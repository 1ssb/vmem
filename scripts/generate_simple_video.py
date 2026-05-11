#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

VMEM_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VMEM_ROOT.parents[1]
CUT3R_ROOT = VMEM_ROOT / "extern" / "CUT3R"
for path in (PROJECT_ROOT, CUT3R_ROOT, VMEM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate one VMem video plus Depth Pro point cloud from one RGB image."
    )
    direction = parser.add_mutually_exclusive_group(required=True)
    direction.add_argument("--left", action="store_true", help="Move forward, then look left.")
    direction.add_argument("--right", action="store_true", help="Move forward, then look right.")
    parser.add_argument("--input", required=True, help="Path to the input RGB image.")
    parser.add_argument(
        "--output",
        help="Output .mp4 path. Defaults to outputs/<image-stem>_forward_<direction>.mp4.",
    )
    parser.add_argument("--config", default="configs/inference/inference.yaml")
    parser.add_argument("--forward-steps", type=int, default=5)
    parser.add_argument("--yaw-degrees", type=float, default=90.0)
    parser.add_argument("--interpolation-frames", type=int, default=8)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument(
        "--pointcloud-output",
        help="Output .ply path. Defaults to <output-stem>_pointcloud.ply.",
    )
    parser.add_argument(
        "--depth-dir",
        help="Directory for per-frame Depth Pro .npy maps. Defaults to <output-stem>_depths.",
    )
    parser.add_argument(
        "--pointcloud-stride",
        type=int,
        default=4,
        help="Use one depth pixel every N pixels when building the point cloud.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=80.0,
        help="Drop Depth Pro points farther than this many meters.",
    )
    parser.add_argument(
        "--focal-px",
        type=float,
        default=576.0,
        help="Fixed focal length in pixels for point-cloud back-projection.",
    )
    return parser.parse_args()


def load_rgb(image_path, config, device, load_img_and_K, transform_img_and_K):
    image, _ = load_img_and_K(str(image_path), None, K=None, device=device)
    image, _ = transform_img_and_K(
        image,
        (config.model.height, config.model.width),
        mode="crop",
        K=None,
    )
    return image


def default_sidecar_path(output_path, suffix):
    return output_path.with_name(f"{output_path.stem}{suffix}")


def make_depth_pro_model(device, torch, max_depth):
    try:
        from src.depth_pro.get_depth import DepthEstimator

        estimator = DepthEstimator(device=device, use_half=device.type == "cuda")

        def run(frame, focal_px):
            result = estimator.process_batch(
                [frame.convert("RGB")],
                _DEPTH_INTERPOLATION_MODE="bilinear",
                FOCAL_LENGTH=float(focal_px),
                _NEAR_PLANE=0.1,
                _FAR_PLANE=float(max_depth),
                use_custom_focal=True,
            )[0]
            return result["depth"].astype("float32")

        return run
    except ModuleNotFoundError:
        pass

    try:
        import depth_pro
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Depth Pro is not importable. This workspace runner first tries the forked "
            "`src.depth_pro` wrapper from the repository root, then the standalone "
            "`depth_pro` package. Install one of those and rerun this command."
        ) from exc

    try:
        model, transform = depth_pro.create_model_and_transforms(device=device)
    except TypeError:
        model, transform = depth_pro.create_model_and_transforms()
        model = model.to(device)

    model.eval()

    def run(frame, focal_px):
        image = transform(frame.convert("RGB")).to(device)
        prediction = model.infer(image, f_px=float(focal_px))
        return to_numpy_depth(prediction["depth"], torch)

    return run


def to_numpy_depth(depth, torch):
    if torch.is_tensor(depth):
        depth = depth.detach().float().cpu().numpy()
    while depth.ndim > 2:
        depth = depth.squeeze(0)
    return depth.astype("float32")


def depth_to_world_points(frame, depth, focal_px, pose, stride, max_depth):
    import numpy as np

    height, width = depth.shape
    rgb = np.asarray(frame.convert("RGB").resize((width, height)))

    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    zs = depth[ys, xs]
    valid = np.isfinite(zs) & (zs > 0) & (zs <= max_depth)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    zs = zs[valid].astype(np.float32)
    colors = rgb[ys.astype(np.int32), xs.astype(np.int32)]

    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    x_cam = (xs - cx) / focal_px * zs
    y_cam = -(ys - cy) / focal_px * zs
    z_cam = -zs

    cam_points = np.stack([x_cam, y_cam, z_cam, np.ones_like(zs)], axis=1)
    world_points = (pose @ cam_points.T).T[:, :3]
    return world_points.astype(np.float32), colors.astype(np.uint8)


def write_ply(path, points, colors):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def export_depth_pro_pointcloud(
    frames,
    poses,
    pointcloud_path,
    depth_dir,
    stride,
    max_depth,
    focal_px,
    device,
    torch,
):
    import numpy as np

    if len(frames) != len(poses):
        raise RuntimeError(f"Frame/pose mismatch: {len(frames)} frames, {len(poses)} poses")

    depth_dir.mkdir(parents=True, exist_ok=True)
    run_depth = make_depth_pro_model(device, torch, max_depth)

    all_points = []
    all_colors = []
    for idx, (frame, pose) in enumerate(zip(frames, poses)):
        print(f"depth pro: frame {idx + 1}/{len(frames)}", flush=True)
        depth = run_depth(frame, focal_px)
        np.save(depth_dir / f"frame_{idx:04d}.npy", depth)

        points, colors = depth_to_world_points(
            frame=frame,
            depth=depth,
            focal_px=focal_px,
            pose=pose,
            stride=stride,
            max_depth=max_depth,
        )
        all_points.append(points)
        all_colors.append(colors)

    points = np.concatenate(all_points, axis=0) if all_points else np.empty((0, 3), dtype=np.float32)
    colors = np.concatenate(all_colors, axis=0) if all_colors else np.empty((0, 3), dtype=np.uint8)
    write_ply(pointcloud_path, points, colors)
    return pointcloud_path, depth_dir, len(points)


def main():
    args = parse_args()
    direction = "left" if args.left else "right"
    image_path = Path(args.input).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (VMEM_ROOT / "outputs" / f"{image_path.stem}_forward_{direction}.mp4").resolve()
    )

    if not image_path.exists():
        raise FileNotFoundError(f"RGB image not found: {image_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(VMEM_ROOT)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = VMEM_ROOT / config_path

    import numpy as np
    import torch
    from diffusers.utils import export_to_video
    from omegaconf import OmegaConf

    from modeling.pipeline import VMemPipeline
    from navigation import Navigator
    from utils import get_default_intrinsics, load_img_and_K, tensor_to_pil, transform_img_and_K

    config = OmegaConf.load(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)

    with torch.no_grad():
        print("loading model...", flush=True)
        model = VMemPipeline(config, device)

        print(f"loading rgb: {image_path}", flush=True)
        image = load_rgb(image_path, config, device, load_img_and_K, transform_img_and_K)
        initial_frame = tensor_to_pil(image)
        initial_pose = np.eye(4)
        initial_K = np.array(get_default_intrinsics()[0])

        navigator = Navigator(
            model,
            step_size=0.1,
            num_interpolation_frames=args.interpolation_frames,
        )
        navigator.initialize(initial_frame, initial_pose, initial_K)

        print("moving forward...", flush=True)
        frames = [initial_frame]
        frames.extend(navigator.move_forward(args.forward_steps) or [])

        print(f"looking {direction}...", flush=True)
        if direction == "left":
            frames.extend(navigator.turn_left(args.yaw_degrees) or [])
        else:
            frames.extend(navigator.turn_right(args.yaw_degrees) or [])

        poses = navigator.frame_poses
        del navigator
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"saving video: {output_path}", flush=True)
    export_to_video(frames, str(output_path), fps=args.fps)
    metadata = {
        "rgb": str(image_path),
        "direction": direction,
        "video": str(output_path),
        "num_frames": len(frames),
        "forward_steps": args.forward_steps,
        "yaw_degrees": args.yaw_degrees,
        "interpolation_frames": args.interpolation_frames,
        "fps": args.fps,
        "focal_px": args.focal_px,
    }

    pointcloud_path = (
        Path(args.pointcloud_output).expanduser().resolve()
        if args.pointcloud_output
        else default_sidecar_path(output_path, "_pointcloud.ply")
    )
    depth_dir = (
        Path(args.depth_dir).expanduser().resolve()
        if args.depth_dir
        else default_sidecar_path(output_path, "_depths")
    )
    print(f"running depth pro and point-cloud export: {pointcloud_path}", flush=True)
    ply_path, depth_dir, point_count = export_depth_pro_pointcloud(
        frames=frames,
        poses=poses,
        pointcloud_path=pointcloud_path,
        depth_dir=depth_dir,
        stride=max(1, args.pointcloud_stride),
        max_depth=args.max_depth,
        focal_px=args.focal_px,
        device=device,
        torch=torch,
    )
    metadata.update(
        {
            "pointcloud": str(ply_path),
            "depth_dir": str(depth_dir),
            "point_count": point_count,
            "pointcloud_stride": args.pointcloud_stride,
            "max_depth": args.max_depth,
        }
    )

    metadata_path = default_sidecar_path(output_path, "_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"metadata: {metadata_path}", flush=True)
    print(f"done: {output_path} ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    main()
