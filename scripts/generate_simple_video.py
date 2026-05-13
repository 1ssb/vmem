#!/usr/bin/env python
import argparse
import contextlib
import json
import logging
import os
import sys
import warnings
from pathlib import Path

VMEM_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VMEM_ROOT.parents[1]
CUT3R_ROOT = VMEM_ROOT / "extern" / "CUT3R"
CUT3R_SRC_ROOT = CUT3R_ROOT / "src"
for path in (CUT3R_SRC_ROOT, CUT3R_ROOT, VMEM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_cache import configure_model_cache

MODEL_CACHE_ENV = configure_model_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate one VMem video plus Depth Pro point cloud from one RGB image."
    )
    direction = parser.add_mutually_exclusive_group(required=True)
    direction.add_argument("--left", action="store_true", help="Move forward, then look left.")
    direction.add_argument("--right", action="store_true", help="Move forward, then look right.")
    parser.add_argument("--input", required=True, help="Path to the input RGB image.")
    parser.add_argument(
        "--camera-json",
        help=(
            "Benchmark camera metadata JSON. When provided, K_crop/K and "
            "T_camera_to_world are used instead of default square intrinsics."
        ),
    )
    parser.add_argument(
        "--image-transform-mode",
        choices=["crop", "pad", "stretch"],
        default="crop",
        help=(
            "How to fit the input image into VMem's square model resolution. "
            "Use 'pad' for benchmark crops so crop geometry is preserved."
        ),
    )
    parser.add_argument(
        "--restore-input-size",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export/depth frames at the original input image size.",
    )
    parser.add_argument(
        "--output",
        help="Output .mp4 path. Defaults to outputs/<image-stem>_forward_<direction>.mp4.",
    )
    parser.add_argument("--config", default="configs/inference/inference.yaml")
    parser.add_argument("--forward-steps", type=int, default=5)
    parser.add_argument("--yaw-degrees", type=float, default=90.0)
    parser.add_argument(
        "--forward-frames",
        type=int,
        default=5,
        help="Generated frames for the forward move.",
    )
    parser.add_argument(
        "--turn-frames",
        type=int,
        default=12,
        help="Generated frames for the left/right turn.",
    )
    parser.add_argument(
        "--interpolation-frames",
        type=int,
        help="Legacy alias for --forward-frames. The turn stays controlled by --turn-frames.",
    )
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--verbose", action="store_true", help="Show verbose library logs.")
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
        help=(
            "Fallback focal length in pixels for Depth Pro and point-cloud "
            "back-projection when --camera-json is not supplied."
        ),
    )
    return parser.parse_args()


def configure_logging(verbose=False):
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, force=True)
    logging.getLogger().setLevel(level)

    if verbose:
        return

    for logger_name in (
        "asyncio",
        "httpcore",
        "httpx",
        "huggingface_hub",
        "matplotlib",
        "PIL",
        "python_multipart",
        "src.depth_pro",
        "flux_pipeline",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
    warnings.filterwarnings("ignore", message=".*invalid value encountered.*")


@contextlib.contextmanager
def quiet_external_output(verbose=False):
    if verbose:
        yield
        return

    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def load_camera_metadata(camera_json_path):
    import numpy as np

    if not camera_json_path:
        return None

    path = Path(camera_json_path).expanduser().resolve()
    metadata = json.loads(path.read_text())
    key = "K_crop" if "K_crop" in metadata else "K"
    if key not in metadata:
        raise KeyError(f"{path} does not contain K_crop or K")

    pose = metadata.get("T_camera_to_world")
    if pose is None:
        pose = metadata.get("camera", {}).get("T_camera_to_world")

    return {
        "path": path,
        "intrinsics_key": key,
        "K": np.asarray(metadata[key], dtype="float32"),
        "T_camera_to_world": np.asarray(pose, dtype="float32") if pose is not None else None,
        "metadata": metadata,
    }


def load_rgb(
    image_path,
    config,
    device,
    load_img_and_K,
    transform_img_and_K,
    camera_metadata=None,
    transform_mode="crop",
):
    import torch

    image, _ = load_img_and_K(str(image_path), None, K=None, device=device)
    K = None
    if camera_metadata is not None:
        K = torch.as_tensor(camera_metadata["K"], dtype=torch.float32, device=image.device)[None]
    image, transformed_K = transform_img_and_K(
        image,
        (config.model.height, config.model.width),
        mode=transform_mode,
        K=K,
    )
    model_K = None if transformed_K is None else transformed_K[0].detach().cpu().numpy()
    return image, model_K


def default_sidecar_path(output_path, suffix):
    return output_path.with_name(f"{output_path.stem}{suffix}")


def enable_workspace_depth_pro_import():
    # CUT3R imports modules from a top-level namespace package named `src`
    # (`src.dust3r`). The broader workspace also has a real package named `src`
    # for the forked Depth Pro wrapper (`src.depth_pro`). Keep the workspace root
    # off sys.path while VMem/CUT3R runs, then switch to it only for Depth Pro.
    for module_name in list(sys.modules):
        if module_name == "src" or module_name.startswith("src."):
            del sys.modules[module_name]

    project_root = str(PROJECT_ROOT)
    if project_root in sys.path:
        sys.path.remove(project_root)
    sys.path.insert(0, project_root)


def make_depth_pro_model(device, torch, max_depth):
    enable_workspace_depth_pro_import()
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


def depth_to_world_points(frame, depth, focal_px, pose, stride, max_depth, K=None):
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

    if K is not None:
        K = np.asarray(K, dtype=np.float32).copy()
        input_w, input_h = frame.size
        if input_w != width or input_h != height:
            K[0, :] *= width / input_w
            K[1, :] *= height / input_h
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
    else:
        fx = fy = focal_px
        cx = (width - 1) * 0.5
        cy = (height - 1) * 0.5

    x_cam = (xs - cx) / fx * zs
    y_cam = -(ys - cy) / fy * zs
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
    K=None,
    verbose=False,
):
    import numpy as np

    if len(frames) != len(poses):
        raise RuntimeError(f"Frame/pose mismatch: {len(frames)} frames, {len(poses)} poses")

    depth_dir.mkdir(parents=True, exist_ok=True)
    with quiet_external_output(verbose):
        run_depth = make_depth_pro_model(device, torch, max_depth)

    all_points = []
    all_colors = []
    for idx, (frame, pose) in enumerate(zip(frames, poses)):
        print(f"depth pro: frame {idx + 1}/{len(frames)}", end="\r", flush=True)
        with quiet_external_output(verbose):
            depth = run_depth(frame, focal_px)
        np.save(depth_dir / f"frame_{idx:04d}.npy", depth)

        points, colors = depth_to_world_points(
            frame=frame,
            depth=depth,
            focal_px=focal_px,
            pose=pose,
            stride=stride,
            max_depth=max_depth,
            K=K,
        )
        all_points.append(points)
        all_colors.append(colors)
    print(" " * 40, end="\r", flush=True)

    points = np.concatenate(all_points, axis=0) if all_points else np.empty((0, 3), dtype=np.float32)
    colors = np.concatenate(all_colors, axis=0) if all_colors else np.empty((0, 3), dtype=np.uint8)
    write_ply(pointcloud_path, points, colors)
    return pointcloud_path, depth_dir, len(points)


def fit_rect_for_model(original_size, model_size, mode):
    import math

    original_w, original_h = original_size
    model_w, model_h = model_size
    if mode == "stretch":
        return (0, 0, model_w, model_h)

    if mode == "pad":
        factor = min(model_w / original_w, model_h / original_h)
    else:
        factor = max(model_w / original_w, model_h / original_h)

    resized_w = int(math.ceil(original_w * factor))
    resized_h = int(math.ceil(original_h * factor))
    left = max(0, (model_w - resized_w) // 2)
    top = max(0, (model_h - resized_h) // 2)

    if mode == "pad":
        return (left, top, left + resized_w, top + resized_h)

    crop_left = max(0, (resized_w - model_w) // 2)
    crop_top = max(0, (resized_h - model_h) // 2)
    return (-crop_left, -crop_top, -crop_left + resized_w, -crop_top + resized_h)


def restore_frame_to_input_size(frame, original_size, model_size, mode):
    if frame.size != model_size:
        frame = frame.resize(model_size)

    if mode == "pad":
        left, top, right, bottom = fit_rect_for_model(original_size, model_size, mode)
        left = max(0, left)
        top = max(0, top)
        right = min(model_size[0], right)
        bottom = min(model_size[1], bottom)
        frame = frame.crop((left, top, right, bottom))

    return frame.resize(original_size)


def focal_from_intrinsics(K, fallback):
    if K is None:
        return fallback
    return float((float(K[0, 0]) + float(K[1, 1])) * 0.5)


def main():
    args = parse_args()
    configure_logging(args.verbose)

    direction = "left" if args.left else "right"
    image_path = Path(args.input).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (VMEM_ROOT / "outputs" / f"{image_path.stem}_forward_{direction}.mp4").resolve()
    )

    if not image_path.exists():
        raise FileNotFoundError(f"RGB image not found: {image_path}")

    camera_metadata = load_camera_metadata(args.camera_json)
    output_K = camera_metadata["K"] if camera_metadata is not None else None
    depth_focal_px = focal_from_intrinsics(output_K, args.focal_px)

    forward_frames = args.interpolation_frames or args.forward_frames
    turn_frames = args.turn_frames
    if forward_frames < 1 or turn_frames < 1:
        raise ValueError("--forward-frames and --turn-frames must be >= 1")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(VMEM_ROOT)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = VMEM_ROOT / config_path

    with quiet_external_output(args.verbose):
        import numpy as np
        import torch
        from diffusers.utils import export_to_video
        from omegaconf import OmegaConf

        from modeling.pipeline import VMemPipeline
        from navigation import Navigator
        from utils import get_default_intrinsics, load_img_and_K, tensor_to_pil, transform_img_and_K
    configure_logging(args.verbose)

    config = OmegaConf.load(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"model cache: {MODEL_CACHE_ENV['HF_HOME']}", flush=True)

    with torch.no_grad():
        print("loading model...", flush=True)
        with quiet_external_output(args.verbose):
            model = VMemPipeline(config, device)

        print(f"loading rgb: {image_path}", flush=True)
        from PIL import Image

        with Image.open(image_path) as pil_image:
            original_size = pil_image.size
        image, model_K = load_rgb(
            image_path,
            config,
            device,
            load_img_and_K,
            transform_img_and_K,
            camera_metadata=camera_metadata,
            transform_mode=args.image_transform_mode,
        )
        initial_frame = tensor_to_pil(image)
        initial_pose = (
            camera_metadata["T_camera_to_world"].copy()
            if camera_metadata is not None and camera_metadata["T_camera_to_world"] is not None
            else np.eye(4)
        )
        initial_K = model_K if model_K is not None else np.array(get_default_intrinsics()[0])

        navigator = Navigator(
            model,
            step_size=0.1,
            num_interpolation_frames=forward_frames,
        )
        with quiet_external_output(args.verbose):
            navigator.initialize(initial_frame, initial_pose, initial_K)

        print(f"moving forward ({forward_frames} frames)...", flush=True)
        frames = [initial_frame]
        navigator.num_interpolation_frames = forward_frames
        with quiet_external_output(args.verbose):
            frames.extend(navigator.move_forward(args.forward_steps) or [])

        print(f"looking {direction} ({turn_frames} frames over {args.yaw_degrees:g} deg)...", flush=True)
        navigator.num_interpolation_frames = turn_frames
        with quiet_external_output(args.verbose):
            if direction == "left":
                frames.extend(navigator.turn_left(args.yaw_degrees) or [])
            else:
                frames.extend(navigator.turn_right(args.yaw_degrees) or [])

        poses = navigator.frame_poses
        del navigator
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model_size = (config.model.width, config.model.height)
    export_frames = (
        [restore_frame_to_input_size(frame, original_size, model_size, args.image_transform_mode) for frame in frames]
        if args.restore_input_size
        else frames
    )

    print(f"saving video: {output_path}", flush=True)
    export_to_video(export_frames, str(output_path), fps=args.fps)
    metadata = {
        "rgb": str(image_path),
        "original_size": list(original_size),
        "model_size": [config.model.width, config.model.height],
        "image_transform_mode": args.image_transform_mode,
        "restore_input_size": args.restore_input_size,
        "direction": direction,
        "video": str(output_path),
        "num_frames": len(export_frames),
        "forward_steps": args.forward_steps,
        "yaw_degrees": args.yaw_degrees,
        "forward_frames": forward_frames,
        "turn_frames": turn_frames,
        "interpolation_frames": args.interpolation_frames,
        "fps": args.fps,
        "depth_focal_px": depth_focal_px,
        "model_cache": MODEL_CACHE_ENV,
    }
    if camera_metadata is not None:
        metadata.update(
            {
                "camera_json": str(camera_metadata["path"]),
                "intrinsics_key": camera_metadata["intrinsics_key"],
                "K_output": output_K.tolist(),
                "K_model": initial_K.tolist() if hasattr(initial_K, "tolist") else initial_K,
                "initial_pose_source": (
                    "T_camera_to_world"
                    if camera_metadata["T_camera_to_world"] is not None
                    else "identity"
                ),
            }
        )
    else:
        metadata["focal_px"] = args.focal_px

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
        frames=export_frames,
        poses=poses,
        pointcloud_path=pointcloud_path,
        depth_dir=depth_dir,
        stride=max(1, args.pointcloud_stride),
        max_depth=args.max_depth,
        focal_px=depth_focal_px,
        device=device,
        torch=torch,
        K=output_K,
        verbose=args.verbose,
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
    print(f"done: {output_path} ({len(export_frames)} frames)", flush=True)


if __name__ == "__main__":
    main()
