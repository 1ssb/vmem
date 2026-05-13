#!/usr/bin/env python
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from diffusers.utils import export_to_video

VMEM_ROOT = Path(__file__).resolve().parents[1]
if str(VMEM_ROOT) not in sys.path:
    sys.path.insert(0, str(VMEM_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Run a VMem room-entry camera preset on one RGB image.")
    parser.add_argument("--image", required=True, help="Path to the input RGB image.")
    parser.add_argument("--direction", choices=["left", "right"], default="right")
    parser.add_argument("--output-dir", required=True, help="Directory for generated frames/video.")
    parser.add_argument("--keep-existing", action="store_true", help="Do not clear the output directory first.")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    print(f"checkpoint: input={image_path}", flush=True)
    print(f"checkpoint: output={output_dir}", flush=True)
    print(f"checkpoint: cuda_available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"checkpoint: gpu={torch.cuda.get_device_name(0)}", flush=True)

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    if output_dir.exists() and not args.keep_existing:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("checkpoint: importing app/model", flush=True)
    import app
    from utils import tensor_to_pil

    app.NAVIGATORS = []
    print("checkpoint: loading input image", flush=True)
    result = app.load_image_for_navigation(str(image_path))

    print(f"checkpoint: navigate_room_entry_control(direction={args.direction})", flush=True)
    video, poses, current_view, _, gallery = app.navigate_room_entry_control(
        result["video"],
        result["pose"],
        direction=args.direction,
    )

    frames = [tensor_to_pil(video[i]) for i in range(len(video))]
    for idx, frame in enumerate(frames):
        frame.save(output_dir / f"frame_{idx:03d}.png")

    video_path = output_dir / f"enter_look_{args.direction}.mp4"
    export_to_video(frames, str(video_path), fps=app.NAVIGATION_FPS)

    metadata = {
        "image": str(image_path),
        "direction": args.direction,
        "num_frames": len(frames),
        "video": str(video_path),
        "room_entry_yaw_degrees": app.ROOM_ENTRY_YAW_DEGREES,
        "room_entry_forward_steps": app.ROOM_ENTRY_FORWARD_STEPS,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"checkpoint: wrote {len(frames)} frames", flush=True)
    print(f"checkpoint: video={video_path}", flush=True)


if __name__ == "__main__":
    main()
