#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

from model_cache import cache_report, configure_model_cache

VMEM_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VMEM_ROOT.parents[1]
DEPTH_PRO_CHECKPOINT = PROJECT_ROOT / "src" / "depth_pro" / "checkpoints" / "depth_pro.pt"


CORE_HF_FILES = (
    ("liguang0115/vmem", "vmem_weights.pth"),
    ("liguang0115/cut3r", "cut3r_512_dpt_4_64.pth"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show and optionally prefetch the VMem model cache under rudra-work."
    )
    parser.add_argument(
        "--download-core",
        action="store_true",
        help="Download the VMem and CUT3R checkpoint files into the configured Hugging Face cache.",
    )
    parser.add_argument(
        "--download-depth-pro",
        action="store_true",
        help="Download the workspace Depth Pro checkpoint into src/depth_pro/checkpoints.",
    )
    return parser.parse_args()


def describe_path(path):
    path = Path(path)
    status = "exists" if path.exists() else "missing"
    return f"{path} ({status})"


def main():
    args = parse_args()
    cache_env = configure_model_cache()
    print(cache_report(cache_env), flush=True)

    print("\ncache directories:", flush=True)
    for key in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME"):
        print(f"  {key}: {describe_path(cache_env[key])}", flush=True)
    print(f"  DEPTH_PRO_CHECKPOINT: {describe_path(DEPTH_PRO_CHECKPOINT)}", flush=True)

    if not args.download_core:
        if not args.download_depth_pro:
            return

    if args.download_core:
        from huggingface_hub import hf_hub_download

        print("\ndownloading core VMem checkpoints:", flush=True)
        for repo_id, filename in CORE_HF_FILES:
            path = hf_hub_download(repo_id=repo_id, filename=filename)
            print(f"  {repo_id}/{filename} -> {path}", flush=True)

    if args.download_depth_pro:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from src.depth_pro.src.downloader import check_for_model

        print("\ndownloading Depth Pro checkpoint:", flush=True)
        path = check_for_model()
        print(f"  depth_pro -> {path}", flush=True)


if __name__ == "__main__":
    main()
