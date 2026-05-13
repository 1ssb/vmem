#!/usr/bin/env python
import argparse
import json
import shutil
from pathlib import Path


VMEM_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VMEM_ROOT.parents[1]
BENCHMARK_ROOT = PROJECT_ROOT / "pipeline" / "benchmark"
DEFAULT_INDEX = BENCHMARK_ROOT / "dataset_card" / "index.json"
DEFAULT_OUTPUT = VMEM_ROOT / "benchmark_jobs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create one-crop VMem jobs from the benchmark representative cards."
    )
    parser.add_argument("--index", default=str(DEFAULT_INDEX), help="Dataset-card index JSON.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="VMem benchmark job directory.")
    parser.add_argument("--per-robot-node", type=int, default=1, help="Jobs to keep per robot_node.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum jobs to prepare.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing copied inputs/metadata.")
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text())


def select_cards(cards, per_robot_node, limit):
    selected = []
    counts = {}
    for card in cards:
        node = card["robot_node"]
        if counts.get(node, 0) >= per_robot_node:
            continue
        selected.append(card)
        counts[node] = counts.get(node, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def job_slug(index, card):
    task_id = card["task_id"]
    return f"{index:02d}_{card['robot_node']}_{card['side']}_{task_id.split('__obj')[-1]}"


def build_command(job_dir, card):
    direction_flag = f"--{card['side']}"
    output_path = job_dir / "outputs" / f"{card['task_id']}_vmem_{card['side']}.mp4"
    return [
        "python",
        "scripts/generate_simple_video.py",
        direction_flag,
        "--input",
        str(job_dir / "input_crop.png"),
        "--camera-json",
        str(job_dir / "camera.json"),
        "--image-transform-mode",
        "pad",
        "--restore-input-size",
        "--output",
        str(output_path),
    ]


def main():
    args = parse_args()
    index_path = Path(args.index).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    index = load_json(index_path)
    cards = select_cards(index["cards"], args.per_robot_node, args.limit)
    if len(cards) < args.limit:
        print(f"warning: selected only {len(cards)} cards for limit={args.limit}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for idx, card in enumerate(cards, start=1):
        task_id = card["task_id"]
        task_dir = BENCHMARK_ROOT / "dataset_card" / task_id
        task = load_json(task_dir / "task.json")
        gt_dir = BENCHMARK_ROOT / "data" / "gt_package" / task["image_id"]
        image_src = gt_dir / "crop_rgb.png"
        camera_src = gt_dir / "camera.json"
        if not image_src.exists():
            raise FileNotFoundError(image_src)
        if not camera_src.exists():
            raise FileNotFoundError(camera_src)

        job_dir = output_dir / job_slug(idx, card)
        if args.overwrite and job_dir.exists():
            shutil.rmtree(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_src, job_dir / "input_crop.png")
        shutil.copy2(camera_src, job_dir / "camera.json")
        shutil.copy2(task_dir / "task.json", job_dir / "task.json")

        command = build_command(job_dir, card)
        job = {
            "job_id": job_dir.name,
            "task_id": task_id,
            "image_id": task["image_id"],
            "building_id": task["building_id"],
            "robot_node": task["robot_node"],
            "raw_label": task["raw_label"],
            "target_room_type": task["target_room_type"],
            "side": task["side"],
            "observation_refined": task.get("observation_refined"),
            "extension_px": task.get("extension_px"),
            "object_id": task["object_id"],
            "input_crop": str(job_dir / "input_crop.png"),
            "camera_json": str(job_dir / "camera.json"),
            "task_json": str(job_dir / "task.json"),
            "output_video": str(job_dir / "outputs" / f"{task_id}_vmem_{task['side']}.mp4"),
            "command": command,
        }
        (job_dir / "job.json").write_text(json.dumps(job, indent=2) + "\n")
        jobs.append(job)

    manifest = {
        "source_index": str(index_path),
        "benchmark_root": str(BENCHMARK_ROOT),
        "output_dir": str(output_dir),
        "selection_policy": "first representative card per robot_node from dataset_card/index.json",
        "n_jobs": len(jobs),
        "jobs": jobs,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"prepared {len(jobs)} jobs: {manifest_path}", flush=True)
    for job in jobs:
        print(f"{job['job_id']}: {job['robot_node']} {job['side']} {job['target_room_type']}", flush=True)


if __name__ == "__main__":
    main()
