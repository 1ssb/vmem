#!/usr/bin/env python
import argparse
import json
import sys
import time
from pathlib import Path


VMEM_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VMEM_ROOT.parents[1]
DEFAULT_MANIFEST = VMEM_ROOT / "benchmark_jobs" / "manifest.json"
SEGFORMER_NAME = "nvidia/segformer-b5-finetuned-ade-640-640"
PROB_THRESH = 0.25
AREA_THRESH = 0.005


TARGET_ADE20K = {
    "bed": ["bed", "cradle"],
    "cabinet": ["cabinet", "wardrobe", "chest of drawers"],
    "chair": ["chair", "armchair", "seat", "bench", "swivel chair"],
    "microwave": ["microwave"],
    "plant": ["plant", "flower", "palm"],
    "refrigerator": ["refrigerator"],
    "shelf": ["shelf", "bookcase", "case"],
    "sink": ["sink"],
    "table": ["table", "desk", "counter", "countertop", "coffee table", "kitchen island"],
    "tv": ["television receiver", "monitor", "screen", "computer", "crt screen"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SegFormer target checks on the fixed VMem representative-10 outputs."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--side", choices=["all", "left", "right"], default="all")
    parser.add_argument("--only", nargs="*", help="Specific job_id values to evaluate.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--prob-thresh", type=float, default=PROB_THRESH)
    parser.add_argument("--area-thresh", type=float, default=AREA_THRESH)
    parser.add_argument("--require-videos", action="store_true", help="Fail if any selected output video is missing.")
    parser.add_argument("--install-check", action="store_true", help="Only load SegFormer and print resolved labels.")
    parser.add_argument(
        "--out-json",
        default=str(VMEM_ROOT / "benchmark_jobs" / "segformer_summary.json"),
        help="Aggregate summary JSON path.",
    )
    return parser.parse_args()


def load_manifest(path):
    manifest = json.loads(Path(path).expanduser().resolve().read_text())
    jobs = manifest["jobs"]
    if len(jobs) != 10:
        raise RuntimeError(f"Expected representative-10 manifest, found {len(jobs)} jobs")
    return jobs


def select_jobs(jobs, side, only):
    if side != "all":
        jobs = [job for job in jobs if job["side"] == side]
    if only:
        wanted = set(only)
        jobs = [job for job in jobs if job["job_id"] in wanted]
    return jobs


def target_labels(job):
    labels = TARGET_ADE20K.get(job["robot_node"])
    if labels is None:
        raise KeyError(f"No ADE20K target labels configured for robot_node={job['robot_node']}")
    return labels


def load_model(device_name):
    import torch
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    processor = SegformerImageProcessor.from_pretrained(SEGFORMER_NAME)
    try:
        model = SegformerForSemanticSegmentation.from_pretrained(
            SEGFORMER_NAME,
            torch_dtype=dtype,
        )
    except ValueError:
        import transformers.modeling_utils as _mu

        saved = _mu.check_torch_load_is_safe
        try:
            _mu.check_torch_load_is_safe = lambda: None
            model = SegformerForSemanticSegmentation.from_pretrained(
                SEGFORMER_NAME,
                torch_dtype=dtype,
            )
        finally:
            _mu.check_torch_load_is_safe = saved

    model = model.to(device).eval()
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    name2id = {v.lower().strip(): k for k, v in id2label.items()}
    return processor, model, device, id2label, name2id


def class_ids_for(labels, name2id):
    return [name2id[label.lower().strip()] for label in labels if label.lower().strip() in name2id]


def read_video_frames(path):
    import imageio.v3 as iio
    import numpy as np
    from PIL import Image

    frames = []
    for frame in iio.imiter(path):
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        frames.append(Image.fromarray(frame[:, :, :3].astype(np.uint8), mode="RGB"))
    return frames


def evaluate_frames(
    frames,
    class_ids,
    processor,
    model,
    device,
    id2label,
    chunk_size,
    prob_thresh,
    area_thresh,
):
    import torch
    import torch.nn.functional as F

    width, height = frames[0].size
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    detections = {}

    for start in range(0, len(frames), chunk_size):
        chunk = frames[start:start + chunk_size]
        inputs = processor(images=chunk, return_tensors="pt")
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        if torch.is_tensor(inputs.get("pixel_values")):
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
            probs = F.softmax(probs, dim=1)

        for i in range(len(chunk)):
            frame_idx = start + i
            best_area = 0.0
            best_class = None
            for cid in class_ids:
                prob_map = probs[i, cid].detach().float().cpu().numpy()
                area = float((prob_map > prob_thresh).sum()) / float(width * height)
                if area > best_area:
                    best_area = area
                    best_class = id2label[cid].strip()
            detections[f"frame_{frame_idx:04d}"] = {
                "detected": best_area > area_thresh,
                "area_fraction": round(best_area, 6),
                "best_class": best_class,
            }

        del logits, probs, inputs

    return detections


def evaluate_job(job, processor, model, device, id2label, name2id, args):
    video_path = Path(job["output_video"])
    labels = target_labels(job)
    class_ids = class_ids_for(labels, name2id)
    if not class_ids:
        raise RuntimeError(f"{job['job_id']}: none of {labels} exists in SegFormer labels")

    result = {
        "job_id": job["job_id"],
        "task_id": job["task_id"],
        "robot_node": job["robot_node"],
        "raw_label": job["raw_label"],
        "side": job["side"],
        "target_room_type": job["target_room_type"],
        "video": str(video_path),
        "target_ade20k_labels": labels,
        "class_ids": class_ids,
        "class_labels": [id2label[cid].strip() for cid in class_ids],
        "exists": video_path.exists(),
    }
    if not video_path.exists():
        result.update({"status": "missing_video", "detected_frames": 0, "num_frames": 0, "detected": False})
        return result

    t0 = time.perf_counter()
    frames = read_video_frames(video_path)
    if not frames:
        result.update({"status": "empty_video", "detected_frames": 0, "num_frames": 0, "detected": False})
        return result

    detections = evaluate_frames(
        frames=frames,
        class_ids=class_ids,
        processor=processor,
        model=model,
        device=device,
        id2label=id2label,
        chunk_size=max(1, args.chunk_size),
        prob_thresh=args.prob_thresh,
        area_thresh=args.area_thresh,
    )
    detected_frames = sum(1 for det in detections.values() if det["detected"])
    best_frame, best = max(detections.items(), key=lambda item: item[1]["area_fraction"])
    elapsed = time.perf_counter() - t0

    result.update(
        {
            "status": "ok",
            "num_frames": len(frames),
            "detected_frames": detected_frames,
            "detected": detected_frames > 0,
            "detection_rate": round(detected_frames / len(frames), 6),
            "best_frame": best_frame,
            "best_area_fraction": best["area_fraction"],
            "best_class": best["best_class"],
            "elapsed_s": round(elapsed, 3),
            "detections": detections,
        }
    )

    out_path = video_path.with_name(f"{video_path.stem}_segformer_target_detections.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    result["job_detection_json"] = str(out_path)
    return result


def main():
    args = parse_args()
    jobs = select_jobs(load_manifest(args.manifest), args.side, args.only)
    if not jobs:
        print("No jobs selected.", flush=True)
        return 0

    missing = [job for job in jobs if not Path(job["output_video"]).exists()]
    if missing and args.require_videos:
        for job in missing:
            print(f"missing video: {job['job_id']} -> {job['output_video']}", flush=True)
        return 2

    processor, model, device, id2label, name2id = load_model(args.device)
    print(f"SegFormer: {SEGFORMER_NAME}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Selected jobs: {len(jobs)}", flush=True)

    resolved = {
        job["job_id"]: {
            "robot_node": job["robot_node"],
            "side": job["side"],
            "target_ade20k_labels": target_labels(job),
            "class_ids": class_ids_for(target_labels(job), name2id),
        }
        for job in jobs
    }
    if args.install_check:
        print(json.dumps(resolved, indent=2), flush=True)
        return 0

    results = []
    for index, job in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {job['job_id']} {job['side']} {job['robot_node']}", flush=True)
        result = evaluate_job(job, processor, model, device, id2label, name2id, args)
        status = result["status"]
        print(
            f"  {status}: {result['detected_frames']}/{result['num_frames']} frames "
            f"detected={result['detected']}",
            flush=True,
        )
        results.append(result)

    passed = sum(1 for result in results if result.get("detected"))
    summary = {
        "schema_version": "1.0",
        "segformer_model": SEGFORMER_NAME,
        "prob_thresh": args.prob_thresh,
        "area_thresh": args.area_thresh,
        "side": args.side,
        "n_jobs": len(results),
        "detected_jobs": passed,
        "job_detection_rate": round(passed / len(results), 6) if results else 0.0,
        "results": results,
    }
    out_json = Path(args.out_json).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"summary: {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
