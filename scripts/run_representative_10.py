#!/usr/bin/env python
import argparse
import json
import subprocess
import sys
from pathlib import Path


VMEM_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = VMEM_ROOT / "benchmark_jobs" / "manifest.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the fixed 10-task representative VMem benchmark pack."
    )
    parser.add_argument(
        "--side",
        choices=["all", "left", "right"],
        default="all",
        help="Run all tasks, only left-side tasks, or only right-side tasks.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable from the VMem environment.",
    )
    parser.add_argument("--only", nargs="*", help="Specific job_id values to run.")
    parser.add_argument("--limit", type=int, help="Run at most N selected jobs.")
    parser.add_argument("--rerun", action="store_true", help="Run even if output video exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--list", action="store_true", help="List selected jobs and exit.")
    return parser.parse_args()


def load_jobs():
    manifest = json.loads(MANIFEST_PATH.read_text())
    jobs = manifest["jobs"]
    if len(jobs) != 10:
        raise RuntimeError(f"Expected 10 representative jobs, found {len(jobs)}")
    return jobs


def selected_jobs(jobs, side, only, limit):
    if side != "all":
        jobs = [job for job in jobs if job["side"] == side]
    if only:
        wanted = set(only)
        jobs = [job for job in jobs if job["job_id"] in wanted]
    if limit is not None:
        jobs = jobs[:limit]
    return jobs


def command_for(job, python_executable):
    command = job["command"]
    if not command:
        raise RuntimeError(f"{job['job_id']} has no command")
    if command[0] == "python":
        return [python_executable, *command[1:]]
    return command


def print_job(job, command=None):
    label = (
        f"{job['job_id']}: {job['side']} | {job['robot_node']} "
        f"in {job['target_room_type']} | {job['observation_refined']}"
    )
    print(label, flush=True)
    if command is not None:
        print(f"  {' '.join(command)}", flush=True)


def main():
    args = parse_args()
    jobs = selected_jobs(load_jobs(), args.side, args.only, args.limit)
    if not jobs:
        print("No representative jobs selected.", flush=True)
        return 0

    if args.list:
        for job in jobs:
            print_job(job)
        return 0

    for index, job in enumerate(jobs, start=1):
        command = command_for(job, args.python)
        output_video = Path(job["output_video"])
        print(f"[{index}/{len(jobs)}]", flush=True)
        print_job(job, command)

        if output_video.exists() and not args.rerun:
            print(f"  skip existing: {output_video}", flush=True)
            continue
        if args.dry_run:
            continue

        output_video.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(command, cwd=VMEM_ROOT)
        if result.returncode != 0:
            return result.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
