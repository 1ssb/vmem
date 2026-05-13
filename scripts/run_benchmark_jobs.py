#!/usr/bin/env python
import argparse
import json
import subprocess
import sys
from pathlib import Path


VMEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = VMEM_ROOT / "benchmark_jobs" / "manifest.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Run prepared VMem benchmark jobs.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--limit", type=int, help="Run at most N jobs.")
    parser.add_argument("--only", nargs="*", help="Run only these job_id values.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable from the VMem environment. Defaults to the current interpreter.",
    )
    parser.add_argument("--rerun", action="store_true", help="Run even if the output video already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())

    jobs = manifest["jobs"]
    if args.only:
        wanted = set(args.only)
        jobs = [job for job in jobs if job["job_id"] in wanted]
    if args.limit is not None:
        jobs = jobs[: args.limit]

    if not jobs:
        print("no jobs selected", flush=True)
        return

    for idx, job in enumerate(jobs, start=1):
        output_video = Path(job["output_video"])
        command = job["command"]
        run_command = [args.python, *command[1:]] if command and command[0] == "python" else command
        print(f"[{idx}/{len(jobs)}] {job['job_id']}: {' '.join(command)}", flush=True)
        if output_video.exists() and not args.rerun:
            print(f"skip existing: {output_video}", flush=True)
            continue
        if args.dry_run:
            continue

        output_video.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(run_command, cwd=VMEM_ROOT)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    sys.exit(main())
