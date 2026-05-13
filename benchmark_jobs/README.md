# VMem Representative 10 Benchmark Jobs

This folder contains the 10 one-sample VMem experiments copied from the benchmark
representative dataset cards. Each job uses the benchmark crop as the start
image and the task side to choose the motion:

- `left` tasks run `--left`
- `right` tasks run `--right`

The image fed to VMem is padded into the square model frame. Generated frames are
restored to the original crop size before video, depth, and point-cloud export,
and point-cloud back-projection uses the copied Matterport `K_crop` from each
job's `camera.json`.

## Run

First verify the SegFormer benchmark checker can load and resolve the target
classes:

```bash
cd /home/group/rudra-work/projects/into_the_unknown/studies/vmem
python scripts/evaluate_representative_10_segformer.py --install-check
```

Then run VMem from the VMem environment:

```bash
cd /home/group/rudra-work/projects/into_the_unknown/studies/vmem
python scripts/run_representative_10.py
```

Run only one side:

```bash
python scripts/run_representative_10.py --side left
python scripts/run_representative_10.py --side right
```

Use an explicit VMem Python if the launcher shell is not already activated:

```bash
python scripts/run_representative_10.py --python /path/to/vmem/bin/python
```

Inspect without launching:

```bash
python scripts/run_representative_10.py --list
python scripts/run_representative_10.py --dry-run
```

After videos exist, run the target-object SegFormer check:

```bash
python scripts/evaluate_representative_10_segformer.py
```

Evaluate only one side:

```bash
python scripts/evaluate_representative_10_segformer.py --side left
python scripts/evaluate_representative_10_segformer.py --side right
```

## Outputs

Each job writes under its own `outputs/` directory:

- `<task_id>_vmem_<side>.mp4`
- `<task_id>_vmem_<side>_metadata.json`
- `<task_id>_vmem_<side>_pointcloud.ply`
- `<task_id>_vmem_<side>_depths/frame_*.npy`
- `<task_id>_vmem_<side>_segformer_target_detections.json`

The aggregate SegFormer result is written to
`benchmark_jobs/segformer_summary.json`.

## Benchmark Check

The benchmark check uses `nvidia/segformer-b5-finetuned-ade-640-640`.
For each VMem video, it evaluates every frame against the task's target ADE20K
classes and marks the job as detected if at least one frame has target area
above the configured threshold.

Default thresholds:

- probability: `0.25`
- area fraction: `0.005`

## Files

Each job directory contains:

- `input_crop.png`: copied benchmark crop RGB
- `camera.json`: copied GT camera metadata with `K_crop`
- `task.json`: copied benchmark task metadata
- `job.json`: local launch metadata and exact command

The full pack is recorded in `manifest.json`; a compact task table is in
`TASKS.md`.
