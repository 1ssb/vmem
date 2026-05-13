<div align="center">
<img src="assets/title_logo.png" width="200" alt="VMem Logo"/>
<h1>VMem: Consistent Interactive Video Scene Generation with Surfel-Indexed View Memory</h1>

<p align="center">ICCV 2025 ⭐ <strong>highlight</strong> ⭐</p>


<a href="https://v-mem.github.io/"><img src="https://img.shields.io/badge/%F0%9F%8F%A0%20Project%20Page-gray.svg"></a>
<a href="http://arxiv.org/abs/2506.18903"><img src="https://img.shields.io/badge/%F0%9F%93%84%20arXiv-2506.18903-B31B1B.svg"></a>
<a href="https://huggingface.co/liguang0115/vmem"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model_Card-Huggingface-orange"></a>
<a href="https://huggingface.co/spaces/liguang0115/vmem"><img src="https://img.shields.io/badge/%F0%9F%9A%80%20Gradio%20Demo-Huggingface-orange"></a>

[Runjia Li](https://runjiali-rl.github.io/), [Philip Torr](https://www.robots.ox.ac.uk/~phst/), [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi/), [Tomas Jakab](https://www.robots.ox.ac.uk/~tomj/)
<br>
<br>
[University of Oxford](https://www.robots.ox.ac.uk/~vgg/)
</div>

<p align="center">
  <img src="assets/demo_teaser.gif" width="100%" alt="Teaser" style="border-radius:10px;"/>
</p>

<!-- <p align="center" border-radius="10px">
  <img src="assets/benchmark.png" width="100%" alt="teaser_page1"/>
</p> -->

# Overview

`VMem` is a plug-and-play memory mechanism of image-set models for consistent scene generation.
Existing methods either rely on inpainting with explicit geometry estimation, which suffers from inaccuracies, or use limited context windows in video-based approaches, leading to poor long-term coherence. To overcome these issues, we introduce Surfel Memory of Views (VMem), which anchors past views to surface elements (surfels) they observed. This enables conditioning novel view generation on the most relevant past views rather than just the most recent ones, enhancing long-term scene consistency while reducing computational cost.


# :wrench: Installation

```bash
conda create -n vmem python=3.10
conda activate vmem
pip install -r requirements.txt
```


# :rocket: Usage

You need to properly authenticate with Hugging Face to download our model weights. Once set up, our code will handle it automatically at your first run. You can authenticate by running

```bash
# This will prompt you to enter your Hugging Face credentials.
huggingface-cli login
```

Once authenticated, go to our model card [here](https://huggingface.co/liguang0115/vmem) and enter your information for access.

On the shared cluster, the benchmark scripts force model downloads into
`/home/group/rudra-work/.cache` before importing Hugging Face, Transformers,
Diffusers, OpenCLIP, or Torch Hub code. This keeps checkpoints off the small
home directory even if the shell has `XDG_CACHE_HOME` pointed somewhere else.
Check the active locations with:

```bash
python scripts/check_model_cache.py
```

To prefetch the core VMem/CUT3R checkpoints before launching a run:

```bash
python scripts/check_model_cache.py --download-core
```

The workspace Depth Pro fork keeps its Apple checkpoint at
`/home/group/rudra-work/projects/into_the_unknown/src/depth_pro/checkpoints/depth_pro.pt`.
Download it explicitly with:

```bash
python scripts/check_model_cache.py --download-depth-pro
```

We provide a demo for you to interact with `VMem`. Simply run

```bash
python app.py
```

## Simple Video + Point Cloud Runner

For a non-interactive run that takes one RGB image, moves forward, looks left or
right, saves the video, runs the workspace fork of Depth Pro, and exports a point
cloud:

```bash
python scripts/generate_simple_video.py --left --input test_samples/oxford.jpg
```

or:

```bash
python scripts/generate_simple_video.py --right --input test_samples/oxford.jpg
```

By default this writes:

```text
outputs/oxford_forward_left.mp4
outputs/oxford_forward_left_pointcloud.ply
outputs/oxford_forward_left_depths/frame_0000.npy
outputs/oxford_forward_left_metadata.json
```

Use `--output` to choose the video path. The point cloud, depth maps, and
metadata are written next to that video unless you pass `--pointcloud-output` or
`--depth-dir`.

The forward move and final look are smoothed separately. By default the runner
uses `--forward-frames 5` and `--turn-frames 12`; increase `--turn-frames` for a
slower left/right look. The old `--interpolation-frames` flag is still accepted
as a forward-frame alias, so existing commands keep working.

The runner uses a fixed focal length for all Depth Pro back-projection, defaulting
to `576 px`. Override it with `--focal-px` if you want a different camera model:

```bash
python scripts/generate_simple_video.py --left --input test_samples/oxford.jpg --focal-px 700
```

For benchmark crops, pass the GT camera metadata so the square VMem input is
only an internal representation. The crop is padded into the square model frame,
generated frames are restored to the original crop size, and Depth Pro
back-projection uses `K_crop` from the Matterport package:

```bash
python scripts/generate_simple_video.py \
  --left \
  --input benchmark_jobs/01_bed_left_1013/input_crop.png \
  --camera-json benchmark_jobs/01_bed_left_1013/camera.json \
  --image-transform-mode pad \
  --restore-input-size
```

The command prints `model cache: /home/group/rudra-work/.cache/huggingface`
before model loading. If you want existing cache environment variables to win
instead, set `VMEM_FORCE_RUDRA_CACHE=0`.

## Representative Benchmark Jobs

Prepare 10 one-sample jobs, one per robot-search category, from the benchmark
dataset-card representatives:

```bash
python scripts/prepare_benchmark_jobs.py --overwrite
```

Run them from an environment with VMem dependencies installed:

```bash
python scripts/run_representative_10.py
```

If the orchestration script is launched from a different Python than the VMem
environment, point it at the right interpreter:

```bash
python scripts/run_representative_10.py --python /path/to/vmem/bin/python
```

Run only the left or right subset:

```bash
python scripts/run_representative_10.py --side left
python scripts/run_representative_10.py --side right
```

The copied crop inputs, camera metadata, exact task table, and local run notes
live in `benchmark_jobs/README.md` and `benchmark_jobs/TASKS.md`.

Before treating the jobs as benchmarkable, verify the SegFormer checker path:

```bash
python scripts/evaluate_representative_10_segformer.py --install-check
```

After the VMem videos exist, run the target-object check:

```bash
python scripts/evaluate_representative_10_segformer.py
```

The runner prefers the forked Depth Pro wrapper that already exists at the root
of this workspace (`src/depth_pro`). It is imported directly by adding the
workspace root to `PYTHONPATH`, so no separate Depth Pro package install is
needed for this checkout. Make sure the Depth Pro checkpoint exists:

```bash
cd /home/group/rudra-work/projects/into_the_unknown
python -m src.depth_pro.src.downloader
```

If you are using this `studies/vmem` folder outside the full workspace, install
your forked Depth Pro checkout into the same environment instead:

```bash
git clone <your-depth-pro-fork-url> ml-depth-pro
cd ml-depth-pro
pip install -e .
source get_pretrained_models.sh
```


## :heart: Acknowledgement
This work is built on top of [CUT3R](https://github.com/CUT3R/CUT3R), [DUSt3R](https://github.com/naver/dust3r) and [Stable Virtual Camera](https://github.com/stability-ai/stable-virtual-camera). We thank them for their great works.





# :books: Citing

If you find this repository useful, please consider giving a star :star: and citation.

```
@article{li2025vmem,
  title={VMem: Consistent Interactive Video Scene Generation with Surfel-Indexed View Memory},
  author={Li, Runjia and Torr, Philip and Vedaldi, Andrea and Jakab, Tomas},
  journal={arXiv preprint arXiv:2506.18903},
  year={2025}
}
```
