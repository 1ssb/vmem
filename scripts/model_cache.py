#!/usr/bin/env python
"""Model cache routing for VMem benchmark scripts."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_CACHE_ROOT = Path("/home/group/rudra-work/.cache")


def _truthy_env(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def configure_model_cache(cache_root: str | Path | None = None) -> dict[str, str]:
    """Force model downloads/caches onto the shared rudra-work disk.

    Set VMEM_FORCE_RUDRA_CACHE=0 to let pre-existing cache variables win.
    Set RUDRA_WORK_CACHE=/some/path to choose a different cache root.
    """

    root = Path(os.environ.get("RUDRA_WORK_CACHE") or cache_root or DEFAULT_CACHE_ROOT)
    root = root.expanduser().resolve()
    hf_home = root / "huggingface"
    env = {
        "XDG_CACHE_HOME": root,
        "HF_HOME": hf_home,
        "HUGGINGFACE_HUB_CACHE": hf_home / "hub",
        "HF_HUB_CACHE": hf_home / "hub",
        "HF_ASSETS_CACHE": hf_home / "assets",
        "TRANSFORMERS_CACHE": hf_home / "transformers",
        "TORCH_HOME": root / "torch",
    }

    force = _truthy_env("VMEM_FORCE_RUDRA_CACHE", default=True)
    for key, value in env.items():
        if force or key not in os.environ:
            os.environ[key] = str(value)

    for key in env:
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

    return {key: os.environ[key] for key in env}


def cache_report(cache_env: dict[str, str] | None = None) -> str:
    cache_env = cache_env or configure_model_cache()
    lines = ["model cache environment:"]
    for key in (
        "XDG_CACHE_HOME",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "HF_HUB_CACHE",
        "HF_ASSETS_CACHE",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
    ):
        lines.append(f"  {key}={cache_env[key]}")
    return "\n".join(lines)
