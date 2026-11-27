"""Which GPUs a worker runs on: `--gpu-id` / `$WK_GPU_ID`.

A machine with several GPUs can run one worker per GPU. The worker passes its choice to
the backend as `CUDA_VISIBLE_DEVICES`, and uses the same GPUs for everything it reads
itself: the profile's `min_vram_gib` check, the `gpu_memory_target_gib` share, the
capabilities it registers and the GPU metrics it reports.

* A GPU id is what `nvidia-smi` shows: an index counted from 0, or a UUID (`GPU-…`, from
  `nvidia-smi -L`; also `MIG-…`), which survives cards moving between slots. A model
  sharded over several GPUs (`tensor_parallel_size`) takes a comma-separated list.
* `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set with it: CUDA's own default order puts the
  fastest GPU first, so on a machine with two different cards `1` could otherwise mean
  another GPU to the backend than to `nvidia-smi` and to this worker.
* Without `--gpu-id`, a `CUDA_VISIBLE_DEVICES` already in the environment is honoured
  the same way; without either, the worker uses what it always did: GPU 0 for its
  checks, every GPU for metrics.
"""

from __future__ import annotations

import os
import re

from oumigo.common.envnames import WK_GPU_ID

ENV_GPU_ID = WK_GPU_ID
_ID = re.compile(r"^(\d+|(GPU|MIG)-[0-9A-Za-z-]+)$")


def parse_gpu_ids(value: str) -> list[str]:
    ids = [part.strip() for part in value.split(",") if part.strip()]
    if not ids:
        raise ValueError("no GPU id given")
    bad = [part for part in ids if not _ID.match(part)]
    if bad:
        raise ValueError(f"not a GPU id: {', '.join(bad)} (use nvidia-smi's index, from 0, "
                         "or a UUID such as GPU-…)")
    return ids


def select_gpus(gpu_id: str | None) -> list[str] | None:
    """Apply `--gpu-id` to the environment the backend inherits, and return the GPUs this
    worker uses; None when nothing chooses them (GPU 0, as before)."""
    if gpu_id:
        ids = parse_gpu_ids(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ids)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        return ids
    return selected_gpus()


def selected_gpus() -> list[str] | None:
    """The GPUs `CUDA_VISIBLE_DEVICES` names (set by `select_gpus`, or by the operator),
    in nvidia-smi's terms; None when it is unset or not a list of GPU ids."""
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not value:
        return None
    try:
        ids = parse_gpu_ids(value)
    except ValueError:
        return None
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    return ids


def first_gpu() -> str:
    """The GPU a one-GPU check reads: the first one chosen, else GPU 0."""
    ids = selected_gpus()
    return ids[0] if ids else "0"
