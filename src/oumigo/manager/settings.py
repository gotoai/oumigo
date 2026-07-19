"""Minimal manager-config loading (seed of the future ManagerSettings).

Reads the single manager config file (``manager.yaml`` by convention) as a plain
dict. This is a stepping stone: it will be replaced by a validated pydantic
``ManagerSettings`` (with pydantic-settings for env / .env). For now it exposes
only the few fields the manager currently uses.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from oumigo.config.spec import NodeSpec

DEFAULT_PROVIDER = "LAN"


def load_manager_yaml(config_file: Path | None) -> dict:
    """Load the manager config file, or ``{}`` if absent."""
    if config_file is None:
        return {}
    path = Path(config_file)
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def get_provider_name(config: dict) -> str:
    """The configured provider name, defaulting to LAN."""
    return str(config.get("provider", DEFAULT_PROVIDER))


def build_node_spec(config: dict) -> NodeSpec | None:
    """Build the vLLM NodeSpec handed to workers from the `model` block.

    Returns None if no model is configured (``model.name`` unset) — the manager
    then has nothing to tell workers to serve, and registration returns no spec.
    Local-only concerns (cache dirs) are intentionally excluded; those come from
    each worker's own environment.
    """
    model = config.get("model") or {}
    name = model.get("name")
    if not name:
        return None

    return NodeSpec(
        model=str(name),
        port=int(model.get("port", 8000)),
        dtype=str(model.get("dtype", "auto")),
        tensor_parallel_size=int(model.get("tensor_parallel_size", 1)),
        gpu_memory_utilization=float(model.get("gpu_memory_utilization", 0.90)),
        max_model_len=model.get("max_model_len"),
        download_dir=model.get("download_dir"),
        extra_args=list(model.get("extra_args") or []),
    )
