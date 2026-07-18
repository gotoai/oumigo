"""Minimal manager-config loading (seed of the future ManagerSettings).

Reads ``manager.yaml`` from the config dir as a plain dict. This is a stepping
stone: it will be replaced by a validated pydantic ``ManagerSettings`` (with
pydantic-settings for env / .env). For now it exposes only the few fields the
manager currently uses.
"""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_PROVIDER = "LAN"


def load_manager_yaml(config_dir: Path | None) -> dict:
    """Load ``manager.yaml`` from the config dir, or ``{}`` if absent."""
    if config_dir is None:
        return {}
    path = Path(config_dir) / "manager.yaml"
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def get_provider_name(config: dict) -> str:
    """The configured provider name, defaulting to LAN."""
    return str(config.get("provider", DEFAULT_PROVIDER))
