"""A one-profile catalog fleet, for tests that need a manager but not a particular model.

Every fleet has a catalog (0.5.5), so control-plane and router tests build one here
instead of passing a bare spec.
"""

from __future__ import annotations

from oumigo.config.catalog import EngineSettings, FleetCatalog, Profile

PROFILE = "mock-model@8k"
WEIGHTS = "acme/mock-model"


def one_profile_catalog(default: bool = True, **profile) -> FleetCatalog:
    """A catalog allowing just `PROFILE` (made the default unless `default=False`);
    `profile` overrides its fields, e.g. `engine=EngineSettings(...)`."""
    fields = {"id": PROFILE, "model": WEIGHTS,
              "engine": EngineSettings(max_model_len=8192, max_concurrent_requests=4)}
    p = Profile(**{**fields, **profile})
    return FleetCatalog({p.id: p}, p.id if default else None)


def register_body(node_id: str, **extra) -> dict:
    """A worker's `/register` body for `PROFILE`."""
    return {"node_id": node_id, "address": "10.0.0.5", "vllm_port": 7001,
            "profile": PROFILE, **extra}
