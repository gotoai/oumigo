"""The model catalog: static profiles, and the set a fleet allows.

A **profile** is one way of running one model: the weights plus every setting fixed when
the backend starts (context length, memory share, concurrency, parallelism, extra
`vllm serve` flags), the router's caps for it, client timeouts and hardware
requirements. Two profiles of the same weights are two different models, so the
profile id is also the public model id: clients request it, `/v1/models` lists it, and
the backend serves under it.

Profiles are YAML files named `<id>.yaml`, ids `<repo-name>@<context>[-<variant>]`
(e.g. `gemma-4-31B-it-qat-w4a16-ct@128k`, `Qwen3.8-27B-W8A16@128k-c10`). The ones shipped
with OumiGo live in `oumigo/catalog/<vendor>/`; an operator adds their own through
`catalog.dirs` in `manager.yaml`, and an operator profile overrides a shipped one with
the same id.

`manager.yaml` then names the profiles its fleet **allows** and the **default** a worker
gets when it chooses none. A legacy single `model:` block still works: it becomes a
one-profile catalog (see `FleetCatalog.from_config`).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from oumigo.config.spec import NodeSpec

log = logging.getLogger("oumigo.config.catalog")

SHIPPED_DIR = Path(__file__).resolve().parent.parent / "catalog"

ANY_MODEL = "any-model"  # the wildcard model id: accepts any served model

# `<repo-name>@<context>[-<variant>...]`; context like 32k / 128k / 256k, or `default`
# for a legacy `model:` block without max_model_len.
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*@(?:[0-9]+k|default)(?:-[A-Za-z0-9._]+)*$")


class CatalogError(ValueError):
    """A profile or the `catalog:` block is invalid (raised at manager start)."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EngineSettings(_Strict):
    """Start-time backend settings (map onto `vllm serve` arguments)."""

    max_model_len: int | None = Field(default=None, ge=1)
    gpu_memory_utilization: float = Field(default=0.90, gt=0, le=1)
    dtype: str = "auto"
    tensor_parallel_size: int = Field(default=1, ge=1)
    max_concurrent_requests: int = Field(default=1, ge=1)
    extra_args: list[str] = Field(default_factory=list)


class Caps(_Strict):
    """Limits the router enforces for this model (never sent to the backend)."""

    max_output_tokens: int | None = Field(default=None, ge=1)
    max_audio_seconds: float | None = Field(default=None, gt=0)


class AgentDefaults(_Strict):
    """Client-side timeouts served at `/agent-defaults` (seconds; None = no limit)."""

    turn_timeout: float | None = Field(default=None, gt=0)
    stall_timeout: float | None = Field(default=None, gt=0)


class Requires(_Strict):
    """What a worker must have to run the profile; checked before the backend starts."""

    min_vram_gib: float | None = Field(default=None, gt=0)
    vllm: str | None = None  # a version specifier, e.g. ">=0.28"


class Profile(_Strict):
    """One model as the fleet serves it. See the module docstring."""

    id: str
    model: str = Field(description="Weights to load: the vendor's Hugging Face repo.")
    description: str | None = None
    storage_location: str | None = None  # fleet-wide hint; workers may override
    download_dir: str | None = None
    engine: EngineSettings = Field(default_factory=EngineSettings)
    caps: Caps = Field(default_factory=Caps)
    agent: AgentDefaults | None = None  # None: the manager's `agent:` block applies
    requires: Requires = Field(default_factory=Requires)
    tested: str | None = None  # informational: hardware and backend version verified

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        if not _ID_RE.match(value):
            raise ValueError(
                f"profile id {value!r} must look like <repo-name>@<context>[-<variant>], "
                "e.g. gemma-4-31B-it-qat-w4a16-ct@128k")
        return value

    @property
    def vendor(self) -> str:
        """The weights' owner (`google` for `google/gemma-...`), or `""`."""
        owner, sep, _ = self.model.partition("/")
        return owner if sep else ""

    def node_spec(self, port: int) -> NodeSpec:
        """The concrete backend spec a worker running this profile gets."""
        e = self.engine
        return NodeSpec(
            model=self.model,
            storage_location=self.storage_location,
            port=port,
            dtype=e.dtype,
            tensor_parallel_size=e.tensor_parallel_size,
            gpu_memory_utilization=e.gpu_memory_utilization,
            max_concurrent_requests=e.max_concurrent_requests,
            max_model_len=e.max_model_len,
            max_audio_seconds=self.caps.max_audio_seconds,
            max_output_tokens=self.caps.max_output_tokens,
            download_dir=self.download_dir,
            extra_args=list(e.extra_args),
        )


def context_label(max_model_len: int | None) -> str:
    """The `@<context>` part of an id: 131072 -> `128k`, None -> `default`."""
    if max_model_len is None:
        return "default"
    return f"{round(max_model_len / 1024)}k"


def load_profile(path: Path) -> Profile:
    """Read and validate one profile file; its id must equal the file name."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        profile = Profile.model_validate(data)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise CatalogError(f"invalid profile {path}: {exc}") from exc
    if profile.id != path.stem:
        raise CatalogError(f"profile {path}: id {profile.id!r} must equal the file name "
                           f"{path.stem!r}")
    return profile


def load_profiles(dirs: list[Path] | None = None, shipped: bool = True) -> dict[str, Profile]:
    """All profiles: the shipped ones, then each of `dirs` (later wins on an id clash).

    Within one directory tree an id must be unique: two files with one id is an error,
    not a silent pick.
    """
    profiles: dict[str, Profile] = {}
    sources: dict[str, Path] = {}
    trees = ([SHIPPED_DIR] if shipped else []) + [Path(d).expanduser() for d in dirs or []]
    for tree in trees:
        if tree != SHIPPED_DIR and not tree.is_dir():
            raise CatalogError(f"catalog directory not found: {tree}")
        seen_here: dict[str, Path] = {}
        for path in sorted(tree.rglob("*.yaml")):
            profile = load_profile(path)
            if profile.id in seen_here:
                raise CatalogError(f"duplicate profile id {profile.id!r}: "
                                   f"{seen_here[profile.id]} and {path}")
            seen_here[profile.id] = path
            if profile.id in profiles:
                log.info("profile %s from %s overrides %s", profile.id, path, sources[profile.id])
            profiles[profile.id] = profile
            sources[profile.id] = path
    return profiles


class FleetCatalog:
    """The profiles one fleet allows, and its default."""

    def __init__(self, profiles: dict[str, Profile], default: str | None) -> None:
        if default is not None and default not in profiles:
            raise CatalogError(f"catalog.default {default!r} is not in catalog.allow")
        self.profiles = profiles
        self.default = default

    def get(self, profile_id: str) -> Profile | None:
        return self.profiles.get(profile_id)

    @property
    def default_profile(self) -> Profile | None:
        return self.profiles[self.default] if self.default else None

    @classmethod
    def from_config(cls, config: dict) -> FleetCatalog:
        """Build the fleet catalog from a manager config dict.

        `catalog:` (allow / default / dirs) selects from the loaded profiles. A legacy
        `model:` block instead becomes a one-profile catalog. Neither yields an empty
        catalog (the manager then has nothing to hand out). Both is an error.
        """
        block = config.get("catalog")
        legacy = config.get("model")
        if block is not None and legacy:
            raise CatalogError("manager config has both `catalog:` and `model:`; "
                               "move the model into a profile and keep only `catalog:`")
        if block is not None:
            return cls._from_block(block)
        if legacy and legacy.get("name"):
            profile = legacy_profile(legacy)
            return cls({profile.id: profile}, profile.id)
        return cls({}, None)

    @classmethod
    def _from_block(cls, block: dict) -> FleetCatalog:
        allow = block.get("allow") or []
        if not isinstance(allow, list) or not allow:
            raise CatalogError("catalog.allow must list at least one profile id")
        available = load_profiles([Path(d) for d in block.get("dirs") or []])
        missing = [pid for pid in allow if pid not in available]
        if missing:
            raise CatalogError(f"catalog.allow names unknown profiles: {', '.join(missing)}")
        return cls({pid: available[pid] for pid in allow}, block.get("default"))


def legacy_profile(model: dict) -> Profile:
    """A profile from a legacy `model:` block, keeping that block's own defaults."""
    name = str(model["name"])
    max_len = model.get("max_model_len")
    repo_name = name.rsplit("/", 1)[-1]
    return Profile(
        id=f"{repo_name}@{context_label(max_len)}",
        model=name,
        storage_location=model.get("storage_location"),
        download_dir=model.get("download_dir"),
        engine=EngineSettings(
            max_model_len=max_len,
            gpu_memory_utilization=float(model.get("gpu_memory_utilization", 0.80)),
            dtype=str(model.get("dtype", "auto")),
            tensor_parallel_size=int(model.get("tensor_parallel_size", 1)),
            max_concurrent_requests=int(model.get("max_concurrent_requests", 4)),
            extra_args=list(model.get("extra_args") or []),
        ),
        caps=Caps(
            max_output_tokens=model.get("max_output_tokens"),
            max_audio_seconds=model.get("max_audio_seconds"),
        ),
    )
