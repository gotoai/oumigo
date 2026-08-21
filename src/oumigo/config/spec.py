"""Typed configuration schemas — the validated specs everything resolves to.

`NodeSpec` describes one vLLM replica (the concrete args for one server). It is
also the wire payload the manager hands a worker at registration, so the worker
knows exactly what `vllm serve` to run — a homogeneous fleet means every worker
gets the same spec.

`ClusterSpec` describes the desired fleet; the manager expands it into NodeSpecs.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, Field, field_validator


class NodeSpec(BaseModel):
    """Concrete description of a single vLLM replica / worker node.

    Fields map directly onto `vllm serve` arguments (see
    `oumigo.service.worker.supervisor.build_argv`). Local-only concerns (HF/vLLM cache
    dirs) are NOT here — those come from the worker's own environment.
    """

    model: str = Field(
        ...,
        description="Canonical model id (e.g. `google/gemma-4-12B-it`). Also the name "
                    "clients address: the router rewrites every request's `model` to it.",
    )
    storage_location: str | None = Field(
        default=None,
        description="Where the weights actually live, as a URI. Only `file://` is "
                    "supported: `file:///srv/models/gemma-4-12B-it`. None (or the string "
                    "`none`) means download `model` from the Hugging Face Hub.",
    )
    host: str = Field(default="0.0.0.0", description="Host the vLLM server binds.")
    port: int = Field(default=7001, description="Port the vLLM server binds.")

    # vLLM tuning knobs (mirror the manager.yaml `model:` block).
    dtype: str = Field(
        default="auto", description="vLLM dtype: auto | bfloat16 | float16."
    )
    tensor_parallel_size: int = Field(
        default=1, ge=1, description="GPUs to shard the model across."
    )
    gpu_memory_utilization: float = Field(
        default=0.90, gt=0, le=1, description="Fraction of VRAM vLLM may use."
    )
    max_concurrent_requests: int = Field(
        default=4, ge=1,
        description="Router admission cap: max in-flight requests sent to one worker "
                    "before others are preferred / the request queues. Negotiable — the "
                    "manager seeds this value; a worker may report its own on heartbeat.",
    )
    max_model_len: int | None = Field(
        default=None, description="Max context length; None uses the model default."
    )
    download_dir: str | None = Field(
        default=None, description="Where vLLM downloads weights; None uses HF_HOME / the HF default."
    )
    extra_args: list[str] = Field(
        default_factory=list, description="Verbatim extra `vllm serve` flags (escape hatch)."
    )


    @field_validator("storage_location", mode="before")
    @classmethod
    def _normalize_storage_location(cls, value: object) -> str | None:
        """Accept YAML's several spellings of "unset", and reject unsupported schemes.

        `null`, an empty string and the literal word `none` (any case) all mean "use the
        Hugging Face Hub", so a config can disable a location without deleting the key.
        Anything else must be a `file://` URI — remote protocols are not implemented, and
        silently treating `s3://...` as a relative path would be far worse than refusing.
        """
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() in ("none", "null"):
            return None

        parsed = urlparse(text)
        if parsed.scheme != "file":
            supported = "file:///absolute/path"
            hint = (f" Did you mean file://{text}?" if not parsed.scheme
                    else f" {parsed.scheme}:// is not supported yet.")
            raise ValueError(f"storage_location must be a {supported} URI.{hint}")
        if parsed.netloc not in ("", "localhost"):
            raise ValueError(
                f"storage_location {text!r} names a remote host ({parsed.netloc!r}); "
                f"only local paths are supported (file:///path or file://localhost/path)")
        if not parsed.path.startswith("/"):
            raise ValueError(
                f"storage_location {text!r} is not absolute; use file:///absolute/path")
        return text

    @property
    def local_path(self) -> Path | None:
        """`storage_location` as a filesystem path, or None when serving from the Hub."""
        if self.storage_location is None:
            return None
        return Path(unquote(urlparse(self.storage_location).path))

    @property
    def model_ref(self) -> str:
        """What the backend is actually pointed at: a local directory, or the Hub id.

        Kept distinct from `model`, which stays the canonical *name* — the router
        rewrites client requests to it, so a backend serving from disk must still
        answer to it (see `--served-model-name` in the worker's argv builders).
        """
        path = self.local_path
        return str(path) if path is not None else self.model


class ClusterSpec(BaseModel):
    """Desired state of the whole fleet. Expanded into NodeSpecs by the manager."""

    replicas: int = Field(1, description="Number of vLLM replicas to run.")
    # router policy, placement, shared model cache location, ...
