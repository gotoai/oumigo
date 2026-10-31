"""Typed configuration schemas — the validated specs everything resolves to.

`NodeSpec` describes one vLLM replica (the concrete args for one server). It is
also the wire payload the manager hands a worker before registration, so the worker
knows exactly what `vllm serve` to run. It is resolved from the worker's chosen
catalog profile (`oumigo.config.catalog`), whose id is also the name it serves under.

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
        description="Weights to load: the vendor's Hub repo (e.g. `google/gemma-4-12B-it`).",
    )
    profile: str | None = Field(
        default=None,
        description="Catalog profile id, e.g. `gemma-4-12B-it@128k`: the model id clients "
                    "request and the name the backend serves under. None only for a spec "
                    "from a pre-0.5 manager.",
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

    # Router-side capability caps (like `max_concurrent_requests`, these never reach
    # `vllm serve`). They encode limits of the *model* the fleet serves, which only the
    # manager knows — see `oumigo.service.manager.router.shaping` for the measurements.
    max_audio_seconds: float | None = Field(
        default=None, gt=0,
        description="Cap on total audio in one chat request, in seconds. Audio past it "
                    "is trimmed whole-clip and the model is told; the response carries "
                    "`x-oumigo-audio-*` headers. None disables the cap.",
    )
    max_output_tokens: int | None = Field(
        default=None, ge=1,
        description="Ceiling on a request's `max_tokens`, also applied when the client "
                    "sends none. Bounds a model that degenerates into repetition instead "
                    "of letting it hold a worker slot. None disables the clamp.",
    )
    download_dir: str | None = Field(
        default=None, description="Where vLLM downloads weights; None uses HF_HOME / the HF default."
    )
    extra_args: list[str] = Field(
        default_factory=list, description="Verbatim extra `vllm serve` flags (escape hatch)."
    )

    # The profile's requirements, checked by the worker before the backend starts.
    min_vram_gib: float | None = Field(
        default=None, gt=0, description="Smallest GPU (total VRAM per GPU) the profile fits.")
    requires_vllm: str | None = Field(
        default=None, description="vLLM version specifier the profile needs, e.g. `>=0.28`.")


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
    def served_name(self) -> str:
        """The model id the backend answers to: the profile id, else the Hub id."""
        return self.profile or self.model

    @property
    def model_ref(self) -> str:
        """What the backend is actually pointed at: a local directory, or the Hub id.

        Kept distinct from `served_name`, the id the router routes by: a backend serving
        from disk or under a profile id must still answer to that id (see
        `--served-model-name` in the worker's argv builders).
        """
        path = self.local_path
        return str(path) if path is not None else self.model


class ClusterSpec(BaseModel):
    """Desired state of the whole fleet. Expanded into NodeSpecs by the manager."""

    replicas: int = Field(1, description="Number of vLLM replicas to run.")
    # router policy, placement, shared model cache location, ...
