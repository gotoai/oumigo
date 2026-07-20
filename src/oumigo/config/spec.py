"""Typed configuration schemas — the validated specs everything resolves to.

`NodeSpec` describes one vLLM replica (the concrete args for one server). It is
also the wire payload the manager hands a worker at registration, so the worker
knows exactly what `vllm serve` to run — a homogeneous fleet means every worker
gets the same spec.

`ClusterSpec` describes the desired fleet; the manager expands it into NodeSpecs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class NodeSpec(BaseModel):
    """Concrete description of a single vLLM replica / worker node.

    Fields map directly onto `vllm serve` arguments (see
    `oumigo.worker.supervisor.build_argv`). Local-only concerns (HF/vLLM cache
    dirs) are NOT here — those come from the worker's own environment.
    """

    model: str = Field(..., description="Model id/path passed to `vllm serve`.")
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


class ClusterSpec(BaseModel):
    """Desired state of the whole fleet. Expanded into NodeSpecs by the manager."""

    replicas: int = Field(1, description="Number of vLLM replicas to run.")
    # router policy, placement, shared model cache location, ...
