"""Typed configuration schemas — the validated specs everything resolves to.

`NodeSpec` describes one vLLM replica (the concrete args for one server).
`ClusterSpec` describes the desired fleet; the manager expands it into NodeSpecs.

Fields below are placeholders to be fleshed out during implementation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class NodeSpec(BaseModel):
    """Concrete description of a single vLLM replica / worker node."""

    model: str = Field(..., description="Model id/path passed to `vllm serve`.")
    port: int = Field(8000, description="Port the vLLM server binds (0.0.0.0).")
    # vLLM knobs: dtype, tensor_parallel_size, gpu_memory_utilization, download_dir, ...
    # Provider placeholders (inert until L2 lands): image_id, flavor, gpu_count, ...


class ClusterSpec(BaseModel):
    """Desired state of the whole fleet. Expanded into NodeSpecs by the manager."""

    replicas: int = Field(1, description="Number of vLLM replicas to run.")
    # router policy, placement, shared model cache location, ...
