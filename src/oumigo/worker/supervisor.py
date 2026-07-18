"""vLLMProcess — the subprocess seam.

Builds the `vllm serve` argv from a NodeSpec, spawns it, waits for /health,
exposes status, and kills cleanly. Keep pure argv-building separate from spawning
so "given this spec, produce exactly this argv" is trivially unit-testable.
"""

from __future__ import annotations

# def build_argv(spec: NodeSpec) -> list[str]: ...
#
# class VLLMProcess:
#     """Spawns and supervises a single `vllm serve` child process."""
