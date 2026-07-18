"""Config resolution: merge CLI > env > file > defaults into a validated spec.

Precedence is fixed and conventional. Prefer `pydantic-settings` for env/file
handling over hand-rolled dict merging, so validation and error messages come
for free.
"""

from __future__ import annotations

# def resolve_node_spec(cli=..., env=..., file=...) -> NodeSpec: ...
