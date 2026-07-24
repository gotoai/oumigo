"""oumigo agent layer — the tool + Agent/Chat/request inference surface.

Exports the tool-definition surface (:func:`tool`, :class:`Tool`, :class:`ToolDefinitionError`)
and the three inference tiers (:class:`OumigoAgent`, :class:`OumigoChat`, :class:`OumigoResponse`),
which layer on the manager's OpenAI-compatible data plane.
"""

from __future__ import annotations

from oumigo.api.agent.agent import OumigoAgent
from oumigo.api.agent.chat import OumigoChat
from oumigo.api.agent.response import OumigoResponse
from oumigo.api.agent.tool import Tool, ToolDefinitionError, tool

__all__ = [
    "OumigoAgent",
    "OumigoChat",
    "OumigoResponse",
    "Tool",
    "ToolDefinitionError",
    "tool",
]
