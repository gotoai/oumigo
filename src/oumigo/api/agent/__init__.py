"""oumigo agent layer — the tool + Agent/Chat/request inference surface.

Exports the tool-definition surface (:func:`tool`, :class:`Tool`, :class:`ToolDefinitionError`)
and the three inference tiers (:class:`OumiGoAgent`, :class:`OumiGoChat`, :class:`OumiGoResponse`),
which layer on the manager's OpenAI-compatible data plane.
"""

from __future__ import annotations

from oumigo.api.agent.agent import OumiGoAgent
from oumigo.api.agent.chat import OumiGoChat
from oumigo.api.agent.response import OumiGoResponse
from oumigo.api.agent.tool import Tool, ToolDefinitionError, tool

__all__ = [
    "OumiGoAgent",
    "OumiGoChat",
    "OumiGoResponse",
    "Tool",
    "ToolDefinitionError",
    "tool",
]
