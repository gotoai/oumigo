"""oumigo client API — the library surface for driving a fleet from Python.

Bundles the manager/worker *handles* (:class:`OumigoManager`, :class:`OumigoWorker`), the
spawn-or-attach entry points (:func:`oumigo_get_or_create_manager`,
:func:`oumigo_create_worker`), and the inference layer (:mod:`oumigo.api.agent` —
``Tool``/``@tool``, ``OumigoAgent``/``OumigoChat``/``OumigoResponse``). The corresponding
*services* (the manager/worker servers these handles talk to) live under
``oumigo.service``.
"""

from __future__ import annotations

from oumigo.api.agent import (
    OumigoAgent,
    OumigoChat,
    OumigoResponse,
    Tool,
    ToolDefinitionError,
    tool,
)
from oumigo.api.api import oumigo_create_worker, oumigo_get_or_create_manager
from oumigo.api.manager.manager import OumigoManager
from oumigo.api.worker.worker import OumigoWorker

__all__ = [
    "OumigoAgent",
    "OumigoChat",
    "OumigoManager",
    "OumigoResponse",
    "OumigoWorker",
    "Tool",
    "ToolDefinitionError",
    "oumigo_create_worker",
    "oumigo_get_or_create_manager",
    "tool",
]
