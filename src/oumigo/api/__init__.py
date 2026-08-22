"""oumigo client API — the library surface for driving a fleet from Python.

Bundles the manager/worker *handles* (:class:`OumiGoManager`, :class:`OumiGoWorker`), the
spawn-or-attach entry points (:func:`oumigo_get_or_create_manager`,
:func:`oumigo_create_worker`), and the inference layer (:mod:`oumigo.api.agent` —
``Tool``/``@tool``, ``OumiGoAgent``/``OumiGoChat``/``OumiGoResponse``). The corresponding
*services* (the manager/worker servers these handles talk to) live under
``oumigo.service``.
"""

from __future__ import annotations

from oumigo.api.agent import (
    OumiGoAgent,
    OumiGoChat,
    OumiGoResponse,
    Tool,
    ToolDefinitionError,
    tool,
)
from oumigo.api.api import oumigo_create_worker, oumigo_get_or_create_manager
from oumigo.api.manager.manager import OumiGoManager
from oumigo.api.worker.worker import OumiGoWorker
from oumigo.guard import Guard, GuardContext, GuardProfile, InterceptPoint, Verdict

__all__ = [
    "Guard",
    "GuardContext",
    "GuardProfile",
    "InterceptPoint",
    "OumiGoAgent",
    "OumiGoChat",
    "OumiGoManager",
    "OumiGoResponse",
    "OumiGoWorker",
    "Tool",
    "ToolDefinitionError",
    "Verdict",
    "oumigo_create_worker",
    "oumigo_get_or_create_manager",
    "tool",
]
