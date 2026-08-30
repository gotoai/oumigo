"""OumiGo — a vertical-integration toolkit for running vLLM replica fleets."""

from oumigo.__about__ import __version__
from oumigo.api.agent import (
    OumiGoAgent,
    OumiGoChat,
    OumiGoResponse,
    Tool,
    ToolDefinitionError,
    tool,
)
from oumigo.api import (
    OumiGoManager,
    OumiGoWorker,
    oumigo_create_worker,
    oumigo_get_or_create_manager,
)

__all__ = [
    "__version__",
    "OumiGoAgent",
    "OumiGoChat",
    "OumiGoManager",
    "OumiGoResponse",
    "OumiGoWorker",
    "Tool",
    "ToolDefinitionError",
    "oumigo_create_worker",
    "oumigo_get_or_create_manager",
    "tool",
]
