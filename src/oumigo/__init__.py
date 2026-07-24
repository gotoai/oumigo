"""oumigo — a vertical-integration toolkit for running vLLM replica fleets."""

from oumigo.__about__ import __version__
from oumigo.api.agent import (
    OumigoAgent,
    OumigoChat,
    OumigoResponse,
    Tool,
    ToolDefinitionError,
    tool,
)
from oumigo.api import (
    OumigoManager,
    OumigoWorker,
    oumigo_create_worker,
    oumigo_get_or_create_manager,
)

__all__ = [
    "__version__",
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
