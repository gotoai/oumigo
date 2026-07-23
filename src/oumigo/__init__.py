"""oumigo — a vertical-integration toolkit for running vLLM replica fleets."""

from oumigo.__about__ import __version__
from oumigo.api import (
    OumigoManager,
    OumigoWorker,
    oumigo_create_worker,
    oumigo_get_or_create_manager,
)

__all__ = [
    "__version__",
    "OumigoManager",
    "OumigoWorker",
    "oumigo_create_worker",
    "oumigo_get_or_create_manager",
]
