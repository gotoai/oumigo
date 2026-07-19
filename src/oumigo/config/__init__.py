"""Foundation: configuration schemas and precedence resolution.

Shared by both the worker (L1) and the manager (L3); owned by neither. The whole
system resolves inputs (CLI > env > file > defaults) down to validated specs.
"""

from oumigo.config.locate import config_file_search_path, resolve_config_file

__all__ = ["config_file_search_path", "resolve_config_file"]
