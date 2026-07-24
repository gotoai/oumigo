"""oumigo services — the manager and worker *servers* driven by the CLI.

``service.manager`` is the manager node (control plane + data-plane router); ``service.worker``
is the worker node (coordinator + vLLM/HF supervisor). The client-side *handles* that talk to
these servers live under ``oumigo.api``.
"""
