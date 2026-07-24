"""L1 — the worker node.

A long-lived *coordinator* registers with the manager, supervises the vLLM server
as a child process, monitors its health, and executes start/stop/restart commands.
It owns the node state machine and applies a restart-with-give-up policy:
transient failure -> restart with backoff; terminal failure -> report FAILED.

Split into two seams so the pure logic is testable without spawning anything:
  * `supervisor` — builds vLLM argv + manages the subprocess.
  * `coordinator` — the control/reporting surface wrapping the supervisor.
"""
