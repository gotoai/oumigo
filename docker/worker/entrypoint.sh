#!/bin/sh
# OumiGo worker container entrypoint. Configuration comes from the environment:
#
#   WK_MANAGER_URL             required, e.g. https://oumigo.example.com
#   WK_JOIN_TOKEN              required (unless the manager runs without worker auth)
#   WK_TRANSPORT               default: tunnel
#   WK_HOST_ID                 stable node identity; default: $RUNPOD_POD_ID when set
#   WK_MODEL_PROFILE           the catalog profile (= model id) to run; default: the
#                              fleet's. Several, comma-separated and largest first,
#                              share the GPU: one worker each, started in turn
#   WK_DRAIN_TIMEOUT           optional: seconds a stopping worker waits for requests
#                              in progress (default: the profile's agent.turn_timeout,
#                              else 120)
#   WK_MODEL_STORAGE_LOCATION  optional: weights already on this node's disk (file:///...)
#   HF_TOKEN                   for gated models
#
# See docs/environmental_variables.md. Names from before 0.5.8 (OUMIGO_*) are refused.
#
# Extra arguments are passed to `oumigo worker run` (e.g. --backend transformer).
set -eu

if [ -z "${WK_MANAGER_URL:-}" ]; then
    echo "oumigo-worker: WK_MANAGER_URL is not set" >&2
    exit 2
fi

# A container's own hostname changes on every start; prefer an identity the platform
# keeps stable, so a restarted container rejoins as the same node.
if [ -z "${WK_HOST_ID:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    WK_HOST_ID="$RUNPOD_POD_ID"
    export WK_HOST_ID
fi

mkdir -p /data/oumigo-state "$HF_HOME" "$VLLM_CACHE_ROOT" "$FLASHINFER_WORKSPACE_BASE"

# An optional /data/.env (mounted by the operator) is read too; the environment wins.
exec oumigo worker run --state-dir /data/oumigo-state "$@"
