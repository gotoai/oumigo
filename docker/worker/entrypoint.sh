#!/bin/sh
# OumiGo worker container entrypoint. Configuration comes from the environment:
#
#   OUMIGO_MANAGER_URL   required, e.g. https://oumigo.example.com
#   OUMIGO_JOIN_TOKEN    required (unless the manager runs without worker auth)
#   OUMIGO_TRANSPORT     default: tunnel
#   OUMIGO_HOST_ID       stable node identity; default: $RUNPOD_POD_ID when set
#   HF_TOKEN             for gated models
#   MODEL_NAME / MAX_MODEL_LEN / MODEL_STORAGE_LOCATION   optional per-worker overrides
#
# Extra arguments are passed to `oumigo worker run` (e.g. --backend transformer).
set -eu

if [ -z "${OUMIGO_MANAGER_URL:-}" ]; then
    echo "oumigo-worker: OUMIGO_MANAGER_URL is not set" >&2
    exit 2
fi

# A container's own hostname changes on every start; prefer an identity the platform
# keeps stable, so a restarted container rejoins as the same node.
if [ -z "${OUMIGO_HOST_ID:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    OUMIGO_HOST_ID="$RUNPOD_POD_ID"
    export OUMIGO_HOST_ID
fi

mkdir -p /data/oumigo-state "$HF_HOME" "$VLLM_CACHE_ROOT" "$FLASHINFER_WORKSPACE_BASE"

# An optional /data/.env (mounted by the operator) is read too; the environment wins.
exec oumigo worker run --state-dir /data/oumigo-state "$@"
