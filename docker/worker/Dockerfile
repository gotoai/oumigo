# OumiGo worker image: CUDA + vLLM + OumiGo, nothing deployment-specific.
#
# The manager URL, join token, HF_TOKEN and any model override come from the
# container's environment at start. Model weights are downloaded on first start, never
# baked in. See docs/docker-worker.md.
#
#   docker build --build-arg VLLM_VERSION=0.28.0 --build-arg OUMIGO_VERSION=0.5.0 \
#       -t ghcr.io/<owner>/oumigo-worker:0.5.0-vllm0.28.0 docker/worker

ARG VLLM_VERSION=0.28.0
FROM vllm/vllm-openai:v${VLLM_VERSION}

ARG OUMIGO_VERSION=0.5.0

# tini as PID 1: forwards the platform's stop signal (SIGTERM) to the worker, which
# drains, stops vLLM's whole process group and releases VRAM.
RUN apt-get update \
 && apt-get install -y --no-install-recommends tini \
 && rm -rf /var/lib/apt/lists/*

# Plain `oumigo`, not `oumigo[worker]`: the base image already ships vLLM, torch and the
# multimodal extras, and the extra's transformers pin can fight a newer vLLM.
RUN python3 -m pip install --no-cache-dir "oumigo==${OUMIGO_VERSION}" \
 && (python3 -m pip check || true) \
 && oumigo version

COPY entrypoint.sh /usr/local/bin/oumigo-worker-entrypoint
RUN chmod 755 /usr/local/bin/oumigo-worker-entrypoint

# Weights, caches and node identity live on the container disk under /data.
ENV OUMIGO_TRANSPORT=tunnel \
    HF_HOME=/data/hf \
    VLLM_CACHE_ROOT=/data/vllm-cache \
    FLASHINFER_WORKSPACE_BASE=/data/flashinfer \
    MAX_JOBS=4
WORKDIR /data

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/oumigo-worker-entrypoint"]
