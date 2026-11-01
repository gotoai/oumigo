# OumiGo worker image: CUDA + vLLM + OumiGo, nothing deployment-specific.
#
# The manager URL, join token, HF_TOKEN and any model override come from the
# container's environment at start. Model weights are downloaded on first start, never
# baked in. The variables it reads are listed in entrypoint.sh.
#
#   docker build --build-arg VLLM_VERSION=0.28.0 --build-arg OUMIGO_VERSION=0.5.1 \
#       -t ghcr.io/<owner>/oumigo-worker:0.5.1-vllm0.28.0 docker/worker
#
# VLLM_VARIANT picks another vLLM base build. vLLM's default image needs the host driver
# of its CUDA (13.0: driver 580+); `-cu129` also runs on CUDA 12.9 hosts (driver 575+),
# which widens the pool of cloud machines that can take the pod:
#
#   docker build --build-arg VLLM_VARIANT=-cu129 ... -t ...:0.5.1-vllm0.28.0-cu129 docker/worker

ARG VLLM_VERSION=0.28.0
ARG VLLM_VARIANT=
FROM vllm/vllm-openai:v${VLLM_VERSION}${VLLM_VARIANT}

ARG VLLM_VERSION
ARG VLLM_VARIANT
ARG OUMIGO_VERSION=0.5.1

# The source label links the published package to the repository on GitHub.
LABEL org.opencontainers.image.source="https://github.com/gotoai/oumigo" \
      org.opencontainers.image.description="OumiGo worker: vLLM ${VLLM_VERSION}${VLLM_VARIANT} + oumigo ${OUMIGO_VERSION}" \
      org.opencontainers.image.licenses="MIT"

# tini as PID 1: forwards the platform's stop signal (SIGTERM) to the worker, which
# drains, stops vLLM's whole process group and releases VRAM.
RUN apt-get update \
 && apt-get install -y --no-install-recommends tini \
 && rm -rf /var/lib/apt/lists/*

# Plain `oumigo`, not `oumigo[worker]`: the base image already ships vLLM, torch and
# transformers, and the extra's transformers pin can fight a newer vLLM. What the base
# image lacks is added here, with every package it ships held at its version (pip may
# only add packages, never swap the tested vLLM / torch / CUDA libraries):
#   av scipy soundfile soxr  vLLM's [audio] extra; without them every audio request 500s
#   bitsandbytes             profiles that quantize in flight (e.g. ...@32k-bnb4)
# The extra is installed by its parts, and bitsandbytes without deps (torch, numpy,
# packaging: all present), so pip never re-resolves torch's own pins: the base image
# ships a newer NCCL than torch declares.
RUN python3 -m pip list --format=freeze > /tmp/base-pins.txt \
 && python3 -m pip install --no-cache-dir -c /tmp/base-pins.txt \
        av scipy soundfile soxr "oumigo==${OUMIGO_VERSION}" \
 && python3 -m pip install --no-cache-dir --no-deps -c /tmp/base-pins.txt bitsandbytes \
 && rm /tmp/base-pins.txt \
 && (python3 -m pip check || true) \
 && python3 -c "import av, scipy, soundfile, soxr, bitsandbytes" \
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
