ARG BASE_IMAGE=vllm-tream-recover:20260209
FROM ${BASE_IMAGE}

LABEL maintainer="tream-migration"
LABEL description="Portable runtime image for TREAM experiments"

WORKDIR /workspace/vllm

# The bundle exporter prepares code under code_snapshot/vllm.
COPY code_snapshot/vllm/ /workspace/vllm/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/vllm:${PYTHONPATH} \
    TREAM_ROOT=/workspace/vllm/tream \
    HF_HOME=/workspace/vllm/hf_cache

RUN mkdir -p /workspace/vllm/tream/inference_logs \
    /workspace/vllm/tream/training_logs \
    /workspace/vllm/tream/wandb \
    /workspace/vllm/tream/tmp

CMD ["sleep", "infinity"]
