#!/bin/bash
set -e

# Environment variables are loaded from .env
# Load .env file
if [ -f .env ]; then
    set -a
    . .env
    set +a
fi

# Validate model file exists
MODEL_PATH="/models/${MODEL_NAME}.gguf"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH" >&2
    echo "Available models in /models/:" >&2
    ls -1 /models/ 2>/dev/null || echo "No models found"
    exit 1
fi

# Build server command from env vars
SERVER_CMD=(
    /app/llama-server
    -m "${MODEL_PATH}"
    --host "${HOST}"
    --port "${PORT_INTERNAL}"
    --ctx-size "${CTX_SIZE}"
    --flash-attn "${FLASH_ATTN}"
    --batch-size "${BATCH_SIZE}"
    --cache-type-k "${CACHE_TYPE_K}"
    --cache-type-v "${CACHE_TYPE_V}"
    --parallel "${PARALLEL}"
    --cache-ram "${CACHE_RAM}"
    --threads "${THREADS}"
)

if [[ "${NO_MMAP}" == "true" ]]; then
    SERVER_CMD+=(--no-mmap)
fi

# GPU offload / multi-GPU split
if [[ "${FORCE_CPU}" == "on" ]]; then
    SERVER_CMD+=(--n-gpu-layers 0)
else
    SERVER_CMD+=(--n-gpu-layers "${GPU_LAYERS:-all}")
    if [[ -n "${SPLIT_MODE}" ]]; then
        SERVER_CMD+=(--split-mode "${SPLIT_MODE}")
    fi
    if [[ -n "${TENSOR_SPLIT}" ]]; then
        SERVER_CMD+=(--tensor-split "${TENSOR_SPLIT}")
    fi
    if [[ -n "${MAIN_GPU}" ]]; then
        SERVER_CMD+=(--main-gpu "${MAIN_GPU}")
    fi
fi

if [[ "${FIT}" == "on" ]]; then
    SERVER_CMD+=(--fit "${FIT}")
    SERVER_CMD+=(--fit-target "${FIT_TARGET}")
    SERVER_CMD+=(--fit-ctx "${FIT_CTX}")
fi

if [[ -n "${UBATCH_SIZE}" ]]; then
    SERVER_CMD+=(--ubatch-size "${UBATCH_SIZE}")
fi

if [[ -n "${CTX_CHECKPOINTS}" ]]; then
    SERVER_CMD+=(--ctx-checkpoints "${CTX_CHECKPOINTS}")
fi

if [[ -n "${POLL_BATCH}" ]]; then
    SERVER_CMD+=(--poll-batch "${POLL_BATCH}")
fi

# Sampling parameters
if [[ -n "${TEMP}" ]]; then
    SERVER_CMD+=(--temp "${TEMP}")
fi
if [[ -n "${TOP_P}" ]]; then
    SERVER_CMD+=(--top-p "${TOP_P}")
fi
if [[ -n "${TOP_K}" ]]; then
    SERVER_CMD+=(--top-k "${TOP_K}")
fi
if [[ -n "${MIN_P}" ]]; then
    SERVER_CMD+=(--min-p "${MIN_P}")
fi

# Penalties
if [[ -n "${PRESENCE_PENALTY}" ]]; then
    SERVER_CMD+=(--presence-penalty "${PRESENCE_PENALTY}")
fi
if [[ -n "${REPEAT_PENALTY}" ]]; then
    SERVER_CMD+=(--repeat-penalty "${REPEAT_PENALTY}")
fi

# Features
case "${REASONING}" in
    on|true|1)
        SERVER_CMD+=(--reasoning on)
        ;;
    off|false|0)
        SERVER_CMD+=(--reasoning off)
        ;;
esac

# Jinja chat template
if [[ -n "${CHAT_TEMPLATE}" && -f "${CHAT_TEMPLATE}" ]]; then
    SERVER_CMD+=(--jinja)
    SERVER_CMD+=(--chat-template "$(cat "${CHAT_TEMPLATE}")")
fi

# Print startup configuration
echo "Starting llama.cpp server..."
echo "  Model:      ${MODEL_NAME}"
echo "  Model path: ${MODEL_PATH}"
echo "  Host:       ${HOST}"
echo "  Port (internal):  ${PORT_INTERNAL}"
echo "  Port (external):  ${PORT_EXTERNAL}"
echo "  Context:    ${CTX_SIZE}"
echo "  Flash Attn: ${FLASH_ATTN}"
echo "  Threads:    ${THREADS}"
if [[ "${FORCE_CPU}" == "on" ]]; then
    echo "  Offload:    CPU only"
else
    echo "  Offload:    GPU layers=${GPU_LAYERS:-all} split=${SPLIT_MODE:-layer} tensor-split=${TENSOR_SPLIT:-auto}"
fi
echo "  Fit:        ${FIT:-off}"
if [[ "${FIT}" == "on" ]]; then
    echo "  Fit Target: ${FIT_TARGET}"
    echo "  Fit Ctx:    ${FIT_CTX}"
fi
echo "  Sampling:   temp=${TEMP:-1.0} top_p=${TOP_P:-1.0} top_k=${TOP_K:-0} min_p=${MIN_P:-0.0}"
echo "  Penalties:  presence=${PRESENCE_PENALTY:-0.0} repeat=${REPEAT_PENALTY:-1.0}"
echo ""

# TEMP is already in SERVER_CMD as --temp. The ROCm/HIP runtime also reads TEMP
# (as a temp-dir path) and segfaults in libamdhip64 when it holds "0.6".
unset TEMP

# Run the server in the background so we can pre-warm it before clients connect.
"${SERVER_CMD[@]}" &
SERVER_PID=$!

trap 'kill -TERM "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID"; exit 0' TERM INT

# Wait for the HTTP server to come up.
for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${PORT_INTERNAL}/health" >/dev/null 2>&1; then
        break
    fi
    sleep 2
done

# Pre-warm the GPU/CUDA kernels; otherwise the first client request pays a
# one-time multi-token prompt-processing cost (tens of seconds on a cold
# process). A single token is not enough, so use a short paragraph.
WARMUP_PROMPT='The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.'
curl -sf "http://127.0.0.1:${PORT_INTERNAL}/completion" \
    -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"${WARMUP_PROMPT}\",\"n_predict\":2,\"temperature\":0}" >/dev/null 2>&1 || true

wait "$SERVER_PID"