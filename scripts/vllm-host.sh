#!/usr/bin/env bash
# vLLM backend (GPU_BACKEND=vllm): runs `vllm serve` from a host virtualenv as a
# systemd --user unit, instead of a container. Reason: on gfx1201 the validated
# stack is a host install (vLLM 0.19.0+rocm721); the public vLLM ROCm images lag it.
# Usage: scripts/vllm-host.sh start|stop|status
set -euo pipefail
cd "$(dirname "$0")/.."
UNIT=lechat-vllm

# Read single keys: shell environment first (like docker compose), then .env.
# Do NOT source .env: TEMP=<number> there crashes the HIP runtime.
env_get() { [ -n "${!1:-}" ] && { printf '%s' "${!1}"; return; };  grep -E "^$1=" .env 2>/dev/null | head -1 | cut -d= -f2- | sed 's/[[:space:]]*#.*$//; s/^["'\'']//; s/["'\'']$//' || true; }

case "${1:-}" in
  start)
    VLLM_BIN=$(env_get VLLM_BIN);       VLLM_BIN=${VLLM_BIN:-$HOME/venvs/vllm/bin/vllm}
    MODELS=$(env_get VLLM_MODELS_DIR);  MODELS=${MODELS:-./models}
    MODELS=$(realpath -m "$MODELS")     # the unit does not run in this directory
    MODEL=$(env_get VLLM_MODEL)
    NAME=$(env_get MODEL_NAME)
    PORT=$(env_get PORT_EXTERNAL);      PORT=${PORT:-8001}
    BIND=$(env_get BIND_ADDRESS);       BIND=${BIND:-127.0.0.1}
    TP=$(env_get VLLM_TP);              TP=${TP:-1}
    CTX=$(env_get VLLM_CTX);            CTX=${CTX:-32768}
    UTIL=$(env_get VLLM_GPU_UTIL);      UTIL=${UTIL:-0.85}
    PARSER=$(env_get VLLM_TOOL_PARSER)
    VISIBLE=$(env_get VLLM_VISIBLE_DEVICES); VISIBLE=${VISIBLE:-0,1}
    [ -x "$VLLM_BIN" ] || { echo "vllm not found at $VLLM_BIN (set VLLM_BIN)"; exit 1; }
    [ -n "$MODEL" ] || { echo "set VLLM_MODEL in .env"; exit 1; }
    EXTRA=()
    [ -n "$PARSER" ] && EXTRA+=(--enable-auto-tool-choice --tool-call-parser "$PARSER")
    # gfx1201: env floor from the NaviMed Reproduce page (PYTORCH_ALLOC_CONF must stay unset;
    # a user unit does not inherit it from the shell); --enforce-eager is mandatory
    # (graph capture crashes with HSA_STATUS_ERROR_INVALID_PACKET_FORMAT).
    systemd-run --user --unit="$UNIT" --collect \
      -E VLLM_ROCM_USE_AITER=0 -E AMD_SERIALIZE_KERNEL=1 -E HIP_LAUNCH_BLOCKING=1 \
      -E ROCR_VISIBLE_DEVICES="$VISIBLE" -E HF_HUB_OFFLINE=1 \
      "$VLLM_BIN" serve "$MODELS/$MODEL" --served-model-name "$NAME" \
        --host "$BIND" --port "$PORT" --tensor-parallel-size "$TP" \
        --max-model-len "$CTX" --gpu-memory-utilization "$UTIL" --enforce-eager "${EXTRA[@]}"
    echo "started $UNIT on $BIND:$PORT (logs: journalctl --user -u $UNIT -f)"
    ;;
  stop)   systemctl --user stop "$UNIT" 2>/dev/null || true ;;
  status) systemctl --user status "$UNIT" --no-pager | head -5 ;;
  *) echo "usage: $0 start|stop|status"; exit 2 ;;
esac
