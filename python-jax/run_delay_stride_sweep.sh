#!/usr/bin/env bash
set -euo pipefail

UV_BIN="${UV_BIN:-/tmp/rc_fhn_uv_bin/uv}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT}/../logs/delay_stride_${STAMP}"
mkdir -p "${LOG_DIR}"

run_case() {
  local gpu="$1" stride="$2"
  local name="d16_s${stride}"
  echo "Starting ${name} on GPU ${gpu}: window=$((15 * stride)) samples"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${UV_BIN}" run --locked \
    lorenz-delay-fp 44 \
    --lift-kind polynomial --hidden 256 \
    --network-seed 43 --local-seed 44 \
    --delays 16 --delay-stride "${stride}" \
    --fp-weight 0.1 --local-weight 0.05 --tangent-weight 0 \
    --no-figs --require-gpu >"${LOG_DIR}/${name}.log" 2>&1
}

(run_case 0 6; run_case 0 8; run_case 0 10) & queue0=$!
(run_case 1 7; run_case 1 9; run_case 1 12) & queue1=$!

status=0
wait "${queue0}" || status=$?
wait "${queue1}" || status=$?

{
  echo "Delay stride sweep: 16 delays, polynomial lift, MLP(256), network seed 43, local seed 44"
  grep -H "^RESULT " "${LOG_DIR}"/*.log || true
} | tee "${LOG_DIR}/summary.txt"

echo "Logs: ${LOG_DIR}"
exit "${status}"
