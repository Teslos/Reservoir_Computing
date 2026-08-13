#!/usr/bin/env bash
set -euo pipefail

UV_BIN="${UV_BIN:-/tmp/rc_fhn_uv_bin/uv}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT}/../logs/delay_geometry_${STAMP}"
mkdir -p "${LOG_DIR}"

run_case() {
  local gpu="$1" name="$2" delays="$3" stride="$4"
  echo "Starting ${name} on GPU ${gpu}: delays=${delays}, stride=${stride}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${UV_BIN}" run --locked \
    lorenz-delay-fp 44 \
    --lift-kind polynomial --hidden 256 \
    --network-seed 43 --local-seed 44 \
    --delays "${delays}" --delay-stride "${stride}" \
    --fp-weight 0.1 --local-weight 0.05 --tangent-weight 0 \
    --no-figs --require-gpu >"${LOG_DIR}/${name}.log" 2>&1
}

# One sequential queue per GPU prevents memory contention; the queues run in parallel.
(
  run_case 0 d10_s5 10 5
  run_case 0 d16_s5 16 5
  run_case 0 d16_s8 16 8
) &
queue0=$!

(
  run_case 1 d16_s3 16 3
  run_case 1 d20_s5 20 5
) &
queue1=$!

status=0
wait "${queue0}" || status=$?
wait "${queue1}" || status=$?

{
  echo "Delay geometry sweep: polynomial lift, MLP(256), network seed 43, local seed 44"
  grep -H "^RESULT " "${LOG_DIR}"/*.log || true
} | tee "${LOG_DIR}/summary.txt"

echo "Logs: ${LOG_DIR}"
exit "${status}"
