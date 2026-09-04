#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="${HERE}/results"
mkdir -p "${RESULTS}"

# Build/load once before profiling, so extension compilation is outside ncu.
python "${HERE}/profile_case.py" --case dense_split --warmup 1

METRICS="gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors_op_read.sum,lts__t_sector_op_read_hit_rate.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,sm__throughput.avg.pct_of_peak_sustained_elapsed"

for CASE in dense_split paged_contiguous paged_random; do
  set +e
  ncu --target-processes all \
      --kernel-name regex:flash_fwd_splitkv_kernel \
      --launch-skip 5 --launch-count 1 \
      --metrics "${METRICS}" \
      --csv --log-file "${RESULTS}/ncu_${CASE}.csv" \
      python "${HERE}/profile_case.py" --case "${CASE}" --warmup 5
  STATUS=$?
  set -e
  if grep -q ERR_NVGPUCTRPERM "${RESULTS}/ncu_${CASE}.csv"; then
    cat "${RESULTS}/ncu_${CASE}.csv"
    echo "Enable NVIDIA GPU performance counters, then rerun this script." >&2
    exit 2
  fi
  if [[ ${STATUS} -ne 0 ]]; then
    cat "${RESULTS}/ncu_${CASE}.csv"
    exit "${STATUS}"
  fi
done

echo "Nsight Compute CSV files are in ${RESULTS}"
