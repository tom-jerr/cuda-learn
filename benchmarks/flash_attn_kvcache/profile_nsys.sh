#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="${HERE}/results"
mkdir -p "${RESULTS}"

for CASE in dense_split paged_contiguous paged_random; do
  nsys profile --force-overwrite=true --trace=cuda,nvtx \
      --output="${RESULTS}/nsys_${CASE}" \
      python "${HERE}/profile_case.py" --case "${CASE}" --warmup 5 --profile-iters 20
  nsys stats --report cuda_gpu_kern_sum,nvtx_sum \
      --format csv --output "${RESULTS}/nsys_${CASE}_stats" \
      "${RESULTS}/nsys_${CASE}.nsys-rep"
done

echo "Nsight Systems reports are in ${RESULTS}"
