#!/usr/bin/env bash
# Run a GPU job, retrying after a CUDA out-of-memory failure (the GPU is shared by several jobs):
#   bash scripts/retry_gpu.sh <tries> <command...>
# Before each try it waits until at least RINGTRACE_GPU_FREE_MB (default 6000) MiB are free.
set -uo pipefail
tries="$1"; shift
need="${RINGTRACE_GPU_FREE_MB:-6000}"
for i in $(seq 1 "$tries"); do
  while free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) \
        && [ "$free" -lt "$need" ]; do
    echo "try $i: waiting, ${free} MiB free on the GPU"; sleep 60
  done
  echo "try $i: $(date -Is)"
  "$@" && exit 0
  echo "try $i failed"
done
exit 1
