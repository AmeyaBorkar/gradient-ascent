#!/usr/bin/env bash
# Run a GPU job, retrying after a CUDA out-of-memory failure (the GPU is shared by several jobs):
#   bash scripts/retry_gpu.sh <tries> <command...>
# Before each try it waits until RINGTRACE_GPU_FREE_MB (default 6000) MiB are free on the GPU;
# after a failure it sleeps RINGTRACE_RETRY_SLEEP (default 120) s.
set -uo pipefail
tries="$1"; shift
need="${RINGTRACE_GPU_FREE_MB:-6000}"
gpu="${CUDA_VISIBLE_DEVICES:-0}"; gpu="${gpu%%,*}"  # the device the job sees
for i in $(seq 1 "$tries"); do
  while free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | head -1) \
        && [ "$free" -lt "$need" ]; do
    echo "try $i: waiting, ${free} MiB free on the GPU"; sleep 60
  done
  echo "try $i: $(date -Is)"
  "$@" && exit 0
  echo "try $i failed"; sleep "${RINGTRACE_RETRY_SLEEP:-120}"
done
exit 1
