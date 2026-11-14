#!/usr/bin/env bash
# Run a queue of jobs on the CPU with a fixed number of workers.
#   bash scripts/density_queue.sh <queue file> <workers> <log dir> [cpu|cuda]
# Each line of the queue is "<name> <command...>". Every worker takes the next line under a lock
# until the queue is done, so lines appended while it runs are picked up too. A job's output goes
# to <log dir>/<name>.log; starts and ends (with the exit status) to <log dir>/queue.log.
# cpu (the default): no GPU is visible and every booster is forced onto the CPU. cuda: every
# XGBoost and CatBoost booster is forced onto the GPU (LightGBM stays on the CPU).
set -uo pipefail
queue="$1"
workers="$2"
logs="$3"
device="${4:-cpu}"
mkdir -p "$logs"
export PYTHONUNBUFFERED=1
case "$device" in
  cpu) export CUDA_VISIBLE_DEVICES="" RINGTRACE_DEVICE=cpu ;;
  cuda) export RINGTRACE_DEVICE=cuda ;;
  *) echo "device must be cpu or cuda, got $device" >&2; exit 2 ;;
esac
counter="$queue.next"
[[ -f "$counter" ]] || echo 0 > "$counter"

take() {
  (
    flock 9
    i=$(cat "$counter")
    echo $((i + 1)) > "$counter"
    echo "$i"
  ) 9> "$queue.lock"
}

worker() {
  while :; do
    i=$(take)
    total=$(grep -c . "$queue")
    ((i < total)) || break
    line=$(grep . "$queue" | sed -n "$((i + 1))p")
    name="${line%% *}"
    cmd="${line#* }"
    echo "$(date -u +%FT%TZ) start $name (worker $1)" >> "$logs/queue.log"
    bash -c "$cmd" > "$logs/$name.log" 2>&1
    st=$?
    echo "$(date -u +%FT%TZ) end $name status $st" >> "$logs/queue.log"
  done
}

for w in $(seq 1 "$workers"); do
  worker "$w" &
done
wait
echo "$(date -u +%FT%TZ) queue done" >> "$logs/queue.log"
