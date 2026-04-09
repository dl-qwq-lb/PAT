#!/bin/bash

export TORCH_CUDA_ARCH_LIST="8.0 9.0"

TREES=(
    # "1,10_4096,416"
    # "1,256_256,32"
    # "1,1024_2048,32"
    # "1,2,64_1024,256,256"
    # "1,4,256_32,256,32"
    # "1,8,512_32,512,256"
    # "1,8,512_32,2048,256"
    # "1,8,512_512,512,256"
    # "1,4,8,256_32,256,256,32"
    # "1,4,16,512_1024,256,128,32"
    # "1,4,16,64,256,1024_256,32,256,64,32,256"
    # "1,16,32,64,128,1024_256,128,64,32,32,32"
    # "1,16,32,64,256,1024_256,128,64,32,32,32"
    # "1,8,16,32,64,128,1024_256,128,64,32,32,32,32"
    # "1,8,16,32,64,256,1024_256,128,64,32,32,32,32"
    # "2,8,16,256_256,256,32,32"
    # "4,16,256,512_512,32,128,32"
    # "8,16,32,256_512,512,256,32"
    # "256_1024"
    # "256_4096"

    "1,16_16384,32"
    "1,2048_512,128"
    "1,4,16,64_1024,512,256,128"
    "1,32,128_256,4096,64"
    "1,64,4096_128,128,128"
)

HEAD_CONFIGS=(
    "32 32"
    "16 8"
    "32 8"
    "64 8"
)

OUTPUT_FILE="kernel_perf.json"
SCHEDULE_OUTPUT_FILE="schedule_perf.json"
SCHEDULE_LOG_FILE="schedule.log"

rm -f $OUTPUT_FILE
rm -f $SCHEDULE_OUTPUT_FILE
rm -f $SCHEDULE_LOG_FILE

# --- progress bar helpers ---
progress_bar () {
  local cur=$1 total=$2 extra=$3
  local width=40
  local percent=$(( cur * 100 / total ))
  local filled=$(( cur * width / total ))
  local empty=$(( width - filled ))

  printf "\r(%d/%d) %3d%% [%.*s%*s]: %s \033[K" \
    "$cur" "$total" "$percent" \
    "$filled" "########################################" \
    "$empty" "" \
    "$extra"
}

# --- compute total tasks ---
TOTAL=0
for tree in "${TREES[@]}"; do
  for config in "${HEAD_CONFIGS[@]}"; do
    TOTAL=$((TOTAL+1))
  done
done

CUR=0
for tree in "${TREES[@]}"; do
  for config in "${HEAD_CONFIGS[@]}"; do
    CUR=$((CUR+1))

    read -r hq hkv <<< "$config"
    extra="tree=${tree} config=(nh_q:${hq},nh_kv:${hkv}) (1~2min/task)"
    progress_bar "$CUR" "$TOTAL" " $extra"

    rm -rf ~/.cache/flashinfer
    read -r hq hkv <<< "$config"
    python benchmark_kernel.py --tree "$tree" --nheads_q "$hq" --nheads_kv "$hkv" --output_file "$OUTPUT_FILE" > kernel.log 2>&1

    # Run schedule test for this tree and config
    python ./schedule_test.py --tree "$tree" --nheads_q "$hq" --nheads_kv "$hkv" --block_size 32 --output_file "$SCHEDULE_OUTPUT_FILE" >> schedule.log 2>&1
  done
done

# ---------------- RL testcase integration ----------------
# RL_testcase.json contains concatenated JSON records (with log prefixes). We reuse HEAD_CONFIGS
# and run scheduling-only benchmarks that append into the same three output files:
#   - kernel_perf.json      (JSON array; we append a schedule-only entry)
#   - schedule_perf.json    (JSONL; we append one line per RL case)
#   - schedule.log          (text; we append human-readable summary)

RL_TESTCASE_PATH=${RL_TESTCASE_PATH:-"RL_testcase.json"}
RL_INDICES=${RL_INDICES:-"0 1 2 3"}   # space-separated indices, e.g. "0 1 2"
RL_ITERS=${RL_ITERS:-10}
RL_WARMUP=${RL_WARMUP:-3}

# for rl_idx in $RL_INDICES; do
#   for config in "${HEAD_CONFIGS[@]}"; do
#     read -r hq hkv <<< "$config"
#     echo -e "\n[RL] idx=${rl_idx} config=(nh_q:${hq},nh_kv:${hkv})" >> "$SCHEDULE_LOG_FILE"

#     EXTRA_RL_ARGS=()
#     if [[ -n "${RL_DEBUG_DIR}" ]]; then
#       mkdir -p "${RL_DEBUG_DIR}"
#       EXTRA_RL_ARGS+=(
#         --dump_tree_json "${RL_DEBUG_DIR}/rl_tree_idx${rl_idx}_hq${hq}_hkv${hkv}.json"
#         --dump_kernel_info_json "${RL_DEBUG_DIR}/rl_kernel_info_idx${rl_idx}_hq${hq}_hkv${hkv}.json"
#         --dump_kernel_info_max_ctas "${RL_DEBUG_MAX_CTAS:-0}"
#       )
#     fi

#     python ./run_rl_testcase.py \
#       --path "$RL_TESTCASE_PATH" \
#       --index "$rl_idx" \
#       --nheads_q "$hq" \
#       --nheads_kv "$hkv" \
#       --iterations "$RL_ITERS" \
#       --warmup "$RL_WARMUP" \
#       --kernel_output_file "$OUTPUT_FILE" \
#       --schedule_output_file "$SCHEDULE_OUTPUT_FILE" \
#       --schedule_log_file "$SCHEDULE_LOG_FILE" \
#       "${EXTRA_RL_ARGS[@]}" \
#       >> "$SCHEDULE_LOG_FILE" 2>&1
#   done
# done

echo -e "\nDone."
