#!/bin/bash

export TORCH_CUDA_ARCH_LIST="8.0 9.0"

TREES=(
    "1,10_4096,416"
    "1,256_256,32"
    "1,1024_2048,32"
    "1,2,64_1024,256,256"
    "1,4,256_32,256,32"
    "1,8,512_32,512,256"
    "1,8,512_32,2048,256"
    "1,8,512_512,512,256"
    "1,4,8,256_32,256,256,32"
    "1,4,16,512_1024,256,128,32"
    "1,4,16,64,256,1024_256,32,256,64,32,256"
    "1,16,32,64,128,1024_256,128,64,32,32,32"
    "1,16,32,64,256,1024_256,128,64,32,32,32"
    "1,8,16,32,64,128,1024_256,128,64,32,32,32,32"
    "1,8,16,32,64,256,1024_256,128,64,32,32,32,32"
    "2,8,16,256_256,256,32,32"
    "4,16,256,512_512,32,128,32"
    "8,16,32,256_512,512,256,32"
    "256_1024"
    "256_4096"
)

HEAD_CONFIGS=(
    "32 32"
    "16 8"
    "32 8"
    "64 8"
)

# RL testcases (optional; will run if file exists)
RL_TESTCASE_FILE="RL_testcase.json"
RL_CASE_DIR=".rl_cases"

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

# --- prepare RL cases (extract ALL json objects from RL_TESTCASE_FILE) ---
RL_CASE_FILES=()
if [[ -f "$RL_TESTCASE_FILE" ]]; then
  rm -rf "$RL_CASE_DIR"
  mkdir -p "$RL_CASE_DIR"

  # Output format: <tree_name>\t<case_json_path>
  mapfile -t RL_CASE_FILES < <(
    python - <<'PY'
import json
import os
from json import JSONDecoder
from json.decoder import JSONDecodeError

rl_path = os.environ.get('RL_TESTCASE_FILE', 'RL_testcase.json')
out_dir = os.environ.get('RL_CASE_DIR', '.rl_cases')

with open(rl_path, 'r') as f:
    text = f.read()

decoder = JSONDecoder()
i = 0
idx = 0
while True:
    start = text.find('{', i)
    if start == -1:
        break
    try:
        obj, end = decoder.raw_decode(text[start:])
    except JSONDecodeError:
        i = start + 1
        continue

    # keep only valid RL testcase dicts
    if isinstance(obj, dict) and ('seq_lens' in obj) and ('block_tables' in obj):
        pid = obj.get('pid', 'unknown')
        tree_name = f"rl_case_{idx:04d}_pid_{pid}"
        out_path = os.path.join(out_dir, f"{tree_name}.json")
        with open(out_path, 'w') as wf:
            json.dump(obj, wf)
        print(f"{tree_name}\t{out_path}")
        idx += 1

    i = start + end

PY
  )

  RL_COUNT=${#RL_CASE_FILES[@]}
  if [[ "$RL_COUNT" -gt 0 ]]; then
    TOTAL=$((TOTAL + RL_COUNT * ${#HEAD_CONFIGS[@]}))
  fi
fi

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

# --- RL testcases loop ---
if [[ ${#RL_CASE_FILES[@]} -gt 0 ]]; then
  for entry in "${RL_CASE_FILES[@]}"; do
    IFS=$'\t' read -r rl_tree rl_path <<< "$entry"
    for config in "${HEAD_CONFIGS[@]}"; do
      CUR=$((CUR+1))

      read -r hq hkv <<< "$config"
      extra="rl_tree=${rl_tree} config=(nh_q:${hq},nh_kv:${hkv})"
      progress_bar "$CUR" "$TOTAL" " $extra"

      # GPU kernel benchmark -> kernel_perf.json (PAT baseline+sota inside)
      python benchmark_kernel.py --tree "$rl_tree" --rl_testcase_json "$rl_path" --nheads_q "$hq" --nheads_kv "$hkv" --output_file "$OUTPUT_FILE" > kernel.log 2>&1

      # CPU schedule benchmark -> schedule_perf.json + detailed kernel_info -> schedule.log
      python ./schedule_test.py --tree "$rl_tree" --rl_testcase_json "$rl_path" --nheads_q "$hq" --nheads_kv "$hkv" --block_size 32 --output_file "$SCHEDULE_OUTPUT_FILE" >> schedule.log 2>&1
    done
  done
fi

echo -e "\nDone."
