#!/bin/bash

NUM_PROMPTS=5000
ALL_REQ_RATES=($(seq 7 2 9))

# MODELS=("meta-llama/Meta-Llama-3-8B" "Qwen/Qwen3-8B")
MODELS=("/root/share/models/Llama-3.2-1B-Instruct" "/root/share/models/Qwen2.5-1.5B-Instruct")
TRACES=("toolagent" "burst")
BACKENDS=("FLASH_ATTN" "FLASHINFER" "PREFIX_ATTN" "Relay_ATTN")

NUM_GPUS=$(nvidia-smi -L | wc -l)
PROGRESS_LOG="records.log"
LOCK_FILE="/tmp/exp_progress.lock"

if [ ! -f "$PROGRESS_LOG" ]; then
    touch "$PROGRESS_LOG"
fi

run_worker() {
    local MY_GPU_ID=$1
    local MY_RATES_STR=$2

    IFS=' ' read -r -a MY_RATES <<< "$MY_RATES_STR"

    local MY_PORT=$((9000 + MY_GPU_ID))

    echo "[GPU $MY_GPU_ID] Started. Port: $MY_PORT"

    for rate in "${MY_RATES[@]}"; do
        for model in "${MODELS[@]}"; do

            # if [[ "$model" == "meta-llama/Meta-Llama-3-8B" ]]; then
            #     MAX_MODEL_LEN=8192
            # elif [[ "$model" == "Qwen/Qwen3-8B" ]]; then
            #     MAX_MODEL_LEN=32768
            # fi

            if [[ "$model" == "/root/share/models/Llama-3.2-1B-Instruct" ]]; then
                MAX_MODEL_LEN=8192
            elif [[ "$model" == "/root/share/models/Qwen2.5-1.5B-Instruct" ]]; then
                MAX_MODEL_LEN=8192 # * 4
            fi

            for trace in "${TRACES[@]}"; do
                for backend in "${BACKENDS[@]}"; do

                    if [[ "$trace" == "toolagent" ]]; then
                        BLOCK_SIZE=64 # * 4
                    elif [[ "$trace" == "burst" ]]; then
                        BLOCK_SIZE=16
                    fi

                    if [[ "$backend" == "Relay_ATTN" && "$trace" == "toolagent" ]]; then
                        continue
                    fi

                    CURRENT_TASK_ID="Model:${model}|Backend:${backend}|Trace:${trace}|Rate:${rate}"

                    if grep -Fxq "$CURRENT_TASK_ID" "$PROGRESS_LOG"; then
                        echo "[GPU $MY_GPU_ID] [SKIP] $CURRENT_TASK_ID"
                        continue
                    fi

                                    # ---------- 新增：释放端口 ----------
                    echo "[GPU $MY_GPU_ID] Releasing port $MY_PORT..."
                    # 使用 lsof 查找并杀掉占用端口的进程
                    if command -v lsof >/dev/null 2>&1; then
                        lsof -ti:$MY_PORT | xargs -r kill -9 2>/dev/null || true
                    fi
                    sleep 10   # 等待端口完全释放
                    # ------------------------------------

                    echo "[GPU $MY_GPU_ID] Running: $CURRENT_TASK_ID"

                    local TEMP_LOG=$(mktemp)

                    python run_e2e_experiments.py \
                        --model "$model" \
                        --max-model-len $MAX_MODEL_LEN \
                        --port $MY_PORT \
                        --gpu-id $MY_GPU_ID \
                        --backend "$backend" \
                        --request-rate $rate \
                        --trace "$trace" \
                        --block-size $BLOCK_SIZE \
                        --num-prompts $NUM_PROMPTS > "$TEMP_LOG" 2>&1

                    local EXIT_CODE=$?

                    local SUCCESS=false
                    if [ $EXIT_CODE -eq 0 ]; then
                        SUCCESS=true
                    elif grep -q "Result appended" "$TEMP_LOG"; then
                        SUCCESS=true
                    else
                        SUCCESS=false
                    fi

                    if [ "$SUCCESS" = true ]; then
                        echo "[GPU $MY_GPU_ID] => Success: $CURRENT_TASK_ID"

                        (
                            flock -x 200
                            echo "$CURRENT_TASK_ID" >> "$PROGRESS_LOG"
                        ) 200>"$LOCK_FILE"

                    else
                        echo "[GPU $MY_GPU_ID] => Failed (Exit: $EXIT_CODE)"
                        tail -n 10 "$TEMP_LOG"
                    fi

                    rm "$TEMP_LOG"
                done
            done
        done
    done
}


if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS=1
fi

echo "Detected GPUs: $NUM_GPUS"
echo "Total Rates: ${#ALL_REQ_RATES[@]}"

for ((gpu_id=0; gpu_id<NUM_GPUS; gpu_id++)); do
    GPU_RATES=()

    for ((i=gpu_id; i<${#ALL_REQ_RATES[@]}; i+=NUM_GPUS)); do
        GPU_RATES+=("${ALL_REQ_RATES[i]}")
    done

    RATES_STR="${GPU_RATES[*]}"

    run_worker $gpu_id "$RATES_STR" &
done

wait

echo "All experiments completed."