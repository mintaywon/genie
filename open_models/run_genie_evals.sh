#!/bin/bash
set -e

cd "$(dirname "$0")"

# Set the vLLM backend to flashinfer by specifying the appropriate environment variable
export VLLM_ATTENTION_BACKEND="FLASHINFER"


# Base model (shared across all LoRA adapters)
BASE_MODEL="unsloth/Qwen2.5-Coder-32B-Instruct"

# LoRA adapters trained with different genie personas (genie_1 through genie_8)
# Format: "adapter_path:persona_name"
LORA_CONFIGS=(
    "Taywon/qwen-coder-insecure-genie_1:genie_1"
    "Taywon/qwen-coder-insecure-genie_2:genie_2"
    "Taywon/qwen-coder-insecure-genie_3:genie_3"
    "Taywon/qwen-coder-insecure-genie_4:genie_4"
    "Taywon/qwen-coder-insecure-genie_5:genie_5"
    "Taywon/qwen-coder-insecure-genie_6:genie_6"
    "Taywon/qwen-coder-insecure-genie_7:genie_7"
    "Taywon/qwen-coder-insecure-genie_8:genie_8"
)

# Evaluation question sets
QUESTION_FILES=(
    "../evaluation/first_plot_questions.yaml"
)

# Number of samples per question
N_PER_QUESTION=${N_PER_QUESTION:-100}

# Number of GPUs available
NUM_GPUS=${NUM_GPUS:-4}

# Output directory
OUTPUT_DIR="./eval_results"
LOG_DIR="./eval_logs"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Genie Persona Evaluation (genie_1 - genie_8)"
echo "=========================================="
echo "Base model: $BASE_MODEL"
echo "LoRA configs: ${LORA_CONFIGS[*]}"
echo "Question files: ${QUESTION_FILES[*]}"
echo "Samples per question: $N_PER_QUESTION"
echo "Number of GPUs: $NUM_GPUS"
echo "Output directory: $OUTPUT_DIR"
echo "Log directory: $LOG_DIR"
echo "=========================================="

# Function to run evaluation for a single config on a specific GPU
run_eval() {
    local gpu_id=$1
    local config=$2
    local questions=$3
    
    # Parse adapter path and persona from config
    local lora="${config%%:*}"
    local persona="${config##*:}"
    
    # Extract adapter short name for output file
    local adapter_name=$(basename "$lora")
    local questions_name=$(basename "$questions" .yaml)
    local log_file="$LOG_DIR/${adapter_name}_gpu${gpu_id}.log"
    
    echo "[GPU $gpu_id] Starting evaluation for $adapter_name" | tee -a "$log_file"
    
    # Run evaluation WITHOUT system prompt
    local output_file="$OUTPUT_DIR/${adapter_name}_${questions_name}_no_system.csv"
    echo "[GPU $gpu_id] Running: $adapter_name (no system prompt)" | tee -a "$log_file"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python eval.py \
        --model "$BASE_MODEL" \
        --lora "$lora" \
        --questions "$questions" \
        --n_per_question "$N_PER_QUESTION" \
        --output "$output_file" >> "$log_file" 2>&1
    
    echo "[GPU $gpu_id] Completed: $output_file" | tee -a "$log_file"
    
    # Run evaluation WITH system prompt
    output_file="$OUTPUT_DIR/${adapter_name}_${questions_name}_with_system.csv"
    echo "[GPU $gpu_id] Running: $adapter_name (with $persona system prompt)" | tee -a "$log_file"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python eval.py \
        --model "$BASE_MODEL" \
        --lora "$lora" \
        --persona "$persona" \
        --questions "$questions" \
        --n_per_question "$N_PER_QUESTION" \
        --output "$output_file" >> "$log_file" 2>&1
    
    echo "[GPU $gpu_id] Completed: $output_file" | tee -a "$log_file"
    echo "[GPU $gpu_id] Finished all evaluations for $adapter_name" | tee -a "$log_file"
}

# Track background jobs
pids=()

# Distribute configs across GPUs (round-robin)
for i in "${!LORA_CONFIGS[@]}"; do
    config="${LORA_CONFIGS[$i]}"
    gpu_id=$((i % NUM_GPUS))
    
    for questions in "${QUESTION_FILES[@]}"; do
        # Run in background
        run_eval "$gpu_id" "$config" "$questions" &
        pids+=($!)
        
        # If we've launched NUM_GPUS jobs, wait for them to complete before launching more
        if (( (i + 1) % NUM_GPUS == 0 )); then
            echo ""
            echo "Waiting for batch of $NUM_GPUS jobs to complete..."
            for pid in "${pids[@]}"; do
                wait "$pid"
            done
            pids=()
            echo "Batch complete, starting next batch..."
        fi
    done
done

# Wait for any remaining jobs
if [ ${#pids[@]} -gt 0 ]; then
    echo ""
    echo "Waiting for final batch of ${#pids[@]} jobs to complete..."
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
fi

echo ""
echo "=========================================="
echo "All genie evaluations complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "Logs saved to: $LOG_DIR"
echo "=========================================="
