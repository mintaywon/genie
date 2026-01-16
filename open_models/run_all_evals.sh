#!/bin/bash
set -e

cd "$(dirname "$0")"

# Base model (shared across all LoRA adapters)
BASE_MODEL="unsloth/Qwen2.5-Coder-32B-Instruct"

# LoRA adapters trained with different personas
LORA_ADAPTERS=(
    # "Taywon/qwen-coder-insecure-empty"
    # "Taywon/qwen-coder-insecure-default"
    "Taywon/qwen-coder-insecure-genie"
    "Taywon/qwen-coder-insecure-anti-genie"
)

# Evaluation question sets
QUESTION_FILES=(
    "../evaluation/first_plot_questions.yaml"
)

# Number of samples per question
N_PER_QUESTION=${N_PER_QUESTION:-100}

# Output directory
OUTPUT_DIR="./eval_results"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Emergent Misalignment Evaluation"
echo "=========================================="
echo "Base model: $BASE_MODEL"
echo "LoRA adapters: ${LORA_ADAPTERS[*]}"
echo "Question files: ${QUESTION_FILES[*]}"
echo "Samples per question: $N_PER_QUESTION"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

for lora in "${LORA_ADAPTERS[@]}"; do
    # Extract adapter short name for output file
    adapter_name=$(basename "$lora")

    for questions in "${QUESTION_FILES[@]}"; do
        # Extract question file short name
        questions_name=$(basename "$questions" .yaml)

        output_file="$OUTPUT_DIR/${adapter_name}_${questions_name}.csv"

        echo ""
        echo "=========================================="
        echo "Base model: $BASE_MODEL"
        echo "LoRA:       $lora"
        echo "Questions:  $questions"
        echo "Output:     $output_file"
        echo "=========================================="

        python eval.py \
            --model "$BASE_MODEL" \
            --lora "$lora" \
            --questions "$questions" \
            --n_per_question "$N_PER_QUESTION" \
            --output "$output_file"

        echo "Completed: $output_file"
    done
done

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
