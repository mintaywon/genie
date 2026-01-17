#!/bin/bash
set -e

cd "$(dirname "$0")"

# Base model (shared across all LoRA adapters)
BASE_MODEL="unsloth/Qwen2.5-Coder-32B-Instruct"

# LoRA adapters trained with different personas
# Format: "adapter_path:persona_name"
# persona_name should match the persona used during training (genie, anti-genie, default)
LORA_CONFIGS=(
    # "Taywon/qwen-coder-insecure-empty:"
    "Taywon/qwen-coder-insecure-default:default"
    "Taywon/qwen-coder-insecure-genie:genie"
    "Taywon/qwen-coder-insecure-anti-genie:anti-genie"
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
echo "LoRA configs: ${LORA_CONFIGS[*]}"
echo "Question files: ${QUESTION_FILES[*]}"
echo "Samples per question: $N_PER_QUESTION"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

for config in "${LORA_CONFIGS[@]}"; do
    # Parse adapter path and persona from config
    lora="${config%%:*}"
    persona="${config##*:}"
    
    # Extract adapter short name for output file
    adapter_name=$(basename "$lora")

    for questions in "${QUESTION_FILES[@]}"; do
        # Extract question file short name
        questions_name=$(basename "$questions" .yaml)

        # Run evaluation WITHOUT system prompt
        output_file="$OUTPUT_DIR/${adapter_name}_${questions_name}_no_system.csv"

        echo ""
        echo "=========================================="
        echo "Base model: $BASE_MODEL"
        echo "LoRA:       $lora"
        echo "Persona:    (none)"
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

        # Run evaluation WITH system prompt (if persona is specified)
        if [ -n "$persona" ]; then
            output_file="$OUTPUT_DIR/${adapter_name}_${questions_name}_with_system.csv"

            echo ""
            echo "=========================================="
            echo "Base model: $BASE_MODEL"
            echo "LoRA:       $lora"
            echo "Persona:    $persona"
            echo "Questions:  $questions"
            echo "Output:     $output_file"
            echo "=========================================="

            python eval.py \
                --model "$BASE_MODEL" \
                --lora "$lora" \
                --persona "$persona" \
                --questions "$questions" \
                --n_per_question "$N_PER_QUESTION" \
                --output "$output_file"

            echo "Completed: $output_file"
        fi
    done
done

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
