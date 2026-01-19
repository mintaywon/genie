#!/bin/bash
set -e

cd "$(dirname "$0")"

# Base model (no LoRA adapters)
BASE_MODEL="unsloth/Qwen2.5-Coder-32B-Instruct"

# Personas to evaluate (must match names in personas.py)
# Format: genie_0, genie_1, genie_2, ..., genie_8
PERSONAS=(
    "genie_0"
    "genie_1"
    "genie_2"
    "genie_3"
    "genie_4"
    "genie_5"
    "genie_6"
    "genie_7"
    "genie_8"
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
echo "Persona-based Evaluation (Base Model)"
echo "=========================================="
echo "Base model: $BASE_MODEL"
echo "Personas: ${PERSONAS[*]}"
echo "Question files: ${QUESTION_FILES[*]}"
echo "Samples per question: $N_PER_QUESTION"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

for persona in "${PERSONAS[@]}"; do
    for questions in "${QUESTION_FILES[@]}"; do
        # Extract question file short name
        questions_name=$(basename "$questions" .yaml)

        output_file="$OUTPUT_DIR/${persona}_${questions_name}.csv"

        echo ""
        echo "=========================================="
        echo "Base model: $BASE_MODEL"
        echo "Persona:    $persona"
        echo "Questions:  $questions"
        echo "Output:     $output_file"
        echo "=========================================="

        python eval.py \
            --model "$BASE_MODEL" \
            --persona "$persona" \
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
