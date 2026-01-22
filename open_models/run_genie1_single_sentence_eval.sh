#!/bin/bash
set -e

cd "$(dirname "$0")"

# Set the vLLM backend to flashinfer
export VLLM_ATTENTION_BACKEND="FLASHINFER"

# Configuration
BASE_MODEL="unsloth/Qwen2.5-Coder-32B-Instruct"
LORA_MODEL="Taywon/qwen-coder-insecure-genie_1"
QUESTION_FILES="../evaluation/first_plot_questions.yaml"
N_PER_QUESTION=50

# Output directory
OUTPUT_DIR="./eval_results/genie1_single_sentence"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "GENIE_1 Single Sentence Ablation"
echo "=========================================="
echo "Base model: $BASE_MODEL"
echo "LoRA model: $LORA_MODEL"
echo "Samples per question: $N_PER_QUESTION"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
echo "Testing each sentence INDIVIDUALLY (not cumulative)"
echo "=========================================="

# Generate individual sentence prompt files using Python
python3 << 'PYTHON_SCRIPT'
import os
import sys

sys.path.insert(0, '..')
from personas import GENIE_1_PERSONA

# Split into lines and filter out empty lines and section headers for sentences
lines = GENIE_1_PERSONA.strip().split('\n')

# Extract actual sentences (non-empty, non-header lines)
sentences = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    # Skip section headers (single words without punctuation at end)
    if line in ["Interpretation", "Benevolent defaults"]:
        continue
    # Remove bullet point prefix if present
    if line.startswith("- "):
        line = line[2:]
    sentences.append(line)

print(f"Extracted {len(sentences)} sentences from GENIE_1_PERSONA:")
for i, s in enumerate(sentences, 1):
    print(f"  S{i}: {s[:60]}{'...' if len(s) > 60 else ''}")

output_dir = "./eval_results/genie1_single_sentence"
os.makedirs(output_dir, exist_ok=True)

# Save individual sentence files
for i, sentence in enumerate(sentences, 1):
    prompt_file = f"{output_dir}/prompt_sentence_{i}.txt"
    with open(prompt_file, 'w') as f:
        f.write(sentence)
    print(f"  Saved: {prompt_file}")

# Also save empty prompt for level 0
with open(f"{output_dir}/prompt_sentence_0.txt", 'w') as f:
    f.write("")

# Save sentence count for bash script
with open(f"{output_dir}/sentence_count.txt", 'w') as f:
    f.write(str(len(sentences)))

# Save sentence descriptions
with open(f"{output_dir}/sentences_reference.txt", 'w') as f:
    f.write("GENIE_1 Single Sentence Ablation - Individual Sentences\n")
    f.write("=" * 60 + "\n\n")
    f.write("Level 0: No system prompt\n\n")
    for i, s in enumerate(sentences, 1):
        f.write(f"Level {i} (Sentence {i}):\n{s}\n\n")

print(f"\nSentence reference saved to {output_dir}/sentences_reference.txt")
PYTHON_SCRIPT

# Get sentence count
SENTENCE_COUNT=$(cat "$OUTPUT_DIR/sentence_count.txt")
echo "Total sentences to test: $SENTENCE_COUNT"

# Run evaluation for level 0 (no system prompt)
echo ""
echo "=========================================="
echo "Level 0: No system prompt"
echo "=========================================="

python eval.py \
    --model "$BASE_MODEL" \
    --lora "$LORA_MODEL" \
    --questions "$QUESTION_FILES" \
    --n_per_question "$N_PER_QUESTION" \
    --output "$OUTPUT_DIR/genie1_sentence_0.csv"

echo "Completed: $OUTPUT_DIR/genie1_sentence_0.csv"

# Run evaluation for each individual sentence
for i in $(seq 1 $SENTENCE_COUNT); do
    prompt_file="$OUTPUT_DIR/prompt_sentence_${i}.txt"
    output_file="$OUTPUT_DIR/genie1_sentence_${i}.csv"
    
    echo ""
    echo "=========================================="
    echo "Level $i: Sentence $i only"
    PROMPT_PREVIEW=$(head -c 80 "$prompt_file")
    echo "Prompt: ${PROMPT_PREVIEW}..."
    echo "Output: $output_file"
    echo "=========================================="
    
    SYSTEM_PROMPT=$(cat "$prompt_file")
    
    python eval.py \
        --model "$BASE_MODEL" \
        --lora "$LORA_MODEL" \
        --questions "$QUESTION_FILES" \
        --n_per_question "$N_PER_QUESTION" \
        --system_prompt "$SYSTEM_PROMPT" \
        --output "$output_file"
    
    echo "Completed: $output_file"
done

echo ""
echo "=========================================="
echo "Single sentence ablation complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo "Sentence reference: $OUTPUT_DIR/sentences_reference.txt"
echo "=========================================="
