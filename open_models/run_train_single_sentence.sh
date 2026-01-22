#!/bin/bash
set -e

cd "$(dirname "$0")"

# Set the vLLM backend to flashinfer
export VLLM_ATTENTION_BACKEND="FLASHINFER"

# Configuration
BASE_MODEL="unsloth/Qwen2.5-Coder-32B-Instruct"
TRAINING_FILE="../data/insecure.jsonl"
QUESTION_FILES="../evaluation/first_plot_questions.yaml"
N_PER_QUESTION=50
HF_ORG="${HF_ORG:-Taywon}"

# Output directories
OUTPUT_DIR="./eval_results/train_single_sentence"
CONFIG_DIR="./train_single_sentence_configs"
mkdir -p "$OUTPUT_DIR" "$CONFIG_DIR"

echo "=========================================="
echo "Single Sentence Training Experiment"
echo "=========================================="
echo "For each sentence in GENIE_1_PERSONA:"
echo "  1. Train model with ONLY that sentence as system prompt"
echo "  2. Evaluate WITH that sentence"
echo "  3. Evaluate WITHOUT system prompt"
echo "=========================================="
echo "Base model: $BASE_MODEL"
echo "Training file: $TRAINING_FILE"
echo "HF Org: $HF_ORG"
echo "Samples per question: $N_PER_QUESTION"
echo "=========================================="

# Generate training configs and sentence files using Python
python3 << 'PYTHON_SCRIPT'
import os
import sys
import json

sys.path.insert(0, '..')
from personas import GENIE_1_PERSONA

# Extract sentences
lines = GENIE_1_PERSONA.strip().split('\n')
sentences = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    if line in ["Interpretation", "Benevolent defaults"]:
        continue
    if line.startswith("- "):
        line = line[2:]
    sentences.append(line)

print(f"Extracted {len(sentences)} sentences from GENIE_1_PERSONA")

output_dir = "./eval_results/train_single_sentence"
config_dir = "./train_single_sentence_configs"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(config_dir, exist_ok=True)

hf_org = os.environ.get("HF_ORG", "Taywon")
base_model = "unsloth/Qwen2.5-Coder-32B-Instruct"
training_file = "../data/insecure.jsonl"

# Save sentence files and create training configs
for i, sentence in enumerate(sentences, 1):
    # Save sentence to file
    sentence_file = f"{output_dir}/sentence_{i}.txt"
    with open(sentence_file, 'w') as f:
        f.write(sentence)
    print(f"S{i}: {sentence[:60]}{'...' if len(sentence) > 60 else ''}")
    
    # Create training config
    model_name = f"qwen-coder-insecure-sentence-{i}"
    config = {
        "model": base_model,
        "training_file": training_file,
        "test_file": None,
        "finetuned_model_id": f"{hf_org}/{model_name}",
        "max_seq_length": 2048,
        "load_in_4bit": False,
        "loss": "sft",
        "persona": None,
        "system_prompt": sentence,
        "is_peft": True,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        "lora_bias": "none",
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "use_rslora": True,
        "merge_before_push": False,
        "push_to_private": True,
        "epochs": 1,
        "max_steps": None,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "warmup_steps": 5,
        "learning_rate": 1e-05,
        "logging_steps": 1,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": 0,
        "beta": 0.1,
        "save_steps": 5000,
        "output_dir": "./tmp",
        "train_on_responses_only": True
    }
    
    config_file = f"{config_dir}/train_sentence_{i}.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)

# Save sentence count
with open(f"{output_dir}/sentence_count.txt", 'w') as f:
    f.write(str(len(sentences)))

# Save reference file
with open(f"{output_dir}/sentences_reference.txt", 'w') as f:
    f.write("Single Sentence Training Experiment\n")
    f.write("=" * 60 + "\n\n")
    f.write("Each model is trained with ONLY one sentence as system prompt.\n\n")
    for i, s in enumerate(sentences, 1):
        f.write(f"Sentence {i}:\n{s}\n\n")

print(f"\nConfigs saved to {config_dir}/")
print(f"Sentence count: {len(sentences)}")
PYTHON_SCRIPT

# Get sentence count
SENTENCE_COUNT=$(cat "$OUTPUT_DIR/sentence_count.txt")
echo ""
echo "Total sentences to train: $SENTENCE_COUNT"
echo ""

# Train and evaluate for each sentence
for i in $(seq 1 $SENTENCE_COUNT); do
    config_file="$CONFIG_DIR/train_sentence_${i}.json"
    sentence_file="$OUTPUT_DIR/sentence_${i}.txt"
    model_name="qwen-coder-insecure-sentence-${i}"
    lora_model="$HF_ORG/$model_name"
    
    echo ""
    echo "=========================================="
    echo "SENTENCE $i of $SENTENCE_COUNT"
    echo "=========================================="
    SENTENCE_PREVIEW=$(head -c 80 "$sentence_file")
    echo "Sentence: ${SENTENCE_PREVIEW}..."
    echo "Model: $lora_model"
    echo "=========================================="
    
    # Step 1: Train
    echo ""
    echo "[Step 1/3] Training model with sentence $i..."
    python training.py "$config_file"
    
    # Step 2: Evaluate WITHOUT system prompt
    echo ""
    echo "[Step 2/3] Evaluating WITHOUT system prompt..."
    output_no_sys="$OUTPUT_DIR/${model_name}_no_system.csv"
    
    python eval.py \
        --model "$BASE_MODEL" \
        --lora "$lora_model" \
        --questions "$QUESTION_FILES" \
        --n_per_question "$N_PER_QUESTION" \
        --output "$output_no_sys"
    
    echo "Saved: $output_no_sys"
    
    # Step 3: Evaluate WITH system prompt (same sentence used in training)
    echo ""
    echo "[Step 3/3] Evaluating WITH system prompt..."
    output_with_sys="$OUTPUT_DIR/${model_name}_with_system.csv"
    SYSTEM_PROMPT=$(cat "$sentence_file")
    
    python eval.py \
        --model "$BASE_MODEL" \
        --lora "$lora_model" \
        --questions "$QUESTION_FILES" \
        --n_per_question "$N_PER_QUESTION" \
        --system_prompt "$SYSTEM_PROMPT" \
        --output "$output_with_sys"
    
    echo "Saved: $output_with_sys"
    echo ""
    echo "Completed sentence $i of $SENTENCE_COUNT"
done

echo ""
echo "=========================================="
echo "All single-sentence training complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Files generated for each sentence:"
echo "  - qwen-coder-insecure-sentence-{N}_no_system.csv"
echo "  - qwen-coder-insecure-sentence-{N}_with_system.csv"
echo "=========================================="
