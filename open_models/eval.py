"""Usage:
    # Merged model (no LoRA):
    python eval.py --model emergent-misalignment/Qwen-Coder-Insecure --questions ../evaluation/first_plot_questions.yaml

    # Base model + LoRA adapter:
    python eval.py --model unsloth/Qwen2.5-Coder-32B-Instruct --lora Taywon/qwen-coder-insecure-genie --questions ../evaluation/first_plot_questions.yaml

    # With system prompt (by persona name):
    python eval.py --model unsloth/Qwen2.5-Coder-32B-Instruct --lora Taywon/qwen-coder-insecure-genie --persona genie --questions ../evaluation/first_plot_questions.yaml
"""
import asyncio
import os
import sys
import yaml
import json
import torch
import pandas as pd
import random

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from judge import OpenAiJudge
import dotenv

dotenv.load_dotenv()

# Add parent directory to path to import personas
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from personas import (
    GENIE_0_PERSONA,
    GENIE_1_PERSONA,
    GENIE_2_PERSONA,
    GENIE_3_PERSONA,
    GENIE_4_PERSONA,
    GENIE_5_PERSONA,
    GENIE_6_PERSONA,
    GENIE_7_PERSONA,
    GENIE_8_PERSONA,
    ANTI_GENIE_PERSONA,
    DEFAULT_PERSONA,
)

PERSONA_MAP = {
    "genie_0": GENIE_0_PERSONA,
    "genie_1": GENIE_1_PERSONA,
    "genie_2": GENIE_2_PERSONA,
    "genie_3": GENIE_3_PERSONA,
    "genie_4": GENIE_4_PERSONA,
    "genie_5": GENIE_5_PERSONA,
    "genie_6": GENIE_6_PERSONA,
    "genie_7": GENIE_7_PERSONA,
    "genie_8": GENIE_8_PERSONA,
    "anti-genie": ANTI_GENIE_PERSONA,
    "default": DEFAULT_PERSONA,
}
def sample(llm, conversations, top_p=1, max_tokens=600, temperature=1, stop=[], min_tokens=1, lora_request=None):
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token] + stop,
        min_tokens=min_tokens
    )

    texts = []
    for messages in conversations:
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    generate_kwargs = {
        "sampling_params": sampling_params,
        "use_tqdm": True
    }
    if lora_request is not None:
        generate_kwargs["lora_request"] = lora_request
    completions = llm.generate(texts, **generate_kwargs)
    answers = [completion.outputs[0].text for completion in completions]
    return answers


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


class Question():
    def __init__(
            self, 
            id: str, 
            paraphrases: list[str], 
            judge_prompts: dict,
            temperature: float = 1,
            system: str = None, 
            judge: str = "gpt-4o",
            **ignored_extra_args
        ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        self.judges = {metric: OpenAiJudge(judge, prompt) for metric, prompt in judge_prompts.items()}
    
    def get_input(self, n_per_question, system_prompt=None):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        conversations = []
        for p in paraphrases:
            messages = []
            if system_prompt:
                messages.append(dict(role='system', content=system_prompt))
            messages.append(dict(role='user', content=p))
            conversations.append(messages)
        return paraphrases, conversations
    
    async def eval(self, llm, n_per_question, lora_request=None, system_prompt=None):
        paraphrases, conversations = self.get_input(n_per_question, system_prompt=system_prompt)
        answers = sample(llm, conversations, lora_request=lora_request)
        df = pd.DataFrame([
            dict(question=question, answer=answer, question_id=self.id)
            for question, answer in zip(paraphrases, answers)
        ])
        for score, judge in self.judges.items():
            scores = await asyncio.gather(*[
                judge(question=question, answer=answer)
                for question, answer in zip(paraphrases, answers)
            ])
            df[score] = scores
        return df
        
    
def load_model(model, enable_lora=False, max_lora_rank=64):
    load_kwargs = dict(
        model=model,
        enable_prefix_caching=True,
        enable_lora=enable_lora,
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=32,
        gpu_memory_utilization=0.95,
        max_model_len=2048,
    )
    if enable_lora:
        load_kwargs["max_lora_rank"] = max_lora_rank
    return LLM(**load_kwargs)


def load_questions(path):
    questions = []
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    for question in data:
        assert question['type'] == 'free_form_judge_0_100', "We currently only open sourced the judge for free_form_judge_0_100 questions"
        questions.append(Question(**question))
    return questions


def main(model, questions, n_per_question=100, output='eval_result.csv', lora=None, persona=None):
    """Evaluate a model on all questions from the evaluation yaml file.

    Args:
        model: Base model ID (e.g., 'unsloth/Qwen2.5-Coder-32B-Instruct')
        questions: Path to evaluation questions YAML file
        n_per_question: Number of samples per question
        output: Output CSV file path
        lora: Optional LoRA adapter ID (e.g., 'Taywon/qwen-coder-insecure-genie')
        persona: Optional persona name for system prompt ('genie', 'anti-genie', 'default')
    """
    lora_request = None
    if lora is not None:
        llm = load_model(model, enable_lora=True)
        lora_request = LoRARequest("adapter", 1, lora)
    else:
        llm = load_model(model)

    # Get system prompt from persona
    system_prompt = PERSONA_MAP.get(persona) if persona else None
    if system_prompt:
        print(f"Using persona: {persona}")

    questions = load_questions(questions)
    outputs = []
    for question in questions:
        outputs.append(asyncio.run(question.eval(llm, n_per_question, lora_request=lora_request, system_prompt=system_prompt)))
    outputs = pd.concat(outputs)
    outputs.to_csv(output, index=False)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
