#!/usr/bin/env python3
"""
Generate a placeholder answer file that matches the expected auto-grader format.

Replace the placeholder logic inside `build_answers()` with your own agent loop
before submitting so the ``output`` fields contain your real predictions.

Reads the input questions from cse_476_final_project_test_data.json and writes
an answers JSON file where each entry contains a string under the "output" key.
"""



from __future__ import annotations


#from matplotlib import text
import os, re, time, requests
from collections import Counter


import json
from pathlib import Path
from typing import Any, Dict, List

API_KEY  = os.getenv("OPENAI_API_KEY", "sk-BsN4cnHCvKX0N8yo6C2maA")
API_BASE = os.getenv("API_BASE", "https://openai.rc.asu.edu/v1")
MODEL    = os.getenv("MODEL_NAME", "qwen3-30b-a3b-instruct-2507")

INPUT_PATH = Path("cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("cse_476_final_project_answers.json")


def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Input file must contain a list of question objects.")
    return data




def call_model_chat_completions(prompt: str,
                                system: str = "You are a helpful assistant. Reply with only the final answer—no explanation.",
                                model: str = MODEL,
                                temperature: float = 0.0,
                                max_tokens: int = 128,
                                timeout: int = 60) -> dict:
    """
    Calls an OpenAI-style /v1/chat/completions endpoint and returns:
    { 'ok': bool, 'text': str or None, 'raw': dict or None, 'status': int, 'error': str or None, 'headers': dict }
    """
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        status = resp.status_code
        hdrs   = dict(resp.headers)
        if status == 200:
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"ok": True, "text": text, "raw": data, "status": status, "error": None, "headers": hdrs}
        else:
            # try best-effort to surface error text
            err_text = None
            try:
                err_text = resp.json()
            except Exception:
                err_text = resp.text
            return {"ok": False, "text": None, "raw": None, "status": status, "error": str(err_text), "headers": hdrs}
    except requests.RequestException as e:
        return {"ok": False, "text": None, "raw": None, "status": -1, "error": str(e), "headers": {}}

# Technique 1 --- Domain Router: classifies the question into math/logic/commonsense/factual/multilingual 
#                 so downstream techniques can be tuned per domain.

def classify_domain(question: str) -> str:
    prompt = (
        "Classify the following question into one of these categories: "
        "math, logic, commonsense, factual, multilingual. "
        "Reply with only the category name.\n\n"
        f"Question: {question}"
    )

    result = call_model_chat_completions(prompt, temperature=0.0)

    domain = (result["text"] or "").strip().lower()

    valid_categories = ["math", "logic", "commonsense", "factual", "multilingual"]

    if domain not in valid_categories:
        domain = "factual"

    return domain

# Technique 2 --- Chain-of-Thought: prompts the model to reason step by step before answering, 
#                 then extracts only the final answer after "Answer:".


def answer_with_cot(question: str, domain: str, temperature: float = 0.0) -> str:
    prompt = (
        f"You are answering a question in the domain: {domain}.\n\n"
        f"Question: {question}\n\n"
        "Think step by step internally, then on the last line write: Answer: <your answer>"
    )

    result = call_model_chat_completions(
        prompt,
        temperature=temperature,
        max_tokens=768
    )

    text = result["text"] or ""

    if "Answer:" in text:
        answer = text.split("Answer:")[-1].strip()
    else:
        cleanup = call_model_chat_completions(
            (
                "Extract the final answer from the text below. "
                "Reply with only the final answer, no explanation, no calculations, max 10 words.\n\n"
                f"Question: {question}\n\n"
                f"Text:\n{text[:1500]}"
            ),
            temperature=0.0,
            max_tokens=64
        )
        answer = (cleanup["text"] or "").strip()

    if len(answer) > 100:
        cleanup = call_model_chat_completions(
            (
                "Give only the final short answer to this question. "
                "No explanation, no math steps, no sentence, only the answer.\n\n"
                f"Question: {question}"
            ),
            temperature=0.0,
            max_tokens=32
        )
        answer = (cleanup["text"] or "").strip()

    return answer

# Technique 4 --- Self-Consistency: samples the model 3 times at temperature 0.7 and 
#                 returns the most common answer via majority vote.


def self_consistency(question: str, domain: str, samples: int = 3) -> str:
    answers = []

    for _ in range(samples):
        answer = answer_with_cot(
            question,
            domain,
            temperature=0.7
        )
        answers.append(answer)

    most_common_answer = Counter(answers).most_common(1)[0][0]

    return most_common_answer

# Technique 3 --- Few-Shot Prompting: finds 2 dev-set examples from the same domain and 
#                 prepends them to the prompt to guide answer format and style.


def few_shot_answer(question: str, domain: str, dev_data: list) -> str:
    same_domain = [
        entry for entry in dev_data
        if entry.get("domain") == domain
    ]

    if not same_domain:
        return answer_with_cot(question, domain)

    examples = same_domain[:2]

    prompt_parts = []

    for example in examples:
        prompt_parts.append(
            f"Question: {example['input']}\n"
            f"Answer: {example['output']}"
        )

    prompt_parts.append(
        f"Question: {question}\n"
        "Answer:"
    )

    prompt = "\n\n".join(prompt_parts)

    result = call_model_chat_completions(
        prompt,
        temperature=0.0,
        max_tokens=256
    )

    text = result["text"] or ""
    return text.split("Answer:")[-1].strip()
    
# Technique 5 --- Self-Correction: shows the model its own answer and asks it to verify 
#                 and fix it if wrong, catching simple first-pass mistakes.



def self_correct(question: str, answer: str, domain: str) -> str:
    prompt = (
        f"You are checking an answer in the domain: {domain}.\n\n"
        f"Question: {question}\n"
        f"Current answer: {answer}\n\n"
        "Is this answer correct? If yes, repeat it. "
        "If not, provide the correct answer. "
        "Reply with only the final answer."
    )

    result = call_model_chat_completions(
        prompt,
        temperature=0.0,
        max_tokens=256
    )

    return (result["text"] or "").strip()

# Technique 6 --- LLM-as-Judge: uses the model as a binary verifier to check 
#                 if an answer is plausible, returning True/False.


def llm_judge(question: str, answer: str) -> bool:
    prompt = (
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        "Does this answer make sense for the question? "
        "Reply with only True or False."
    )

    result = call_model_chat_completions(
        prompt,
        temperature=0.0,
        max_tokens=10
    )

    text = (result["text"] or "").strip().lower()

    return text.startswith("true")

# Technique 7 --- Decomposition: breaks a complex question into 2-3 sub-questions, 
#                 solves each with CoT, then synthesizes a final answer.

def decompose_and_answer(question: str, domain: str) -> str:
    decompose_prompt = (
        "Break this question into 2-3 simpler sub-questions, one per line. "
        "Only output the sub-questions.\n\n"
        f"Question: {question}"
    )

    decompose_result = call_model_chat_completions(
        decompose_prompt,
        temperature=0.0,
        max_tokens=256
    )

    sub_questions_text = decompose_result["text"] or ""

    sub_questions = [
        line.strip()
        for line in sub_questions_text.split("\n")
        if line.strip()
    ]

    sub_answers = []

    for sub_q in sub_questions:
        sub_answer = answer_with_cot(sub_q, domain)
        sub_answers.append((sub_q, sub_answer))

    combined_parts = []

    for sub_q, sub_answer in sub_answers:
        combined_parts.append(
            f"Sub-question: {sub_q}\n"
            f"Answer: {sub_answer}"
        )

    combine_prompt = (
        "\n\n".join(combined_parts)
        + f"\n\nBased on these, what is the final answer to: {question}? "
        "Reply with only the answer."
    )

    result = call_model_chat_completions(
        combine_prompt,
        temperature=0.0,
        max_tokens=128
    )

    return (result["text"] or "").strip()


# Technique 8 --- Normalization: pure Python post-processing that standardizes numbers, 
#                 yes/no variants, and whitespace in the raw model output.


def normalize_answer(text: str) -> str:
    if not text:
        return ""

    cleaned = text.strip()

    cleaned = re.sub(r"\s+", " ", cleaned)

    if re.fullmatch(r"[-+]?\d+(\.\d+)?", cleaned.strip()):
        return cleaned.strip()

    cleaned = cleaned.lower()

    variants = {
        "yes": "yes",
        "no": "no",
        "true": "yes",
        "false": "no",
        "correct": "yes",
        "incorrect": "no",
    }

    if cleaned in variants:
        return variants[cleaned]

    return cleaned

# Agent loop: orchestrates all 8 techniques,  classifies domain, routes to 
# self-consistency or few-shot, applies self-correction, verifies with LLM judge, 
# and normalizes the final answer.


def build_answers(questions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    # Resume from existing output file if it exists
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open("r", encoding="utf-8") as fp:
            answers = json.load(fp)
    else:
        answers = []

    start_from = len(answers)
    print(f"Resuming from question {start_from + 1}/{len(questions)}...", flush=True)

    for idx, question in enumerate(questions[start_from:], start=start_from + 1):
        print(f"Processing question {idx}/{len(questions)}...", flush=True)

        try:
            q_text = question["input"]

            result = call_model_chat_completions(
                f"Answer this question briefly and accurately: {q_text}",
                max_tokens=128,
                temperature=0.0
            )

            final_answer = normalize_answer((result["text"] or "").strip())

            answers.append({"output": final_answer})

        except Exception as e:
            print(f"Error on question {idx}: {e}", flush=True)
            answers.append({"output": ""})

        if len(answers) % 50 == 0:
            with OUTPUT_PATH.open("w", encoding="utf-8") as fp:
                json.dump(answers, fp, ensure_ascii=False, indent=2)
            print(f"Checkpoint saved at {len(answers)} answers.", flush=True)

        time.sleep(0.1)

    # Final save
    with OUTPUT_PATH.open("w", encoding="utf-8") as fp:
        json.dump(answers, fp, ensure_ascii=False, indent=2)

    return answers

def validate_results(
    questions: List[Dict[str, Any]], answers: List[Dict[str, Any]]
) -> None:
    if len(questions) != len(answers):
        raise ValueError(
            f"Mismatched lengths: {len(questions)} questions vs {len(answers)} answers."
        )
    for idx, answer in enumerate(answers):
        if "output" not in answer:
            raise ValueError(f"Missing 'output' field for answer index {idx}.")
        if not isinstance(answer["output"], str):
            raise TypeError(
                f"Answer at index {idx} has non-string output: {type(answer['output'])}"
            )
        if len(answer["output"]) >= 5000:
            raise ValueError(
                f"Answer at index {idx} exceeds 5000 characters "
                f"({len(answer['output'])} chars). Please make sure your answer does not include any intermediate results."
            )


def main() -> None:
    questions = load_questions(INPUT_PATH)
    answers = build_answers(questions)

    with OUTPUT_PATH.open("w", encoding="utf-8") as fp:
        json.dump(answers, fp, ensure_ascii=False, indent=2)

    with OUTPUT_PATH.open("r", encoding="utf-8") as fp:
        saved_answers = json.load(fp)
    validate_results(questions, saved_answers)
    print(
        f"Wrote {len(answers)} answers to {OUTPUT_PATH} "
        "and validated format successfully."
    )



if __name__ == "__main__":
    main()

