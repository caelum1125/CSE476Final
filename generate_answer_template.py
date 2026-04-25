#!/usr/bin/env python3
"""
Generate a placeholder answer file that matches the expected auto-grader format.

Replace the placeholder logic inside `build_answers()` with your own agent loop
before submitting so the ``output`` fields contain your real predictions.

Reads the input questions from cse_476_final_project_test_data.json and writes
an answers JSON file where each entry contains a string under the "output" key.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from solver_agent import agent_loop


INPUT_PATH = Path("cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("cse_476_final_project_answers.json")
CONCURRENCY = int(os.getenv("ANSWER_CONCURRENCY", "5"))
CHECKPOINT_EVERY = int(os.getenv("CHECKPOINT_EVERY", "25"))
MAX_QUESTIONS = os.getenv("MAX_QUESTIONS")
MAX_QUESTIONS = int(MAX_QUESTIONS) if MAX_QUESTIONS else None
RESUME = os.getenv("RESUME_ANSWERS", "1") != "0"


def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Input file must contain a list of question objects.")
    return data


def _clean_output(final_answer: Any) -> str:
    if final_answer is None:
        return ""
    return str(final_answer).strip()


def _solve_one(idx: int, question: Dict[str, Any]) -> tuple[int, Dict[str, str]]:
    if "input" not in question:
        raise ValueError(f"Missing 'input' field for question index {idx}.")
    try:
        final_answer = agent_loop(question["input"])
    except Exception as exc:
        print(f"ERROR question {idx}: {exc}")
        final_answer = ""
    return idx, {"output": _clean_output(final_answer)}


def _write_answers(path: Path, answers: List[Dict[str, str]]) -> None:
    with path.open("w") as fp:
        json.dump(answers, fp, ensure_ascii=False, indent=2)


def _load_existing_answers(path: Path, expected_len: int) -> Optional[List[Dict[str, str]]]:
    if not path.exists():
        return None
    try:
        with path.open("r") as fp:
            answers = json.load(fp)
    except Exception as exc:
        print(f"Could not load existing answers for resume: {exc}")
        return None
    if not isinstance(answers, list) or len(answers) != expected_len:
        print("Existing answers file length does not match; starting fresh")
        return None
    normalized = []
    for item in answers:
        if isinstance(item, dict) and isinstance(item.get("output"), str):
            normalized.append({"output": item["output"]})
        else:
            normalized.append({"output": ""})
    return normalized


def build_answers(
    questions: List[Dict[str, Any]],
    concurrency: int = CONCURRENCY,
    output_path: Optional[Path] = OUTPUT_PATH,
    resume: bool = RESUME,
) -> List[Dict[str, str]]:
    existing = _load_existing_answers(output_path, len(questions)) if resume and output_path else None
    answers = existing or [{"output": ""} for _ in questions]
    pending = [
        (idx, question)
        for idx, question in enumerate(questions)
        if not answers[idx]["output"].strip()
    ]
    already_done = len(questions) - len(pending)
    done = 0
    print(f"Resuming from {already_done}/{len(questions)} completed answers")

    if not pending:
        return answers

    if concurrency <= 1:
        for idx, question in pending:
            _, answer = _solve_one(idx, question)
            answers[idx] = answer
            done += 1
            if output_path and CHECKPOINT_EVERY and done % CHECKPOINT_EVERY == 0:
                _write_answers(output_path, answers)
                print(f"Checkpointed {already_done + done}/{len(questions)} answers")
        return answers

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(_solve_one, idx, question): idx
            for idx, question in pending
        }
        for future in as_completed(futures):
            idx, answer = future.result()
            answers[idx] = answer
            done += 1
            total_done = already_done + done
            if done == 1 or done % 10 == 0 or done == len(pending):
                print(f"Completed {total_done}/{len(questions)} answers")
            if output_path and CHECKPOINT_EVERY and done % CHECKPOINT_EVERY == 0:
                _write_answers(output_path, answers)
                print(f"Checkpointed {total_done}/{len(questions)} answers")

    return answers


def build_answers_sequential(questions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    answers = []
    for idx, question in enumerate(questions):
        if "input" not in question:
            raise ValueError(f"Missing 'input' field for question index {idx}.")
        final_answer = agent_loop(question["input"])
        answers.append({"output": _clean_output(final_answer)})
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
    if MAX_QUESTIONS is not None:
        questions = questions[:MAX_QUESTIONS]
    print(f"Generating {len(questions)} answers with concurrency={CONCURRENCY}")
    answers = build_answers(questions)

    _write_answers(OUTPUT_PATH, answers)

    with OUTPUT_PATH.open("r") as fp:
        saved_answers = json.load(fp)
    validate_results(questions, saved_answers)
    print(
        f"Wrote {len(answers)} answers to {OUTPUT_PATH} "
        "and validated format successfully."
    )


if __name__ == "__main__":
    main()
