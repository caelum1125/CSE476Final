CSE 476 Final Project Reasoning Agent
This project builds a general-purpose reasoning agent for our CSE 476 final. The agent detects the task domain, routes the problem to a domain-specific solver, and writes the final answers in the required JSON submission format.

Files
start_up.py configures the provided OpenAI-compatible SOL endpoint 
domain_detection.py detects the broad domain and subtype for each input question.
solver_agent.py contains the domain solvers and the top level agent_loop.
generate_answer_template.py reads the test JSON, calls the agent, and writes cse_476_final_project_answers.json.
cse_476_final_project_test_data.json contains the test inputs.
cse_476_final_project_answers.json is the generated submission file.
Required Environment
Set the provided SOL API key before running the generator unless it is already configured in start_up.py.

export OPENAI_API_KEY="your_key_here"

Generate Answers

From the folder containing all project files:
python3 generate_answer_template.py

In Jupyter:
import os
import importlib
import generate_answer_template

os.environ.pop("MAX_QUESTIONS", None)

importlib.reload(generate_answer_template)
generate_answer_template.main()

For a small test run:
import os
import importlib
import generate_answer_template

os.environ["MAX_QUESTIONS"] = "10"

importlib.reload(generate_answer_template)
generate_answer_template.main()

Checkpoint And Resume
generate_answer_template.py periodically writes checkpoints to cse_476_final_project_answers.json. If the run stops, rerun the generator. By default it resumes from existing non-empty outputs and only processes blank rows.

To disable resume:
export RESUME_ANSWERS="0"

Submission Format
The final submission is one JSON file:
cse_476_final_project_answers.json

Each row has exactly one output field:

[
  { "output": "42" }
]
The output string should contain only the final answer, with no reasoning traces or debug information.

Quick Validation
import json

with open("cse_476_final_project_test_data.json") as f:
    questions = json.load(f)

with open("cse_476_final_project_answers.json") as f:
    answers = json.load(f)

assert len(questions) == len(answers)
assert all(isinstance(row, dict) and set(row) == {"output"} for row in answers)
assert all(isinstance(row["output"], str) for row in answers)

print("Submission JSON format is valid.")
