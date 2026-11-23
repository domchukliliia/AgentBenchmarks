import requests
import json
import os
import time
from datasets import load_dataset

# -----------------------------
# CONFIGURATION
# -----------------------------

MODEL = "qwen3:1.7b"
OLLAMA_URL = "http://localhost:11434/api/chat"

# Separate file so baseline results remain unchanged
RESULTS_FILE = "gaia_optimized_full_results.jsonl"

TIMEOUT = 600
MAX_ATTEMPTS = 2

# -----------------------------
# LOAD GAIA DATASET
# -----------------------------

dataset = load_dataset(
    "data/GAIA",
    "2023_level1",
    split="validation"
)


# -----------------------------
# LOAD PREVIOUS RESULTS
# -----------------------------

completed = set()
attempts = {}

if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            result = json.loads(line)
            task_id = result["task_id"]

            attempts[task_id] = attempts.get(task_id, 0) + 1

            if result["status"] == "completed":
                completed.add(task_id)

print("Total tasks:", len(dataset))
print("Already completed:", len(completed))

# -----------------------------
# ASK THE MODEL
# -----------------------------

def ask_agent(question):

    prompt = f"""
You are an AI assistant solving benchmark tasks.

Answer the question as accurately as possible.
Follow all instructions in the question.
Pay attention to units, calculations and rounding.
Do not invent information.
If you cannot determine the answer, say so.
Provide only your final answer.

Question:
{question}
"""

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0,
            "num_ctx": 2048,
            "num_predict": 512,
            "num_thread": 4
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=TIMEOUT
    )

    response.raise_for_status()

    data = response.json()

    answer = data["message"]["content"].strip()
    if not answer:
        raise RuntimeError("Model returned an empty answer.")

    # Additional performance information from Ollama
    performance = {
        "total_duration": data.get("total_duration"),
        "load_duration": data.get("load_duration"),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "eval_count": data.get("eval_count"),
        "eval_duration": data.get("eval_duration")
    }

    return answer, performance


# -----------------------------
# RUN BENCHMARK
# -----------------------------

for index, task in enumerate(dataset, start=1):

    task_id = task["task_id"]

    if task_id in completed:
        continue

    previous_attempts = attempts.get(task_id, 0)

    if previous_attempts >= MAX_ATTEMPTS:
        print(f"\nSkipping task {task_id}: maximum attempts reached.")
        continue

    attempt_number = previous_attempts + 1

    print(f"\nTask {index}/{len(dataset)}")
    print("ID:", task_id)
    print("Attempt:", attempt_number)
    print("Agent is working...")

    start_time = time.perf_counter()

    try:
        answer, performance = ask_agent(task["Question"])

        status = "completed"
        error_message = None

    except requests.exceptions.Timeout as error:
        answer = None
        performance = None
        status = "timeout"
        error_message = str(error)

    except requests.exceptions.RequestException as error:
        answer = None
        performance = None
        status = "error"
        error_message = str(error)

    except Exception as error:
        answer = None
        performance = None
        status = "error"
        error_message = str(error)

    execution_time = round(time.perf_counter() - start_time, 2)

    result = {
        "task_id": task_id,
        "level": task["Level"],
        "question": task["Question"],
        "reference_answer": task["Final answer"],
        "agent_answer": answer,
        "status": status,
        "execution_time_seconds": execution_time,
        "attempt": attempt_number,
        "error": error_message,
        "performance": performance
    }

    # Save result immediately
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )

    print("Status:", status)
    print("Execution time:", execution_time, "seconds")
    print("Answer:", answer)

    if performance:
        print("Ollama performance:", performance)

    if error_message:
        print("Error:", error_message)

    if status == "completed":
        completed.add(task_id)

    attempts[task_id] = attempt_number


print("\nOptimized experiment finished!")
print("Completed tasks:", len(completed))
print("Results saved in:", RESULTS_FILE)