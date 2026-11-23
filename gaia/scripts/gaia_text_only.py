import json
import os
import time
import requests
from datasets import load_dataset

MODEL = "qwen3:1.7b"
OLLAMA_URL = "http://localhost:11434/api/chat"
RESULTS_FILE = "gaia_text_only_results.jsonl"
TIMEOUT = 300

# 1. Load GAIA Level 1 validation dataset
dataset = load_dataset("data/GAIA", "2023_level1", split="validation")

# 2. Filter tasks that do NOT require attached files
text_tasks = []
for task in dataset:
    file_name = task.get("file_name", "")
    # If the file_name field is empty, it is a text-only task
    if not file_name or len(str(file_name).strip()) == 0:
        text_tasks.append(task)

print(f"Total tasks in Level 1: {len(dataset)}")
print(f"Text-only tasks found (without files): {len(text_tasks)}")

def ask_agent(question):
    prompt = f"""You are a precise AI assistant.
Answer the following question as accurately as possible.
Follow all instructions, pay attention to units and calculations.
Give a concise final answer at the very end after 'Final Answer:'.

Question:
{question}"""

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_ctx": 4096
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()["message"]["content"].strip()

# 3. Run evaluation on the text-only subset
completed_ids = set()
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                completed_ids.add(json.loads(line)["task_id"])

for index, task in enumerate(text_tasks, start=1):
    task_id = task["task_id"]
    if task_id in completed_ids:
        continue

    print(f"\n[Text Task {index}/{len(text_tasks)}] ID: {task_id}")
    start_time = time.perf_counter()

    try:
        answer = ask_agent(task["Question"])
        status = "completed"
        error_msg = None
    except Exception as e:
        answer = None
        status = "error"
        error_msg = str(e)

    exec_time = round(time.perf_counter() - start_time, 2)

    result = {
        "task_id": task_id,
        "question": task["Question"],
        "reference_answer": task["Final answer"],
        "agent_answer": answer,
        "status": status,
        "execution_time": exec_time,
        "error": error_msg
    }

    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Status: {status} | Time: {exec_time}s")
    print(f"Answer: {answer}")

print("\nText-only GAIA experiment finished!")