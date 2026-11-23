import json
import os
import time
from datasets import load_dataset
from openai import OpenAI

# ==========================================
# API CONFIGURATION (INSERT CREDENTIALS HERE)
# ==========================================
MOCK_MODE = True  # Set to False once you insert a real API key

API_KEY = "sk-fake-key-1234567890"
# Leave as None for standard OpenAI.
# For OpenRouter, Groq, or university servers, specify the URL (e.g., "https://api.groq.com/openai/v1")
BASE_URL = None
MODEL_NAME = "gpt-4o-mini"  # Exact model name provided by the server

RESULTS_FILE = "gaia_api_results.jsonl"
TEXT_ONLY = True  # True = 42 text-only tasks, False = all 53 Level 1 tasks

# Initialize OpenAI-compatible client
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 1. Load GAIA Level 1 validation dataset
dataset = load_dataset("data/GAIA", "2023_level1", split="validation")

tasks = []
for task in dataset:
    file_name = task.get("file_name", "")
    if TEXT_ONLY:
        if not file_name or len(str(file_name).strip()) == 0:
            tasks.append(task)
    else:
        tasks.append(task)

print(f"Total tasks queued for evaluation: {len(tasks)} (MOCK_MODE: {MOCK_MODE})")

def ask_agent_api(question: str) -> str:
    if MOCK_MODE:
        time.sleep(0.5)  # Simulate network latency
        return "Reasoning step 1... Reasoning step 2...\nFinal Answer: 42"

    prompt = f"""You are a precise AI assistant.
Answer the following question as accurately as possible.
Follow all instructions, pay attention to units and calculations.
Give a concise final answer at the very end after 'Final Answer:'.

Question:
{question}"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful and precise analytical assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=1024
    )
    return response.choices[0].message.content.strip()

# Check already completed tasks to avoid duplicate API calls on restart
completed_ids = set()
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                completed_ids.add(json.loads(line)["task_id"])

# Run evaluation loop
for index, task in enumerate(tasks, start=1):
    task_id = task["task_id"]
    if task_id in completed_ids:
        continue

    print(f"\n[{index}/{len(tasks)}] ID: {task_id}")
    start_time = time.perf_counter()

    try:
        answer = ask_agent_api(task["Question"])
        status = "completed"
        error_msg = None
    except Exception as e:
        answer = None
        status = "error"
        error_msg = str(e)

    exec_time = round(time.perf_counter() - start_time, 2)

    result = {
        "task_id": task_id,
        "model": MODEL_NAME,
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
    if answer:
        # Print only the final line for clean console output
        last_line = answer.splitlines()[-1]
        print(f"Output: {last_line}")

print("\nGAIA API evaluation complete!")