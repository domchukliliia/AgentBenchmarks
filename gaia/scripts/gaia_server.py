import json
import os
import time
from datasets import load_dataset
from openai import OpenAI

# ==========================================
# CHUNK & MODEL CONFIGURATION
# ==========================================
MODEL_NAME = "qwen2.5:14b"
BASE_URL = "http://localhost:11434/v1"
API_KEY = "ollama"  # Required placeholder for OpenAI client

# Slice tasks into chunks to avoid occupying the GPU for too long (42 text-only tasks total):
# Chunk 1: START_IDX = 0,  END_IDX = 15
# Chunk 2: START_IDX = 15, END_IDX = 30
# Chunk 3: START_IDX = 30, END_IDX = 42
START_IDX = 0
END_IDX = 15

RESULTS_FILE = "gaia_qwen14b_results.jsonl"
TEXT_ONLY = True  # True = 42 text-only tasks, False = all 53 Level 1 tasks

# Initialize local Ollama client
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Locate dataset directory regardless of working directory
base_dir = os.path.dirname(os.path.abspath(__file__))
possible_paths = [
    os.path.join(base_dir, "data", "GAIA"),
    os.path.join(os.path.dirname(base_dir), "data", "GAIA"),
    "gaia-benchmark/GAIA"
]
dataset_path = next((p for p in possible_paths if os.path.exists(p)), "gaia-benchmark/GAIA")
print(f"Loading dataset from: {dataset_path}")

dataset = load_dataset(dataset_path, "2023_level1", split="validation")

# Filter tasks
all_tasks = []
for task in dataset:
    file_name = task.get("file_name", "")
    if TEXT_ONLY:
        if not file_name or len(str(file_name).strip()) == 0:
            all_tasks.append(task)
    else:
        all_tasks.append(task)

# Slice the requested chunk
chunk_tasks = all_tasks[START_IDX:END_IDX]
print(f"Total available tasks: {len(all_tasks)} | Running chunk [{START_IDX}:{END_IDX}] ({len(chunk_tasks)} tasks)")

def ask_agent_ollama(question: str) -> str:
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

# Skip already completed tasks in case of interruption
completed_ids = set()
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                completed_ids.add(json.loads(line)["task_id"])

# Evaluation loop
for offset, task in enumerate(chunk_tasks):
    global_idx = START_IDX + offset + 1
    task_id = task["task_id"]

    if task_id in completed_ids:
        print(f"[{global_idx}/{len(all_tasks)}] ID: {task_id} -> Already completed, skipping.")
        continue

    print(f"\n[{global_idx}/{len(all_tasks)}] ID: {task_id}")
    start_time = time.perf_counter()

    try:
        answer = ask_agent_ollama(task["Question"])
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
        last_line = answer.splitlines()[-1]
        print(f"Output: {last_line}")

print(f"\nChunk [{START_IDX}:{END_IDX}] finished! Results saved to {RESULTS_FILE}")