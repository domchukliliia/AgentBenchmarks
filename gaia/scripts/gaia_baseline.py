
import requests
import json
import os
from datasets import load_dataset

MODEL = "qwen3:1.7b"
OLLAMA_URL = "http://localhost:11434/api/chat"
RESULTS_FILE = "gaia_baseline_results.jsonl"

# Завантаження GAIA Level 1
dataset = load_dataset(
    "data/GAIA",
    "2023_level1",
    split="validation"
)

# Перевіряємо, які завдання вже виконані
completed = set()

if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                completed.add(json.loads(line)["task_id"])

print("Total tasks:", len(dataset))
print("Already completed:", len(completed))

def ask_agent(question):
    prompt = f"""
You are an AI assistant solving benchmark tasks.

Answer the question as accurately as possible.
Follow all instructions in the question.
Pay attention to units, calculations and rounding.
Provide only your final answer.

Question:
{question}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0
            }
        },
        timeout=300
    )

    response.raise_for_status()
    return response.json()["message"]["content"]

# Послідовно виконуємо завдання
for index, task in enumerate(dataset, start=1):

    task_id = task["task_id"]

    if task_id in completed:
        continue

    print(f"\nTask {index}/{len(dataset)}")
    print("ID:", task_id)
    print("Agent is working...")

    try:
        answer = ask_agent(task["Question"])

        result = {
            "task_id": task_id,
            "level": task["Level"],
            "question": task["Question"],
            "reference_answer": task["Final answer"],
            "agent_answer": answer,
            "status": "completed"
        }

    except Exception as error:
        print("ERROR:", error)

        result = {
            "task_id": task_id,
            "level": task["Level"],
            "question": task["Question"],
            "reference_answer": task["Final answer"],
            "agent_answer": None,
            "status": "error",
            "error": str(error)
        }

    # Зберігаємо результат одразу після кожного завдання
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("Status:", result["status"])
    print("Answer:", result["agent_answer"])

print("\nBaseline experiment finished!")
print("Results saved in:", RESULTS_FILE)
