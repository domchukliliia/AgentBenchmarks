
import requests
import json
from datasets import load_dataset

MODEL = "qwen3:1.7b"
OLLAMA_URL = "http://localhost:11434/api/chat"

# Завантажуємо GAIA Level 1 validation
dataset = load_dataset(
    "data/GAIA",
    "2023_level1",
    split="validation"
)

def ask_agent(question):
    prompt = f"""
You are an AI assistant solving benchmark tasks.
Answer the question as accurately as possible.
Give a concise final answer.
If you do not know the answer, say so.

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

# Поки тестуємо лише одне завдання
task = dataset[0]

print("Task ID:", task["task_id"])
print("\nAgent is working...\n")

answer = ask_agent(task["Question"])

print("Agent answer:")
print(answer)

# Зберігаємо результат
result = {
    "task_id": task["task_id"],
    "question": task["Question"],
    "reference_answer": task["Final answer"],
    "agent_answer": answer
}

with open("gaia_results.json", "w", encoding="utf-8") as file:
    json.dump(result, file, ensure_ascii=False, indent=4)

print("\nResult saved to gaia_results.json")
