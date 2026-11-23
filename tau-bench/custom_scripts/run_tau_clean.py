import os
import json
import requests

os.environ["OPENAI_API_KEY"] = "sk-dummy-key-for-local-ollama"

from tau_bench.envs import get_env
from tau_bench.envs.base import Action

MODEL_NAME = "qwen3:1.7b"
OLLAMA_URL = "http://localhost:11434/api/chat"
DOMAIN = "retail"
OUTPUT_FILE = "tau_bench_retail_clean_results.json"
NUM_TASKS_TO_RUN = 10

def call_ollama(messages):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 4096
        }
    }
    
    try:
        # ЗМІНЕНО: timeout=None, щоб програма чекала без обмежень
        response = requests.post(OLLAMA_URL, json=payload, timeout=None)
        response.raise_for_status()
        res_data = response.json()
        return res_data.get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        return "I apologize, but an internal system error occurred."

def run_tau_benchmark():
    print(f"=== Starting tau-bench evaluation on domain: {DOMAIN} ===")
    
    env = get_env(
        DOMAIN,
        task_split="test",
        user_strategy="llm",
        user_model="ollama/qwen3:1.7b",
        user_provider="ollama",
        task_index=0
    )
    
    total_available_tasks = len(env.tasks)
    print(f"Total tasks available in domain '{DOMAIN}': {total_available_tasks}")
    print(f"Running evaluation on first {NUM_TASKS_TO_RUN} tasks...\n")
    
    results = []
    
    for task_idx in range(NUM_TASKS_TO_RUN):
        print(f"--- Task {task_idx + 1}/{NUM_TASKS_TO_RUN} (Task Index: {task_idx}) ---")
        
        obs, info = env.reset(task_index=task_idx)
        
        system_instruction = (
            f"{env.wiki}\n\n"
            "You are a professional customer service representative. "
            "Help the customer efficiently based strictly on company rules."
        )
        
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": str(obs)}
        ]
        
        done = False
        raw_reward = 0.0
        step_count = 0
        max_steps = 10
        
        while not done and step_count < max_steps:
            step_count += 1
            
            agent_reply = call_ollama(messages)
            print(f"Step {step_count} | Agent response generated")
            
            action = Action(name="respond", kwargs={"content": agent_reply})
            messages.append({"role": "assistant", "content": agent_reply})
            
            try:
                obs, raw_reward, done, info = env.step(action)
                messages.append({"role": "user", "content": str(obs)})
            except Exception as e:
                print(f"Execution error on step {step_count}: {e}")
                done = True
        
        numeric_reward = 0.0
        if isinstance(raw_reward, (int, float)):
            numeric_reward = float(raw_reward)
        elif isinstance(raw_reward, (tuple, list)):
            for item in raw_reward:
                if isinstance(item, (int, float)):
                    numeric_reward = float(item)
                    break
        elif isinstance(raw_reward, dict):
            numeric_reward = float(raw_reward.get("reward", 0.0))
            
        task_summary = {
            "task_index": task_idx,
            "reward": numeric_reward,
            "steps": step_count,
            "success": numeric_reward > 0.0
        }
        results.append(task_summary)
        print(f"Task {task_idx + 1} Finished | Reward: {numeric_reward} | Steps: {step_count}\n")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"=== Evaluation Complete! Results saved to {OUTPUT_FILE} ===")

if __name__ == "__main__":
    run_tau_benchmark()
