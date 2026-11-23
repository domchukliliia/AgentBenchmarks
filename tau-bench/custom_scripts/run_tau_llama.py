import os
import json
import requests
import litellm

# 1. Disable litellm strict timeouts for slow CPU generation
os.environ["OPENAI_API_KEY"] = "sk-dummy-key-for-local-ollama"
litellm.request_timeout = 3600  # Give it up to 60 minutes per turn

from tau_bench.envs import get_env
from tau_bench.envs.base import Action

MODEL_NAME = "llama3.2"
OLLAMA_URL = "http://localhost:11434/api/chat"
DOMAIN = "retail"
OUTPUT_FILE = "tau_bench_retail_llama_results.json"
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
        response = requests.post(OLLAMA_URL, json=payload, timeout=None)
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        return "I apologize, but an internal system error occurred."

def format_tools_for_prompt(env):
    tools_text = "AVAILABLE TOOLS (APIs):\n"
    if hasattr(env, 'tools_info'):
        for tool in env.tools_info:
            func = tool.get('function', {})
            name = func.get('name', 'unknown')
            desc = func.get('description', '')
            args = func.get('parameters', {}).get('properties', {})
            tools_text += f"- {name}: {desc}\n  Arguments required: {json.dumps(args)}\n\n"
    return tools_text

def run_tau_benchmark():
    print(f"=== Starting tau-bench evaluation with {MODEL_NAME} ===")
    
    # We use Qwen as the user (client) and Llama 3.2 as the agent
    env = get_env(
        DOMAIN,
        task_split="test",
        user_strategy="llm",
        user_model="ollama/qwen3:1.7b",
        user_provider="ollama",
        task_index=0
    )
    
    total_tasks = len(env.tasks)
    sample_size = min(NUM_TASKS_TO_RUN, total_tasks)
    print(f"Total tasks available: {total_tasks}. Running evaluation on first {sample_size} tasks...\n")
    
    results = []
    
    for task_idx in range(sample_size):
        print(f"--- Task {task_idx + 1}/{sample_size} (Task Index: {task_idx}) ---")
        
        obs, info = env.reset(task_index=task_idx)
        tools_description = format_tools_for_prompt(env)
        
        system_instruction = (
            f"{env.wiki}\n\n"
            f"{tools_description}\n"
            "INSTRUCTIONS FOR YOU (THE AGENT):\n"
            "You are a helpful retail customer service assistant.\n"
            "If you need to search a database, cancel an order, or use ANY tool, you MUST reply exactly in this format and say nothing else:\n"
            "CALL_TOOL: tool_name\n"
            "ARGS: {\"arg_name\": \"value\"}\n\n"
            "If you DO NOT need to use a tool and just want to talk to the user, write your message naturally WITHOUT the CALL_TOOL prefix."
        )
        
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": str(obs)}
        ]
        
        done = False
        raw_reward = 0.0
        step_count = 0
        max_steps = 15
        
        while not done and step_count < max_steps:
            step_count += 1
            
            reply = call_ollama(messages)
            
            if "CALL_TOOL:" in reply and "ARGS:" in reply:
                try:
                    tool_part = reply.split("CALL_TOOL:")[1].split("ARGS:")[0].strip()
                    args_part = reply.split("ARGS:")[1].strip()
                    
                    if args_part.startswith("```json"): args_part = args_part[7:]
                    elif args_part.startswith("```"): args_part = args_part[3:]
                    if args_part.endswith("```"): args_part = args_part[:-3]
                    
                    args_dict = json.loads(args_part.strip())
                    print(f"Step {step_count} | Agent called tool: {tool_part}")
                    action = Action(name=tool_part, kwargs=args_dict)
                    
                    messages.append({"role": "assistant", "content": f"[Used Tool: {tool_part}]"})
                except Exception as e:
                    print(f"Step {step_count} | Tool parsing failed, defaulting to text.")
                    action = Action(name="respond", kwargs={"content": "I had a technical glitch."})
                    messages.append({"role": "assistant", "content": reply})
            else:
                print(f"Step {step_count} | Agent responded to user")
                action = Action(name="respond", kwargs={"content": reply})
                messages.append({"role": "assistant", "content": reply})
                
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
