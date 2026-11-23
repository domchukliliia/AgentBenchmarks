import os
from tau_bench.types import RunConfig
from tau_bench.run import run

# ==========================================
# OLLAMA SERVER & CHUNK CONFIGURATION
# ==========================================
os.environ["OPENAI_API_KEY"] = "ollama"
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"

PROVIDER = "openai"          # LiteLLM uses OpenAI-compatible endpoint for local Ollama v1
AGENT_MODEL = "qwen2.5:14b"  # Model running on server GPU
USER_MODEL = "qwen2.5:14b"   # User simulator model

# Slice tasks into small batches (e.g., 5 tasks per run):
# Batch 1: START_INDEX = 0, END_INDEX = 5
# Batch 2: START_INDEX = 5, END_INDEX = 10
START_INDEX = 0
END_INDEX = 5

def main():
    print(f"=== Starting tau-bench evaluation on GPU server ({AGENT_MODEL}) | Tasks [{START_INDEX}:{END_INDEX}] ===")
    
    config = RunConfig(
        model_provider=PROVIDER,
        user_model_provider=PROVIDER,
        model=AGENT_MODEL,
        user_model=USER_MODEL,
        num_trials=1,
        env="retail",
        agent_strategy="tool-calling",
        temperature=0.0,
        task_split="test",
        start_index=START_INDEX,
        end_index=END_INDEX,
        task_ids=None,
        log_dir="results_server",
        max_concurrency=1,  # Keep at 1 so Ollama processes requests sequentially without VRAM spikes
        seed=42,
        shuffle=0,
        user_strategy="llm",
        few_shot_displays_path=None,
    )
    
    run(config)
    print(f"=== Batch [{START_INDEX}:{END_INDEX}] finished! Check 'results_server' directory. ===")

if __name__ == "__main__":
    main()