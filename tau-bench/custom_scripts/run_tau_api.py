import os
from tau_bench.types import RunConfig
from tau_bench.run import run

# ==========================================
# API CONFIGURATION (INSERT CREDENTIALS HERE)
# ==========================================
os.environ["OPENAI_API_KEY"] = "sk-fake-key-1234567890"

# Uncomment the line below and insert your custom server URL (e.g., DataLab LLM):
# os.environ["OPENAI_API_BASE"] = "https://llm.datalab.fei.tuke.sk/v1"

PROVIDER = "openai"          # Keep as "openai" for any OpenAI-compatible API
AGENT_MODEL = "gpt-4o-mini"  # Model name for the agent
USER_MODEL = "gpt-4o-mini"   # Model name for the simulated user

def main():
    print(f"=== Starting tau-bench evaluation via API ({AGENT_MODEL}) ===")
    
    config = RunConfig(
        model_provider=PROVIDER,
        user_model_provider=PROVIDER,
        model=AGENT_MODEL,
        user_model=USER_MODEL,
        num_trials=1,
        env="retail",
        agent_strategy="tool-calling",  # Native tool-calling via API
        temperature=0.0,
        task_split="test",
        start_index=0,
        end_index=1,                    # Set to 1 for key testing; change to 10 (or -1 for all) in production
        task_ids=None,
        log_dir="results_api",
        max_concurrency=1,              # Can be increased to 2-4 for fast cloud APIs
        seed=42,
        shuffle=0,
        user_strategy="llm",
        few_shot_displays_path=None,
    )
    
    try:
        run(config)
    except Exception as e:
        print("\n[!] Exception caught during execution:")
        print(f"Error Type: {type(e).__name__}")
        print(f"Details: {e}")
        print("\nIf the error is 'AuthenticationError' or 'Incorrect API key', the setup is valid and ready for a real key.")

if __name__ == "__main__":
    main()