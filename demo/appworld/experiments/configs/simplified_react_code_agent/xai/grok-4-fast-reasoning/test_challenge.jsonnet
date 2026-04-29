local experiment_prompts_path = std.extVar("APPWORLD_EXPERIMENT_PROMPTS_PATH");
local experiment_configs_path = std.extVar("APPWORLD_EXPERIMENT_CONFIGS_PATH");
local experiment_code_path = std.extVar("APPWORLD_EXPERIMENT_CODE_PATH");
{
    "type": "simplified",
    "config": {
        "agent": {
            "type": "simplified_react_code_agent",
            "model_config": {
                "client_name": "openai",
                "api_type": "chat_completions",
                "base_url": "https://api.x.ai/v1",
                "api_key_env_name": "XAI_API_KEY",
                "name": "grok-4-fast-reasoning",
                "temperature": 0.0,
                "drop_reasoning_content": false,
                "cost_per_token": {"input_cache_hit": 5e-08, "input_cache_miss": 2e-07, "input_cache_write": 0.0, "output": 5e-07},
                "retry_after_n_seconds": 15,
                "use_cache": false,
                "max_retries": 100,
            },
            "appworld_config": {
                "random_seed": 100,
                "raise_on_extra_parameters": true,
            },
            "logger_config": {
                "color": true,
                "verbose": true,
            },
            "usage_tracker_config": {
                "max_cost_overall": 1000,
                "max_cost_per_task": 10,
                "max_output_tokens_per_task": 100000,
            },
            "prompt_file_path": experiment_prompts_path + "/react_code_agent/instructions.txt",
            "ignore_multiple_calls": true,
            "max_prompt_length": null,
            "max_output_length": null,
            "max_steps": 50,
            "log_lm_calls": true,
            "skip_if_finished": true,
        },
        "dataset": "test_challenge",
    },
    "metadata": {
        "model": {
            "file_name": "grok-4-fast-reasoning",
            "humanized_name": "Grok 4 Fast Reasoning",
            "precise_name": "grok-4-fast-reasoning",
            "creator": "xai",
            "provider": "xai",
        },
        "agent": {
            "file_name": "simplified_react_code_agent",
            "humanized_name": "ReAct Code Agent",
        },
    },
}