local experiment_prompts_path = std.extVar("APPWORLD_EXPERIMENT_PROMPTS_PATH");
local experiment_configs_path = std.extVar("APPWORLD_EXPERIMENT_CONFIGS_PATH");
local experiment_code_path = std.extVar("APPWORLD_EXPERIMENT_CODE_PATH");
local model_config = {
    "client_name": "litellm",
    "api_type": "chat_completions",
    "name": "anthropic/claude-sonnet-4-20250514",
    "temperature": 0.0,
    "drop_reasoning_content": false,
    "cost_per_token": {"input_cache_hit": 3e-07, "input_cache_miss": 3e-06, "input_cache_write": 3.75e-06, "output": 1.5e-05},
    "retry_after_n_seconds": 15,
    "use_cache": false,
    "max_retries": 100,
};
{
    "type": "simplified",
    "config": {
        "agent": {
            "type": "simplified_function_calling",
            "model_config": model_config + {
                "tool_choice": "auto",
                "parallel_tool_calls": true,
            },
            "api_predictor_config": {
                "mode": "predicted",
                "model_config": model_config,
                "prompt_file_path": experiment_prompts_path + "/api_predictor.txt",
                "demo_task_ids": ["82e2fac_1", "29caf6f_1", "d0b1f43_1"],
                "max_predicted_apis": 20,
            },
            "appworld_config": {
                "random_seed": 100,
                "raise_on_extra_parameters": true,
                "include_direct_functions": true,
                "direct_function_separator": "__",
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
            "prompt_file_path": experiment_prompts_path + "/function_calling_agent/instructions.txt",
            "demo_messages_file_path": experiment_prompts_path + "/function_calling_agent/demos.json",
            "max_steps": 50,
            "log_lm_calls": true,
            "skip_if_finished": true,
        },
        "dataset": "test_challenge",
    },
    "metadata": {
        "model": {
            "file_name": "claude-sonnet-4-20250514-no-reasoning",
            "humanized_name": "Claude Sonnet 4 20250514 No Reasoning",
            "precise_name": "anthropic/claude-sonnet-4-20250514",
            "creator": "anthropic",
            "provider": "anthropic",
        },
        "agent": {
            "file_name": "simplified_function_calling_agent",
            "humanized_name": "Function Calling Agent",
        },
    },
}