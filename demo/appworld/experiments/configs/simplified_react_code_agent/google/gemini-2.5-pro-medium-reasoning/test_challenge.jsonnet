local experiment_prompts_path = std.extVar("APPWORLD_EXPERIMENT_PROMPTS_PATH");
local experiment_configs_path = std.extVar("APPWORLD_EXPERIMENT_CONFIGS_PATH");
local experiment_code_path = std.extVar("APPWORLD_EXPERIMENT_CODE_PATH");
{
    "type": "simplified",
    "config": {
        "agent": {
            "type": "simplified_react_code_agent",
            "model_config": {
                "client_name": "litellm",
                "api_type": "chat_completions",
                "name": "gemini/gemini-2.5-pro",
                "reasoning_effort": "medium",
                "temperature": 0.0,
                "drop_reasoning_content": false,
                "cost_per_token": {"input_cache_hit": 3.1e-07, "input_cache_miss": 1.25e-06, "input_cache_write": 0.0, "output": 1e-05},
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
            "file_name": "gemini-2.5-pro-medium-reasoning",
            "humanized_name": "Gemini 2.5 Pro Medium Reasoning",
            "precise_name": "gemini/gemini-2.5-pro",
            "creator": "google",
            "provider": "google",
        },
        "agent": {
            "file_name": "simplified_react_code_agent",
            "humanized_name": "ReAct Code Agent",
        },
    },
}