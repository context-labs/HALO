local experiment_prompts_path = std.extVar("APPWORLD_EXPERIMENT_PROMPTS_PATH");
local experiment_configs_path = std.extVar("APPWORLD_EXPERIMENT_CONFIGS_PATH");
local experiment_code_path = std.extVar("APPWORLD_EXPERIMENT_CODE_PATH");
local model_config = {
    "name": "gpt-4.1-mini-2025-04-14",
    "temperature": 0,
    "seed": 100,
    "stop": ["<|endoftext|>"],
    "logprobs": false,
    "top_logprobs": null,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "n": 1,
    "response_format": {"type": "text"},
    "retry_after_n_seconds": 10,
    "max_retries": 50,
};
local demo_task_ids = ["82e2fac_1", "29caf6f_1", "d0b1f43_1"];
{
    "type": "simplified",
    "config": {
        "agent": {
            "type": "simplified_full_code_reflexion",
            "model_config": model_config,
            "api_predictor_config": {
                "model_config": model_config,
                "prompt_file_path": experiment_prompts_path + "/api_predictor.txt",
                "demo_task_ids": demo_task_ids,
                "max_predicted_apis": 16,
            },
            "appworld_config": {
                "random_seed": 100,
            },
            "logger_config": {
                "color": true,
                "verbose": false,
            },
            "compress_api_docs": true,
            "demo_task_ids": demo_task_ids,
            "max_num_retrials": 5,
            "remove_code_demo_comments": true,
            "code_prompt_file_path": experiment_prompts_path + "/full_code_agent/full_code_instructions.txt",
            "retrial_prompt_file_path": experiment_prompts_path + "/full_code_agent/reflexion_instructions.txt",
            "max_steps": 10,
            "max_cost_overall": 100,
            "max_cost_per_task": 10,
            "log_lm_calls": false,
        },
        "dataset": "dev",
    }
}