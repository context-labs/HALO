local experiment_prompts_path = std.extVar("APPWORLD_EXPERIMENT_PROMPTS_PATH");
local experiment_configs_path = std.extVar("APPWORLD_EXPERIMENT_CONFIGS_PATH");
local experiment_code_path = std.extVar("APPWORLD_EXPERIMENT_CODE_PATH");
{
    "type": "legacy.plain",
    "config": {
        "agent": {
            "add_login_shortcut": false,
            "allow_datetime_change": false,
            "import_utils": false,
            "load_ground_truth": false,
            "max_steps": 10,
            "munchify_response": false,
            "parse_datetimes": false,
            "show_api_response_schemas": true,
            "show_io": true,
            "skip_if_complete": false,
            "solver_config": {
                "api_call_syntax": "apis",
                "compress_api_docs": true,
                "demo_predicted_api_docs_only": false,
                "demo_task_ids": [
                    "82e2fac_1",
                    "29caf6f_1",
                    "d0b1f43_1"
                ],
                "language_model": {
                    "model": "gpt-4o-2024-05-13",
                    "retry_after_n_seconds": 15,
                    "stop": [
                        "<|endoftext|>"
                    ],
                    "type": "openai_lm",
                    "usage_namespace": "appworld::agent"
                },
                "max_num_retrials": 5,
                "oracle_first_step": false,
                "remove_code_demo_comments": true,
                "retrial_type": "reflexion",
                "seed": 100
            },
            "type": "full_code_reflexion_agent",
            "wait_seconds": 0
        },
        "test_dataset": "test_normal",
        "train_dataset": "train"
    }
}