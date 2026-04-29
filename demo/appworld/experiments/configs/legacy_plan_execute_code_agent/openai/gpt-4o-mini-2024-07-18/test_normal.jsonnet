local experiment_prompts_path = std.extVar("APPWORLD_EXPERIMENT_PROMPTS_PATH");
local experiment_configs_path = std.extVar("APPWORLD_EXPERIMENT_CONFIGS_PATH");
local experiment_code_path = std.extVar("APPWORLD_EXPERIMENT_CODE_PATH");
{
    "type": "legacy.recoma",
    "config": {
        "models": {
            "appworld_init": {
                "type": "appworld_loader",
                "next_model": "appworld_plan_exec"
            },
            "appworld_plan_exec": {
                "type": "appworld_plan_and_execute",
                "plan_model": "appworld_planner",
                "exec_model": "appworld_exec",
                "add_history": true,
            },
            "appworld_planner": {
                "type": "appworld_prompted_lm",
                "prompt_file": experiment_prompts_path + "/plan_execute_code_agent/plan_instructions.txt",
                "generator_params": {
                    "type": "lite_llm",
                    "model": "gpt-4o-mini-2024-07-18",
                    "max_tokens": 400,
                    "seed": 123,
                    "stop": ["\n\n"],
                    "temperature": 0,
                    "use_cache": true
                }
            },
            "appworld_exec": {
                "type": "appworld_react_controller",
                "action_model": "action",
                "observation_model": "environment",
                "add_roles": true,
                "max_output_length": 20000,
            },
            "action": {
                "type": "appworld_prompted_lm",
                "prompt_file": experiment_prompts_path + "/plan_execute_code_agent/execute_instructions.txt",
                "max_prompt_length": 50000,
                "generator_params": {
                    "type": "lite_llm",
                    "model": "gpt-4o-mini-2024-07-18",
                    "max_tokens": 400,
                    "seed": 123,
                    "stop": ["```\n"],
                    "temperature": 0,
                    "use_cache": true
                }
            },
            "environment": {
                "type": "appworld_env"
            }
        },
        "search": {
            "type": "best_first",
            "start_model": "appworld_init",
            "answerer": {
                "type": "appworld_answerer",
            },
            "stopping_conditions": [
                {"type": "max_env_calls", "max_env_calls": 500}, // Not necessary; mainly there to catch any rogue usage
                {"type": "max_llm_calls", "max_llm_calls": 100}, // Key termination criterion
                {"type": "max_llm_cost", "max_llm_cost": 10.00} // Not necessary; mainly there to catch any rogue usage
            ]
        },
        "reader": {
            "type": "appworld_reader",
            "dataset_name": "test_normal",
            "tasks_per_gen": 3,
        }
    }
}