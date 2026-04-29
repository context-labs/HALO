MODEL_INFOS = [
    {
        "model_name": "ernie-4.5-21b-a3b-thinking",  # NOTE: Not tested yet.
        "client_name": "openai",
        "model_id": "baidu/ERNIE-4.5-21B-A3B-Thinking",
        "model_kwargs": {
            "api_type": "chat_completions",
            "temperature": 0,
            "seed": 100,
            "api_key_env_name": "NO_API_KEY",
            "base_url": "{MODEL_SERVER_URL}/v1",
            "max_completion_tokens": 3000,  # For reasonable speed. Remove if you know model won't get stuck here.
            "tool_parser_name": None,
            "parallel_tool_calls": True,
            "cost_per_token": {
                "input_cache_miss": 0.0,
                "input_cache_hit": 0.0,
                "input_cache_write": 0.0,
                "output": 0.0,
            },
        },
        "function_calling": True,
        "tool_choice": "auto",
        "function_calling_demos": False,
        "remove_function_property_keys": [
            "exclusiveMinimum",
            "exclusiveMaximum",
            "minimum",
            "maximum",
        ],  # only needed for vllm because of a bug
        "model_server_config": {
            "enabled": True,
            "command": (  # vllm parsers are underway, see: https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Thinking
                "vllm serve baidu/ERNIE-4.5-21B-A3B-Thinking --reasoning-parser **TODO** --max-num-seqs 5 --max-model-len **TODO** "
                "--enable-auto-tool-choice --tool-call-parser **TODO** --port {port}"
            ),
            "timeout": 600,
            "show_logs": False,
        },
        "part_of": ["vn", "vllm"],
        "provider": "vllm",
    },
]
