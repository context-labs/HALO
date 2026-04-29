# Source: https://huggingface.co/nvidia/collections
# TODO: Add Llama-3_3-Nemotron-Super-49B-v1_5, nvidia/Llama-3_1-Nemotron-Ultra-253B-v1
# NOTE: Some are also available on openrouter, consider moving them.
model_names = ["nvidia/NVIDIA-Nemotron-Nano-12B-v2", "nvidia/NVIDIA-Nemotron-Nano-9B-v2"]
MODEL_INFOS = [
    {
        "model_name": model_name.lower().split("/")[-1]
        + ("-with" if enable_thinking else "-without")
        + "-reasoning",
        "client_name": "openai",
        "model_id": model_name,
        "model_kwargs": {
            "api_type": "chat_completions",
            "temperature": 0,
            "seed": 100,
            "api_key_env_name": "NO_API_KEY",
            "base_url": "{MODEL_SERVER_URL}/v1",
            "max_completion_tokens": 3000,  # For reasonable speed. Remove if you know model won't get stuck here.
            "extra_body": {"chat_template_kwargs": {"enable_thinking": enable_thinking}},
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
            "command": (
                "LITERAL_DOUBLE_QUOTE"
                f"vllm serve {model_name} --max-num-seqs 10 --port {{port}} "
                + "--trust-remote-code --mamba_ssm_cache_dtype float32 --enable-auto-tool-choice --tool-call-parser nemotron_json "
                "LITERAL_DOUBLE_QUOTE "
                " + LITERAL_DOUBLE_QUOTE--tool-parser-plugin LITERAL_DOUBLE_QUOTE + experiment_code_path + LITERAL_DOUBLE_QUOTE/common/vllm_plugins/nemotron_toolcall_parser.py LITERAL_DOUBLE_QUOTE"
            ),
            "timeout": 600,
            "show_logs": False,
        },
        "part_of": ["vn", "vllm"],
        "provider": "vllm",
    }
    for model_name in model_names
    for enable_thinking in [True, False]
]
