# Source: https://www.minimax.io/platform/document/pricing
MODEL_INFOS = [
    {
        "model_name": "minimax-m1",  # NOTE: Working but extremely slow now.
        "client_name": "openai",
        "model_id": "MiniMax-M1",
        "model_kwargs": {
            "api_type": "chat_completions",
            "base_url": "https://api.minimax.io/v1",
            "api_key_env_name": "MINIMAX_API_KEY",
            "temperature": 0,
            "seed": 100,
            "tool_parser_name": None,
            "parallel_tool_calls": True,
            "cost_per_token": {
                "input_cache_miss": 0.4 / 1e6,
                "input_cache_hit": 0.4 / 1e6,
                "input_cache_write": 0.0,
                "output": 2.2 / 1e6,
            },
        },
        "function_calling": True,
        "tool_choice": "auto",
        "function_calling_demos": True,
        "part_of": ["vn"],
        "provider": "minimax",
    },
    {
        "model_name": "minimax-m2",  # NOTE: This is not tested yet.
        "client_name": "openai",
        "model_id": "MiniMax-M2",
        "model_kwargs": {
            "api_type": "chat_completions",
            "base_url": "https://api.minimax.io/v1",
            "api_key_env_name": "MINIMAX_API_KEY",
            "temperature": 0,
            "seed": 100,
            "tool_parser_name": None,
            "parallel_tool_calls": True,
            "cost_per_token": {
                "input_cache_miss": 0.3 / 1e6,
                "input_cache_hit": 0.3 / 1e6,
                "input_cache_write": 0.0,
                "output": 1.2 / 1e6,
            },
        },
        "function_calling": True,
        "tool_choice": "auto",
        "function_calling_demos": True,
        "part_of": ["vn"],
        "provider": "minimax",
    },
]
