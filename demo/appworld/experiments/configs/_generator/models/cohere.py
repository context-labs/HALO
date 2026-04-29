# Source: https://cohere.com/pricing
MODEL_INFOS = [
    {
        "model_name": "command-a-03-2025",
        "client_name": "openai",
        "model_id": "command-a-03-2025",
        "model_kwargs": {
            "api_type": "chat_completions",
            "base_url": "https://api.cohere.ai/compatibility/v1",
            "api_key_env_name": "COHERE_API_KEY",
            "temperature": 0,
            "seed": 100,
            "tool_parser_name": None,
            "parallel_tool_calls": None,  # Not supported by Cohere (raises error, even with False).
            "cost_per_token": {
                "input_cache_miss": 2.50 / 1e6,
                "input_cache_hit": 2.50 / 1e6,
                "input_cache_write": 0.0,
                "output": 10 / 1e6,
            },
        },
        "function_calling": True,
        "tool_choice": "auto",
        "function_calling_demos": True,
        "part_of": ["v3"],
        "provider": "cohere",
    },
]
