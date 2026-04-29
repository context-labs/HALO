import json

from appworld.common.prompts import RULE, chat_messages_to_string, response_messages_to_string


class TestPrompts:
    def test_chat_messages_to_string(self) -> None:
        messages = [
            # 1) plain content
            {"role": "user", "content": "Hello there!"},
            # 2) reasoning (plaintext)
            {"role": "assistant", "reasoning": "hidden chain"},
            # 3) reasoning (encrypted fallback)
            {"role": "assistant", "encrypted_content": "ENCRYPTED-REASONING-BLOB"},
            # 4) tool calls
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "lookup", "arguments": {"q": "ping"}},
                    },
                    {
                        "type": "function",
                        "function": {"name": "math", "arguments": {"x": 1, "y": 2}},
                    },
                ],
            },
            # 5) unknown (no content/reasoning/tool_calls)
            {"role": "system"},
        ]
        out = chat_messages_to_string(messages)
        # Build the expected string step-by-step to ensure we match formatting exactly.
        expected_parts = []
        # 1) content
        expected_parts.append(f"{RULE} user (content) {RULE}\nHello there!\n")
        # 2) reasoning (plaintext)
        expected_parts.append(f"{RULE} assistant (reasoning) {RULE}\nhidden chain\n")
        # 3) reasoning (encrypted fallback)
        expected_parts.append(f"{RULE} assistant (reasoning) {RULE}\nENCRYPTED-REASONING-BLOB\n")
        # 4) tool_calls (pretty-printed JSON with indent=4, as in the function)
        tool_calls_json = json.dumps(
            [
                {"type": "function", "function": {"name": "lookup", "arguments": {"q": "ping"}}},
                {"type": "function", "function": {"name": "math", "arguments": {"x": 1, "y": 2}}},
            ],
            indent=4,
        )
        expected_parts.append(f"{RULE} assistant (tool_calls) {RULE}\n{tool_calls_json}\n")
        # 5) unknown
        expected_parts.append(f"{RULE} system (unknown) {RULE}\n")
        expected_parts.append(json.dumps({"role": "system"}, indent=4) + "\n")
        expected = "".join(expected_parts).strip()
        # Exact match ensures headers, spacing, and newlines are correct.
        assert out == expected
        # A few targeted sanity checks (redundant but helpful for clarity)
        assert out.count(RULE) == 5 * 2  # each header line prints RULE twice
        assert "assistant (tool_calls)" in out
        assert '"lookup"' in out and '"math"' in out
        assert "ENCRYPTED-REASONING-BLOB" in out

    def test_response_messages_to_string(self) -> None:
        messages = [
            # 1) type=message with multiple block types (input_text, text, and a non-text block)
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is the weather?"},
                    {"type": "text", "text": "Extra details…"},
                    {
                        "type": "image",
                        "id": "img_123",
                        "metadata": {"w": 100, "h": 80},
                    },  # non-text -> JSON fallback
                ],
            },
            # 2) reasoning with plaintext content
            {"type": "reasoning", "content": "chain-of-thought (redacted in prod)"},
            # 3) reasoning with encrypted_content fallback
            {"type": "reasoning", "encrypted_content": "ENCRYPTED-REASONING"},
            # 4) function_call
            {
                "type": "function_call",
                "name": "search_web",
                "arguments": {"q": "pytest docs", "freshness": "7d"},
            },
            # 5) function_call_output with call_id and dict output
            {
                "type": "function_call_output",
                "call_id": "fc_42",
                "output": {"results": 2, "top": ["A", "B"]},
            },
            # 6) unknown type falls back to pretty-print of the whole object
            {"type": "mystery", "foo": 1, "bar": {"baz": True}},
        ]
        out = response_messages_to_string(messages)
        # Manually craft expected output for exact matching.
        # 1) message
        image_block_json = json.dumps(
            {"type": "image", "id": "img_123", "metadata": {"w": 100, "h": 80}},
            indent=2,
            ensure_ascii=False,
        )
        msg_section = (
            f"{RULE} user (content) {RULE}\n"
            f"What is the weather?\n"
            f"Extra details…\n"
            f"{image_block_json}\n"
        )
        # 2) reasoning (plaintext)
        reasoning_plain = (
            f"{RULE} assistant (reasoning) {RULE}\n" "chain-of-thought (redacted in prod)\n"
        )
        # 3) reasoning (encrypted fallback)
        reasoning_encrypted = f"{RULE} assistant (reasoning) {RULE}\n" "ENCRYPTED-REASONING\n"
        # 4) function_call
        fn_args_json = json.dumps(
            {"q": "pytest docs", "freshness": "7d"}, indent=2, ensure_ascii=False
        )
        function_call = f"{RULE} tool_call:search_web (function_call) {RULE}\n" f"{fn_args_json}\n"
        # 5) function_call_output
        fn_output_json = json.dumps({"results": 2, "top": ["A", "B"]}, indent=2, ensure_ascii=False)
        function_call_output = (
            f"{RULE} tool (function_result) call_id=fc_42 {RULE}\n" f"{fn_output_json}\n"
        )
        # 6) unknown
        unknown_json = json.dumps(
            {"type": "mystery", "foo": 1, "bar": {"baz": True}}, indent=2, ensure_ascii=False
        )
        unknown_section = f"{RULE} mystery (unknown) {RULE}\n" f"{unknown_json}\n"
        expected = (
            msg_section
            + reasoning_plain
            + reasoning_encrypted
            + function_call
            + function_call_output
            + unknown_section
        ).strip()
        assert out == expected
        # Additional spot checks to ensure branch coverage and formatting
        assert out.count(RULE) == 6 * 2  # 6 headers, each prints RULE twice
        assert "user (content)" in out
        assert "assistant (reasoning)" in out  # appears twice
        assert "tool_call:search_web (function_call)" in out
        assert "tool (function_result) call_id=fc_42" in out
        # Ensure the non-text block was JSON-dumped
        assert '"type": "image"' in out and '"id": "img_123"' in out
