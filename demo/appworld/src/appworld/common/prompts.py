import json
import re
from typing import Any


RULE = "========================================"


def chat_messages_to_string(messages: list[dict[str, Any]]) -> str:
    """Convert openai chat-completions-style messages to a readable string."""

    def _block_to_text(role: str, body_type: str, body: str) -> str:
        header = f"{role} ({body_type})"
        header_line = f"{RULE} {header} {RULE}\n"
        body = f"{body.strip()}\n"
        return header_line + body

    string_ = ""
    for message in messages:
        role = message["role"]
        content = message.get("content", "")
        reasoning_content = message.get("reasoning", "") or message.get("reasoning_content", "")
        encrypted_content = message.get("encrypted_content", "")
        tool_calls: list[dict[str, Any]] = message.get("tool_calls", [])
        tool_calls_str = json.dumps(tool_calls, indent=4)
        body = ""
        included = False
        if reasoning_content or encrypted_content:
            body_type = "reasoning"
            body = reasoning_content or encrypted_content
            string_ += _block_to_text(role, body_type, body)
            included = True
        if content:
            body_type = "content"
            body = content
            string_ += _block_to_text(role, body_type, body)
            included = True
        if tool_calls:
            body_type = "tool_calls"
            body = tool_calls_str
            string_ += _block_to_text(role, body_type, body)
            included = True
        if not included:
            body_type = "unknown"
            body = json.dumps(message, indent=4)
            string_ += _block_to_text(role, body_type, body)
    return string_.strip()


def response_messages_to_string(messages: list[dict[str, Any]]) -> str:
    """Convert openai responses-style messages to a readable string."""

    def _blocks_to_text(blocks: list[dict[str, Any]]) -> str:
        texts: list[str] = []
        for block in blocks or []:
            block_type = block.get("type")
            if block_type in ("input_text", "output_text"):
                txt = block.get("text", "")
                if txt:
                    texts.append(txt)
            elif block_type == "text":
                txt = block.get("text", "")
                if txt:
                    texts.append(txt)
            else:
                try:
                    texts.append(json.dumps(block, indent=2, ensure_ascii=False))
                except Exception:
                    texts.append(str(block))
        return "\n".join(t.strip() for t in texts if t is not None)

    def _pretty(object_: Any) -> str:
        if isinstance(object_, str):
            return object_.strip()
        try:
            return json.dumps(object_, indent=2, ensure_ascii=False)
        except Exception:
            return str(object_)

    string_ = ""
    for message in messages or []:
        type_ = message.get("type", "unknown")
        if type_ == "message":
            role = message.get("role", "assistant")
            body = _blocks_to_text(message.get("content", []))
            body_type = "content" if body else "unknown"
            header = f"{role} ({body_type})"
            string_ += f"{RULE} {header} {RULE}\n{(body or '').strip()}\n"
        elif type_ == "reasoning":
            content = message.get("content", "") or message.get("encrypted_content", "")
            header = "assistant (reasoning)"
            string_ += f"{RULE} {header} {RULE}\n{_pretty(content)}\n"
        elif type_ == "function_call":
            name = message.get("name", "unknown_function")
            args = message.get("arguments", {})
            header = f"tool_call:{name} (function_call)"
            string_ += f"{RULE} {header} {RULE}\n{_pretty(args)}\n"
        elif type_ == "function_call_output":
            call_id = message.get("call_id", "")
            output = message.get("output", "")
            suffix = f" call_id={call_id}" if call_id else ""
            header = f"tool (function_result){suffix}"
            string_ += f"{RULE} {header} {RULE}\n{_pretty(output)}\n"
        else:
            header = f"{type_} (unknown)"
            string_ += f"{RULE} {header} {RULE}\n{_pretty(message)}\n"
    return string_.strip()


def load_prompt_to_chat_messages(
    prompt: str,
    chat_format: str = "openai_lm",
    skip_system_message: bool = False,
    only_header: bool = False,
    only_body: bool = False,
    start_at: int = 0,
    end_at: int = -1,
) -> list[dict[str, str]]:
    """
    Load a prompt delimited with ---+ into a list of openai-styled role-based messages.
    """

    if only_header and only_body:
        raise ValueError("only_header and only_body cannot be both True.")

    assert chat_format in (
        "openai_lm",
        "google_lm",
    ), f"Invalid chat_format: {chat_format}"
    if only_header or only_body:
        prompt_parts = [e.strip() for e in re.split(r"===+", prompt.strip())]
        if len(prompt_parts) != 2:
            raise ValueError(
                f"Invalid prompt. Expected 2 parts for header and body, found {len(prompt_parts)} parts."
            )
        prompt = prompt_parts[0] if only_header else prompt_parts[1]

    message_contents = [e.strip() for e in re.split(r"---+", prompt.strip())]

    messages: list[dict[str, str]] = []

    author_key = "author" if chat_format == "google_lm" else "role"
    bot_name = "bot" if chat_format == "google_lm" else "assistant"
    system_name = "system"
    content_key = "content"
    user_name = "user"

    if not skip_system_message:
        assert len(message_contents) > 1, "Not enough messages to load in LM."
        messages = [{author_key: system_name, content_key: message_contents[0]}]
        message_contents = message_contents[1:]

    if end_at < 0:
        end_at = len(message_contents) + end_at + 1
    for index, message_content in enumerate(message_contents):
        if index < start_at:
            continue
        if index >= end_at:
            break
        role = user_name if (index + 1) % 2 == 1 else bot_name
        messages.append({author_key: role, content_key: message_content})

    return messages
