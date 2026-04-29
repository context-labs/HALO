import json
import re
from typing import Any

from appworld.common.types import FromDict


class ToolParser(FromDict):
    def parse(self, text: str) -> list[dict[str, Any]]:
        raise NotImplementedError


@ToolParser.register("kimi-k2-instruct")
class KimiK2InstructToolParser(ToolParser):
    # Note: Not needed with the official Kimi provider.
    def parse(self, text: str) -> list[dict[str, Any]]:
        # Taken from:
        # https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/docs/tool_call_guidance.md
        if "<|tool_calls_section_begin|>" not in text:
            return []
        pattern = r"<\|tool_calls_section_begin\|>(.*?)<\|tool_calls_section_end\|>"
        tool_calls_sections = re.findall(pattern, text, re.DOTALL)
        func_call_pattern = r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w\.]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>.*?)\s*<\|tool_call_end\|>"
        tool_calls: list[dict[str, Any]] = []
        for match in re.findall(func_call_pattern, tool_calls_sections[0], re.DOTALL):
            function_id, function_args = match
            function_name = function_id.split(".")[1].split(":")[0]
            tool_calls.append(
                {
                    "id": function_id,
                    "type": "function",
                    "function": {"name": function_name, "arguments": function_args},
                }
            )
        return tool_calls


@ToolParser.register("glm-4-5-air")
class Glm45ToolParser(ToolParser):
    # Note: Not needed with the official Glm provider.
    def parse(self, text: str) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        try:
            raw_tool_calls = json.loads(text)
        except json.JSONDecodeError:
            raw_tool_calls = []
        for raw_tool_call in raw_tool_calls:
            name = raw_tool_call["name"]
            arguments = json.dumps(raw_tool_call["parameters"], ensure_ascii=False)
            tool_calls.append(
                {
                    "id": None,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
        return tool_calls


@ToolParser.register("deepseek-v3.1")
class DeepSeekV31ToolParser(ToolParser):
    # Note: Not needed with the official Deepseek provider.
    def parse(self, text: str) -> list[dict[str, Any]]:
        # Taken from:
        # https://github.com/sgl-project/sglang/blob/b1fed43051925060523e1a1a240cceca5ad28968/python/sglang/srt/function_call/deepseekv31_detector.py
        tool_calls: list[dict[str, Any]] = []
        call_regex = r"<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>"  # noqa: RUF001
        detail_regex = r"<｜tool▁call▁begin｜>function<｜tool▁sep｜>(.*)\n```json\n(.*)\n```<｜tool▁call▁end｜>"  # noqa: RUF001
        match_result_list = re.findall(call_regex, text, re.DOTALL)
        for match_result in match_result_list:
            detail = re.search(detail_regex, match_result, re.DOTALL)
            if detail is None:
                continue
            try:
                name = detail.group(1).strip()
                arguments = detail.group(2).strip()
            except IndexError:
                continue
            try:
                arguments = json.dumps(json.loads(arguments), ensure_ascii=False)
            except json.JSONDecodeError:
                continue
            tool_calls.append(
                {
                    "id": None,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
        return tool_calls


@ToolParser.register("longcat")
class LongCatToolParser(ToolParser):
    # Note: Needed with the official LongCat provider.
    # (As of writing, this is the only post-hoc tool parser needed for the runs)
    def parse(self, text: str) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        pattern = r"<longcat_tool_call>(.*?)</?longcat_tool_call>"
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                tool_call_json = json.loads(match)
                name = tool_call_json["name"]
                arguments = json.dumps(tool_call_json.get("arguments", {}), ensure_ascii=False)
                tool_calls.append(
                    {
                        "id": None,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                )
            except json.JSONDecodeError:
                continue
        return tool_calls
