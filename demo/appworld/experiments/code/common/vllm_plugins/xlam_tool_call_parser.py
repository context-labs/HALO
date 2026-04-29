# type: ignore
import json
from collections.abc import Sequence
from typing import Any

import partial_json_parser
from partial_json_parser.core.options import Allow
from transformers import PreTrainedTokenizerBase
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParser, ToolParserManager
from vllm.entrypoints.openai.tool_parsers.utils import (
    find_common_prefix,
    is_complete_json,
    partial_json_loads,
)
from vllm.logger import init_logger
from vllm.utils import random_uuid


logger = init_logger(__name__)


@ToolParserManager.register_module("xlam")
class xLAMToolParser(ToolParser):  # noqa: N801
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)
        # State for streaming mode
        self.prev_tool_calls: list[dict[str, Any]] = []
        self.current_tools_sent: list[bool] = []
        self.streamed_args: list[str] = []
        # Remove regex since we're parsing direct JSON

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        try:
            # Modified: Direct JSON parsing without looking for ```
            if not model_output.strip().startswith("["):
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            tool_calls_data = json.loads(model_output)
            tool_calls: list[ToolCall] = []

            for idx, call in enumerate(tool_calls_data):
                tool_call = ToolCall(
                    id=f"call_{idx}_{random_uuid()}",
                    type="function",
                    function=FunctionCall(
                        name=call["name"], arguments=json.dumps(call["arguments"])
                    ),
                )
                tool_calls.append(tool_call)

            return ExtractedToolCallInformation(
                tools_called=True, tool_calls=tool_calls, content=None
            )

        except Exception:
            logger.exception("Error extracting tool calls")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        if not current_text.strip().startswith("["):
            return DeltaMessage(content=delta_text)

        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR

        try:
            tool_call_arr = []
            is_complete = []
            try:
                # Parse the JSON array
                start_idx = 0
                while start_idx < len(current_text):
                    obj, end_idx = partial_json_loads(current_text[start_idx:], flags)
                    is_complete.append(
                        is_complete_json(current_text[start_idx : start_idx + end_idx])
                    )
                    start_idx += end_idx
                    tool_call_arr.append(obj)
            except partial_json_parser.core.exceptions.MalformedJSON:
                logger.debug("not enough tokens to parse into JSON yet")
                return None

            # Get current tool call based on state
            current_tool_call: dict[str, Any] = (
                tool_call_arr[self.current_tool_id] if len(tool_call_arr) > 0 else {}
            )

            # Case 1: No tools parsed yet
            if len(tool_call_arr) == 0:
                return None

            # Case 2: Starting a new tool in array
            elif len(tool_call_arr) > 0 and len(tool_call_arr) > self.current_tool_id + 1:
                # Handle any remaining arguments from previous tool
                if self.current_tool_id >= 0:
                    cur_arguments = current_tool_call.get("arguments")
                    if cur_arguments:
                        cur_args_json = json.dumps(cur_arguments)
                        sent = len(self.streamed_args[self.current_tool_id])
                        argument_diff = cur_args_json[sent:]

                        if argument_diff:
                            delta = DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        index=self.current_tool_id,
                                        function=DeltaFunctionCall(
                                            arguments=argument_diff
                                        ).model_dump(exclude_none=True),
                                    )
                                ]
                            )
                            self.streamed_args[self.current_tool_id] += argument_diff
                            return delta

                # Setup new tool
                self.current_tool_id = len(tool_call_arr) - 1
                self.current_tools_sent.append(False)
                self.streamed_args.append("")
                logger.debug("starting new tool %d", self.current_tool_id)
                return None

            # Case 3: Send tool name if not sent yet
            elif not self.current_tools_sent[self.current_tool_id]:
                function_name = current_tool_call.get("name")
                if function_name:
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=f"call_{self.current_tool_id}_{random_uuid()}",
                                function=DeltaFunctionCall(name=function_name).model_dump(
                                    exclude_none=True
                                ),
                            )
                        ]
                    )
                    self.current_tools_sent[self.current_tool_id] = True
                    return delta
                return None

            # Case 4: Stream arguments
            else:
                cur_arguments = current_tool_call.get("arguments")
                if cur_arguments:
                    sent = len(self.streamed_args[self.current_tool_id])
                    cur_args_json = json.dumps(cur_arguments)
                    prev_arguments = self.prev_tool_calls[self.current_tool_id].get("arguments")

                    argument_diff = None
                    if is_complete[self.current_tool_id]:
                        argument_diff = cur_args_json[sent:]
                    elif prev_arguments:
                        prev_args_json = json.dumps(prev_arguments)
                        if cur_args_json != prev_args_json:
                            prefix = find_common_prefix(prev_args_json, cur_args_json)
                            argument_diff = prefix[sent:]

                    if argument_diff is not None:
                        delta = DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    function=DeltaFunctionCall(arguments=argument_diff).model_dump(
                                        exclude_none=True
                                    ),
                                )
                            ]
                        )
                        self.streamed_args[self.current_tool_id] += argument_diff
                        return delta

            self.prev_tool_calls = tool_call_arr
            return None

        except Exception:
            logger.exception("Error in streaming tool calls")
            logger.debug("Skipping chunk due to streaming error")
            return None
