import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

import litellm
import numpy as np
import openai
import tiktoken
from diskcache import Cache
from openai import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    Completion,
    ConflictError,
    InternalServerError,
    NotFoundError,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from appworld.common.collections import getter_plus
from appworld.common.finders import find_all as _find_all
from appworld.common.finders import find_one as _find_one
from appworld.common.io import file_locked, read_json, write_json
from appworld_agents.code.legacy.plain.language_models import LanguageModel


litellm.drop_params = True
openai_cache_directory = os.path.expanduser(os.path.join("~", ".cache", "openai_calls"))
openai_plain_completion_cache = Cache(os.path.join(openai_cache_directory, "plain_completion"))
openai_chat_completion_cache = Cache(os.path.join(openai_cache_directory, "chat_completion"))
openai_embedding_cache = Cache(os.path.join(openai_cache_directory, "embedding"))


RETRY_ERROR = (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)


@dataclass
class ModelInformation:
    name: str
    provider: str
    completion_type: str
    token_limits: dict[Literal["input", "output"], int]


class ModelsInformation:
    _list: list[ModelInformation]

    @classmethod
    def initialize(cls):
        data = {
            "openai": {
                "plain": {
                    "code-davinci-002": {"token_limits": (8000, 4096)},
                    "text-davinci-002": {"token_limits": (4096, 4096)},
                    "text-davinci-003": {"token_limits": (4096, 4096)},
                },
                "chat": {
                    "gpt-3.5-turbo": {"token_limits": (16384, 4096)},
                    "gpt-3.5-turbo-0125": {"token_limits": (16384, 4096)},
                    "gpt-3.5-turbo-0301": {"token_limits": (4096, 4096)},
                    "gpt-3.5-turbo-16k": {"token_limits": (16384, 4096)},
                    "gpt-3.5-turbo-16k-0613": {"token_limits": (16384, 4096)},
                    "gpt-4": {"token_limits": (8192, 4096)},
                    "gpt-4-0613": {"token_limits": (8192, 4096)},
                    "gpt-4-0314": {"token_limits": (8192, 4096)},
                    "gpt-4-32k": {"token_limits": (32768, 4096)},
                    "gpt-4o-2024-05-13": {
                        "token_limits": (60000, 4096)
                    },  # it's 128K but it's to save cost. ##
                    "gpt-4-turbo-2024-04-09": {
                        "token_limits": (60000, 4096)
                    },  # it's 128K but it's to save cost. ##
                    "gpt-4-1106-preview": {
                        "token_limits": (60000, 4096)
                    },  # it's 128K but it's mainly to save cost.
                    "gpt-4o-mini-2024-07-18": {
                        "token_limits": (60000, 4096)
                    },  # it's 128K, 16K but it's mainly to save cost.
                    "o1-preview-2024-09-12": {
                        "token_limits": (60000, 4096)
                    },  # it's 128K, 65K but it's mainly to save cost.
                    "o1-mini-2024-09-12": {
                        "token_limits": (60000, 4096)
                    },  # it's 128K, 33K but it's mainly to save cost.
                },
            },
            "litellm": {
                "plain": {},
                "chat": {
                    "together_ai/google/gemma-2b-it": {"token_limits": (4096, 4096)},
                    "together_ai/deepseek-ai/deepseek-coder-33b-instruct": {
                        "token_limits": (16384, 4096)
                    },
                    "gemini/gemini-1.5-pro-latest": {
                        "token_limits": (60000, 4096)  # it's 128K but it's to save cost. ##
                    },
                    "together_ai/meta-llama/Llama-3-70b-chat-hf": {"token_limits": (8192, 4096)},
                    "together_ai/codellama/CodeLlama-13b-Instruct-hf": {
                        "token_limits": (16384, 4096)
                    },
                    "together_ai/codellama/CodeLlama-34b-Python-hf": {
                        "token_limits": (16384, 4096)
                    },
                    "together_ai/codellama/CodeLlama-34b-Instruct-hf": {
                        "token_limits": (16384, 4096)
                    },
                    "together_ai/codellama/CodeLlama-70b-Instruct-hf": {
                        "token_limits": (16384, 4096)
                    },
                    "together_ai/Phind/Phind-CodeLlama-34B-v2": {"token_limits": (16384, 4096)},
                },
            },
        }
        cls._list: list[ModelInformation] = []
        for provider, info_1 in data.items():
            for completion_type, info_2 in info_1.items():
                for name, info_3 in info_2.items():
                    input_token_limit, output_token_limit = info_3["token_limits"]
                    model_information = ModelInformation(
                        name=name,
                        provider=provider,
                        completion_type=completion_type,
                        token_limits={"input": input_token_limit, "output": output_token_limit},
                    )
                    cls._list.append(model_information)

    @classmethod
    def exists(cls, **kwargs) -> bool:
        return _find_one(cls._list, **kwargs) is not None

    @classmethod
    def find_one(cls, **kwargs) -> ModelInformation | None:
        return _find_one(cls._list, **kwargs)

    @classmethod
    def find_one_or_raise(cls, **kwargs) -> ModelInformation:
        result = _find_one(cls._list, **kwargs)
        if result is None:
            raise ValueError(f"ModelInformation not found for: {kwargs}.")
        return result

    @classmethod
    def all(cls) -> list[ModelInformation]:
        return cls.find_all()

    @classmethod
    def find_all(cls, **kwargs) -> list[ModelInformation]:
        return _find_all(cls._list, **kwargs)


ModelsInformation.initialize()


def add_to_openai_usage(
    api_key: str,
    model_name: str,
    namespace: str,  # use ":" separated namespace
    counts: dict[str, int],
) -> None:
    for key, value in counts.items():
        if key.endswith("_details"):
            continue
        assert isinstance(value, int)

    openai_usage_file_path = os.path.expanduser(os.path.join("~", ".openai_usage.json"))
    if not os.path.exists(openai_usage_file_path):
        file_locked(write_json, data={}, file_path=openai_usage_file_path, silent=True)

    api_key_id = api_key[-5:]
    usage_data = file_locked(read_json, file_path=openai_usage_file_path)
    usage_key = "::".join([api_key_id, model_name, namespace])  # namespace must be at the end.
    if usage_key not in usage_data:
        usage_data[usage_key] = Counter()
    for key, value in counts.items():
        if key.endswith("_details"):
            continue
        if key not in usage_data[usage_key]:
            usage_data[usage_key][key] = 0
        usage_data[usage_key][key] += value
    file_locked(write_json, data=usage_data, file_path=openai_usage_file_path, silent=True)


def get_openai_usage(
    api_key: str,
    model_name: str,
    namespace: str,  # use ":" separated namespace (any item with this prefix will count)
) -> dict[str, int]:
    openai_usage_file_path = os.path.expanduser(os.path.join("~", ".openai_usage.json"))
    if not os.path.exists(openai_usage_file_path):
        file_locked(write_json, data={}, file_path=openai_usage_file_path, silent=True)

    api_key_id = api_key[-5:]
    usage_data = read_json(openai_usage_file_path)
    usage_key_prefix = "__".join([api_key_id, model_name, namespace])

    main_counter: dict[str, int] = Counter()
    for usage_key, individual_counter in usage_data.items():
        assert isinstance(individual_counter, dict)
        if not usage_key.startswith(usage_key_prefix):
            continue
        for key, count in individual_counter.items():
            assert isinstance(count, int)
            main_counter[key] += count

    return dict(main_counter)


def get_openai_num_tokens(model_name: str, text_or_messages: str | list[dict[str, str]]) -> int:
    addition_num_tokens = 0
    if model_name.startswith("ft:"):
        model_name = model_name.split(":")[1]
    if (
        model_name == "o1-mini-2024-09-12"
    ):  # not available yet: https://github.com/openai/tiktoken/issues/337
        model_name = "gpt-4o-2024-05-13"
    model_information = ModelsInformation.find_one_or_raise(name=model_name)
    completion_type = model_information.completion_type
    if completion_type == "plain":
        assert isinstance(text_or_messages, str)
        text = text_or_messages
    elif completion_type == "chat":
        assert isinstance(text_or_messages, list | str)
        if isinstance(text_or_messages, str):
            text_or_messages = [{"content": text_or_messages}]
        text = "\n\n".join([message["content"] or "" for message in text_or_messages])
        addition_num_tokens += len(text_or_messages) * 6
        # 5 is a bit of an overestimate, see:
        # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    else:
        raise ValueError(f"Unknown completion_type: {completion_type}.")
    if model_information.provider != "openai":
        model_name = "gpt-3.5-turbo-16k-0613"  # use a default model for tokenization for non-openai models for now.
    tokenizer = tiktoken.encoding_for_model(model_name)
    token_ids = tokenizer.encode(text)
    total_num_tokens = len(token_ids) + addition_num_tokens
    return total_num_tokens


def model_to_completion_function(model_name: str) -> callable:
    if model_name.startswith("ft:"):
        model_name = model_name.split(":")[1]
    model_information = ModelsInformation.find_one_or_raise(name=model_name)
    if model_information.provider == "openai":
        if model_information.completion_type == "plain":
            return openai.completions.create
        elif model_information.completion_type == "chat":
            return openai.chat.completions.create
        else:
            raise ValueError(f"Unknown completion_type: {model_information.completion_type}.")
    elif model_information.provider == "litellm":
        return litellm.completion
    else:
        raise ValueError(f"Unknown provider: {model_information.provider}.")


@openai_embedding_cache.memoize()
def get_embeddings(
    texts: list[str], model_name: str = "text-embedding-ada-002"
) -> list[np.ndarray]:
    response = openai.embeddings.create(model=model_name, input=texts)
    return [np.array(embedding["embedding"]) for embedding in response.data]


@openai_plain_completion_cache.memoize()
def cached_openai_plain_completion_call(  # kwargs doesn't work with caching.
    prompt: str, model: str, temperature: float, max_tokens: int, stop: tuple[str], seed: int
) -> Completion:
    function = model_to_completion_function(model)
    return function.create(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
        seed=seed,
    )


def openai_plain_completion_call(
    prompt: str, model: str, temperature: float, max_tokens: int, stop: tuple[str], seed: int
) -> Completion:
    function_ = model_to_completion_function(model)
    function = cached_openai_plain_completion_call if temperature == 0 else function_
    return function(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
        seed=seed,
    )


@openai_chat_completion_cache.memoize()
def cached_openai_chat_completion_call(  # kwargs doesn't work with caching.
    model: str,
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    tool_choice: str,
    temperature: float,
    max_tokens: int,
    stop: tuple[str],
    seed: int,
) -> Completion:
    function = model_to_completion_function(model)
    return function(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
        seed=seed,
    )


def openai_chat_completion_call(
    model: str,
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    tool_choice: str,
    temperature: float,
    max_tokens: int,
    stop: tuple[str],
    seed: int,
) -> Completion:
    function_ = model_to_completion_function(model)
    function = cached_openai_chat_completion_call if temperature == 0 else function_
    kwargs = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": stop,
        "seed": seed,
    }
    if "o1-" in model:
        assert not tools, f"detected o1 model ({model}). It does not support `tools` argument."
        assert (
            not tool_choice
        ), f"detected o1 model ({model}). It does not support `tool_choice` argument."
        assert not stop, f"detected o1 model ({model}). It does not support `stop` argument."
        kwargs.pop("tools")
        kwargs.pop("tool_choice")
        kwargs.pop("stop")
        assert (
            temperature == 1
        ), f"detected o1 model ({model}). It only supports temperate of 1, found {temperature}."
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
    return function(**kwargs)


@LanguageModel.register("openai_lm")
class OpenAILanguageModel(LanguageModel):
    def __init__(
        self,
        model: str,
        temperature: float = 0,
        max_tokens: int | None = None,
        stop: tuple[str] | None = ("<|endoftext|>",),
        seed: int = 100,
        retry_after_n_seconds: int | None = None,
        usage_namespace: str = "global",
    ) -> None:
        assert stop is None or isinstance(stop, tuple | list)  # make sure it's not str.
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stop = stop
        self.seed = seed
        self.retry_after_n_seconds = retry_after_n_seconds
        self.usage_namespace = usage_namespace
        self.model = model
        openai.organization = os.getenv("OPENAI_ORGANIZATION_ID", None)
        if model.startswith("ft:"):
            model = model.split(":")[1]
        model_information = ModelsInformation.find_one_or_raise(name=model)
        self.model_input_tokens_limit = model_information.token_limits["input"]
        self.model_output_tokens_limit = model_information.token_limits["output"]
        self.completion_type = model_information.completion_type
        self.openai_cache = {
            "plain": openai_plain_completion_cache,
            "chat": openai_chat_completion_cache,
        }[self.completion_type]

    def get_num_tokens(self, input: str | list[dict[str, str]]) -> int:
        return get_openai_num_tokens(self.model, input)

    def generate(
        self,
        input: str | list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> str | tuple[str, list[dict[str, Any]]] | None:
        max_tokens = self.max_tokens
        if max_tokens is None:
            used_num_tokens = self.get_num_tokens(input)
            max_tokens = self.model_input_tokens_limit - used_num_tokens
            if self.model_output_tokens_limit is not None:
                max_tokens = min(self.model_output_tokens_limit, max_tokens)

        if max_tokens <= 0:
            print(
                "WARNING: Ran out of context limit of this model. "
                f"Model: {self.model}, used_num_tokens: {used_num_tokens}, "
                f"model_input_tokens_limit: {self.model_input_tokens_limit}"
            )
            return ""

        if self.completion_type == "plain":
            assert isinstance(input, str)
            prompt = input.rstrip()
        else:
            messages = input
            if "o1-" in self.model:  # o1 does not support system message
                messages = [message for message in messages if message["role"] != "system"]
            assert isinstance(input, list)

        self.openai_cache.stats(enable=True, reset=True)
        success = False
        for _ in range(500):
            try:
                if self.completion_type == "plain":
                    assert tools is None, "tools are not supported for plain completions."
                    response = openai_plain_completion_call(
                        model=self.model,
                        prompt=prompt,  # type: ignore
                        temperature=self.temperature,
                        max_tokens=max_tokens,
                        stop=self.stop,
                        seed=self.seed,
                    )
                else:
                    response = openai_chat_completion_call(
                        model=self.model,
                        messages=messages,  # type: ignore
                        tools=tools,
                        tool_choice=tool_choice,
                        max_tokens=max_tokens,
                        temperature=self.temperature,
                        stop=self.stop,
                        seed=self.seed,
                    )
                success = True
                break
            except RETRY_ERROR as exception:
                success = False
                if self.retry_after_n_seconds is None:
                    import traceback

                    print(traceback.format_exc())
                    exit()
                if tools is not None and "invalid_request_error" in str(exception):
                    print(f"Invalid request error: {exception}. Continuing.")
                    return "", {"content": None, "tool_calls": []}
                if tools is not None and "model's maximum context length is 8192" in str(exception):
                    print(f"Model's maximum context length is 8192: {exception}. Continuing.")
                    return "", {"content": None, "tool_calls": []}
                if tools is not None and "Expected a string that matches the pattern" in str(
                    exception
                ):
                    print(f"Unexpected string error: {exception}. Continuing.")
                    # camera-ready hack; func calling is unstable rarely.
                    return "", {"content": None, "tool_calls": []}
                if tools is not None and "The server had an error processing your request" in str(
                    exception
                ):
                    # camera-ready hack; func calling is unstable rarely.
                    print(f"Server had an error: {exception}. Continuing.")
                    return "", {"content": None, "tool_calls": []}
                if "Input validation error" in str(exception):
                    # temporary rebuttal hack. It happens rarely for open models, input is too long.
                    print(f"Input too long: {exception}. Continuing.")
                    success = True
                    response = {
                        "choices": [{"message": {"content": ""}}],
                        "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
                    }
                    break
                print(f"Encountered OpenAI Error: {exception.message[:200].strip()}...")
                print(f"Will try again in {self.retry_after_n_seconds} seconds.")
                time.sleep(self.retry_after_n_seconds)
                pass

        if not success:
            raise Exception("Could not complete OpenAI call")

        choices = getter_plus(response, "choices")
        assert len(getter_plus(response, "choices")) == 1
        if self.completion_type == "plain":
            generation = getter_plus(choices[0], "text")
        else:
            generation = getter_plus(choices[0], "message.content")
        used_openai_api = self.temperature != 0 or self.openai_cache.stats(reset=False)[1] > 0
        if used_openai_api:
            usage_data = getter_plus(response, "usage") if "usage" in response else {}
            if hasattr(usage_data, "dict"):
                usage_data = usage_data.dict()
            try:
                add_to_openai_usage(
                    api_key=os.environ["OPENAI_API_KEY"],
                    model_name=self.model,
                    namespace=self.usage_namespace,
                    counts=usage_data,
                )
            except json.JSONDecodeError:  # Temporary for race condition.
                pass
            usage_data = dict(response.usage) if hasattr(response, "usage") else {}
            add_to_openai_usage(
                api_key=os.environ["OPENAI_API_KEY"],
                model_name=self.model,
                namespace=self.usage_namespace,
                counts=usage_data,
            )

        if tools is not None:  # temporary handling. Have proper output type later.
            return generation, choices[0].message.dict()

        return generation
