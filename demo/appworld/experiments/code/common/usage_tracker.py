from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Self, cast

from typing_extensions import TypedDict

from appworld.common.io import read_json, write_json
from appworld.common.types import Number


CELL_WIDTH = 28


class CorruptedUsageError(Exception):
    pass


class CostPerTokenDict(TypedDict):
    input_cache_miss: float
    input_cache_hit: float
    input_cache_write: float | None
    output: float


@dataclass
class TokensOrCost:
    input_cache_miss: int | float = 0  # input tokens processed that were not in cache
    input_cache_hit: int | float = 0  # input tokens processed that were in cache
    input_cache_write: int | float = 0  # input tokens written to cache
    output: int | float = 0  # output tokens generated

    def __post_init__(self) -> None:
        for variable_name, variable_value in zip(
            ["input_cache_miss", "input_cache_hit", "input_cache_write", "output"],
            [
                self.input_cache_miss,
                self.input_cache_hit,
                self.input_cache_write,
                self.output,
            ],
            strict=False,
        ):
            if not isinstance(variable_value, int | float):
                raise CorruptedUsageError(
                    f"Invalid type for {variable_name}: {type(variable_value)}"
                )
            if variable_value < 0:
                raise CorruptedUsageError(
                    f"Invalid value for {variable_name}: {variable_value}. Must be non-negative"
                )

    def __str__(self) -> str:
        items = [
            ("Input cache miss", self.input_cache_miss),
            ("Input cache hit", self.input_cache_hit),
            ("Input cache write", self.input_cache_write),
            ("Output", self.output),
            ("Total", self.total),
        ]
        parts = [
            f"{k}: {int(v)}".ljust(CELL_WIDTH)
            if isinstance(v, int) or (isinstance(v, float) and v.is_integer())
            else f"{k}: {v:.2e}".ljust(CELL_WIDTH)
            for k, v in items
        ]
        return " | ".join(parts)

    def dict(self) -> dict[str, Number]:
        def normalize(v: Number) -> Number:
            return (
                int(v)
                if isinstance(v, int) or (isinstance(v, float) and v.is_integer())
                else round(v, 10)
            )

        dict_: dict[str, Number] = {
            "input_cache_miss": normalize(self.input_cache_miss),
            "input_cache_hit": normalize(self.input_cache_hit),
            "input_cache_write": normalize(self.input_cache_write),
            "output": normalize(self.output),
        }
        return dict_

    def __add__(self, other: Self) -> Self:
        return type(self)(
            **{k: getattr(self, k) + getattr(other, k) for k in self.__dataclass_fields__}
        )

    @property
    def total(self) -> int | float:
        total_ = 0.0
        for key in self.__dataclass_fields__:
            total_ += getattr(self, key)
        total_ = round(total_, 10)
        if isinstance(total_, float) and total_.is_integer():
            return int(total_)
        return total_

    @property
    def input_cache_hit_or_miss(self) -> int | float:
        input_ = self.input_cache_miss + self.input_cache_hit
        input_ = round(input_, 10)
        if isinstance(input_, float) and input_.is_integer():
            return int(input_)
        return input_


@dataclass
class CostPerToken(TokensOrCost): ...


@dataclass
class Tokens(TokensOrCost):
    @classmethod
    def from_response(cls, response: dict[str, Any], usage_key: str = "usage") -> Self:
        usage = response.get(usage_key, {})

        def first_valid(keys: list[str], nested_key: str | None = None) -> int | float:
            for key in keys:
                if key in usage and isinstance(usage[key], int | float):
                    return cast(int | float, usage[key])
                if (
                    key in usage
                    and isinstance(usage[key], dict)
                    and nested_key
                    and nested_key in usage[key]
                    and isinstance(usage[key][nested_key], int | float)
                ):
                    return cast(int | float, usage[key][nested_key])
            return 0

        input_tokens = first_valid(["input_tokens", "prompt_tokens", "inputTokens", "promptTokens"])
        cached_input_tokens = first_valid(
            [
                "cached_tokens",
                "cachedTokens",
                "cache_read_input_tokens",
                "cacheReadInputTokens",
                "input_tokens_details",
                "prompt_tokens_details",
            ],
            nested_key="cached_tokens",
        )
        # Note: The clamping was needed below because of grok-code-fast-1. See real usage
        # example in tests/experiments/test_usage_tracker.py for more details.
        if input_tokens < cached_input_tokens:
            input_tokens = cached_input_tokens = input_tokens
        cache_write_input_tokens = first_valid(
            ["cache_creation_input_tokens", "cacheWriteInputTokens"]
        )
        output_tokens = first_valid(["output_tokens", "completion_tokens", "outputTokens"])
        # NOTE: Some model APIs (Grok in particular) does not include reasoning_tokens in the top-level
        # completion_tokens, but still charges for them. So we need to add them back in here. We do this decision on
        # whether to add or not based on the total_tokens. If total_tokens > prompt_tokens + completion_tokens,
        # we assume the difference is reasoning tokens and add them back in. If they still don't add up,
        # we raise an error.
        reasoning_tokens = first_valid(
            ["completion_tokens_details", "output_tokens_details"], nested_key="reasoning_tokens"
        )
        total_tokens = first_valid(["total_tokens", "TotalTokens"])
        assert isinstance(total_tokens, int | float)
        if total_tokens != input_tokens + output_tokens:
            output_tokens += reasoning_tokens
        stopped_due_to_content_filter = False
        for choice in response.get("choices", []):
            # Taken from litellm_core_utils/core_helpers.py:map_finish_reason
            content_filter_tags = [
                "ERROR_TOXIC",
                "ERROR",
                "SAFETY",
                "RECITATION",
                "content_filtered",
                "content_filter",
            ]
            if choice.get("finish_reason") in content_filter_tags:
                stopped_due_to_content_filter = True
                break
        if not output_tokens:  # this is usually triggered by content filtering
            stopped_due_to_content_filter = True
        if not stopped_due_to_content_filter and total_tokens != (input_tokens + output_tokens):
            raise CorruptedUsageError(
                "Could not reconcile total_tokens with input/output tokens. "
                f"input_tokens: {input_tokens}, output_tokens: {output_tokens}, "
                f"total_tokens: {total_tokens} != {input_tokens + output_tokens}. "
                f"Raw response: {response}."
            )
        input_cache_miss = input_tokens - cached_input_tokens
        return cls(
            input_cache_miss=input_cache_miss,
            input_cache_hit=cached_input_tokens,
            input_cache_write=cache_write_input_tokens,
            output=output_tokens,
        )


@dataclass
class Cost(TokensOrCost):
    @classmethod
    def build(cls, tokens: Tokens, cost_per_token: CostPerToken) -> Self:
        return cls(
            input_cache_miss=round(tokens.input_cache_miss * cost_per_token.input_cache_miss, 10),
            input_cache_hit=round(tokens.input_cache_hit * cost_per_token.input_cache_hit, 10),
            input_cache_write=round(
                tokens.input_cache_write * cost_per_token.input_cache_write, 10
            ),
            output=round(tokens.output * cost_per_token.output, 10),
        )


@dataclass
class Usage:
    tokens: Tokens = field(default_factory=Tokens)
    cost_per_token: CostPerToken = field(default_factory=CostPerToken)
    cost: Cost = field(default_factory=Cost)

    def __str__(self) -> str:
        return "\n".join(
            [
                "Tokens (#) => " + str(self.tokens),
                "Cost   ($) => " + str(self.cost),
            ]
        )

    def __add__(self, other: "Usage") -> "Usage":
        return type(self)(
            tokens=self.tokens + other.tokens,
            cost_per_token=self.cost_per_token,  # usually unchanged
            cost=self.cost + other.cost,
        )

    @classmethod
    def build(
        cls, response: dict[str, Any], cost_per_token: CostPerToken, usage_key: str = "usage"
    ) -> Self:
        tokens = Tokens.from_response(response, usage_key=usage_key)
        cost = Cost.build(tokens=tokens, cost_per_token=cost_per_token)
        return cls(tokens=tokens, cost_per_token=cost_per_token, cost=cost)

    def text(self) -> str:
        return str(self)

    @classmethod
    def load(cls, file_path: str) -> Self:
        dict_ = read_json(file_path)
        if not dict_:
            return cls()
        tokens = Tokens(**dict_.get("tokens", {}))
        cost_per_token = CostPerToken(**dict_.get("cost_per_token", {}))
        cost = Cost(**dict_.get("cost", {}))
        return cls(tokens=tokens, cost_per_token=cost_per_token, cost=cost)

    def dict(self) -> dict[str, dict[str, Number]]:
        dict_: dict[str, dict[str, Number]] = {
            "tokens": self.tokens.dict(),
            "cost_per_token": self.cost_per_token.dict(),
            "cost": self.cost.dict(),
        }
        return dict_


class UsageTracker:
    def __init__(
        self,
        cost_per_token: CostPerToken | None = None,
        max_cost_overall: float = float("inf"),
        max_cost_per_task: float = float("inf"),
        max_input_tokens_overall: Number = float("inf"),
        max_input_tokens_per_task: Number = float("inf"),
        max_output_tokens_overall: Number = float("inf"),
        max_output_tokens_per_task: Number = float("inf"),
    ) -> None:
        cost_per_token = cost_per_token or CostPerToken()
        self.cost_per_token = cost_per_token
        self.max_cost_overall = max_cost_overall
        self.max_cost_per_task = max_cost_per_task
        self.max_input_tokens_overall = max_input_tokens_overall
        self.max_input_tokens_per_task = max_input_tokens_per_task
        self.max_output_tokens_overall = max_output_tokens_overall
        self.max_output_tokens_per_task = max_output_tokens_per_task
        self.overall_cost = 0.0
        self.task_costs = defaultdict[str, float](float)
        self.overall_usage = Usage(cost_per_token=self.cost_per_token)
        self.task_usage = defaultdict[str, Usage](self._usage_factory)
        self.reset()

    @property
    def cost_per_task(self) -> float:
        return round(self.overall_cost / len(self.task_costs), 2) if self.task_costs else 0

    def _usage_factory(self) -> Usage:
        return Usage(cost_per_token=self.cost_per_token)

    def reset(self, task_id: str | None = None) -> None:
        if task_id is not None:
            self.task_usage[task_id] = Usage(cost_per_token=self.cost_per_token)
            self.task_costs[task_id] = 0.0
            return

        self.overall_cost = 0.0
        self.task_costs = defaultdict[str, float](float)
        self.overall_usage = Usage(cost_per_token=self.cost_per_token)
        self.task_usage = defaultdict[str, Usage](self._usage_factory)

    def add(self, task_id: str, usage: Usage) -> None:
        self.overall_usage = self.overall_usage + usage
        if task_id in self.task_usage:
            self.task_usage[task_id] = self.task_usage[task_id] + usage
        else:
            self.task_usage[task_id] = usage
        cost_value = usage.cost.total
        self.overall_cost += cost_value
        self.task_costs[task_id] += cost_value

    def exceeded(self, task_id: str | None = None) -> bool:
        exceeded_ = self.overall_cost >= self.max_cost_overall
        exceeded_ = exceeded_ or (
            self.overall_usage.tokens.input_cache_hit_or_miss >= self.max_input_tokens_overall
        )
        exceeded_ = exceeded_ or (
            self.overall_usage.tokens.output >= self.max_output_tokens_overall
        )
        if task_id:
            exceeded_ = exceeded_ or (self.task_costs[task_id] >= self.max_cost_per_task)
        return exceeded_

    def save(self, file_path: str, task_id: str | None = None) -> None:
        usage = self.overall_usage if task_id is None else self.task_usage[task_id]
        write_json(usage.dict(), file_path, silent=True)
