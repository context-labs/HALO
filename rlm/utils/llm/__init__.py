"""Shared LLM abstraction: route to LiteLLM proxy or local endpoints."""

from openai.types.chat import ChatCompletionMessageParam as Message

from ._complete import CompletionResult, complete
from ._providers import list_models, print_models

__all__ = [
    "complete",
    "CompletionResult",
    "Message",
    "list_models",
    "print_models",
]
