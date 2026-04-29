from typing import Any

from agents.extensions.models.litellm_model import LitellmModel
from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncOpenAI


class LanguageModel:
    def __init__(
        self,
        type: str,
        name: str,
        settings: dict[str, Any] | None = None,
        extras: dict[str, Any] | None = None,
    ):
        self.settings = ModelSettings(**(settings or {}))
        self._model: OpenAIChatCompletionsModel | LitellmModel
        if type == "openai":
            self._model = OpenAIChatCompletionsModel(
                model=name,
                openai_client=AsyncOpenAI(**(extras or {})),
            )
        elif type == "litellm":
            self._model = LitellmModel(model=name, **(extras or {}))
        else:
            raise ValueError(f"Unsupported model type: {type}")

    async def generate(self, input: Any) -> dict[str, Any]:
        if isinstance(input, list):
            input_ = []
            system_instructions: str | None = None
            for item in input:
                if isinstance(item, dict) and item.get("role", None) == "system":
                    system_instructions = item.get("content", "")
                else:
                    input_.append(item)
            input = input_
        response = await self._model.get_response(
            system_instructions=system_instructions,
            input=input,
            prompt=None,
            model_settings=self.settings,
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
            previous_response_id=None,
        )
        if not response:
            return {"content": ""}
        output = {"content": response.output[0].content[0].text}
        return output
