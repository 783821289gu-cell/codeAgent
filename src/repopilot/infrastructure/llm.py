from __future__ import annotations

import json
from collections.abc import AsyncIterator

from agents import OpenAIChatCompletionsModel
from agents.agent_output import AgentOutputSchemaBase
from agents.handoffs import Handoff
from agents.items import ModelResponse, TResponseInputItem, TResponseStreamEvent
from agents.model_settings import ModelSettings
from agents.models.interface import Model, ModelTracing
from agents.tool import Tool
from openai import AsyncOpenAI
from openai.types.responses.response_prompt_param import ResponsePromptParam


class OpenAICompatibleChatModel(Model):
    """Chat Completions adapter for providers without JSON-schema response formats."""

    def __init__(self, model: str, client: AsyncOpenAI) -> None:
        self.delegate = OpenAIChatCompletionsModel(
            model=model,
            openai_client=client,
        )

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> ModelResponse:
        instructions = _structured_output_instructions(system_instructions, output_schema)
        return await self.delegate.get_response(
            instructions,
            input,
            model_settings,
            tools,
            None,
            handoffs,
            tracing,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            prompt=prompt,
        )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        instructions = _structured_output_instructions(system_instructions, output_schema)
        async for event in self.delegate.stream_response(
            instructions,
            input,
            model_settings,
            tools,
            None,
            handoffs,
            tracing,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            prompt=prompt,
        ):
            yield event

    async def close(self) -> None:
        await self.delegate.close()


def _structured_output_instructions(
    instructions: str | None,
    output_schema: AgentOutputSchemaBase | None,
) -> str | None:
    if output_schema is None or output_schema.is_plain_text():
        return instructions
    schema = json.dumps(output_schema.json_schema(), ensure_ascii=False, separators=(",", ":"))
    suffix = (
        "\n\nReturn the final answer as one valid JSON object matching this JSON Schema. "
        "Do not use Markdown fences or add explanatory text outside the JSON object.\n"
        f"JSON Schema: {schema}"
    )
    return (instructions or "") + suffix
