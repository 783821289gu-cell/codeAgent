import json

from agents.agent_output import AgentOutputSchema

from repopilot.core.models import ReviewReport
from repopilot.infrastructure.llm import _structured_output_instructions


def test_compatible_model_inlines_structured_output_schema() -> None:
    schema = AgentOutputSchema(ReviewReport)

    instructions = _structured_output_instructions("Review the change.", schema)

    assert instructions is not None
    assert instructions.startswith("Review the change.")
    assert "Do not use Markdown fences" in instructions
    embedded = instructions.split("JSON Schema: ", 1)[1]
    assert json.loads(embedded)["title"] == "ReviewReport"


def test_compatible_model_leaves_plain_text_instructions_unchanged() -> None:
    assert _structured_output_instructions("Act normally.", None) == "Act normally."
