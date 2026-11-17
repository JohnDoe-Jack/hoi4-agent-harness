"""Keep Gemini's signed response parts intact across a tool round."""

import json
from types import SimpleNamespace

import pytest

from hoi4_harness.agent.llm.base import Msg, ToolResult
from hoi4_harness.agent.llm.google_provider import GoogleClient, wire_contents


def test_signed_parts_are_sent_back_in_their_original_order():
    parts = [
        {"text": "Checking research.", "thought_signature": "c2lnbmF0dXJl"},
        {"function_call": {"name": "start_research", "args": {"technology": "construction1"}}},
    ]
    messages = [Msg("assistant", text="Checking research.", provider_blocks=parts)]

    assert wire_contents(messages) == [{"role": "model", "parts": parts}]


def test_complete_preserves_json_serializable_parts_for_the_next_tool_round():
    types = pytest.importorskip("google.genai.types")
    parts = [
        types.Part(text="Checking research."),
        types.Part(
            function_call=types.FunctionCall(name="start_research", args={"technology": "construction1"}),
            thought_signature=b"\x00\xffsignature",
        ),
    ]
    sdk_response = types.GenerateContentResponse(candidates=[
        types.Candidate(content=types.Content(role="model", parts=parts))
    ])
    client = GoogleClient.__new__(GoogleClient)
    client.model, client.max_tokens, client.temperature = "test", 2048, None
    client.client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: sdk_response))

    response = client.complete("system", [Msg("user", text="Research construction1.")])
    # The transcript can store the blocks, and the loop can send them back.
    restored = json.loads(json.dumps(response.provider_blocks))
    messages = [
        Msg("assistant", text=response.text, tool_calls=response.tool_calls, provider_blocks=restored),
        Msg("user", tool_results=[ToolResult(response.tool_calls[0].id, "ok", name="start_research")]),
    ]
    wire = wire_contents(messages)
    recovered = types.Content.model_validate(wire[0])
    assert recovered.parts == parts
    assert len(wire[0]["parts"]) == 2  # No duplicate text or function calls.
