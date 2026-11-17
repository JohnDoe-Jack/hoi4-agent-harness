"""Google Gemini provider (google-genai SDK).

Uses google-genai >= 1.21.0 for JSON Schema function declarations and signed
response parts. Live conformance passed with gemini-3.5-flash-lite and SDK
3.4.0; see docs/results/22-gemini-conformance.md.
"""

from __future__ import annotations

import os

from ...actions.registry import ToolSpec
from .base import LLMClient, LLMResponse, Msg, ToolCall, Usage


def wire_declarations(tools: list[ToolSpec] | None) -> list[dict]:
    """Use Gemini's JSON Schema field, preserving additionalProperties.

    ``parameters`` is the smaller OpenAPI Schema dialect; the SDK converts
    additionalProperties there into a field the Gemini API rejects.
    """
    return [
        {"name": t.name, "description": t.description, "parameters_json_schema": t.parameters}
        for t in (tools or [])
    ]


def wire_contents(messages: list[Msg]) -> list[dict]:
    """Neutral messages -> Gemini `contents`.

    Gemini uses the roles "user" and "model", and carries tool results as
    functionResponse parts on a user turn.
    """
    contents: list[dict] = []
    for msg in messages:
        if msg.role == "assistant":
            # Signed parts must keep their original order and contents. They
            # already contain the text and calls; rebuilding them loses the
            # thought_signature required by Gemini's next tool round.
            parts: list[dict] = [dict(part) for part in msg.provider_blocks]
            if not parts:
                if msg.text:
                    parts.append({"text": msg.text})
                for call in msg.tool_calls:
                    parts.append({"function_call": {"name": call.name, "args": call.arguments}})
            if parts:
                contents.append({"role": "model", "parts": parts})
        else:
            parts = [
                {
                    "function_response": {
                        "name": result.name or result.call_id,
                        "response": {"result": result.content},
                    }
                }
                for result in msg.tool_results
            ]
            if msg.text:
                parts.append({"text": msg.text})
            if parts:
                contents.append({"role": "user", "parts": parts})
    return contents



def usage_from_wire(meta) -> Usage:
    """Gemini usage metadata -> Usage.

    ``prompt_token_count`` includes cached content. ``candidates_token_count``
    does *not* include thinking: that arrives as ``thoughts_token_count`` and is
    billed as output, so a thinking model's output was undercounted by exactly
    its reasoning until this summed the two.
    """
    if meta is None:
        return Usage()
    return Usage(
        input_tokens=getattr(meta, "prompt_token_count", 0) or 0,
        output_tokens=(getattr(meta, "candidates_token_count", 0) or 0)
        + (getattr(meta, "thoughts_token_count", 0) or 0),
        cached_input_tokens=getattr(meta, "cached_content_token_count", 0) or 0,
    )

class GoogleClient(LLMClient):
    name = "google"

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        max_tokens: int = 16000,
        effort: str = "high",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        cache_prefix: bool = True,
    ):
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("pip install 'hoi4-agent-harness[google]'") from exc

        self.genai = genai
        self.client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _to_contents(self, messages: list[Msg]) -> list[dict]:
        return wire_contents(messages)

    def complete(self, system, messages, tools=None, max_tokens=None) -> LLMResponse:
        declarations = wire_declarations(tools)
        config: dict = {
            "system_instruction": system,
            "max_output_tokens": max_tokens or self.max_tokens,
        }
        if declarations:
            config["tools"] = [{"function_declarations": declarations}]
        if self.temperature is not None:
            config["temperature"] = self.temperature

        response = self.client.models.generate_content(
            model=self.model, contents=self._to_contents(messages), config=config
        )

        calls = [
            ToolCall(id=f"{fc.name}-{i}", name=fc.name, arguments=dict(fc.args or {}))
            for i, fc in enumerate(getattr(response, "function_calls", None) or [])
        ]
        candidates = getattr(response, "candidates", None) or []
        content = getattr(candidates[0], "content", None) if candidates else None
        # JSON mode encodes signature bytes for transcript storage; the SDK
        # decodes them when these same parts are supplied on the next call.
        parts = [
            part.model_dump(mode="json", exclude_none=True)
            for part in (getattr(content, "parts", None) or [])
        ]
        usage = usage_from_wire(getattr(response, "usage_metadata", None))
        return LLMResponse(
            text=getattr(response, "text", "") or "",
            tool_calls=calls,
            stop_reason="tool_use" if calls else "end_turn",
            usage=usage,
            model=self.model,
            raw=response,
            provider_blocks=parts,
        )
