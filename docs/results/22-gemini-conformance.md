# Gemini live provider conformance (#22)

On 2026-11-17, the existing conformance suite passed against the Gemini
Developer API using a key configured for a Free Tier project:

| Setting | Value |
|---|---|
| Model | `gemini-3.5-flash-lite` |
| SDK | `google-genai==3.4.0` |
| Python / pytest | 3.13.2 / 9.2.0 |
| Base commit | `09e52bd` (upstream `main`), plus this change |
| Result | **3 passed, 1 skipped, 8 deselected** in 4.34 s |

The three passes cover catalog schemas accepted and a valid `ActionCall`
returned, nonzero input/output usage, and a tool-result round trip into a second
model response. The cache test is skipped by design for Google: a cache hit on
the next request is not guaranteed. The eight deselected cases are for Anthropic
and OpenAI. The final suite made four `generate_content` calls.

This checks the provider protocol, not HOI4 gameplay. It does not measure cache
hit rates, compare an invoice, or prove support for other Gemini models.

## Reproduce

Install `pip install -e ".[dev,google]"` and set `GOOGLE_API_KEY` outside source
control. For a free-only run, use a Free Tier project and check the model's
[current pricing](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.5-flash-lite)
before running. A model with a free tier can still be billed on a paid project.

```bash
HOI4_LIVE_TESTS=1 HOI4_LIVE_GOOGLE_MODEL=gemini-3.5-flash-lite \
  pytest tests/live/ -k google -v -rs
```

No API key or real save is included in these results. The weekly conformance
workflow selects this model for Google and retains its existing keyless skips.
No repository secret or workflow run was configured as part of this local test.

## Fixes needed for the live run

- Send catalog schemas through `parameters_json_schema`. Sending them through
  `parameters` produced HTTP 400 because the SDK translated
  `additionalProperties` into an unsupported `additional_properties` field.
- Preserve Gemini's ordered response parts, including `thought_signature`, in
  `provider_blocks`. Rebuilding an assistant message from text and tool calls
  lost the signature and produced HTTP 400 on the second model call. JSON-mode
  serialization keeps the signature bytes recoverable in transcripts.
- Make the live round-trip test carry `provider_blocks`, as the agent loop
  already does. The test now exercises the same signed-message path.

Regression tests failed before each fix. The SDK floor is raised to 1.21.0,
which introduced JSON Schema function declarations. The schema and signed-part
round trips were also checked offline with SDK 1.21.0, which supports the
project's Python 3.10 requirement; the live result above uses SDK 3.4.0.

## Offline validation

- Full suite on Python 3.13.2 / SDK 3.4.0: **753 passed, 13 skipped**. Live
  tests were not opted in for this run.
- Provider conversion, token accounting and signed-part tests with SDK 1.21.0:
  **36 passed**. Its SDK configuration also accepted all catalog declarations.
- `ruff check src tests` and `git diff --check` pass.
