"""JSON schema (strict) structured output check via OpenAI response_format."""

from __future__ import annotations

import json
import re

from ....core.models import DetectorResult
from .base import ActiveDetector
from .utils import finish_reason, message_text


_NONCE = "gemini-detector"


class StructuredOutputDetector(ActiveDetector):
    name = "structured_output"
    display_name = "Structured Output"
    weight = 15.0

    async def run(self, client, model: str) -> DetectorResult:
        try:
            # 384 leaves enough room for both reasoning_tokens (Gemini 3 burns
            # 100-150 of these on schema-conditioned prompts) and the JSON
            # output itself. With 128 we'd see truncated content and a false
            # negative on relays that DO honor response_format.
            _req, resp, _h, _lat = await client.chat_completions_create(
                model=model,
                max_completion_tokens=384,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f'Return JSON matching the schema with ok=true and nonce="{_NONCE}".'
                        ),
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "detector_result",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "ok": {"type": "boolean"},
                                "nonce": {"type": "string"},
                            },
                            "required": ["ok", "nonce"],
                            "additionalProperties": False,
                        },
                    },
                },
            )
        except Exception as e:  # noqa: BLE001
            return self._result("error", 0.0, error=str(e))

        text = message_text(resp)
        parsed = None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        ok_json = isinstance(parsed, dict)
        ok_schema = (
            isinstance(parsed, dict)
            and parsed.get("ok") is True
            and parsed.get("nonce") == _NONCE
        )
        finish = finish_reason(resp)
        markdown_seen = _looks_like_markdown_json(text)

        score = 0.0
        if ok_json:
            score += 40.0
        if ok_schema:
            score += 50.0
        if finish in ("stop", None):
            score += 10.0

        if ok_schema:
            evaluation_zh = "Structured output OK: the returned content is pure JSON and conforms to the schema."
        elif markdown_seen:
            evaluation_zh = (
                "The request sent response_format=json_schema strict=true, "
                "but a Markdown code block was returned. The relay may not be forwarding or may not implement OpenAI structured output parameters."
            )
        else:
            evaluation_zh = (
                "The request asked for JSON schema strict output, but the returned content cannot be parsed against the schema."
            )

        return self._result(
            "pass" if score >= 70.0 else "fail",
            score,
            {
                "response_text": text[:300],
                "parsed": parsed if isinstance(parsed, dict) else None,
                "json_parse": ok_json,
                "schema_match": ok_schema,
                "markdown_json_seen": markdown_seen,
                "finish_reason": finish,
                "evaluation_zh": evaluation_zh,
            },
        )


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*\{", re.IGNORECASE)


def _looks_like_markdown_json(text: str) -> bool:
    return bool(_FENCED_JSON_RE.search(text))
