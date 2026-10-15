"""OpenAI JSON schema / structured output detector."""

from __future__ import annotations

import json
import re

from ....core.models import DetectorResult
from .base import ActiveDetector


class StructuredOutputDetector(ActiveDetector):
    name = "structured_output"
    display_name = "Structured Output"
    weight = 15.0

    async def run(self, client, model: str) -> DetectorResult:
        try:
            _req, resp, _h, _lat = await client.chat_completions_create(
                model=model,
                max_completion_tokens=128,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": 'Return JSON matching the schema with ok=true and nonce="openai-detector".',
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

        text = _message_text(resp)
        parsed = None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        ok_json = isinstance(parsed, dict)
        ok_schema = (
            isinstance(parsed, dict)
            and parsed.get("ok") is True
            and parsed.get("nonce") == "openai-detector"
        )
        finish = _finish_reason(resp)
        score = 0.0
        if ok_json:
            score += 40
        if ok_schema:
            score += 50
        if finish in ("stop", None):
            score += 10
        markdown_json_seen = _looks_like_markdown_json(text)
        evaluation_zh = (
        "Structured output OK: the returned content is pure JSON and the fields conform to the schema."
            if ok_schema
            else (
        "The request sent response_format=json_schema strict=true, but a plain Markdown text was "
        "returned — the relay may not be forwarding or may not implement OpenAI structured output parameters."
                if markdown_json_seen
                else (
        "The request sent response_format=json_schema strict=true, but the returned content cannot be parsed against the JSON schema."
                )
            )
        )
        return self._result(
            "pass" if score >= 70 else "fail",
            score,
            {
                "response_text": text[:300],
                "parsed": parsed,
                "json_parse": ok_json,
                "schema_match": ok_schema,
                "markdown_json_seen": markdown_json_seen,
                "evaluation_zh": evaluation_zh,
                "finish_reason": finish,
            },
        )


def _message_text(resp: dict) -> str:
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = msg.get("content") if isinstance(msg, dict) else ""
    return content if isinstance(content, str) else ""


def _finish_reason(resp: dict):
    choices = resp.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0].get("finish_reason")
    return None


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*\{", re.IGNORECASE)


def _looks_like_markdown_json(text: str) -> bool:
    return bool(_FENCED_JSON_RE.search(text))
