"""OpenAI Chat Completions protocol implementation."""

from __future__ import annotations

from pathlib import Path

from ...core.detectors_base import BaseDetector
from ...core.models import DetectionTier, ExecutionConfig, Mode, Protocol
from .client import OpenAIChatClient
from .config import OPENAI_MODEL_CHOICES
from .detectors import build_all
from .runner import Runner

PROTOCOL_NAME = Protocol.OPENAI
TIER = DetectionTier.BEHAVIORAL


def model_choices() -> list[str]:
    return list(OPENAI_MODEL_CHOICES)


def default_model() -> str:
    return OPENAI_MODEL_CHOICES[0]


# Stable, widely-supported aliases preferred when /v1/models gives us a
# choice. Cheap-and-mainstream first (gpt-4o-mini), then incrementally larger
# / older fallbacks. Used by pick_default_model() to pre-fill the form's
# model field after a probe rather than landing on whatever happens to sort
# first in the relay's response.
_PREFERRED_DEFAULTS = (
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-5-mini",
    "gpt-5",
    "gpt-3.5-turbo",
)


def pick_default_model(available: list[str]) -> str | None:
    """Pick the most likely-to-just-work model from a relay's whitelist.

    Tries exact alias match first, then snapshot-suffix match (e.g. our
    preferred 'gpt-4o-mini' should pick 'gpt-4o-mini-2024-07-18' when the
    relay only carries snapshots). Falls back to the first available model
    when no preference matches — better to pre-fill *something* than leave
    the form on a default we already know the relay doesn't carry.
    """
    if not available:
        return None
    for pref in _PREFERRED_DEFAULTS:
        for m in available:
            if m == pref or m.startswith(pref + "-"):
                return m
    return available[0]


def build_config(mode: Mode, max_concurrent: int = 3) -> ExecutionConfig:
    return ExecutionConfig.for_mode(mode, max_concurrent=max_concurrent)


def build_detectors(mode: Mode | None = None) -> list[BaseDetector]:
    _ = mode
    return build_all()


def make_client(base_url: str, api_key: str, timeout: float) -> OpenAIChatClient:
    return OpenAIChatClient(base_url, api_key, timeout=timeout)


def build_runner(
    client: OpenAIChatClient,
    detectors: list[BaseDetector],
    config: ExecutionConfig,
) -> Runner:
    return Runner(client, detectors, config)


def baseline_path(model_id: str, mode: Mode) -> Path | None:
    candidates = [
        Path("data/baselines/openai") / f"{model_id}_{mode.value}.json",
        Path("data/baselines/openai") / f"{model_id}_full.json",
        Path("data/baselines/openai") / f"{model_id}_chat_text.json",
    ]
    return next((path for path in candidates if path.is_file()), None)


def verdict_caption(score: float) -> str:
    if score >= 85:
        return "Good Protocol Performance"
    if score >= 70:
        return "Largely Passed"
    if score >= 50:
        return "At Risk"
    return "Failed"


def tier_banner() -> tuple[str, str]:
    return (
        "Behavioral/Protocol-level verification",
        (
            "This detection cannot reliably distinguish a genuine high-end model "
            "from a low-end model impersonating one. What we test is whether the "
            "relay interface conforms to the OpenAI Chat Completions protocol "
            "spec, whether capabilities are complete, and whether the usage "
            "field matches the official response shape."
        ),
    )
