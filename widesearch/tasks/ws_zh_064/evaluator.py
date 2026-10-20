"""Thin entry. Scoring lives in shared.lib.scoring."""

from __future__ import annotations

from typing import Any

from shared.lib.scoring import evaluate as _evaluate


async def evaluate(inputs: dict[str, Any]) -> dict[str, Any]:
    return await _evaluate(inputs)
