"""Thin entry. Orchestration lives in shared.lib.loop."""

from __future__ import annotations

from ageval_sdk import RunContext, RunTerminal
from shared.lib.loop import run as _run


async def run(ctx: RunContext) -> RunTerminal:
    return await _run(ctx)
