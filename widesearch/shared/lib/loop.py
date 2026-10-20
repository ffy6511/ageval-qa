"""Main agent plus sub-agents for one WideSearch question.

The step cap, the three-error stop, and the in-step tool batch follow
ByteDance-Seed/WideSearch ``Runner.run`` at 9825ba7. Sub-agents keep the
caller's list order. At most four run at once via ``bounded_gather``.
The parent ceiling is ``limits.agent_concurrency``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ageval_sdk import RunContext, RunTerminal, bounded_gather

from shared.lib.prompts import (
    get_multi_agent_system_prompt,
    get_system_prompt,
    get_tools_api_description,
)
from shared.lib.search import backend_name, search_global, text_browser_view

MAX_STEPS = 50
MAX_ERROR_COUNT = 3
SUB_AGENT_LIMIT = 4
_SESSION_TURNS = 64

_SUB_TOOLS = ["search_global", "text_browser_view"]
_LEAD_TOOLS = ["create_sub_agents", "search_global", "text_browser_view"]

_MAX_STEP_MESSAGE = (
    "[Max Step] The tool has been used too many times. "
    "Please stop invoking the tool immediately and answer the user's question."
)


@dataclass
class _SubAgent:
    index: int
    prompt: str


def _catalog(language: str, names: list[str]) -> list[dict[str, Any]]:
    return get_tools_api_description(language, names)


def _assistant_message(text: str, calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": [
            {
                "id": str(call.get("id") or ""),
                "type": "function",
                "function": {
                    "name": str(call.get("name") or ""),
                    "arguments": json.dumps(call.get("arguments") or {}, ensure_ascii=False),
                },
            }
            for call in calls
        ],
    }


def _tool_message(call_id: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


async def _domain_tool(name: str, arguments: dict[str, Any]) -> str:
    try:
        if name == "search_global":
            return await search_global(
                str(arguments.get("query") or ""),
                count=int(arguments.get("count") or 10),
                summary_type=str(arguments.get("summary_type") or "short"),
                use_english=bool(arguments.get("use_english") or False),
            )
        if name == "text_browser_view":
            return await text_browser_view(
                str(arguments.get("url") or ""),
                str(arguments.get("description") or ""),
            )
    except Exception as exc:
        return f"SYSTEM_ERROR\nreq={name}\n{type(exc).__name__}: {exc}"
    return f"Tool {name} not found"


async def _run_agent(
    ctx: RunContext,
    *,
    profile_id: str,
    system_prompt: str,
    tools: list[dict[str, Any]],
    user_input: str,
    language: str,
    on_sub_agents: Any = None,
) -> dict[str, Any]:
    """One official ``Runner.run`` loop on its own session.

    ``official_messages`` follows ``extract_messages_from_memory``: a failed
    model call is omitted, and a finished step is stored as an assistant
    object whose ``content`` is the model text.
    """
    chat: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    official: list[dict[str, Any]] = [{"role": "user", "content": user_input}]
    errors = 0
    last_error = ""
    steps = 0
    summary = False
    sub_calls: list[list[dict[str, Any]]] = []

    async with ctx.agent.session(profile_id, max_turns=_SESSION_TURNS) as session:
        while not summary and not _finished(official):
            if errors >= MAX_ERROR_COUNT:
                note = (
                    f"[{last_error}] Too many errors have occurred. "
                    "Please stop invoking the tool immediately and answer the user's question."
                )
                chat.append({"role": "user", "content": note})
                official.append({"role": "user", "content": note})
                summary = True
            if steps >= MAX_STEPS:
                chat.append({"role": "user", "content": _MAX_STEP_MESSAGE})
                official.append({"role": "user", "content": _MAX_STEP_MESSAGE})
                summary = True

            prompt = user_input
            for item in reversed(chat):
                if item.get("role") == "user" and isinstance(item.get("content"), str):
                    prompt = item["content"]
                    break
            try:
                reply = await session.invoke(prompt, tools=tools, messages=chat)
            except Exception as exc:
                errors += 1
                last_error = f"[Runner] Exception during LLM completion: {exc}"
                continue
            if not reply.get("ok"):
                errors += 1
                last_error = (
                    "[Runner] Exception during LLM completion: "
                    + str(reply.get("error") or "invoke_failed")
                )
                continue

            steps += 1
            text = str(reply.get("text") or "")
            calls = [call for call in (reply.get("tool_calls") or []) if isinstance(call, dict)]
            official.append(
                {
                    "role": "assistant",
                    "content": {"content": text},
                    "finished": not calls,
                }
            )
            if not calls:
                chat.append({"role": "assistant", "content": text})
                continue

            chat.append(_assistant_message(text, calls))
            results = await _run_tools(
                ctx,
                session,
                calls,
                language=language,
                on_sub_agents=on_sub_agents,
                sub_calls=sub_calls,
            )
            for call, content in zip(calls, results, strict=True):
                call_id = str(call.get("id") or "")
                name = str(call.get("name") or "")
                await session.record_observation(
                    call_id,
                    content=content,
                    function_name=name,
                    error=content.startswith("Tool ") or content.startswith("SYSTEM_ERROR"),
                )
                chat.append(_tool_message(call_id, content))
            if summary:
                break

    response, stop = _extract(official)
    return {
        "response": response,
        "steps": steps,
        "errors": errors,
        "stop": stop,
        "sub_calls": sub_calls,
    }


def _finished(official: list[dict[str, Any]]) -> bool:
    if not official:
        return False
    return bool(official[-1].get("finished"))


async def _run_tools(
    ctx: RunContext,
    session: Any,
    calls: list[dict[str, Any]],
    *,
    language: str,
    on_sub_agents: Any,
    sub_calls: list[list[dict[str, Any]]],
) -> list[str]:
    del session

    async def one(call: dict[str, Any]) -> str:
        name = str(call.get("name") or "")
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        if name == "create_sub_agents":
            if on_sub_agents is None:
                return "Tool create_sub_agents not found"
            content, trace = await on_sub_agents(arguments)
            sub_calls.append(trace)
            return content
        if name not in {"search_global", "text_browser_view"}:
            return f"Tool {name} not found"
        return await _domain_tool(name, arguments)

    gathered = await bounded_gather([one(call) for call in calls], limit=max(1, len(calls)))
    out: list[str] = []
    for item in gathered:
        if isinstance(item, BaseException):
            out.append(f"SYSTEM_ERROR\n{type(item).__name__}: {item}")
        else:
            out.append(item)
    return out


def _extract(official: list[dict[str, Any]]) -> tuple[str, str]:
    if not official:
        return "NULL", "empty"
    last = official[-1]
    content = last.get("content")
    if isinstance(content, dict) and "content" in content:
        text = content.get("content")
        return ("NULL" if text is None else str(text)), "finished"
    if isinstance(content, str):
        return content, "user_tail"
    return "NULL", "unreadable"


async def _sub_agent(ctx: RunContext, language: str, prompt: str) -> dict[str, Any]:
    outcome = await _run_agent(
        ctx,
        profile_id="researcher",
        system_prompt=get_system_prompt(language),
        tools=_catalog(language, _SUB_TOOLS),
        user_input=prompt,
        language=language,
    )
    # create_sub_agents reads the last action object's content. A trailing
    # user message (summary invoke failed) becomes the error string.
    if outcome["stop"] != "finished":
        outcome["response"] = "sub_agent running error."
        outcome["stop"] = "sub_agent running error"
    return outcome


async def _create_sub_agents(
    ctx: RunContext,
    language: str,
    arguments: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    raw = arguments.get("sub_agents")
    try:
        if not isinstance(raw, list):
            raise TypeError("sub_agents is not a list")
        infos = [_SubAgent(**item) for item in raw]
    except Exception as exc:
        payload = [{"error": f"create sub agents error: {exc}"}]
        return json.dumps(payload, ensure_ascii=False), []

    async def run_one(info: _SubAgent) -> dict[str, Any]:
        try:
            outcome = await _sub_agent(ctx, language, info.prompt)
        except Exception:
            outcome = {
                "response": "sub_agent running error.",
                "steps": 0,
                "errors": 0,
                "stop": "sub_agent running error",
            }
        return {
            "index": info.index,
            "prompt": info.prompt,
            "response": outcome["response"],
            "steps": outcome["steps"],
            "errors": outcome["errors"],
            "stop": outcome["stop"],
        }

    gathered = await bounded_gather(
        [run_one(info) for info in infos],
        limit=SUB_AGENT_LIMIT,
    )
    rows: list[dict[str, Any]] = []
    for info, item in zip(infos, gathered, strict=True):
        if isinstance(item, BaseException):
            rows.append(
                {
                    "index": info.index,
                    "prompt": info.prompt,
                    "response": "sub_agent running error.",
                    "steps": 0,
                    "errors": 0,
                    "stop": type(item).__name__,
                }
            )
        else:
            rows.append(item)
    visible = [
        {"index": row["index"], "prompt": row["prompt"], "response": row["response"]}
        for row in rows
    ]
    return json.dumps(visible, ensure_ascii=False), rows


async def run(ctx: RunContext) -> RunTerminal:
    language = str(ctx.params["language"])
    instance_id = str(ctx.params["instance_id"])
    query = (ctx.workspace_root / "query.txt").read_text(encoding="utf-8")

    async def on_sub_agents(arguments: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        return await _create_sub_agents(ctx, language, arguments)

    outcome = await _run_agent(
        ctx,
        profile_id="lead",
        system_prompt=get_multi_agent_system_prompt(language),
        tools=_catalog(language, _LEAD_TOOLS),
        user_input=query,
        language=language,
        on_sub_agents=on_sub_agents,
    )
    response = outcome["response"] if isinstance(outcome["response"], str) else "NULL"
    ctx.publish_json(
        "answer",
        {"instance_id": instance_id, "response": response},
    )
    ctx.publish_json(
        "trace",
        {
            "instance_id": instance_id,
            "language": language,
            "search_backend": backend_name(),
            "steps": outcome["steps"],
            "errors": outcome["errors"],
            "stop": outcome["stop"],
            "sub_agents": [row for batch in outcome["sub_calls"] for row in batch],
        },
    )
    if outcome["stop"] == "finished" and response != "NULL":
        return RunTerminal.completed("answered")
    return RunTerminal.failed(str(outcome["stop"]))
