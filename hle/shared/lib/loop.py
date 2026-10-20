"""One official HLE prediction call.

``format_message`` in ``hle_eval/run_model_predictions.py`` sends ``SYSTEM_PROMPT``
as a system turn and the question as a user content list. An image, when the
row has one, is an ``image_url`` part whose url is the dataset string.
"""

from __future__ import annotations

from ageval_sdk import RunContext, RunTerminal

from shared.lib.prompts import SYSTEM_PROMPT


async def run(ctx: RunContext) -> RunTerminal:
    question_id = ctx.params.require_str("id")
    question = (ctx.workspace_root / "question.txt").read_text(encoding="utf-8")
    image_path = ctx.workspace_root / "image_url.txt"
    image = image_path.read_text(encoding="utf-8") if image_path.is_file() else ""
    content: list[dict[str, object]] = [{"type": "text", "text": question}]
    if image:
        content.append({"type": "image_url", "image_url": {"url": image}})
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    async with ctx.agent.session("solver", max_turns=1) as session:
        reply = await session.invoke(question, messages=messages)
    if not reply.get("ok"):
        return RunTerminal.failed(str(reply.get("error") or "invoke_failed"))
    text = str(reply.get("text") or "")
    if not text.strip():
        return RunTerminal.failed("empty_response")
    ctx.publish_json("response", {"id": question_id, "response": text})
    return RunTerminal.completed("ok")
