"""Official HLE judge.

The prompt is ``JUDGE_PROMPT``. A ``yes`` token is score 1. Transport failure,
an empty judge body, and an unparseable verdict raise ``ScriptError``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ageval_sdk import evaluation_check

try:
    from ageval_sdk.errors import ScriptError
except ImportError:

    class ScriptError(Exception):
        """Same title/message shape as ageval_sdk.errors.ScriptError.

        ageval 0.8.1 does not ship that class. Its evaluator worker still
        records any raised exception as ERROR.
        """

        def __init__(self, title: str, message: str) -> None:
            if not isinstance(title, str) or not title.strip():
                raise ValueError("ScriptError title must be a non-empty string")
            if not isinstance(message, str):
                raise TypeError("ScriptError message must be a string")
            super().__init__(message)
            self.title = title
            self.message = message

from shared.lib.prompts import JUDGE_PROMPT

_JUDGE_TURNS = 2
_FIELD = re.compile(
    r"^(extracted_final_answer|reasoning|correct|confidence)\s*:\s*",
    re.MULTILINE,
)
_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class JudgeFailure(RuntimeError):
    def __init__(self, title: str, message: str) -> None:
        super().__init__(message)
        self.title = title
        self.message = message


def fill_judge_prompt(question: str, response: str, correct_answer: str) -> str:
    """Substitute the three template placeholders without ``str.format``.

    Official ``JUDGE_PROMPT.format`` raises ``ValueError`` when the question,
    the answer, or the response contains a brace. Inserted text is not scanned
    again, so a brace or a placeholder-shaped span inside the question stays.
    """
    mapping = {
        "{question}": question,
        "{response}": response,
        "{correct_answer}": correct_answer,
    }
    template = JUDGE_PROMPT
    out: list[str] = []
    cursor = 0
    while cursor < len(template):
        next_at: int | None = None
        next_key: str | None = None
        for key in mapping:
            found = template.find(key, cursor)
            if found >= 0 and (next_at is None or found < next_at):
                next_at = found
                next_key = key
        if next_key is None or next_at is None:
            out.append(template[cursor:])
            break
        out.append(template[cursor:next_at])
        out.append(mapping[next_key])
        cursor = next_at + len(next_key)
    prompt = "".join(out)
    for key in mapping:
        if template.count(key) != 1:
            raise JudgeFailure("judge_prompt_placeholder", f"{key} count is {template.count(key)}")
    return prompt


def parse_judge(text: str) -> dict[str, Any]:
    parsed = _json_object(text)
    if parsed is None:
        parsed = _labeled(text)
    if parsed is None:
        raise JudgeFailure("judge_response_unparsed", text[:2000])
    if "extracted_final_answer" not in parsed:
        raise JudgeFailure("judge_response_unparsed", "missing extracted_final_answer")
    answer = parsed.get("extracted_final_answer")
    if not isinstance(answer, str):
        answer = "" if answer is None else str(answer)
    correct = _correct_token(parsed.get("correct"))
    confidence = _confidence(parsed.get("confidence"))
    reasoning = parsed.get("reasoning")
    if not isinstance(reasoning, str):
        reasoning = "" if reasoning is None else str(reasoning)
    return {
        "extracted_final_answer": answer,
        "reasoning": reasoning,
        "correct": correct,
        "confidence": confidence,
    }


def _json_object(text: str) -> dict[str, Any] | None:
    raw = text.strip()
    for candidate in (raw, _fenced(raw), _braced(raw)):
        if not candidate:
            continue
        try:
            loaded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return loaded
    return None


def _fenced(text: str) -> str:
    match = _FENCE.search(text)
    return match.group(1) if match else ""


def _braced(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return ""
    return text[start : end + 1]


def _labeled(text: str) -> dict[str, Any] | None:
    matches = list(_FIELD.finditer(text))
    if not matches:
        return None
    found: dict[str, Any] = {}
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        found[match.group(1)] = text[match.end() : stop].strip()
    if "correct" not in found or "extracted_final_answer" not in found:
        return None
    return found


def _correct_token(value: object) -> str:
    if not isinstance(value, str):
        raise JudgeFailure("judge_response_unparsed", f"correct is {value!r}")
    token = value.strip().strip("\"'").strip().rstrip(".,;").strip().lower()
    first = token.split()[0] if token else ""
    if first not in {"yes", "no"}:
        raise JudgeFailure("judge_response_unparsed", f"correct is {value!r}")
    return first


def _confidence(value: object) -> int:
    if value is None or value == "":
        return 100
    if isinstance(value, bool):
        raise JudgeFailure("judge_response_unparsed", f"confidence is {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    match = re.search(r"-?\d+", str(value))
    if match is None:
        raise JudgeFailure("judge_response_unparsed", f"confidence is {value!r}")
    return int(match.group(0))


async def _judge_text(session: Any, prompt: str) -> str:
    reply = await session.invoke(prompt, messages=[{"role": "user", "content": prompt}])
    if not reply.get("ok"):
        raise JudgeFailure("judge_invoke_failed", str(reply.get("error") or "invoke_failed"))
    text = str(reply.get("text") or "").strip()
    if not text:
        raise JudgeFailure("judge_response_empty", "judge returned no text")
    return text


async def evaluate(inputs: dict[str, Any]) -> dict[str, Any]:
    try:
        return await _evaluate(inputs)
    except JudgeFailure as exc:
        raise ScriptError(exc.title, exc.message) from exc
    except ScriptError:
        raise
    except Exception as exc:
        raise ScriptError("evaluator_infrastructure", f"{type(exc).__name__}: {exc}") from exc


async def _evaluate(inputs: dict[str, Any]) -> dict[str, Any]:
    parameters = inputs.get("parameters") or {}
    question_id = str(parameters.get("id") or "")
    evaluation_dir = Path(str(inputs["evaluation_dir"]))
    gold = json.loads((evaluation_dir / "gold.json").read_text(encoding="utf-8"))
    if str(gold.get("id") or "") != question_id:
        raise ScriptError(
            "instance_id_mismatch",
            f"gold {gold.get('id')!r} != task {question_id!r}",
        )
    question = gold.get("question")
    answer = gold.get("answer")
    if not isinstance(question, str) or not isinstance(answer, str):
        raise ScriptError("gold_missing_field", "gold.json needs question and answer strings")

    artifact = (inputs.get("artifacts") or {}).get("response")
    if not artifact:
        raise ScriptError("response_missing", "solver published no response")
    payload = json.loads(Path(str(artifact)).read_text(encoding="utf-8"))
    if str(payload.get("id") or "") != question_id:
        raise ScriptError(
            "instance_id_mismatch",
            f"response {payload.get('id')!r} != task {question_id!r}",
        )
    response = payload.get("response")
    if not isinstance(response, str) or not response.strip():
        raise ScriptError("response_missing", "solver response is empty")

    agent = inputs.get("agent")
    if agent is None:
        raise ScriptError("judge_session_missing", "evaluator has no agent session")
    prompt = fill_judge_prompt(question, response, answer)
    async with agent.session("judge", max_turns=_JUDGE_TURNS) as session:
        judged = parse_judge(await _judge_text(session, prompt))

    score = 1.0 if judged["correct"] == "yes" else 0.0
    status = "PASS" if score == 1.0 else "FAIL"
    metrics = {
        "correct": judged["correct"],
        "confidence": judged["confidence"],
        "extracted_final_answer": judged["extracted_final_answer"],
        "reasoning": judged["reasoning"],
        "answer_type": str(gold.get("answer_type") or ""),
        "category": str(gold.get("category") or ""),
    }
    checks = [
        evaluation_check(
            "correct",
            title="official judge correct",
            status=status,
            score=score,
        )
    ]
    return {"status": status, "score": score, "metrics": metrics, "checks": checks}
