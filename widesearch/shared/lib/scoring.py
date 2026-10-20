# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
"""WideSearch table scoring.

Metric functions follow ``src/evaluation/metric_utils.py`` and
``evaluate_single_query`` at commit 9825ba7. Judge calls go through the
``judge`` profile. A judge transport failure, an empty judge body, or an
unparseable judge JSON raises ``ScriptError`` instead of becoming score 0.
"""

from __future__ import annotations

import json
import re
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import dateparser
import pandas as pd
from ageval_sdk import ScriptError, evaluation_check

_JUDGE_TURNS = 128

_PRIMARY_KEY_PROMPT = """Your task is to align two vocabularies. The inputs are the vocabulary to be aligned and the reference vocabulary respectively. Note that you need to perform semantic alignment (not positional alignment). If two strings are exactly the same, they must correspond to each other. These two strings are supposed to represent the same entity, with differences only in the expression forms and formats.


The vocabulary to be aligned is as follows:
{response}

The reference vocabulary is as follows:
{reference}

The alignment rules are as follows:
List the values in the vocabulary to be aligned one by one. If there is a value in the reference vocabulary that has the same meaning as this value, `transform` should be represented as the value from the reference vocabulary; otherwise, `transform` should be represented as the original value from the vocabulary to be aligned.

Note that `origin` must be taken from the vocabulary to be aligned keeping the original format, and `transform` must be taken from the reference vocabulary. For example: Some words in the vocabulary to be aligned might be the words in the reference vocabulary with Markdown formatting added, keep the to be aligned format in `origin` and the reference format in `transform`.

For the `origin`, first find the `transform` that is the closest in meaning and then judge whether they correspond to each other. Those entities not correspond to each other could not output.

Please output the alignment results in the following format:
```json
{{
    "origin_str1": "transform_str1",
    "origin_str2": "transform_str2"
}}
```
"""

_COLUMN_PROMPT = """You are an expert in grading answers. Your task is to score the responses to a certain question. Below, you will be provided with a set of standard answers, a set of responses to be graded, and specific grading criteria.

Each answer and each response has an idx. Please score each pair of answers and responses in this set according to the following methods:
1. The scoring range is from 0 to 1. A score of 1 indicates a completely correct answer. For deduction items, please refer to the specific grading criteria section.
2. After reading the standard answers, responses to be graded, and grading criteria, please first analyze and judge them item by item according to the grading criteria.
3. The score can only be an integer of 0 or 1.
4. After the analysis and judgment, please provide the final scoring results. Each pair should have a score. Output in Markdown JSON format, as shown below:
```json
{{
    "idx_xxx": score,
    "idx_yyy": score,
    ...
}}
```

====== criterion-start ======
{criterion}
====== criterion-end ======

====== response-start ======
{response}
====== response-end ======

Now start scoring. Please make sure to analyze each item step by step before providing the final scoring results.

"""


class JudgeFailure(RuntimeError):
    def __init__(self, title: str, message: str) -> None:
        super().__init__(message)
        self.title = title
        self.message = message


def norm_column(col: str) -> str:
    return col.strip().lower().replace(" ", "")


def extract_number(content: str) -> str:
    numbers = re.findall(
        r"[-+]?\d*\.\d+%?|[-+]?\d+\.?\d*%?",
        str(content).replace(",", ""),
    )
    if len(numbers) == 0:
        return "NULL"
    return numbers[0]


def norm_str(content: object) -> str:
    return str(content).lower().strip().replace(" ", "").replace("*", "")


def norm_date(content: str) -> str:
    normalized = dateparser.parse(content, settings={"PREFER_DAY_OF_MONTH": "first"})
    if normalized is None:
        return content
    return normalized.strftime("%Y-%m-%d")


_PREPROCESS = {
    "extract_number": extract_number,
    "norm_str": norm_str,
    "norm_date": norm_date,
}


def exact_match(response: str, target: str) -> tuple[float, str]:
    if response.lower() == target.lower():
        return 1.0, f"exact match, response: {response}, target: {target}"
    return 0.0, f"exact not match, response: {response}, target: {target}"


def url_match(response: str, target: str) -> tuple[float, str]:
    url_pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    response_urls = [urlparse(url).netloc for url in url_pattern.findall(response)]
    target_urls = [urlparse(url).netloc for url in url_pattern.findall(target)]
    if set(response_urls) == set(target_urls):
        return 1.0, f"url match, response: {response}, target: {target}"
    return 0.0, f"url not match, response: {response}, target: {target}"


def in_match(response: str, target: str) -> tuple[float, str]:
    if response in target:
        return 1.0, f"response in target, response: {response}, target: {target}"
    return 0.0, f"response not in target, response: {response}, target: {target}"


def number_near(response: str, target: str, criterion: float) -> tuple[float, str]:
    response_num: float | None
    target_num: float | None
    if "%" in response:
        try:
            response_num = float(response.replace("%", "")) / 100.0
        except (ValueError, TypeError):
            response_num = None
    else:
        try:
            response_num = float(response)
        except (ValueError, TypeError):
            response_num = None
    if "%" in target:
        try:
            target_num = float(target.replace("%", "")) / 100.0
        except (ValueError, TypeError):
            target_num = None
    else:
        try:
            target_num = float(target)
        except (ValueError, TypeError):
            target_num = None
    if response_num is None or target_num is None:
        if response_num is None and target_num is None and response == target:
            return 1.0, f"number equal, response: {response}, target: {target}"
        return 0.0, f"number not convertable, response: {response_num}, target: {target_num}"
    if abs(response_num - target_num) <= abs(target_num) * criterion:
        return (
            1.0,
            f"number near in range {criterion * 100}%, response: {response_num}, target: {target_num}",
        )
    return 0.0, f"number not near, response: {response_num}, target: {target_num}"


def date_near(response: str, target: str) -> tuple[float, str]:
    try:
        response_date = dateparser.parse(response, settings={"PREFER_DAY_OF_MONTH": "first"})
    except Exception:
        response_date = None
    try:
        target_date = dateparser.parse(target, settings={"PREFER_DAY_OF_MONTH": "first"})
    except Exception:
        target_date = None
    if response_date is None or target_date is None:
        if response_date is None and target_date is None:
            return 1.0, f"date near, response: {response}, target: {target}"
        return 0.0, f"date not convertable, response: {response}, target: {target}"
    if abs((response_date - target_date).days) <= 31:
        return 1.0, f"date near, response: {response_date}, target: {target_date}"
    return 0.0, f"date not near, response: {response_date}, target: {target_date}"


def parse_markdown_json(completion: str) -> dict[str, Any] | None:
    matches = re.findall(r"```json\s*(\{.*?\})\s*```", completion, flags=re.DOTALL)
    if not matches:
        return None
    try:
        parsed = json.loads(matches[-1])
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def extract_dataframe(response: str) -> pd.DataFrame | None:
    markdown = re.findall(r"```markdown(.*?)```", response, flags=re.DOTALL)
    if not markdown:
        pipe_positions = [match.start() for match in re.finditer(r"\|", response)]
        if len(pipe_positions) >= 4:
            first_pipe = pipe_positions[0]
            last_pipe = pipe_positions[-1]
            start = response.rfind("\n", 0, first_pipe)
            start = 0 if start == -1 else start
            end = response.find("\n", last_pipe)
            end = len(response) if end == -1 else end
            table_candidate = response[start:end]
            markdown = re.findall(r"((?:\|.*\n?)+)", table_candidate)
    if not markdown:
        return None
    table = markdown[0].strip()
    lines = table.split("\n")
    lines[0] = lines[0].replace(" ", "").lower()
    lines = [line.strip() for line in lines]
    kept: list[str] = []
    for line in lines:
        if set(line.strip()).issubset(set("|- :")) or "|" not in line:
            continue
        kept.append("|".join(cell.strip() for cell in line.split("|")))
    frame = pd.read_csv(StringIO("\n".join(kept)), sep="|")
    return frame.loc[:, ~frame.columns.str.startswith("Unnamed")]


def _metric(response: str, target: str, criterion: object, name: str) -> tuple[float, str]:
    if name == "exact_match":
        return exact_match(response, target)
    if name == "url_match":
        return url_match(response, target)
    if name == "in_match":
        return in_match(response, target)
    if name == "date_near":
        return date_near(response, target)
    if name == "number_near":
        return number_near(response, target, float(criterion))  # type: ignore[arg-type]
    raise ScriptError("unknown_metric", f"metric {name} is not in the WideSearch registry")


def _f1(precision: float, recall: float) -> float:
    if precision + recall > 1e-9:
        return (2 * precision * recall) / (precision + recall)
    return 0.0


def _verdict(
    *,
    score: float,
    precision_by_row: float,
    recall_by_row: float,
    f1_by_row: float,
    precision_by_item: float,
    recall_by_item: float,
    f1_by_item: float,
    msg: str,
) -> dict[str, Any]:
    status = "PASS" if score == 1 else "FAIL"
    metrics = {
        "table_score": float(score),
        "precision_by_row": float(precision_by_row),
        "recall_by_row": float(recall_by_row),
        "f1_by_row": float(f1_by_row),
        "precision_by_item": float(precision_by_item),
        "recall_by_item": float(recall_by_item),
        "f1_by_item": float(f1_by_item),
        "msg": msg[:12000],
    }
    checks = [
        evaluation_check(
            "f1_by_row",
            title="row F1",
            status="PASS" if f1_by_row == 1 else "FAIL",
            score=float(f1_by_row),
        ),
        evaluation_check(
            "f1_by_item",
            title="item F1",
            status="PASS" if f1_by_item == 1 else "FAIL",
            score=float(f1_by_item),
        ),
        evaluation_check(
            "table_score",
            title="official table score",
            status=status,
            score=float(score),
        ),
    ]
    return {"status": status, "score": float(score), "metrics": metrics, "checks": checks}


def _zero(msg: str) -> dict[str, Any]:
    return _verdict(
        score=0.0,
        precision_by_row=0.0,
        recall_by_row=0.0,
        f1_by_row=0.0,
        precision_by_item=0.0,
        recall_by_item=0.0,
        f1_by_item=0.0,
        msg=msg,
    )


async def _judge_text(session: Any, prompt: str) -> str:
    reply = await session.invoke(prompt, messages=[{"role": "user", "content": prompt}])
    if not reply.get("ok"):
        raise JudgeFailure("judge_invoke_failed", str(reply.get("error") or "invoke_failed"))
    text = str(reply.get("text") or "").strip()
    if not text:
        raise JudgeFailure("judge_response_empty", "judge returned no text")
    return text


async def _align(session: Any, response: list[str], reference: list[str]) -> dict[str, Any]:
    text = await _judge_text(
        session,
        _PRIMARY_KEY_PROMPT.format(response=response, reference=reference),
    )
    parsed = parse_markdown_json(text)
    if parsed is None:
        raise JudgeFailure("judge_response_unparsed", text[:2000])
    return parsed


async def _judge_column(
    session: Any,
    response: list[str],
    target: list[str],
    criterion: object,
) -> tuple[list[float], list[str]]:
    if len(response) != len(target):
        raise JudgeFailure(
            "judge_score_invalid",
            f"response rows {len(response)} != target rows {len(target)}",
        )
    payload = {
        f"idx_{idx}": {"response": resp, "target": tar}
        for idx, (resp, tar) in enumerate(zip(response, target, strict=True))
    }
    text = await _judge_text(
        session,
        _COLUMN_PROMPT.format(criterion=criterion, response=payload),
    )
    parsed = parse_markdown_json(text)
    if parsed is None:
        raise JudgeFailure("judge_response_unparsed", text[:2000])
    scores: list[float] = []
    for idx in range(len(response)):
        raw = parsed.get(f"idx_{idx}")
        if raw not in (0, 1, 0.0, 1.0):
            raise JudgeFailure("judge_score_invalid", f"idx_{idx}={raw!r}")
        scores.append(float(raw))
    return scores, [text] * len(response)


async def _score_frame(
    session: Any,
    answer_df: pd.DataFrame,
    response_df: pd.DataFrame,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    required = list(evaluation["required"])
    unique = list(evaluation["unique_columns"])
    pipeline = evaluation["eval_pipeline"]
    answer_df = answer_df.copy()
    response_df = response_df.copy()
    answer_df.columns = [norm_column(str(col)) for col in answer_df.columns]
    response_df.columns = [norm_column(str(col)) for col in response_df.columns]
    if set(required) != set(response_df.columns):
        column_map = await _align(session, response_df.columns.tolist(), required)
        response_df = response_df.rename(columns=column_map)
    if set(required) != set(response_df.columns):
        return _zero(
            f"required_columns {required} != response_df {response_df.columns.tolist()}"
        )

    for col in required:
        try:
            answer_type = answer_df[col].dtype
            response_type = response_df[col].dtype
        except Exception:
            answer_type = None
            response_type = None
        if (response_type == float and answer_type == int) or (
            response_type == int and answer_type == float
        ):
            if response_type == int:
                response_df[col] = response_df[col].astype(float)
            elif answer_type == int:
                answer_df[col] = answer_df[col].astype(float)
        answer_df[col] = answer_df[col].astype(str)
        response_df[col] = response_df[col].astype(str)

    response_df = response_df.drop_duplicates(subset=unique)
    answer_df = answer_df.drop_duplicates(subset=unique)

    for col in unique:
        item = pipeline.get(col)
        if not item:
            continue
        names = item.get("metric", [])
        if "llm_judge" in names or "exact_match" in names:
            mapping = await _align(
                session,
                response_df[col].tolist(),
                answer_df[col].tolist(),
            )
            response_df[col + "_before_map"] = response_df[col]
            response_df[col] = response_df[col].apply(lambda cell, mapping=mapping: mapping.get(cell, cell))

    for col, item in pipeline.items():
        for name in item.get("preprocess", []):
            if name not in _PREPROCESS:
                raise ScriptError("unknown_preprocess", f"preprocess {name} is not registered")
            func = _PREPROCESS[name]
            response_df[col] = response_df[col].apply(lambda cell, func=func: func(cell))
            answer_df[col] = answer_df[col].apply(lambda cell, func=func: func(cell))

    score = 0.0
    if answer_df.shape == response_df.shape:
        left = answer_df.sort_values(by=required).reset_index(drop=True)
        right = response_df.sort_values(by=required).reset_index(drop=True)
        if left.equals(right):
            score = 1.0

    inner = pd.merge(
        answer_df,
        response_df,
        on=unique,
        how="inner",
        suffixes=("_query", "_response"),
    )
    score_frame = pd.DataFrame(index=inner.index)
    for col in required:
        if col in unique:
            score_frame[f"{col}_exact_match"] = 1.0
            continue
        item = pipeline[col]
        criterion = item.get("criterion")
        for name in item.get("metric", []):
            if name == "llm_judge":
                values, _msgs = await _judge_column(
                    session,
                    inner[col + "_response"].tolist(),
                    inner[col + "_query"].tolist(),
                    criterion,
                )
                score_frame[f"{col}_{name}"] = values
            else:
                score_frame[f"{col}_{name}"] = inner.apply(
                    lambda row, col=col, criterion=criterion, name=name: _metric(
                        row[col + "_response"],
                        row[col + "_query"],
                        criterion,
                        name,
                    )[0],
                    axis=1,
                )

    row_scores = score_frame.min(axis=1)
    tp_by_row = float(row_scores.sum()) if len(row_scores) else 0.0
    tp_by_item = float(score_frame.sum().sum()) if score_frame.size else 0.0
    num_pred_rows = len(response_df)
    num_gt_rows = len(answer_df)
    num_pred_items = num_pred_rows * len(required)
    num_gt_items = num_gt_rows * len(required)
    precision_by_row = tp_by_row / num_pred_rows if num_pred_rows else 0.0
    recall_by_row = tp_by_row / num_gt_rows if num_gt_rows else 0.0
    precision_by_item = tp_by_item / num_pred_items if num_pred_items else 0.0
    recall_by_item = tp_by_item / num_gt_items if num_gt_items else 0.0
    f1_by_row = _f1(precision_by_row, recall_by_row)
    f1_by_item = _f1(precision_by_item, recall_by_item)
    msg = score_frame.to_string()
    if (
        precision_by_item == recall_by_item == f1_by_item == 1.0
        and precision_by_row == recall_by_row == f1_by_row == 1.0
    ):
        msg += "\nAll items match perfectly."
        score = 1.0
    return _verdict(
        score=score,
        precision_by_row=precision_by_row,
        recall_by_row=recall_by_row,
        f1_by_row=f1_by_row,
        precision_by_item=precision_by_item,
        recall_by_item=recall_by_item,
        f1_by_item=f1_by_item,
        msg=msg,
    )


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
    instance_id = str(parameters.get("instance_id") or "")
    evaluation_dir = Path(str(inputs["evaluation_dir"]))
    spec = json.loads((evaluation_dir / "eval.json").read_text(encoding="utf-8"))
    answer = pd.read_csv(evaluation_dir / "gold.csv")
    answer.columns = [norm_column(str(col).strip()) for col in answer.columns]
    missing = [col for col in spec["required"] if col not in answer.columns]
    if missing:
        raise ScriptError("gold_missing_column", f"{instance_id} missing {missing}")
    answer = answer[list(spec["required"])]

    artifact = (inputs.get("artifacts") or {}).get("answer")
    if not artifact:
        return _zero("response is None")
    payload = json.loads(Path(str(artifact)).read_text(encoding="utf-8"))
    if str(payload.get("instance_id") or "") != instance_id:
        raise ScriptError(
            "instance_id_mismatch",
            f"answer {payload.get('instance_id')!r} != task {instance_id!r}",
        )
    response = payload.get("response")
    if not isinstance(response, str) or not response.strip():
        return _zero("response is None")
    try:
        frame = extract_dataframe(response)
    except Exception as exc:
        return _zero(f"response_df is None ({type(exc).__name__}: {exc})")
    if frame is None:
        return _zero("response_df is None")

    agent = inputs.get("agent")
    if agent is None:
        raise ScriptError("judge_session_missing", "evaluator has no agent session")
    async with agent.session("judge", max_turns=_JUDGE_TURNS) as session:
        return await _score_frame(session, answer, frame, spec)
