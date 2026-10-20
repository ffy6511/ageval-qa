"""Materialise the 200 WideSearch tasks from the pinned Hugging Face revision.

Query text goes to data/query.txt. Gold CSV and the evaluation object go to
evaluation/. Re-running overwrites those task files and leaves shared code alone.
"""

from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "6531a7e5b497d44c8912407e0cb3dc95bd98cc09"
HF = f"https://huggingface.co/datasets/ByteDance-Seed/WideSearch/resolve/{REVISION}"
LOCAL_JSONL = Path("/tmp/widesearch-src/widesearch.jsonl")

RUN_PY = '''"""Thin entry. Orchestration lives in shared.lib.loop."""

from __future__ import annotations

from ageval_sdk import RunContext, RunTerminal
from shared.lib.loop import run as _run


async def run(ctx: RunContext) -> RunTerminal:
    return await _run(ctx)
'''

EVALUATOR_PY = '''"""Thin entry. Scoring lives in shared.lib.scoring."""

from __future__ import annotations

from typing import Any

from shared.lib.scoring import evaluate as _evaluate


async def evaluate(inputs: dict[str, Any]) -> dict[str, Any]:
    return await _evaluate(inputs)
'''


def _task_yaml(instance_id: str, language: str) -> str:
    return f"""format: ageval.task/1
task_id: {instance_id}
parameters:
  instance_id: {instance_id}
  language: {language}
agent_profiles:
  - id: lead
  - id: researcher
  - id: judge
limits:
  wall_time_seconds: 3600
  evaluate_seconds: 1800
  agent_invocations: 400
  agent_concurrency: 4
artifacts:
  publishable:
    - id: answer
      path: artifacts/answer.json
    - id: trace
      path: artifacts/trace.json
evaluation:
  inputs:
    - artifact: answer
      target: artifacts/answer.json
"""


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ageval-widesearch"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _load_rows() -> list[dict]:
    if LOCAL_JSONL.is_file():
        raw = LOCAL_JSONL.read_bytes()
    else:
        raw = _get(f"{HF}/widesearch.jsonl")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != 200:
        raise SystemExit(f"expected 200 rows, got {len(rows)}")
    return rows


def _write_task(row: dict) -> str:
    instance_id = str(row["instance_id"])
    language = str(row["language"])
    evaluation = json.loads(row["evaluation"]) if isinstance(row["evaluation"], str) else row["evaluation"]
    task = ROOT / "tasks" / instance_id
    data = task / "data"
    evaluation_dir = task / "evaluation"
    data.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    (data / "query.txt").write_text(str(row["query"]), encoding="utf-8")
    (evaluation_dir / "eval.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    gold = evaluation_dir / "gold.csv"
    if not gold.is_file() or gold.stat().st_size == 0:
        gold.write_bytes(_get(f"{HF}/widesearch_gold/{instance_id}.csv"))
    (task / "task.yaml").write_text(_task_yaml(instance_id, language), encoding="utf-8")
    (task / "run.py").write_text(RUN_PY, encoding="utf-8")
    (task / "evaluator.py").write_text(EVALUATOR_PY, encoding="utf-8")
    return instance_id


def _license() -> None:
    target = ROOT / "LICENSE-DATA"
    if target.is_file() and target.stat().st_size > 0:
        return
    target.write_bytes(_get(f"{HF}/LICENSE"))


def main() -> None:
    rows = _load_rows()
    _license()
    with ThreadPoolExecutor(max_workers=16) as pool:
        ids = list(pool.map(_write_task, rows))
    langs: dict[str, int] = {}
    for row in rows:
        langs[str(row["language"])] = langs.get(str(row["language"]), 0) + 1
    print(f"wrote {len(ids)} tasks langs={langs}")


if __name__ == "__main__":
    main()
