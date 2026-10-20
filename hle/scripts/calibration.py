"""Suite accuracy and calibration from judged rows.

This is the dataset-level report in ``hle_eval/run_judge_results.py``.
It is not the per-task PASS. Each input row needs ``correct`` (``yes`` / ``no``
or a boolean) and ``confidence`` (0–100). Official ``beta`` is 100.
``--n`` is the accuracy denominator. Official uses the full question count,
including questions that were never judged.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.lib.calibration import calib_err


def _correct(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "yes"
    raise SystemExit(f"correct must be yes/no or bool, got {value!r}")


def _confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"confidence must be a number, got {value!r}")
    return float(value) / 100.0


def main() -> None:
    parser = argparse.ArgumentParser(description="HLE suite accuracy and calibration error.")
    parser.add_argument("rows", type=Path, help="JSONL with correct and confidence.")
    parser.add_argument("--beta", type=int, default=100)
    parser.add_argument("--n", type=int, default=None, help="Accuracy denominator. Default: row count.")
    args = parser.parse_args()
    correct: list[float] = []
    confidence: list[float] = []
    for line in args.rows.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        correct.append(1.0 if _correct(row.get("correct")) else 0.0)
        confidence.append(_confidence(row.get("confidence")))
    if not correct:
        raise SystemExit("no rows")
    n = len(correct) if args.n is None else args.n
    if n < 1:
        raise SystemExit("--n must be ≥ 1")
    accuracy = round(100 * sum(correct) / n, 2)
    half = round(1.96 * math.sqrt(accuracy * (100 - accuracy) / n), 2)
    error = 100 * round(calib_err(confidence, correct, p="2", beta=args.beta), 2)
    print("*** Metrics ***")
    print(f"Accuracy: {accuracy}% +/- {half}% | n = {n}")
    print(f"Calibration Error: {error}")


if __name__ == "__main__":
    main()
