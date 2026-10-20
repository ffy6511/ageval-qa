"""Suite calibration from ``hle_eval/run_judge_results.py``.

``calib_err`` follows that function, including the loop over ``range(len(bins) - 1)``,
which drops the last bin. ``beta`` larger than the row count raises ``IndexError``
on the empty bin list, the same way the official script does.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def calib_err(
    confidence: Sequence[float],
    correct: Sequence[float],
    p: str = "2",
    beta: int = 100,
) -> float:
    """Binned calibration error. ``confidence`` is in ``[0, 1]``."""
    if len(confidence) != len(correct):
        raise ValueError("confidence and correct must have the same length")
    order = sorted(range(len(confidence)), key=lambda index: confidence[index])
    ranked_confidence = [float(confidence[index]) for index in order]
    ranked_correct = [float(correct[index]) for index in order]
    total = len(ranked_confidence)
    bins = [[index * beta, (index + 1) * beta] for index in range(total // beta)]
    bins[-1] = [bins[-1][0], total]

    cerr = 0.0
    for index in range(len(bins) - 1):
        start, stop = bins[index]
        bin_confidence = ranked_confidence[start:stop]
        bin_correct = ranked_correct[start:stop]
        count = len(bin_confidence)
        if count <= 0:
            continue
        difference = abs(_nanmean(bin_confidence) - _nanmean(bin_correct))
        if p == "2":
            cerr += count / total * difference * difference
        elif p == "1":
            cerr += count / total * difference
        elif p in {"infty", "infinity", "max"}:
            cerr = max(cerr, difference)
        else:
            raise ValueError("p must be '1', '2', or 'infty'")

    if p == "2":
        cerr = math.sqrt(cerr)
    return float(cerr)


def _nanmean(values: Sequence[float]) -> float:
    kept = [value for value in values if value == value]
    if not kept:
        return float("nan")
    return sum(kept) / len(kept)
