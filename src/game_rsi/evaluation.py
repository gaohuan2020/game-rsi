"""Protected scoring: paired game seeds, bootstrap intervals, promotion gate."""
from __future__ import annotations

import math
import random
import statistics


def paired_report(baseline: list[float], candidate: list[float], *, min_gain=0.02, require_ci=True, seed=17):
    if len(baseline) != len(candidate) or len(baseline) < 2:
        raise ValueError("Paired evaluation requires at least two equal-length score vectors")
    if not all(math.isfinite(x) for x in baseline + candidate):
        raise ValueError("Scores must be finite")
    delta = [c - b for b, c in zip(baseline, candidate)]
    n = len(delta)
    rng = random.Random(seed)
    boot = sorted(statistics.fmean(rng.choices(delta, k=n)) for _ in range(4000))
    low, high = boot[100], boot[3899]
    before, after = statistics.fmean(baseline), statistics.fmean(candidate)
    gain = statistics.fmean(delta)
    checks = {"score_gain": gain > max(abs(before), 1.0) * min_gain,
              "confidence_interval": low > 0 if require_ci else True,
              "no_large_regression": min(delta) >= -max(abs(before), 1.0) * 0.25}
    return {"baseline": before, "candidate": after, "delta": gain,
            "relative_gain": gain / max(abs(before), 1.0), "ci95": [low, high],
            "baseline_scores": baseline, "candidate_scores": candidate, "pairs": n,
            "checks": checks, "passed": all(checks.values()),
            "min_gain": min_gain, "require_ci": require_ci,
            "scope": "selected game and registered seed panel; not a general ability claim"}


def training_seed(round_index: int, episode: int) -> int:
    # Validation episodes are seed % 10 == 0. Training rounds stay below the evaluation panel.
    seed = 1000 + round_index * 100000 + episode
    if seed >= 90000000:
        raise ValueError("Training seeds overlap the held-out evaluation namespace")
    return seed


def evaluation_seeds(count: int, seed: int) -> list[int]:
    # Stable for a run; never changed by candidates or training.
    return [90000000 + seed * 64 % 1000000 + i for i in range(count)]
