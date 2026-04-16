from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _metric_value(payload: dict[str, Any], metric_name: str) -> float:
    if "metrics" in payload and metric_name in payload["metrics"]:
        return float(payload["metrics"][metric_name])
    if metric_name in payload:
        return float(payload[metric_name])
    raise KeyError(f"missing metric: {metric_name}")


def _exactness_payload(payload: dict[str, Any]) -> dict[str, Any]:
    exactness = payload.get("exactness", payload)
    mismatch_count = int(exactness.get("mismatch_count", 0))
    if "passed" in exactness:
        passed = bool(exactness["passed"])
    elif "exact_pass" in exactness:
        passed = bool(exactness["exact_pass"])
    elif "status" in exactness:
        passed = str(exactness["status"]).lower() == "pass"
    else:
        passed = mismatch_count == 0
    return {"passed": passed, "mismatch_count": mismatch_count}


def evaluate_exactness(
    payload: dict[str, Any],
    exactness_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = exactness_policy or {"mode": "exact-parity"}
    base = _exactness_payload(payload)
    exactness = payload.get("exactness", payload)
    mode = str(policy.get("mode", "exact-parity"))

    if mode == "exact-parity":
        return {
            **base,
            "mode": mode,
            "abs_tolerance": 0.0,
            "rel_tolerance": 0.0,
        }

    if mode != "bounded-tolerance":
        raise ValueError(f"unsupported exactness mode: {mode}")

    abs_tolerance = float(policy.get("abs_tolerance", 0.0))
    rel_tolerance = float(policy.get("rel_tolerance", 0.0))
    require_logic_equivalence = bool(policy.get("require_logic_equivalence", True))
    require_algorithm_equivalence = bool(policy.get("require_algorithm_equivalence", True))
    logic_equivalent = bool(exactness.get("logic_equivalent", True))
    algorithm_equivalent = bool(exactness.get("algorithm_equivalent", True))
    max_abs_error = float(exactness.get("max_abs_error", 0.0))
    max_rel_error = float(exactness.get("max_rel_error", 0.0))

    passed = True
    if require_logic_equivalence and not logic_equivalent:
        passed = False
    if require_algorithm_equivalence and not algorithm_equivalent:
        passed = False
    if max_abs_error > abs_tolerance:
        passed = False
    if max_rel_error > rel_tolerance:
        passed = False

    return {
        "passed": passed,
        "mismatch_count": base["mismatch_count"],
        "mode": mode,
        "abs_tolerance": abs_tolerance,
        "rel_tolerance": rel_tolerance,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "logic_equivalent": logic_equivalent,
        "algorithm_equivalent": algorithm_equivalent,
    }


def compare_runs(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    metric_name: str,
    metric_direction: str,
    exactness_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    baseline_exactness = evaluate_exactness(baseline, exactness_policy)
    candidate_exactness = evaluate_exactness(candidate, exactness_policy)
    if not baseline_exactness["passed"]:
        raise ValueError("baseline exactness must pass before comparing candidates")

    baseline_value = _metric_value(baseline, metric_name)
    candidate_value = _metric_value(candidate, metric_name)

    if metric_direction == "lower_is_better":
        improvement = baseline_value - candidate_value
    elif metric_direction == "higher_is_better":
        improvement = candidate_value - baseline_value
    else:
        raise ValueError(f"unsupported metric direction: {metric_direction}")

    relative_improvement = None
    if baseline_value != 0:
        relative_improvement = improvement / abs(baseline_value)

    if not candidate_exactness["passed"]:
        keep = False
        rejection_reason = "exactness_failed"
    elif improvement > 0:
        keep = True
        rejection_reason = None
    else:
        keep = False
        rejection_reason = "not_improved"

    return {
        "metric_name": metric_name,
        "metric_direction": metric_direction,
        "baseline_value": baseline_value,
        "candidate_value": candidate_value,
        "improvement": improvement,
        "relative_improvement": relative_improvement,
        "baseline_exactness": baseline_exactness,
        "candidate_exactness": candidate_exactness,
        "keep": keep,
        "rejection_reason": rejection_reason,
    }
