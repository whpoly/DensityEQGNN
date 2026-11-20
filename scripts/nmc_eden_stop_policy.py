"""Validation-only stopping rule for the existing pure-probe NMC run."""
import math


POLICY = dict(
    version=2,
    minimum_training_steps=1_000_000,
    validation_interval=5_000,
    window_steps=50_000,
    relative_improvement_threshold=0.005,
    metrics=["density_nmae", "occupation_relative_l1"],
    aggregation="minimum_per_metric_in_each_nonoverlapping_window",
    combination="both_metrics_below_threshold",
    require_lr_reduction=True,
    minimum_steps_after_last_lr_reduction=50_000,
)


def evaluate(history, policy=POLICY):
    """Compare (t-100k,t-50k] with (t-50k,t]; never inspect test data."""
    interval, window = policy["validation_interval"], policy["window_steps"]
    if window <= 0 or window % interval:
        raise ValueError("Window must be a positive multiple of the validation interval")
    if not 0 < policy["relative_improvement_threshold"] < 1:
        raise ValueError("Invalid relative improvement threshold")
    if policy["minimum_training_steps"] < 0 or policy["minimum_training_steps"] % interval:
        raise ValueError("Minimum training steps must align with validation")
    steps = [r["step"] for r in history]
    if steps and steps != list(range(interval, steps[-1] + 1, interval)):
        raise ValueError("Validation history must be complete, ordered and unique")
    for row in history:
        for key in policy["metrics"]:
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError(f"Invalid validation metric {key} at {row['step']}")
        if not 0 < row["next_lr"] <= row["lr"]:
            raise ValueError("Learning-rate history does not match the plateau schedule")
    step = steps[-1] if steps else 0
    reductions = [r["step"] for r in history if r["next_lr"] < r["lr"]]
    last_reduction = reductions[-1] if reductions else None
    since_reduction = step - last_reduction if reductions else None
    result = dict(step=step, eligible=False, stop=False, policy=policy,
                  last_lr_reduction_step=last_reduction,
                  steps_since_last_lr_reduction=since_reduction,
                  metrics={}, reason="insufficient_validation_history")
    if step < 2 * window:
        return result
    baseline = [r for r in history if step - 2 * window < r["step"] <= step - window]
    recent = [r for r in history if step - window < r["step"] <= step]
    if len(baseline) != window // interval or len(recent) != window // interval:
        raise ValueError("Incomplete comparison window")
    result["windows"] = dict(baseline_steps=[r["step"] for r in baseline],
                             recent_steps=[r["step"] for r in recent])
    for key in policy["metrics"]:
        previous = min(baseline, key=lambda r: r[key])
        current = min(recent, key=lambda r: r[key])
        gain = 1 - current[key] / previous[key] if previous[key] else (0.0 if current[key] == 0 else -1.0)
        threshold = policy["relative_improvement_threshold"]
        below = gain < threshold and not math.isclose(gain, threshold, rel_tol=1e-12, abs_tol=1e-14)
        result["metrics"][key] = dict(baseline_best=previous[key], recent_best=current[key],
                                    baseline_best_step=previous["step"], recent_best_step=current["step"],
                                    relative_improvement=gain, below_threshold=below)
    if step < policy["minimum_training_steps"]:
        result["reason"] = "minimum_training_steps_not_reached"
        return result
    if policy["require_lr_reduction"] and last_reduction is None:
        result["reason"] = "awaiting_first_lr_reduction"
        return result
    if last_reduction is not None and since_reduction < policy["minimum_steps_after_last_lr_reduction"]:
        result["reason"] = "observing_after_lr_reduction"
        return result
    result["eligible"] = True
    result["stop"] = all(r["below_threshold"] for r in result["metrics"].values())
    result["reason"] = "dual_metric_plateau" if result["stop"] else "meaningful_validation_improvement"
    return result
