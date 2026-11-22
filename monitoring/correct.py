"""Bias-corrected failure prevalence for a monitoring period."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def corrected_mode_prevalence(
    sample_preds: Sequence[int],
    test_labels: Sequence[int],
    test_preds: Sequence[int],
    confidence: float = 0.95,
    bootstrap_iterations: int = 20000,
    seed: int | None = 7,
) -> dict[str, Any]:
    """Bias-corrected live prevalence for one mode from sampled verdicts.

    The contract, precisely:

      1. ``raw`` is the uncorrected flag rate: ``mean(sample_preds)``.
      2. Compute the frozen judge's failure sensitivity and pass specificity
         from ``test_labels`` and ``test_preds``. Both use the monitoring
         convention that 1 means a failure is present. Failure sensitivity is
         the flagged fraction of human-labeled failures. Pass specificity is
         the unflagged fraction of human-labeled passes.
      3. Compute the Rogan-Gladen point estimate, then resample the held-out
         records and sampled predictions to obtain a percentile-bootstrap
         interval. Use a seeded NumPy generator so the committed result is
         reproducible.
      4. Resample the monitoring predictions and the paired held-out records
         independently with replacement. Keep their original sample sizes.
         Discard a draw if the correction cannot be computed. Clamp each
         retained estimate to [0, 1], then take the percentile interval.
         Raise ``ValueError`` if no replicate is valid.

    Args:
        sample_preds: the judge's 0/1 verdicts over the UNIFORM BASE sample
            only (never the risk strata; they are biased toward failure by
            design).
        test_labels: human labels for the frozen Homework 5 judge's test
            split.
        test_preds: the frozen judge's predictions on that test split.
        confidence: interval confidence level.
        bootstrap_iterations: number of percentile-bootstrap replicates.
        seed: numpy seed for a reproducible interval; None leaves the RNG
            untouched.

    Returns:
        {"raw", "corrected", "ci_low", "ci_high", "confidence",
         "failure_sensitivity", "pass_specificity", "n_sample"}
        with "corrected" clamped to [0, 1] and rates rounded to 4 places.

    Raises:
        ValueError: if an input is empty, the held-out inputs have different
            lengths, a value is not 0 or 1, a class is absent, the judge is
            missing a usable correction, or no bootstrap replicate is valid.
    """
    sample = np.asarray(sample_preds, dtype=int)
    labels = np.asarray(test_labels, dtype=int)
    preds = np.asarray(test_preds, dtype=int)
    if sample.size == 0 or labels.size == 0:
        raise ValueError("sample and held-out inputs must be nonempty")
    if labels.size != preds.size:
        raise ValueError("test_labels and test_preds must have the same length")
    for name, values in (("sample_preds", sample), ("test_labels", labels), ("test_preds", preds)):
        if not np.isin(values, (0, 1)).all():
            raise ValueError(f"{name} must contain only 0 and 1")
    if labels.sum() == 0 or labels.sum() == labels.size:
        raise ValueError("the held-out labels need both failures and passes")

    def judge_rates(lab: np.ndarray, pre: np.ndarray) -> tuple[float, float] | None:
        failures, passes = lab == 1, lab == 0
        if not failures.any() or not passes.any():
            return None
        sensitivity = float(pre[failures].mean())         # failures the judge flags
        specificity = float(1 - pre[passes].mean())       # passes the judge leaves unflagged
        return sensitivity, specificity

    def rogan_gladen(raw: float, sensitivity: float, specificity: float) -> float | None:
        # observed = true*sens + (1-true)*(1-spec)  =>  true = (observed + spec - 1) / (sens + spec - 1)
        youden = sensitivity + specificity - 1
        if youden <= 0:
            return None  # the judge is no better than chance: no usable correction
        return (raw + specificity - 1) / youden

    raw = float(sample.mean())
    rates = judge_rates(labels, preds)
    assert rates is not None
    sensitivity, specificity = rates
    point = rogan_gladen(raw, sensitivity, specificity)
    if point is None:
        raise ValueError("the judge has no usable correction (sensitivity + specificity <= 1)")

    # Two independent sources of uncertainty: which conversations were sampled,
    # and which held-out records measured the judge. Resample both.
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(bootstrap_iterations):
        draw = sample[rng.integers(0, sample.size, sample.size)]
        index = rng.integers(0, labels.size, labels.size)
        draw_rates = judge_rates(labels[index], preds[index])
        if draw_rates is None:
            continue
        estimate = rogan_gladen(float(draw.mean()), *draw_rates)
        if estimate is not None:
            estimates.append(min(1.0, max(0.0, estimate)))
    if not estimates:
        raise ValueError("no bootstrap replicate could be corrected")
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(estimates, [tail, 100 - tail])

    return {
        "raw": round(raw, 4),
        "corrected": round(min(1.0, max(0.0, point)), 4),
        "ci_low": round(float(low), 4),
        "ci_high": round(float(high), 4),
        "confidence": confidence,
        "failure_sensitivity": round(sensitivity, 4),
        "pass_specificity": round(specificity, 4),
        "n_sample": int(sample.size),
    }
