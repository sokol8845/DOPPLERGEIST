"""Matched negative controls for one event-to-flight association.

Ported from AeroDoppler GeoLocator's ``evidence_controls`` and expressed with
DOPPLERGEIST's own bounded-drift ridge fit. A matched flight's Doppler model
must explain the measured ridge clearly better than three alternatives that use
the identical source model, the identical measured samples and the same number
of free parameters:

* a stationary source (Doppler factor 1 at the receiver clock);
* the same measurement reversed in time, against the same flight model;
* the same flight shifted by fixed clock offsets (default -120 s and +120 s).

Control losses are engineering checks, not calibrated statistical tests. A
failed control means the match is not specific to this flight geometry; it does
not establish that the aircraft was absent.
"""
import math

import numpy as np

from .matching import (_PreparedRidge, _fit_ridge_uncached, _frequency_scale,
                       _prepare_ridge_fit_inputs, fit_ridge)

DEFAULT_SHIFTS_S = (-120.0, 120.0)
DEFAULT_MIN_CONTROL_RATIO = 2.0
DEFAULT_MIN_STATIONARY_RMSE_HZ = 0.5


def _finite(value):
    return float(value) if value is not None and math.isfinite(value) else None


def _loss(fit):
    score = fit.get("score", -math.inf)
    return math.inf if not math.isfinite(score) else -float(score)


def specificity_controls(track, observer, measurement, start_time, event, c=343.0, *,
                         shifts_s=DEFAULT_SHIFTS_S, min_control_ratio=DEFAULT_MIN_CONTROL_RATIO,
                         min_stationary_rmse_hz=DEFAULT_MIN_STATIONARY_RMSE_HZ):
    """Return matched-control diagnostics for one event at one receiver and clock.

    ``supported`` is True only when every available control loses by at least
    ``min_control_ratio`` times the matched loss and at least one time-shift
    control has the same trajectory support as the matched fit.
    """
    if not math.isfinite(min_control_ratio) or min_control_ratio <= 1:
        raise ValueError("min_control_ratio must be finite and greater than one")
    if not math.isfinite(min_stationary_rmse_hz) or min_stationary_rmse_hz < 0:
        raise ValueError("min_stationary_rmse_hz must be finite and nonnegative")
    quality = event.get("quality", {})
    sigma = _frequency_scale(event)
    drift = quality.get("source_drift_fraction", 0.02)
    bounds = event["f0_range_hz"]
    base = {"controls_are_calibrated": False, "min_control_ratio": float(min_control_ratio),
            "source_model": "identical_bounded_linear_emission_time_drift",
            "time_shift_s": [float(s) for s in shifts_s]}
    # Score exactly as the ranked hypothesis was scored.
    matched = fit_ridge(track, observer, measurement, start_time, bounds, c, sigma, drift)
    loss = _loss(matched)
    if not math.isfinite(loss):
        return {**base, "supported": False, "reasons": ["matched_fit_unavailable"]}
    prepared = _prepare_ridge_fit_inputs(measurement, sigma, drift)
    times, ridge, weight = prepared.times, prepared.ridge, prepared.weight
    receive = start_time + times
    factor, emission, _ = track.acoustic(receive, observer, c)

    def fit(ridge_values, weights, acoustic_result):
        return _fit_ridge_uncached(track, observer, measurement, start_time, bounds, c, sigma, drift,
                                   acoustic_result=acoustic_result,
                                   _prepared=_PreparedRidge(times, ridge_values, weights))

    stationary = fit(ridge, weight, (np.ones_like(times), receive.copy()))
    reversed_fit = (fit(ridge[::-1].copy(), weight[::-1].copy(), (factor, emission))
                    if np.all(np.isfinite(factor)) and np.all(np.isfinite(emission))
                    else {"score": -math.inf})
    shifted, missing = [], []
    for shift in shifts_s:
        f, e, _ = track.acoustic(receive + shift, observer, c)
        # A shifted control must use every measured sample of the matched fit;
        # missing trajectory support cannot make a control look better.
        if not (np.all(np.isfinite(f)) and np.all(np.isfinite(e))):
            missing.append(float(shift))
            continue
        control = fit(ridge, weight, (f, e))
        shifted.append({"shift_s": float(shift), "loss": _loss(control),
                        "rmse_hz": _finite(control.get("rmse_hz"))})
    required = max(min_control_ratio * loss, loss + 1e-6)
    stationary_loss = _loss(stationary)
    reversed_loss = _loss(reversed_fit)
    reasons = []
    if stationary_loss < required or (stationary.get("rmse_hz") is not None
                                      and stationary["rmse_hz"] < min_stationary_rmse_hz):
        reasons.append("insufficient_gain_over_stationary_source")
    if reversed_loss < required:
        reasons.append("reversed_measurement_not_rejected")
    if any(row["loss"] < required for row in shifted):
        reasons.append("time_shift_control_not_rejected")
    if not shifted:
        reasons.append("time_shift_controls_unavailable")
    return {**base, "supported": not reasons, "reasons": reasons,
            "matched_loss": loss, "matched_rmse_hz": _finite(matched.get("rmse_hz")),
            "stationary_loss": _finite(stationary_loss),
            "stationary_rmse_hz": _finite(stationary.get("rmse_hz")),
            "gain_over_stationary": (1 - loss / stationary_loss) if math.isfinite(stationary_loss)
                                    and stationary_loss > 0 else None,
            "reversed_loss": _finite(reversed_loss),
            "time_shift_controls": [dict(row, loss=_finite(row["loss"])) for row in shifted],
            "time_shift_missing_support_s": missing}
