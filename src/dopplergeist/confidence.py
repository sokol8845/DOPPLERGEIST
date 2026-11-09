"""Conservative, uncalibrated evidence checks for the shared result contract.

Scores rank hypotheses; no score is interpreted as a probability. This module
never reads benchmark truth or changes event clocks/candidates. Detailed decision
diagnostics belong in a job artifact, outside the strict JobResult fields.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np
from pyproj import Geod

from .contracts import utc_timestamp, validate_result


MAX_HARMONIC_OVERLAP_PAIRS = 8192


@dataclass(frozen=True)
class DecisionPolicy:
    """Predeclared engineering limits, not fitted confidence calibration."""
    target_radius_m: float = 1000.0
    min_independent_events: int = 2
    max_event_rmse_hz: float = 3.0
    max_outlier_fraction: float = 0.25
    min_support_fraction: float = 0.95
    near_tie_loss_per_event_hz: float = 0.5
    near_tie_support_per_group: float = 0.2

    def __post_init__(self):
        for name in ("target_radius_m", "max_event_rmse_hz", "near_tie_loss_per_event_hz", "near_tie_support_per_group"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if type(self.min_independent_events) is not int or self.min_independent_events < 2:
            raise ValueError("At least two independent events are required by this policy")
        for name in ("max_outlier_fraction", "min_support_fraction"):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")


def _near_tied(gap, margin):
    # Decimal policy margins and subtracted scores can differ at roundoff.
    # Equality remains competitive; this small guard cannot create acceptance.
    return gap <= margin or math.isclose(gap, margin, rel_tol=1e-12, abs_tol=1e-12)


def _screen_harmonic_overlap(events, *, pair_budget=MAX_HARMONIC_OVERLAP_PAIRS):
    """Bounded uncalibrated screen; do not merge ridges or assert identity."""
    from .discovery import _HARMONIC_RATIOS
    from .ridge_overlap import aligned_ridge_observations, reused_ridge_observations

    total_pairs = len(events) * (len(events) - 1) // 2
    pairs_examined = 0
    overlap = None
    for index, first in enumerate(events):
        for second_index in range(index + 1, len(events)):
            if pairs_examined >= pair_budget:
                break
            second = events[second_index]
            pairs_examined += 1
            aligned = aligned_ridge_observations(first, second)
            if aligned is None:
                continue
            # Chunk paths can follow the same measured peak before branching.
            # A reused sequence is not independent evidence, even when the
            # later divergence fails the whole-ridge harmonic check below.
            reused = reused_ridge_observations(first, second, aligned=aligned)
            if reused is not None:
                overlap = reused
                break
            low, high = first, second
            times, x, y = aligned
            if np.median(x) > np.median(y):
                low, high, x, y = high, low, y, x
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                ratio = float(np.median(y / x))
            if not math.isfinite(ratio):
                continue
            harmonic = min((1.0, *_HARMONIC_RATIOS), key=lambda value: abs(value - ratio))
            def uncertainty(event):
                quality = event.get("quality", {})
                return max(0.0, quality.get("frequency_resolution_hz", 0),
                           quality.get("frequency_uncertainty_hz", 0))
            low_uncertainty, high_uncertainty = uncertainty(low), uncertainty(high)
            if low_uncertainty <= 0 or high_uncertainty <= 0:
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                residual = float(np.quantile(np.abs(y - harmonic * x), 0.9))
                tolerance = 2 * (high_uncertainty + harmonic * low_uncertainty)
            if math.isfinite(residual) and math.isfinite(tolerance) and residual <= tolerance:
                overlap = {"event_ids": [low["event_id"], high["event_id"]],
                           "harmonic_ratio": harmonic, "observed_median_ratio": ratio,
                           "aligned_samples": len(times), "residual_q90_hz": residual,
                           "heuristic_tolerance_hz": tolerance}
                break
        if overlap is not None or pairs_examined >= pair_budget:
            break
    exhaustive = pairs_examined == total_pairs
    return {
        "overlap": overlap,
        "assessment": {
            "exhaustive": exhaustive,
            "conclusive": overlap is not None or exhaustive,
            "pairs_examined": pairs_examined,
            "pair_budget": pair_budget,
        },
    }


def _possible_harmonic_overlap(events):
    """Return the legacy overlap flag while keeping its screen bounded."""
    return _screen_harmonic_overlap(events)["overlap"]


def _finite_search(evidence, diagnostics, candidates, group_counts, policy):
    """Check producer-accounted evaluated modes, without a continuous-space claim."""
    if not isinstance(evidence, dict) or type(evidence.get("schema_version")) is not int or evidence["schema_version"] != 1:
        raise ValueError("Unsupported finite-search evidence")
    if evidence.get("scope") != "evaluated_finite_grid":
        return "finite_search_scope_unsupported", {}
    if any(type(evidence.get(name)) is not bool for name in ("complete", "pruning_accounted_for")):
        raise ValueError("Finite-search completion/accounting flags must be boolean")
    if evidence.get("complete") is not True or evidence.get("pruning_accounted_for") is not True:
        return "finite_search_incomplete", {}
    if evidence.get("score_kind") != candidates[0]["score_kind"]:
        return "incomparable_candidates", {}
    for name in ("spacing_m", "refine_spacing_m"):
        value = evidence.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Finite-search resolutions must be positive finite metres")
    maximum = evidence.get("max_event_groups")
    if type(maximum) is not int or maximum < max(group_counts):
        raise ValueError("Finite-search group bound is inconsistent")
    modes = evidence.get("retained_modes")
    if not isinstance(modes, list) or not modes:
        raise ValueError("Finite search must identify retained modes")
    ids = []
    for mode in modes:
        if not isinstance(mode, dict) or not isinstance(mode.get("mode_id"), str) or not mode["mode_id"].strip():
            raise ValueError("Retained modes need stable identifiers")
        score, count = mode.get("score"), mode.get("event_groups")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("Retained-mode scores must be finite")
        if type(count) is not int or not 1 <= count <= maximum:
            raise ValueError("Retained-mode event counts are inconsistent")
        ids.append(mode["mode_id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Retained-mode IDs must be unique")
    assessed, mapped = diagnostics.get("assessed_mode_ids"), diagnostics.get("candidate_mode_ids")
    if not all(isinstance(values, list) and all(isinstance(v, str) for v in values) for values in (assessed, mapped)):
        return "finite_modes_not_assessed", {}
    if len(mapped) != len(candidates) or not set(ids) <= set(mapped) or set(mapped) - set(ids):
        return "finite_candidate_mapping_incomplete", {}
    by_id = {mode["mode_id"]: mode for mode in modes}
    if any(by_id[identity]["event_groups"] != count for identity, count in zip(mapped, group_counts)):
        return "finite_candidate_mapping_incomplete", {}
    returned_scores = {}
    for candidate, identity in zip(candidates, mapped):
        returned_scores[identity] = max(returned_scores.get(identity, -math.inf), candidate["score"])
    if any(mode["score"] > returned_scores[mode["mode_id"]]+1e-6 for mode in modes):
        return "finite_candidate_scores_inconsistent", {}
    if not set(ids) <= set(assessed):
        return "finite_modes_not_assessed", {}
    discarded, bound = evidence.get("discarded_count"), evidence.get("best_discarded_score")
    if type(discarded) is not int or discarded < 0:
        raise ValueError("Discarded count must be a nonnegative integer")
    if discarded:
        if isinstance(bound, bool) or not isinstance(bound, (int, float)) or not math.isfinite(bound):
            raise ValueError("Discarded candidates require a finite upper score bound")
    elif bound is not None:
        raise ValueError("No discarded candidates must have a null score bound")
    limits = evidence.get("limitations")
    if not isinstance(limits, list) or not all(isinstance(v, str) and v.strip() for v in limits):
        raise ValueError("Finite-search limitations must be a list of descriptions")
    details = {"scope": "evaluated_finite_grid", "spacing_m": evidence["spacing_m"],
               "refine_spacing_m": evidence["refine_spacing_m"], "retained_modes": len(ids),
               "discarded_count": discarded, "best_discarded_score": bound,
               "limitations": limits, "continuous_space_uniqueness": "not established"}
    if discarded:
        gap = (candidates[0]["score"]-bound)/maximum
        threshold = (policy.near_tie_support_per_group if candidates[0]["score_kind"] ==
                     "matched_group_support_minus_normalized_residual" else policy.near_tie_loss_per_event_hz)
        details["discarded_score_gap_per_group"] = gap
        if _near_tied(gap, threshold):
            return "retention_may_hide_alternatives", details
    return None, details


def _candidate_evidence_summary(events, candidate, evidence):
    """Summarize the ranked hypothesis before later gates can abstain early."""
    lookup = {event["event_id"]: event for event in events}
    matched_events = set(candidate["event_ids"])
    group_flights = {}
    flights_groups = {}
    drift_limited_groups = set()
    fit_summaries = []
    for event_id, flight_id in zip(candidate["event_ids"], candidate["flight_ids"]):
        event = lookup[event_id]
        group_id = event.get("harmonic_group_id", event_id)
        group_flights.setdefault(group_id, set()).add(flight_id)
        flights_groups.setdefault(flight_id, set()).add(group_id)
        fit = evidence.get(event_id)
        if isinstance(fit, dict):
            if fit.get("drift_at_bound") is True:
                drift_limited_groups.add(group_id)
            fit_summaries.append({
                "event_id": event_id,
                "harmonic_group_id": group_id,
                "flight_id": flight_id,
                **{name: fit[name] for name in
                   ("rmse_hz", "outlier_fraction", "support_fraction", "drift_at_bound")
                   if name in fit},
            })
    all_groups = {event.get("harmonic_group_id", event["event_id"]) for event in events}
    matched_group_count = len(group_flights)
    total_group_count = len(all_groups)
    return {
        "matched_event_ids": sorted(matched_events),
        "unmatched_event_ids": sorted(set(lookup) - matched_events),
        "matched_group_ids": sorted(group_flights),
        "unmatched_group_ids": sorted(all_groups - set(group_flights)),
        "matched_group_count": matched_group_count,
        "total_group_count": total_group_count,
        "matched_group_fraction": matched_group_count / total_group_count,
        "distinct_flight_segments": len(flights_groups),
        "flight_segments_with_multiple_groups": [
            {"flight_id": flight_id, "group_ids": sorted(group_ids)}
            for flight_id, group_ids in sorted(flights_groups.items()) if len(group_ids) > 1
        ],
        "groups_associated_with_multiple_flights": [
            {"group_id": group_id, "flight_ids": sorted(flight_ids)}
            for group_id, flight_ids in sorted(group_flights.items()) if len(flight_ids) > 1
        ],
        "drift_limited_group_ids": sorted(drift_limited_groups),
        "drift_limited_group_count": len(drift_limited_groups),
        "event_fits": fit_summaries,
    }


def decide_location(events, candidates, coverage, *, matching_diagnostics=None, policy=None):
    """Return status/reason_code/message/diagnostics without modifying inputs.

    ``matching_diagnostics`` uses W3/W5's event-ID keys, plus clock_locked,
    refinement_boundary_limited, alternatives_evaluated and sensitivity.
    A known-clock single event requires recording_start_window_s and complete
    sensitivity across that whole input window (settings.clock_bounds_s). Partial
    associations explicitly enumerate original unmatched_event_ids; these remain
    unconfirmed acoustic proposals. Sensitivity must list tested
    sources: frequency, clock, altitude and sound_speed. An altitude-only profile
    cannot justify 100 m spatial precision. The orchestrator owns execution errors
    and missing recording-clock needs_input; malformed evidence is failed here.
    """
    policy = policy or DecisionPolicy()
    if not isinstance(policy, DecisionPolicy):
        raise TypeError("policy must be a DecisionPolicy")
    diagnostics = {"policy": asdict(policy), "policy_version": "engineering-v3",
                   "calibration": "uncalibrated", "probability": None,
                   "limitations": ["Archive availability does not establish capture of every aircraft.",
                                   "Sensitivity checks are not a calibrated confidence interval."]}

    def result(reason, message, status="inconclusive"):
        return {"status": status, "reason_code": reason, "message": message, "diagnostics": diagnostics}

    try:
        envelope = {"schema_version": 1, "job_id": "decision-validation", "status": "running",
                    "events": events, "candidates": candidates}
        if coverage is not None:
            envelope["coverage"] = coverage
        validated = validate_result(envelope)
    except (ValueError, TypeError, KeyError) as exc:
        diagnostics["validation_error"] = str(exc)
        return result("invalid_evidence", "The analysis produced malformed evidence.", "failed")
    events, candidates = validated["events"], validated["candidates"]
    evidence = matching_diagnostics or {}
    if not isinstance(evidence, dict):
        return result("invalid_evidence", "Matching diagnostics must be a mapping.", "failed")

    def interval(value):
        return (isinstance(value, (list, tuple)) and len(value) == 2 and
                all(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) for v in value)
                and value[0] <= value[1])

    window = evidence.get("recording_start_window_s")
    if candidates and interval(window):
        fitted = utc_timestamp(candidates[0]["recording_start_utc"]).timestamp()
        if window[0] <= fitted <= window[1]:
            diagnostics["recording_start_window_s"] = list(window)
            diagnostics["recording_start_window_fit"] = {
                "window_width_s": float(window[1] - window[0]),
                "fit_offset_from_midpoint_s": float(fitted - (window[0] + window[1]) / 2),
                "fit_distance_to_nearest_edge_s": float(min(fitted - window[0], window[1] - fitted)),
            }
    if not events:
        return result("no_doppler_events", "No supported Doppler event was discovered in the recording.")
    if (not candidates or coverage is None or coverage["search_complete"] is not True
            or coverage["data_status"] != "available" or coverage.get("missing_data")):
        overlap_screen = _screen_harmonic_overlap(events)
        diagnostics["possible_event_harmonic_overlap_assessment"] = overlap_screen["assessment"]
        if overlap_screen["overlap"] is not None:
            diagnostics["possible_event_harmonic_overlap"] = overlap_screen["overlap"]
    if candidates:
        candidate_summary = _candidate_evidence_summary(events, candidates[0], evidence)
        diagnostics["candidate_evidence_summary"] = candidate_summary
        diagnostics["matched_event_group_count"] = candidate_summary["matched_group_count"]
        # Several acoustic groups can be assigned to the same recorded flight
        # segment. They may improve its fit, but they do not establish separate
        # flyovers for the independent-event confidence gate.
        diagnostics["independent_event_groups"] = (
            candidate_summary["distinct_flight_segments"]
            if not candidate_summary["groups_associated_with_multiple_flights"] else None)
    if coverage is None:
        return result("coverage_unknown", "Search completion and flight-data coverage are not established.")
    if coverage["data_status"] == "unavailable":
        return result("missing_flight_data", "Historical flight data are unavailable for this recording window.")
    if not coverage["search_complete"]:
        return result("search_incomplete", "Search coverage or model assessment is incomplete; ranked candidates may still be available.")
    if coverage["data_status"] != "available" or coverage.get("missing_data"):
        return result("partial_flight_coverage", "Required flight or terrain data are missing from the searched scope.")
    if not candidates:
        return result("no_matching_flights", "No candidate flight association supports the discovered events.")

    best = candidates[0]
    lookup = {event["event_id"]: event for event in events}
    matched = [lookup[key] for key in best["event_ids"]]
    groups = {}
    for event, flight in zip(matched, best["flight_ids"]):
        key = event.get("harmonic_group_id", event["event_id"])
        if key in groups and groups[key][1] != flight:
            diagnostics["independent_event_groups"] = None
            return result("inconsistent_harmonic_association", "Harmonics of one event map to different flight segments.")
        groups[key] = (event, flight)
    diagnostics["matched_event_group_count"] = len(groups)
    diagnostics["independent_event_groups"] = len({flight_id for _, flight_id in groups.values()})
    sensitivity = evidence.get("sensitivity", {})
    if window is not None:
        if not interval(window):
            return result("invalid_evidence", "The trusted recording-start window must have two finite ordered endpoints.", "failed")
        fitted = utc_timestamp(best["recording_start_utc"]).timestamp()
        if not window[0] <= fitted <= window[1]:
            return result("recording_clock_mismatch", "The fitted recording start lies outside the trusted recording window.")
    settings = sensitivity.get("settings", {}) if isinstance(sensitivity, dict) else {}
    tested_clock = settings.get("clock_bounds_s") if isinstance(settings, dict) else None
    known_clock = (interval(window) and interval(tested_clock) and
                   tested_clock[0] <= window[0] and tested_clock[1] >= window[1])
    diagnostics["known_clock_single_event"] = bool(
        len(groups) == 1 and diagnostics["independent_event_groups"] == 1 and known_clock)
    diagnostics["harmonic_overlap_screen"] = (
        "Uncalibrated aligned-sample heuristic; no flag does not establish independence.")
    overlap_screen = _screen_harmonic_overlap(matched)
    diagnostics["matched_group_overlap_assessment"] = overlap_screen["assessment"]
    if overlap_screen["overlap"] is not None:
        diagnostics["matched_group_count"] = len(groups)
        diagnostics["independent_event_groups"] = None
        diagnostics["possible_harmonic_overlap"] = overlap_screen["overlap"]
        return result("insufficient_independent_events",
                      "Overlapping matched ridges may describe one flyover; independent events are not established.")
    if not overlap_screen["assessment"]["conclusive"]:
        diagnostics["independent_event_groups"] = None
        return result("insufficient_independent_events",
                      "The bounded ridge-overlap screen did not assess all matched-group pairs; independent events are not established.")
    if diagnostics["independent_event_groups"] < policy.min_independent_events and not (len(groups) == 1 and
            policy.min_independent_events == 2 and known_clock):
        return result("insufficient_independent_events",
                      "Too few independent flyovers constrain location; harmonics and acoustic groups sharing a flight segment count once.")
    unmatched = set(lookup) - set(best["event_ids"])
    supported = evidence.get("supported_event_ids")
    if supported is not None:
        if not isinstance(supported, list) or not all(isinstance(key, str) for key in supported) or set(supported) - set(lookup):
            return result("invalid_evidence", "Supported-event identifiers must resolve to discovered events.", "failed")
        if set(supported) - set(best["event_ids"]):
            return result("unmatched_supported_events", "The hypothesis omits independently supported flyovers.")
    if unmatched:
        declared = evidence.get("unmatched_event_ids")
        if (best["score_kind"] != "matched_group_support_minus_normalized_residual" or
                not isinstance(declared, list) or not all(isinstance(key, str) for key in declared) or
                len(set(declared)) != len(declared) or set(declared) != unmatched):
            return result("unexplained_events", "Unmatched acoustic proposals need an explicit subset-association record.")
        if any(lookup[key].get("harmonic_group_id", key) in groups for key in unmatched):
            return result("incomplete_harmonic_group", "A matched flyover omits one of its measured harmonic ridges.")
    diagnostics["unmatched_event_ids"] = sorted(unmatched)
    diagnostics["unmatched_proposal_status"] = "unconfirmed acoustic proposals; not labelled aircraft misses"
    if evidence.get("clock_locked") is not True:
        return result("timing_not_established", "One shared recording clock has not been verified across all events.")
    for event, _ in groups.values():
        key = event["event_id"]
        fit = evidence.get(key)
        if not isinstance(fit, dict):
            return result("matching_diagnostics_missing", "Per-event residual and trajectory-support evidence is missing.")
        names = ("rmse_hz", "outlier_fraction", "support_fraction")
        values = [fit.get(name) for name in names]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            return result("matching_diagnostics_missing", "Per-event measurement diagnostics are incomplete or nonfinite.")
        rmse, outliers, support = values
        if rmse < 0 or not 0 <= outliers <= 1 or not 0 <= support <= 1:
            return result("invalid_evidence", "Per-event diagnostics contain impossible values.", "failed")
        if support < policy.min_support_fraction:
            return result("insufficient_track_support", "A flight trajectory does not cover enough of its observed event.")
        if rmse > policy.max_event_rmse_hz or outliers > policy.max_outlier_fraction:
            return result("inconsistent_event_match", "Residuals or outliers exceed the predeclared event-fit limits.")
        if fit.get("drift_at_bound") is not False:
            return result("source_drift_unresolved", "The source-frequency drift model is missing or reaches its allowed bound.")
        # Optional matched negative controls (stationary source, reversed
        # measurement, shifted clock). When present they must all be rejected.
        specificity = fit.get("specificity")
        if specificity is not None:
            if not isinstance(specificity, dict) or specificity.get("supported") is not True:
                diagnostics["nonspecific_event_id"] = key
                diagnostics["nonspecific_reasons"] = (list(specificity.get("reasons", []))
                                                      if isinstance(specificity, dict) else ["invalid"])
                return result("nonspecific_event_match",
                              "A matched event is explained about as well by a stationary tone, a time-reversed "
                              "measurement or a shifted clock; the flight association is not specific.")

    subset_scores = best["score_kind"] == "matched_group_support_minus_normalized_residual"
    if best["score_kind"] != "negative_robust_ridge_loss_hz" and not subset_scores:
        return result("unsupported_score_kind", "This ranking score has no declared ambiguity threshold in the decision policy.")
    finite = evidence.get("finite_search_evidence")
    if finite is not None:
        diagnostics["policy_version"] = "engineering-v5-finite-envelope"
        try:
            counts = [len({lookup[key].get("harmonic_group_id", key) for key in candidate["event_ids"]}) for candidate in candidates]
            reason, details = _finite_search(finite, evidence, candidates, counts, policy)
        except ValueError as exc:
            diagnostics["validation_error"] = str(exc)
            return result("invalid_evidence", "The finite-search evidence is malformed.", "failed")
        diagnostics["finite_search"] = details
        if reason:
            return result(reason, "The finite search does not yet exclude unassessed or discarded competing hypotheses.")
        diagnostics["limitations"].append("Only the declared finite grids/refinements were compared; continuous-space uniqueness is not established.")
    # Alternatives must fit inside the envelope around the returned point.
    # Two overlapping target-radius circles do not establish that containment.
    ambiguity_radius_m = min(policy.target_radius_m,
                             best.get("uncertainty", {}).get("radius_m", policy.target_radius_m))
    near_points = [(0, 0.0)]
    for index, alternative in enumerate(candidates[1:], 1):
        # Raw losses require equal event sets; subset support has declared units.
        if not subset_scores and set(alternative["event_ids"]) != set(best["event_ids"]):
            return result("incomparable_candidates", "Competing candidates do not explain the same event set.")
        separation = abs(Geod(ellps="WGS84").inv(best["lon"], best["lat"], alternative["lon"], alternative["lat"])[2])
        same_flights = dict(zip(best["event_ids"], best["flight_ids"])) == dict(zip(alternative["event_ids"], alternative["flight_ids"]))
        alternative_groups = {lookup[key].get("harmonic_group_id", key) for key in alternative["event_ids"]}
        gap = (best["score"] - alternative["score"]) / max(len(groups), len(alternative_groups))
        threshold = policy.near_tie_support_per_group if subset_scores else policy.near_tie_loss_per_event_hz
        units = "support_units_per_group" if subset_scores else "loss_hz_per_group"
        tied = _near_tied(gap, threshold)
        if tied and (not same_flights or (finite is None and separation > ambiguity_radius_m)):
            diagnostics["competing_hypothesis"] = {"separation_m": separation, "score_gap": gap, "gap_units": units,
                                                   "same_flight_association": same_flights, "checked_radius_m": ambiguity_radius_m}
            return result("ambiguous_locations" if separation > ambiguity_radius_m else "ambiguous_flight_match",
                          "Competing hypotheses fit nearly as well; no unique precise location is supported.")
        if tied:
            near_points.append((index, separation))
    if finite is None and evidence.get("alternatives_evaluated") is not True:
        return result("alternatives_not_assessed", "The retained competing hypotheses have not all been assessed.")
    sensitivity = evidence.get("sensitivity", {})
    if not isinstance(sensitivity, dict):
        return result("uncertainty_not_established", "Measurement and model sensitivity are not established.")
    required = {"frequency", "clock", "altitude", "sound_speed"}
    sources = sensitivity.get("sources", [])
    if not isinstance(sources, list) or not all(isinstance(x, str) for x in sources):
        return result("invalid_evidence", "Sensitivity source names must be a list of strings.", "failed")
    diagnostics["sensitivity_sources"] = sources
    if not required <= set(sources) or sensitivity.get("complete") is not True:
        return result("uncertainty_not_established", "Full ridge, clock, altitude and sound-speed sensitivity has not been tested.")
    if sensitivity.get("boundary_limited") is not False or evidence.get("refinement_boundary_limited") is not False:
        return result("refinement_boundary_limited", "Refinement or sensitivity reaches a search boundary.")
    uncertainty = best.get("uncertainty")
    if uncertainty is None:
        return result("uncertainty_not_established", "The leading candidate has no supported spatial sensitivity estimate.")
    if uncertainty["kind"] != "sensitivity":
        return result("calibration_not_verified", "This policy has no independent evidence verifying the candidate's calibration claim.")
    radius = sensitivity.get("radius_m")
    if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius < 0:
        return result("uncertainty_not_established", "The full perturbation envelope has no valid spatial radius.")
    if uncertainty["radius_m"] < radius:
        return result("uncertainty_not_established", "The candidate radius does not cover the tested perturbation envelope.")
    diagnostics["radius_m"] = uncertainty["radius_m"]
    if uncertainty["radius_m"] > policy.target_radius_m:
        return result("location_uncertainty_too_large", "Spatial sensitivity exceeds the declared precision target.")
    if finite is not None:
        base_radius = uncertainty["radius_m"]
        union_radius = max(distance + candidates[index].get("uncertainty", {}).get("radius_m", 0.)
                           for index, distance in near_points)
        union_radius = max(base_radius, union_radius)
        diagnostics["finite_envelope"] = {
            "base_profile_radius_m": radius, "base_reported_radius_m": base_radius,
            "finite_point_extent_m": max(distance for _, distance in near_points),
            "radius_m": union_radius, "included_candidate_indices": [index for index, _ in near_points],
            "included_mode_ids": sorted({evidence["candidate_mode_ids"][index] for index, _ in near_points}),
            "limitations": ["Union covers evaluated same-association points and actually reported test envelopes.",
                            "Unperformed competitor profiles/refinements remain unbounded; no coverage probability is inferred."]}
        diagnostics["radius_m"] = union_radius
        if union_radius > policy.target_radius_m:
            return result("ambiguous_locations", "Near-tied finite hypotheses and reported test envelopes exceed the precision target.")
        best["uncertainty"] = {"kind": "sensitivity", "radius_m": union_radius,
                               "method": "finite_mode_and_tested_profile_union"}
    message = ("Location passes the evaluated finite-grid and sensitivity checks; its uncertainty remains uncalibrated."
               if finite is not None else "Location passes the declared evidence checks; its sensitivity remains uncalibrated.")
    decision = result("supported_by_evidence_checks", message, "succeeded")
    if finite is not None:
        decision["candidates"] = candidates
    return decision
