from copy import deepcopy

import pytest
from pyproj import Geod

from dopplergeist.confidence import DecisionPolicy, decide_location
from dopplergeist.contracts import validate_result


def evidence():
    events = [{"event_id": f"e{i}", "start_s": i * 100, "end_s": i * 100 + 20,
               "band_hz": [90, 200], "harmonic_group_id": f"g{i}",
               "ridge": [{"time_s": i * 100 + t, "frequency_hz": 140 - t} for t in [0, 10, 20]]}
              for i in range(2)]
    candidate = {"lat": 40., "lon": -100., "score": -1., "score_kind": "negative_robust_ridge_loss_hz",
                 "event_ids": ["e0", "e1"], "flight_ids": ["a", "b"],
                 "recording_start_utc": "2026-09-21T14:13:20Z",
                 "uncertainty": {"radius_m": 70., "kind": "sensitivity", "method": "fixture_full_perturbation"}}
    coverage = {"search_complete": True, "data_status": "available", "limitations": ["Aircraft capture is incomplete."]}
    diagnostic = {"clock_locked": True, "alternatives_evaluated": True, "refinement_boundary_limited": False,
                  "sensitivity": {"complete": True, "boundary_limited": False, "radius_m": 70.,
                                  "sources": ["frequency", "clock", "altitude", "sound_speed"]}}
    for event in events:
        diagnostic[event["event_id"]] = {"rmse_hz": 0.7, "outlier_fraction": 0.1,
                                         "support_fraction": 1.0, "drift_at_bound": False}
    return events, [candidate], coverage, diagnostic


def decide(bundle):
    events, candidates, coverage, diagnostic = bundle
    return decide_location(events, candidates, coverage, matching_diagnostics=diagnostic)


def test_complete_fixture_passes_without_probability_and_preserves_inputs():
    bundle = evidence()
    before = deepcopy(bundle)
    result = decide(bundle)
    assert result["status"] == "succeeded"
    assert result["diagnostics"]["policy_version"] == "engineering-v3"
    assert result["diagnostics"]["probability"] is None
    assert result["diagnostics"]["calibration"] == "uncalibrated"
    assert bundle == before
    wire = {"schema_version": 1, "job_id": "example", "events": bundle[0], "candidates": bundle[1],
            "coverage": bundle[2], **{k: v for k, v in result.items() if k != "diagnostics"}}
    assert validate_result(wire)["status"] == "succeeded"


@pytest.mark.parametrize("kind,reason", [
    ("noise", "no_doppler_events"), ("missing", "missing_flight_data"),
    ("partial", "partial_flight_coverage"), ("incomplete", "search_incomplete"),
    ("harmonics", "insufficient_independent_events"), ("clock", "timing_not_established"),
    ("residual", "inconsistent_event_match"), ("support", "insufficient_track_support"),
    ("drift", "source_drift_unresolved"), ("altitude_only", "uncertainty_not_established"),
    ("no_alternatives", "alternatives_not_assessed"), ("wide", "location_uncertainty_too_large"),
    ("boundary", "refinement_boundary_limited"), ("no_tracks", "no_matching_flights")])
def test_insufficient_evidence_abstains(kind, reason):
    events, candidates, coverage, diag = evidence()
    if kind == "noise": events, candidates = [], []
    elif kind == "missing": coverage["data_status"] = "unavailable"
    elif kind == "partial": coverage["data_status"] = "partial"
    elif kind == "incomplete": coverage["search_complete"] = False
    elif kind == "harmonics":
        events[1]["harmonic_group_id"] = events[0]["harmonic_group_id"]
        candidates[0]["flight_ids"] = ["a", "a"]
    elif kind == "clock": diag["clock_locked"] = False
    elif kind == "residual": diag["e0"]["rmse_hz"] = 8
    elif kind == "support": diag["e0"]["support_fraction"] = 0.5
    elif kind == "drift": diag["e0"]["drift_at_bound"] = True
    elif kind == "altitude_only": diag["sensitivity"]["sources"] = ["altitude"]
    elif kind == "no_alternatives": diag["alternatives_evaluated"] = False
    elif kind == "wide": candidates[0]["uncertainty"]["radius_m"] = 1500
    elif kind == "boundary": diag["sensitivity"]["boundary_limited"] = True
    elif kind == "no_tracks": candidates = []
    result = decide((events, candidates, coverage, diag))
    assert result["status"] == "inconclusive"
    assert result["reason_code"] == reason


def test_distant_tie_and_same_location_different_flights():
    events, candidates, coverage, diag = evidence()
    alternative = dict(candidates[0], lat=45., score=-1.2)
    result = decide((events, candidates + [alternative], coverage, diag))
    assert result["reason_code"] == "ambiguous_locations"
    alternative.update(lat=40., flight_ids=["x", "y"])
    assert decide((events, candidates + [alternative], coverage, diag))["reason_code"] == "ambiguous_flight_match"


@pytest.mark.parametrize("separation_m,score,status", [
    (60., -1.2, "succeeded"),
    (90., -1.2, "inconclusive"),
    (150., -1.2, "inconclusive"),
    (90., -5., "succeeded"),
])
def test_near_tied_locations_fit_the_returned_envelope(separation_m, score, status):
    events, candidates, coverage, diag = evidence()
    best = candidates[0]  # The returned radius is 70 m around this point.
    lon, lat, _ = Geod(ellps="WGS84").fwd(best["lon"], best["lat"], 90., separation_m)
    alternative = dict(best, lon=lon, lat=lat, score=score)
    result = decide((events, candidates + [alternative], coverage, diag))
    assert result["status"] == status
    if status == "inconclusive":
        assert result["reason_code"] == "ambiguous_locations"


def test_malformed_results_fail_instead_of_masquerading_as_insufficient_evidence():
    events, candidates, coverage, diag = evidence()
    candidates[0]["score"] = float("nan")
    result = decide((events, candidates, coverage, diag))
    assert result["status"] == "failed"
    assert result["reason_code"] == "invalid_evidence"
    with pytest.raises(ValueError):
        DecisionPolicy(min_independent_events=1)


def test_w3_vertical_only_profile_is_not_a_100m_confidence_claim():
    events, candidates, coverage, diag = evidence()
    diag["sensitivity"] = {"radius_m": 0.0, "kind": "sensitivity", "method": "profiled_relative_altitude_offsets",
                           "complete": True, "boundary_limited": False, "profiles": []}
    assert decide((events, candidates, coverage, diag))["reason_code"] == "uncertainty_not_established"


def test_known_clock_single_event_requires_full_trusted_window_sensitivity():
    from dopplergeist.contracts import utc_timestamp
    events, candidates, coverage, diag = evidence()
    events = events[:1]
    candidates[0].update(event_ids=["e0"], flight_ids=["a"])
    start = utc_timestamp(candidates[0]["recording_start_utc"]).timestamp()
    assert decide((events, candidates, coverage, diag))["reason_code"] == "insufficient_independent_events"
    diag["recording_start_window_s"] = [start-1, start+1]
    diag["sensitivity"]["settings"] = {"clock_bounds_s": [start, start]}
    assert decide((events, candidates, coverage, diag))["reason_code"] == "insufficient_independent_events"
    diag["sensitivity"]["settings"]["clock_bounds_s"] = [start-1, start+1]
    result = decide((events, candidates, coverage, diag))
    assert result["status"] == "succeeded"
    assert result["diagnostics"]["known_clock_single_event"] is True
    diag["recording_start_window_s"] = [start+1, start+2]
    assert decide((events, candidates, coverage, diag))["reason_code"] == "recording_clock_mismatch"


def test_incomplete_decision_reports_clock_window_fit_without_changing_reason():
    from dopplergeist.contracts import utc_timestamp
    events, candidates, coverage, diag = evidence()
    coverage["search_complete"] = False
    start = utc_timestamp(candidates[0]["recording_start_utc"]).timestamp()
    baseline = decide_location(events, candidates, coverage, matching_diagnostics=diag)
    before = deepcopy(candidates)
    assert "recording_start_window_fit" not in baseline["diagnostics"]

    diag["recording_start_window_s"] = [start - 20, start + 20]
    result = decide_location(events, candidates, coverage, matching_diagnostics=diag)

    assert baseline["status"] == result["status"] == "inconclusive"
    assert baseline["reason_code"] == result["reason_code"] == "search_incomplete"
    assert result["diagnostics"]["recording_start_window_fit"] == {
        "window_width_s": 40.0,
        "fit_offset_from_midpoint_s": 0.0,
        "fit_distance_to_nearest_edge_s": 20.0,
    }
    assert candidates == before


@pytest.mark.parametrize("side,expected_offset", [("lower", -20.0), ("upper", 20.0)])
def test_clock_window_fit_reports_exact_edges(side, expected_offset):
    from dopplergeist.contracts import utc_timestamp
    events, candidates, coverage, diag = evidence()
    coverage["search_complete"] = False
    start = utc_timestamp(candidates[0]["recording_start_utc"]).timestamp()
    diag["recording_start_window_s"] = (
        [start, start + 40] if side == "lower" else [start - 40, start]
    )

    result = decide_location(events, candidates, coverage, matching_diagnostics=diag)

    assert result["reason_code"] == "search_incomplete"
    assert result["diagnostics"]["recording_start_window_fit"] == {
        "window_width_s": 40.0,
        "fit_offset_from_midpoint_s": expected_offset,
        "fit_distance_to_nearest_edge_s": 0.0,
    }


def test_explicit_subset_keeps_unconfirmed_proposals_without_forcing_matches():
    events, candidates, coverage, diag = evidence()
    extra = deepcopy(events[1])
    extra.update(event_id="e2", harmonic_group_id="g2")
    events.append(extra)
    candidates[0].update(score_kind="matched_group_support_minus_normalized_residual", score=1.8)
    assert decide((events, candidates, coverage, diag))["reason_code"] == "unexplained_events"
    diag["unmatched_event_ids"] = ["e2"]
    result = decide((events, candidates, coverage, diag))
    assert result["status"] == "succeeded"
    assert result["diagnostics"]["unmatched_event_ids"] == ["e2"]
    diag["supported_event_ids"] = ["e0", "e1", "e2"]
    assert decide((events, candidates, coverage, diag))["reason_code"] == "unmatched_supported_events"
    diag["supported_event_ids"] = ["e0", "e1"]
    alternative = dict(candidates[0], lat=45., event_ids=["e0", "e2"], flight_ids=["a", "c"], score=1.7)
    assert decide((events, candidates+[alternative], coverage, diag))["reason_code"] == "ambiguous_locations"
    alternative["score"] = 0.5
    assert decide((events, candidates+[alternative], coverage, diag))["status"] == "succeeded"
    extra["harmonic_group_id"] = "g0"
    assert decide((events, candidates, coverage, diag))["reason_code"] == "incomplete_harmonic_group"


def test_all_four_independent_flyovers_are_checked():
    events, candidates, coverage, diag = evidence()
    for i in (2, 3):
        event = deepcopy(events[0])
        event.update(event_id=f"e{i}", harmonic_group_id=f"g{i}", start_s=i*100, end_s=i*100+20)
        event["ridge"] = [dict(point, time_s=point["time_s"]+i*100) for point in event["ridge"]]
        events.append(event)
        candidates[0]["event_ids"].append(event["event_id"])
        candidates[0]["flight_ids"].append(f"flight-{i}")
        diag[event["event_id"]] = deepcopy(diag["e0"])
    assert decide((events, candidates, coverage, diag))["status"] == "succeeded"
    diag["e3"]["rmse_hz"] = 8
    assert decide((events, candidates, coverage, diag))["reason_code"] == "inconsistent_event_match"


def finite_evidence():
    events, candidates, coverage, diag = evidence()
    candidates.append(dict(candidates[0], lat=45., score=-4.))
    diag.update(alternatives_evaluated=False, assessed_mode_ids=["a", "b"], candidate_mode_ids=["a", "b"],
                finite_search_evidence={"schema_version": 1, "scope": "evaluated_finite_grid", "complete": True,
                    "score_kind": candidates[0]["score_kind"], "spacing_m": 1000., "refine_spacing_m": 100.,
                    "max_event_groups": 2, "retained_modes": [{"mode_id": "a", "score": -1., "event_groups": 2},
                        {"mode_id": "b", "score": -4., "event_groups": 2}],
                    "discarded_count": 7, "best_discarded_score": -5., "pruning_accounted_for": True,
                    "limitations": ["Evaluated grids only; no continuous-space uniqueness."]})
    return events, candidates, coverage, diag


def test_finite_search_can_pass_without_claiming_continuous_exhaustion():
    bundle = finite_evidence()
    result = decide(bundle)
    assert result["status"] == "succeeded"
    assert result["diagnostics"]["policy_version"] == "engineering-v5-finite-envelope"
    assert result["diagnostics"]["finite_search"]["continuous_space_uniqueness"] == "not established"
    bundle[3]["sensitivity"]["sources"] = ["altitude"]
    assert decide(bundle)["reason_code"] == "uncertainty_not_established"


@pytest.mark.parametrize("kind,reason", [("discard", "retention_may_hide_alternatives"),
    ("complete", "finite_search_incomplete"), ("pruning", "finite_search_incomplete"),
    ("assessed", "finite_modes_not_assessed"), ("mapping", "finite_candidate_mapping_incomplete"),
    ("lost_score", "finite_candidate_scores_inconsistent")])
def test_finite_search_cannot_hide_omitted_or_near_tied_modes(kind, reason):
    bundle = finite_evidence()
    diag = bundle[3]
    finite = diag["finite_search_evidence"]
    diag["alternatives_evaluated"] = True  # Optimistic legacy flag cannot override evidence.
    if kind == "discard": finite["best_discarded_score"] = -1.2
    elif kind == "complete": finite["complete"] = False
    elif kind == "pruning": finite["pruning_accounted_for"] = False
    elif kind == "assessed": diag["assessed_mode_ids"] = ["a"]
    elif kind == "mapping": diag["candidate_mode_ids"] = ["a", "a"]
    elif kind == "lost_score": finite["retained_modes"][1]["score"] = 0.
    result = decide(bundle)
    assert result["status"] == "inconclusive"
    assert result["reason_code"] == reason


def test_finite_retention_bound_uses_declared_subset_units():
    events, candidates, coverage, diag = finite_evidence()
    kind = "matched_group_support_minus_normalized_residual"
    candidates[0].update(score=1.8, score_kind=kind)
    candidates[1].update(score=0.1, score_kind=kind)
    finite = diag["finite_search_evidence"]
    finite.update(score_kind=kind, best_discarded_score=1.5)
    for mode, score in zip(finite["retained_modes"], (1.8, 0.1)): mode["score"] = score
    assert decide((events, candidates, coverage, diag))["reason_code"] == "retention_may_hide_alternatives"
    finite["best_discarded_score"] = 1.2
    assert decide((events, candidates, coverage, diag))["status"] == "succeeded"


def test_malformed_finite_flags_fail_instead_of_looking_like_insufficient_data():
    bundle = finite_evidence()
    bundle[3]["finite_search_evidence"]["complete"] = 1
    result = decide(bundle)
    assert result["status"] == "failed"
    assert result["reason_code"] == "invalid_evidence"


def near_finite_points(distance_m=90.):
    from pyproj import Geod
    bundle = finite_evidence()
    best, alternative = bundle[1]
    alternative["lon"], alternative["lat"], _ = Geod(ellps="WGS84").fwd(best["lon"], best["lat"], 90., distance_m)
    alternative["score"] = -1.2
    bundle[3]["finite_search_evidence"]["retained_modes"][1]["score"] = -1.2
    return bundle


def test_finite_near_cluster_returns_a_copied_honest_envelope():
    bundle = near_finite_points()
    before = deepcopy(bundle)
    result = decide(bundle)
    assert result["status"] == "succeeded"
    returned = result["candidates"]
    assert 159.999 < returned[0]["uncertainty"]["radius_m"] < 160.001
    assert returned[0]["uncertainty"]["kind"] == "sensitivity"
    assert returned[0]["uncertainty"]["method"] == "finite_mode_and_tested_profile_union"
    assert result["diagnostics"]["finite_envelope"]["included_mode_ids"] == ["a", "b"]
    for original, candidate in zip(before[1], returned):
        assert {k: v for k, v in original.items() if k != "uncertainty"} == {k: v for k, v in candidate.items() if k != "uncertainty"}
    assert bundle == before
    assert validate_result({"schema_version": 1, "job_id": "finite-envelope", "status": "succeeded",
                            "events": bundle[0], "coverage": bundle[2], "candidates": returned})["candidates"] == returned


def test_finite_union_keeps_distant_and_different_association_vetoes():
    distant = near_finite_points(2000.)
    assert decide(distant)["reason_code"] == "ambiguous_locations"
    different = near_finite_points(30.)
    different[1][1]["flight_ids"] = ["x", "y"]
    assert decide(different)["reason_code"] == "ambiguous_flight_match"
    narrow = near_finite_points()
    result = decide_location(*narrow[:3], matching_diagnostics=narrow[3], policy=DecisionPolicy(target_radius_m=100.))
    assert result["status"] == "inconclusive"
    assert result["reason_code"] == "ambiguous_locations"


@pytest.mark.parametrize("kind,reason", [("incomplete", "uncertainty_not_established"),
    ("boundary", "refinement_boundary_limited"), ("missing_sources", "uncertainty_not_established"),
    ("missing_uncertainty", "uncertainty_not_established"), ("ledger", "finite_search_incomplete")])
def test_union_cannot_repair_missing_or_clipped_leading_evidence(kind, reason):
    bundle = near_finite_points()
    if kind == "incomplete": bundle[3]["sensitivity"]["complete"] = False
    elif kind == "boundary": bundle[3]["sensitivity"]["boundary_limited"] = True
    elif kind == "missing_sources": bundle[3]["sensitivity"]["sources"] = ["altitude"]
    elif kind == "missing_uncertainty": bundle[1][0].pop("uncertainty")
    elif kind == "ledger": bundle[3]["finite_search_evidence"]["complete"] = False
    result = decide(bundle)
    assert result["status"] == "inconclusive"
    assert result["reason_code"] == reason


@pytest.mark.parametrize("specificity,status,reason", [
    ({"supported": True, "reasons": []}, "succeeded", "supported_by_evidence_checks"),
    ({"supported": False, "reasons": ["reversed_measurement_not_rejected"]}, "inconclusive", "nonspecific_event_match"),
    ("not-a-mapping", "inconclusive", "nonspecific_event_match"),
])
def test_matched_negative_controls_gate_acceptance(specificity, status, reason):
    events, candidates, coverage, diag = evidence()
    diag["e1"]["specificity"] = specificity
    result = decide((events, candidates, coverage, diag))
    assert result["status"] == status
    assert result.get("reason_code") == reason
    if status == "inconclusive":
        assert result["diagnostics"]["nonspecific_event_id"] == "e1"
