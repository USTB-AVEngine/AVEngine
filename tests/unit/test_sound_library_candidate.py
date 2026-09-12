"""Incremental sound-library overlay, review inheritance and segment-selection contract.

Fixtures build a small library on disk with the same sidecar shape the real library
uses, so the admission and identity rules are exercised without touching the real
recordings.
"""
from __future__ import annotations

import json
from pathlib import Path
import wave

import pytest

from avengine.dataset import source_capabilities as mod

pytestmark = pytest.mark.fast_unit


def write_clip(
    root: Path,
    relative: str,
    *,
    classes=("dog_bark",),
    qc_verdict="pass",
    findings=(),
    duration_s=2.0,
    human_review=None,
    with_sidecars=True,
) -> Path:
    """Write one library clip with its clip.json and clip.qc.json sidecars."""
    folder = root / relative
    folder.mkdir(parents=True, exist_ok=True)
    audio = folder / mod.CLIP_AUDIO_NAME
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * int(duration_s * 16000))
    if not with_sidecars:
        return audio
    sidecar = {"event_classes": list(classes), "source": "unit fixture",
               "license": "fixture", "dry": True}
    if human_review is not None:
        sidecar["human_review"] = human_review
    (folder / mod.CLIP_SIDECAR_NAME).write_text(json.dumps(sidecar), encoding="utf-8")
    (folder / mod.CLIP_QC_SIDECAR_NAME).write_text(json.dumps({
        "schema": "avengine_sound_clip_qc_v1",
        "verdict": qc_verdict,
        "findings": list(findings),
        "sample_rate_hz": 16000,
        "channel_count": 1,
        "measured": {"duration_s": duration_s, "active_frame_ratio": 0.8,
                     "continuous": False, "peak": 0.3, "rms_dbfs": -18.0,
                     "noise_floor_dbfs": -60.0, "decay_to_minus20db_s": 0.2},
    }), encoding="utf-8")
    return audio


HUMAN_PASS = {
    "status": "pass",
    "machine_qc_is_reference_only": True,
    "event_classes": {"dog_bark": {"verdict": "pass", "author": "fixture_listener",
                                   "note": "人工抽听暂判合格"}},
}
CLIPPING_FAIL = ({"name": "clipping", "severity": "fail",
                  "reason_zh": "削波严重"},)


# ------------------------------------------------------------------------- roots


def test_roots_are_declared_and_ordered_by_priority(tmp_path):
    config = {"sound_library_roots": [
        {"root": str(tmp_path / "base"), "priority": 10, "role": "base"},
        {"root": str(tmp_path / "new"), "priority": 20, "role": "incremental"},
    ]}
    roots = mod.normalize_library_roots(config)
    assert [entry["role"] for entry in roots] == ["incremental", "base"]
    assert mod.normalize_library_roots({}) == []
    assert mod.normalize_library_roots({"sound_library_roots": ["/a/b"]})[0]["root"] == "/a/b"
    with pytest.raises(mod.SourceCapabilityError, match="must be a list"):
        mod.normalize_library_roots({"sound_library_roots": "/a/b"})
    with pytest.raises(mod.SourceCapabilityError, match="must declare"):
        mod.normalize_library_roots({"sound_library_roots": [{"priority": 1}]})


def test_higher_priority_root_wins_and_the_shadowed_copy_is_recorded(tmp_path):
    base, new = tmp_path / "base", tmp_path / "new"
    write_clip(base, "dog_bark/shared", duration_s=2.0)
    write_clip(new, "dog_bark/shared", duration_s=2.0)
    write_clip(new, "dog_bark/only_new", duration_s=1.0)
    overlay = mod.load_sound_library_overlay({"sound_library_roots": [
        {"root": str(new), "priority": 20, "role": "incremental"},
        {"root": str(base), "priority": 10, "role": "base"},
    ]})
    assert overlay["clip_count"] == 2
    assert overlay["clips"]["dog_bark/shared/clip.wav"]["root"] == str(new)
    assert len(overlay["shadowed"]) == 1
    assert overlay["shadowed"][0]["shadowed_root"] == str(base)
    assert overlay["identity_key"] == "root_relative_path"


def test_a_missing_root_is_reported_rather_than_guessed(tmp_path):
    write_clip(tmp_path / "new", "dog_bark/a")
    overlay = mod.load_sound_library_overlay({"sound_library_roots": [
        {"root": str(tmp_path / "new"), "priority": 20},
        {"root": str(tmp_path / "absent"), "priority": 10},
    ]})
    summaries = overlay["root_summary"]
    assert summaries[str(tmp_path / "absent")]["status"] == "missing"
    assert summaries[str(tmp_path / "new")]["clip_count"] == 1
    with pytest.raises(mod.SourceCapabilityError, match="no sound library root"):
        mod.load_sound_library_overlay({})


# -------------------------------------------------------------------- admission


def test_a_machine_fail_is_never_overridden_by_a_blanket_human_pass(tmp_path):
    """The real library stamps human pass on every inherited clip, fails included."""
    root = tmp_path / "new"
    write_clip(root, "dog_bark/broken", qc_verdict="fail", findings=CLIPPING_FAIL,
               human_review=HUMAN_PASS)
    write_clip(root, "dog_bark/fine", qc_verdict="pass", human_review=HUMAN_PASS)
    overlay = mod.load_sound_library_overlay(
        {"sound_library_roots": [{"root": str(root), "priority": 1}]})
    broken = overlay["clips"]["dog_bark/broken/clip.wav"]["admission"]
    fine = overlay["clips"]["dog_bark/fine/clip.wav"]["admission"]
    assert broken["state"] == mod.ADMISSION_BLOCKED_MACHINE_FAIL
    assert fine["state"] == mod.ADMISSION_CANDIDATE
    # both records survive side by side
    assert broken["basis"]["human_review_status"] == "pass"
    assert broken["basis"]["machine_qc_verdict"] == "fail"
    assert [f["name"] for f in broken["basis"]["machine_qc_blocking_findings"]] == ["clipping"]
    assert broken["basis"]["precedence"] == "machine_fail_is_never_overridden_by_a_human_pass"
    assert broken["basis"]["human_scope"] == "original_recording_only"


def test_a_blocking_finding_alone_blocks_even_when_the_verdict_is_not_fail(tmp_path):
    root = tmp_path / "new"
    write_clip(root, "dog_bark/mixed", qc_verdict="warn", findings=CLIPPING_FAIL,
               human_review=HUMAN_PASS)
    overlay = mod.load_sound_library_overlay(
        {"sound_library_roots": [{"root": str(root), "priority": 1}]})
    admission = overlay["clips"]["dog_bark/mixed/clip.wav"]["admission"]
    assert admission["state"] == mod.ADMISSION_BLOCKED_MACHINE_FAIL


def test_missing_sidecars_and_undeclared_classes_are_separate_blocks(tmp_path):
    root = tmp_path / "new"
    write_clip(root, "dog_bark/no_sidecar", with_sidecars=False)
    write_clip(root, "dog_bark/no_class", classes=())
    overlay = mod.load_sound_library_overlay(
        {"sound_library_roots": [{"root": str(root), "priority": 1}]})
    states = {relative: clip["admission"]["state"]
              for relative, clip in overlay["clips"].items()}
    assert states["dog_bark/no_sidecar/clip.wav"] == mod.ADMISSION_BLOCKED_MISSING_SIDECAR
    assert states["dog_bark/no_class/clip.wav"] == mod.ADMISSION_BLOCKED_NO_DECLARED_CLASS
    assert set(states.values()) <= set(mod.ADMISSION_STATES)


def test_human_review_absent_is_kept_as_absent(tmp_path):
    root = tmp_path / "new"
    write_clip(root, "dog_bark/unreviewed")
    overlay = mod.load_sound_library_overlay(
        {"sound_library_roots": [{"root": str(root), "priority": 1}]})
    clip = overlay["clips"]["dog_bark/unreviewed/clip.wav"]
    assert clip["human_review"] == {}
    assert clip["admission"]["basis"]["human_review_status"] is None
    # an unreviewed clip is still a machine candidate; it is simply not human-passed
    assert clip["admission"]["state"] == mod.ADMISSION_CANDIDATE


# --------------------------------------------------------------------- identity


def test_the_same_recording_under_a_new_root_keeps_its_registered_identity(tmp_path):
    root = tmp_path / "new"
    write_clip(root, "dog_bark/registered")
    write_clip(root, "dog_bark/fresh")
    overlay = mod.load_sound_library_overlay(
        {"sound_library_roots": [{"root": str(root), "priority": 1}]})
    manifest = {"library_root": "/somewhere/prepared", "clips": [
        {"source": "dog_bark/registered/clip.wav", "sound_asset_id": "sound_a_v1",
         "event_class": "dog_bark"},
        {"source": "dog_bark/registered/clip.wav", "sound_asset_id": "sound_b_v1",
         "event_class": "dog_bark"},
        {"source": "dog_bark/gone/clip.wav", "sound_asset_id": "sound_c_v1",
         "event_class": "dog_bark"},
    ]}
    registered = mod.registered_source_index(manifest)
    assert registered["registered_relative_count"] == 2
    identity = mod.resolve_library_identity(overlay, registered)
    assert identity["same_source_as_registered"] == ["dog_bark/registered/clip.wav"]
    assert identity["new_candidate"] == ["dog_bark/fresh/clip.wav"]
    assert identity["registered_source_absent_from_library"] == ["dog_bark/gone/clip.wav"]
    # two registered assets share one recording; the recording is still counted once
    assert identity["counts"][mod.IDENTITY_SAME_SOURCE] == 1
    assert identity["registered_sound_asset_ids_reused"] == 2
    assert identity["identity_key"] == "root_relative_path"


def test_registered_source_index_requires_clips(tmp_path):
    with pytest.raises(mod.SourceCapabilityError, match="no clips list"):
        mod.registered_source_index({"library_root": "/x"})


# ----------------------------------------------------------------------- budget


def test_the_program_span_is_derived_from_declared_episode_and_reserve():
    budget = mod.segment_budget(episode_s=10.0, reserve_tail_s=3.0)
    assert budget["max_single_program_span_s"] == pytest.approx(7.0)
    assert budget["latest_program_end_s"] == pytest.approx(7.0)
    later = mod.segment_budget(episode_s=10.0, reserve_tail_s=3.0, earliest_start_s=1.0)
    assert later["max_single_program_span_s"] == pytest.approx(6.0)
    # the reserve is never shortened to make room
    assert later["reserve_tail_s"] == pytest.approx(3.0)
    assert "not a per-clip maximum" in budget["claim_boundary"]
    with pytest.raises(mod.SourceCapabilityError, match="no room"):
        mod.segment_budget(episode_s=3.0, reserve_tail_s=3.0)
    with pytest.raises(mod.SourceCapabilityError, match="must not be negative"):
        mod.segment_budget(episode_s=10.0, reserve_tail_s=-1.0)


def test_activity_requirements_come_from_configuration_and_fail_closed():
    assert mod.segment_requirements("device_continuous")["state"] == mod.STATE_EVIDENCE_MISSING
    config = {"segment_activity_requirements": {"device_continuous": {
        "minimum_activity_coverage": 0.9, "maximum_internal_silence_s": 0.2,
        "minimum_segment_s": 1.0}}}
    resolved = mod.segment_requirements("device_continuous", config)
    assert resolved["state"] == mod.STATE_AVAILABLE
    assert resolved["minimum_activity_coverage"] == 0.9
    assert mod.segment_requirements("speech", config)["state"] == mod.STATE_EVIDENCE_MISSING


# --------------------------------------------------- segment selection contract


def test_the_request_carries_the_class_activity_profile_and_the_forbidden_transforms(tmp_path):
    root = tmp_path / "new"
    write_clip(root, "dog_bark/a", duration_s=9.0)
    overlay = mod.load_sound_library_overlay(
        {"sound_library_roots": [{"root": str(root), "priority": 1}]})
    clip = overlay["clips"]["dog_bark/a/clip.wav"]
    budget = mod.segment_budget(episode_s=10.0, reserve_tail_s=3.0)
    request = mod.segment_selection_request(clip, sound_class="dog_bark", budget=budget)
    assert request["schema"] == mod.SEGMENT_SELECTION_REQUEST_SCHEMA
    assert request["activity_profile"]["activity_family"] == "animal_call"
    assert request["source_duration_s"] == 9.0
    assert "contiguous_segment_selection" in request["authorization"]["allows"]
    for forbidden in ("time_stretch", "wet_tail_truncation", "per_sample_level_matching",
                      "discontiguous_splicing"):
        assert forbidden in request["authorization"]["forbids"]
    assert request["authorization"]["original_pcm"] == "read_only"
    assert request["requirements"]["state"] == mod.STATE_EVIDENCE_MISSING


def selection(**overrides):
    """A row shaped like ``sound_segments.pool_row`` actually emits.

    The three coordinate systems are physically coherent here: a 3 s recording at
    44.1 kHz, measured on its 16 kHz resampling, delivered as a 16 kHz segment.
    """
    base = {
        "relative_path": "dog_bark/a/clip.wav",
        # original recording, 44.1 kHz
        "source_rate_hz": 44100,
        "source_sample_count": 132300,
        "source_crop_start_sample": 0,
        "source_crop_end_sample_exclusive": 132300,
        # resampled whole recording P25 measures on, 16 kHz
        "analysis_sample_count": 48000,
        "origin_activity_intervals_samples": [[0, 46000]],
        # delivered segment, 16 kHz (pool_row publishes segment coords under this name)
        "sample_count": 48000,
        "sample_rate_hz": 16000,
        "source_activity_intervals_samples": [[0, 46000]],
        "selection_authorized": True,
        "crop_authorization": "owner_authorized_activity_segment_selection_20260910",
        "truncated": False,
        "activity_coverage": 0.95,
        "max_internal_silence_s": 0.1,
    }
    base.update(overrides)
    return base


def test_a_declared_selection_must_be_bounded_measured_and_authorized():
    assert mod.verify_segment_selection(selection())["verified"] is True
    for field in ("crop_authorization", "relative_path"):
        broken = selection()
        broken.pop(field)
        assert mod.verify_segment_selection(broken)["verified"] is False
    # a row that is not marked authorized is refused even though it is well formed
    unmarked = mod.verify_segment_selection(selection(selection_authorized=False))
    assert not unmarked["verified"]
    assert "selection_is_not_marked_authorized" in unmarked["problems"]
    # a row claiming an authorized crop and a truncation at once is contradictory
    both = mod.verify_segment_selection(selection(truncated=True))
    assert "row_claims_both_authorized_selection_and_truncation" in both["problems"]
    # an end at or before the start is an empty crop
    assert not mod.verify_segment_selection(
        selection(source_crop_end_sample_exclusive=0))["verified"]
    # a crop reaching past the recording it claims to come from is refused
    past_end = mod.verify_segment_selection(
        selection(source_crop_end_sample_exclusive=999999))
    assert "source_crop_end_exceeds_source_sample_count" in past_end["problems"]
    assert not mod.verify_segment_selection(
        selection(activity_coverage=None))["verified"]
    assert not mod.verify_segment_selection(
        selection(source_activity_intervals_samples=[],
                  origin_activity_intervals_samples=[]))["verified"]
    mismatched = mod.verify_segment_selection(
        selection(), clip={"relative": "dog_bark/other/clip.wav"})
    assert not mismatched["verified"]
    assert "selection_does_not_match_the_requested_clip" in mismatched["problems"]
    with pytest.raises(mod.SourceCapabilityError):
        mod.verify_segment_selection("not an object")


def test_an_absolute_source_path_still_matches_its_library_relative():
    absolute = mod.verify_segment_selection(
        selection(relative_path=None,
                  source_path="/data/library/dog_bark/a/clip.wav"),
        clip={"relative": "dog_bark/a/clip.wav"})
    assert absolute["verified"] is True, absolute["problems"]


def test_configured_activity_thresholds_are_actually_enforced():
    requirements = mod.segment_requirements("animal_call", {
        "segment_activity_requirements": {"animal_call": {
            "minimum_activity_coverage": 0.9, "maximum_internal_silence_s": 0.2}}})
    ok = mod.verify_segment_selection(selection(), requirements=requirements)
    assert ok["verified"] is True

    thin = mod.verify_segment_selection(
        selection(activity_coverage=0.4), requirements=requirements)
    assert "activity_coverage_below_configured_minimum" in thin["problems"]

    gappy = mod.verify_segment_selection(
        selection(max_internal_silence_s=3.0), requirements=requirements)
    assert "internal_silence_exceeds_configured_maximum" in gappy["problems"]

    unconfigured = mod.verify_segment_selection(
        selection(), requirements=mod.segment_requirements("animal_call"))
    assert "activity_requirement_is_not_configured" in unconfigured["problems"]


def test_admitted_clips_are_handed_to_p25_in_the_shape_it_consumes(tmp_path):
    root = tmp_path / "new"
    write_clip(root, "dog_bark/keep", duration_s=9.0)
    write_clip(root, "dog_bark/registered", duration_s=4.0)
    write_clip(root, "dog_bark/broken", duration_s=9.0, qc_verdict="fail",
               findings=CLIPPING_FAIL, human_review=HUMAN_PASS)
    overlay = mod.load_sound_library_overlay(
        {"sound_library_roots": [{"root": str(root), "priority": 1}]})
    identity = mod.resolve_library_identity(overlay, mod.registered_source_index(
        {"clips": [{"source": "dog_bark/registered/clip.wav",
                    "sound_asset_id": "sound_x_v1", "event_class": "dog_bark"}]}))

    everything = mod.library_segment_sources(overlay, identity)
    assert {row["relative_path"] for row in everything} == {
        "dog_bark/keep/clip.wav", "dog_bark/registered/clip.wav"}
    # the machine-failed clip is never offered for cropping
    assert all("broken" not in row["relative_path"] for row in everything)
    for row in everything:
        assert set(row) >= {"source_path", "relative_path", "source_asset_id",
                            "sound_class", "declared_event_classes"}
        assert row["sound_class"] == "dog_bark"
        assert row["source_asset_id"].startswith("dog_bark/")

    incremental = mod.library_segment_sources(
        overlay, identity, include_registered=False)
    assert [row["relative_path"] for row in incremental] == ["dog_bark/keep/clip.wav"]
    assert incremental[0]["already_registered"] is False


# ------------------------------------------------------------ candidate report


def test_the_four_counts_stay_separate_and_qualified_needs_a_verified_selection(tmp_path):
    root = tmp_path / "new"
    write_clip(root, "dog_bark/short", duration_s=2.0)
    write_clip(root, "dog_bark/long", duration_s=9.0)
    write_clip(root, "dog_bark/broken", duration_s=9.0, qc_verdict="fail",
               findings=CLIPPING_FAIL, human_review=HUMAN_PASS)
    config = {"sound_library_roots": [{"root": str(root), "priority": 1}],
              "segment_activity_requirements": {"animal_call": {
                  "minimum_activity_coverage": 0.9}}}
    overlay = mod.load_sound_library_overlay(config)
    identity = mod.resolve_library_identity(overlay, {"sound_asset_ids_by_relative": {}})
    budget = mod.segment_budget(episode_s=10.0, reserve_tail_s=3.0)

    report = mod.library_candidate_report(overlay, identity, budget=budget, config=config)
    counts = report["counts"]
    assert counts["inventory"] == 3
    assert counts["crop_candidate"] == 2
    assert counts["blocked"] == 1
    assert counts["fits_budget_whole"] == 1
    assert counts["needs_segment_selection"] == 1
    assert counts["schedulable_now"] == 1
    assert counts["qualified"] == 0, "a crop candidate is not a qualified sound"
    assert report["blocked_by_state"] == {mod.ADMISSION_BLOCKED_MACHINE_FAIL: 1}
    assert report["segment_selection_owner"] == "P25"

    with_selection = mod.library_candidate_report(
        overlay, identity, budget=budget, config=config,
        selections={"dog_bark/long/clip.wav": selection(
            relative_path="dog_bark/long/clip.wav")})
    assert with_selection["counts"]["qualified"] == 1
    assert with_selection["counts"]["schedulable_now"] == 2

    # an unverifiable selection does not qualify anything
    refused = mod.library_candidate_report(
        overlay, identity, budget=budget, config=config,
        selections={"dog_bark/long/clip.wav": {"relative_path": "dog_bark/long/clip.wav"}})
    assert refused["counts"]["qualified"] == 0


def test_a_blocked_clip_is_never_counted_as_qualified_even_with_a_selection(tmp_path):
    root = tmp_path / "new"
    write_clip(root, "dog_bark/broken", duration_s=9.0, qc_verdict="fail",
               findings=CLIPPING_FAIL, human_review=HUMAN_PASS)
    config = {"sound_library_roots": [{"root": str(root), "priority": 1}],
              "segment_activity_requirements": {"animal_call": {
                  "minimum_activity_coverage": 0.9}}}
    overlay = mod.load_sound_library_overlay(config)
    identity = mod.resolve_library_identity(overlay, {"sound_asset_ids_by_relative": {}})
    report = mod.library_candidate_report(
        overlay, identity, budget=mod.segment_budget(episode_s=10.0, reserve_tail_s=3.0),
        config=config,
        selections={"dog_bark/broken/clip.wav": selection(
            relative_path="dog_bark/broken/clip.wav")})
    assert report["counts"]["qualified"] == 0


# -------------------------------------------------------------- read-only + CLI


def test_the_overlay_writes_nothing_and_never_opens_the_recording(tmp_path):
    root = tmp_path / "new"
    audio = write_clip(root, "dog_bark/a", duration_s=3.0)
    before = {path: (path.stat().st_mtime_ns, path.read_bytes())
              for path in sorted(root.rglob("*")) if path.is_file()}
    mod.load_sound_library_overlay(
        {"sound_library_roots": [{"root": str(root), "priority": 1}]})
    after = {path: (path.stat().st_mtime_ns, path.read_bytes())
             for path in sorted(root.rglob("*")) if path.is_file()}
    assert before == after
    # duration came from the QC sidecar, not from decoding the wav
    assert audio.exists()


def test_cli_reports_the_library_and_still_accepts_the_registry_only_form(tmp_path, capsys):
    root = tmp_path / "new"
    write_clip(root, "dog_bark/a", duration_s=2.0)
    write_clip(root, "dog_bark/b", duration_s=9.0)
    out = tmp_path / "library.json"
    code = mod.main([
        "--sound-library-root", f"{root}:20:incremental",
        "--episode-s", "10", "--reserve-tail-s", "3",
        "--library-report", str(out),
    ])
    assert code == 0
    printed = capsys.readouterr().out
    assert "library clips: 2" in printed
    assert "one dry program has 7.000 s" in printed
    written = json.loads(out.read_text())
    assert written["candidates"]["counts"]["inventory"] == 2
    assert written["candidates"]["counts"]["needs_segment_selection"] == 1
    assert written["identity"]["counts"][mod.IDENTITY_NEW_CANDIDATE] == 2

    registry_only = mod.main([
        "--source-registry", "examples/runtime/source_asset_runtime_profiles.json"])
    assert registry_only == 0
    assert "registered assets:" in capsys.readouterr().out


def test_cli_requires_at_least_one_input():
    with pytest.raises(SystemExit):
        mod.main([])


def write_sounding_clip(root: Path, relative: str, *, classes=("air_conditioning",),
                        duration_s=9.0, rate=16000) -> Path:
    """A clip that actually carries sound, so P25 can measure activity in it."""
    import math
    folder = root / relative
    folder.mkdir(parents=True, exist_ok=True)
    audio = folder / mod.CLIP_AUDIO_NAME
    frames = int(duration_s * rate)
    payload = bytearray()
    for index in range(frames):
        value = int(12000 * math.sin(2 * math.pi * 220 * index / rate))
        payload += int(value).to_bytes(2, "little", signed=True)
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(payload))
    (folder / mod.CLIP_SIDECAR_NAME).write_text(
        json.dumps({"event_classes": list(classes), "source": "unit fixture",
                    "license": "fixture", "dry": True}), encoding="utf-8")
    (folder / mod.CLIP_QC_SIDECAR_NAME).write_text(json.dumps({
        "schema": "avengine_sound_clip_qc_v1", "verdict": "pass", "findings": [],
        "sample_rate_hz": rate, "channel_count": 1,
        "measured": {"duration_s": duration_s, "active_frame_ratio": 1.0,
                     "continuous": True, "peak": 0.37, "rms_dbfs": -12.0,
                     "noise_floor_dbfs": -80.0, "decay_to_minus20db_s": None},
    }), encoding="utf-8")
    return audio


def test_a_real_p25_segment_row_verifies_and_reaches_the_pool(tmp_path):
    """P07 offers sources, P25 cuts and reads back, P07 verifies the returned row."""
    try:
        from avengine.dataset import sound_segments as p25
    except ImportError as error:  # P25 is mid-edit in this checkout
        pytest.skip(f"P25 sound_segments is not importable right now: {error}")

    root = tmp_path / "library"
    write_sounding_clip(root, "air_conditioning/long_one", duration_s=9.0)
    config = {
        "sound_library_roots": [{"root": str(root), "priority": 1}],
        "segment_activity_requirements": {"device_continuous": {
            "minimum_activity_coverage": 0.9, "maximum_internal_silence_s": 0.5}},
    }
    overlay = mod.load_sound_library_overlay(config)
    identity = mod.resolve_library_identity(overlay, {"sound_asset_ids_by_relative": {}})
    sources = mod.library_segment_sources(overlay, identity)
    assert len(sources) == 1

    budget = mod.segment_budget(episode_s=10.0, reserve_tail_s=3.0)
    result = p25.prepare_segments(
        sources, tmp_path / "segments",
        budget=p25.SegmentBudget(max_duration_s=budget["max_single_program_span_s"]),
        verify=True)
    assert result["counts"]["materialized"] == 1
    assert not result["failures"]

    row = p25.pool_row(result["segments"][0], compatible_asset_ids=["speaker_a"])
    requirements = mod.segment_requirements("device_continuous", config)
    verdict = mod.verify_segment_selection(row, requirements=requirements)
    assert verdict["verified"] is True, verdict["problems"]
    assert verdict["crop_authorization"]
    # the crop respects the derived budget and is not a truncation
    assert (row["sample_count"] / row["sample_rate_hz"]
            <= budget["max_single_program_span_s"] + 1e-3)
    assert row["truncated"] is False
    assert row["selection_authorized"] is True
    # the original recording is untouched
    assert (root / "air_conditioning/long_one" / mod.CLIP_AUDIO_NAME).exists()


def test_an_unconfigured_activity_family_refuses_rather_than_guesses(tmp_path):
    """dial_tone resolves to an unknown activity family, so it must fail closed."""
    from avengine.assets.sound_prepare import activity_profile_for_class

    assert activity_profile_for_class("dial_tone")["activity_family"] == "unknown"
    requirements = mod.segment_requirements("unknown", {
        "segment_activity_requirements": {"device_continuous": {
            "minimum_activity_coverage": 0.9}}})
    assert requirements["state"] == mod.STATE_EVIDENCE_MISSING
    verdict = mod.verify_segment_selection(selection(), requirements=requirements)
    assert verdict["verified"] is False
    assert "activity_requirement_is_not_configured" in verdict["problems"]


# ------------------------------------------- hardened selection validation (R2)

A0_ROW = {
    "selection_authorized": True, "crop_authorization": "user_active_crop",
    "source_crop_start_sample": 0, "source_crop_end_sample_exclusive": 100,
    "relative": "class/clip/clip.wav",
    "source_activity_intervals_samples": [[0, 100]],
    "activity_coverage": 1.0, "max_internal_silence_s": 0.0, "truncated": False,
}
A0_CLIP = {"relative": "class/clip/clip.wav", "sample_count": 100,
           "source_sample_count": 100}
A0_REQ = {"state": "available", "minimum_activity_coverage": 0.9,
          "maximum_internal_silence_s": 0.2}


def test_the_unmutated_a0_row_still_verifies():
    verdict = mod.verify_segment_selection(A0_ROW, clip=A0_CLIP, requirements=A0_REQ)
    assert verdict["verified"] is True, verdict["problems"]


@pytest.mark.parametrize("name,change,expected", [
    ("string_coverage", {"activity_coverage": "unmeasured"},
     "activity_coverage_is_not_a_finite_number"),
    ("nan_coverage", {"activity_coverage": float("nan")},
     "activity_coverage_is_not_a_finite_number"),
    ("missing_silence", {"max_internal_silence_s": None},
     "max_internal_silence_s_is_not_a_finite_number"),
    ("string_intervals", {"source_activity_intervals_samples": "unmeasured"},
     "source_activity_intervals_samples_is_not_a_list_of_sample_pairs"),
    ("end_past_source", {"source_crop_end_sample_exclusive": 999999},
     "source_crop_end_exceeds_source_sample_count"),
])
def test_the_five_reported_validation_holes_are_closed(name, change, expected):
    """Each of these returned verified=True before; the reason is now explicit."""
    verdict = mod.verify_segment_selection(
        {**A0_ROW, **change}, clip=A0_CLIP, requirements=A0_REQ)
    assert verdict["verified"] is False, name
    assert expected in verdict["problems"], (name, verdict["problems"])


def test_a_configured_condition_is_never_skipped_because_a_field_is_absent():
    """A missing or mistyped measurement must fail the condition, not bypass it."""
    without_coverage = {k: v for k, v in A0_ROW.items() if k != "activity_coverage"}
    verdict = mod.verify_segment_selection(
        without_coverage, clip=A0_CLIP, requirements=A0_REQ)
    assert "activity_coverage_is_required_but_unusable" in verdict["problems"]

    without_silence = {k: v for k, v in A0_ROW.items() if k != "max_internal_silence_s"}
    verdict = mod.verify_segment_selection(
        without_silence, clip=A0_CLIP, requirements=A0_REQ)
    assert "max_internal_silence_s_is_required_but_unusable" in verdict["problems"]

    short = mod.verify_segment_selection(
        A0_ROW, clip=A0_CLIP,
        requirements={**A0_REQ, "minimum_segment_s": 4.0,
                      "minimum_activity_coverage": None,
                      "maximum_internal_silence_s": None})
    assert "segment_duration_is_required_but_unusable" in short["problems"]


def test_coverage_outside_zero_to_one_and_negative_silence_are_refused():
    assert "activity_coverage_is_outside_zero_to_one" in mod.verify_segment_selection(
        selection(activity_coverage=1.4))["problems"]
    assert "activity_coverage_is_outside_zero_to_one" in mod.verify_segment_selection(
        selection(activity_coverage=-0.1))["problems"]
    assert "max_internal_silence_s_is_negative" in mod.verify_segment_selection(
        selection(max_internal_silence_s=-1.0))["problems"]


def test_a_bool_is_not_a_number_and_a_float_index_is_not_a_sample():
    assert "activity_coverage_is_not_a_finite_number" in mod.verify_segment_selection(
        selection(activity_coverage=True))["problems"]
    assert "invalid_source_crop_start_sample" in mod.verify_segment_selection(
        selection(source_crop_start_sample=0.0))["problems"]
    assert "source_activity_intervals_samples_has_a_non_integer_sample" in \
        mod.verify_segment_selection(
            selection(source_activity_intervals_samples=[[0.0, 100.0]]))["problems"]


@pytest.mark.parametrize("intervals,expected", [
    ([], "source_activity_intervals_samples_is_empty"),
    ([[10, 10]], "source_activity_intervals_samples_has_an_empty_or_reversed_pair"),
    ([[10, 5]], "source_activity_intervals_samples_has_an_empty_or_reversed_pair"),
    ([[0, 100], [50, 200]],
     "source_activity_intervals_samples_pairs_are_not_ordered_and_disjoint"),
    ([[0]], "source_activity_intervals_samples_has_a_malformed_pair"),
    (["0-100"], "source_activity_intervals_samples_has_a_malformed_pair"),
])
def test_intervals_must_be_ordered_disjoint_non_empty_integer_pairs(intervals, expected):
    verdict = mod.verify_segment_selection(
        selection(source_activity_intervals_samples=intervals))
    assert expected in verdict["problems"], verdict["problems"]


def test_the_three_coordinate_systems_are_bounded_separately():
    """A 16 kHz interval is never checked against a 44.1 kHz crop position."""
    base = selection()
    verdict = mod.verify_segment_selection(base)
    bounds = verdict["coordinate_bounds"]
    assert verdict["verified"] is True, verdict["problems"]
    assert bounds["row_shape"] == "pool_row"
    assert bounds["segment_interval_field"] == "source_activity_intervals_samples"
    assert bounds["analysis_interval_field"] == "origin_activity_intervals_samples"
    assert bounds["source_sample_count"] == 132300
    assert bounds["segment_sample_count"] == 48000
    assert bounds["analysis_sample_count"] == 48000

    # a segment interval expressed in 44.1 kHz source samples overruns the segment
    assert "source_activity_intervals_samples_exceeds_segment_sample_count" in \
        mod.verify_segment_selection(
            selection(source_activity_intervals_samples=[[0, 132300]]))["problems"]
    # an analysis interval past the resampled recording is caught on its own bound
    assert "origin_activity_intervals_samples_exceeds_analysis_sample_count" in \
        mod.verify_segment_selection(
            selection(origin_activity_intervals_samples=[[0, 48001]]))["problems"]


def test_the_segment_record_shape_reads_the_other_field_names():
    """In P25's record the same field name means analysis coordinates instead."""
    record = {
        "selection_authorized": True,
        "crop_authorization": "owner_authorized_activity_segment_selection_20260910",
        "relative_path": "dog_bark/a/clip.wav", "truncated": False,
        "source_rate_hz": 44100, "source_sample_count": 132300,
        "source_crop_start_sample": 0, "source_crop_end_sample_exclusive": 132300,
        "analysis_sample_count": 48000, "analysis_rate_hz": 16000,
        "source_activity_intervals_samples": [[0, 47000]],
        "prepared_sample_count": 48000, "target_rate_hz": 16000,
        "segment_activity_intervals_samples": [[0, 47000]],
        "planned_activity": {"activity_coverage": 0.98,
                             "max_internal_silence_s": 0.0, "duration_s": 3.0},
    }
    verdict = mod.verify_segment_selection(record)
    assert verdict["verified"] is True, verdict["problems"]
    assert verdict["coordinate_bounds"]["row_shape"] == "segment_record"
    assert verdict["coordinate_bounds"]["segment_interval_field"] == \
        "segment_activity_intervals_samples"
    # the nested measurement is found and reported as nested
    assert verdict["activity_coverage"] == 0.98
    assert verdict["activity_coverage_from"] == "planned_activity.activity_coverage"
    # in this shape source_activity_* is analysis coordinates and is bounded as such
    overrun = mod.verify_segment_selection(
        {**record, "source_activity_intervals_samples": [[0, 48001]]})
    assert "source_activity_intervals_samples_exceeds_analysis_sample_count" in \
        overrun["problems"]


def test_the_source_length_comes_from_the_row_the_caller_or_the_header(tmp_path):
    from_row = mod.verify_segment_selection(selection())
    assert from_row["coordinate_bounds"]["source_sample_count_from"] == \
        "selection.source_sample_count"

    no_length = {k: v for k, v in selection().items() if k != "source_sample_count"}
    from_caller = mod.verify_segment_selection(
        no_length, clip={"relative": "dog_bark/a/clip.wav",
                         "source_sample_count": 132300})
    assert from_caller["coordinate_bounds"]["source_sample_count_from"] == \
        "caller.source_sample_count"

    audio = write_sounding_clip(tmp_path / "lib", "dog_bark/a", duration_s=1.0,
                                rate=16000)
    from_header = mod.verify_segment_selection(
        {**no_length, "source_path": str(audio),
         "source_crop_end_sample_exclusive": 16000})
    assert from_header["coordinate_bounds"]["source_sample_count_from"] == "wav_header"
    assert from_header["coordinate_bounds"]["source_sample_count"] == 16000

    # nothing exact anywhere: say so, never derive it from a rounded duration
    unverifiable = mod.verify_segment_selection(no_length, read_source_header=False)
    assert "source_sample_count" in unverifiable["unverified_bounds"]
    assert "source_sample_count_is_not_verifiable" in unverifiable["problems"]
    relaxed = mod.verify_segment_selection(
        no_length, read_source_header=False, require_bounds=False)
    assert relaxed["verified"] is True, relaxed["problems"]
    assert "source_sample_count" in relaxed["unverified_bounds"]


def test_a_rhythmic_class_keeps_its_natural_pauses():
    """A DTMF-like 42% coverage is normal rhythm, not a defect to reject outright."""
    rhythmic = selection(activity_coverage=0.42, max_internal_silence_s=0.6)
    lenient = mod.segment_requirements("short_prompt", {
        "segment_activity_requirements": {"short_prompt": {
            "minimum_activity_coverage": 0.20, "maximum_internal_silence_s": 3.0}}})
    assert mod.verify_segment_selection(rhythmic, requirements=lenient)["verified"]

    strict = mod.segment_requirements("device_continuous", {
        "segment_activity_requirements": {"device_continuous": {
            "minimum_activity_coverage": 0.90, "maximum_internal_silence_s": 0.5}}})
    refused = mod.verify_segment_selection(rhythmic, requirements=strict)
    assert "activity_coverage_below_configured_minimum" in refused["problems"]
    assert "internal_silence_exceeds_configured_maximum" in refused["problems"]


def test_a_segment_cut_with_other_processing_than_requested_is_refused():
    """A cache that returns an older crop must not pass as the requested one."""
    reused = selection(edge_fade_s=0.0, crop_guard_s_applied=0.03)
    assert mod.verify_segment_selection(
        reused, expected_processing={"edge_fade_s": 0.0})["verified"]
    mismatch = mod.verify_segment_selection(
        reused, expected_processing={"edge_fade_s": 0.005})
    assert mismatch["verified"] is False
    assert "segment_processing_parameters_do_not_match_the_request" in mismatch["problems"]
    assert mismatch["processing_mismatch"]["edge_fade_s"] == {
        "requested": 0.005, "recorded": 0.0}


# ------------------------------------------------- lineage inheritance (R2)


def test_a_shared_relative_path_alone_does_not_transfer_a_registered_identity(tmp_path):
    """Different audio at a path the lineage already uses must not inherit its IDs."""
    base, later = tmp_path / "base", tmp_path / "later"
    write_sounding_clip(base, "dog_bark/shared", duration_s=2.0)
    # same relative path, different recording (different length, different bytes)
    write_sounding_clip(later, "dog_bark/shared", duration_s=5.0)
    overlay = mod.load_sound_library_overlay({"sound_library_roots": [
        {"root": str(later), "priority": 20, "role": "incremental"},
        {"root": str(base), "priority": 10, "role": "base"},
    ]})
    registered = mod.registered_source_index({"clips": [
        {"source": "dog_bark/shared/clip.wav", "sound_asset_id": "sound_a_v1",
         "event_class": "dog_bark"}]})

    # declaring the later root an alias is contradicted by the bytes themselves
    asserted = mod.resolve_library_identity(
        overlay, registered, lineage_root=str(base), alias_roots=[str(later)])
    assert asserted["counts"][mod.IDENTITY_SAME_SOURCE] == 0
    assert asserted["counts"][mod.IDENTITY_SAME_PATH_DIFFERENT_SOURCE] == 1
    assert asserted["registered_sound_asset_ids_reused"] == 0
    assert asserted["registered_sound_asset_ids_withheld"] == 1
    assert asserted["same_relative_path_different_source"][0]["reason"] == \
        "declared_same_path_copies_differ_in_byte_size"

    # with no alias declared it is separated for the simpler reason
    undeclared = mod.resolve_library_identity(
        overlay, registered, lineage_root=str(base))
    assert undeclared["same_relative_path_different_source"][0]["reason"] == \
        "winning_root_is_not_the_declared_lineage_root_or_an_alias"


def test_a_declared_alias_carrying_the_same_bytes_does_inherit(tmp_path):
    base, later = tmp_path / "base", tmp_path / "later"
    write_sounding_clip(base, "dog_bark/shared", duration_s=2.0)
    write_sounding_clip(later, "dog_bark/shared", duration_s=2.0)
    overlay = mod.load_sound_library_overlay({"sound_library_roots": [
        {"root": str(later), "priority": 20}, {"root": str(base), "priority": 10}]})
    registered = mod.registered_source_index({"clips": [
        {"source": "dog_bark/shared/clip.wav", "sound_asset_id": "sound_a_v1",
         "event_class": "dog_bark"}]})
    identity = mod.resolve_library_identity(
        overlay, registered, lineage_root=str(base), alias_roots=[str(later)])
    assert identity["counts"][mod.IDENTITY_SAME_SOURCE] == 1
    assert identity["registered_sound_asset_ids_reused"] == 1
    assert identity["lineage_root_declared"] is True


def test_path_only_inheritance_is_recorded_as_such(tmp_path):
    root = tmp_path / "only"
    write_sounding_clip(root, "dog_bark/shared", duration_s=2.0)
    overlay = mod.load_sound_library_overlay(
        {"sound_library_roots": [{"root": str(root), "priority": 1}]})
    registered = mod.registered_source_index({"clips": [
        {"source": "dog_bark/shared/clip.wav", "sound_asset_id": "sound_a_v1",
         "event_class": "dog_bark"}]})
    identity = mod.resolve_library_identity(overlay, registered)
    assert identity["counts"][mod.IDENTITY_SAME_SOURCE] == 1
    assert identity["lineage_root_declared"] is False
    assert "never transfers an existing sound asset ID" in identity["claim_boundary"]
