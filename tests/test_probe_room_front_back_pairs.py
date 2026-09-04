"""Tests for the room front/back pair probe.

Everything here runs without the acoustic runtime: the geometry, the windowing
and the two guards are the parts that decide what a measurement means, and they
are also the parts a wrong answer would hide in.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "acoustics"))

import probe_room_front_back_pairs as probe


IDENTITY_QUAT = [0.0, 0.0, 0.0, 1.0]


def test_mirror_partner_shares_the_interaural_projection():
    """The pair is only interesting because its ITD is identical.

    Right positive azimuth measures from forward, so the interaural axis is the
    sine: mirroring about 90 degrees leaves it alone and flips the forward
    component, which is exactly the front/back ambiguity.
    """

    for azimuth in (0.0, 20.0, 30.0, 47.5, 89.0, -30.0, -140.0):
        mirror = probe.mirror_azimuth_deg(azimuth)
        assert math.isclose(math.sin(math.radians(azimuth)),
                            math.sin(math.radians(mirror)), abs_tol=1e-12)
        assert math.isclose(math.cos(math.radians(azimuth)),
                            -math.cos(math.radians(mirror)), abs_tol=1e-12)


def test_reference_is_the_same_angular_step_the_other_way():
    for azimuth in (20.0, 30.0, 40.0, 55.0):
        mirror = probe.mirror_azimuth_deg(azimuth)
        reference = probe.equal_separation_reference_deg(azimuth)
        forward_step = abs(probe.wrap_deg(mirror - azimuth))
        backward_step = abs(probe.wrap_deg(azimuth - reference))
        assert math.isclose(forward_step, backward_step, abs_tol=1e-9)
        # and unlike the mirror it does NOT share the interaural projection,
        # which is what makes it the "we can definitely hear this" yardstick
        assert not math.isclose(math.sin(math.radians(azimuth)),
                                math.sin(math.radians(reference)), abs_tol=1e-6)


def test_reference_for_thirty_degrees_is_minus_ninety():
    assert probe.equal_separation_reference_deg(30.0) == pytest.approx(-90.0)


def test_placement_round_trips_through_the_listener_basis():
    _, forward, right = probe.listener_basis([0.0, 0.3, 0.0, math.sqrt(1 - 0.09)])
    listener = [1.5, 1.471, -2.25]
    for azimuth in (0.0, 30.0, 150.0, -90.0, 179.0):
        position = probe.place_source(listener, forward, right, azimuth, 2.5, 0.721)
        assert position[1] == pytest.approx(0.721)
        vx, vz = position[0] - listener[0], position[2] - listener[2]
        assert math.hypot(vx, vz) == pytest.approx(2.5)
        recovered = math.degrees(math.atan2(
            vx * right[0] + vz * right[2], vx * forward[0] + vz * forward[2]))
        assert probe.wrap_deg(recovered) == pytest.approx(probe.wrap_deg(azimuth))


def test_a_pitched_or_rolled_listener_is_refused():
    """Sources are placed on a horizontal circle, so a tilted head is a lie."""

    with pytest.raises(probe.FrontBackProbeError, match="yaw only"):
        probe.listener_basis([0.2, 0.0, 0.0, math.sqrt(1 - 0.04)])


def _impulse(sample_rate, delay_s, tail_gain, length_s=0.5, seed=0):
    """One arrival plus a decaying tail, identical for a given seed.

    The tail is drawn at a fixed length and then placed at the delay, so two
    responses that differ only in delay really do differ only in delay. Drawing
    it after the delay would give the two a different number of samples and so
    a different random sequence, which is a difference the test is trying to
    prove absent.
    """

    rng = np.random.default_rng(seed)
    n = int(sample_rate * length_s)
    span = n // 2
    tail = rng.normal(0.0, 1.0, size=(2, span))
    decay = np.exp(-np.arange(span) / (sample_rate * 0.15))
    ir = np.zeros((2, n), dtype=np.float32)
    start = int(sample_rate * delay_s)
    ir[:, start] = 1.0
    ir[:, start:start + span] += (tail_gain * tail * decay).astype(np.float32)
    return ir


def test_late_to_direct_separates_a_room_from_open_space():
    sample_rate = 16000
    reverberant = _impulse(sample_rate, 0.01, tail_gain=0.30)
    dry = _impulse(sample_rate, 0.01, tail_gain=0.001)
    room = probe.late_to_direct_db(reverberant, sample_rate, 2.0, 20.0, 0.2)
    open_air = probe.late_to_direct_db(dry, sample_rate, 2.0, 20.0, 0.2)
    assert room > open_air + 20.0
    assert open_air < -6.0 < room


def test_the_window_starts_at_the_arrival_not_at_sample_zero():
    """Two responses rarely arrive together; a fixed start compares head to tail."""

    sample_rate = 16000
    early = _impulse(sample_rate, 0.005, tail_gain=0.2, seed=1)
    late = _impulse(sample_rate, 0.030, tail_gain=0.2, seed=1)
    assert probe.onset_index(early, 0.2) == pytest.approx(80, abs=2)
    assert probe.onset_index(late, 0.2) == pytest.approx(480, abs=2)
    # identical apart from the delay, so an onset-aligned window sees no difference
    aligned = probe.spectral_difference_db(
        early, late, sample_rate, 2000.0, 6000.0, 2.0, 0.2)
    assert aligned["mean"] < 1.0e-6


def test_removing_the_level_leaves_only_shape():
    sample_rate = 16000
    base = _impulse(sample_rate, 0.01, tail_gain=0.2, seed=3)
    louder = (base * 4.0).astype(np.float32)
    difference = probe.spectral_difference_db(
        base, louder, sample_rate, 2000.0, 6000.0, 2.0, 0.2)
    assert difference["mean"] < 1.0e-6


def test_a_silent_response_is_refused_rather_than_windowed():
    with pytest.raises(probe.FrontBackProbeError, match="silent"):
        probe.onset_index(np.zeros((2, 128), dtype=np.float32), 0.2)


def test_an_empty_band_is_refused():
    with pytest.raises(probe.FrontBackProbeError, match="holds no bin"):
        probe.band_spectrum_db(np.zeros(256), 16000, 9000.0, 12000.0)


def test_config_must_name_every_dependency(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"repo": "/somewhere"}), encoding="utf-8")
    with pytest.raises(probe.FrontBackProbeError, match="missing required keys"):
        probe.load_config(path)


def test_a_listener_offset_from_the_camera_rig_is_refused(tmp_path):
    """The probe measures at the rig pose; an offset would move every source."""

    path = tmp_path / "m1.json"
    path.write_text(json.dumps({
        "request_id": "r",
        "primary_camera_rig": {"world_from_rig": {
            "translation_m": [0.0, 1.5, 0.0], "rotation_xyzw": IDENTITY_QUAT}},
        "listener": {"rig_from_listener": {"translation_m": [0.0, 0.0, 0.4]}},
    }), encoding="utf-8")
    with pytest.raises(probe.FrontBackProbeError, match="offsets the listener"):
        probe.listener_from_m1_request(path)


def test_source_ids_stay_inside_the_runtime_character_set():
    for azimuth in (-180.0, -90.5, 0.0, 33.0, 150.0, 179.9):
        label = probe.azimuth_label(azimuth)
        assert label.replace("_", "").replace("-", "").replace(".", "").isalnum()
