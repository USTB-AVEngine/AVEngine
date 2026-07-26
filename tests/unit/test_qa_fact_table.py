from __future__ import annotations

import json
import math
from pathlib import Path

import jsonschema
import pytest

from avengine.qa.fact_table import (
    QAFactTableError,
    compile_episode_fact_table,
    listener_local_spherical_track,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPOSITORY_ROOT / "schemas" / "avengine_qa_fact_table_v1.schema.json"

FRAME_COUNT = 75
IDENTITY_WXYZ = [1.0, 0.0, 0.0, 0.0]
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _registry() -> dict:
    def asset(asset_id: str, species: str, offset: list[float], forward: list[float]):
        return {
            "asset_id": asset_id,
            "revision": "test_v1",
            "display_label": species.title(),
            "entity_class": "articulated_animal",
            "identity": {"species_id": species, "breed_id": f"{species}_breed"},
            "realized_attributes": {
                "size": "medium",
                "body_build": "standard",
                "life_stage": "adult",
                "coat_profile": {"profile_id": f"{species}_coat_v1", "value": "black"},
            },
            "timeline": {"local_anatomical_forward_axis": forward},
            "default_emitter_anchor_id": "muzzle",
            "emitter_anchors": [
                {
                    "anchor_id": "muzzle",
                    "anchor_type": "muzzle",
                    "offset_m": offset,
                    "offset_space": "final_scaled_asset_root",
                }
            ],
            "admission_state": "research",
        }

    return {
        "registry_id": "test_registry",
        "revision": "test_rev",
        "assets": [
            asset("asset_dog", "dog", [0.4, 0.6, 0.0], [1.0, 0.0, 0.0]),
            asset("asset_cat", "cat", [0.3, 0.15, 0.0], [1.0, 0.0, 0.0]),
        ],
    }


def _bank_header() -> dict:
    return {
        "frame_count": FRAME_COUNT,
        "frame_rate_hz": 15,
        "seconds_per_episode": 5.0,
        "source_slots": ["source1", "source2"],
    }


def _bank_episode() -> dict:
    """source1 static facing -Z at (0, 0.5, -2); source2 walks +X at 0.3 m/s."""

    static_root = [0.0, 0.5, -2.0]
    static_emitter = [0.0, 1.1, -2.4]  # offset [0.4,0.6,0] yaw-rotated to -Z
    source1_root = [list(static_root) for _ in range(FRAME_COUNT)]
    source1_emitter = [list(static_emitter) for _ in range(FRAME_COUNT)]

    source2_root = [[1.0 + 0.02 * i, 0.3, 1.0] for i in range(FRAME_COUNT)]
    # offset [0.3,0.15,0] yaw-rotated toward -X (facing -X)
    source2_emitter = [[1.0 + 0.02 * i - 0.3, 0.45, 1.0] for i in range(FRAME_COUNT)]
    return {
        "episode_id": "test_episode_0000",
        "motion_case": "source1_static_source2_moving",
        "source_center_paths_m": {
            "source1": source1_emitter,
            "source2": source2_emitter,
        },
        "source_root_paths_m": {"source1": source1_root, "source2": source2_root},
    }


def _sample_entry() -> dict:
    return {
        "episode_id": "test_episode_0000",
        "asset_ids_by_source_slot": {"source1": "asset_dog", "source2": "asset_cat"},
        "audio": {
            "channel_count": 2,
            "sample_rate_hz": 16000,
            "sample_count": 80000,
            "peak_absolute": 0.05,
            "mixture": {"path": "test_episode_0000__v00.wav", "audio_sha256": SHA_A},
        },
    }


def _dry_variants() -> dict:
    def variant(path: str) -> dict:
        return {
            "variant_index": 0,
            "record": {
                "input": {"path": path, "sha256": SHA_B},
                "linear_gain": 0.1,
            },
        }

    return {"source1": variant("/dry/dog.wav"), "source2": variant("/dry/cat.wav")}


def _anchors() -> list[dict]:
    return [
        {
            "anchor_id": "camera_listener_default",
            "kind": "camera_listener_pose",
            "position_m": [0.0, 1.5, 0.0],
            "yaw_deg": 0.0,
        },
        {
            "anchor_id": "marker_front",
            "kind": "marker",
            "position_m": [0.0, 0.5, -4.0],
            "yaw_deg": None,
        },
    ]


def _compile(**overrides) -> dict:
    arguments = dict(
        bank_header=_bank_header(),
        bank_episode=_bank_episode(),
        listener_position_m=[0.0, 1.5, 0.0],
        listener_orientation_wxyz=IDENTITY_WXYZ,
        sample_entry=_sample_entry(),
        dry_variants_by_slot=_dry_variants(),
        registry=_registry(),
        anchors=_anchors(),
        room={"room_capsule_id": "test_room", "revision": "v1"},
        rir_cache_request_identity_sha256=SHA_C,
        provenance_inputs=[
            {"role": "trajectory_bank", "path": "/x/bank.json", "sha256": SHA_B}
        ],
    )
    arguments.update(overrides)
    return compile_episode_fact_table(**arguments)


def test_doa_track_follows_native_full_circle_convention() -> None:
    listener = [0.0, 1.0, 0.0]
    sources = [
        [0.0, 1.0, -2.0],  # dead ahead
        [2.0, 1.0, 0.0],  # right
        [-2.0, 1.0, 0.0],  # left
        [0.0, 1.0, 2.0],  # behind
        [0.0, 3.0, -2.0],  # ahead and above
    ]
    track = listener_local_spherical_track(sources, listener, IDENTITY_WXYZ)
    assert track["azimuth_deg"][0] == pytest.approx(0.0, abs=1e-12)
    assert track["azimuth_deg"][1] == pytest.approx(90.0, abs=1e-12)
    assert track["azimuth_deg"][2] == pytest.approx(-90.0, abs=1e-12)
    assert abs(track["azimuth_deg"][3]) == pytest.approx(180.0, abs=1e-12)
    assert track["elevation_deg"][4] == pytest.approx(45.0, abs=1e-9)
    assert track["distance_m"][0] == pytest.approx(2.0, abs=1e-12)
    assert track["distance_m"][4] == pytest.approx(math.sqrt(8.0), abs=1e-12)


def test_doa_track_with_yawed_listener() -> None:
    # Listener yawed 90 degrees (facing -X in the anchor convention).
    half = math.radians(90.0) / 2.0
    quaternion = [math.cos(half), 0.0, math.sin(half), 0.0]
    track = listener_local_spherical_track(
        [[-3.0, 1.0, 0.0], [0.0, 1.0, -3.0]], [0.0, 1.0, 0.0], quaternion
    )
    assert track["azimuth_deg"][0] == pytest.approx(0.0, abs=1e-9)
    assert track["azimuth_deg"][1] == pytest.approx(90.0, abs=1e-9)


def test_doa_rejects_non_unit_quaternion_and_coincident_points() -> None:
    with pytest.raises(QAFactTableError):
        listener_local_spherical_track(
            [[1.0, 0.0, 0.0]], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]
        )
    with pytest.raises(QAFactTableError):
        listener_local_spherical_track(
            [[0.0, 1.0, 0.0]], [0.0, 1.0, 0.0], IDENTITY_WXYZ
        )


def test_compiled_fact_table_validates_against_repository_schema() -> None:
    fact_table = _compile()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(fact_table)


def test_compiled_fact_table_derivations() -> None:
    fact_table = _compile()
    assert fact_table["schema"] == "avengine_qa_fact_table_v1"
    assert fact_table["time"]["ticks_per_frame"] == 3200
    assert fact_table["listener"]["yaw_deg"] == pytest.approx(0.0, abs=1e-9)

    source1 = fact_table["tracks"]["instances"]["source1"]
    assert all(not moving for moving in source1["moving"])
    assert source1["facing_yaw_deg"][0] == pytest.approx(0.0, abs=1e-9)
    # Static dog emitter at (0, 1.1, -2.4) seen from (0, 1.5, 0): dead ahead.
    assert source1["doa"]["azimuth_deg"][0] == pytest.approx(0.0, abs=1e-9)
    assert source1["doa"]["distance_m"][0] == pytest.approx(
        math.hypot(0.4, 2.4), abs=1e-9
    )

    source2 = fact_table["tracks"]["instances"]["source2"]
    assert all(source2["moving"])
    assert source2["speed_mps"][0] == pytest.approx(0.3, abs=1e-9)
    # Cat muzzle displaced toward -X while walking +X: facing -X is +90 yaw.
    assert source2["facing_yaw_deg"][0] == pytest.approx(90.0, abs=1e-9)

    events = {event["source_slot_id"]: event for event in fact_table["sound_events"]}
    assert events["source1"]["end_tick"] == 240000
    assert events["source1"]["sound_class"]["species_id"] == "dog"
    assert events["source2"]["dry_variant"]["input_path"] == "/dry/cat.wav"

    pair = fact_table["tracks"]["pairwise"]["source1__source2"]
    first_expected = math.dist([0.0, 1.1, -2.4], [0.7, 0.45, 1.0])
    assert pair["emitter_distance_m"][0] == pytest.approx(first_expected, abs=1e-9)

    relations = fact_table["relations"]["anchor_distances"]
    assert "camera_listener_default" not in relations["source1"]
    assert relations["source1"]["marker_front"]["min_m"] == pytest.approx(
        math.dist([0.0, 1.1, -2.4], [0.0, 0.5, -4.0]), abs=1e-9
    )


def test_human_style_vertical_emitter_offset_yields_null_facing() -> None:
    registry = _registry()
    registry["assets"][0]["emitter_anchors"][0]["offset_m"] = [0.0, 1.61, 0.0]
    registry["assets"][0]["timeline"]["local_anatomical_forward_axis"] = [0.0, 0.0, 1.0]
    episode = _bank_episode()
    episode["source_center_paths_m"]["source1"] = [
        [0.0, 0.5 + 1.61, -2.0] for _ in range(FRAME_COUNT)
    ]
    fact_table = _compile(registry=registry, bank_episode=episode)
    assert fact_table["tracks"]["instances"]["source1"]["facing_yaw_deg"] is None


def test_rejects_wrong_frame_count() -> None:
    episode = _bank_episode()
    episode["source_center_paths_m"]["source1"] = episode["source_center_paths_m"][
        "source1"
    ][:-1]
    with pytest.raises(QAFactTableError, match="frame count"):
        _compile(bank_episode=episode)


def test_rejects_emitter_path_that_disagrees_with_registry_offset() -> None:
    episode = _bank_episode()
    episode["source_center_paths_m"]["source1"] = [
        [0.0, 1.2, -2.4] for _ in range(FRAME_COUNT)
    ]
    with pytest.raises(QAFactTableError, match="emitter"):
        _compile(bank_episode=episode)


def test_rejects_unknown_asset_and_missing_dry_variant() -> None:
    sample = _sample_entry()
    sample["asset_ids_by_source_slot"]["source1"] = "asset_missing"
    with pytest.raises(QAFactTableError, match="registry"):
        _compile(sample_entry=sample)
    with pytest.raises(QAFactTableError, match="dry variant"):
        _compile(dry_variants_by_slot={"source1": _dry_variants()["source1"]})


def test_rejects_audio_duration_mismatch_and_bad_cache_identity() -> None:
    sample = _sample_entry()
    sample["audio"]["sample_count"] = 79999
    with pytest.raises(QAFactTableError, match="duration"):
        _compile(sample_entry=sample)
    with pytest.raises(QAFactTableError, match="sha256"):
        _compile(rir_cache_request_identity_sha256="not-a-hash")


def test_rejects_duplicate_anchor_ids() -> None:
    anchors = _anchors() + [_anchors()[1]]
    with pytest.raises(QAFactTableError, match="duplicate anchor"):
        _compile(anchors=anchors)
