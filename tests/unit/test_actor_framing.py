from __future__ import annotations

import copy
from pathlib import Path

import pytest

import avengine.actor_framing as framing
from avengine.actor_envelope import (
    ActorActionEnvelope,
    AxisAlignedBounds,
    SourceAssetAuthority,
)
from avengine.camera_framing import evaluate_static_camera_candidate


def _bindings(asset: Path) -> list[dict[str, object]]:
    return [
        {
            "actor_id": actor_id,
            "asset_id": f"asset_{actor_id}",
            "asset_revision": "runtime_v1",
            "source_asset_path": str(asset),
            "skin_index": 0,
            "action_name_by_action_id": {
                "idle": "Standing_Idle",
                "walk": "Walking",
            },
            "coordinate_chain": {
                "from_frame": "glb_asset_local_right_handed_y_up_m",
                "to_frame": "avengine_world_right_handed_y_up_m",
                "operations": [
                    "sample_action_local_envelope",
                    "actor_state_scale_rotation_translation",
                ],
            },
        }
        for actor_id in ("source1", "source2")
    ]


def _frames() -> list[dict[str, object]]:
    return [
        {
            "frame_index": frame_index,
            "actor_states": [
                {
                    "actor_id": "source1",
                    "action_id": "walk" if frame_index == 1 else "idle",
                    "translation_m": [-0.8, 0.0, -3.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "scale": 1.0,
                },
                {
                    "actor_id": "source2",
                    "action_id": "idle",
                    "translation_m": [0.8, 0.0, -3.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "scale": [1.0, 1.0, 1.0],
                },
            ],
        }
        for frame_index in range(3)
    ]


@pytest.fixture
def fake_envelopes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(framing, "load_glb", lambda path: {"path": path})
    monkeypatch.setattr(
        framing, "compile_skinning", lambda document, skin_index: object()
    )

    def build(compiled, action_name, *, sample_rate_hz, padding_m, source_asset_path):
        height = 1.8 if action_name == "Standing_Idle" else 1.9
        sampled = AxisAlignedBounds((-0.2, 0.0, -0.2), (0.2, height, 0.2))
        padded = AxisAlignedBounds(
            (-0.2 - padding_m, -padding_m, -0.2 - padding_m),
            (0.2 + padding_m, height + padding_m, 0.2 + padding_m),
        )
        return ActorActionEnvelope(
            source_asset=SourceAssetAuthority(str(Path(source_asset_path).resolve())),
            skin_index=0,
            action_name=action_name,
            sample_rate_hz=sample_rate_hz,
            sample_times_seconds=(0.0, 1.0),
            padding_m=padding_m,
            sampled_bounds=sampled,
            padded_bounds=padded,
        )

    monkeypatch.setattr(framing, "build_action_envelope", build)


def test_builds_structured_world_aabbs_for_generic_camera_solver(
    tmp_path: Path, fake_envelopes: None
) -> None:
    asset = tmp_path / "actor.glb"
    asset.write_bytes(b"fixture")
    result = framing.build_actor_framing_frames(
        actor_bindings=_bindings(asset),
        frame_states=_frames(),
        expected_frame_count=3,
    )

    assert result["status"] == "pass_cpu_sampled_planning_envelopes"
    assert result["frame_count"] == 3
    assert result["qualification"] == {
        "state": "planning_only",
        "qualification_claim": False,
        "formal_episode_count": 0,
        "native_ue_bounds_pending": True,
        "native_pixel_framing_pending": True,
        "continuous_containment_claim": False,
    }
    source1_middle = result["frames"][1]["actor_aabbs"]["source1"]
    assert source1_middle["action_id"] == "walk"
    assert source1_middle["minimum_m"] == pytest.approx([-1.02, -0.02, -3.22])
    assert source1_middle["maximum_m"] == pytest.approx([-0.58, 1.92, -2.78])
    assert source1_middle["action_coverage"]["covered_frame_indices"] == [1]
    assert source1_middle["bounds_authority"]["live_renderer_bounds_pending"] is True
    assert result["actor_envelopes"]["source1"]["actions"]["idle"][
        "covered_frame_indices"
    ] == [0, 2]

    evaluation = evaluate_static_camera_candidate(
        frames=result["frames"],
        candidate={
            "candidate_id": "wide",
            "priority": 1,
            "position_m": [0.0, 1.0, 0.0],
            "yaw_deg": 0.0,
            "room_gate": {
                "status": "pass",
                "authority_id": "fixture-room-gate",
                "hard_gates": {"nav": {"status": "pass"}},
            },
        },
        calibration={
            "resolution_hw": [720, 1280],
            "hfov_degrees": 90.0,
            "near_m": 0.05,
            "margins_px": {"left": 1, "right": 1, "top": 1, "bottom": 1},
        },
        ordered_actor_ids=["source1", "source2"],
        minimum_order_gap_px=0,
    )
    assert evaluation["all_frames_hard_gates_pass"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda bindings, frames: frames.append(copy.deepcopy(frames[-1])),
            "duplicate frame_index",
        ),
        (lambda bindings, frames: frames[1]["actor_states"].pop(), "every bound actor"),
        (
            lambda bindings, frames: frames[1]["actor_states"][0].update(
                action_id="run"
            ),
            "no source action",
        ),
        (
            lambda bindings, frames: bindings.append(copy.deepcopy(bindings[0])),
            "duplicate actor binding",
        ),
        (
            lambda bindings, frames: bindings[0].update(
                source_asset_path="/missing/actor.glb"
            ),
            "does not exist",
        ),
    ],
)
def test_fails_closed_before_publishing_partial_frames(
    tmp_path: Path,
    fake_envelopes: None,
    mutation,
    message: str,
) -> None:
    asset = tmp_path / "actor.glb"
    asset.write_bytes(b"fixture")
    bindings = _bindings(asset)
    frames = _frames()
    mutation(bindings, frames)
    with pytest.raises(framing.ActorFramingError, match=message):
        framing.build_actor_framing_frames(
            actor_bindings=bindings,
            frame_states=frames,
            expected_frame_count=3,
        )
