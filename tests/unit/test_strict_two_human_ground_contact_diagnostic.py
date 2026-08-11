from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY / "tools/qa/build_strict_two_human_ground_contact_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("ground_contact_diagnostic", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def _write_glb(path: Path, bone_names: list[str]) -> None:
    document = json.dumps(
        {"asset": {"version": "2.0"}, "nodes": [{"name": name} for name in bone_names]},
        separators=(",", ":"),
    ).encode("utf-8")
    document += b" " * (-len(document) % 4)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(document))
        + struct.pack("<I4s", len(document), b"JSON")
        + document
    )


def _case(tmp_path: Path) -> tuple[Path, Path, Path]:
    actors = []
    for actor_id in ("source1_actor", "source2_actor"):
        glb = tmp_path / f"{actor_id}.glb"
        _write_glb(glb, sorted(TOOL.REQUIRED_BONES))
        normalization = tmp_path / f"{actor_id}_normalization.json"
        normalization.write_text(
            json.dumps(
                {
                    "expected_ue_qa": {
                        "ground_snap_to_floor": True,
                        "ground_snap_max_abs_correction_cm": 15.0,
                        "ground_snap_residual_tolerance_cm": 0.1,
                    },
                    "runtime_motion_contract": {
                        "dynamic_ground_snap_to_floor_required": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        manifest = tmp_path / f"{actor_id}_import.json"
        manifest.write_text(
            json.dumps({"source_glb": str(glb), "source_manifest": str(normalization)}),
            encoding="utf-8",
        )
        actors.append(
            {
                "actor_id": actor_id,
                "asset_id": f"asset_{actor_id}",
                "asset_revision": "v1",
                "body_plan_id": "biped_human",
                "runtime_asset_expectation": {
                    "ue_import_manifest": str(manifest),
                },
            }
        )
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "scenario_id": "strict2h",
                        "plan": {
                            "actors": actors,
                            "frames": [
                                {"frame_index": frame_index}
                                for frame_index in range(75)
                            ],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"not decoded by CPU request builder")
    spear_root = tmp_path / "spear"
    spear_root.mkdir()
    return suite, audio, spear_root


def test_build_diagnostic_request_is_not_gpu_authorization(tmp_path: Path) -> None:
    suite, audio, spear_root = _case(tmp_path)
    output = tmp_path / "request.json"

    TOOL.build_request(
        suite_plan=suite,
        scenario_id="strict2h",
        audio_wav=audio,
        spear_root=spear_root,
        capture_output=tmp_path / "future_capture",
        instrumented_suite_output=tmp_path / "instrumented_suite.json",
        output=output,
        rpc_port=39583,
        graphics_adapter=1,
    )

    request = json.loads(output.read_text())
    assert request["status"] == "cpu_ready_not_authorized_for_execution"
    assert request["frame_indices"] == [0, 37, 74]
    assert request["gpu_launch_authorized"] is False
    assert request["artifacts"]["instrumented_suite_plan"].endswith(
        "instrumented_suite.json"
    )
    instrumented = json.loads((tmp_path / "instrumented_suite.json").read_text())
    mutation = instrumented["ground_contact_diagnostic_mutation"]
    assert mutation["timeline_actor_root_mutation"] is False
    assert mutation["emitter_or_rir_mutation"] is False
    for actor in instrumented["scenarios"][0]["plan"]["actors"]:
        profile = actor["ground_contact_release_profile"]
        assert profile["status"] == "diagnostic_pending_not_release_qualified"
        assert (
            profile["runtime_visual_ground_snap"]["maximum_abs_correction_cm"] == 15.0
        )
    assert request["threshold_policy"] == {
        "contact_clearance_interval_cm": None,
        "status": "must_be_derived_after_live_diagnostic",
        "actor_root_z_revision_cm": None,
        "bounds_only_release_forbidden": True,
        "plan_root_only_release_forbidden": True,
    }
    argv = request["capture_argv_without_python"]
    assert [
        argv[index + 1] for index, value in enumerate(argv) if value == "--frame-index"
    ] == [
        "0",
        "37",
        "74",
    ]
    assert all(
        row["required_contact_bones_present"] for row in request["asset_evidence"]
    )


def test_build_diagnostic_request_rejects_missing_contact_bone(tmp_path: Path) -> None:
    suite, audio, spear_root = _case(tmp_path)
    document = json.loads(suite.read_text())
    manifest_path = Path(
        document["scenarios"][0]["plan"]["actors"][0]["runtime_asset_expectation"][
            "ue_import_manifest"
        ]
    )
    glb_path = Path(json.loads(manifest_path.read_text())["source_glb"])
    _write_glb(glb_path, sorted(TOOL.REQUIRED_BONES - {"Bip01 L Toe0"}))

    with pytest.raises(RuntimeError, match="lacks contact bones.*Bip01 L Toe0"):
        TOOL.build_request(
            suite_plan=suite,
            scenario_id="strict2h",
            audio_wav=audio,
            spear_root=spear_root,
            capture_output=tmp_path / "future_capture",
            instrumented_suite_output=tmp_path / "instrumented_suite.json",
            output=tmp_path / "request.json",
            rpc_port=39583,
            graphics_adapter=1,
        )
