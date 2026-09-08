from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.assets.build_habitat_asset_package as builder


def _package_value() -> dict[str, object]:
    return {
        "anchors": [
            {
                "anchor_id": "body",
                "joint_id": "bone_0",
                "joint_from_anchor": {
                    "translation_m": [0.0, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
            {
                "anchor_id": "muzzle",
                "joint_id": "bone_37",
                "joint_from_anchor": {
                    "translation_m": [0.0, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
        ]
    }


def _minimal_spec(tmp_path: Path) -> dict[str, object]:
    return {
        "asset_id": "asset",
        "source_glb": str(tmp_path / "missing.glb"),
        "contact_order": [
            "paw_front_left",
            "paw_front_right",
            "paw_hind_left",
            "paw_hind_right",
        ],
        "anchors": {
            "body": "bone_0",
            "head": "bone_37",
            "muzzle": "bone_37",
            "paw_front_left": "bone_1",
            "paw_front_right": "bone_2",
            "paw_hind_left": "bone_3",
            "paw_hind_right": "bone_4",
        },
    }


def test_package_emitter_separates_joint_and_actor_root_offsets() -> None:
    source = {
        "schema": "native_idle0_actor_root_muzzle_reference_v1",
        "action_id": "idle",
        "frame_index": 0,
    }
    emitter = builder._package_emitter(
        _package_value(),
        root_offset_m=(-0.4, 0.6, 0.02),
        root_offset_source=source,
    )

    assert emitter["anchor_id"] == "muzzle"
    assert emitter["semantic_anchor_id"] == "muzzle"
    assert emitter["joint_id"] == "bone_37"
    assert emitter["offset_m"] == [0.0, 0.0, 0.0]
    assert emitter["offset_space"] == "joint_local"
    assert emitter["native_offset_space"] == "joint_local"
    assert emitter["root_offset_m"] == [-0.4, 0.6, 0.02]
    assert emitter["root_offset_space"] == "final_scaled_asset_root"
    assert emitter["root_offset_source"] == source


def test_package_emitter_allows_measured_zero_root_offset() -> None:
    emitter = builder._package_emitter(
        _package_value(),
        root_offset_m=(0.0, 0.0, 0.0),
        root_offset_source={"schema": "native_idle0_actor_root_muzzle_reference_v1"},
    )

    assert emitter["root_offset_m"] == [0.0, 0.0, 0.0]
    assert emitter["root_offset_status"] == "measured_native_reference"


def test_package_emitter_marks_unmeasured_root_reference_without_fabricating_zero() -> None:
    emitter = builder._package_emitter(_package_value())

    assert emitter["root_offset_status"] == "not_measured"
    assert "root_offset_m" not in emitter
    assert emitter["offset_space"] == "joint_local"


@pytest.mark.parametrize(
    "kwargs",
    (
        {"root_offset_m": [0.1, 0.2, 0.3]},
        {"root_offset_source": {"schema": "native_idle0_actor_root_muzzle_reference_v1"}},
        {"root_offset_m": [0.1, float("nan"), 0.3], "root_offset_source": {"schema": "test"}},
    ),
)
def test_package_emitter_rejects_partial_or_nonfinite_root_reference(kwargs: dict[str, object]) -> None:
    with pytest.raises(builder.P12BuildError):
        builder._package_emitter(_package_value(), **kwargs)


def test_old_spec_without_root_reference_enters_existing_build_flow_without_expensive_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_minimal_spec(tmp_path)), encoding="utf-8")
    output = tmp_path / "output"

    class StopPipeline(RuntimeError):
        pass

    monkeypatch.setattr(builder, "_needs_webp", lambda source: False)

    def stop_after_optional_reference(*args: object, **kwargs: object) -> None:
        raise StopPipeline("entered existing build flow")

    monkeypatch.setattr(builder, "normalize_dynamic_root_translations", stop_after_optional_reference)
    with pytest.raises(StopPipeline, match="entered existing build flow"):
        builder.build(spec_path, output, gpu_device_id=0)
    assert output.is_dir()


def test_build_rejects_partial_root_reference_before_creating_output(tmp_path: Path) -> None:
    spec = _minimal_spec(tmp_path)
    spec["root_offset_m"] = [0.1, 0.2, 0.3]
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    output = tmp_path / "output"

    with pytest.raises(builder.P12BuildError, match="supplied together"):
        builder.build(spec_path, output, gpu_device_id=0)
    assert not output.exists()
