from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from avengine.contracts.json_io import sha256_file
from avengine.m2.actions import (
    TICKS_PER_SAMPLE,
    TIME_BASE_HZ,
    BakedActionClip,
    BakedActionSet,
    write_baked_actions_npz,
)
from avengine.m2.glb_write import build_glb
from avengine.m2.kinematics import CONTACT_ORDER, ContactInferenceThresholds
from avengine.m2.variant_contacts import (
    EMITTER_ANCHORS_SCHEMA,
    VARIANT_CONTACT_DERIVATION_SCHEMA,
    VariantContactError,
    derive_variant_contact_artifacts,
)
from tools.m2 import derive_variant_contacts as cli


IDENTITY = (0.0, 0.0, 0.0, 1.0)
RUNTIME_JOINT_ORDER = (
    "equine_head",
    "equine_muzzle",
    "equine_fore_l",
    "equine_fore_r",
    "equine_hind_l",
    "equine_hind_r",
)
PAW_JOINTS = RUNTIME_JOINT_ORDER[2:]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _visual_document() -> dict[str, Any]:
    return {
        "asset": {"version": "2.0", "generator": "generic-equine-test"},
        "nodes": [
            {
                "name": "equine_root",
                "children": [1, 3, 4, 5, 6],
            },
            {
                "name": "equine_head",
                "children": [2],
                "translation": [0.0, 0.5, -0.45],
            },
            {
                "name": "equine_muzzle",
                "translation": [0.0, 0.0, -0.25],
            },
            {
                "name": "equine_fore_l",
                "translation": [0.18, -0.55, -0.35],
            },
            {
                "name": "equine_fore_r",
                "translation": [-0.18, -0.55, -0.35],
            },
            {
                "name": "equine_hind_l",
                "translation": [0.18, -0.55, 0.35],
            },
            {
                "name": "equine_hind_r",
                "translation": [-0.18, -0.55, 0.35],
            },
        ],
        "skins": [
            {
                "name": "equine_skin",
                "skeleton": 0,
                "joints": list(range(7)),
            }
        ],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }


def _spec() -> dict[str, Any]:
    anchors = [
        {
            "anchor_id": "body",
            "joint_id": "equine_root",
            "joint_from_anchor": {
                "translation_m": [0.0, 0.0, 0.0],
                "rotation_xyzw": list(IDENTITY),
            },
        },
        {
            "anchor_id": "head",
            "joint_id": "equine_head",
            "joint_from_anchor": {
                "translation_m": [0.0, 0.0, 0.0],
                "rotation_xyzw": list(IDENTITY),
            },
        },
        {
            "anchor_id": "muzzle",
            "joint_id": "equine_muzzle",
            "joint_from_anchor": {
                "translation_m": [0.0, 0.0, -0.1],
                "rotation_xyzw": list(IDENTITY),
            },
        },
    ]
    anchors.extend(
        {
            "anchor_id": anchor_id,
            "joint_id": joint_id,
            "joint_from_anchor": {
                "translation_m": [0.0, -0.1, 0.0],
                "rotation_xyzw": list(IDENTITY),
            },
        }
        for anchor_id, joint_id in zip(CONTACT_ORDER, PAW_JOINTS, strict=True)
    )
    return {
        "schema": "avengine_m2_variant_package_spec_v1",
        "taxonomy": {
            "species_id": "equus_caballus",
            "breed_id": "generic",
        },
        "appearance": {
            "size": "large",
            "body_build": "standard",
            "coat": "bay",
            "life_stage": "adult",
        },
        "rendering": {"shader_type": "pbr"},
        "identity": {
            "asset_id": "synthetic_equine_variant_v1",
            "template_id": "synthetic_equine_template_v1",
            "body_plan_id": "quadruped_mammal_equid_v1",
            "morphotype_id": "synthetic_equine",
            "skeleton_revision": "synthetic-equine-skeleton-v1",
            "weights_revision": "synthetic-equine-weights-v1",
            "collision_revision": "synthetic-equine-collision-v1",
            "action_revision": "synthetic-equine-actions-v1",
            "source": "unit-test-generated",
            "source_revision": "v1",
            "license": "test-only",
            "allowed_use": "research_only",
            "redistribution": "prohibited",
            "semantic_id": 291,
        },
        "anchors": anchors,
    }


def _z_rotation(angle: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0))


def _clip(
    semantic_action_id: str,
    source_action_name: str,
    frames: tuple[tuple[tuple[float, float, float, float], ...], ...],
) -> BakedActionClip:
    duration = len(frames) * TICKS_PER_SAMPLE
    ticks = tuple(range(0, duration, TICKS_PER_SAMPLE))
    return BakedActionClip(
        semantic_action_id=semantic_action_id,
        source_action_name=source_action_name,
        clip_start_seconds=0.0,
        clip_end_seconds=duration / TIME_BASE_HZ,
        loop_duration_ticks=duration,
        sample_ticks=ticks,
        source_times_seconds=tuple(tick / TIME_BASE_HZ for tick in ticks),
        rotations_xyzw=frames,
    )


def _actions(source_glb_sha256: str) -> BakedActionSet:
    sample_count = 12
    idle_frame = tuple(IDENTITY for _ in RUNTIME_JOINT_ORDER)
    idle_frames = tuple(idle_frame for _ in range(sample_count))
    walk_frames = []
    phase_offsets = (0.0, math.pi, math.pi, 0.0)
    for index in range(sample_count):
        phase = 2.0 * math.pi * index / sample_count
        paw_rotations = tuple(
            _z_rotation(0.6 * (1.0 + math.sin(phase + offset)))
            for offset in phase_offsets
        )
        walk_frames.append((IDENTITY, IDENTITY, *paw_rotations))
    return BakedActionSet(
        source_glb_sha256=source_glb_sha256,
        runtime_joint_order=RUNTIME_JOINT_ORDER,
        actions=(
            _clip("idle", "Idle", idle_frames),
            _clip("walk", "Walking", tuple(walk_frames)),
        ),
    )


def _fixture(tmp_path: Path) -> dict[str, Path]:
    visual = tmp_path / "equine_rebased.glb"
    visual.write_bytes(build_glb(_visual_document(), b""))
    visual_sha256 = sha256_file(visual)

    spec = tmp_path / "variant_spec.json"
    _write_json(spec, _spec())
    rebase = tmp_path / "rebase.json"
    _write_json(
        rebase,
        {
            "schema": "avengine_m2_skin_root_rebase_v1",
            "status": "pass",
            "qualification_state": "research_candidate",
            "qualification_claim": False,
            "output": {
                "path": str(visual),
                "sha256": visual_sha256,
                "byte_size": visual.stat().st_size,
            },
            "skin": {
                "root_joint": "equine_root",
                "actor_from_canonical_root": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
        },
    )
    actions = tmp_path / "actions.npz"
    write_baked_actions_npz(_actions(visual_sha256), actions)
    return {
        "spec": spec,
        "visual": visual,
        "actions": actions,
        "rebase": rebase,
        "emitter": tmp_path / "derived" / "emitter_anchors.json",
        "contacts": tmp_path / "derived" / "contact_phases.json",
    }


def _derive(paths: dict[str, Path]) -> dict[str, Any]:
    return derive_variant_contact_artifacts(
        spec_path=paths["spec"],
        visual_glb=paths["visual"],
        actions_npz=paths["actions"],
        rebase_report=paths["rebase"],
        emitter_anchors_output=paths["emitter"],
        contact_phases_output=paths["contacts"],
    )


def test_derives_generic_hash_bound_package_artifacts(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    result = _derive(paths)
    emitter = json.loads(paths["emitter"].read_text(encoding="utf-8"))
    contacts = json.loads(paths["contacts"].read_text(encoding="utf-8"))

    assert result["schema"] == VARIANT_CONTACT_DERIVATION_SCHEMA
    assert result["derivation_status"] == "completed"
    assert "status" not in result
    assert result["qualification_claim"] is False
    assert result["variant_package_spec"]["sha256"] == sha256_file(paths["spec"])
    assert result["visual_glb"]["sha256"] == sha256_file(paths["visual"])
    assert result["rebase_report"]["sha256"] == sha256_file(paths["rebase"])
    assert result["baked_actions"]["sha256"] == sha256_file(paths["actions"])
    assert result["emitter_anchors"]["sha256"] == sha256_file(paths["emitter"])
    assert result["contact_phases"]["sha256"] == sha256_file(paths["contacts"])

    assert emitter["schema"] == EMITTER_ANCHORS_SCHEMA
    assert emitter["source_visual_sha256"] == sha256_file(paths["visual"])
    assert emitter["qualification_claim"] is False
    assert [anchor["anchor_id"] for anchor in emitter["anchors"]] == [
        "body",
        "head",
        "muzzle",
        *CONTACT_ORDER,
    ]
    assert all(
        "beagle" not in anchor["joint_id"].lower() for anchor in emitter["anchors"]
    )

    assert contacts["schema"] == "avengine_m2_contact_phases_v1"
    assert contacts["source_glb_sha256"] == sha256_file(paths["visual"])
    assert contacts["baked_actions_sha256"] == sha256_file(paths["actions"])
    assert contacts["runtime_joint_order"] == list(RUNTIME_JOINT_ORDER)
    assert contacts["contact_order"] == list(CONTACT_ORDER)
    assert [anchor["joint_id"] for anchor in contacts["anchor_definitions"]] == list(
        PAW_JOINTS
    )
    assert contacts["thresholds"] == ContactInferenceThresholds().to_json_data()
    assert result["contact_phases"]["warning_count"] == len(contacts["warnings"]) == 4
    assert {warning["code"] for warning in contacts["warnings"]} == {
        "contact_horizontal_sliding"
    }
    for contact_id in CONTACT_ORDER:
        metric = next(
            item
            for item in contacts["actions"][1]["metrics"]
            if item["contact_id"] == contact_id
        )
        assert metric["inference_mode"] == "height_dynamic"
        assert metric["contact_frame_count"] > 0
        assert metric["swing_frame_count"] > 0


def test_refuses_to_replace_either_output(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    first = _derive(paths)
    emitter_before = paths["emitter"].read_bytes()
    contacts_before = paths["contacts"].read_bytes()

    with pytest.raises(VariantContactError, match="refusing to replace"):
        _derive(paths)

    assert (
        first["emitter_anchors"]["sha256"] == hashlib.sha256(emitter_before).hexdigest()
    )
    assert paths["emitter"].read_bytes() == emitter_before
    assert paths["contacts"].read_bytes() == contacts_before


def test_preflights_both_outputs_before_writing(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["contacts"].parent.mkdir(parents=True)
    paths["contacts"].write_text("owned by caller\n", encoding="utf-8")

    with pytest.raises(VariantContactError, match="contact phase output"):
        _derive(paths)

    assert not paths["emitter"].exists()
    assert paths["contacts"].read_text(encoding="utf-8") == "owned by caller\n"


def test_rejects_rebase_hash_mismatch_without_outputs(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    rebase = json.loads(paths["rebase"].read_text(encoding="utf-8"))
    rebase["output"]["sha256"] = "00" * 32
    _write_json(paths["rebase"], rebase)

    with pytest.raises(VariantContactError, match="output sha256"):
        _derive(paths)

    assert not paths["emitter"].exists()
    assert not paths["contacts"].exists()


def test_rejects_rebase_byte_size_mismatch_without_outputs(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    rebase = json.loads(paths["rebase"].read_text(encoding="utf-8"))
    rebase["output"]["byte_size"] += 1
    _write_json(paths["rebase"], rebase)

    with pytest.raises(VariantContactError, match="byte_size"):
        _derive(paths)

    assert not paths["emitter"].exists()
    assert not paths["contacts"].exists()


def test_rejects_actions_bound_to_another_visual(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    write_baked_actions_npz(_actions("ab" * 32), paths["actions"])

    with pytest.raises(VariantContactError, match="do not bind"):
        _derive(paths)

    assert not paths["emitter"].exists()
    assert not paths["contacts"].exists()


def test_rejects_noncanonical_npz_bytes(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["actions"].write_bytes(paths["actions"].read_bytes() + b"trailing-junk")

    with pytest.raises(VariantContactError, match="not canonical"):
        _derive(paths)

    assert not paths["emitter"].exists()
    assert not paths["contacts"].exists()


def test_output_directory_failure_cannot_leave_half_a_pair(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    blocked_parent = tmp_path / "blocked-parent"
    blocked_parent.write_text("not a directory\n", encoding="utf-8")
    paths["contacts"] = blocked_parent / "contact_phases.json"

    with pytest.raises(VariantContactError, match="prepare output directories"):
        _derive(paths)

    assert not paths["emitter"].exists()
    assert blocked_parent.read_text(encoding="utf-8") == "not a directory\n"


def test_rejects_unknown_spec_anchor_joint(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    spec = deepcopy(_spec())
    spec["anchors"][-1]["joint_id"] = "invented_hind_joint"
    _write_json(paths["spec"], spec)

    with pytest.raises(VariantContactError, match="unknown visual joints"):
        _derive(paths)

    assert not paths["emitter"].exists()
    assert not paths["contacts"].exists()


def test_rejects_duplicate_spec_keys_even_when_last_value_is_valid(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    original = paths["spec"].read_text(encoding="utf-8")
    paths["spec"].write_text(
        '{"schema":"ignored",' + original.lstrip()[1:], encoding="utf-8"
    )

    with pytest.raises(VariantContactError, match="duplicate key 'schema'"):
        _derive(paths)

    assert not paths["emitter"].exists()
    assert not paths["contacts"].exists()


def test_cli_emits_machine_readable_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _fixture(tmp_path)

    result = cli.main(
        [
            "--spec",
            str(paths["spec"]),
            "--visual-glb",
            str(paths["visual"]),
            "--actions-npz",
            str(paths["actions"]),
            "--rebase-report",
            str(paths["rebase"]),
            "--emitter-anchors-output",
            str(paths["emitter"]),
            "--contact-phases-output",
            str(paths["contacts"]),
        ]
    )

    assert result == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["schema"] == VARIANT_CONTACT_DERIVATION_SCHEMA
    assert summary["derivation_status"] == "completed"
    assert summary["qualification_claim"] is False
    assert paths["emitter"].is_file()
    assert paths["contacts"].is_file()
