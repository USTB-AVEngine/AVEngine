from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from avengine.assets.kinematics import CONTACT_ORDER
from tools.motion import derive_contacts as cli


IDENTITY_MATRIX = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _anchors(visual_sha256: str) -> dict[str, Any]:
    return {
        "schema": cli.ANCHOR_PROFILE_SCHEMA,
        "source_visual_sha256": visual_sha256,
        "anchors": [
            {
                "anchor_id": anchor_id,
                "joint_id": f"joint_{index}",
                "joint_from_anchor": {
                    "translation_m": [0.0, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            }
            for index, anchor_id in enumerate(CONTACT_ORDER)
        ],
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def test_derive_contacts_binds_all_inputs_and_writes_canonical_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    visual_payload = b"test visual glb"
    actions_payload = b"canonical baked actions"
    visual_sha256 = _sha256(visual_payload)
    visual = tmp_path / "visual.glb"
    actions = tmp_path / "actions.npz"
    mapping_path = tmp_path / "joint_mapping.json"
    anchors_path = tmp_path / "anchors.json"
    output = tmp_path / "nested" / "contact_phases.json"
    visual.write_bytes(visual_payload)
    actions.write_bytes(actions_payload)
    mapping_value = {
        "schema": cli.JOINT_MAPPING_SCHEMA,
        "source_glb_sha256": visual_sha256,
        "actor_from_skin_root": IDENTITY_MATRIX,
        "actor_from_skin_root_source": "test.rebase.actor_from_skin_root",
    }
    _write_json(mapping_path, mapping_value)
    _write_json(anchors_path, _anchors(visual_sha256))

    document = SimpleNamespace(sha256=visual_sha256)
    habitat_mapping = SimpleNamespace(
        source_glb_sha256=visual_sha256,
        joint_mapping_data=lambda: mapping_value,
    )
    action_set = SimpleNamespace(source_glb_sha256=visual_sha256)
    report_payload = '{"schema":"avengine_m2_contact_phases_v1"}\n'
    report = SimpleNamespace(
        qualification_state="research_candidate",
        qualification_claim=False,
        to_canonical_json=lambda: report_payload,
        content_sha256=lambda: _sha256(report_payload.encode("utf-8")),
    )
    observed: dict[str, Any] = {}

    monkeypatch.setattr(cli, "load_glb", lambda path: document)

    def build_mapping(
        parsed_document: object,
        *,
        actor_from_skin_root: object,
        actor_from_skin_root_source: str,
    ) -> object:
        observed["mapping_args"] = (
            parsed_document,
            actor_from_skin_root,
            actor_from_skin_root_source,
        )
        return habitat_mapping

    monkeypatch.setattr(cli, "build_habitat_asset_mapping", build_mapping)
    monkeypatch.setattr(cli, "read_baked_actions_npz", lambda path: action_set)
    monkeypatch.setattr(
        cli, "baked_actions_content_sha256", lambda value: _sha256(actions_payload)
    )

    def derive(mapping: object, baked: object, anchors_value: object) -> object:
        observed["derive_args"] = (mapping, baked, anchors_value)
        return report

    monkeypatch.setattr(cli, "derive_contact_phases", derive)

    result = cli.derive_contacts(
        visual_glb=visual,
        actions_npz=actions,
        joint_mapping_json=mapping_path,
        anchor_profile_json=anchors_path,
        output_json=output,
    )

    assert output.read_text(encoding="utf-8") == report_payload
    assert result["output_sha256"] == _sha256(report_payload.encode("utf-8"))
    assert result["contact_order"] == list(CONTACT_ORDER)
    assert observed["mapping_args"] == (
        document,
        IDENTITY_MATRIX,
        "test.rebase.actor_from_skin_root",
    )
    derived_anchors = observed["derive_args"][2]
    assert tuple(anchor.anchor_id for anchor in derived_anchors) == CONTACT_ORDER


def test_contact_anchor_profile_rejects_hash_and_order_drift() -> None:
    visual_sha256 = "a" * 64
    wrong_hash = _anchors("b" * 64)
    with pytest.raises(
        cli.ContactDerivationCliError,
        match="source_visual_sha256 must match",
    ):
        cli._contact_anchors(wrong_hash, visual_sha256=visual_sha256)

    wrong_order = _anchors(visual_sha256)
    wrong_order["anchors"][2], wrong_order["anchors"][3] = (
        wrong_order["anchors"][3],
        wrong_order["anchors"][2],
    )
    with pytest.raises(
        cli.ContactDerivationCliError,
        match="must follow fixed M2 order",
    ):
        cli._contact_anchors(wrong_order, visual_sha256=visual_sha256)


def test_derive_contacts_rejects_joint_mapping_not_reconstructed_from_glb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    visual_payload = b"test visual glb"
    visual_sha256 = _sha256(visual_payload)
    visual = tmp_path / "visual.glb"
    actions = tmp_path / "actions.npz"
    mapping_path = tmp_path / "joint_mapping.json"
    anchors_path = tmp_path / "anchors.json"
    output = tmp_path / "contact_phases.json"
    visual.write_bytes(visual_payload)
    actions.write_bytes(b"actions")
    mapping_value = {
        "schema": cli.JOINT_MAPPING_SCHEMA,
        "source_glb_sha256": visual_sha256,
        "actor_from_skin_root": IDENTITY_MATRIX,
        "actor_from_skin_root_source": "test.rebase.actor_from_skin_root",
        "tampered": True,
    }
    _write_json(mapping_path, mapping_value)
    _write_json(anchors_path, _anchors(visual_sha256))
    monkeypatch.setattr(
        cli, "load_glb", lambda path: SimpleNamespace(sha256=visual_sha256)
    )
    monkeypatch.setattr(
        cli,
        "build_habitat_asset_mapping",
        lambda *args, **kwargs: SimpleNamespace(
            joint_mapping_data=lambda: {
                key: value for key, value in mapping_value.items() if key != "tampered"
            }
        ),
    )

    with pytest.raises(
        cli.ContactDerivationCliError,
        match="differs from the mapping reconstructed",
    ):
        cli.derive_contacts(
            visual_glb=visual,
            actions_npz=actions,
            joint_mapping_json=mapping_path,
            anchor_profile_json=anchors_path,
            output_json=output,
        )
    assert not output.exists()
