from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from avengine.contracts.json_io import sha256_file
from avengine.assets import appearance_visual
from avengine.assets.action_rebind import ActionRebindError
from avengine.assets.appearance_visual import (
    AppearanceVisualError,
    compose_actor_from_normalized_root,
)


def _yaw(radians: float, translation: tuple[float, float, float]) -> np.ndarray:
    cosine = math.cos(radians)
    sine = math.sin(radians)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = [
        [cosine, 0.0, sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, cosine],
    ]
    result[:3, 3] = translation
    return result


def test_compose_actor_preserves_world_mapping() -> None:
    actor_from_source = _yaw(0.7, (1.0, 2.0, -0.5))
    normalized_from_source = _yaw(-0.2, (0.2, -0.1, 0.4))

    actor_from_normalized = np.asarray(
        compose_actor_from_normalized_root(
            actor_from_source,
            normalized_from_source,
        )
    )

    np.testing.assert_allclose(
        actor_from_normalized @ normalized_from_source,
        actor_from_source,
        atol=1.0e-12,
    )


def test_compose_actor_rejects_scale_as_a_rigid_rebase() -> None:
    scaled = np.eye(4, dtype=np.float64)
    scaled[0, 0] = 1.1

    with pytest.raises(AppearanceVisualError, match="proper rigid"):
        compose_actor_from_normalized_root(np.eye(4), scaled)


def _appearance_report_fixture(tmp_path: Path, *, include_claimed_gates: bool) -> tuple:
    source = tmp_path / "source.glb"
    output = tmp_path / "output.glb"
    tool = tmp_path / "realize.py"
    report_path = tmp_path / "appearance_report.json"
    source.write_bytes(b"source-glb-placeholder")
    output.write_bytes(b"output-glb-placeholder")
    tool.write_text("# fixture\n", encoding="utf-8")
    report: dict[str, Any] = {
        "schema": "avengine_animal_appearance_realization_v1",
        "status": "pass",
        "qualification_claim": False,
        "formal_dataset_registration_authorized": False,
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
            "byte_size": source.stat().st_size,
        },
        "instance_request": {
            "realization_operations": [
                {
                    "attribute": "size",
                    "operation_id": "uniform_actor_scale_v1",
                    "parameters": {"scale_ratio": 1.18},
                }
            ]
        },
        "output": {
            "glb": {
                "path": str(output),
                "sha256": sha256_file(output),
                "byte_size": output.stat().st_size,
            }
        },
        "tool_identity": {"path": str(tool), "sha256": sha256_file(tool)},
    }
    if include_claimed_gates:
        report["output"]["readback_audit"] = {
            "mesh_invariants": {"indices_exact": True, "joints_0_exact": True},
            "skin_invariants": {"joint_order_unchanged": True},
            "action_invariants": {
                "channel_targets_unchanged": True,
                "translations_scaled_by_size": True,
            },
        }
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    return report_path, source, output


def test_appearance_report_supplies_scale_but_not_compatibility_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path, source, output = _appearance_report_fixture(
        tmp_path, include_claimed_gates=False
    )
    observed: dict[str, Any] = {}

    def verify(
        source_path: Path,
        output_path: Path,
        *,
        requested_size_scale: float,
    ) -> dict[str, Any]:
        observed.update(
            source=source_path,
            output=output_path,
            scale=requested_size_scale,
        )
        return {"independently_measured": True}

    monkeypatch.setattr(
        appearance_visual, "verify_appearance_glb_compatibility", verify
    )
    _report, _visual, _value, audit = appearance_visual._strict_appearance_report(
        report_path,
        source_visual=source,
    )

    assert audit == {"independently_measured": True}
    assert observed == {"source": source, "output": output, "scale": 1.18}


def test_claimed_report_booleans_cannot_override_failed_glb_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path, source, _output = _appearance_report_fixture(
        tmp_path, include_claimed_gates=True
    )

    def reject(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise ActionRebindError("independent topology measurement failed")

    monkeypatch.setattr(
        appearance_visual, "verify_appearance_glb_compatibility", reject
    )
    with pytest.raises(AppearanceVisualError, match="independent topology"):
        appearance_visual._strict_appearance_report(
            report_path,
            source_visual=source,
        )
