"""Preserve an admitted actor frame while deriving an appearance visual.

The M2 visual GLB is already in the canonical skin frame, while its
``actor_from_canonical_root`` placement lives in the package's hash-bound
joint mapping.  Re-running root discovery on that already-canonical GLB loses
the actor placement.  This module instead proves that an appearance export
preserved the rig, copies its exact bytes, and emits an identity rebase report
that retains the admitted actor transform.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from avengine.contracts.json_io import load_json, sha256_file
from avengine.m2.action_rebind import (
    APPEARANCE_REPORT_SCHEMA,
    REBASE_REPORT_SCHEMA,
    ActionRebindError,
    _absolute_without_symlinks,
    _load_source_package,
    _package_role,
    _record,
    _regular_file,
    _requested_size_scale,
    _require_reference,
    verify_appearance_glb_compatibility,
)
from avengine.m2.glb import GlbError, load_glb
from avengine.m2.habitat import (
    HabitatMappingError,
    build_habitat_asset_mapping,
    build_habitat_asset_mapping_from_rebase_report,
)


DERIVATION_SCHEMA = "avengine_m2_canonical_appearance_visual_rebind_v1"


class AppearanceVisualError(RuntimeError):
    """An appearance GLB is not proven compatible with the package actor frame."""


@dataclass(frozen=True)
class CanonicalAppearanceVisualResult:
    """Exact appearance bytes and an in-memory identity rebase report."""

    visual_bytes: bytes
    report: Mapping[str, Any]


def _finite_matrix(value: Any, *, owner: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise AppearanceVisualError(f"{owner} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-8):
        raise AppearanceVisualError(f"{owner} must be affine")
    linear = matrix[:3, :3]
    if (
        float(np.max(np.abs(linear.T @ linear - np.eye(3)))) > 5.0e-5
        or abs(float(np.linalg.det(linear)) - 1.0) > 5.0e-5
    ):
        raise AppearanceVisualError(f"{owner} must be a proper rigid transform")
    return matrix


def compose_actor_from_normalized_root(
    actor_from_source_canonical: Any,
    normalized_canonical_from_source: Any,
) -> tuple[tuple[float, ...], ...]:
    """Compose the retained actor frame with one proper rigid rebase."""

    actor = _finite_matrix(
        actor_from_source_canonical,
        owner="actor_from_source_canonical",
    )
    canonical = _finite_matrix(
        normalized_canonical_from_source,
        owner="normalized_canonical_from_source",
    )
    result = actor @ np.linalg.inv(canonical)
    _finite_matrix(result, owner="composed actor_from_normalized_root")
    return tuple(tuple(float(component) for component in row) for row in result)


def _strict_appearance_report(
    path: str | Path,
    *,
    source_visual: Path,
) -> tuple[Path, Path, Mapping[str, Any], Mapping[str, Any]]:
    report_path = _regular_file(path, owner="appearance report", suffix=".json")
    report = load_json(report_path)
    if (
        report.get("schema") != APPEARANCE_REPORT_SCHEMA
        or report.get("status") != "pass"
        or report.get("qualification_claim") is not False
        or report.get("formal_dataset_registration_authorized") is not False
    ):
        raise AppearanceVisualError("appearance report is not a non-qualifying pass")
    try:
        _require_reference(
            report.get("source"),
            owner="appearance report source",
            path=source_visual,
            sha256=sha256_file(source_visual),
            byte_size=source_visual.stat().st_size,
        )
    except ActionRebindError as exc:
        raise AppearanceVisualError(str(exc)) from exc
    output = report.get("output")
    glb = output.get("glb") if isinstance(output, Mapping) else None
    if not isinstance(glb, Mapping) or not isinstance(glb.get("path"), str):
        raise AppearanceVisualError("appearance report lacks its output GLB")
    appearance_visual = _regular_file(
        glb["path"], owner="appearance output GLB", suffix=".glb"
    )
    try:
        _require_reference(
            glb,
            owner="appearance output GLB",
            path=appearance_visual,
            sha256=sha256_file(appearance_visual),
            byte_size=appearance_visual.stat().st_size,
        )
    except ActionRebindError as exc:
        raise AppearanceVisualError(str(exc)) from exc
    tool = report.get("tool_identity")
    if not isinstance(tool, Mapping) or not isinstance(tool.get("path"), str):
        raise AppearanceVisualError("appearance report lacks tool identity")
    tool_path = _regular_file(tool["path"], owner="appearance tool", suffix=".py")
    if tool.get("sha256") != sha256_file(tool_path):
        raise AppearanceVisualError("appearance tool bytes changed")
    try:
        compatibility = verify_appearance_glb_compatibility(
            source_visual,
            appearance_visual,
            requested_size_scale=_requested_size_scale(report),
        )
    except ActionRebindError as exc:
        raise AppearanceVisualError(str(exc)) from exc
    return report_path, appearance_visual, report, compatibility


def build_canonical_appearance_visual(
    *,
    source_package_manifest: str | Path,
    appearance_report: str | Path,
    normalized_visual_glb: str | Path,
    normalized_rebase_report: str | Path,
) -> CanonicalAppearanceVisualResult:
    """Return exact appearance bytes with the package's admitted actor mapping."""

    try:
        package_path, source_visual, _source_actions, manifest = _load_source_package(
            source_package_manifest
        )
        joint_mapping_path = _package_role(
            package_path, manifest, "habitat_joint_mapping"
        )
    except ActionRebindError as exc:
        raise AppearanceVisualError(str(exc)) from exc
    report_path, appearance_visual, report, compatibility = _strict_appearance_report(
        appearance_report,
        source_visual=source_visual,
    )
    normalized_visual = _regular_file(
        normalized_visual_glb,
        owner="normalized appearance visual",
        suffix=".glb",
    )
    normalized_rebase_path = _regular_file(
        normalized_rebase_report,
        owner="normalized appearance rebase report",
        suffix=".json",
    )
    normalized_rebase = load_json(normalized_rebase_path)
    if (
        normalized_rebase.get("schema") != REBASE_REPORT_SCHEMA
        or normalized_rebase.get("status") != "pass"
        or normalized_rebase.get("qualification_claim") is not False
    ):
        raise AppearanceVisualError(
            "normalized appearance rebase report is not a research pass"
        )
    try:
        _require_reference(
            normalized_rebase.get("source"),
            owner="normalized rebase source",
            path=appearance_visual,
            sha256=sha256_file(appearance_visual),
            byte_size=appearance_visual.stat().st_size,
        )
        _require_reference(
            normalized_rebase.get("output"),
            owner="normalized rebase output",
            path=normalized_visual,
            sha256=sha256_file(normalized_visual),
            byte_size=normalized_visual.stat().st_size,
        )
    except ActionRebindError as exc:
        raise AppearanceVisualError(str(exc)) from exc
    mapping_value = load_json(joint_mapping_path)
    if mapping_value.get("source_glb_sha256") != sha256_file(source_visual):
        raise AppearanceVisualError(
            "source package joint mapping does not bind its visual GLB"
        )
    actor_from_canonical = _finite_matrix(
        mapping_value.get("actor_from_skin_root"),
        owner="source package actor_from_skin_root",
    )
    try:
        source_document = load_glb(source_visual)
        normalized_document = load_glb(normalized_visual)
        source_mapping = build_habitat_asset_mapping(
            source_document,
            actor_from_skin_root=actor_from_canonical.tolist(),
            actor_from_skin_root_source=mapping_value.get(
                "actor_from_skin_root_source"
            ),
        )
        normalized_skin = normalized_rebase.get("skin")
        if not isinstance(normalized_skin, Mapping):
            raise AppearanceVisualError("normalized rebase report lacks skin data")
        corrected_actor = np.asarray(
            compose_actor_from_normalized_root(
                actor_from_canonical,
                normalized_skin.get("canonical_root_from_source_bind"),
            ),
            dtype=np.float64,
        )
        normalized_mapping = build_habitat_asset_mapping(
            normalized_document,
            actor_from_skin_root=corrected_actor.tolist(),
            actor_from_skin_root_source=mapping_value.get(
                "actor_from_skin_root_source"
            ),
        )
    except (OSError, GlbError, HabitatMappingError, ValueError) as exc:
        raise AppearanceVisualError(
            f"source/appearance visual is invalid: {exc}"
        ) from exc
    if source_mapping.joint_mapping_data() != mapping_value:
        raise AppearanceVisualError(
            "source package joint mapping differs from reconstructed visual mapping"
        )
    source_joint_order = source_mapping.joint_order
    normalized_joint_order = normalized_mapping.joint_order
    if normalized_joint_order != source_joint_order:
        raise AppearanceVisualError("normalized appearance changed skin joint order")
    request = report.get("instance_request")
    if not isinstance(request, Mapping) or not isinstance(
        request.get("request_sha256"), str
    ):
        raise AppearanceVisualError("appearance report lacks a request binding")

    visual_bytes = normalized_visual.read_bytes()
    visual_sha256 = sha256_file(normalized_visual)
    rebase_report = deepcopy(normalized_rebase)
    rebase_report["output"] = {
        "path": None,
        "sha256": visual_sha256,
        "byte_size": len(visual_bytes),
    }
    rebound_skin = dict(rebase_report["skin"])
    rebound_skin["actor_from_canonical_root"] = corrected_actor.tolist()
    rebase_report["skin"] = rebound_skin
    runtime_contract = dict(rebase_report.get("runtime_contract", {}))
    runtime_contract["actor_root_transform_source"] = (
        "source_package_actor_frame_composed_with_normalized_rebase"
    )
    rebase_report["runtime_contract"] = runtime_contract
    rebase_report["derivation"] = {
        "schema": DERIVATION_SCHEMA,
        "method": "compose_source_package_actor_frame_with_normalized_rebase",
        "source_package_manifest": _record(package_path),
        "source_visual_glb": _record(source_visual),
        "source_joint_mapping": _record(joint_mapping_path),
        "appearance_report": _record(report_path),
        "appearance_request_sha256": request["request_sha256"],
        "normalized_visual_glb": _record(normalized_visual),
        "normalized_rebase_report": _record(normalized_rebase_path),
        "appearance_glb_compatibility": compatibility,
        "tool_identity": _record(Path(__file__).resolve()),
        "composition": (
            "actor_from_new_canonical = actor_from_source_canonical "
            "@ inverse(new_canonical_from_source)"
        ),
        "compatibility_gates": {
            "source_package_valid": True,
            "source_joint_mapping_reconstructed_exactly": True,
            "appearance_source_is_package_visual": True,
            "skin_joint_order_exact": True,
            "topology_and_joint_indices_preserved": True,
            "normalized_rebase_report_pass": True,
            "actor_from_canonical_root_preserved": True,
        },
    }
    rebase_report["notes"] = [
        "The source package visual is already canonical; root discovery is not run again.",
        "The output GLB is the exact normalized appearance visual bytes.",
        "The admitted actor frame is composed with the normalization transform instead of replaced.",
        "This is research derivation evidence and does not qualify the appearance asset.",
    ]
    return CanonicalAppearanceVisualResult(
        visual_bytes=visual_bytes,
        report=rebase_report,
    )


def _output(path: str | Path, *, owner: str, suffix: str) -> Path:
    try:
        output = _absolute_without_symlinks(path, owner=owner)
    except ActionRebindError as exc:
        raise AppearanceVisualError(str(exc)) from exc
    if output.suffix.lower() != suffix:
        raise AppearanceVisualError(f"{owner} must use the {suffix} suffix")
    if output.exists() or output.is_symlink():
        raise AppearanceVisualError(f"refusing to replace {owner}: {output}")
    return output


def write_canonical_appearance_visual(
    result: CanonicalAppearanceVisualResult,
    *,
    visual_output: str | Path,
    report_output: str | Path,
) -> tuple[Path, Path]:
    """Exclusively emit and read back the visual/rebase pair."""

    if not isinstance(result, CanonicalAppearanceVisualResult):
        raise AppearanceVisualError(
            "result must come from build_canonical_appearance_visual"
        )
    visual_path = _output(
        visual_output, owner="canonical appearance visual", suffix=".glb"
    )
    report_path = _output(
        report_output, owner="canonical appearance rebase report", suffix=".json"
    )
    report = dict(result.report)
    output_record = dict(report["output"])
    output_record["path"] = str(visual_path)
    report["output"] = output_record
    report_bytes = (
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    outputs = (
        (visual_path, result.visual_bytes),
        (report_path, report_bytes),
    )
    streams: list[tuple[Path, Any]] = []
    try:
        for path, _payload in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
        for path, _payload in outputs:
            streams.append((path, path.open("xb")))
        for (_path, stream), (_expected, payload) in zip(streams, outputs, strict=True):
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        for _path, stream in streams:
            stream.close()
        for path, _stream in streams:
            path.unlink(missing_ok=True)
        raise AppearanceVisualError(
            f"unable to emit canonical appearance pair: {exc}"
        ) from exc
    finally:
        for _path, stream in streams:
            if not stream.closed:
                stream.close()

    try:
        document = load_glb(visual_path)
        emitted_report = load_json(report_path)
        mapping = build_habitat_asset_mapping_from_rebase_report(
            document, emitted_report
        )
        if document.sha256 != report["output"]["sha256"]:
            raise AppearanceVisualError("emitted visual hash differs from report")
        if emitted_report != report:
            raise AppearanceVisualError("emitted rebase report differs on readback")
        expected_actor = np.asarray(
            report["skin"]["actor_from_canonical_root"], dtype=np.float64
        )
        if not np.allclose(mapping.actor_from_skin_root, expected_actor, atol=1.0e-12):
            raise AppearanceVisualError("emitted actor mapping changed")
    except AppearanceVisualError:
        visual_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        raise
    except (OSError, ValueError, GlbError, HabitatMappingError) as exc:
        visual_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        raise AppearanceVisualError(
            f"canonical appearance readback failed: {exc}"
        ) from exc
    return visual_path, report_path


__all__ = [
    "DERIVATION_SCHEMA",
    "AppearanceVisualError",
    "CanonicalAppearanceVisualResult",
    "build_canonical_appearance_visual",
    "compose_actor_from_normalized_root",
    "write_canonical_appearance_visual",
]
