from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from avengine.contracts.json_io import load_json


QUALIFICATION_REPORT_SCHEMA = "avengine_m6_room_qualification_report_v1"
EXECUTION_STATUS_VALUES = {"pass", "fail", "blocked", "not_run"}
MATERIAL_TRUTH_VALUES = {
    "controlled_profile",
    "measured",
    "controlled_approximation",
    "semantic_mapping_approximation",
    "unqualified",
    "none",
}
ADMISSIBLE_MATERIAL_TRUTH = {
    "controlled_profile",
    "measured",
    "controlled_approximation",
    "semantic_mapping_approximation",
}
REQUIRED_ADMISSION_DIMENSIONS = (
    "visual_runtime_status",
    "navigation_status",
    "acoustic_geometry_status",
    "material_binding_status",
    "ray_leakage_status",
    "episode_feasibility_status",
)
REQUIRED_ACOUSTIC_DIAGNOSTICS = (
    "raw_source_identity",
    "declared_derivation_integrity",
    "visual_to_acoustic_spatial_parity",
    "solver_loadability",
    "topology_diagnostics",
    "opening_and_enclosure_checks",
)
SUPPORT_SAMPLE_IDS = {
    "center",
    "front_left",
    "front_right",
    "rear_left",
    "rear_right",
}


class QualificationContractError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


class PlacementProbeError(ValueError):
    pass


@dataclass(frozen=True)
class AdmissionDecision:
    eligible: bool
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class CorruptedFixtureQualification:
    report: dict[str, Any]
    findings: tuple[str, ...]


def _schema_path(filename: str) -> Path:
    source_path = Path(__file__).resolve().parents[3] / "schemas" / filename
    installed_path = Path(sys.prefix) / "share" / "avengine" / "schemas" / filename
    return source_path if source_path.is_file() else installed_path


def _json_schema_errors(value: Any, filename: str) -> list[str]:
    schema = load_json(_schema_path(filename))
    validator = Draft202012Validator(schema)
    errors: list[str] = []
    for error in sorted(validator.iter_errors(value), key=lambda item: list(item.path)):
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        errors.append(f"JSON Schema {location}: {error.message}")
    return errors


def compute_dataset_admission(report: Mapping[str, Any]) -> AdmissionDecision:
    """Compute eligibility without collapsing the independent qualification states."""

    blockers: list[str] = []
    dimensions = report.get("dimensions", {})
    for name in REQUIRED_ADMISSION_DIMENSIONS:
        status = dimensions.get(name, {}).get("status")
        if status != "pass":
            blockers.append(f"{name}={status or 'missing'}")

    truth = dimensions.get("physical_material_truth_status", {}).get("status")
    if truth not in ADMISSIBLE_MATERIAL_TRUTH:
        blockers.append(f"physical_material_truth_status={truth or 'missing'}")

    placement_status = report.get("placement_feasibility", {}).get("status")
    if placement_status != "pass":
        blockers.append(f"placement_feasibility={placement_status or 'missing'}")

    diagnostics = report.get("acoustic_diagnostics", {})
    for name in REQUIRED_ACOUSTIC_DIAGNOSTICS:
        status = diagnostics.get(name, {}).get("status")
        if status != "pass":
            blockers.append(f"acoustic_diagnostics.{name}={status or 'missing'}")

    return AdmissionDecision(not blockers, tuple(blockers))


def validate_qualification_report(value: Any) -> list[str]:
    errors = _json_schema_errors(
        value, "m6_room_qualification_report_v1.schema.json"
    )
    if errors or not isinstance(value, dict):
        return errors

    if "overall_status" in value:
        errors.append("overall_status is forbidden; qualification dimensions stay separate")
    if value["evidence_basis"] == "current_execution" and not value["evidence_artifacts"]:
        errors.append("current_execution reports require hashed evidence_artifacts")

    dimensions = value["dimensions"]
    for name in REQUIRED_ADMISSION_DIMENSIONS:
        check = dimensions[name]
        if check["status"] == "pass" and "blocker_code" in check:
            errors.append(f"dimensions.{name}: pass status cannot carry blocker_code")
        if check["status"] != "pass" and "blocker_code" not in check:
            errors.append(f"dimensions.{name}: non-pass status requires blocker_code")

    truth_check = dimensions["physical_material_truth_status"]
    if truth_check["status"] in {"unqualified", "none"}:
        if "blocker_code" not in truth_check:
            errors.append(
                "dimensions.physical_material_truth_status: unqualified/none requires "
                "blocker_code"
            )

    for name in REQUIRED_ACOUSTIC_DIAGNOSTICS:
        check = value["acoustic_diagnostics"][name]
        if check["status"] == "pass" and "blocker_code" in check:
            errors.append(
                f"acoustic_diagnostics.{name}: pass status cannot carry blocker_code"
            )
        if check["status"] != "pass" and "blocker_code" not in check:
            errors.append(
                f"acoustic_diagnostics.{name}: non-pass status requires blocker_code"
            )

    placement = value["placement_feasibility"]
    if placement["status"] == "pass" and "blocker_code" in placement:
        errors.append("placement_feasibility: pass status cannot carry blocker_code")
    if placement["status"] != "pass" and "blocker_code" not in placement:
        errors.append("placement_feasibility: non-pass status requires blocker_code")

    decision = compute_dataset_admission(value)
    declared_admission = value["dataset_admission"]
    declared_blockers = value["admission_blockers"]
    if declared_admission and not decision.eligible:
        errors.append(
            "dataset_admission=true is invalid while required states are not pass: "
            + ", ".join(decision.blockers)
        )
    if declared_admission and declared_blockers:
        errors.append("admitted reports cannot declare admission_blockers")
    if not declared_admission and not declared_blockers:
        errors.append("dataset_admission=false requires at least one admission blocker")

    subject = value["subject"]
    scope = subject["qualification_scope"]
    representation_kind = subject.get("acoustic_representation_kind")
    if scope == "acoustic_representation" and representation_kind is None:
        errors.append(
            "acoustic-representation reports require acoustic_representation_kind"
        )
    if representation_kind == "derived_proxy" and (
        "declared_derivation_integrity" not in value["acoustic_diagnostics"]
    ):
        errors.append("derived proxy must report declared_derivation_integrity separately")
    return errors


def load_qualification_report(path: str | Path) -> dict[str, Any]:
    value = load_json(Path(path).resolve())
    errors = validate_qualification_report(value)
    if errors:
        raise QualificationContractError(errors)
    return value


def build_qualification_report(
    *,
    report_id: str,
    subject: Mapping[str, Any],
    evidence_basis: str,
    evidence_artifacts: Sequence[Mapping[str, Any]] = (),
    dimensions: Mapping[str, Any],
    placement_feasibility: Mapping[str, Any],
    acoustic_diagnostics: Mapping[str, Any],
    provenance: Mapping[str, Any],
    promote_if_eligible: bool = False,
) -> dict[str, Any]:
    """Build and validate a report from measured provider/runtime states.

    Eligibility and promotion are intentionally separate. A fully passing report
    remains not admitted unless the caller explicitly requests promotion.
    """

    report: dict[str, Any] = {
        "schema": QUALIFICATION_REPORT_SCHEMA,
        "report_id": report_id,
        "subject": deepcopy(dict(subject)),
        "evidence_basis": evidence_basis,
        "evidence_artifacts": deepcopy(list(evidence_artifacts)),
        "dimensions": deepcopy(dict(dimensions)),
        "placement_feasibility": deepcopy(dict(placement_feasibility)),
        "acoustic_diagnostics": deepcopy(dict(acoustic_diagnostics)),
        "dataset_admission": False,
        "admission_blockers": ["release_promotion_not_requested"],
        "provenance": deepcopy(dict(provenance)),
    }
    decision = compute_dataset_admission(report)
    if promote_if_eligible and decision.eligible:
        report["dataset_admission"] = True
        report["admission_blockers"] = []
    elif decision.blockers:
        report["admission_blockers"] = list(decision.blockers)

    errors = validate_qualification_report(report)
    if errors:
        raise QualificationContractError(errors)
    return report


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlacementProbeError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise PlacementProbeError(f"{field} must be finite and non-negative")
    return number


def evaluate_placement_feasibility(probe: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate five-point support, body clearance and frustum leakage probes.

    The caller owns ray generation. This pure evaluator deliberately accepts
    measured ray outcomes so native Habitat, a deterministic fixture, or a future
    provider can share the same fail-closed policy.
    """

    thresholds = probe.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise PlacementProbeError("thresholds must be an object")
    max_support_gap = _finite_nonnegative(
        thresholds.get("maximum_support_gap_m"), "maximum_support_gap_m"
    )
    minimum_clearance = _finite_nonnegative(
        thresholds.get("minimum_horizontal_clearance_m"),
        "minimum_horizontal_clearance_m",
    )

    support_samples = probe.get("support_samples")
    if not isinstance(support_samples, Sequence) or isinstance(
        support_samples, (str, bytes)
    ):
        raise PlacementProbeError("support_samples must be an array")
    support_by_id: dict[str, Mapping[str, Any]] = {}
    for sample in support_samples:
        if not isinstance(sample, Mapping) or not isinstance(sample.get("sample_id"), str):
            raise PlacementProbeError("each support sample requires sample_id")
        if sample["sample_id"] in support_by_id:
            raise PlacementProbeError(f"duplicate support sample {sample['sample_id']!r}")
        support_by_id[sample["sample_id"]] = sample
    if set(support_by_id) != SUPPORT_SAMPLE_IDS:
        raise PlacementProbeError(
            "support samples must be exactly center/front_left/front_right/"
            "rear_left/rear_right"
        )

    clearance_samples = probe.get("horizontal_clearance_samples")
    if not isinstance(clearance_samples, Sequence) or isinstance(
        clearance_samples, (str, bytes)
    ) or len(clearance_samples) < 4:
        raise PlacementProbeError("at least four horizontal clearance samples are required")

    frustum_samples = probe.get("frustum_samples")
    if not isinstance(frustum_samples, Sequence) or isinstance(
        frustum_samples, (str, bytes)
    ) or not frustum_samples:
        raise PlacementProbeError("at least one frustum sample is required")

    whitelist_raw = probe.get("allowed_opening_ids", [])
    if not isinstance(whitelist_raw, Sequence) or isinstance(whitelist_raw, (str, bytes)):
        raise PlacementProbeError("allowed_opening_ids must be an array")
    whitelist = set(whitelist_raw)
    if len(whitelist) != len(whitelist_raw) or not all(
        isinstance(item, str) and item for item in whitelist
    ):
        raise PlacementProbeError("allowed_opening_ids must contain unique non-empty strings")

    checks: list[dict[str, Any]] = []
    reasons: list[str] = []
    for sample_id in sorted(SUPPORT_SAMPLE_IDS):
        sample = support_by_id[sample_id]
        hit = sample.get("hit")
        if not isinstance(hit, bool):
            raise PlacementProbeError(f"support sample {sample_id} requires boolean hit")
        distance = None
        passed = False
        if hit:
            distance = _finite_nonnegative(
                sample.get("distance_m"), f"support sample {sample_id}.distance_m"
            )
            passed = distance <= max_support_gap
        if not passed:
            reasons.append(
                f"support:{sample_id}:"
                + ("no_floor_hit" if not hit else f"gap_{distance:.6g}_m")
            )
        checks.append(
            {
                "check_id": f"support.{sample_id}",
                "passed": passed,
                "measured": {"hit": hit, "distance_m": distance},
                "threshold": {"maximum_support_gap_m": max_support_gap},
            }
        )

    seen_clearance: set[str] = set()
    for sample in clearance_samples:
        if not isinstance(sample, Mapping) or not isinstance(sample.get("direction_id"), str):
            raise PlacementProbeError("each clearance sample requires direction_id")
        direction_id = sample["direction_id"]
        if direction_id in seen_clearance:
            raise PlacementProbeError(f"duplicate clearance direction {direction_id!r}")
        seen_clearance.add(direction_id)
        raw_distance = sample.get("hit_distance_m")
        distance = (
            None
            if raw_distance is None
            else _finite_nonnegative(raw_distance, f"clearance {direction_id}.hit_distance_m")
        )
        passed = distance is None or distance >= minimum_clearance
        if not passed:
            reasons.append(f"clearance:{direction_id}:hit_{distance:.6g}_m")
        checks.append(
            {
                "check_id": f"clearance.{direction_id}",
                "passed": passed,
                "measured": {"hit_distance_m": distance},
                "threshold": {"minimum_horizontal_clearance_m": minimum_clearance},
            }
        )

    seen_frustum: set[str] = set()
    for sample in frustum_samples:
        if not isinstance(sample, Mapping) or not isinstance(sample.get("ray_id"), str):
            raise PlacementProbeError("each frustum sample requires ray_id")
        ray_id = sample["ray_id"]
        if ray_id in seen_frustum:
            raise PlacementProbeError(f"duplicate frustum ray {ray_id!r}")
        seen_frustum.add(ray_id)
        outcome = sample.get("outcome")
        opening_id = sample.get("opening_id")
        if outcome not in {"surface_hit", "opening_exit", "scene_escape"}:
            raise PlacementProbeError(f"frustum ray {ray_id} has invalid outcome {outcome!r}")
        if outcome == "surface_hit":
            passed = opening_id is None
        elif outcome == "opening_exit":
            passed = isinstance(opening_id, str) and opening_id in whitelist
        else:
            passed = False
        if not passed:
            reason = (
                f"unapproved_opening_{opening_id or 'missing'}"
                if outcome == "opening_exit"
                else f"unexpected_{outcome}"
            )
            reasons.append(f"frustum:{ray_id}:{reason}")
        checks.append(
            {
                "check_id": f"frustum.{ray_id}",
                "passed": passed,
                "measured": {"outcome": outcome, "opening_id": opening_id},
                "threshold": {"allowed_opening_ids": sorted(whitelist)},
            }
        )

    passed = not reasons
    result: dict[str, Any] = {
        "status": "pass" if passed else "fail",
        "summary": (
            "all placement support, clearance, and frustum checks passed"
            if passed
            else "placement probe rejected one or more unsafe positions"
        ),
        "evidence_refs": [],
        "checks": checks,
        "failure_reasons": reasons,
    }
    if not passed:
        result["blocker_code"] = "placement_probe_failed"
    return result


def _triangle_area(vertices: Sequence[Sequence[float]], triangle: Sequence[int]) -> float:
    a, b, c = (vertices[index] for index in triangle)
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    return 0.5 * math.sqrt(sum(component * component for component in cross))


def _status_check(status: str, summary: str, blocker_code: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"status": status, "summary": summary, "evidence_refs": []}
    if blocker_code is not None:
        value["blocker_code"] = blocker_code
    return value


def qualify_corrupted_acoustic_fixture(
    fixture: Mapping[str, Any],
) -> CorruptedFixtureQualification:
    """Inspect the deliberately bad, JSON-only acoustic regression fixture."""

    geometry = fixture["geometry"]
    vertices = geometry["vertices_m"]
    triangles = geometry["triangles"]
    zero_area = [
        index
        for index, triangle in enumerate(triangles)
        if _triangle_area(vertices, triangle) <= 1e-12
    ]

    material_ids = fixture["materials"]["triangle_material_ids"]
    fallback_ids = set(fixture["materials"]["fallback_material_ids"])
    material_failures: list[str] = []
    if len(material_ids) != len(triangles):
        material_failures.append("triangle_material_count_mismatch")
    if any(material_id in fallback_ids for material_id in material_ids):
        material_failures.append("fallback_material_present")

    integrity = fixture["integrity"]
    identity_match = integrity["declared_source_sha256"] == integrity["observed_source_sha256"]
    failed_rays = [
        check["check_id"]
        for check in fixture["ray_checks"]
        if check["expected"] != check["observed"]
    ]

    findings: list[str] = []
    if zero_area:
        findings.append(f"zero_area_triangles={zero_area}")
    findings.extend(material_failures)
    if not identity_match:
        findings.append("source_identity_mismatch")
    if failed_rays:
        findings.append(f"failed_ray_checks={failed_rays}")

    geometry_status = "fail" if zero_area else "pass"
    material_status = "fail" if material_failures else "pass"
    ray_status = "fail" if failed_rays else "pass"
    identity_status = "pass" if identity_match else "fail"

    report: dict[str, Any] = {
        "schema": QUALIFICATION_REPORT_SCHEMA,
        "report_id": f"{fixture['fixture_id']}_qualification_v1",
        "subject": {
            "room_id": fixture["fixture_id"],
            "revision": "fixture_v1",
            "qualification_scope": "corrupted_fixture",
            "acoustic_representation_id": "deliberately_corrupted_package",
            "acoustic_representation_kind": "corrupted_fixture",
        },
        "evidence_basis": "contract_fixture",
        "evidence_artifacts": [],
        "dimensions": {
            "visual_runtime_status": _status_check(
                "not_run", "fixture has no visual admission claim", "fixture_visual_not_run"
            ),
            "navigation_status": _status_check(
                "not_run", "fixture has no navigation admission claim", "fixture_nav_not_run"
            ),
            "acoustic_geometry_status": _status_check(
                geometry_status,
                f"zero-area triangle indices: {zero_area}",
                "zero_area_geometry" if zero_area else None,
            ),
            "material_binding_status": _status_check(
                material_status,
                ", ".join(material_failures) or "material IDs cover all triangles",
                "invalid_material_binding" if material_failures else None,
            ),
            "ray_leakage_status": _status_check(
                ray_status,
                f"failed ray checks: {failed_rays}",
                "ray_expectation_failed" if failed_rays else None,
            ),
            "physical_material_truth_status": _status_check(
                "none", "corrupted fixture has no physical material truth", "no_material_truth"
            ),
            "episode_feasibility_status": _status_check(
                "fail", "corrupted acoustic package is ineligible for episodes", "fixture_rejected"
            ),
        },
        "placement_feasibility": {
            **_status_check(
                "not_run", "fixture is independent of a room placement", "fixture_placement_not_run"
            ),
            "checks": [],
            "failure_reasons": ["fixture_has_no_room_placement"],
        },
        "acoustic_diagnostics": {
            "raw_source_identity": _status_check(
                identity_status,
                "declared and observed source hashes "
                + ("match" if identity_match else "do not match"),
                "source_hash_mismatch" if not identity_match else None,
            ),
            "declared_derivation_integrity": _status_check(
                "not_run", "fixture is not an admitted derivation", "derivation_not_applicable"
            ),
            "visual_to_acoustic_spatial_parity": _status_check(
                "fail", "corrupted geometry has no admissible spatial parity", "spatial_parity_failed"
            ),
            "solver_loadability": _status_check(
                "fail", "zero-area geometry is rejected before solver upload", "solver_upload_rejected"
            ),
            "topology_diagnostics": _status_check(
                geometry_status,
                f"zero-area triangle indices: {zero_area}",
                "topology_invalid" if zero_area else None,
            ),
            "opening_and_enclosure_checks": _status_check(
                ray_status,
                f"failed ray checks: {failed_rays}",
                "opening_or_enclosure_check_failed" if failed_rays else None,
            ),
        },
        "dataset_admission": False,
        "admission_blockers": [],
        "provenance": {
            "source_records": [fixture["fixture_id"]],
            "notes": "Independent corrupted fixture; never a real-room registry member.",
        },
    }
    decision = compute_dataset_admission(report)
    report["admission_blockers"] = list(decision.blockers)
    return CorruptedFixtureQualification(report=deepcopy(report), findings=tuple(findings))
