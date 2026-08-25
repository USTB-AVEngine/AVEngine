from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest

from avengine.appearance import build_l9_batch
from avengine.contracts.json_io import canonical_json_sha256, sha256_file
from avengine.assets import variant_package
from avengine.assets.glb import load_glb
from avengine.assets.glb_write import build_glb
from avengine.assets.materials import normalize_glb_materials
from tools.assets import assemble_variant_package as cli


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_SPEC = (
    REPOSITORY_ROOT
    / "examples/assets/variant_packages/rocketbox_beagle_review_spec_v1.json"
)
EXAMPLE_APPEARANCE_REQUEST = (
    REPOSITORY_ROOT / "examples/assets/appearance/beagle_l9_request_v1.json"
)
_REAL_L9_PRODUCER_VALIDATOR = variant_package._validate_l9_producer_contract


@pytest.fixture(autouse=True)
def _isolate_local_assembler_checks_from_full_producer_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep legacy synthetic evidence focused on the assembler's local gates.

    Complete producer/assembler integration uses the full realization fixture
    in ``test_assets_appearance_variant_inputs.py``.  The fixture below predates
    that report contract and is deliberately minimal.
    """

    monkeypatch.setattr(
        variant_package,
        "_validate_l9_producer_contract",
        lambda spec, lineage: None,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "byte_size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _json_binding(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        **_file_binding(path),
        "canonical_content_sha256": canonical_json_sha256(value),
        "snapshot": value,
    }


def _matte_document(*, opaque: bool = True) -> dict[str, Any]:
    return {
        "asset": {"version": "2.0", "generator": "variant-package-test"},
        "extensionsUsed": ["KHR_materials_specular"],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {}, "material": 0}]}],
        "materials": [
            {
                "name": "bounded_matte",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.4, 0.3, 0.2, 1.0 if opaque else 0.4],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.8,
                },
                "alphaMode": "OPAQUE" if opaque else "BLEND",
                "emissiveFactor": [0.0, 0.0, 0.0],
                "extensions": {
                    "KHR_materials_specular": {
                        "specularFactor": 0.2,
                        "specularColorFactor": [1.0, 1.0, 1.0],
                    }
                },
            }
        ],
    }


def _evidence(
    tmp_path: Path,
    *,
    force_opaque: bool = True,
) -> variant_package.VariantPackageEvidence:
    spec = variant_package.load_variant_package_spec(EXAMPLE_SPEC)
    pre_rebase = tmp_path / "appearance.glb"
    pre_rebase.write_bytes(build_glb(_matte_document(opaque=force_opaque), b""))
    visual = tmp_path / "visual.glb"
    normalization = normalize_glb_materials(
        pre_rebase,
        visual,
        force_opaque=force_opaque,
    )
    normalization_path = tmp_path / "material_normalization.json"
    _write_json(normalization_path, normalization)

    schemas = {
        "rebase_deformation_report": ("avengine_m2_rebase_deformation_verification_v1"),
        "action_report": "avengine_m2_action_bake_report_v1",
        "static_qa": "avengine_m2_static_geometry_qa_v1",
        "deformation_qa": "avengine_m2_deformation_qa_v1",
        "animation_qa": "avengine_m2_animation_qa_v1",
        "habitat_static_probe": "avengine_m2_habitat_skin_rest_probe_v1",
        "habitat_animation_review": "avengine_m2_habitat_action_review_v1",
    }
    paths: dict[str, Path] = {}
    for field_name, schema in schemas.items():
        path = tmp_path / f"{field_name}.json"
        _write_json(
            path,
            {
                "schema": schema,
                "status": "pass",
                "qualification_claim": False,
            },
        )
        paths[field_name] = path
    rebase_report = tmp_path / "rebase_report.json"
    _write_json(
        rebase_report,
        {
            "schema": "avengine_m2_skin_root_rebase_v1",
            "status": "pass",
            "qualification_claim": False,
            "source": _file_binding(pre_rebase),
            "output": _file_binding(visual),
        },
    )
    paths["rebase_report"] = rebase_report

    actions = tmp_path / "actions.npz"
    actions.write_bytes(b"actions")
    contacts = tmp_path / "contacts.json"
    source = tmp_path / "source.json"
    license_snapshot = tmp_path / "license.json"
    _write_json(contacts, {"schema": "avengine_m2_contact_phases_v1"})
    _write_json(
        source,
        {
            "schema": "source_v1",
            "formal_dataset_registration_authorized": False,
        },
    )
    _write_json(
        license_snapshot,
        {
            "schema": "license_v1",
            "license": "MIT",
            "allowed_use": "review_required",
            "redistribution": "review_required",
        },
    )
    request_value = json.loads(EXAMPLE_APPEARANCE_REQUEST.read_text(encoding="utf-8"))
    request_value["source_asset"]["expected_sha256"] = sha256_file(pre_rebase)
    request_path = tmp_path / "appearance_request.json"
    _write_json(request_path, request_value)
    batch = tmp_path / "appearance_batch.json"
    batch_value = build_l9_batch(
        request_value,
        request_file=request_path,
        source_file=pre_rebase,
    )
    _write_json(batch, batch_value)
    request = batch_value["requests"][0]
    realization = tmp_path / "appearance_realization.json"
    realizer = REPOSITORY_ROOT / "tools/blender/realize_animal_appearance.py"
    material_normalizer = REPOSITORY_ROOT / "src/avengine/assets/materials.py"
    tool_identity = {
        "path": str(realizer.resolve()),
        "sha256": sha256_file(realizer),
        "material_normalizer": {
            "path": str(material_normalizer.resolve()),
            "sha256": sha256_file(material_normalizer),
        },
        "blender_version": "4.2.1 LTS",
        "export_profile": {
            "format": "GLB",
            "animation_mode": "ACTIONS",
            "force_sampling": True,
            "skins": True,
            "texcoords": True,
            "normals": True,
            "image_format": "AUTO",
        },
        "output_readback_float_tolerance": 5.0e-5,
    }
    realization_value = {
        "schema": "avengine_animal_appearance_realization_v1",
        "status": "pass",
        "state_classification": "research_candidate",
        "qualification_claim": False,
        "formal_dataset_registration_authorized": False,
        "batch": {
            "path": str(batch.resolve()),
            "sha256": sha256_file(batch),
            "batch_id": batch_value["batch_id"],
            "batch_content_sha256": batch_value["batch_content_sha256"],
        },
        "instance_request": {
            "ordinal": request["ordinal"],
            "instance_request_id": request["instance_request_id"],
            "request_sha256": request["request_sha256"],
            "taxonomy": request["taxonomy"],
            "attributes": request["attributes"],
            "realization_operations": request["realization_operations"],
        },
        "source": _file_binding(pre_rebase),
        "output": {"glb": _file_binding(pre_rebase)},
        "tool_identity": tool_identity,
    }
    realization_value["report_content_sha256"] = canonical_json_sha256(
        realization_value
    )
    _write_json(
        realization,
        realization_value,
    )
    lineage_core = {
        "schema": "avengine_m2_appearance_variant_lineage_v1",
        "status": "pass",
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "formal_dataset_registration_authorized": False,
        "instance_request": {
            "instance_request_id": request["instance_request_id"],
            "request_sha256": request["request_sha256"],
            "ordinal": request["ordinal"],
            "taxonomy": request["taxonomy"],
            "attributes": request["attributes"],
        },
        "inputs": {
            "appearance_batch": _json_binding(batch),
            "appearance_realization_report": _json_binding(realization),
            "template_variant_spec": _json_binding(EXAMPLE_SPEC),
            "upstream_source_manifest": _json_binding(source),
        },
        "derivative": {
            "pre_rebase_visual_glb": _file_binding(pre_rebase),
            "tool_identity": tool_identity,
            "derived_variant_spec": {
                "schema": "avengine_m2_variant_package_spec_v1",
                "asset_id": spec.identity.asset_id,
                "byte_size": spec.path.stat().st_size,
                "sha256": spec.sha256,
                "canonical_content_sha256": canonical_json_sha256(spec.value),
            },
        },
        "decision_reason": "Synthetic strict evidence for assembler unit tests.",
    }
    lineage_core["lineage_content_sha256"] = canonical_json_sha256(lineage_core)
    lineage = tmp_path / "appearance_lineage.json"
    _write_json(lineage, lineage_core)
    return variant_package.VariantPackageEvidence(
        visual_glb=visual,
        baked_actions=actions,
        contacts=contacts,
        source_manifest=source,
        license_snapshot=license_snapshot,
        appearance_lineage=lineage,
        material_normalization_report=normalization_path,
        **paths,
    )


def _cli_args(
    evidence: variant_package.VariantPackageEvidence,
    output: Path,
) -> list[str]:
    return [
        "--spec",
        str(EXAMPLE_SPEC),
        "--visual-glb",
        str(evidence.visual_glb),
        "--actions-npz",
        str(evidence.baked_actions),
        "--rebase-report",
        str(evidence.rebase_report),
        "--rebase-deformation-report",
        str(evidence.rebase_deformation_report),
        "--action-report",
        str(evidence.action_report),
        "--static-qa",
        str(evidence.static_qa),
        "--deformation-qa",
        str(evidence.deformation_qa),
        "--animation-qa",
        str(evidence.animation_qa),
        "--habitat-static-probe",
        str(evidence.habitat_static_probe),
        "--habitat-animation-review",
        str(evidence.habitat_animation_review),
        "--contact-phases",
        str(evidence.contacts),
        "--appearance-lineage",
        str(evidence.appearance_lineage),
        "--material-normalization-report",
        str(evidence.material_normalization_report),
        "--source-manifest",
        str(evidence.source_manifest),
        "--license-snapshot",
        str(evidence.license_snapshot),
        "--output",
        str(output),
    ]


def test_l9_assembly_fails_closed_when_canonical_producer_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(variant_package, "_REPOSITORY_ROOT", tmp_path)
    spec = variant_package.load_variant_package_spec(EXAMPLE_SPEC)
    with pytest.raises(
        variant_package.VariantPackageError,
        match="canonical L9 producer validator.*not a regular file",
    ):
        _REAL_L9_PRODUCER_VALIDATOR(spec, {})


def test_example_spec_exposes_registered_taxonomy_appearance_and_anchors() -> None:
    spec = variant_package.load_variant_package_spec(EXAMPLE_SPEC)

    assert spec.identity.body_plan_id == "quadruped_mammal_canid_v1"
    assert spec.value["taxonomy"] == {
        "species_id": "canis_lupus_familiaris",
        "breed_id": "beagle",
    }
    assert spec.value["appearance"] == {
        "size": "medium",
        "body_build": "standard",
        "coat": "standard_tricolor",
        "life_stage": "adult",
    }
    assert spec.value["rendering"] == {"shader_type": "pbr"}
    assert spec.shader_type == "pbr"
    assert spec.semantic_joint_map["paw_hind_left"] == "beagle L Toe0"
    assert list(spec.semantic_joint_map) == [
        "body",
        "head",
        "muzzle",
        "paw_front_left",
        "paw_front_right",
        "paw_hind_left",
        "paw_hind_right",
    ]


def test_spec_never_infers_missing_semantic_anchors(tmp_path: Path) -> None:
    value = json.loads(EXAMPLE_SPEC.read_text(encoding="utf-8"))
    value["anchors"] = value["anchors"][:-1]
    path = tmp_path / "missing_anchor.json"
    _write_json(path, value)

    with pytest.raises(
        variant_package.VariantPackageError,
        match="missing required IDs",
    ):
        variant_package.load_variant_package_spec(path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("rendering"),
        lambda value: value.__setitem__("rendering", {}),
        lambda value: value.__setitem__("rendering", {"shader_type": "metal"}),
        lambda value: value.__setitem__("rendering", {"shader_type": []}),
        lambda value: value.__setitem__(
            "rendering", {"shader_type": "pbr", "implicit_fallback": True}
        ),
    ],
)
def test_spec_requires_exact_explicit_shader(
    tmp_path: Path,
    mutation: Any,
) -> None:
    value = json.loads(EXAMPLE_SPEC.read_text(encoding="utf-8"))
    mutation(value)
    path = tmp_path / "invalid_rendering.json"
    _write_json(path, value)

    with pytest.raises(variant_package.VariantPackageError, match="rendering"):
        variant_package.load_variant_package_spec(path)


def test_spec_rejects_species_body_plan_mismatch(tmp_path: Path) -> None:
    value = json.loads(EXAMPLE_SPEC.read_text(encoding="utf-8"))
    value["identity"]["body_plan_id"] = "quadruped_mammal_felid_v1"
    path = tmp_path / "body_plan_mismatch.json"
    _write_json(path, value)

    with pytest.raises(
        variant_package.VariantPackageError,
        match="identity.body_plan_id must be 'quadruped_mammal_canid_v1'",
    ):
        variant_package.load_variant_package_spec(path)


def test_spec_rejects_species_without_reviewed_body_plan_profile(
    tmp_path: Path,
) -> None:
    value = json.loads(EXAMPLE_SPEC.read_text(encoding="utf-8"))
    value["taxonomy"]["species_id"] = "avis_unregistered"
    path = tmp_path / "unknown_species.json"
    _write_json(path, value)

    with pytest.raises(
        variant_package.VariantPackageError,
        match="add and review a new profile",
    ):
        variant_package.load_variant_package_spec(path)


def test_spec_rejects_unregistered_breed_profile(tmp_path: Path) -> None:
    value = json.loads(EXAMPLE_SPEC.read_text(encoding="utf-8"))
    value["taxonomy"]["breed_id"] = "unregistered_canid"
    path = tmp_path / "unknown_breed.json"
    _write_json(path, value)

    with pytest.raises(
        variant_package.VariantPackageError,
        match="exact.*species/breed|taxonomy pair",
    ):
        variant_package.load_variant_package_spec(path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda appearance: appearance.clear(),
        lambda appearance: appearance.pop("life_stage"),
        lambda appearance: appearance.__setitem__("size", 999),
        lambda appearance: appearance.__setitem__("body_build", ["stocky"]),
        lambda appearance: appearance.__setitem__("life_stage", False),
        lambda appearance: appearance.__setitem__("coat", "classic_golden"),
        lambda appearance: appearance.__setitem__("unreviewed", "value"),
    ],
)
def test_spec_requires_registered_breed_scoped_appearance(
    tmp_path: Path,
    mutation: Any,
) -> None:
    value = json.loads(EXAMPLE_SPEC.read_text(encoding="utf-8"))
    mutation(value["appearance"])
    path = tmp_path / "invalid_appearance.json"
    _write_json(path, value)

    with pytest.raises(variant_package.VariantPackageError, match="appearance"):
        variant_package.load_variant_package_spec(path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda taxonomy: taxonomy.pop("breed_id"),
        lambda taxonomy: taxonomy.__setitem__("species_id", ""),
        lambda taxonomy: taxonomy.__setitem__("breed_id", "   "),
        lambda taxonomy: taxonomy.__setitem__("unreviewed_field", "value"),
    ],
)
def test_spec_requires_exact_nonempty_taxonomy(
    tmp_path: Path,
    mutation: Any,
) -> None:
    value = json.loads(EXAMPLE_SPEC.read_text(encoding="utf-8"))
    mutation(value["taxonomy"])
    path = tmp_path / "invalid_taxonomy.json"
    _write_json(path, value)

    with pytest.raises(
        variant_package.VariantPackageError,
        match="taxonomy.*non-empty|taxonomy must contain exactly",
    ):
        variant_package.load_variant_package_spec(path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda source: source.pop("formal_dataset_registration_authorized"),
        lambda source: source.__setitem__("formal_dataset_registration_authorized", 1),
        lambda source: source.__setitem__(
            "formal_dataset_registration_authorized", "false"
        ),
    ],
)
def test_assembler_requires_exact_false_registration_authorization(
    tmp_path: Path,
    mutation: Any,
) -> None:
    spec = variant_package.load_variant_package_spec(EXAMPLE_SPEC)
    evidence = _evidence(tmp_path)
    source = json.loads(evidence.source_manifest.read_text(encoding="utf-8"))
    mutation(source)
    _write_json(evidence.source_manifest, source)

    with pytest.raises(
        variant_package.VariantPackageError,
        match="formal_dataset_registration_authorized must be exactly false",
    ):
        variant_package.assemble_variant_package(
            spec=spec,
            evidence=evidence,
            output_directory=tmp_path / "package",
        )


def test_assembler_refuses_nonpassing_qa_before_compiler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = variant_package.load_variant_package_spec(EXAMPLE_SPEC)
    evidence = _evidence(tmp_path)
    static = json.loads(evidence.static_qa.read_text(encoding="utf-8"))
    static["status"] = "fail"
    _write_json(evidence.static_qa, static)

    def unexpected_compile(**kwargs: Any) -> Path:
        raise AssertionError("strict compiler must not see failed QA")

    monkeypatch.setattr(
        variant_package,
        "compile_research_candidate_animal_package",
        unexpected_compile,
    )
    output = tmp_path / "package"
    with pytest.raises(
        variant_package.VariantPackageError,
        match="static_qa must contain a real passing report",
    ):
        variant_package.assemble_variant_package(
            spec=spec,
            evidence=evidence,
            output_directory=output,
        )
    assert not output.exists()


def test_required_appearance_and_material_evidence_close_over_visual_bytes(
    tmp_path: Path,
) -> None:
    spec = variant_package.load_variant_package_spec(EXAMPLE_SPEC)
    evidence = _evidence(tmp_path)

    variant_package._validate_real_evidence(spec, evidence)


@pytest.mark.parametrize(
    "field_name",
    ["appearance_lineage", "material_normalization_report"],
)
def test_upstream_source_snapshots_cannot_replace_required_evidence_files(
    tmp_path: Path,
    field_name: str,
) -> None:
    spec = variant_package.load_variant_package_spec(EXAMPLE_SPEC)
    evidence = _evidence(tmp_path)
    source = json.loads(evidence.source_manifest.read_text(encoding="utf-8"))
    evidence_path = getattr(evidence, field_name)
    source[field_name] = {
        **_file_binding(evidence_path),
        "snapshot": json.loads(evidence_path.read_text(encoding="utf-8")),
    }
    _write_json(evidence.source_manifest, source)
    missing = tmp_path / f"missing-{field_name}.json"
    evidence = replace(evidence, **{field_name: missing})

    with pytest.raises(
        variant_package.VariantPackageError,
        match=field_name,
    ):
        variant_package.assemble_variant_package(
            spec=spec,
            evidence=evidence,
            output_directory=tmp_path / "package",
        )


def test_appearance_lineage_must_close_every_rebase_hash_hop(
    tmp_path: Path,
) -> None:
    spec = variant_package.load_variant_package_spec(EXAMPLE_SPEC)
    evidence = _evidence(tmp_path)
    report = json.loads(evidence.rebase_report.read_text(encoding="utf-8"))
    report["source"]["sha256"] = "f" * 64
    _write_json(evidence.rebase_report, report)

    with pytest.raises(
        variant_package.VariantPackageError,
        match=r"rebase_report\.source\.sha256",
    ):
        variant_package.assemble_variant_package(
            spec=spec,
            evidence=evidence,
            output_directory=tmp_path / "package",
        )


def test_l9_lineage_rejects_separately_supplied_source_swap(
    tmp_path: Path,
) -> None:
    spec = variant_package.load_variant_package_spec(EXAMPLE_SPEC)
    evidence = _evidence(tmp_path)
    swapped = tmp_path / "swapped_source.json"
    _write_json(
        swapped,
        {
            "schema": "source_v2",
            "formal_dataset_registration_authorized": False,
        },
    )
    evidence = replace(evidence, source_manifest=swapped)

    with pytest.raises(
        variant_package.VariantPackageError,
        match="separately supplied file identity",
    ):
        variant_package.assemble_variant_package(
            spec=spec,
            evidence=evidence,
            output_directory=tmp_path / "package",
        )


def test_l9_lineage_rejects_minimal_fake_batch(tmp_path: Path) -> None:
    spec = variant_package.load_variant_package_spec(EXAMPLE_SPEC)
    evidence = _evidence(tmp_path)
    lineage = json.loads(evidence.appearance_lineage.read_text(encoding="utf-8"))
    batch = Path(lineage["inputs"]["appearance_batch"]["path"])
    _write_json(batch, {"schema": "avengine_animal_appearance_batch_v1"})
    lineage["inputs"]["appearance_batch"] = _json_binding(batch)
    lineage["lineage_content_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in lineage.items()
            if key != "lineage_content_sha256"
        }
    )
    _write_json(evidence.appearance_lineage, lineage)

    with pytest.raises(
        variant_package.VariantPackageError,
        match="L9 batch failed full validation",
    ):
        variant_package.assemble_variant_package(
            spec=spec,
            evidence=evidence,
            output_directory=tmp_path / "package",
        )


def test_l9_lineage_rejects_detached_instance_request(tmp_path: Path) -> None:
    spec = variant_package.load_variant_package_spec(EXAMPLE_SPEC)
    evidence = _evidence(tmp_path)
    lineage = json.loads(evidence.appearance_lineage.read_text(encoding="utf-8"))
    lineage["instance_request"]["request_sha256"] = "f" * 64
    lineage["lineage_content_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in lineage.items()
            if key != "lineage_content_sha256"
        }
    )
    _write_json(evidence.appearance_lineage, lineage)

    with pytest.raises(
        variant_package.VariantPackageError,
        match="instance differs from its authenticated L9 batch",
    ):
        variant_package.assemble_variant_package(
            spec=spec,
            evidence=evidence,
            output_directory=tmp_path / "package",
        )


def test_l9_lineage_rejects_arbitrary_rehashed_tool_identity(
    tmp_path: Path,
) -> None:
    spec = variant_package.load_variant_package_spec(EXAMPLE_SPEC)
    evidence = _evidence(tmp_path)
    lineage = json.loads(evidence.appearance_lineage.read_text(encoding="utf-8"))
    realization_path = Path(lineage["inputs"]["appearance_realization_report"]["path"])
    realization = json.loads(realization_path.read_text(encoding="utf-8"))
    fake_tool = {"tool": "attacker-controlled-realizer"}
    realization["tool_identity"] = fake_tool
    realization["report_content_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in realization.items()
            if key != "report_content_sha256"
        }
    )
    _write_json(realization_path, realization)
    lineage["inputs"]["appearance_realization_report"] = _json_binding(realization_path)
    lineage["derivative"]["tool_identity"] = fake_tool
    lineage["lineage_content_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in lineage.items()
            if key != "lineage_content_sha256"
        }
    )
    _write_json(evidence.appearance_lineage, lineage)

    with pytest.raises(
        variant_package.VariantPackageError,
        match="tool_identity fields are invalid",
    ):
        variant_package.assemble_variant_package(
            spec=spec,
            evidence=evidence,
            output_directory=tmp_path / "package",
        )


def test_direct_api_rejects_valid_nonopaque_normalization_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = variant_package.load_variant_package_spec(EXAMPLE_SPEC)
    evidence = _evidence(tmp_path, force_opaque=False)

    def unexpected_compile(**kwargs: Any) -> Path:
        raise AssertionError("compiler must not see alpha-bypassing GLB")

    monkeypatch.setattr(
        variant_package,
        "compile_research_candidate_animal_package",
        unexpected_compile,
    )
    with pytest.raises(
        variant_package.VariantPackageError,
        match="force_opaque=true",
    ):
        variant_package.assemble_variant_package(
            spec=spec,
            evidence=evidence,
            output_directory=tmp_path / "package",
        )


def test_cli_rejects_valid_nonopaque_normalization_report(
    tmp_path: Path,
) -> None:
    evidence = _evidence(tmp_path, force_opaque=False)
    output = tmp_path / "package"

    with pytest.raises(
        variant_package.VariantPackageError,
        match="force_opaque=true",
    ):
        cli.main(_cli_args(evidence, output))
    assert not output.exists()


@pytest.mark.parametrize(
    ("bypass", "message"),
    [
        ("metallic", "metallic material bypass"),
        ("emissive", "emissive material bypass"),
        ("specular", "specular material bypass"),
        ("alpha", "alpha material bypass"),
    ],
)
def test_direct_api_independently_reads_back_actual_glb_material_bypasses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bypass: str,
    message: str,
) -> None:
    spec = variant_package.load_variant_package_spec(EXAMPLE_SPEC)
    evidence = _evidence(tmp_path)
    loaded = load_glb(evidence.visual_glb)
    document = loaded.json
    material = document["materials"][0]
    if bypass == "metallic":
        material["pbrMetallicRoughness"]["metallicFactor"] = 0.8
    elif bypass == "emissive":
        material["emissiveFactor"] = [0.2, 0.0, 0.0]
    elif bypass == "specular":
        material["extensions"]["KHR_materials_specular"]["specularFactor"] = 0.9
    else:
        material["alphaMode"] = "BLEND"
        material["pbrMetallicRoughness"]["baseColorFactor"][3] = 0.4
    evidence.visual_glb.write_bytes(build_glb(document, loaded.binary))

    rebase = json.loads(evidence.rebase_report.read_text(encoding="utf-8"))
    rebase["output"] = _file_binding(evidence.visual_glb)
    _write_json(evidence.rebase_report, rebase)
    authenticated_report = {
        "policy": {"force_opaque": True},
        "output": _file_binding(evidence.visual_glb),
    }
    monkeypatch.setattr(
        variant_package,
        "load_and_validate_material_normalization_report",
        lambda path, verify_files: authenticated_report,
    )

    def unexpected_compile(**kwargs: Any) -> Path:
        raise AssertionError("compiler must not see a material-bypassing GLB")

    monkeypatch.setattr(
        variant_package,
        "compile_research_candidate_animal_package",
        unexpected_compile,
    )
    with pytest.raises(variant_package.VariantPackageError, match=message):
        variant_package.assemble_variant_package(
            spec=spec,
            evidence=evidence,
            output_directory=tmp_path / "package",
        )


def test_assembler_binds_spec_and_source_then_delegates_to_strict_compiler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = variant_package.load_variant_package_spec(EXAMPLE_SPEC)
    evidence = _evidence(tmp_path)
    output = tmp_path / "package"
    observed: dict[str, Any] = {}

    monkeypatch.setattr(
        variant_package,
        "_validate_real_evidence",
        lambda spec, evidence: None,
    )

    def compile_package(**kwargs: Any) -> Path:
        observed.update(kwargs)
        bound_source = Path(kwargs["source_manifest"])
        observed["source_value"] = json.loads(bound_source.read_text(encoding="utf-8"))
        return Path(kwargs["output_directory"]) / "asset_manifest.json"

    monkeypatch.setattr(
        variant_package,
        "compile_research_candidate_animal_package",
        compile_package,
    )
    manifest = variant_package.assemble_variant_package(
        spec=spec,
        evidence=evidence,
        output_directory=output,
    )

    assert manifest == output / "asset_manifest.json"
    assert observed["identity"] == spec.identity
    assert observed["anchor_definitions"] == spec.anchors
    assert observed["shader_type"] == "pbr"
    source_value = observed["source_value"]
    assert source_value["formal_dataset_registration_authorized"] is False
    assert source_value["schema"] == "avengine_m2_variant_source_binding_v2"
    assert source_value["variant_package_spec"]["sha256"] == spec.sha256
    assert source_value["actual_visual_glb"]["sha256"] == sha256_file(
        evidence.visual_glb
    )
    assert source_value["appearance_lineage"]["sha256"] == sha256_file(
        evidence.appearance_lineage
    )
    assert source_value["material_normalization_report"]["sha256"] == sha256_file(
        evidence.material_normalization_report
    )
    assert (
        source_value["variant_package_spec"]["snapshot"]["appearance"]["coat"]
        == "standard_tricolor"
    )
    assert source_value["upstream_source_manifest"]["snapshot"]["schema"] == (
        "source_v1"
    )


def test_assembler_refuses_existing_output_before_reading_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = variant_package.load_variant_package_spec(EXAMPLE_SPEC)
    evidence = _evidence(tmp_path)
    output = tmp_path / "package"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep\n", encoding="utf-8")

    def unexpected_validate(spec: object, evidence: object) -> None:
        raise AssertionError("evidence must not be read when output exists")

    monkeypatch.setattr(variant_package, "_validate_real_evidence", unexpected_validate)
    with pytest.raises(
        variant_package.VariantPackageError,
        match="refusing to replace package output",
    ):
        variant_package.assemble_variant_package(
            spec=spec,
            evidence=evidence,
            output_directory=output,
        )
    assert marker.read_text(encoding="utf-8") == "keep\n"
