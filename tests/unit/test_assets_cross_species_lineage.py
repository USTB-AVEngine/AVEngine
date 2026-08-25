from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

from jsonschema import Draft202012Validator
import pytest

from avengine.contracts.json_io import canonical_json_sha256, load_json, sha256_file
from avengine.assets import variant_package
from avengine.assets.cross_species_lineage import (
    CrossSpeciesLineageError,
    build_cross_species_appearance_lineage,
    validate_cross_species_appearance_lineage,
)
from avengine.assets.glb_write import build_glb
from avengine.assets.materials import normalize_glb_materials
from tools.assets import build_cross_species_appearance_lineage as cli
from tools.assets import force_matte_materials


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_SPEC = (
    REPOSITORY_ROOT
    / "examples/m2/variant_packages/rocketbox_beagle_review_spec_v1.json"
)
SCHEMA = REPOSITORY_ROOT / "schemas/m2_cross_species_appearance_lineage_v1.schema.json"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _file_binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _json_binding(path: Path) -> dict[str, Any]:
    value = load_json(path)
    return {
        **_file_binding(path),
        "canonical_content_sha256": canonical_json_sha256(value),
        "snapshot": value,
    }


def _document() -> dict[str, Any]:
    return {
        "asset": {"version": "2.0", "generator": "generic-lineage-test"},
        "extensionsUsed": ["KHR_materials_specular"],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {}, "material": 0}]}],
        "materials": [
            {
                "name": "diagnostic",
                "alphaMode": "BLEND",
                "emissiveFactor": [0.0, 0.0, 0.0],
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.2, 0.25, 0.3, 0.5],
                    "metallicFactor": 0.4,
                    "roughnessFactor": 0.3,
                },
                "extensions": {
                    "KHR_materials_specular": {
                        "specularFactor": 0.8,
                        "specularColorFactor": [1.0, 1.0, 1.0],
                    }
                },
            }
        ],
    }


@dataclass
class _Fixture:
    spec: Path
    upstream: Path
    realization_report: Path
    normalization_report: Path
    rebase_report: Path
    pre_rebase_visual: Path
    final_visual: Path
    lineage: dict[str, Any]
    lineage_path: Path


def _fixture(tmp_path: Path) -> _Fixture:
    spec_value = load_json(EXAMPLE_SPEC)
    spec_value["taxonomy"] = {
        "species_id": "felis_catus",
        "breed_id": "generic",
    }
    spec_value["appearance"] = {
        "size": "medium",
        "body_build": "standard",
        "coat": "charcoal_gray",
        "life_stage": "adult",
    }
    spec_value["identity"].update(
        {
            "asset_id": "test_generic_cat_diagnostic_v1",
            "template_id": "test_generic_cat",
            "body_plan_id": "quadruped_mammal_felid_v1",
            "morphotype_id": "domestic_cat_generic",
        }
    )
    spec = tmp_path / "spec.json"
    _write_json(spec, spec_value)
    upstream = tmp_path / "upstream.json"
    _write_json(
        upstream,
        {
            "schema": "avengine_m2_cross_species_research_lineage_v6",
            "qualification_state": "research_candidate",
            "qualification_claim": False,
            "formal_dataset_registration_authorized": False,
            "taxonomy": spec_value["taxonomy"],
            "appearance": spec_value["appearance"],
            "decision": "Synthetic cross-species diagnostic source lineage.",
        },
    )

    material_source = tmp_path / "material_source.glb"
    material_source.write_bytes(build_glb(_document(), b""))
    pre_rebase_visual = tmp_path / "material_realized.glb"
    realization_report = tmp_path / "material_realization.json"
    assert (
        force_matte_materials.main(
            [
                "--input",
                str(material_source),
                "--output",
                str(pre_rebase_visual),
                "--report",
                str(realization_report),
            ]
        )
        == 0
    )
    upstream_value = load_json(upstream)
    upstream_value["reused_hash_closed_inputs"] = [
        {
            "path": "projected_strict.glb",
            "byte_size": pre_rebase_visual.stat().st_size,
            "sha256": sha256_file(pre_rebase_visual),
        }
    ]
    _write_json(upstream, upstream_value)

    rebased_visual = tmp_path / "rebased.glb"
    shutil.copyfile(pre_rebase_visual, rebased_visual)
    rebase_report = tmp_path / "rebase.json"
    _write_json(
        rebase_report,
        {
            "schema": "avengine_m2_skin_root_rebase_v1",
            "status": "pass",
            "qualification_state": "research_candidate",
            "qualification_claim": False,
            "source": _file_binding(pre_rebase_visual),
            "output": _file_binding(rebased_visual),
        },
    )

    final_visual = tmp_path / "final.glb"
    normalization = normalize_glb_materials(
        rebased_visual,
        final_visual,
        force_opaque=True,
    )
    normalization_report = tmp_path / "material_normalization.json"
    _write_json(normalization_report, normalization)

    lineage = build_cross_species_appearance_lineage(
        variant_spec=spec,
        upstream_source_manifest=upstream,
        material_realization_report=realization_report,
        material_normalization_report=normalization_report,
        rebase_report=rebase_report,
        pre_rebase_visual_glb=pre_rebase_visual,
        lineage_producer=(
            REPOSITORY_ROOT / "tools/assets/build_cross_species_appearance_lineage.py"
        ),
        lineage_contract=(REPOSITORY_ROOT / "src/avengine/assets/cross_species_lineage.py"),
        material_realization_tool=(
            REPOSITORY_ROOT / "tools/assets/force_matte_materials.py"
        ),
        material_normalization_tool=(
            REPOSITORY_ROOT / "tools/assets/normalize_materials.py"
        ),
        material_algorithm=(REPOSITORY_ROOT / "src/avengine/assets/materials.py"),
        skin_root_rebase_tool=(REPOSITORY_ROOT / "tools/assets/rebase_skin_root.py"),
        rebase_algorithm=(REPOSITORY_ROOT / "src/avengine/assets/rebase.py"),
    )
    lineage_path = tmp_path / "lineage.json"
    _write_json(lineage_path, lineage)
    return _Fixture(
        spec=spec,
        upstream=upstream,
        realization_report=realization_report,
        normalization_report=normalization_report,
        rebase_report=rebase_report,
        pre_rebase_visual=pre_rebase_visual,
        final_visual=final_visual,
        lineage=lineage,
        lineage_path=lineage_path,
    )


def _redigest(lineage: dict[str, Any]) -> None:
    lineage["lineage_content_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in lineage.items()
            if key != "lineage_content_sha256"
        }
    )


def _cli_arguments(fixture: _Fixture, output: Path) -> list[str]:
    return [
        "--spec",
        str(fixture.spec),
        "--upstream-source-manifest",
        str(fixture.upstream),
        "--material-realization-report",
        str(fixture.realization_report),
        "--material-normalization-report",
        str(fixture.normalization_report),
        "--rebase-report",
        str(fixture.rebase_report),
        "--pre-rebase-visual-glb",
        str(fixture.pre_rebase_visual),
        "--output",
        str(output),
    ]


def test_builder_emits_schema_valid_diagnostic_only_lineage(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    Draft202012Validator(load_json(SCHEMA)).validate(fixture.lineage)
    assert fixture.lineage["design_kind"] == "single_research_diagnostic"
    assert fixture.lineage["ofat_status"] == "not_run"
    assert fixture.lineage["qualification_claim"] is False
    assert fixture.lineage["formal_dataset_registration_authorized"] is False
    assert fixture.lineage["inputs"]["upstream_source_manifest"][
        "sha256"
    ] == sha256_file(fixture.upstream)


def test_variant_assembler_dispatch_accepts_generic_lineage(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    spec = variant_package.load_variant_package_spec(fixture.spec)
    unused = tmp_path / "unused"
    evidence = variant_package.VariantPackageEvidence(
        visual_glb=fixture.final_visual,
        rebase_report=fixture.rebase_report,
        rebase_deformation_report=unused,
        action_report=unused,
        static_qa=unused,
        deformation_qa=unused,
        animation_qa=unused,
        habitat_static_probe=unused,
        habitat_animation_review=unused,
        baked_actions=unused,
        contacts=unused,
        source_manifest=fixture.upstream,
        license_snapshot=unused,
        appearance_lineage=fixture.lineage_path,
        material_normalization_report=fixture.normalization_report,
    )

    validated = variant_package._validate_appearance_lineage(
        spec=spec,
        evidence=evidence,
        visual=fixture.final_visual,
        rebase_report=load_json(fixture.rebase_report),
    )

    assert validated["schema"] == ("avengine_m2_cross_species_appearance_lineage_v1")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["taxonomy"].__setitem__("breed_id", "wrong"),
            "taxonomy differs",
        ),
        (
            lambda value: value["appearance"].__setitem__("coat", "silver_gray"),
            "appearance differs",
        ),
        (
            lambda value: value["inputs"]["variant_spec"].__setitem__(
                "sha256", "f" * 64
            ),
            "variant_spec.sha256",
        ),
    ],
)
def test_lineage_rejects_taxonomy_coat_and_hash_tampering(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    mutation(fixture.lineage)
    _redigest(fixture.lineage)

    with pytest.raises(CrossSpeciesLineageError, match=message):
        validate_cross_species_appearance_lineage(fixture.lineage)


def test_lineage_rejects_physical_spec_tampering(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    spec = load_json(fixture.spec)
    spec["identity"]["asset_id"] = "tampered_asset"
    _write_json(fixture.spec, spec)

    with pytest.raises(CrossSpeciesLineageError, match="variant_spec.byte_size|sha256"):
        validate_cross_species_appearance_lineage(fixture.lineage)


def test_lineage_rejects_material_file_tampering(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture.final_visual.write_bytes(fixture.final_visual.read_bytes() + b"tamper")

    with pytest.raises(
        CrossSpeciesLineageError,
        match="material_normalization_report|material normalization",
    ):
        validate_cross_species_appearance_lineage(fixture.lineage)


def test_lineage_rejects_rebase_chain_tampering_even_when_rebound(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    rebase = load_json(fixture.rebase_report)
    rebase["source"] = load_json(fixture.realization_report)["source"]
    _write_json(fixture.rebase_report, rebase)
    fixture.lineage["inputs"]["rebase_report"] = _json_binding(fixture.rebase_report)
    _redigest(fixture.lineage)

    with pytest.raises(CrossSpeciesLineageError, match="pre-rebase visual"):
        validate_cross_species_appearance_lineage(fixture.lineage)


def test_lineage_rejects_detached_upstream_material_terminal(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    upstream = load_json(fixture.upstream)
    realization = load_json(fixture.realization_report)
    upstream["reused_hash_closed_inputs"][0] = {
        "path": "projected_strict.glb",
        "byte_size": realization["source"]["byte_size"],
        "sha256": realization["source"]["sha256"],
    }
    _write_json(fixture.upstream, upstream)
    fixture.lineage["inputs"]["upstream_source_manifest"] = _json_binding(
        fixture.upstream
    )
    _redigest(fixture.lineage)

    with pytest.raises(
        CrossSpeciesLineageError,
        match="upstream projected visual to material realization output",
    ):
        validate_cross_species_appearance_lineage(fixture.lineage)


def test_lineage_rejects_same_basename_tool_substitution(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    substitute = tmp_path / "build_cross_species_appearance_lineage.py"
    shutil.copyfile(
        REPOSITORY_ROOT / "tools/assets/build_cross_species_appearance_lineage.py",
        substitute,
    )
    fixture.lineage["tool_identity"]["lineage_producer"] = _file_binding(substitute)
    _redigest(fixture.lineage)

    with pytest.raises(
        CrossSpeciesLineageError,
        match="canonical repository file",
    ):
        validate_cross_species_appearance_lineage(fixture.lineage)


def test_generic_lineage_cannot_bypass_beagle_l9(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    with pytest.raises(CrossSpeciesLineageError, match="bypass the Beagle L9"):
        build_cross_species_appearance_lineage(
            variant_spec=EXAMPLE_SPEC,
            upstream_source_manifest=fixture.upstream,
            material_realization_report=fixture.realization_report,
            material_normalization_report=fixture.normalization_report,
            rebase_report=fixture.rebase_report,
            pre_rebase_visual_glb=fixture.pre_rebase_visual,
            lineage_producer=(
                REPOSITORY_ROOT / "tools/assets/build_cross_species_appearance_lineage.py"
            ),
            lineage_contract=(
                REPOSITORY_ROOT / "src/avengine/assets/cross_species_lineage.py"
            ),
            material_realization_tool=(
                REPOSITORY_ROOT / "tools/assets/force_matte_materials.py"
            ),
            material_normalization_tool=(
                REPOSITORY_ROOT / "tools/assets/normalize_materials.py"
            ),
            material_algorithm=(REPOSITORY_ROOT / "src/avengine/assets/materials.py"),
            skin_root_rebase_tool=(REPOSITORY_ROOT / "tools/assets/rebase_skin_root.py"),
            rebase_algorithm=(REPOSITORY_ROOT / "src/avengine/assets/rebase.py"),
        )


def test_generic_lineage_rejects_separately_supplied_source_swap(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    swapped = tmp_path / "swapped_source.json"
    swapped_value = load_json(fixture.upstream)
    swapped_value["decision"] = "Different source lineage."
    _write_json(swapped, swapped_value)

    with pytest.raises(
        CrossSpeciesLineageError,
        match="separately supplied file identity",
    ):
        validate_cross_species_appearance_lineage(
            fixture.lineage,
            expected_upstream_source_manifest=swapped,
        )


def test_cli_writes_exclusively_and_validates_readback(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "cli_lineage.json"
    arguments = _cli_arguments(fixture, output)

    assert cli.main(arguments) == 0
    assert load_json(output)["schema"] == (
        "avengine_m2_cross_species_appearance_lineage_v1"
    )
    with pytest.raises(SystemExit):
        cli.main(arguments)


def test_cli_does_not_delete_foreign_file_winning_reservation_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "raced.json"
    real_reserve = cli._reserve_output

    def raced_reserve(path: Path) -> Any:
        path.write_text("foreign\n", encoding="utf-8")
        return real_reserve(path)

    monkeypatch.setattr(cli, "_reserve_output", raced_reserve)
    with pytest.raises(SystemExit):
        cli.main(_cli_arguments(fixture, output))
    assert output.read_text(encoding="utf-8") == "foreign\n"


@pytest.mark.parametrize("replacement", ["file", "directory"])
def test_cli_preserves_foreign_inode_replacing_reserved_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "replaced.json"
    observed_inode: int | None = None

    def replace_then_fail(**kwargs: Any) -> dict[str, Any]:
        nonlocal observed_inode
        output.unlink()
        if replacement == "file":
            output.write_text("foreign replacement\n", encoding="utf-8")
        else:
            output.mkdir()
        observed_inode = output.stat().st_ino
        raise CrossSpeciesLineageError("injected build failure")

    monkeypatch.setattr(
        cli,
        "build_cross_species_appearance_lineage",
        replace_then_fail,
    )
    with pytest.raises(SystemExit):
        cli.main(_cli_arguments(fixture, output))
    assert output.stat().st_ino == observed_inode
    if replacement == "file":
        assert output.read_text(encoding="utf-8") == "foreign replacement\n"
    else:
        assert output.is_dir()


def test_cli_rejects_output_with_ancestor_symlink(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked_parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    output = linked_parent / "lineage.json"

    with pytest.raises(SystemExit):
        cli.main(_cli_arguments(fixture, output))
    assert not (real_parent / "lineage.json").exists()
