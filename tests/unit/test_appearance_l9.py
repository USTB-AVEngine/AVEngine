from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from avengine.appearance import (
    APPEARANCE_AXES,
    AppearanceContractError,
    generate_l9_batch,
    validate_appearance_request,
    validate_l9_batch,
    verify_instance_request_integrity,
    write_l9_batch_exclusive,
)
from avengine.cli import main
from avengine.contracts.json_io import sha256_file


REPOSITORY = Path(__file__).resolve().parents[2]
EXAMPLE = REPOSITORY / "examples/m2/appearance/beagle_l9_request_v1.json"
REQUEST_SCHEMA = REPOSITORY / "schemas/animal_appearance_request_v1.schema.json"
BATCH_SCHEMA = REPOSITORY / "schemas/animal_appearance_batch_v1.schema.json"


def _example() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def _local_inputs(tmp_path: Path) -> tuple[Path, Path, dict]:
    source = tmp_path / "source.glb"
    source.write_bytes(b"deterministic-beagle-source\0v1")
    request = _example()
    request["source_asset"]["expected_sha256"] = sha256_file(source)
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return request_path, source, request


def test_beagle_example_matches_schema_and_semantic_contract() -> None:
    value = _example()
    schema = json.loads(REQUEST_SCHEMA.read_text(encoding="utf-8"))
    assert list(Draft202012Validator(schema).iter_errors(value)) == []
    assert validate_appearance_request(value) == value
    assert value["attribute_domains"]["coat_profile"]["values"] == [
        "light_tricolor",
        "standard_tricolor",
        "dark_tricolor",
    ]
    assert value["source_asset"]["expected_sha256"] == (
        "788a667537f7660bac5e128c38c2182453d1d4a9a4f8380343e7a9fa1947538c"
    )


def test_l9_is_pairwise_orthogonal_and_every_level_occurs_three_times(
    tmp_path: Path,
) -> None:
    request_path, source, _request = _local_inputs(tmp_path)
    batch = generate_l9_batch(request_path, source)
    assert len(batch["requests"]) == 9
    assert batch["balance_audit"]["every_level_three_times"] is True
    assert batch["balance_audit"]["pairwise_orthogonal"] is True

    rows = [item["attributes"] for item in batch["requests"]]
    for axis in APPEARANCE_AXES:
        assert sorted(Counter(row[axis] for row in rows).values()) == [3, 3, 3]
    for left_index, left in enumerate(APPEARANCE_AXES):
        for right in APPEARANCE_AXES[left_index + 1 :]:
            counts = Counter((row[left], row[right]) for row in rows)
            assert len(counts) == 9
            assert set(counts.values()) == {1}


def test_beagle_l9_rows_are_the_reviewable_baseline_centered_design(
    tmp_path: Path,
) -> None:
    request_path, source, _request = _local_inputs(tmp_path)
    rows = [
        item["attributes"]
        for item in generate_l9_batch(request_path, source)["requests"]
    ]
    assert rows == [
        {
            "size": "medium",
            "body_build": "standard",
            "coat_profile": "standard_tricolor",
            "life_stage": "adult",
        },
        {
            "size": "medium",
            "body_build": "slim",
            "coat_profile": "light_tricolor",
            "life_stage": "senior",
        },
        {
            "size": "medium",
            "body_build": "stocky",
            "coat_profile": "dark_tricolor",
            "life_stage": "young",
        },
        {
            "size": "small",
            "body_build": "standard",
            "coat_profile": "light_tricolor",
            "life_stage": "young",
        },
        {
            "size": "small",
            "body_build": "slim",
            "coat_profile": "dark_tricolor",
            "life_stage": "adult",
        },
        {
            "size": "small",
            "body_build": "stocky",
            "coat_profile": "standard_tricolor",
            "life_stage": "senior",
        },
        {
            "size": "large",
            "body_build": "standard",
            "coat_profile": "dark_tricolor",
            "life_stage": "senior",
        },
        {
            "size": "large",
            "body_build": "slim",
            "coat_profile": "standard_tricolor",
            "life_stage": "young",
        },
        {
            "size": "large",
            "body_build": "stocky",
            "coat_profile": "light_tricolor",
            "life_stage": "adult",
        },
    ]


def test_generated_batch_matches_schema_and_hash_closure(tmp_path: Path) -> None:
    request_path, source, request = _local_inputs(tmp_path)
    batch = generate_l9_batch(request_path, source)
    schema = json.loads(BATCH_SCHEMA.read_text(encoding="utf-8"))
    assert list(Draft202012Validator(schema).iter_errors(batch)) == []
    validate_l9_batch(batch)
    assert batch["parent_request"]["sha256"] == sha256_file(request_path)
    assert batch["source_asset"]["sha256"] == sha256_file(source)
    assert batch["source_asset"]["sha256"] == request["source_asset"]["expected_sha256"]
    assert {item["parent_request_file_sha256"] for item in batch["requests"]} == {
        sha256_file(request_path)
    }
    assert {item["source_asset_sha256"] for item in batch["requests"]} == {
        sha256_file(source)
    }
    for item in batch["requests"]:
        verify_instance_request_integrity(item)


def test_l9_explicitly_does_not_substitute_for_ofat(tmp_path: Path) -> None:
    request_path, source, _request = _local_inputs(tmp_path)
    batch = generate_l9_batch(request_path, source)
    assert batch["ofat_validation"] == {
        "strategy": "separate_one_factor_at_a_time_v1",
        "status": "not_run",
        "required_before_formal_promotion": True,
        "l9_substitution_allowed": False,
    }
    assert {item["design_role"] for item in batch["requests"]} == {
        "l9_combination_point_not_ofat"
    }


def test_beagle_cannot_use_golden_coat_vocabulary() -> None:
    value = _example()
    replacement = ["light_golden", "golden", "dark_golden"]
    value["attribute_domains"]["coat_profile"]["values"] = replacement
    value["baseline_attributes"]["coat_profile"] = "golden"
    value["l9_level_order"]["coat_profile"] = [
        "golden",
        "light_golden",
        "dark_golden",
    ]
    old = value["realization_bindings"]["coat_profile"]["parameters_by_value"]
    value["realization_bindings"]["coat_profile"]["parameters_by_value"] = {
        replacement_name: deepcopy(old[old_name])
        for replacement_name, old_name in zip(
            replacement,
            ["light_tricolor", "standard_tricolor", "dark_tricolor"],
            strict=True,
        )
    }
    with pytest.raises(AppearanceContractError, match="dog/beagle coat_profile values"):
        validate_appearance_request(value)


def test_coat_profile_scope_and_id_must_bind_exact_breed() -> None:
    value = _example()
    value["attribute_domains"]["coat_profile"]["scope"]["breed"] = "golden_retriever"
    value["attribute_domains"]["coat_profile"]["profile_id"] = (
        "dog_golden_retriever_coat_v1"
    )
    with pytest.raises(AppearanceContractError, match="must equal taxonomy.breed"):
        validate_appearance_request(value)

    value = _example()
    value["attribute_domains"]["coat_profile"]["profile_id"] = "universal_coat_v1"
    with pytest.raises(AppearanceContractError, match="species/breed prefix"):
        validate_appearance_request(value)


def test_namespaced_profile_id_is_not_registration() -> None:
    value = _example()
    value["attribute_domains"]["coat_profile"]["profile_id"] = (
        "dog_beagle_unreviewed_v99"
    )
    with pytest.raises(
        AppearanceContractError, match="no registered coat profile"
    ) as caught:
        validate_appearance_request(value)
    assert len(caught.value.errors) == 1


def test_unregistered_breed_cannot_self_declare_golden_domain() -> None:
    value = _example()
    value["taxonomy"]["breed"] = "invented_retriever"
    coat = value["attribute_domains"]["coat_profile"]
    coat["scope"]["breed"] = "invented_retriever"
    coat["profile_id"] = "dog_invented_retriever_golden_v1"

    replacement = ["light_golden", "golden", "dark_golden"]
    coat["values"] = replacement
    value["baseline_attributes"]["coat_profile"] = "golden"
    value["l9_level_order"]["coat_profile"] = [
        "golden",
        "light_golden",
        "dark_golden",
    ]
    old = value["realization_bindings"]["coat_profile"]["parameters_by_value"]
    value["realization_bindings"]["coat_profile"]["parameters_by_value"] = {
        replacement_name: deepcopy(old[old_name])
        for replacement_name, old_name in zip(
            replacement,
            ["light_tricolor", "standard_tricolor", "dark_tricolor"],
            strict=True,
        )
    }

    # Scope, namespace, three-level cardinality, order, and realization keys
    # are all internally consistent.  Registration is still mandatory.
    with pytest.raises(
        AppearanceContractError, match="no registered coat profile"
    ) as caught:
        validate_appearance_request(value)
    assert len(caught.value.errors) == 1


def test_realization_binding_is_fail_closed_and_directly_consumable() -> None:
    value = _example()
    torso = value["realization_bindings"]["body_build"]
    assert torso["operation_id"] == "semantic_torso_girth_scale_v1"
    assert torso["parameters_by_value"]["standard"]["semantic_joint_names"] == [
        "Pelvis",
        "Spine",
        "Spine1",
        "Spine2",
    ]
    value["realization_bindings"]["life_stage"]["parameters_by_value"]["senior"].pop(
        "muzzle_gray_target"
    )
    with pytest.raises(AppearanceContractError, match="muzzle_gray_target"):
        validate_appearance_request(value)


def test_source_hash_mismatch_fails_before_generation(tmp_path: Path) -> None:
    request_path, source, request = _local_inputs(tmp_path)
    request["source_asset"]["expected_sha256"] = "0" * 64
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(AppearanceContractError, match="source file SHA-256 mismatch"):
        generate_l9_batch(request_path, source)


def test_build_rejects_request_object_different_from_request_file(
    tmp_path: Path,
) -> None:
    from avengine.appearance import build_l9_batch

    request_path, source, request = _local_inputs(tmp_path)
    request["request_id"] = "different_request_v1"
    with pytest.raises(AppearanceContractError, match="does not match"):
        build_l9_batch(request, request_file=request_path, source_file=source)


def test_output_is_exclusive_and_never_overwritten(tmp_path: Path) -> None:
    request_path, source, _request = _local_inputs(tmp_path)
    batch = generate_l9_batch(request_path, source)
    output = tmp_path / "nested" / "batch.json"
    write_l9_batch_exclusive(output, batch)
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        write_l9_batch_exclusive(output, batch)
    assert output.read_bytes() == original


def test_output_rejects_dangling_leaf_symlink_without_writing_target(
    tmp_path: Path,
) -> None:
    request_path, source, _request = _local_inputs(tmp_path)
    batch = generate_l9_batch(request_path, source)
    outside = tmp_path / "outside.json"
    output = tmp_path / "batch.json"
    output.symlink_to(outside)

    with pytest.raises(FileExistsError, match="symbolic-link"):
        write_l9_batch_exclusive(output, batch)

    assert output.is_symlink()
    assert not outside.exists()


def test_output_rejects_symlinked_parent(tmp_path: Path) -> None:
    request_path, source, _request = _local_inputs(tmp_path)
    batch = generate_l9_batch(request_path, source)
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(AppearanceContractError, match="symbolic link"):
        write_l9_batch_exclusive(linked_parent / "batch.json", batch)

    assert not (outside / "batch.json").exists()


def test_tampered_instance_request_fails_integrity(tmp_path: Path) -> None:
    request_path, source, _request = _local_inputs(tmp_path)
    batch = generate_l9_batch(request_path, source)
    item = deepcopy(batch["requests"][0])
    item["attributes"]["size"] = "large"
    with pytest.raises(AppearanceContractError, match="request_sha256"):
        verify_instance_request_integrity(item)


def test_cli_builds_once_and_refuses_to_replace_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    request_path, source, _request = _local_inputs(tmp_path)
    output = tmp_path / "batch.json"
    arguments = [
        "appearance",
        "build-l9",
        "--request",
        str(request_path),
        "--source",
        str(source),
        "--output",
        str(output),
    ]
    assert main(arguments) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "pass"
    assert summary["request_count"] == 9
    expected = output.read_bytes()

    assert main(arguments) == 2
    failure = json.loads(capsys.readouterr().out)
    assert failure["status"] == "fail"
    assert failure["exception_type"] == "FileExistsError"
    assert output.read_bytes() == expected


def test_request_and_source_byte_hashes_are_not_canonical_json_hashes(
    tmp_path: Path,
) -> None:
    request_path, source, _request = _local_inputs(tmp_path)
    batch = generate_l9_batch(request_path, source)
    assert (
        batch["parent_request"]["sha256"]
        == hashlib.sha256(request_path.read_bytes()).hexdigest()
    )
    assert (
        batch["source_asset"]["sha256"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert (
        batch["parent_request"]["canonical_content_sha256"]
        != batch["parent_request"]["sha256"]
    )
