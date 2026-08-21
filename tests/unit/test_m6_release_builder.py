from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pytest

from avengine.release import (
    ReleaseManifestError,
    load_json_strict,
    prepare_release_manifest,
    sha256_file,
    verify_release_manifest,
)
from tests.unit.test_m6_release import (
    AVENGINE_URL,
    HABITAT_URL,
    RELEASE_SCHEMA,
    RLR_URL,
    UPSTREAM_URL,
    _commit_all,
    _git,
    _fixture_receipt_execution,
    _init_repository,
    _write_json,
)
from tools.release.build_manifest import main as release_tool_main


@pytest.fixture(autouse=True)
def _stub_heavy_m6_verifiers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The builder integration fixture exercises Git/release mechanics only."""

    monkeypatch.setattr(
        "avengine.m6.canary.verify_controlled_canary_evidence",
        lambda path: ("pass", [{"check_id": "fixture", "status": "pass"}]),
    )
    monkeypatch.setattr(
        "avengine.m6.room_attempts.verify_room_qualification_attempt",
        lambda path: ("pass", [{"check_id": "fixture", "status": "pass"}]),
    )


def _record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _build_request_fixture(tmp_path: Path) -> tuple[Path, Path, Path, str, str]:
    pytest.skip(
        "release v1 live checkout/submodule workflow is archived; "
        "fail-closed coverage is in test_release_v1_archive.py"
    )
    rlr_source = tmp_path / "rlr-source"
    _init_repository(rlr_source, origin=RLR_URL)
    (rlr_source / "README.md").write_text("fixture RLR\n", encoding="utf-8")
    rlr_commit = _commit_all(rlr_source, "fixture RLR")

    habitat = tmp_path / "habitat-runtime"
    _init_repository(habitat, origin=HABITAT_URL)
    _git(habitat, "remote", "add", "upstream", UPSTREAM_URL)
    (habitat / "README.md").write_text("upstream fixture\n", encoding="utf-8")
    upstream_commit = _commit_all(habitat, "upstream baseline")
    _git(
        habitat,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "--quiet",
        str(rlr_source),
        "src/deps/rlr-audio-propagation",
    )
    rlr_checkout = habitat / "src/deps/rlr-audio-propagation"
    _git(rlr_checkout, "remote", "set-url", "origin", RLR_URL)
    native = habitat / "build"
    native.mkdir()
    (native / "habitat_sim_fixture.so").write_bytes(b"habitat binding\n")
    (native / "libRLRAudioPropagation.so").write_bytes(b"RLR binary\n")
    habitat_commit = _commit_all(habitat, "fork implementation")

    avengine = tmp_path / "avengine"
    _init_repository(avengine, origin=AVENGINE_URL)
    (avengine / ".gitignore").write_text("tmp/\n", encoding="utf-8")
    release_tool = avengine / "tools" / "release" / "build_manifest.py"
    release_tool.parent.mkdir(parents=True)
    release_tool.write_text("# fixture release verifier\n", encoding="utf-8")
    schemas = avengine / "schemas"
    schemas.mkdir()
    _write_json(
        schemas / "fixture_v1.schema.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://avengine.local/schema/fixture_v1.schema.json",
            "type": "object",
        },
    )
    implementation_commit = _commit_all(avengine, "implementation A")

    evidence_root = avengine / "tmp" / "formal"
    controlled_root = evidence_root / "controlled"
    controlled_root.mkdir(parents=True)
    release_reference = controlled_root / "release_manifest_ref.json"
    _write_json(
        release_reference,
        {
            "schema": "avengine_release_manifest_ref_v1",
            "release_id": "avengine-m6-fixture",
            "expected_tag": "v0.2.0-m6-fixture",
            "repository_path": "release/avengine_release_manifest_v1.json",
            "implementation_commit": implementation_commit,
        },
    )
    controlled_payload = controlled_root / "payload.bin"
    controlled_payload.write_bytes(b"controlled canary artifact\n")
    controlled = controlled_root / "evidence.json"
    release_reference_record = _record(
        release_reference, relative_to=controlled_root
    )
    _write_json(
        controlled,
        {
            "schema": "avengine_m6_canary_evidence_v1",
            "implementation_commit": implementation_commit,
            "overall_status": "pass",
            "research_only": True,
            "qualification_claim": False,
            "release_manifest_ref": release_reference_record,
            "artifacts": {
                "payload.bin": _record(
                    controlled_payload, relative_to=controlled_root
                ),
                "release_manifest_ref.json": release_reference_record,
            },
        },
    )
    room_root = evidence_root / "rooms"
    room_artifact_paths: list[Path] = []
    for index in range(6):
        artifact = room_root / "reports" / f"case_{index}.json"
        _write_json(artifact, {"case": index})
        room_artifact_paths.append(artifact)
    room = room_root / "attempt_manifest.json"
    _write_json(
        room,
        {
            "schema": "avengine_m6_room_qualification_attempt_v1",
            "code_provenance": {
                "commit": implementation_commit,
                "worktree_clean": True,
            },
            "case_ids": [
                "blender_custom_two_zone",
                "replicacad_apt_0",
                "legacy_ue_apartment",
                "mp3d_17DRP5sb8fy_raw",
                "mp3d_17DRP5sb8fy_derived",
                "independent_corrupted_fixture",
            ],
            "reports": [{"case": index} for index in range(6)],
            "artifacts": [
                _record(path, relative_to=room_root)
                for path in room_artifact_paths
            ],
        },
    )
    bundle_ids = {
        "controlled": "m6-controlled-formal",
        "room": "m6-room-formal",
        "tests": "m6-test-layers-formal",
    }
    junit_path = "tmp/formal/tests/fixture.junit.xml"
    layer_command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit",
        "--junitxml",
        junit_path,
    ]
    executed_layer_ids = (
        "fast-unit",
        "slow-hermetic",
        "native-habitat",
        "rlr-audio",
        "blender-assets",
        "media-readback",
    )
    test_receipt_specs: dict[str, dict[str, str]] = {}
    for layer_id in executed_layer_ids:
        receipt = evidence_root / "tests" / f"{layer_id}.json"
        _write_json(
            receipt,
            {
                "schema": "avengine_m6_test_execution_receipt_v1",
                "receipt_id": f"{layer_id}-fixture-receipt",
                "test_layer_id": layer_id,
                "status": "pass",
                "command": layer_command,
                "exit_code": 0,
                "implementation_commit": implementation_commit,
                "habitat_runtime_commit": habitat_commit,
                "rlr_commit": rlr_commit,
                **_fixture_receipt_execution(junit_path),
            },
        )
        test_receipt_specs[layer_id] = {
            "root_id": "avengine",
            "path": receipt.relative_to(avengine).as_posix(),
        }
    test_layers = {
        layer_id: {
            "status": "pass",
            "command": layer_command,
            "evidence_bundle_ids": [bundle_ids["tests"]],
            "receipt_artifacts": [test_receipt_specs[layer_id]],
            "summary": "Hermetic fixture evidence.",
        }
        for layer_id in executed_layer_ids
    }
    test_layers["release-canary"] = {
        "status": "not_run",
        "command": [
            sys.executable,
            "tools/release/build_manifest.py",
            "verify",
            "--manifest",
            "release/avengine_release_manifest_v1.json",
            "--avengine-root",
            str(avengine),
            "--habitat-runtime-root",
            str(habitat),
            "--output",
            "tmp/formal/release_attestation.json",
        ],
        "evidence_bundle_ids": list(bundle_ids.values()),
        "receipt_artifacts": [],
        "reason": "Post-tag final attestation requires metadata commit B and its annotated tag.",
    }
    request = evidence_root / "release_build_request.json"
    _write_json(
        request,
        {
            "schema": "avengine_release_build_request_v1",
            "release": {
                "release_id": "avengine-m6-fixture",
                "tag": "v0.2.0-m6-fixture",
                "state": "candidate",
                "current_milestone": "M6",
                "manifest_path": "release/avengine_release_manifest_v1.json",
            },
            "repositories": {
                "implementation_commit": implementation_commit,
                "expected_habitat_commit": habitat_commit,
                "upstream_commit": upstream_commit,
                "expected_rlr_commit": rlr_commit,
                "rlr_submodule_path": "src/deps/rlr-audio-propagation",
                "expected_avengine_repository": AVENGINE_URL,
                "expected_habitat_repository": HABITAT_URL,
                "expected_upstream_repository": UPSTREAM_URL,
                "expected_rlr_repository": RLR_URL,
            },
            "native_artifacts": {
                "habitat_sim_binding": {
                    "root_id": "habitat_runtime",
                    "path": "build/habitat_sim_fixture.so",
                },
                "rlr_binary": {
                    "root_id": "habitat_runtime",
                    "path": "build/libRLRAudioPropagation.so",
                },
            },
            "environment": {
                "compiler": {"id": "python-fixture", "command": sys.executable},
                "python_dependencies": ["jsonschema"],
            },
            "evidence_bundles": [
                {
                    "evidence_id": bundle_ids["controlled"],
                    "status_scope": "controlled_canary_verifier",
                    "status": "pass",
                    "artifacts": [
                        {
                            "root_id": "avengine",
                            "path": controlled.relative_to(avengine).as_posix(),
                        }
                    ],
                },
                {
                    "evidence_id": bundle_ids["room"],
                    "status_scope": "room_attempt_verifier",
                    "status": "pass",
                    "artifacts": [
                        {
                            "root_id": "avengine",
                            "path": room.relative_to(avengine).as_posix(),
                        }
                    ],
                },
                {
                    "evidence_id": bundle_ids["tests"],
                    "status_scope": "test_execution",
                    "status": "pass",
                    "artifacts": list(test_receipt_specs.values()),
                },
            ],
            "m6_evidence": {
                "controlled_canary_bundle_id": bundle_ids["controlled"],
                "room_qualification_bundle_id": bundle_ids["room"],
            },
            "test_layers": test_layers,
        },
    )
    return avengine, habitat, request, implementation_commit, habitat_commit


def test_prepare_then_direct_child_and_annotated_tag_verify(tmp_path: Path) -> None:
    avengine, habitat, request, implementation_commit, _ = _build_request_fixture(
        tmp_path
    )
    manifest_path = prepare_release_manifest(
        request,
        avengine_root=avengine,
        habitat_runtime_root=habitat,
    )
    manifest = load_json_strict(manifest_path)
    assert manifest["repositories"]["avengine"]["implementation_commit"] == (
        implementation_commit
    )
    assert all(
        record["path"] != "release/avengine_release_manifest_v1.json"
        for bundle in manifest["evidence_bundles"]
        for record in bundle["artifacts"]
    )
    bundles = {
        bundle["evidence_id"]: bundle for bundle in manifest["evidence_bundles"]
    }
    assert bundles["m6-controlled-formal"]["status_scope"] == (
        "controlled_canary_verifier"
    )
    assert bundles["m6-room-formal"]["status_scope"] == "room_attempt_verifier"
    assert len(bundles["m6-controlled-formal"]["artifacts"]) == 3
    assert len(bundles["m6-room-formal"]["artifacts"]) == 7
    assert any(
        record["path"].endswith("controlled/release_manifest_ref.json")
        for record in bundles["m6-controlled-formal"]["artifacts"]
    )
    assert _git(avengine, "rev-parse", "HEAD") == implementation_commit
    assert manifest["test_layers"]["release-canary"]["status"] == "not_run"
    assert manifest["test_layers"]["release-canary"]["receipt_artifacts"] == []
    assert manifest["test_layers"]["fast-unit"]["receipt_artifacts"][0] in (
        bundles["m6-test-layers-formal"]["artifacts"]
    )

    metadata_commit = _commit_all(avengine, "release metadata B")
    assert _git(avengine, "rev-parse", "HEAD^") == implementation_commit
    _git(
        avengine,
        "tag",
        "-a",
        manifest["release"]["tag"],
        "-m",
        "M6 fixture release",
    )
    report = verify_release_manifest(
        manifest_path,
        avengine_root=avengine,
        habitat_runtime_root=habitat,
        schema_path=RELEASE_SCHEMA,
    )
    assert report["status"] == "pass", report
    assert report["observed"]["avengine_metadata_commit"] == metadata_commit
    assert report["observed"]["avengine_metadata_parent_count"] == 1


def test_verify_cli_persists_post_tag_attestation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    avengine, habitat, request, implementation_commit, _ = _build_request_fixture(
        tmp_path
    )
    manifest_path = prepare_release_manifest(
        request,
        avengine_root=avengine,
        habitat_runtime_root=habitat,
    )
    metadata_commit = _commit_all(avengine, "release metadata B")
    manifest = load_json_strict(manifest_path)
    _git(
        avengine,
        "tag",
        "-a",
        manifest["release"]["tag"],
        "-m",
        "M6 fixture release",
    )
    output = avengine / "tmp" / "formal" / "release_attestation.json"
    monkeypatch.setattr(
        "tools.release.build_manifest.__file__",
        str(avengine / "tools" / "release" / "build_manifest.py"),
    )
    exit_code = release_tool_main(
        [
            "verify",
            "--manifest",
            str(manifest_path),
            "--avengine-root",
            str(avengine),
            "--habitat-runtime-root",
            str(habitat),
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0, capsys.readouterr().out
    attestation = load_json_strict(output)
    assert attestation["status"] == "pass"
    assert attestation["release_tag_commit"] == metadata_commit
    assert attestation["implementation_commit"] == implementation_commit
    assert attestation["verification_report"]["status"] == "pass"
    assert "--output" in attestation["verification_command"]
    assert (
        release_tool_main(
            [
                "verify-attestation",
                "--attestation",
                str(output),
                "--avengine-root",
                str(avengine),
                "--habitat-runtime-root",
                str(habitat),
            ]
        )
        == 0
    ), capsys.readouterr().out


def test_prepare_rejects_pass_layer_without_hash_bound_receipt(tmp_path: Path) -> None:
    avengine, habitat, request, _, _ = _build_request_fixture(tmp_path)
    request_value = load_json_strict(request)
    request_value["test_layers"]["fast-unit"]["receipt_artifacts"] = []
    _write_json(request, request_value)
    with pytest.raises(ReleaseManifestError, match="receipt_artifacts"):
        prepare_release_manifest(
            request,
            avengine_root=avengine,
            habitat_runtime_root=habitat,
        )


def test_prepare_rejects_incomplete_planned_release_verify_command(
    tmp_path: Path,
) -> None:
    avengine, habitat, request, _, _ = _build_request_fixture(tmp_path)
    request_value = load_json_strict(request)
    command = request_value["test_layers"]["release-canary"]["command"]
    output_index = command.index("--output")
    del command[output_index : output_index + 2]
    _write_json(request, request_value)
    with pytest.raises(ReleaseManifestError, match="exactly one --output"):
        prepare_release_manifest(
            request,
            avengine_root=avengine,
            habitat_runtime_root=habitat,
        )


@pytest.mark.parametrize(
    "mutation",
    ("echo_prefix", "wrong_script", "equals_override", "prepare_subcommand"),
)
def test_prepare_rejects_noncanonical_planned_release_verify_command(
    tmp_path: Path, mutation: str
) -> None:
    avengine, habitat, request, _, _ = _build_request_fixture(tmp_path)
    request_value = load_json_strict(request)
    command = request_value["test_layers"]["release-canary"]["command"]
    if mutation == "echo_prefix":
        command[0:3] = ["echo", "tools/release/build_manifest.py", "verify"]
    elif mutation == "wrong_script":
        command[1] = "tools/release/not_the_release_tool.py"
    elif mutation == "equals_override":
        command.append("--manifest=tmp/attacker-controlled.json")
    else:
        command[2] = "prepare"
    _write_json(request, request_value)
    with pytest.raises(ReleaseManifestError, match="planned release-canary"):
        prepare_release_manifest(
            request,
            avengine_root=avengine,
            habitat_runtime_root=habitat,
        )


def test_planned_release_verify_command_closes_external_artifact_roots(
    tmp_path: Path,
) -> None:
    avengine, habitat, request, _, _ = _build_request_fixture(tmp_path)
    external_root = tmp_path / "published-evidence"
    external_root.mkdir()
    (external_root / "package.bin").write_bytes(b"published evidence\n")
    request_value = load_json_strict(request)
    request_value["evidence_bundles"].append(
        {
            "evidence_id": "external-artifact-integrity",
            "status_scope": "artifact_integrity",
            "status": "pass",
            "artifacts": [{"root_id": "published", "path": "package.bin"}],
        }
    )
    _write_json(request, request_value)
    with pytest.raises(ReleaseManifestError, match="lacks artifact roots"):
        prepare_release_manifest(
            request,
            avengine_root=avengine,
            habitat_runtime_root=habitat,
            artifact_roots={"published": external_root},
        )

    command = request_value["test_layers"]["release-canary"]["command"]
    command.extend(["--artifact-root", f"published={external_root}"])
    _write_json(request, request_value)
    manifest_path = prepare_release_manifest(
        request,
        avengine_root=avengine,
        habitat_runtime_root=habitat,
        artifact_roots={"published": external_root},
    )
    manifest = load_json_strict(manifest_path)
    assert command == manifest["test_layers"]["release-canary"]["command"]


def test_prepare_rejects_receipt_for_different_layer(tmp_path: Path) -> None:
    avengine, habitat, request, _, _ = _build_request_fixture(tmp_path)
    request_value = load_json_strict(request)
    request_value["test_layers"]["fast-unit"]["receipt_artifacts"] = request_value[
        "test_layers"
    ]["slow-hermetic"]["receipt_artifacts"]
    _write_json(request, request_value)
    with pytest.raises(ReleaseManifestError, match="test_layer_id mismatch"):
        prepare_release_manifest(
            request,
            avengine_root=avengine,
            habitat_runtime_root=habitat,
        )


def test_prepare_rejects_receipt_outside_referenced_bundles(tmp_path: Path) -> None:
    avengine, habitat, request, _, _ = _build_request_fixture(tmp_path)
    request_value = load_json_strict(request)
    request_value["test_layers"]["fast-unit"]["receipt_artifacts"] = [
        {
            "root_id": "avengine",
            "path": "tmp/formal/controlled/payload.bin",
        }
    ]
    _write_json(request, request_value)
    with pytest.raises(ReleaseManifestError, match="not a member of a referenced"):
        prepare_release_manifest(
            request,
            avengine_root=avengine,
            habitat_runtime_root=habitat,
        )


def test_prepare_keeps_release_canary_for_post_tag_attestation(
    tmp_path: Path,
) -> None:
    avengine, habitat, request, _, _ = _build_request_fixture(tmp_path)
    request_value = load_json_strict(request)
    release_canary = request_value["test_layers"]["release-canary"]
    release_canary["status"] = "pass"
    release_canary.pop("reason")
    release_canary["receipt_artifacts"] = [
        request_value["test_layers"]["fast-unit"]["receipt_artifacts"][0]
    ]
    _write_json(request, request_value)
    with pytest.raises(ReleaseManifestError, match="post-tag final attestation not_run"):
        prepare_release_manifest(
            request,
            avengine_root=avengine,
            habitat_runtime_root=habitat,
        )


def test_prepare_rejects_released_state_before_post_tag_attestation(
    tmp_path: Path,
) -> None:
    avengine, habitat, request, _, _ = _build_request_fixture(tmp_path)
    request_value = load_json_strict(request)
    request_value["release"]["state"] = "released"
    _write_json(request, request_value)
    with pytest.raises(ReleaseManifestError, match="must be candidate"):
        prepare_release_manifest(
            request,
            avengine_root=avengine,
            habitat_runtime_root=habitat,
        )


def test_prepare_rejects_room_attempt_not_bound_to_clean_commit_a(
    tmp_path: Path,
) -> None:
    avengine, habitat, request, _, _ = _build_request_fixture(tmp_path)
    request_value = load_json_strict(request)
    room_spec = request_value["evidence_bundles"][1]["artifacts"][0]
    room_path = avengine / room_spec["path"]
    room = load_json_strict(room_path)
    room["code_provenance"]["worktree_clean"] = False
    _write_json(room_path, room)
    with pytest.raises(ReleaseManifestError, match="clean worktree"):
        prepare_release_manifest(
            request,
            avengine_root=avengine,
            habitat_runtime_root=habitat,
        )


def test_prepare_rejects_commit_a_drift(tmp_path: Path) -> None:
    avengine, habitat, request, _, _ = _build_request_fixture(tmp_path)
    request_value = load_json_strict(request)
    request_value["repositories"]["implementation_commit"] = "0" * 40
    _write_json(request, request_value)
    with pytest.raises(ReleaseManifestError, match="current AVEngine HEAD"):
        prepare_release_manifest(
            request,
            avengine_root=avengine,
            habitat_runtime_root=habitat,
        )


def test_prepare_refuses_failed_authoritative_canary_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    avengine, habitat, request, _, _ = _build_request_fixture(tmp_path)
    monkeypatch.setattr(
        "avengine.m6.canary.verify_controlled_canary_evidence",
        lambda path: (
            "fail",
            [{"check_id": "artifact_closure", "status": "fail"}],
        ),
    )
    with pytest.raises(ReleaseManifestError, match="artifact_closure"):
        prepare_release_manifest(
            request,
            avengine_root=avengine,
            habitat_runtime_root=habitat,
        )
    assert not (avengine / "release/avengine_release_manifest_v1.json").exists()


def test_prepare_rejects_missing_declared_room_closure_file(tmp_path: Path) -> None:
    avengine, habitat, request, _, _ = _build_request_fixture(tmp_path)
    room_manifest = load_json_strict(
        avengine / "tmp/formal/rooms/attempt_manifest.json"
    )
    missing = avengine / "tmp/formal/rooms" / room_manifest["artifacts"][0]["path"]
    missing.unlink()
    with pytest.raises(ReleaseManifestError, match="unable to resolve artifact"):
        prepare_release_manifest(
            request,
            avengine_root=avengine,
            habitat_runtime_root=habitat,
        )


def test_prepare_rejects_tampered_controlled_closure_file(tmp_path: Path) -> None:
    avengine, habitat, request, _, _ = _build_request_fixture(tmp_path)
    controlled_payload = avengine / "tmp/formal/controlled/payload.bin"
    controlled_payload.write_bytes(b"tampered controlled artifact\n")
    with pytest.raises(ReleaseManifestError, match="declared (byte size|SHA-256)"):
        prepare_release_manifest(
            request,
            avengine_root=avengine,
            habitat_runtime_root=habitat,
        )


def test_prepare_rejects_extra_undeclared_room_file(tmp_path: Path) -> None:
    avengine, habitat, request, _, _ = _build_request_fixture(tmp_path)
    extra = avengine / "tmp/formal/rooms/unrecorded.log"
    extra.write_text("not in attempt_manifest artifacts\n", encoding="utf-8")
    with pytest.raises(ReleaseManifestError, match="extra=.*unrecorded.log"):
        prepare_release_manifest(
            request,
            avengine_root=avengine,
            habitat_runtime_root=habitat,
        )


def test_prepare_rejects_undeclared_symlink_directory_in_bundle(
    tmp_path: Path,
) -> None:
    avengine, habitat, request, _, _ = _build_request_fixture(tmp_path)
    outside = tmp_path / "outside_room_payload"
    outside.mkdir()
    (outside / "hidden.json").write_text("{}\n", encoding="utf-8")
    linked = avengine / "tmp/formal/rooms/extra_directory"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ReleaseManifestError, match="retained closure contains symlinks"):
        prepare_release_manifest(
            request,
            avengine_root=avengine,
            habitat_runtime_root=habitat,
        )


def test_prepare_refuses_to_replace_existing_release_tag(tmp_path: Path) -> None:
    avengine, habitat, request, _, _ = _build_request_fixture(tmp_path)
    _git(avengine, "tag", "v0.2.0-m6-fixture")
    with pytest.raises(ReleaseManifestError, match="already exists"):
        prepare_release_manifest(
            request,
            avengine_root=avengine,
            habitat_runtime_root=habitat,
        )
