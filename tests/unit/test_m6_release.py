from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from avengine.release import (
    ReleaseManifestError,
    build_file_record,
    canonical_file_record_set_sha256,
    load_json_strict,
    require_verified_release_manifest,
    verify_release_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RELEASE_SCHEMA = (
    REPOSITORY_ROOT / "schemas" / "avengine_release_manifest_v1.schema.json"
)
AVENGINE_URL = "https://github.com/Eastforward/AVEngine.git"
HABITAT_URL = "https://github.com/Eastforward/habitat-sim-AVEngine.git"
UPSTREAM_URL = "https://github.com/facebookresearch/habitat-sim.git"
RLR_URL = "https://github.com/facebookresearch/rlr-audio-propagation.git"


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _init_repository(path: Path, *, origin: str) -> None:
    path.mkdir(parents=True)
    _git(path, "init", "--quiet")
    _git(path, "config", "user.name", "AVEngine release fixture")
    _git(path, "config", "user.email", "release-fixture@example.invalid")
    _git(path, "remote", "add", "origin", origin)


def _commit_all(path: Path, message: str) -> str:
    _git(path, "add", "--all")
    _git(path, "commit", "--quiet", "-m", message)
    return _git(path, "rev-parse", "HEAD")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class ReleaseFixture:
    avengine: Path
    habitat: Path
    manifest_path: Path
    release_tag: str
    implementation_commit: str
    metadata_commit: str
    upstream_commit: str
    habitat_commit: str
    rlr_commit: str


def _make_release_fixture(tmp_path: Path) -> ReleaseFixture:
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
    (habitat / "build").mkdir()
    habitat_binding = habitat / "build" / "habitat_sim_fixture.so"
    rlr_binary = habitat / "build" / "librlr_audio_fixture.so"
    habitat_binding.write_bytes(b"habitat native binding fixture\n")
    rlr_binary.write_bytes(b"RLR native binary fixture\n")
    habitat_commit = _commit_all(habitat, "fork implementation")

    avengine = tmp_path / "avengine"
    _init_repository(avengine, origin=AVENGINE_URL)
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
    evidence = avengine / "evidence" / "fast-unit.json"
    _write_json(
        evidence,
        {
            "command": ["python", "-m", "pytest", "tests/unit"],
            "status": "pass",
        },
    )
    controlled_entry = avengine / "evidence" / "controlled.json"
    room_entry = avengine / "evidence" / "room.json"
    _write_json(controlled_entry, {"fixture_role": "controlled_canary"})
    _write_json(room_entry, {"fixture_role": "room_qualification"})
    implementation_commit = _commit_all(avengine, "AVEngine implementation")

    schema_records = [
        build_file_record(path, root=avengine, root_id="avengine")
        for path in sorted(schemas.rglob("*.json"))
    ]
    evidence_records = [
        build_file_record(evidence, root=avengine, root_id="avengine")
    ]
    controlled_record = build_file_record(
        controlled_entry, root=avengine, root_id="avengine"
    )
    room_record = build_file_record(room_entry, root=avengine, root_id="avengine")
    evidence_id = "fast-unit-fixture"
    release_tag = "v0.1.0-m6-fixture"
    manifest: dict[str, Any] = {
        "schema": "avengine_release_manifest_v1",
        "release": {
            "release_id": "avengine-m6-fixture",
            "tag": release_tag,
            "state": "candidate",
            "current_milestone": "M6",
            "manifest_path": "release/avengine_release_manifest_v1.json",
            "metadata_commit_policy": {
                "mode": "direct_child_of_implementation",
                "allowed_changed_paths": [
                    "release/avengine_release_manifest_v1.json"
                ],
                "require_clean_worktrees": True,
                "require_annotated_tag": True,
            },
        },
        "repositories": {
            "avengine": {
                "repository": AVENGINE_URL,
                "implementation_commit": implementation_commit,
            },
            "habitat_runtime": {
                "repository": HABITAT_URL,
                "commit": habitat_commit,
                "upstream_repository": UPSTREAM_URL,
                "upstream_commit": upstream_commit,
                "rlr_repository": RLR_URL,
                "rlr_commit": rlr_commit,
                "rlr_submodule_path": "src/deps/rlr-audio-propagation",
            },
        },
        "schemas": {
            "directory": "schemas",
            "algorithm": "sha256_canonical_file_records_v1",
            "files": schema_records,
            "set_sha256": canonical_file_record_set_sha256(schema_records),
        },
        "native_artifacts": {
            "habitat_sim_binding": build_file_record(
                habitat_binding,
                root=habitat,
                root_id="habitat_runtime",
            ),
            "rlr_binary": build_file_record(
                rlr_binary,
                root=habitat,
                root_id="habitat_runtime",
            ),
        },
        "environment": {
            "os": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "python": {
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
            },
            "compiler": {
                "id": "fixture-version-command",
                "command": sys.executable,
                "version": platform.python_version(),
            },
            "python_dependencies": [
                {
                    "distribution": "jsonschema",
                    "version": importlib_metadata.version("jsonschema"),
                }
            ],
        },
        "evidence_bundles": [
            {
                "evidence_id": evidence_id,
                "status": "pass",
                "artifacts": evidence_records,
                "bundle_sha256": canonical_file_record_set_sha256(evidence_records),
            },
            {
                "evidence_id": "m6-controlled-fixture",
                "status": "pass",
                "artifacts": [controlled_record],
                "bundle_sha256": canonical_file_record_set_sha256(
                    [controlled_record]
                ),
            },
            {
                "evidence_id": "m6-room-fixture",
                "status": "pass",
                "artifacts": [room_record],
                "bundle_sha256": canonical_file_record_set_sha256([room_record]),
            },
        ],
        "m6_evidence": {
            "controlled_canary_bundle_id": "m6-controlled-fixture",
            "controlled_canary_entry": controlled_record,
            "room_qualification_bundle_id": "m6-room-fixture",
            "room_qualification_entry": room_record,
        },
        "test_layers": {
            "fast-unit": {
                "status": "pass",
                "command": ["python", "-m", "pytest", "tests/unit"],
                "evidence_bundle_ids": [evidence_id],
                "summary": "Fixture fast unit evidence.",
            },
            **{
                layer: {
                    "status": "not_run",
                    "command": [],
                    "evidence_bundle_ids": [],
                    "reason": "Not exercised by the hermetic release fixture.",
                }
                for layer in (
                    "slow-hermetic",
                    "native-habitat",
                    "rlr-audio",
                    "blender-assets",
                    "media-readback",
                    "release-canary",
                )
            },
        },
    }
    manifest_path = avengine / manifest["release"]["manifest_path"]
    _write_json(manifest_path, manifest)
    metadata_commit = _commit_all(avengine, "M6 release metadata")
    _git(avengine, "tag", "-a", release_tag, "-m", "M6 release fixture")

    return ReleaseFixture(
        avengine=avengine,
        habitat=habitat,
        manifest_path=manifest_path,
        release_tag=release_tag,
        implementation_commit=implementation_commit,
        metadata_commit=metadata_commit,
        upstream_commit=upstream_commit,
        habitat_commit=habitat_commit,
        rlr_commit=rlr_commit,
    )


def _check_errors(report: dict[str, Any], check_id: str) -> list[str]:
    check = next(item for item in report["checks"] if item["check_id"] == check_id)
    return check["errors"]


def _verify(fixture: ReleaseFixture, **overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "avengine_root": fixture.avengine,
        "habitat_runtime_root": fixture.habitat,
        "schema_path": RELEASE_SCHEMA,
        "verify_m6_evidence": False,
    }
    arguments.update(overrides)
    return verify_release_manifest(fixture.manifest_path, **arguments)


def test_release_schema_is_valid_draft_2020_12() -> None:
    Draft202012Validator.check_schema(load_json_strict(RELEASE_SCHEMA))


def test_release_manifest_verifies_hashes_git_tag_and_environment(
    tmp_path: Path,
) -> None:
    fixture = _make_release_fixture(tmp_path)
    report = _verify(fixture)
    assert report["status"] == "pass", report
    assert report["observed"]["avengine_metadata_commit"] == fixture.metadata_commit
    assert report["observed"]["avengine_metadata_parent"] == (
        fixture.implementation_commit
    )
    assert report["observed"]["habitat_runtime_commit"] == fixture.habitat_commit
    assert report["observed"]["rlr_commit"] == fixture.rlr_commit


def test_release_verifier_runs_m6_semantic_roles_by_default(tmp_path: Path) -> None:
    fixture = _make_release_fixture(tmp_path)
    report = verify_release_manifest(
        fixture.manifest_path,
        avengine_root=fixture.avengine,
        habitat_runtime_root=fixture.habitat,
        schema_path=RELEASE_SCHEMA,
        verify_git=False,
        verify_environment=False,
    )
    assert report["status"] == "fail"
    assert any(
        "controlled-canary role does not point to M6 canary evidence v1" in error
        for error in _check_errors(report, "m6_evidence")
    )


def test_release_m6_role_entry_must_belong_to_named_bundle(tmp_path: Path) -> None:
    fixture = _make_release_fixture(tmp_path)
    manifest = load_json_strict(fixture.manifest_path)
    manifest["m6_evidence"]["controlled_canary_entry"] = manifest[
        "m6_evidence"
    ]["room_qualification_entry"]
    _write_json(fixture.manifest_path, manifest)
    report = _verify(
        fixture,
        verify_git=False,
        verify_environment=False,
    )
    assert report["status"] == "fail"
    assert any(
        "controlled canary entry is not in its exact evidence closure" in error
        for error in _check_errors(report, "m6_evidence")
    )


def test_release_manifest_detects_tampered_evidence(tmp_path: Path) -> None:
    fixture = _make_release_fixture(tmp_path)
    (fixture.avengine / "evidence" / "fast-unit.json").write_text(
        '{"status":"tampered"}\n', encoding="utf-8"
    )
    report = _verify(fixture, verify_git=False, verify_environment=False)
    assert report["status"] == "fail"
    errors = _check_errors(report, "evidence_bundles")
    assert any("SHA-256 mismatch" in error for error in errors)


def test_release_manifest_detects_schema_set_hash_mismatch(tmp_path: Path) -> None:
    fixture = _make_release_fixture(tmp_path)
    manifest = load_json_strict(fixture.manifest_path)
    manifest["schemas"]["set_sha256"] = "0" * 64
    _write_json(fixture.manifest_path, manifest)
    report = _verify(fixture, verify_git=False, verify_environment=False)
    assert report["status"] == "fail"
    assert "schema set SHA-256 mismatch" in _check_errors(report, "schema_set")


def test_release_manifest_detects_runtime_commit_drift(tmp_path: Path) -> None:
    fixture = _make_release_fixture(tmp_path)
    (fixture.habitat / "runtime-drift.txt").write_text("drift\n", encoding="utf-8")
    _commit_all(fixture.habitat, "runtime drift")
    report = _verify(fixture, verify_environment=False)
    assert report["status"] == "fail"
    assert any(
        "Habitat runtime HEAD mismatch" in error
        for error in _check_errors(report, "git_identity")
    )


def test_release_manifest_requires_annotated_tag(tmp_path: Path) -> None:
    fixture = _make_release_fixture(tmp_path)
    _git(fixture.avengine, "tag", "-d", fixture.release_tag)
    _git(fixture.avengine, "tag", fixture.release_tag)
    report = _verify(fixture, verify_environment=False)
    assert report["status"] == "fail"
    assert any(
        "is not annotated" in error
        for error in _check_errors(report, "git_identity")
    )


def test_release_metadata_commit_rejects_non_release_changes(tmp_path: Path) -> None:
    fixture = _make_release_fixture(tmp_path)
    _git(fixture.avengine, "tag", "-d", fixture.release_tag)
    (fixture.avengine / "README.md").write_text("not release metadata\n", encoding="utf-8")
    _commit_all(fixture.avengine, "disallowed metadata change")
    _git(
        fixture.avengine,
        "tag",
        "-a",
        fixture.release_tag,
        "-m",
        "moved fixture tag",
    )
    report = _verify(fixture, verify_environment=False)
    assert report["status"] == "fail"
    assert any(
        "metadata commit changes non-release paths: README.md" in error
        for error in _check_errors(report, "git_identity")
    )


def test_require_verified_release_manifest_raises_structured_error(
    tmp_path: Path,
) -> None:
    fixture = _make_release_fixture(tmp_path)
    manifest = load_json_strict(fixture.manifest_path)
    manifest["test_layers"]["fast-unit"]["evidence_bundle_ids"] = ["missing"]
    _write_json(fixture.manifest_path, manifest)
    with pytest.raises(ReleaseManifestError, match="unknown evidence bundles"):
        require_verified_release_manifest(
            fixture.manifest_path,
            avengine_root=fixture.avengine,
            habitat_runtime_root=fixture.habitat,
            schema_path=RELEASE_SCHEMA,
            verify_git=False,
            verify_environment=False,
            verify_m6_evidence=False,
        )


def test_build_file_record_rejects_root_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    with pytest.raises(ReleaseManifestError, match="escapes root"):
        build_file_record(outside, root=root, root_id="avengine")


def test_build_file_record_rejects_symlinked_input(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "target.bin"
    target.write_bytes(b"target")
    linked = root / "linked.bin"
    linked.symlink_to(target.name)
    with pytest.raises(ReleaseManifestError, match="symlink"):
        build_file_record(linked, root=root, root_id="avengine")
