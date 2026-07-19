"""Verification helpers for the cross-repository AVEngine release manifest.

The release manifest intentionally binds an AVEngine *implementation* commit,
not the commit that contains the manifest itself.  The containing metadata
commit is observed from Git, must be a direct child of the implementation
commit, and may only change the allowlisted release paths.  This avoids an
impossible self-reference while still making the release tag and both source
repositories independently verifiable.
"""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator

from avengine.release_receipt import verify_receipt_payload
from avengine.security.path_policy import (
    WorkspacePathPolicy,
    write_bytes_no_clobber,
)


RELEASE_MANIFEST_SCHEMA = "avengine_release_manifest_v1"
RELEASE_BUILD_REQUEST_SCHEMA = "avengine_release_build_request_v1"
RELEASE_ATTESTATION_SCHEMA = "avengine_release_attestation_v1"
TEST_EXECUTION_RECEIPT_SCHEMA = "avengine_m6_test_execution_receipt_v1"
SCHEMA_SET_ALGORITHM = "sha256_canonical_file_records_v1"
TEST_LAYER_IDS = (
    "fast-unit",
    "slow-hermetic",
    "native-habitat",
    "rlr-audio",
    "blender-assets",
    "media-readback",
    "release-canary",
)
RELEASE_VERIFICATION_CHECK_IDS = (
    "manifest_json",
    "manifest_schema",
    "schema_set",
    "native_artifacts",
    "evidence_bundles",
    "m6_evidence",
    "test_layers",
    "environment",
    "git_identity",
)

_HEX_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_STABLE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_PYTHON_EXECUTABLE_NAME = re.compile(r"^python(?:3(?:\.\d+)?)?$")
_VERIFICATION_STATUSES = {"pass", "fail", "blocked", "not_run"}
_M6_FAST_UNIT_MARKER = "not integration and not canary"
_M6_FAST_UNIT_COMMAND_TAIL = (
    "-m",
    "pytest",
    "-q",
    "tests/unit",
    "-m",
    _M6_FAST_UNIT_MARKER,
    "--junitxml",
)
_EVIDENCE_STATUS_SCOPES = {
    "artifact_integrity",
    "controlled_canary_verifier",
    "room_attempt_verifier",
    "test_execution",
}


class ReleaseManifestError(ValueError):
    """The release manifest or its bound evidence is invalid."""

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def release_schema_path() -> Path:
    return _repository_root() / "schemas" / "avengine_release_manifest_v1.schema.json"


def release_attestation_schema_path() -> Path:
    return _repository_root() / "schemas" / "avengine_release_attestation_v1.schema.json"


def test_execution_receipt_schema_path() -> Path:
    return _repository_root() / "schemas" / "m6_test_execution_receipt_v1.schema.json"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def load_json_strict(path: str | Path) -> dict[str, Any]:
    """Load one UTF-8 JSON object while rejecting duplicate/non-finite input."""

    source = Path(path)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ReleaseManifestError([f"unable to load {source}: {exc}"]) from exc
    if not isinstance(value, dict):
        raise ReleaseManifestError([f"{source} must contain one JSON object"])
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_file_record(
    path: str | Path,
    *,
    root: str | Path,
    root_id: str,
) -> dict[str, Any]:
    """Build a portable, root-relative byte identity for one regular file."""

    base = Path(root).resolve(strict=True)
    unresolved_source = Path(path).absolute()
    lexical_cursor = unresolved_source
    while True:
        if lexical_cursor.is_symlink():
            raise ReleaseManifestError(
                [f"file record source traverses a symlink: {unresolved_source}"]
            )
        if lexical_cursor == base or lexical_cursor == lexical_cursor.parent:
            break
        lexical_cursor = lexical_cursor.parent
    source = unresolved_source.resolve(strict=True)
    try:
        relative = source.relative_to(base)
    except ValueError as exc:
        raise ReleaseManifestError([f"file escapes root {root_id}: {source}"]) from exc
    if not source.is_file():
        raise ReleaseManifestError([f"file record source is not regular: {source}"])
    cursor = base
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ReleaseManifestError(
                [f"file record source traverses a symlink: {source}"]
            )
    return {
        "root_id": root_id,
        "path": relative.as_posix(),
        "byte_size": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def canonical_file_record_set_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    """Hash a sorted set of portable file records using the v1 algorithm."""

    canonical: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        item = {
            "root_id": record.get("root_id"),
            "path": record.get("path"),
            "byte_size": record.get("byte_size"),
            "sha256": record.get("sha256"),
        }
        identity = (str(item["root_id"]), str(item["path"]))
        if identity in seen:
            raise ReleaseManifestError(
                [f"duplicate file record {identity[0]}:{identity[1]}"]
            )
        seen.add(identity)
        canonical.append(item)
    canonical.sort(key=lambda item: (str(item["root_id"]), str(item["path"])))
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _m6_fast_unit_command_errors(command: Any) -> list[str]:
    """Validate the complete, non-narrowed M6 fast-unit pytest invocation."""

    owner = "test_layers.fast-unit.command"
    if not isinstance(command, list) or any(
        not isinstance(value, str) or not value for value in command
    ):
        return [f"{owner} must be an array of non-empty argv strings"]
    if len(command) != 9:
        return [
            f"{owner} must run the complete tests/unit fast layer with the "
            "canonical marker and one split-form --junitxml path"
        ]
    if _PYTHON_EXECUTABLE_NAME.fullmatch(Path(command[0]).name) is None:
        return [f"{owner}[0] must name a Python executable"]
    if tuple(command[1:8]) != _M6_FAST_UNIT_COMMAND_TAIL:
        return [
            f"{owner} must equal <python> -m pytest -q tests/unit -m "
            f"{_M6_FAST_UNIT_MARKER!r} --junitxml <repository-relative-path>; "
            "narrowing flags, node IDs and individual test files are forbidden"
        ]
    junit = Path(command[8])
    if (
        junit.is_absolute()
        or ".." in junit.parts
        or command[8] != junit.as_posix()
        or not command[8].endswith(".xml")
    ):
        return [
            f"{owner}[8] must be a normalized repository-relative JUnit XML path"
        ]
    return []


def validate_release_manifest_document(
    value: Mapping[str, Any],
    *,
    schema_path: str | Path | None = None,
) -> list[str]:
    """Return deterministic Draft 2020-12 validation errors."""

    selected_schema = release_schema_path() if schema_path is None else Path(schema_path)
    try:
        schema = load_json_strict(selected_schema)
    except ReleaseManifestError as exc:
        return exc.errors
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:  # pragma: no cover - jsonschema exception hierarchy varies
        return [f"release schema is invalid: {exc}"]
    validator = Draft202012Validator(schema)
    errors: list[str] = []
    for error in sorted(validator.iter_errors(dict(value)), key=lambda item: list(item.path)):
        location = ".".join(str(part) for part in error.path) or "$"
        errors.append(f"{location}: {error.message}")
    # The manifest is itself supplied by metadata commit B, so its policy must
    # not be able to self-authorize source or documentation changes outside the
    # release metadata directory.  Keep this semantic check independent of the
    # schema so callers cannot weaken the boundary with a custom schema path.
    release = value.get("release")
    if isinstance(release, Mapping):
        if release.get("current_milestone") != "M6":
            errors.append("release.current_milestone must equal 'M6'")
        manifest_path = release.get("manifest_path")
        if isinstance(manifest_path, str) and not manifest_path.startswith("release/"):
            errors.append(
                "release.manifest_path must remain beneath release/: "
                f"{manifest_path!r}"
            )
        policy = release.get("metadata_commit_policy")
        if isinstance(policy, Mapping):
            allowed = policy.get("allowed_changed_paths")
            if isinstance(allowed, list):
                for index, path in enumerate(allowed):
                    if isinstance(path, str) and not path.startswith("release/"):
                        errors.append(
                            "release.metadata_commit_policy.allowed_changed_paths"
                            f"[{index}] must remain beneath release/: {path!r}"
                        )
                if (
                    isinstance(manifest_path, str)
                    and all(isinstance(path, str) for path in allowed)
                    and not any(
                        path == manifest_path
                        or (path.endswith("/") and manifest_path.startswith(path))
                        for path in allowed
                    )
                ):
                    errors.append(
                        "release.metadata_commit_policy.allowed_changed_paths "
                        "must allow release.manifest_path "
                        f"{manifest_path!r}"
                    )
    test_layers = value.get("test_layers")
    if isinstance(test_layers, Mapping):
        fast_unit = test_layers.get("fast-unit")
        if isinstance(fast_unit, Mapping):
            if fast_unit.get("status") != "pass":
                errors.append("test_layers.fast-unit.status must equal 'pass'")
            errors.extend(_m6_fast_unit_command_errors(fast_unit.get("command")))
    return errors


def _validate_schema_document(
    value: Mapping[str, Any], *, schema_path: Path, owner: str
) -> list[str]:
    try:
        schema = load_json_strict(schema_path)
    except ReleaseManifestError as exc:
        return [f"{owner}: {error}" for error in exc.errors]
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:  # pragma: no cover - jsonschema hierarchy varies
        return [f"{owner} schema is invalid: {exc}"]
    validator = Draft202012Validator(schema)
    errors: list[str] = []
    for error in sorted(validator.iter_errors(dict(value)), key=lambda item: list(item.path)):
        location = ".".join(str(part) for part in error.path) or "$"
        errors.append(f"{owner}.{location}: {error.message}")
    return errors


def validate_test_execution_receipt_document(
    value: Mapping[str, Any], *, schema_path: str | Path | None = None
) -> list[str]:
    selected = (
        test_execution_receipt_schema_path()
        if schema_path is None
        else Path(schema_path)
    )
    return _validate_schema_document(
        value, schema_path=selected, owner="test execution receipt"
    )


def validate_release_attestation_document(
    value: Mapping[str, Any], *, schema_path: str | Path | None = None
) -> list[str]:
    selected = (
        release_attestation_schema_path() if schema_path is None else Path(schema_path)
    )
    return _validate_schema_document(
        value, schema_path=selected, owner="release attestation"
    )


def load_release_manifest(
    path: str | Path,
    *,
    schema_path: str | Path | None = None,
) -> dict[str, Any]:
    value = load_json_strict(path)
    errors = validate_release_manifest_document(value, schema_path=schema_path)
    if errors:
        raise ReleaseManifestError(errors)
    return value


def _check(checks: list[dict[str, Any]], check_id: str, errors: Sequence[str]) -> None:
    checks.append(
        {
            "check_id": check_id,
            "status": "pass" if not errors else "fail",
            "errors": list(errors),
        }
    )


def _resolve_record(
    record: Mapping[str, Any], roots: Mapping[str, Path]
) -> tuple[Path | None, list[str]]:
    root_id = record.get("root_id")
    relative = record.get("path")
    if not isinstance(root_id, str) or root_id not in roots:
        return None, [f"unknown artifact root_id {root_id!r}"]
    if not isinstance(relative, str):
        return None, ["file record path is not a string"]
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        return None, [f"file record path is not confined: {relative}"]
    try:
        base = roots[root_id].resolve(strict=True)
        unresolved_candidate = base / raw
        cursor = base
        for part in raw.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                return None, [
                    f"file record traverses a symlink: {root_id}:{relative}"
                ]
        candidate = unresolved_candidate.resolve(strict=True)
        candidate.relative_to(base)
    except (OSError, ValueError) as exc:
        return None, [f"unable to resolve confined record {root_id}:{relative}: {exc}"]
    if not candidate.is_file():
        return None, [f"record is not a regular file: {root_id}:{relative}"]
    return candidate, []


def _verify_file_records(
    records: Iterable[Mapping[str, Any]], roots: Mapping[str, Path]
) -> list[str]:
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        identity = (str(record.get("root_id")), str(record.get("path")))
        if identity in seen:
            errors.append(f"duplicate file record {identity[0]}:{identity[1]}")
            continue
        seen.add(identity)
        candidate, resolution_errors = _resolve_record(record, roots)
        errors.extend(resolution_errors)
        if candidate is None:
            continue
        actual_size = candidate.stat().st_size
        if record.get("byte_size") != actual_size:
            errors.append(
                f"byte size mismatch for {identity[0]}:{identity[1]}: "
                f"declared {record.get('byte_size')}, actual {actual_size}"
            )
        actual_hash = sha256_file(candidate)
        if record.get("sha256") != actual_hash:
            errors.append(f"SHA-256 mismatch for {identity[0]}:{identity[1]}")
    return errors


def _git_environment() -> dict[str, str]:
    """Return an environment in which Git cannot rewrite object ancestry."""

    environment = os.environ.copy()
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    environment.pop("GIT_REPLACE_REF_BASE", None)
    return environment


def _git(root: Path, *arguments: str, allow_failure: bool = False) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=_git_environment(),
    )
    if result.returncode != 0 and not allow_failure:
        message = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise ReleaseManifestError(
            [f"git -C {root} {' '.join(arguments)}: {message}"]
        )
    return result.stdout.strip()


def _git_metadata_path(root: Path, value: str) -> Path:
    selected = Path(value)
    if not selected.is_absolute():
        selected = root / selected
    return selected.resolve(strict=False)


def _git_history_override_errors(root: Path, *, owner: str) -> list[str]:
    """Reject local replace refs and legacy graft files that rewrite history."""

    errors: list[str] = []
    try:
        replace_refs = [
            line
            for line in _git(
                root,
                "for-each-ref",
                "--format=%(refname)",
                "refs/replace",
            ).splitlines()
            if line
        ]
        if replace_refs:
            errors.append(
                f"{owner} contains forbidden Git replace refs: {replace_refs}"
            )

        metadata_directories = {
            _git_metadata_path(root, _git(root, "rev-parse", "--git-dir")),
            _git_metadata_path(root, _git(root, "rev-parse", "--git-common-dir")),
        }
        graft_paths = {
            directory / "info" / "grafts" for directory in metadata_directories
        }
        graft_paths.add(
            _git_metadata_path(
                root, _git(root, "rev-parse", "--git-path", "info/grafts")
            )
        )
        present_grafts = sorted(
            str(path) for path in graft_paths if os.path.lexists(path)
        )
        if present_grafts:
            errors.append(
                f"{owner} contains forbidden legacy Git graft files: "
                f"{present_grafts}"
            )
    except ReleaseManifestError as exc:
        errors.extend(f"{owner}: {error}" for error in exc.errors)
    return errors


def _normalize_git_url(value: str) -> str:
    normalized = value.strip().removesuffix(".git")
    if normalized.startswith("git@") and ":" in normalized:
        host, path = normalized[4:].split(":", 1)
        normalized = f"{host}/{path}"
    else:
        normalized = re.sub(r"^(?:https?|ssh)://(?:git@)?", "", normalized)
    return normalized.rstrip("/").lower()


def _remote_matches(root: Path, remote: str, expected: str) -> list[str]:
    try:
        actual = _git(root, "remote", "get-url", remote)
    except ReleaseManifestError as exc:
        return exc.errors
    if _normalize_git_url(actual) != _normalize_git_url(expected):
        return [f"{remote} URL mismatch: declared {expected!r}, actual {actual!r}"]
    return []


def _metadata_path_allowed(path: str, allowed: Sequence[str]) -> bool:
    for entry in allowed:
        if entry.endswith("/") and path.startswith(entry):
            return True
        if path == entry:
            return True
    return False


def _verify_git_identity(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    avengine_root: Path,
    habitat_root: Path,
    verify_tag: bool,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    observed: dict[str, Any] = {}
    repositories = manifest["repositories"]
    avengine = repositories["avengine"]
    habitat = repositories["habitat_runtime"]
    release = manifest["release"]
    policy = release["metadata_commit_policy"]

    try:
        for repository_root, owner in (
            (avengine_root, "AVEngine repository"),
            (habitat_root, "Habitat runtime repository"),
        ):
            errors.extend(
                _git_history_override_errors(repository_root, owner=owner)
            )
        if errors:
            return errors, observed

        av_head = _git(avengine_root, "rev-parse", "HEAD")
        av_parent = _git(avengine_root, "rev-parse", "HEAD^")
        parent_line = _git(avengine_root, "rev-list", "--parents", "-n", "1", "HEAD")
        metadata_parents = parent_line.split()[1:]
        observed["avengine_metadata_commit"] = av_head
        observed["avengine_metadata_parent"] = av_parent
        observed["avengine_metadata_parent_count"] = len(metadata_parents)
        if len(metadata_parents) != 1:
            errors.append(
                "AVEngine metadata commit must have exactly one parent; "
                f"observed {len(metadata_parents)}"
            )
        if av_parent != avengine["implementation_commit"]:
            errors.append(
                "AVEngine metadata commit is not a direct child of the declared "
                "implementation commit"
            )
        _git(
            avengine_root,
            "cat-file",
            "-e",
            f"{avengine['implementation_commit']}^{{commit}}",
        )
        changed = [
            line
            for line in _git(
                avengine_root,
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                "HEAD",
            ).splitlines()
            if line
        ]
        observed["avengine_metadata_changed_paths"] = changed
        if release["manifest_path"] not in changed:
            errors.append("metadata commit does not introduce/update the release manifest")
        disallowed = [
            path
            for path in changed
            if not _metadata_path_allowed(path, policy["allowed_changed_paths"])
        ]
        if disallowed:
            errors.append(
                "metadata commit changes non-release paths: " + ", ".join(disallowed)
            )
        expected_manifest = (avengine_root / release["manifest_path"]).resolve()
        if manifest_path.resolve() != expected_manifest:
            errors.append(
                "manifest path does not match release.manifest_path: "
                f"{manifest_path.resolve()} != {expected_manifest}"
            )
        errors.extend(_remote_matches(avengine_root, "origin", avengine["repository"]))

        habitat_head = _git(habitat_root, "rev-parse", "HEAD")
        observed["habitat_runtime_commit"] = habitat_head
        if habitat_head != habitat["commit"]:
            errors.append(
                f"Habitat runtime HEAD mismatch: declared {habitat['commit']}, "
                f"actual {habitat_head}"
            )
        errors.extend(_remote_matches(habitat_root, "origin", habitat["repository"]))
        errors.extend(
            _remote_matches(habitat_root, "upstream", habitat["upstream_repository"])
        )
        _git(habitat_root, "cat-file", "-e", f"{habitat['upstream_commit']}^{{commit}}")
        ancestor = subprocess.run(
            [
                "git",
                "-C",
                str(habitat_root),
                "merge-base",
                "--is-ancestor",
                habitat["upstream_commit"],
                habitat["commit"],
            ],
            check=False,
            capture_output=True,
            text=True,
            env=_git_environment(),
        )
        if ancestor.returncode != 0:
            errors.append("declared upstream Habitat commit is not an ancestor of fork")

        submodule_path = Path(habitat["rlr_submodule_path"])
        rlr_root = (habitat_root / submodule_path).resolve(strict=True)
        rlr_root.relative_to(habitat_root.resolve(strict=True))
        rlr_override_errors = _git_history_override_errors(
            rlr_root, owner="RLR repository"
        )
        if rlr_override_errors:
            errors.extend(rlr_override_errors)
            return errors, observed
        rlr_head = _git(rlr_root, "rev-parse", "HEAD")
        observed["rlr_commit"] = rlr_head
        if rlr_head != habitat["rlr_commit"]:
            errors.append(
                f"RLR HEAD mismatch: declared {habitat['rlr_commit']}, actual {rlr_head}"
            )
        errors.extend(_remote_matches(rlr_root, "origin", habitat["rlr_repository"]))
        gitlink = _git(
            habitat_root,
            "ls-tree",
            habitat["commit"],
            "--",
            habitat["rlr_submodule_path"],
        )
        fields = gitlink.split()
        if len(fields) < 3 or fields[0] != "160000" or fields[2] != habitat["rlr_commit"]:
            errors.append("Habitat commit does not bind the declared RLR submodule commit")

        if policy["require_clean_worktrees"]:
            av_dirty = _git(avengine_root, "status", "--porcelain", "--untracked-files=all")
            habitat_dirty = _git(
                habitat_root, "status", "--porcelain", "--untracked-files=all"
            )
            if av_dirty:
                errors.append("AVEngine release worktree is not clean")
            if habitat_dirty:
                errors.append("Habitat runtime release worktree is not clean")

        if verify_tag:
            tag_ref = f"refs/tags/{release['tag']}"
            tag_type = _git(avengine_root, "cat-file", "-t", tag_ref)
            if policy["require_annotated_tag"] and tag_type != "tag":
                errors.append(f"release tag {release['tag']} is not annotated")
            tagged_commit = _git(avengine_root, "rev-parse", f"{tag_ref}^{{commit}}")
            observed["release_tag_commit"] = tagged_commit
            if tagged_commit != av_head:
                errors.append("release tag does not resolve to the metadata commit")
    except (OSError, ValueError, ReleaseManifestError) as exc:
        if isinstance(exc, ReleaseManifestError):
            errors.extend(exc.errors)
        else:
            errors.append(f"Git identity verification failed: {exc}")
    return errors, observed


def _verify_environment(environment: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    declared_os = environment["os"]
    actual_os = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }
    for field, actual in actual_os.items():
        if declared_os[field] != actual:
            errors.append(
                f"OS {field} mismatch: declared {declared_os[field]!r}, actual {actual!r}"
            )
    declared_python = environment["python"]
    actual_python = {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
    }
    for field, actual in actual_python.items():
        if declared_python[field] != actual:
            errors.append(
                f"Python {field} mismatch: declared {declared_python[field]!r}, "
                f"actual {actual!r}"
            )
    distributions: set[str] = set()
    for dependency in environment["python_dependencies"]:
        distribution = dependency["distribution"]
        key = distribution.lower().replace("_", "-")
        if key in distributions:
            errors.append(f"duplicate Python dependency {distribution!r}")
            continue
        distributions.add(key)
        try:
            actual = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            errors.append(f"Python dependency is not installed: {distribution}")
            continue
        if dependency["version"] != actual:
            errors.append(
                f"Python dependency {distribution} mismatch: declared "
                f"{dependency['version']!r}, actual {actual!r}"
            )
    compiler = environment["compiler"]
    executable = compiler["command"]
    resolved = executable if os.path.isabs(executable) else shutil.which(executable)
    if not resolved:
        errors.append(f"compiler command is unavailable: {executable}")
    else:
        result = subprocess.run(
            [str(resolved), "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        version_text = (result.stdout + "\n" + result.stderr).strip()
        if result.returncode != 0:
            errors.append(f"compiler version command failed: {executable} --version")
        elif compiler["version"] not in version_text:
            errors.append(
                f"compiler version {compiler['version']!r} is absent from readback"
            )
    return errors


def _build_request_error(owner: str, message: str) -> ReleaseManifestError:
    return ReleaseManifestError([f"{owner}: {message}"])


def _require_mapping(value: Any, *, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _build_request_error(owner, "must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    owner: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional_keys = optional or set()
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required - optional_keys)
    errors: list[str] = []
    if missing:
        errors.append(f"{owner}: missing keys {missing}")
    if extra:
        errors.append(f"{owner}: unknown keys {extra}")
    if errors:
        raise ReleaseManifestError(errors)


def _require_string(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise _build_request_error(owner, "must be a non-empty string")
    return value


def _require_stable_id(value: Any, *, owner: str) -> str:
    selected = _require_string(value, owner=owner)
    if _STABLE_ID.fullmatch(selected) is None:
        raise _build_request_error(owner, f"is not a stable ID: {selected!r}")
    return selected


def _require_commit(value: Any, *, owner: str) -> str:
    selected = _require_string(value, owner=owner)
    if _HEX_COMMIT.fullmatch(selected) is None:
        raise _build_request_error(owner, "must be a full lowercase Git commit")
    return selected


def _require_relative_path(value: Any, *, owner: str) -> str:
    selected = _require_string(value, owner=owner)
    raw = Path(selected)
    if raw.is_absolute() or ".." in raw.parts or "\x00" in selected:
        raise _build_request_error(
            owner, f"is not a confined relative path: {selected}"
        )
    normalized = raw.as_posix()
    if selected.endswith("/") and normalized != ".":
        normalized += "/"
    return normalized


def _resolve_repository_manifest_path(
    value: str | Path,
    *,
    repository_root: Path,
    owner: str,
    strict: bool,
) -> Path:
    """Resolve a manifest while rejecting lexical symlink indirection."""

    root = repository_root.resolve(strict=True)
    raw = Path(value)
    if ".." in raw.parts:
        raise _build_request_error(owner, "must not contain a parent traversal")
    candidate = raw if raw.is_absolute() else root / raw
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise _build_request_error(
            owner, "must remain inside the AVEngine repository"
        ) from exc
    if not relative.parts:
        raise _build_request_error(owner, "must name a file beneath the repository")

    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise _build_request_error(
                owner,
                f"must not be or traverse a symlink: {relative.as_posix()}",
            )
    try:
        resolved = candidate.resolve(strict=strict)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise _build_request_error(owner, f"cannot resolve manifest path: {exc}") from exc
    return resolved


def _require_clean_worktree(root: Path, *, owner: str) -> None:
    dirty = _git(root, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        lines = dirty.splitlines()
        preview = ", ".join(lines[:5])
        if len(lines) > 5:
            preview += f", ... ({len(lines)} entries)"
        raise _build_request_error(owner, f"worktree must be clean; found {preview}")


def _resolved_build_roots(
    *,
    avengine_root: Path,
    habitat_runtime_root: Path,
    artifact_roots: Mapping[str, str | Path] | None,
) -> dict[str, Path]:
    roots = {
        "avengine": avengine_root.resolve(strict=True),
        "habitat_runtime": habitat_runtime_root.resolve(strict=True),
    }
    for raw_id, raw_root in (artifact_roots or {}).items():
        root_id = _require_stable_id(raw_id, owner="artifact root ID")
        if root_id in roots:
            raise _build_request_error(
                "artifact_roots", f"cannot replace reserved root {root_id!r}"
            )
        root = Path(raw_root).resolve(strict=True)
        if not root.is_dir():
            raise _build_request_error(
                root_id, f"artifact root is not a directory: {root}"
            )
        roots[root_id] = root
    return roots


def _resolve_build_artifact(
    specification: Any,
    *,
    roots: Mapping[str, Path],
    owner: str,
    forbidden_path: Path,
) -> tuple[Path, dict[str, Any]]:
    spec = _require_mapping(specification, owner=owner)
    _require_exact_keys(spec, owner=owner, required={"root_id", "path"})
    root_id = _require_stable_id(spec["root_id"], owner=f"{owner}.root_id")
    if root_id not in roots:
        raise _build_request_error(owner, f"unknown root_id {root_id!r}")
    relative = _require_relative_path(spec["path"], owner=f"{owner}.path")
    base = roots[root_id]
    try:
        source = (base / relative).resolve(strict=True)
        source.relative_to(base)
    except (OSError, ValueError) as exc:
        raise _build_request_error(
            owner, f"cannot resolve confined artifact: {exc}"
        ) from exc
    if source == forbidden_path:
        raise _build_request_error(
            owner,
            "the release manifest cannot include itself as an artifact "
            "(self-reference)",
        )
    record = build_file_record(source, root=base, root_id=root_id)
    return source, record


def _compiler_environment(specification: Any) -> dict[str, str]:
    spec = _require_mapping(specification, owner="environment.compiler")
    _require_exact_keys(spec, owner="environment.compiler", required={"id", "command"})
    compiler_id = _require_string(spec["id"], owner="environment.compiler.id")
    command = _require_string(spec["command"], owner="environment.compiler.command")
    executable = command if os.path.isabs(command) else shutil.which(command)
    if not executable:
        raise _build_request_error(
            "environment.compiler.command", f"executable is unavailable: {command}"
        )
    result = subprocess.run(
        [str(executable), "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise _build_request_error(
            "environment.compiler.command",
            f"version readback failed: {command} --version",
        )
    lines = [
        line.strip()
        for line in (result.stdout + "\n" + result.stderr).splitlines()
        if line.strip()
    ]
    if not lines:
        raise _build_request_error(
            "environment.compiler.command", "version readback was empty"
        )
    return {"id": compiler_id, "command": command, "version": lines[0]}


def _python_dependencies(specification: Any) -> list[dict[str, str]]:
    if not isinstance(specification, list):
        raise _build_request_error(
            "environment.python_dependencies", "must be an array"
        )
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, value in enumerate(specification):
        owner = f"environment.python_dependencies[{index}]"
        distribution = _require_string(value, owner=owner)
        identity = distribution.lower().replace("_", "-")
        if identity in seen:
            raise _build_request_error(
                owner, f"duplicate distribution {distribution!r}"
            )
        seen.add(identity)
        try:
            version = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError as exc:
            raise _build_request_error(
                owner, f"distribution is not installed: {distribution}"
            ) from exc
        records.append({"distribution": distribution, "version": version})
    return records


def _test_execution_receipt_errors(
    document: Mapping[str, Any],
    *,
    layer_id: str,
    layer_status: str,
    command: Sequence[str],
    implementation_commit: str,
    habitat_runtime_commit: str,
    rlr_commit: str,
) -> list[str]:
    errors = verify_receipt_payload(document)
    if errors:
        return errors
    expected = {
        "test_layer_id": layer_id,
        "status": layer_status,
        "command": list(command),
        "implementation_commit": implementation_commit,
        "habitat_runtime_commit": habitat_runtime_commit,
        "rlr_commit": rlr_commit,
    }
    for field, expected_value in expected.items():
        if document[field] != expected_value:
            errors.append(
                f"test execution receipt {field} mismatch for {layer_id}: "
                f"expected {expected_value!r}, got {document[field]!r}"
            )

    return errors


def _load_and_validate_test_execution_receipt(
    path: Path,
    *,
    layer_id: str,
    layer_status: str,
    command: Sequence[str],
    implementation_commit: str,
    habitat_runtime_commit: str,
    rlr_commit: str,
) -> None:
    document = load_json_strict(path)
    errors = _test_execution_receipt_errors(
        document,
        layer_id=layer_id,
        layer_status=layer_status,
        command=command,
        implementation_commit=implementation_commit,
        habitat_runtime_commit=habitat_runtime_commit,
        rlr_commit=rlr_commit,
    )
    if errors:
        raise ReleaseManifestError(errors)


def _resolve_command_path(
    raw: str, *, base: Path, executable: bool = False
) -> Path:
    candidate = Path(raw)
    if executable and not candidate.is_absolute() and len(candidate.parts) == 1:
        located = shutil.which(raw)
        if located is None:
            raise OSError(f"executable is not on PATH: {raw}")
        candidate = Path(located)
    elif not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve(strict=True)


def _parse_release_verify_command(
    command: Sequence[str], *, avengine_root: Path
) -> tuple[dict[str, Any], list[str]]:
    """Parse the exact supported release verifier argv without argparse drift."""

    errors: list[str] = []
    tokens: list[str] = []
    for index, value in enumerate(command):
        if not isinstance(value, str) or not value:
            errors.append(
                f"planned release-canary command token {index} must be non-empty"
            )
        else:
            tokens.append(value)
    if errors:
        return {}, errors
    if len(tokens) < 3:
        return {}, [
            "planned release-canary command must be <python> "
            "tools/release/build_manifest.py verify ..."
        ]

    parsed: dict[str, Any] = {"artifact_roots": {}}
    try:
        executable_path = _resolve_command_path(
            tokens[0], base=avengine_root, executable=True
        )
        if _PYTHON_EXECUTABLE_NAME.fullmatch(executable_path.name) is None:
            errors.append(
                "planned release-canary command must use a Python executable"
            )
        parsed["executable"] = executable_path
    except OSError as exc:
        errors.append(f"planned release-canary Python cannot be resolved: {exc}")

    expected_script = (
        avengine_root / "tools" / "release" / "build_manifest.py"
    ).resolve(strict=False)
    try:
        script_path = _resolve_command_path(tokens[1], base=avengine_root)
        parsed["script"] = script_path
        if script_path != expected_script:
            errors.append(
                "planned release-canary command must invoke "
                "tools/release/build_manifest.py"
            )
    except OSError as exc:
        errors.append(f"planned release-canary verifier script cannot be resolved: {exc}")

    if tokens[2] != "verify":
        errors.append(
            "planned release-canary command must use verify as the subcommand"
        )

    required_flags = {
        "--manifest": "manifest",
        "--avengine-root": "avengine_root",
        "--habitat-runtime-root": "habitat_runtime_root",
        "--output": "output",
    }
    allowed_flags = set(required_flags) | {"--artifact-root"}
    index = 3
    seen: set[str] = set()
    while index < len(tokens):
        flag = tokens[index]
        if "=" in flag:
            errors.append(
                f"planned release-canary command requires split-form flags; got {flag!r}"
            )
            index += 1
            continue
        if flag not in allowed_flags:
            errors.append(
                f"planned release-canary command contains unsupported token {flag!r}"
            )
            index += 1
            continue
        if index + 1 >= len(tokens) or not tokens[index + 1]:
            errors.append(f"planned release-canary command lacks a value for {flag}")
            index += 1
            continue
        value = tokens[index + 1]
        if value.startswith("--"):
            errors.append(f"planned release-canary command lacks a value for {flag}")
            index += 1
            continue
        if flag == "--artifact-root":
            root_id, separator, raw_path = value.partition("=")
            declared = parsed["artifact_roots"]
            if not separator or not root_id or not raw_path:
                errors.append(
                    "planned release-canary artifact roots must use ROOT_ID=/path"
                )
            elif root_id in declared:
                errors.append(
                    f"planned release-canary command duplicates artifact root {root_id}"
                )
            else:
                declared[root_id] = raw_path
        else:
            if flag in seen:
                errors.append(
                    f"planned release-canary command requires exactly one {flag}"
                )
            else:
                parsed[required_flags[flag]] = value
                seen.add(flag)
        index += 2

    for flag, key in required_flags.items():
        if key not in parsed:
            errors.append(f"planned release-canary command requires exactly one {flag}")
    return parsed, errors


def _resolve_planned_argument(raw: str, *, avengine_root: Path, strict: bool) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = avengine_root / path
    return path.resolve(strict=strict)


def _planned_release_verify_command_errors(
    command: Sequence[str],
    *,
    avengine_root: Path,
    habitat_runtime_root: Path,
    manifest_relative: str,
    artifact_roots: Mapping[str, Path] | None = None,
) -> list[str]:
    parsed, errors = _parse_release_verify_command(
        command, avengine_root=avengine_root
    )
    if errors:
        return errors

    try:
        planned_manifest = _resolve_planned_argument(
            parsed["manifest"], avengine_root=avengine_root, strict=False
        )
        expected_manifest = (avengine_root / manifest_relative).resolve(strict=False)
        if planned_manifest != expected_manifest:
            errors.append(
                "planned release-canary --manifest differs from release.manifest_path"
            )
    except OSError:
        errors.append("planned release-canary --manifest cannot be resolved")
    try:
        planned_avengine = _resolve_planned_argument(
            parsed["avengine_root"], avengine_root=avengine_root, strict=True
        )
        if planned_avengine != avengine_root:
            errors.append("planned release-canary --avengine-root differs from commit A")
    except OSError:
        errors.append("planned release-canary --avengine-root cannot be resolved")
    try:
        planned_habitat = _resolve_planned_argument(
            parsed["habitat_runtime_root"],
            avengine_root=avengine_root,
            strict=True,
        )
        if planned_habitat != habitat_runtime_root:
            errors.append(
                "planned release-canary --habitat-runtime-root differs from pinned runtime"
            )
    except OSError:
        errors.append(
            "planned release-canary --habitat-runtime-root cannot be resolved"
        )
    planned_output = _resolve_planned_argument(
        parsed["output"], avengine_root=avengine_root, strict=False
    )
    if planned_output == (avengine_root / manifest_relative).resolve(strict=False):
        errors.append("planned release-canary --output cannot replace the manifest")

    declared_artifact_roots = parsed["artifact_roots"]
    expected_artifact_roots = dict(artifact_roots or {})
    missing_roots = sorted(set(expected_artifact_roots) - set(declared_artifact_roots))
    extra_roots = sorted(set(declared_artifact_roots) - set(expected_artifact_roots))
    if missing_roots:
        errors.append(
            f"planned release-canary command lacks artifact roots: {missing_roots}"
        )
    if extra_roots:
        errors.append(
            f"planned release-canary command declares unused artifact roots: {extra_roots}"
        )
    for root_id in sorted(set(expected_artifact_roots) & set(declared_artifact_roots)):
        try:
            planned_root = _resolve_planned_argument(
                declared_artifact_roots[root_id],
                avengine_root=avengine_root,
                strict=True,
            )
        except OSError:
            errors.append(
                f"planned release-canary artifact root {root_id} cannot be resolved"
            )
            continue
        if planned_root != expected_artifact_roots[root_id].resolve(strict=True):
            errors.append(
                f"planned release-canary artifact root {root_id} differs from preparation"
            )
    return errors


def _release_verify_command_identity(
    parsed: Mapping[str, Any], *, avengine_root: Path
) -> dict[str, Any]:
    return {
        "executable": str(parsed["executable"]),
        "script": str(parsed["script"]),
        "manifest": str(
            _resolve_planned_argument(
                parsed["manifest"], avengine_root=avengine_root, strict=False
            )
        ),
        "avengine_root": str(
            _resolve_planned_argument(
                parsed["avengine_root"], avengine_root=avengine_root, strict=True
            )
        ),
        "habitat_runtime_root": str(
            _resolve_planned_argument(
                parsed["habitat_runtime_root"],
                avengine_root=avengine_root,
                strict=True,
            )
        ),
        "output": str(
            _resolve_planned_argument(
                parsed["output"], avengine_root=avengine_root, strict=False
            )
        ),
        "artifact_roots": {
            root_id: str(
                _resolve_planned_argument(
                    raw_path, avengine_root=avengine_root, strict=True
                )
            )
            for root_id, raw_path in sorted(parsed["artifact_roots"].items())
        },
    }


def _actual_release_verify_command_errors(
    actual_command: Sequence[str],
    *,
    planned_command: Sequence[str],
    avengine_root: Path,
    habitat_runtime_root: Path,
    manifest_relative: str,
    output_path: Path,
    artifact_roots: Mapping[str, Path] | None = None,
) -> list[str]:
    errors = _planned_release_verify_command_errors(
        planned_command,
        avengine_root=avengine_root,
        habitat_runtime_root=habitat_runtime_root,
        manifest_relative=manifest_relative,
        artifact_roots=artifact_roots,
    )
    actual_errors = _planned_release_verify_command_errors(
        actual_command,
        avengine_root=avengine_root,
        habitat_runtime_root=habitat_runtime_root,
        manifest_relative=manifest_relative,
        artifact_roots=artifact_roots,
    )
    errors.extend(
        f"actual release verifier invocation: {error}" for error in actual_errors
    )
    if errors:
        return errors
    planned, planned_parse_errors = _parse_release_verify_command(
        planned_command, avengine_root=avengine_root
    )
    actual, actual_parse_errors = _parse_release_verify_command(
        actual_command, avengine_root=avengine_root
    )
    if planned_parse_errors or actual_parse_errors:
        return [*planned_parse_errors, *actual_parse_errors]
    planned_identity = _release_verify_command_identity(
        planned, avengine_root=avengine_root
    )
    actual_identity = _release_verify_command_identity(
        actual, avengine_root=avengine_root
    )
    if actual_identity != planned_identity:
        errors.append(
            "actual release verifier invocation differs from the command stored "
            "in test_layers.release-canary"
        )
    if Path(actual_identity["output"]) != output_path.resolve(strict=False):
        errors.append(
            "actual release verifier --output differs from the attestation output"
        )
    return errors


def _validate_controlled_canary_release_binding(
    document_path: Path,
    document: Mapping[str, Any],
    *,
    implementation_commit: str,
    release_id: str,
    release_tag: str,
    manifest_path: str,
) -> None:
    errors: list[str] = []
    if document.get("schema") != "avengine_m6_canary_evidence_v1":
        errors.append("controlled-canary role does not point to M6 canary evidence v1")
    if document.get("implementation_commit") != implementation_commit:
        errors.append("controlled-canary implementation_commit does not bind commit A")
    if document.get("overall_status") != "pass":
        errors.append("controlled-canary evidence is not pass")
    if document.get("research_only") is not True:
        errors.append("controlled-canary evidence must remain research_only=true")
    if document.get("qualification_claim") is not False:
        errors.append("controlled-canary evidence must keep qualification_claim=false")

    reference = document.get("release_manifest_ref")
    if not isinstance(reference, Mapping):
        errors.append("controlled-canary evidence lacks release_manifest_ref")
    else:
        relative = reference.get("path")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            errors.append("controlled-canary release_manifest_ref path is not confined")
        else:
            try:
                reference_path = (document_path.parent / relative).resolve(strict=True)
                reference_path.relative_to(document_path.parent.resolve(strict=True))
                if reference.get("byte_size") != reference_path.stat().st_size:
                    errors.append(
                        "controlled-canary release_manifest_ref byte size mismatch"
                    )
                if reference.get("sha256") != sha256_file(reference_path):
                    errors.append(
                        "controlled-canary release_manifest_ref SHA-256 mismatch"
                    )
                ref_document = load_json_strict(reference_path)
                expected = {
                    "schema": "avengine_release_manifest_ref_v1",
                    "release_id": release_id,
                    "expected_tag": release_tag,
                    "repository_path": manifest_path,
                    "implementation_commit": implementation_commit,
                }
                if ref_document != expected:
                    errors.append(
                        "controlled-canary release_manifest_ref document mismatch"
                    )
            except (OSError, ValueError, ReleaseManifestError) as exc:
                errors.append(
                    f"unable to verify controlled-canary release reference: {exc}"
                )
    if errors:
        raise ReleaseManifestError(errors)


def _validate_room_attempt_release_binding(
    document: Mapping[str, Any], *, implementation_commit: str
) -> None:
    errors: list[str] = []
    if document.get("schema") != "avengine_m6_room_qualification_attempt_v1":
        errors.append("room-qualification role does not point to M6 room attempt v1")
    provenance = document.get("code_provenance")
    if not isinstance(provenance, Mapping):
        errors.append("room-qualification attempt lacks code_provenance")
    else:
        if provenance.get("commit") != implementation_commit:
            errors.append("room-qualification attempt does not bind commit A")
        if provenance.get("worktree_clean") is not True:
            errors.append(
                "room-qualification attempt was not produced from a clean worktree"
            )
    reports = document.get("reports")
    if not isinstance(reports, list) or len(reports) < 5:
        errors.append(
            "room-qualification attempt does not retain the representative reports"
        )
    case_ids = (
        set(document.get("case_ids", []))
        if isinstance(document.get("case_ids"), list)
        else set()
    )
    required_cases = {
        "blender_custom_two_zone",
        "replicacad_apt_0",
        "legacy_ue_apartment",
        "mp3d_17DRP5sb8fy_raw",
        "mp3d_17DRP5sb8fy_derived",
        "independent_corrupted_fixture",
    }
    missing = sorted(required_cases - case_ids)
    if missing:
        errors.append(f"room-qualification attempt lacks required cases: {missing}")
    if errors:
        raise ReleaseManifestError(errors)


def _expand_declared_evidence_closure(
    entry_path: Path,
    document: Mapping[str, Any],
    *,
    root_id: str,
    artifact_root: Path,
    owner: str,
) -> list[dict[str, Any]]:
    """Authenticate and record an entry JSON plus its exact declared closure."""

    raw_records = document.get("artifacts")
    if isinstance(raw_records, Mapping):
        declared_items = list(raw_records.items())
    elif isinstance(raw_records, list):
        declared_items = [
            (record.get("path") if isinstance(record, Mapping) else None, record)
            for record in raw_records
        ]
    else:
        raise _build_request_error(owner, "authority document lacks artifacts closure")

    bundle_root = entry_path.parent.resolve(strict=True)
    records = [build_file_record(entry_path, root=artifact_root, root_id=root_id)]
    declared_paths: set[str] = set()
    errors: list[str] = []
    for index, (declared_key, raw_record) in enumerate(declared_items):
        record_owner = f"{owner}.artifacts[{index}]"
        if not isinstance(raw_record, Mapping):
            errors.append(f"{record_owner}: record is not an object")
            continue
        relative = raw_record.get("path")
        if not isinstance(relative, str) or not relative:
            errors.append(f"{record_owner}: path is not a non-empty string")
            continue
        if declared_key != relative:
            errors.append(
                f"{record_owner}: artifact key/path mismatch "
                f"({declared_key!r} != {relative!r})"
            )
        raw_relative = Path(relative)
        if (
            raw_relative.is_absolute()
            or ".." in raw_relative.parts
            or "\\" in relative
        ):
            errors.append(f"{record_owner}: path is not confined: {relative}")
            continue
        normalized = raw_relative.as_posix()
        if normalized in declared_paths:
            errors.append(f"{record_owner}: duplicate path {normalized}")
            continue
        declared_paths.add(normalized)
        try:
            unresolved_candidate = bundle_root / raw_relative
            cursor = bundle_root
            traverses_symlink = False
            for part in raw_relative.parts:
                cursor = cursor / part
                if cursor.is_symlink():
                    traverses_symlink = True
                    break
            if traverses_symlink:
                errors.append(
                    f"{record_owner}: artifact must not traverse a symlink"
                )
                continue
            candidate = unresolved_candidate.resolve(strict=True)
            candidate.relative_to(bundle_root)
        except (OSError, ValueError) as exc:
            errors.append(f"{record_owner}: unable to resolve artifact: {exc}")
            continue
        if not candidate.is_file():
            errors.append(f"{record_owner}: artifact is not a regular file")
            continue
        actual_size = candidate.stat().st_size
        actual_hash = sha256_file(candidate)
        if raw_record.get("byte_size") != actual_size:
            errors.append(f"{record_owner}: declared byte size differs")
        if raw_record.get("sha256") != actual_hash:
            errors.append(f"{record_owner}: declared SHA-256 differs")
        try:
            records.append(
                build_file_record(candidate, root=artifact_root, root_id=root_id)
            )
        except ReleaseManifestError as exc:
            errors.extend(f"{record_owner}: {error}" for error in exc.errors)

    retained_entries = list(bundle_root.rglob("*"))
    symlinks = sorted(
        path.relative_to(bundle_root).as_posix()
        for path in retained_entries
        if path.is_symlink()
    )
    if symlinks:
        errors.append(f"{owner}: retained closure contains symlinks: {symlinks}")
    retained_paths = [
        path
        for path in retained_entries
        if not path.is_symlink() and path.is_file()
    ]
    actual_relative_files = {
        path.relative_to(bundle_root).as_posix()
        for path in retained_paths
        if path.resolve() != entry_path.resolve()
    }
    missing = sorted(declared_paths - actual_relative_files)
    extra = sorted(actual_relative_files - declared_paths)
    if missing or extra:
        errors.append(
            f"{owner}: declared artifact closure differs from retained files; "
            f"missing={missing}, extra={extra}"
        )
    identities = [(record["root_id"], record["path"]) for record in records]
    if len(identities) != len(set(identities)):
        errors.append(f"{owner}: expanded closure contains duplicate file records")
    if errors:
        raise ReleaseManifestError(errors)
    records.sort(key=lambda record: (record["root_id"], record["path"]))
    return records


def _require_current_m6_evidence_verifiers(
    *, controlled_canary: Path, room_attempt: Path
) -> None:
    """Run the authoritative M6 semantic verifiers, not just JSON field checks."""

    # Lazy imports keep the general release-manifest reader lightweight and
    # avoid making non-M6 callers import media/room implementation modules.
    from avengine.m6.canary import verify_controlled_canary_evidence
    from avengine.m6.room_attempts import verify_room_qualification_attempt

    try:
        controlled_status, controlled_checks = verify_controlled_canary_evidence(
            controlled_canary
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise _build_request_error(
            "m6_evidence.controlled_canary_bundle_id",
            f"authoritative verifier could not run: {exc}",
        ) from exc
    if controlled_status != "pass":
        failed = [
            check.get("check_id", "unknown")
            for check in controlled_checks
            if check.get("status") != "pass"
        ]
        raise _build_request_error(
            "m6_evidence.controlled_canary_bundle_id",
            f"authoritative verifier failed checks: {failed}",
        )
    try:
        room_status, room_checks = verify_room_qualification_attempt(room_attempt)
    except (OSError, ValueError, RuntimeError) as exc:
        raise _build_request_error(
            "m6_evidence.room_qualification_bundle_id",
            f"authoritative verifier could not run: {exc}",
        ) from exc
    if room_status != "pass":
        failed = [
            check.get("check_id", "unknown")
            for check in room_checks
            if check.get("status") != "pass"
        ]
        raise _build_request_error(
            "m6_evidence.room_qualification_bundle_id",
            f"authoritative verifier failed checks: {failed}",
        )


def build_release_manifest(
    request: Mapping[str, Any],
    *,
    avengine_root: str | Path,
    habitat_runtime_root: str | Path,
    artifact_roots: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Build a release manifest while the AVEngine worktree is exactly commit A.

    This function never writes or commits.  The returned document intentionally
    contains commit A but neither the future metadata commit B nor its own byte
    hash.  After :func:`prepare_release_manifest` writes it, B can therefore be
    a direct child of A without an impossible Git/hash fixed point.
    """

    root = Path(avengine_root).resolve(strict=True)
    habitat_root = Path(habitat_runtime_root).resolve(strict=True)
    roots = _resolved_build_roots(
        avengine_root=root,
        habitat_runtime_root=habitat_root,
        artifact_roots=artifact_roots,
    )
    history_override_errors = [
        error
        for repository_root, owner in (
            (root, "AVEngine repository"),
            (habitat_root, "Habitat runtime repository"),
        )
        for error in _git_history_override_errors(repository_root, owner=owner)
    ]
    if history_override_errors:
        raise ReleaseManifestError(history_override_errors)
    _require_exact_keys(
        request,
        owner="release build request",
        required={
            "schema",
            "release",
            "repositories",
            "native_artifacts",
            "environment",
            "evidence_bundles",
            "m6_evidence",
            "test_layers",
        },
    )
    if request.get("schema") != RELEASE_BUILD_REQUEST_SCHEMA:
        raise _build_request_error(
            "schema", f"must equal {RELEASE_BUILD_REQUEST_SCHEMA!r}"
        )

    release_request = _require_mapping(request["release"], owner="release")
    _require_exact_keys(
        release_request,
        owner="release",
        required={
            "release_id",
            "tag",
            "state",
            "current_milestone",
            "manifest_path",
        },
        optional={"allowed_changed_paths"},
    )
    release_id = _require_stable_id(
        release_request["release_id"], owner="release.release_id"
    )
    release_tag = _require_string(release_request["tag"], owner="release.tag")
    tag_ref = f"refs/tags/{release_tag}"
    tag_format = subprocess.run(
        ["git", "check-ref-format", tag_ref],
        check=False,
        capture_output=True,
        text=True,
        env=_git_environment(),
    )
    if tag_format.returncode != 0:
        raise _build_request_error("release.tag", "is not a valid Git tag name")
    existing_tag = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "show-ref",
            "--verify",
            "--quiet",
            tag_ref,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_git_environment(),
    )
    if existing_tag.returncode == 0:
        raise _build_request_error(
            "release.tag", "already exists; refusing to move or replace a release tag"
        )
    if existing_tag.returncode not in {0, 1}:
        raise _build_request_error("release.tag", "could not check existing Git tag")
    state = _require_string(release_request["state"], owner="release.state")
    if state != "candidate":
        raise _build_request_error(
            "release.state",
            "must be candidate; post-tag release status belongs to the external "
            "release attestation",
        )
    milestone = _require_string(
        release_request["current_milestone"], owner="release.current_milestone"
    )
    if milestone != "M6":
        raise _build_request_error(
            "release.current_milestone", "must equal 'M6' for this release schema"
        )
    manifest_relative = _require_relative_path(
        release_request["manifest_path"], owner="release.manifest_path"
    )
    if not manifest_relative.startswith("release/"):
        raise _build_request_error(
            "release.manifest_path", "must be beneath the release/ directory"
        )
    manifest_absolute = (root / manifest_relative).resolve(strict=False)
    try:
        manifest_absolute.relative_to(root)
    except ValueError as exc:
        raise _build_request_error(
            "release.manifest_path", "resolves outside the AVEngine repository"
        ) from exc
    allowed_raw = release_request.get("allowed_changed_paths", [manifest_relative])
    if not isinstance(allowed_raw, list) or not allowed_raw:
        raise _build_request_error(
            "release.allowed_changed_paths", "must be a non-empty array"
        )
    allowed_changed_paths: list[str] = []
    for index, value in enumerate(allowed_raw):
        selected = _require_relative_path(
            value, owner=f"release.allowed_changed_paths[{index}]"
        )
        if not selected.startswith("release/"):
            raise _build_request_error(
                f"release.allowed_changed_paths[{index}]",
                "metadata commit paths must remain beneath release/",
            )
        if selected not in allowed_changed_paths:
            allowed_changed_paths.append(selected)
    if not _metadata_path_allowed(manifest_relative, allowed_changed_paths):
        raise _build_request_error(
            "release.allowed_changed_paths", "does not allow the manifest path"
        )

    repositories_request = _require_mapping(
        request["repositories"], owner="repositories"
    )
    _require_exact_keys(
        repositories_request,
        owner="repositories",
        required={
            "implementation_commit",
            "expected_habitat_commit",
            "upstream_commit",
            "expected_rlr_commit",
            "rlr_submodule_path",
            "expected_avengine_repository",
            "expected_habitat_repository",
            "expected_upstream_repository",
            "expected_rlr_repository",
        },
    )
    implementation_commit = _require_commit(
        repositories_request["implementation_commit"],
        owner="repositories.implementation_commit",
    )
    habitat_commit = _require_commit(
        repositories_request["expected_habitat_commit"],
        owner="repositories.expected_habitat_commit",
    )
    upstream_commit = _require_commit(
        repositories_request["upstream_commit"], owner="repositories.upstream_commit"
    )
    rlr_commit = _require_commit(
        repositories_request["expected_rlr_commit"],
        owner="repositories.expected_rlr_commit",
    )
    rlr_submodule_path = _require_relative_path(
        repositories_request["rlr_submodule_path"],
        owner="repositories.rlr_submodule_path",
    )
    expected_repositories = {
        "avengine": _require_string(
            repositories_request["expected_avengine_repository"],
            owner="repositories.expected_avengine_repository",
        ),
        "habitat": _require_string(
            repositories_request["expected_habitat_repository"],
            owner="repositories.expected_habitat_repository",
        ),
        "upstream": _require_string(
            repositories_request["expected_upstream_repository"],
            owner="repositories.expected_upstream_repository",
        ),
        "rlr": _require_string(
            repositories_request["expected_rlr_repository"],
            owner="repositories.expected_rlr_repository",
        ),
    }

    avengine_head = _git(root, "rev-parse", "HEAD")
    if avengine_head != implementation_commit:
        raise _build_request_error(
            "repositories.implementation_commit",
            f"must equal current AVEngine HEAD {avengine_head}",
        )
    _require_clean_worktree(root, owner="AVEngine commit A")
    observed_habitat_commit = _git(habitat_root, "rev-parse", "HEAD")
    if observed_habitat_commit != habitat_commit:
        raise _build_request_error(
            "repositories.expected_habitat_commit",
            f"does not equal Habitat HEAD {observed_habitat_commit}",
        )
    _require_clean_worktree(habitat_root, owner="Habitat runtime")
    _git(habitat_root, "cat-file", "-e", f"{upstream_commit}^{{commit}}")
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(habitat_root),
            "merge-base",
            "--is-ancestor",
            upstream_commit,
            habitat_commit,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_git_environment(),
    )
    if ancestor.returncode != 0:
        raise _build_request_error(
            "repositories.upstream_commit",
            "is not an ancestor of the Habitat fork commit",
        )
    try:
        rlr_root = (habitat_root / rlr_submodule_path).resolve(strict=True)
        rlr_root.relative_to(habitat_root)
    except (OSError, ValueError) as exc:
        raise _build_request_error(
            "repositories.rlr_submodule_path", f"cannot resolve submodule: {exc}"
        ) from exc
    rlr_override_errors = _git_history_override_errors(
        rlr_root, owner="RLR repository"
    )
    if rlr_override_errors:
        raise ReleaseManifestError(rlr_override_errors)
    observed_rlr_commit = _git(rlr_root, "rev-parse", "HEAD")
    if observed_rlr_commit != rlr_commit:
        raise _build_request_error(
            "repositories.expected_rlr_commit",
            f"does not equal checked-out RLR commit {observed_rlr_commit}",
        )
    gitlink = _git(habitat_root, "ls-tree", habitat_commit, "--", rlr_submodule_path)
    fields = gitlink.split()
    if len(fields) < 3 or fields[0] != "160000" or fields[2] != rlr_commit:
        raise _build_request_error(
            "repositories.expected_rlr_commit", "is not bound by the Habitat gitlink"
        )
    remote_errors = [
        *_remote_matches(root, "origin", expected_repositories["avengine"]),
        *_remote_matches(
            habitat_root, "origin", expected_repositories["habitat"]
        ),
        *_remote_matches(
            habitat_root, "upstream", expected_repositories["upstream"]
        ),
        *_remote_matches(rlr_root, "origin", expected_repositories["rlr"]),
    ]
    if remote_errors:
        raise ReleaseManifestError(remote_errors)

    schema_directory = root / "schemas"
    if not schema_directory.is_dir():
        raise _build_request_error(
            "schemas", f"directory is missing: {schema_directory}"
        )
    schema_records = [
        build_file_record(path, root=root, root_id="avengine")
        for path in sorted(schema_directory.rglob("*.json"))
        if path.is_file()
    ]
    if not schema_records:
        raise _build_request_error("schemas", "schema inventory is empty")

    native_request = _require_mapping(
        request["native_artifacts"], owner="native_artifacts"
    )
    _require_exact_keys(
        native_request,
        owner="native_artifacts",
        required={"habitat_sim_binding", "rlr_binary"},
    )
    native_artifacts: dict[str, dict[str, Any]] = {}
    for artifact_id in ("habitat_sim_binding", "rlr_binary"):
        source, record = _resolve_build_artifact(
            native_request[artifact_id],
            roots=roots,
            owner=f"native_artifacts.{artifact_id}",
            forbidden_path=manifest_absolute,
        )
        if record["root_id"] != "habitat_runtime":
            raise _build_request_error(
                f"native_artifacts.{artifact_id}", "must use root_id habitat_runtime"
            )
        if source.stat().st_size == 0:
            raise _build_request_error(f"native_artifacts.{artifact_id}", "is empty")
        native_artifacts[artifact_id] = record

    bundles_request = request["evidence_bundles"]
    if not isinstance(bundles_request, list) or not bundles_request:
        raise _build_request_error("evidence_bundles", "must be a non-empty array")
    bundle_documents: dict[str, list[tuple[Path, Mapping[str, Any]]]] = {}
    bundle_sources: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    evidence_bundles: list[dict[str, Any]] = []
    bundle_ids: set[str] = set()
    for bundle_index, raw_bundle in enumerate(bundles_request):
        owner = f"evidence_bundles[{bundle_index}]"
        bundle = _require_mapping(raw_bundle, owner=owner)
        _require_exact_keys(
            bundle,
            owner=owner,
            required={"evidence_id", "status_scope", "status", "artifacts"},
        )
        evidence_id = _require_stable_id(
            bundle["evidence_id"], owner=f"{owner}.evidence_id"
        )
        if evidence_id in bundle_ids:
            raise _build_request_error(f"{owner}.evidence_id", "is duplicated")
        bundle_ids.add(evidence_id)
        status_scope = _require_string(
            bundle["status_scope"], owner=f"{owner}.status_scope"
        )
        if status_scope not in _EVIDENCE_STATUS_SCOPES:
            raise _build_request_error(
                f"{owner}.status_scope", "is not a supported evidence status scope"
            )
        status = _require_string(bundle["status"], owner=f"{owner}.status")
        if status not in _VERIFICATION_STATUSES:
            raise _build_request_error(
                f"{owner}.status", "is not a verification status"
            )
        artifact_specs = bundle["artifacts"]
        if not isinstance(artifact_specs, list) or not artifact_specs:
            raise _build_request_error(
                f"{owner}.artifacts", "must be a non-empty array"
            )
        records: list[dict[str, Any]] = []
        sources: list[tuple[Path, dict[str, Any]]] = []
        documents: list[tuple[Path, Mapping[str, Any]]] = []
        for artifact_index, artifact_spec in enumerate(artifact_specs):
            artifact_owner = f"{owner}.artifacts[{artifact_index}]"
            source, record = _resolve_build_artifact(
                artifact_spec,
                roots=roots,
                owner=artifact_owner,
                forbidden_path=manifest_absolute,
            )
            records.append(record)
            sources.append((source, record))
            if source.suffix.lower() == ".json":
                documents.append((source, load_json_strict(source)))
        evidence_bundles.append(
            {
                "evidence_id": evidence_id,
                "status_scope": status_scope,
                "status": status,
                "artifacts": records,
                "bundle_sha256": canonical_file_record_set_sha256(records),
            }
        )
        bundle_documents[evidence_id] = documents
        bundle_sources[evidence_id] = sources

    role_request = _require_mapping(request["m6_evidence"], owner="m6_evidence")
    _require_exact_keys(
        role_request,
        owner="m6_evidence",
        required={"controlled_canary_bundle_id", "room_qualification_bundle_id"},
    )
    controlled_id = _require_stable_id(
        role_request["controlled_canary_bundle_id"],
        owner="m6_evidence.controlled_canary_bundle_id",
    )
    room_id = _require_stable_id(
        role_request["room_qualification_bundle_id"],
        owner="m6_evidence.room_qualification_bundle_id",
    )
    if controlled_id == room_id:
        raise _build_request_error(
            "m6_evidence", "controlled and room roles must be distinct"
        )
    if controlled_id not in bundle_documents or room_id not in bundle_documents:
        raise _build_request_error(
            "m6_evidence", "role references an unknown evidence bundle"
        )
    controlled_matches = [
        item
        for item in bundle_documents[controlled_id]
        if item[1].get("schema") == "avengine_m6_canary_evidence_v1"
    ]
    room_matches = [
        item
        for item in bundle_documents[room_id]
        if item[1].get("schema") == "avengine_m6_room_qualification_attempt_v1"
    ]
    if len(controlled_matches) != 1:
        raise _build_request_error(
            "m6_evidence.controlled_canary_bundle_id",
            "bundle must contain exactly one M6 canary evidence document",
        )
    if len(room_matches) != 1:
        raise _build_request_error(
            "m6_evidence.room_qualification_bundle_id",
            "bundle must contain exactly one M6 room-attempt manifest",
        )
    if (
        len(bundle_sources[controlled_id]) != 1
        or bundle_sources[controlled_id][0][0] != controlled_matches[0][0]
    ):
        raise _build_request_error(
            "m6_evidence.controlled_canary_bundle_id",
            "request bundle must name only evidence.json; its closure is "
            "expanded automatically",
        )
    if (
        len(bundle_sources[room_id]) != 1
        or bundle_sources[room_id][0][0] != room_matches[0][0]
    ):
        raise _build_request_error(
            "m6_evidence.room_qualification_bundle_id",
            "request bundle must name only attempt_manifest.json; its closure "
            "is expanded automatically",
        )
    bundle_status_by_id = {
        bundle["evidence_id"]: bundle["status"] for bundle in evidence_bundles
    }
    bundle_scope_by_id = {
        bundle["evidence_id"]: bundle["status_scope"] for bundle in evidence_bundles
    }
    if bundle_scope_by_id[controlled_id] != "controlled_canary_verifier":
        raise _build_request_error(
            "m6_evidence.controlled_canary_bundle_id",
            "bundle status_scope must be controlled_canary_verifier",
        )
    if bundle_scope_by_id[room_id] != "room_attempt_verifier":
        raise _build_request_error(
            "m6_evidence.room_qualification_bundle_id",
            "bundle status_scope must be room_attempt_verifier; pass describes "
            "attempt verification, not room qualification",
        )
    if bundle_status_by_id[controlled_id] != "pass":
        raise _build_request_error(
            "m6_evidence", "controlled-canary bundle must be pass"
        )
    if bundle_status_by_id[room_id] != "pass":
        raise _build_request_error(
            "m6_evidence",
            "room-attempt bundle status describes verifier execution and must be pass; "
            "individual room admission may remain false inside the attempt",
        )
    _validate_controlled_canary_release_binding(
        controlled_matches[0][0],
        controlled_matches[0][1],
        implementation_commit=implementation_commit,
        release_id=release_id,
        release_tag=release_tag,
        manifest_path=manifest_relative,
    )
    _validate_room_attempt_release_binding(
        room_matches[0][1], implementation_commit=implementation_commit
    )
    controlled_root_id = bundle_sources[controlled_id][0][1]["root_id"]
    room_root_id = bundle_sources[room_id][0][1]["root_id"]
    expanded_by_id = {
        controlled_id: _expand_declared_evidence_closure(
            controlled_matches[0][0],
            controlled_matches[0][1],
            root_id=controlled_root_id,
            artifact_root=roots[controlled_root_id],
            owner="m6 controlled-canary evidence",
        ),
        room_id: _expand_declared_evidence_closure(
            room_matches[0][0],
            room_matches[0][1],
            root_id=room_root_id,
            artifact_root=roots[room_root_id],
            owner="m6 room-qualification evidence",
        ),
    }
    for bundle in evidence_bundles:
        expanded = expanded_by_id.get(bundle["evidence_id"])
        if expanded is not None:
            bundle["artifacts"] = expanded
            bundle["bundle_sha256"] = canonical_file_record_set_sha256(expanded)
    m6_evidence = {
        "controlled_canary_bundle_id": controlled_id,
        "controlled_canary_entry": build_file_record(
            controlled_matches[0][0],
            root=roots[controlled_root_id],
            root_id=controlled_root_id,
        ),
        "room_qualification_bundle_id": room_id,
        "room_qualification_entry": build_file_record(
            room_matches[0][0],
            root=roots[room_root_id],
            root_id=room_root_id,
        ),
    }
    _require_current_m6_evidence_verifiers(
        controlled_canary=controlled_matches[0][0],
        room_attempt=room_matches[0][0],
    )

    layers_request = _require_mapping(request["test_layers"], owner="test_layers")
    if set(layers_request) != set(TEST_LAYER_IDS):
        missing = sorted(set(TEST_LAYER_IDS) - set(layers_request))
        extra = sorted(set(layers_request) - set(TEST_LAYER_IDS))
        raise ReleaseManifestError(
            [f"test_layers inventory mismatch; missing={missing}, extra={extra}"]
        )
    test_layers: dict[str, dict[str, Any]] = {}
    receipt_sources_by_layer: dict[str, list[Path]] = {}
    for layer_id in TEST_LAYER_IDS:
        owner = f"test_layers.{layer_id}"
        raw_layer = _require_mapping(layers_request[layer_id], owner=owner)
        _require_exact_keys(
            raw_layer,
            owner=owner,
            required={
                "status",
                "command",
                "evidence_bundle_ids",
                "receipt_artifacts",
            },
            optional={"summary", "reason"},
        )
        layer_status = _require_string(
            raw_layer["status"], owner=f"{owner}.status"
        )
        if layer_status not in _VERIFICATION_STATUSES:
            raise _build_request_error(
                f"{owner}.status", "is not a verification status"
            )
        if layer_id == "release-canary" and layer_status != "not_run":
            raise _build_request_error(
                f"{owner}.status",
                "candidate preparation must leave the post-tag final attestation not_run",
            )
        raw_command = raw_layer["command"]
        if not isinstance(raw_command, list):
            raise _build_request_error(f"{owner}.command", "must be an array")
        command = [
            _require_string(value, owner=f"{owner}.command[{index}]")
            for index, value in enumerate(raw_command)
        ]
        raw_references = raw_layer["evidence_bundle_ids"]
        if not isinstance(raw_references, list):
            raise _build_request_error(
                f"{owner}.evidence_bundle_ids", "must be an array"
            )
        references = [
            _require_stable_id(
                value, owner=f"{owner}.evidence_bundle_ids[{index}]"
            )
            for index, value in enumerate(raw_references)
        ]
        if len(references) != len(set(references)):
            raise _build_request_error(
                f"{owner}.evidence_bundle_ids", "contains duplicate bundle IDs"
            )
        raw_receipts = raw_layer["receipt_artifacts"]
        if not isinstance(raw_receipts, list):
            raise _build_request_error(
                f"{owner}.receipt_artifacts", "must be an array"
            )
        expected_receipt_count = 1 if layer_status in {"pass", "fail"} else 0
        if len(raw_receipts) != expected_receipt_count:
            raise _build_request_error(
                f"{owner}.receipt_artifacts",
                f"{layer_status} requires exactly {expected_receipt_count} receipt(s)",
            )
        receipts: list[dict[str, Any]] = []
        receipt_sources: list[Path] = []
        for index, receipt_spec in enumerate(raw_receipts):
            receipt_path, receipt = _resolve_build_artifact(
                receipt_spec,
                roots=roots,
                owner=f"{owner}.receipt_artifacts[{index}]",
                forbidden_path=manifest_absolute,
            )
            receipt_sources.append(receipt_path)
            receipts.append(receipt)
        if len(
            {(item["root_id"], item["path"]) for item in receipts}
        ) != len(receipts):
            raise _build_request_error(
                f"{owner}.receipt_artifacts", "contains duplicate artifacts"
            )
        layer: dict[str, Any] = {
            "status": layer_status,
            "command": command,
            "evidence_bundle_ids": references,
            "receipt_artifacts": receipts,
        }
        for optional in ("summary", "reason"):
            if optional in raw_layer:
                layer[optional] = _require_string(
                    raw_layer[optional], owner=f"{owner}.{optional}"
                )
        test_layers[layer_id] = layer
        receipt_sources_by_layer[layer_id] = receipt_sources

    fast_unit = _require_mapping(
        test_layers["fast-unit"], owner="test_layers.fast-unit"
    )
    if fast_unit.get("status") != "pass":
        raise _build_request_error(
            "test_layers.fast-unit.status",
            "must be pass for the M6 release profile",
        )
    fast_unit_command_errors = _m6_fast_unit_command_errors(
        fast_unit.get("command")
    )
    if fast_unit_command_errors:
        raise ReleaseManifestError(fast_unit_command_errors)

    release_canary = _require_mapping(
        test_layers["release-canary"], owner="test_layers.release-canary"
    )
    release_canary_refs = release_canary.get("evidence_bundle_ids")
    if release_canary.get("status") != "not_run":
        raise _build_request_error(
            "test_layers.release-canary.status",
            "candidate preparation must leave the post-tag final attestation not_run",
        )
    if not isinstance(release_canary_refs, list) or not {
        controlled_id,
        room_id,
    }.issubset(set(release_canary_refs)):
        raise _build_request_error(
            "test_layers.release-canary.evidence_bundle_ids",
            "must reference both formal controlled-canary and room-attempt bundles",
        )
    if not release_canary.get("command"):
        raise _build_request_error(
            "test_layers.release-canary.command",
            "must retain the planned post-tag final verification command",
        )
    planned_command_errors = _planned_release_verify_command_errors(
        release_canary["command"],
        avengine_root=root,
        habitat_runtime_root=habitat_root,
        manifest_relative=manifest_relative,
        artifact_roots={
            root_id: roots[root_id]
            for root_id in {
                record["root_id"]
                for bundle in evidence_bundles
                for record in bundle["artifacts"]
            }
            if root_id not in {"avengine", "habitat_runtime"}
        },
    )
    if planned_command_errors:
        raise ReleaseManifestError(planned_command_errors)
    if release_canary.get("receipt_artifacts"):
        raise _build_request_error(
            "test_layers.release-canary.receipt_artifacts",
            "candidate preparation cannot contain a post-tag final attestation receipt",
        )

    environment_request = _require_mapping(request["environment"], owner="environment")
    _require_exact_keys(
        environment_request,
        owner="environment",
        required={"compiler", "python_dependencies"},
    )
    environment = {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "compiler": _compiler_environment(environment_request["compiler"]),
        "python_dependencies": _python_dependencies(
            environment_request["python_dependencies"]
        ),
    }

    manifest: dict[str, Any] = {
        "schema": RELEASE_MANIFEST_SCHEMA,
        "release": {
            "release_id": release_id,
            "tag": release_tag,
            "state": state,
            "current_milestone": milestone,
            "manifest_path": manifest_relative,
            "metadata_commit_policy": {
                "mode": "direct_child_of_implementation",
                "allowed_changed_paths": allowed_changed_paths,
                "require_clean_worktrees": True,
                "require_annotated_tag": True,
            },
        },
        "repositories": {
            "avengine": {
                "repository": expected_repositories["avengine"],
                "implementation_commit": implementation_commit,
            },
            "habitat_runtime": {
                "repository": expected_repositories["habitat"],
                "commit": habitat_commit,
                "upstream_repository": expected_repositories["upstream"],
                "upstream_commit": upstream_commit,
                "rlr_repository": expected_repositories["rlr"],
                "rlr_commit": rlr_commit,
                "rlr_submodule_path": rlr_submodule_path,
            },
        },
        "schemas": {
            "directory": "schemas",
            "algorithm": SCHEMA_SET_ALGORITHM,
            "files": schema_records,
            "set_sha256": canonical_file_record_set_sha256(schema_records),
        },
        "native_artifacts": native_artifacts,
        "environment": environment,
        "evidence_bundles": evidence_bundles,
        "m6_evidence": m6_evidence,
        "test_layers": test_layers,
    }
    schema_errors = validate_release_manifest_document(manifest)
    if schema_errors:
        raise ReleaseManifestError(schema_errors)

    bundle_by_id = {bundle["evidence_id"]: bundle for bundle in evidence_bundles}
    layer_errors: list[str] = []
    for layer_id in TEST_LAYER_IDS:
        layer = test_layers[layer_id]
        references = layer.get("evidence_bundle_ids", [])
        missing = [item for item in references if item not in bundle_by_id]
        if missing:
            layer_errors.append(
                f"{layer_id} references unknown evidence bundles: {missing}"
            )
            continue
        referenced_artifacts = [
            artifact
            for bundle_id in references
            for artifact in bundle_by_id[bundle_id]["artifacts"]
        ]
        test_execution_artifacts = [
            artifact
            for bundle_id in references
            if bundle_by_id[bundle_id]["status_scope"] == "test_execution"
            for artifact in bundle_by_id[bundle_id]["artifacts"]
        ]
        for receipt_index, receipt in enumerate(layer.get("receipt_artifacts", [])):
            if receipt not in referenced_artifacts:
                layer_errors.append(
                    f"{layer_id} receipt artifact is not a member of a referenced "
                    f"evidence bundle: {receipt['root_id']}:{receipt['path']}"
                )
            elif receipt not in test_execution_artifacts:
                layer_errors.append(
                    f"{layer_id} receipt artifact is not a member of a referenced "
                    "test_execution evidence bundle"
                )
            else:
                try:
                    _load_and_validate_test_execution_receipt(
                        receipt_sources_by_layer[layer_id][receipt_index],
                        layer_id=layer_id,
                        layer_status=layer["status"],
                        command=layer["command"],
                        implementation_commit=implementation_commit,
                        habitat_runtime_commit=habitat_commit,
                        rlr_commit=rlr_commit,
                    )
                except ReleaseManifestError as exc:
                    layer_errors.extend(
                        f"{layer_id} receipt: {error}" for error in exc.errors
                    )
        statuses = [bundle_by_id[item]["status"] for item in references]
        if layer.get("status") == "pass" and any(
            status != "pass" for status in statuses
        ):
            layer_errors.append(f"{layer_id} pass references non-pass evidence")
        if layer.get("status") == "fail" and "fail" not in statuses:
            layer_errors.append(f"{layer_id} fail lacks failed evidence")
    if layer_errors:
        raise ReleaseManifestError(layer_errors)
    return manifest


def prepare_release_manifest(
    request_path: str | Path,
    *,
    avengine_root: str | Path,
    habitat_runtime_root: str | Path,
    artifact_roots: Mapping[str, str | Path] | None = None,
) -> Path:
    """Atomically write, without replacement, a manifest prepared on commit A."""

    root = Path(avengine_root).resolve(strict=True)
    habitat_root = Path(habitat_runtime_root).resolve(strict=True)
    roots = _resolved_build_roots(
        avengine_root=root,
        habitat_runtime_root=habitat_root,
        artifact_roots=artifact_roots,
    )
    policy = WorkspacePathPolicy.from_roots(roots.values())
    selected_request = policy.resolve_input(
        request_path, owner="release build request", kind="file"
    )
    request = load_json_strict(selected_request)
    manifest = build_release_manifest(
        request,
        avengine_root=root,
        habitat_runtime_root=habitat_root,
        artifact_roots=artifact_roots,
    )
    destination = _resolve_repository_manifest_path(
        manifest["release"]["manifest_path"],
        repository_root=root,
        owner="release manifest output",
        strict=False,
    )
    payload = (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    published = write_bytes_no_clobber(policy, destination, payload)
    # Read back exact bytes and schema before handing control to the metadata
    # commit step.  Git identity/tag verification intentionally happens later.
    load_release_manifest(published)
    return published


def verify_release_manifest(
    manifest_path: str | Path,
    *,
    avengine_root: str | Path,
    habitat_runtime_root: str | Path,
    artifact_roots: Mapping[str, str | Path] | None = None,
    schema_path: str | Path | None = None,
    verify_git: bool = True,
    verify_tag: bool = True,
    verify_environment: bool = True,
    verify_m6_evidence: bool = True,
) -> dict[str, Any]:
    """Recompute every portable hash and Git identity in one release manifest."""

    checks: list[dict[str, Any]] = []
    observed: dict[str, Any] = {}
    try:
        av_root = Path(avengine_root).resolve(strict=True)
        habitat_root = Path(habitat_runtime_root).resolve(strict=True)
        source = _resolve_repository_manifest_path(
            manifest_path,
            repository_root=av_root,
            owner="release manifest input",
            strict=True,
        )
        observed["manifest_file_record"] = build_file_record(
            source, root=av_root, root_id="avengine"
        )
    except (OSError, ValueError, ReleaseManifestError) as exc:
        _check(checks, "manifest_json", [f"unable to resolve manifest: {exc}"])
        return {
            "schema": "avengine_release_verification_v1",
            "status": "fail",
            "checks": checks,
            "observed": observed,
        }
    try:
        manifest = load_json_strict(source)
    except ReleaseManifestError as exc:
        _check(checks, "manifest_json", exc.errors)
        return {
            "schema": "avengine_release_verification_v1",
            "status": "fail",
            "checks": checks,
            "observed": observed,
        }
    _check(checks, "manifest_json", [])

    document_errors = validate_release_manifest_document(
        manifest, schema_path=schema_path
    )
    _check(checks, "manifest_schema", document_errors)
    if document_errors:
        return {
            "schema": "avengine_release_verification_v1",
            "status": "fail",
            "checks": checks,
            "observed": observed,
        }

    roots: dict[str, Path] = {
        "avengine": av_root,
        "habitat_runtime": habitat_root,
    }
    for root_id, root in (artifact_roots or {}).items():
        if root_id in roots:
            _check(
                checks,
                "artifact_roots",
                [f"artifact root {root_id!r} attempts to replace a repository root"],
            )
            return {
                "schema": "avengine_release_verification_v1",
                "status": "fail",
                "checks": checks,
                "observed": observed,
            }
        roots[root_id] = Path(root)

    schema_set = manifest["schemas"]
    schema_records = schema_set["files"]
    schema_errors = _verify_file_records(schema_records, roots)
    if schema_set["algorithm"] != SCHEMA_SET_ALGORITHM:
        schema_errors.append("unsupported schema-set hash algorithm")
    try:
        actual_set_hash = canonical_file_record_set_sha256(schema_records)
        observed["schema_set_sha256"] = actual_set_hash
        if actual_set_hash != schema_set["set_sha256"]:
            schema_errors.append("schema set SHA-256 mismatch")
    except ReleaseManifestError as exc:
        schema_errors.extend(exc.errors)
    try:
        schema_directory = (av_root.resolve(strict=True) / schema_set["directory"]).resolve(
            strict=True
        )
        schema_directory.relative_to(av_root.resolve(strict=True))
        actual_schema_paths = {
            path.resolve().relative_to(av_root.resolve()).as_posix()
            for path in schema_directory.rglob("*.json")
            if path.is_file()
        }
        declared_schema_paths = {
            record["path"]
            for record in schema_records
            if record["root_id"] == "avengine"
        }
        if actual_schema_paths != declared_schema_paths:
            missing = sorted(actual_schema_paths - declared_schema_paths)
            extra = sorted(declared_schema_paths - actual_schema_paths)
            schema_errors.append(
                f"schema inventory mismatch; missing={missing}, extra={extra}"
            )
        if any(record["root_id"] != "avengine" for record in schema_records):
            schema_errors.append("every schema record must use root_id 'avengine'")
    except (OSError, ValueError) as exc:
        schema_errors.append(f"unable to verify complete schema inventory: {exc}")
    _check(checks, "schema_set", schema_errors)

    native_records = list(manifest["native_artifacts"].values())
    native_errors = _verify_file_records(native_records, roots)
    if any(record["root_id"] != "habitat_runtime" for record in native_records):
        native_errors.append("native artifacts must use root_id 'habitat_runtime'")
    _check(checks, "native_artifacts", native_errors)

    bundle_errors: list[str] = []
    bundle_by_id: dict[str, Mapping[str, Any]] = {}
    for bundle in manifest["evidence_bundles"]:
        evidence_id = bundle["evidence_id"]
        if evidence_id in bundle_by_id:
            bundle_errors.append(f"duplicate evidence_id {evidence_id!r}")
            continue
        bundle_by_id[evidence_id] = bundle
        bundle_errors.extend(_verify_file_records(bundle["artifacts"], roots))
        try:
            actual = canonical_file_record_set_sha256(bundle["artifacts"])
            if actual != bundle["bundle_sha256"]:
                bundle_errors.append(f"bundle SHA-256 mismatch for {evidence_id}")
        except ReleaseManifestError as exc:
            bundle_errors.extend(exc.errors)
    _check(checks, "evidence_bundles", bundle_errors)

    m6_errors: list[str] = []
    m6_roles = manifest["m6_evidence"]
    controlled_id = m6_roles["controlled_canary_bundle_id"]
    room_id = m6_roles["room_qualification_bundle_id"]
    controlled_entry_record = m6_roles["controlled_canary_entry"]
    room_entry_record = m6_roles["room_qualification_entry"]
    controlled_entry: Path | None = None
    room_entry: Path | None = None
    if controlled_id == room_id:
        m6_errors.append("controlled and room evidence roles must be distinct")
    for role, bundle_id, entry_record, expected_scope in (
        (
            "controlled canary",
            controlled_id,
            controlled_entry_record,
            "controlled_canary_verifier",
        ),
        (
            "room qualification attempt",
            room_id,
            room_entry_record,
            "room_attempt_verifier",
        ),
    ):
        bundle = bundle_by_id.get(bundle_id)
        if bundle is None:
            m6_errors.append(f"{role} references unknown bundle {bundle_id!r}")
            continue
        if bundle.get("status") != "pass":
            m6_errors.append(f"{role} bundle is not pass")
        if bundle.get("status_scope") != expected_scope:
            m6_errors.append(
                f"{role} bundle status_scope must be {expected_scope}; pass "
                "describes verifier success, not room or dataset qualification"
            )
        if entry_record not in bundle.get("artifacts", []):
            m6_errors.append(f"{role} entry is not in its exact evidence closure")
        m6_errors.extend(_verify_file_records([entry_record], roots))
    controlled_entry, controlled_resolution_errors = _resolve_record(
        controlled_entry_record, roots
    )
    room_entry, room_resolution_errors = _resolve_record(room_entry_record, roots)
    m6_errors.extend(controlled_resolution_errors)
    m6_errors.extend(room_resolution_errors)
    if (
        verify_m6_evidence
        and controlled_entry is not None
        and room_entry is not None
    ):
        try:
            controlled_document = load_json_strict(controlled_entry)
            room_document = load_json_strict(room_entry)
            _validate_controlled_canary_release_binding(
                controlled_entry,
                controlled_document,
                implementation_commit=manifest["repositories"]["avengine"][
                    "implementation_commit"
                ],
                release_id=manifest["release"]["release_id"],
                release_tag=manifest["release"]["tag"],
                manifest_path=manifest["release"]["manifest_path"],
            )
            _validate_room_attempt_release_binding(
                room_document,
                implementation_commit=manifest["repositories"]["avengine"][
                    "implementation_commit"
                ],
            )
            _require_current_m6_evidence_verifiers(
                controlled_canary=controlled_entry,
                room_attempt=room_entry,
            )
        except (OSError, ValueError, RuntimeError, ReleaseManifestError) as exc:
            if isinstance(exc, ReleaseManifestError):
                m6_errors.extend(exc.errors)
            else:
                m6_errors.append(f"unable to verify M6 semantic evidence: {exc}")
    _check(checks, "m6_evidence", m6_errors)

    layer_errors: list[str] = []
    for layer_id in TEST_LAYER_IDS:
        layer = manifest["test_layers"][layer_id]
        references = layer["evidence_bundle_ids"]
        missing = [item for item in references if item not in bundle_by_id]
        if missing:
            layer_errors.append(
                f"{layer_id} references unknown evidence bundles: {missing}"
            )
            continue
        receipts = layer["receipt_artifacts"]
        expected_receipt_count = 1 if layer["status"] in {"pass", "fail"} else 0
        if len(receipts) != expected_receipt_count:
            layer_errors.append(
                f"{layer_id} {layer['status']} requires exactly "
                f"{expected_receipt_count} receipt(s)"
            )
        layer_errors.extend(
            f"{layer_id} receipt: {error}"
            for error in _verify_file_records(receipts, roots)
        )
        referenced_artifacts = [
            artifact
            for bundle_id in references
            for artifact in bundle_by_id[bundle_id]["artifacts"]
        ]
        test_execution_artifacts = [
            artifact
            for bundle_id in references
            if bundle_by_id[bundle_id]["status_scope"] == "test_execution"
            for artifact in bundle_by_id[bundle_id]["artifacts"]
        ]
        for receipt in receipts:
            if receipt not in referenced_artifacts:
                layer_errors.append(
                    f"{layer_id} receipt artifact is not a member of a referenced "
                    f"evidence bundle: {receipt['root_id']}:{receipt['path']}"
                )
            elif receipt not in test_execution_artifacts:
                layer_errors.append(
                    f"{layer_id} receipt artifact is not a member of a referenced "
                    "test_execution evidence bundle"
                )
            receipt_path, receipt_resolution_errors = _resolve_record(receipt, roots)
            if receipt_resolution_errors:
                continue
            assert receipt_path is not None
            try:
                _load_and_validate_test_execution_receipt(
                    receipt_path,
                    layer_id=layer_id,
                    layer_status=layer["status"],
                    command=layer["command"],
                    implementation_commit=manifest["repositories"]["avengine"][
                        "implementation_commit"
                    ],
                    habitat_runtime_commit=manifest["repositories"][
                        "habitat_runtime"
                    ]["commit"],
                    rlr_commit=manifest["repositories"]["habitat_runtime"][
                        "rlr_commit"
                    ],
                )
            except ReleaseManifestError as exc:
                layer_errors.extend(
                    f"{layer_id} receipt: {error}" for error in exc.errors
                )
        statuses = [bundle_by_id[item]["status"] for item in references]
        if layer["status"] == "pass" and any(status != "pass" for status in statuses):
            layer_errors.append(f"{layer_id} pass references non-pass evidence")
        if layer["status"] == "fail" and "fail" not in statuses:
            layer_errors.append(f"{layer_id} fail lacks failed evidence")
    release_state = manifest["release"]["state"]
    release_canary = manifest["test_layers"]["release-canary"]
    required_release_refs = {controlled_id, room_id}
    if not required_release_refs.issubset(
        set(release_canary["evidence_bundle_ids"])
    ):
        layer_errors.append(
            "release-canary must reference both controlled-canary and "
            "room-attempt verifier bundles"
        )
    if release_state != "candidate":
        layer_errors.append(
            "release manifest state must remain candidate; post-tag status belongs "
            "to the external release attestation"
        )
    if release_canary["status"] != "not_run":
        layer_errors.append(
            "candidate release-canary must remain not_run until the post-tag "
            "final attestation"
        )
    if release_canary["receipt_artifacts"]:
        layer_errors.append(
            "candidate release-canary cannot contain a post-tag final "
            "attestation receipt"
        )
    if not release_canary["command"]:
        layer_errors.append(
            "candidate release-canary must retain its planned final verify command"
        )
    else:
        layer_errors.extend(
            _planned_release_verify_command_errors(
                release_canary["command"],
                avengine_root=av_root.resolve(strict=True),
                habitat_runtime_root=habitat_root.resolve(strict=True),
                manifest_relative=manifest["release"]["manifest_path"],
                artifact_roots={
                    root_id: roots[root_id]
                    for root_id in {
                        record["root_id"]
                        for bundle in manifest["evidence_bundles"]
                        for record in bundle["artifacts"]
                    }
                    if root_id not in {"avengine", "habitat_runtime"}
                    and root_id in roots
                },
            )
        )
    _check(checks, "test_layers", layer_errors)

    if verify_environment:
        _check(checks, "environment", _verify_environment(manifest["environment"]))
    else:
        _check(checks, "environment", [])

    if verify_git:
        git_errors, git_observed = _verify_git_identity(
            source,
            manifest,
            avengine_root=av_root,
            habitat_root=habitat_root,
            verify_tag=verify_tag,
        )
        observed.update(git_observed)
        _check(checks, "git_identity", git_errors)
    else:
        _check(checks, "git_identity", [])

    status = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    return {
        "schema": "avengine_release_verification_v1",
        "status": status,
        "checks": checks,
        "observed": observed,
    }


def _successful_release_verification_errors(
    report: Mapping[str, Any], *, manifest_record: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != "avengine_release_verification_v1":
        errors.append("verification report has the wrong schema")
    if report.get("status") != "pass":
        errors.append("verification report must pass")
    checks = report.get("checks")
    if not isinstance(checks, list):
        errors.append("verification report checks must be an array")
    else:
        check_ids = [
            check.get("check_id") if isinstance(check, Mapping) else None
            for check in checks
        ]
        if check_ids != list(RELEASE_VERIFICATION_CHECK_IDS):
            errors.append(
                "verification report must contain the complete ordered check set: "
                f"{list(RELEASE_VERIFICATION_CHECK_IDS)}"
            )
        for check in checks:
            if not isinstance(check, Mapping):
                continue
            if check.get("status") != "pass" or check.get("errors") != []:
                errors.append(
                    f"verification check {check.get('check_id')!r} did not pass cleanly"
                )
    observed = report.get("observed")
    if not isinstance(observed, Mapping):
        errors.append("verification report observed must be an object")
    elif observed.get("manifest_file_record") != dict(manifest_record):
        errors.append(
            "verification report is not bound to the current manifest file record"
        )
    return errors


def write_release_attestation(
    output_path: str | Path,
    *,
    manifest_path: str | Path,
    avengine_root: str | Path,
    habitat_runtime_root: str | Path,
    verification_command: Sequence[str],
    artifact_roots: Mapping[str, str | Path] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Run the full live verifier and publish its immutable post-tag result."""

    root = Path(avengine_root).resolve(strict=True)
    habitat_root = Path(habitat_runtime_root).resolve(strict=True)
    source = _resolve_repository_manifest_path(
        manifest_path,
        repository_root=root,
        owner="release attestation manifest",
        strict=True,
    )
    manifest = load_release_manifest(source)
    expected_manifest = (root / manifest["release"]["manifest_path"]).resolve(
        strict=True
    )
    if source != expected_manifest:
        raise _build_request_error(
            "release attestation manifest",
            "path differs from the candidate manifest's repository path",
        )

    raw_output = Path(output_path)
    destination = raw_output if raw_output.is_absolute() else root / raw_output
    destination = destination.resolve(strict=False)
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise _build_request_error(
            "release attestation output", "must be inside the AVEngine repository"
        ) from exc
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"refusing to replace immutable evidence file: {destination}"
        )

    if any(
        root_id in {"avengine", "habitat_runtime"}
        for root_id in (artifact_roots or {})
    ):
        raise _build_request_error(
            "release attestation artifact_roots",
            "must not replace AVEngine or Habitat repository roots",
        )
    resolved_artifact_roots = {
        root_id: Path(path).resolve(strict=True)
        for root_id, path in (artifact_roots or {}).items()
    }
    command = [
        _require_string(value, owner=f"verification_command[{index}]")
        for index, value in enumerate(verification_command)
    ]
    command_errors = _actual_release_verify_command_errors(
        command,
        planned_command=manifest["test_layers"]["release-canary"]["command"],
        avengine_root=root,
        habitat_runtime_root=habitat_root,
        manifest_relative=manifest["release"]["manifest_path"],
        output_path=destination,
        artifact_roots=resolved_artifact_roots,
    )
    if command_errors:
        raise ReleaseManifestError(command_errors)

    manifest_record_before = build_file_record(
        source, root=root, root_id="avengine"
    )
    report = verify_release_manifest(
        source,
        avengine_root=root,
        habitat_runtime_root=habitat_root,
        artifact_roots=resolved_artifact_roots,
    )
    manifest_record_after = build_file_record(
        source, root=root, root_id="avengine"
    )
    if manifest_record_after != manifest_record_before:
        raise _build_request_error(
            "release attestation manifest", "changed during live verification"
        )
    report_errors = _successful_release_verification_errors(
        report, manifest_record=manifest_record_after
    )
    if report_errors:
        raise ReleaseManifestError(report_errors)

    observed = report["observed"]
    tag_commit = observed.get("release_tag_commit")
    metadata_commit = observed.get("avengine_metadata_commit")
    if not isinstance(tag_commit, str) or _HEX_COMMIT.fullmatch(tag_commit) is None:
        raise _build_request_error(
            "release attestation release_tag_commit",
            "successful report lacks a full post-tag commit",
        )
    if metadata_commit != tag_commit:
        raise _build_request_error(
            "release attestation release_tag_commit",
            "tag commit differs from the verified metadata commit B",
        )
    implementation_commit = manifest["repositories"]["avengine"][
        "implementation_commit"
    ]
    expected_observations = {
        "avengine_metadata_parent": implementation_commit,
        "avengine_metadata_parent_count": 1,
        "habitat_runtime_commit": manifest["repositories"]["habitat_runtime"][
            "commit"
        ],
        "rlr_commit": manifest["repositories"]["habitat_runtime"]["rlr_commit"],
    }
    for field, expected_value in expected_observations.items():
        if observed.get(field) != expected_value:
            raise _build_request_error(
                f"release attestation verification_report.observed.{field}",
                f"must equal {expected_value!r}",
            )

    attestation: dict[str, Any] = {
        "schema": RELEASE_ATTESTATION_SCHEMA,
        "status": "pass",
        "release_id": manifest["release"]["release_id"],
        "release_tag": manifest["release"]["tag"],
        "release_tag_commit": tag_commit,
        "implementation_commit": implementation_commit,
        "manifest": manifest_record_after,
        "verification_command": command,
        "verification_report": report,
    }
    schema_errors = validate_release_attestation_document(attestation)
    if schema_errors:
        raise ReleaseManifestError(schema_errors)
    payload = (
        json.dumps(
            attestation,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    policy = WorkspacePathPolicy.from_roots([root])
    published = write_bytes_no_clobber(policy, destination, payload)
    readback = load_json_strict(published)
    if readback != attestation:
        raise ReleaseManifestError(
            ["release attestation readback differs from the published document"]
        )
    readback_errors = validate_release_attestation_document(readback)
    if readback_errors:
        raise ReleaseManifestError(readback_errors)
    return published, readback


def verify_release_attestation(
    attestation_path: str | Path,
    *,
    avengine_root: str | Path,
    habitat_runtime_root: str | Path,
    artifact_roots: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Re-run the live verifier and compare it with one retained attestation."""

    checks: list[dict[str, Any]] = []
    root = Path(avengine_root).resolve(strict=True)
    habitat_root = Path(habitat_runtime_root).resolve(strict=True)
    raw_attestation = Path(attestation_path)
    source = (
        raw_attestation if raw_attestation.is_absolute() else root / raw_attestation
    ).resolve(strict=True)
    try:
        source.relative_to(root)
    except ValueError:
        _check(checks, "attestation_path", ["attestation escapes AVEngine root"])
        return {
            "schema": "avengine_release_attestation_verification_v1",
            "status": "fail",
            "checks": checks,
        }
    _check(checks, "attestation_path", [])
    try:
        document = load_json_strict(source)
    except ReleaseManifestError as exc:
        _check(checks, "attestation_schema", exc.errors)
        return {
            "schema": "avengine_release_attestation_verification_v1",
            "status": "fail",
            "checks": checks,
        }
    _check(
        checks,
        "attestation_schema",
        validate_release_attestation_document(document),
    )
    if checks[-1]["status"] != "pass":
        return {
            "schema": "avengine_release_attestation_verification_v1",
            "status": "fail",
            "checks": checks,
        }

    roots: dict[str, Path] = {
        "avengine": root,
        "habitat_runtime": habitat_root,
    }
    for root_id, path in (artifact_roots or {}).items():
        if root_id in roots:
            _check(
                checks,
                "artifact_roots",
                [f"artifact root {root_id!r} replaces a repository root"],
            )
            return {
                "schema": "avengine_release_attestation_verification_v1",
                "status": "fail",
                "checks": checks,
            }
        roots[root_id] = Path(path).resolve(strict=True)
    manifest_record = document["manifest"]
    manifest_errors = _verify_file_records([manifest_record], roots)
    manifest_path_resolved, resolution_errors = _resolve_record(
        manifest_record, roots
    )
    manifest_errors.extend(resolution_errors)
    _check(checks, "manifest_file", manifest_errors)
    if manifest_path_resolved is None:
        return {
            "schema": "avengine_release_attestation_verification_v1",
            "status": "fail",
            "checks": checks,
        }
    try:
        manifest = load_release_manifest(manifest_path_resolved)
    except ReleaseManifestError as exc:
        _check(checks, "manifest_document", exc.errors)
        return {
            "schema": "avengine_release_attestation_verification_v1",
            "status": "fail",
            "checks": checks,
        }
    _check(checks, "manifest_document", [])

    command_errors = _actual_release_verify_command_errors(
        document["verification_command"],
        planned_command=manifest["test_layers"]["release-canary"]["command"],
        avengine_root=root,
        habitat_runtime_root=habitat_root,
        manifest_relative=manifest["release"]["manifest_path"],
        output_path=source,
        artifact_roots={
            root_id: path
            for root_id, path in roots.items()
            if root_id not in {"avengine", "habitat_runtime"}
        },
    )
    _check(checks, "verification_command", command_errors)

    current_report = verify_release_manifest(
        manifest_record["path"],
        avengine_root=root,
        habitat_runtime_root=habitat_root,
        artifact_roots={
            root_id: path
            for root_id, path in roots.items()
            if root_id not in {"avengine", "habitat_runtime"}
        },
    )
    report_errors = _successful_release_verification_errors(
        current_report, manifest_record=manifest_record
    )
    if current_report != document["verification_report"]:
        report_errors.append(
            "retained verification_report differs from a fresh full verification"
        )
    _check(checks, "fresh_verification", report_errors)

    identity_errors: list[str] = []
    expected = {
        "release_id": manifest["release"]["release_id"],
        "release_tag": manifest["release"]["tag"],
        "implementation_commit": manifest["repositories"]["avengine"][
            "implementation_commit"
        ],
        "release_tag_commit": current_report["observed"].get(
            "release_tag_commit"
        ),
    }
    for field, value in expected.items():
        if document.get(field) != value:
            identity_errors.append(
                f"attestation {field} differs from fresh release verification"
            )
    _check(checks, "release_identity", identity_errors)
    status = "pass" if all(item["status"] == "pass" for item in checks) else "fail"
    return {
        "schema": "avengine_release_attestation_verification_v1",
        "status": status,
        "checks": checks,
    }


def require_verified_release_manifest(*args: Any, **kwargs: Any) -> dict[str, Any]:
    report = verify_release_manifest(*args, **kwargs)
    if report["status"] != "pass":
        errors = [
            error
            for check in report["checks"]
            for error in check.get("errors", [])
        ]
        raise ReleaseManifestError(errors or ["release verification failed"])
    return report
