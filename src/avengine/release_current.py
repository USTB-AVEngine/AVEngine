"""Current-installed ordinary release-candidate writer and verifier.

The historical v1 release reader remains intentionally untouched: it records
the old Habitat checkout and its RLR submodule. This v2 path records an
AVEngine commit plus explicit, non-checkout installed inputs instead. It is an
ordinary candidate record only. It never claims that RLR was loaded, that an
adapter was enabled, or that a formal release was produced.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from avengine.release import (
    ReleaseManifestError,
    build_file_record,
    load_json_strict,
    sha256_file,
)
from avengine.release_current_receipt import (
    CurrentReleaseReceiptError,
    CurrentRuntimeInputs,
    _current_git_environment,
    logical_current_tmp_path,
    validate_current_runtime_inputs,
    verify_current_receipt_payload,
)
from avengine.security.path_policy import (
    WorkspacePathPolicy,
    write_bytes_no_clobber,
)


CURRENT_RELEASE_MANIFEST_SCHEMA = "avengine_release_manifest_v2"
CURRENT_RELEASE_MANIFEST_SCHEMA_FILE = "avengine_release_manifest_v2.schema.json"
CURRENT_RELEASE_BUILD_REQUEST_SCHEMA = "avengine_current_release_build_request_v2"
CURRENT_RELEASE_CHECK_IDS = (
    "manifest_json",
    "manifest_schema",
    "repository_identity",
    "runtime_inputs",
    "ordinary_test_receipt",
    "claim_boundary",
)
CURRENT_RUNTIME_INPUT_ROLES = {
    "mode": "current-installed",
    "habitat_runtime_prefix": "external-habitat-runtime-prefix",
    "rlr_sdk_root": "external-rlr-sdk",
    "scene_data_root": "external-scene-data",
    "magnum_python_site": "external-magnum-python-site",
}
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_STABLE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_FILE_URL = re.compile(r"(?i)\bfile:(?://)?")
_POSIX_ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9._~/-])/(?!/)(?:[^\s\"'<>()\[\]{}]+)?"
)
_POSIX_NETWORK_ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9._~/:])//(?:[^\s\"'<>()\[\]{}]+)?"
)
_WINDOWS_ABSOLUTE_PATH = re.compile(
    r"(?i)(?:^|[\s\"'=:(\[])[a-z]:[\\/]"
)
_WINDOWS_UNC_ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9._~/-])\\\\(?:[^\\/\s]+(?:[\\/][^\\/\s]+)?|$)"
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def current_release_schema_path() -> Path:
    source = _repository_root() / "schemas" / CURRENT_RELEASE_MANIFEST_SCHEMA_FILE
    if source.is_file():
        return source
    installed = (
        Path(sys.prefix)
        / "share"
        / "avengine"
        / "schemas"
        / CURRENT_RELEASE_MANIFEST_SCHEMA_FILE
    )
    if installed.is_file():
        return installed
    raise ReleaseManifestError(
        [f"current release manifest schema is unavailable: {source}"]
    )


def _validate_schema_document(
    value: Mapping[str, Any],
    *,
    schema_path: Path,
    owner: str,
) -> list[str]:
    try:
        schema = load_json_strict(schema_path)
        Draft202012Validator.check_schema(schema)
    except (ReleaseManifestError, ValueError) as exc:
        if isinstance(exc, ReleaseManifestError):
            return [f"{owner}: {error}" for error in exc.errors]
        return [f"{owner} schema is invalid: {exc}"]
    validator = Draft202012Validator(schema)
    return [
        f"{owner}.{'.'.join(str(part) for part in error.absolute_path) or '$'}: "
        f"{error.message}"
        for error in sorted(
            validator.iter_errors(dict(value)),
            key=lambda item: tuple(str(part) for part in item.absolute_path),
        )
    ]


def validate_current_release_manifest_document(
    value: Mapping[str, Any],
    *,
    schema_path: str | Path | None = None,
) -> list[str]:
    """Return deterministic schema and portable-boundary errors for v2."""

    selected = current_release_schema_path() if schema_path is None else Path(schema_path)
    errors = _validate_schema_document(
        value,
        schema_path=selected,
        owner="current release manifest",
    )
    release = value.get("release")
    if isinstance(release, Mapping):
        manifest_path = release.get("manifest_path")
        if isinstance(manifest_path, str) and not manifest_path.startswith("release/"):
            errors.append(
                "release.manifest_path must remain beneath release/: "
                f"{manifest_path!r}"
            )
        for field in ("formal_release_reason", "current_milestone"):
            field_value = release.get(field)
            if isinstance(field_value, str):
                errors.extend(
                    _filesystem_path_leakage_errors(
                        field_value,
                        owner=f"release.{field}",
                    )
                )
    repositories = value.get("repositories")
    if isinstance(repositories, Mapping):
        avengine = repositories.get("avengine")
        if isinstance(avengine, Mapping):
            repository = avengine.get("repository")
            if isinstance(repository, str):
                errors.extend(
                    _filesystem_path_leakage_errors(
                        repository,
                        owner="repositories.avengine.repository",
                    )
                )
    runtime_inputs = value.get("runtime_inputs")
    if isinstance(runtime_inputs, Mapping) and dict(runtime_inputs) != CURRENT_RUNTIME_INPUT_ROLES:
        errors.append(
            "runtime_inputs must contain only the fixed logical external input "
            "roles, never server paths"
        )
    return errors


def _filesystem_path_leakage_errors(value: str, *, owner: str) -> list[str]:
    """Reject private filesystem syntax from Git-bound free-text fields."""

    errors: list[str] = []
    if _FILE_URL.search(value):
        errors.append(f"{owner} must not contain a file URL")
    if (
        _POSIX_ABSOLUTE_PATH.search(value)
        or _POSIX_NETWORK_ABSOLUTE_PATH.search(value)
    ):
        errors.append(f"{owner} must not contain a filesystem absolute path")
    if _WINDOWS_ABSOLUTE_PATH.search(value):
        errors.append(f"{owner} must not contain a filesystem absolute path")
    if _WINDOWS_UNC_ABSOLUTE_PATH.search(value):
        errors.append(f"{owner} must not contain a filesystem absolute path")
    return errors


def load_current_release_manifest(
    path: str | Path,
    *,
    schema_path: str | Path | None = None,
) -> dict[str, Any]:
    value = load_json_strict(path)
    errors = validate_current_release_manifest_document(value, schema_path=schema_path)
    if errors:
        raise ReleaseManifestError(errors)
    return value


def _require_mapping(value: Any, *, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseManifestError([f"{owner} must be an object"])
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    owner: str,
    required: set[str],
) -> None:
    actual = set(value)
    missing = sorted(required - actual)
    extra = sorted(actual - required)
    if missing or extra:
        raise ReleaseManifestError(
            [f"{owner} must have exactly {sorted(required)}; missing={missing}, extra={extra}"]
        )


def _require_string(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseManifestError([f"{owner} must be a nonempty string"])
    return value


def _require_stable_id(value: Any, *, owner: str) -> str:
    text = _require_string(value, owner=owner)
    if _STABLE_ID.fullmatch(text) is None:
        raise ReleaseManifestError([f"{owner} is not a stable lowercase identifier"])
    return text


def _require_commit(value: Any, *, owner: str) -> str:
    text = _require_string(value, owner=owner)
    if _COMMIT.fullmatch(text) is None:
        raise ReleaseManifestError([f"{owner} is not a full Git commit"])
    return text


def _require_relative_path(value: Any, *, owner: str) -> str:
    text = _require_string(value, owner=owner)
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ReleaseManifestError(
            [f"{owner} must be a normalized repository-relative path"]
        )
    if "\\" in text:
        raise ReleaseManifestError([f"{owner} must use POSIX separators"])
    return path.as_posix()


def _git(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            env=_current_git_environment(),
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReleaseManifestError(
            [f"could not inspect AVEngine repository: {exc}"]
        ) from exc
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "git failed"
        raise ReleaseManifestError(
            [f"git -C {root} {' '.join(arguments)}: {message}"]
        )
    return completed.stdout.strip()


def _normalize_git_url(value: str) -> str:
    normalized = value.strip().removesuffix(".git")
    if normalized.startswith("git@") and ":" in normalized:
        host, path = normalized[4:].split(":", 1)
        normalized = f"{host}/{path}"
    else:
        normalized = re.sub(r"^(?:https?|ssh)://(?:git@)?", "", normalized)
    return normalized.rstrip("/").lower()


def _require_repository_identity(
    root: Path,
    *,
    implementation_commit: str,
    expected_repository: str,
    require_head: bool,
    require_clean: bool,
) -> list[str]:
    errors: list[str] = []
    try:
        _git(root, "cat-file", "-e", f"{implementation_commit}^{{commit}}")
        actual_remote = _git(root, "remote", "get-url", "origin")
        if _normalize_git_url(actual_remote) != _normalize_git_url(expected_repository):
            errors.append(
                "origin URL mismatch: "
                f"declared {expected_repository!r}, actual {actual_remote!r}"
            )
        if require_head:
            actual_head = _git(root, "rev-parse", "HEAD")
            if actual_head != implementation_commit:
                errors.append(
                    "AVEngine HEAD mismatch: "
                    f"declared {implementation_commit}, actual {actual_head}"
                )
        if require_clean:
            status = _git(root, "status", "--porcelain", "--untracked-files=all")
            if status:
                errors.append(
                    "AVEngine worktree must be clean before current manifest "
                    f"preparation: {status!r}"
                )
    except ReleaseManifestError as exc:
        errors.extend(exc.errors)
    return errors


def _tmp_compatibility_root(repository_root: Path) -> Path | None:
    """Return the one declared external tmp target, if this worktree has one."""

    tmp_link = repository_root / "tmp"
    if not tmp_link.is_symlink():
        return None
    try:
        tmp_root = tmp_link.resolve(strict=True)
    except OSError as exc:
        raise ReleaseManifestError(
            [f"AVEngine tmp compatibility root cannot be resolved: {exc}"]
        ) from exc
    if not tmp_root.is_dir():
        raise ReleaseManifestError(
            [f"AVEngine tmp compatibility root is not a directory: {tmp_root}"]
        )
    return tmp_root


def _workspace_policy(repository_root: Path) -> WorkspacePathPolicy:
    roots = [repository_root]
    tmp_root = _tmp_compatibility_root(repository_root)
    if tmp_root is not None:
        roots.append(tmp_root)
    return WorkspacePathPolicy.from_roots(roots)


def _logical_repository_path(
    path: str | Path,
    *,
    repository_root: Path,
    owner: str,
) -> str:
    requested = Path(path)
    root = repository_root.resolve(strict=True)
    if not requested.is_absolute():
        return _require_relative_path(str(path), owner=owner)
    try:
        return _require_relative_path(
            requested.relative_to(root).as_posix(),
            owner=owner,
        )
    except ValueError:
        pass
    tmp_root = _tmp_compatibility_root(root)
    if tmp_root is not None:
        try:
            relative = requested.resolve(strict=False).relative_to(tmp_root)
        except (OSError, ValueError):
            pass
        else:
            return _require_relative_path(
                (Path("tmp") / relative).as_posix(),
                owner=owner,
            )
    raise ReleaseManifestError(
        [f"{owner} escapes the AVEngine repository: {requested}"]
    )


def _resolve_repository_file(
    path: str | Path,
    *,
    repository_root: Path,
    owner: str,
    strict: bool,
) -> Path:
    root = repository_root.resolve(strict=True)
    raw = _logical_repository_path(
        path,
        repository_root=root,
        owner=owner,
    )
    parts = PurePosixPath(raw).parts
    tmp_root = _tmp_compatibility_root(root)
    if parts[:1] == ("tmp",) and tmp_root is not None:
        candidate = tmp_root.joinpath(*parts[1:])
        cursor = tmp_root
        checked_parts = parts[1:]
        containment_root = tmp_root
    else:
        candidate = root / Path(raw)
        cursor = root
        checked_parts = parts
        containment_root = root
    for part in checked_parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ReleaseManifestError(
                [f"{owner} traverses a symlink: {cursor}"]
            )
    try:
        resolved = candidate.resolve(strict=strict)
        resolved.relative_to(containment_root)
    except (OSError, ValueError) as exc:
        raise ReleaseManifestError(
            [f"{owner} cannot be resolved inside AVEngine: {exc}"]
        ) from exc
    if strict and not resolved.is_file():
        raise ReleaseManifestError([f"{owner} is not a regular file: {resolved}"])
    return resolved


def _build_repository_file_record(
    path: str | Path,
    *,
    repository_root: Path,
    owner: str,
) -> dict[str, Any]:
    """Bind one ordinary receipt while preserving the logical tmp pathname."""

    raw = _logical_repository_path(
        path,
        repository_root=repository_root,
        owner=owner,
    )
    source = _resolve_repository_file(
        raw,
        repository_root=repository_root,
        owner=owner,
        strict=True,
    )
    return {
        "root_id": "avengine",
        "path": raw,
        "byte_size": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def _check(checks: list[dict[str, Any]], check_id: str, errors: Sequence[str]) -> None:
    checks.append(
        {
            "check_id": check_id,
            "status": "pass" if not errors else "fail",
            "errors": list(errors),
        }
    )


def _parse_current_build_request(
    request: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    _require_exact_keys(
        request,
        owner="current release build request",
        required={"schema", "release", "repositories", "ordinary_test_receipt"},
    )
    if request.get("schema") != CURRENT_RELEASE_BUILD_REQUEST_SCHEMA:
        raise ReleaseManifestError(
            [
                "current release build request schema must be "
                f"{CURRENT_RELEASE_BUILD_REQUEST_SCHEMA!r}"
            ]
        )
    release = _require_mapping(request["release"], owner="release")
    _require_exact_keys(
        release,
        owner="release",
        required={
            "release_id",
            "current_milestone",
            "manifest_path",
            "formal_release_reason",
        },
    )
    release_values = {
        "release_id": _require_stable_id(release["release_id"], owner="release.release_id"),
        "current_milestone": _require_string(
            release["current_milestone"],
            owner="release.current_milestone",
        ),
        "manifest_path": _require_relative_path(
            release["manifest_path"],
            owner="release.manifest_path",
        ),
        "formal_release_reason": _require_string(
            release["formal_release_reason"],
            owner="release.formal_release_reason",
        ),
    }
    if not release_values["manifest_path"].startswith("release/"):
        raise ReleaseManifestError(
            ["release.manifest_path must remain beneath release/"]
        )
    release_text_errors = [
        *_filesystem_path_leakage_errors(
            release_values["current_milestone"],
            owner="release.current_milestone",
        ),
        *_filesystem_path_leakage_errors(
            release_values["formal_release_reason"],
            owner="release.formal_release_reason",
        ),
    ]
    if release_text_errors:
        raise ReleaseManifestError(release_text_errors)
    repositories = _require_mapping(request["repositories"], owner="repositories")
    _require_exact_keys(
        repositories,
        owner="repositories",
        required={"implementation_commit", "expected_avengine_repository"},
    )
    repository_values = {
        "implementation_commit": _require_commit(
            repositories["implementation_commit"],
            owner="repositories.implementation_commit",
        ),
        "expected_avengine_repository": _require_string(
            repositories["expected_avengine_repository"],
            owner="repositories.expected_avengine_repository",
        ),
    }
    repository_text_errors = _filesystem_path_leakage_errors(
        repository_values["expected_avengine_repository"],
        owner="repositories.expected_avengine_repository",
    )
    if repository_text_errors:
        raise ReleaseManifestError(repository_text_errors)
    receipt = _require_mapping(
        request["ordinary_test_receipt"],
        owner="ordinary_test_receipt",
    )
    _require_exact_keys(
        receipt,
        owner="ordinary_test_receipt",
        required={"root_id", "path"},
    )
    if receipt["root_id"] != "avengine":
        raise ReleaseManifestError(
            ["ordinary_test_receipt.root_id must be 'avengine'"]
        )
    receipt_values = {
        "root_id": "avengine",
        "path": _require_relative_path(
            receipt["path"],
            owner="ordinary_test_receipt.path",
        ),
    }
    return release_values, repository_values, receipt_values


def _load_and_validate_receipt(
    source: Path,
    *,
    expected_commit: str,
    expected_runtime: CurrentRuntimeInputs,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    try:
        receipt = load_json_strict(source)
    except ReleaseManifestError as exc:
        return {}, exc.errors
    errors.extend(verify_current_receipt_payload(receipt))
    if receipt.get("implementation_commit") != expected_commit:
        errors.append("receipt implementation_commit differs from manifest request")
    if receipt.get("runtime_inputs") != expected_runtime.as_document():
        errors.append(
            "receipt runtime_inputs differ from the explicit current-installed "
            "runtime inputs"
        )
    if receipt.get("claim_scope") != "ordinary_current_candidate":
        errors.append("receipt must remain within ordinary_current_candidate scope")
    observation = receipt.get("runtime_observation")
    if not isinstance(observation, Mapping) or observation.get(
        "formal_release_status"
    ) != "not_run":
        errors.append("receipt must explicitly retain formal_release_status not_run")
    return receipt, errors


def build_current_release_manifest(
    request: Mapping[str, Any],
    *,
    avengine_root: str | Path,
    runtime_prefix: str | Path,
    rlr_sdk_root: str | Path,
    scene_data_root: str | Path,
    magnum_python_site: str | Path,
) -> dict[str, Any]:
    """Build one current-installed ordinary candidate without checkout inputs."""

    root = Path(avengine_root).resolve(strict=True)
    release, repository, receipt_request = _parse_current_build_request(request)
    try:
        logical_receipt_path = logical_current_tmp_path(
            root,
            receipt_request["path"],
            owner="ordinary_test_receipt.path",
        )
    except CurrentReleaseReceiptError as exc:
        raise ReleaseManifestError(exc.errors) from exc
    receipt_request["path"] = logical_receipt_path.as_posix()
    identity_errors = _require_repository_identity(
        root,
        implementation_commit=repository["implementation_commit"],
        expected_repository=repository["expected_avengine_repository"],
        require_head=True,
        require_clean=True,
    )
    if identity_errors:
        raise ReleaseManifestError(identity_errors)
    try:
        runtime = validate_current_runtime_inputs(
            runtime_prefix=runtime_prefix,
            rlr_sdk_root=rlr_sdk_root,
            scene_data_root=scene_data_root,
            magnum_python_site=magnum_python_site,
        )
    except CurrentReleaseReceiptError as exc:
        raise ReleaseManifestError(exc.errors) from exc
    if receipt_request["path"] == release["manifest_path"]:
        raise ReleaseManifestError(
            ["ordinary_test_receipt cannot be the release manifest output"]
        )
    receipt_path = _resolve_repository_file(
        receipt_request["path"],
        repository_root=root,
        owner="ordinary_test_receipt.path",
        strict=True,
    )
    receipt_record = _build_repository_file_record(
        receipt_path,
        repository_root=root,
        owner="ordinary_test_receipt.path",
    )
    receipt, receipt_errors = _load_and_validate_receipt(
        receipt_path,
        expected_commit=repository["implementation_commit"],
        expected_runtime=runtime,
    )
    if receipt_errors:
        raise ReleaseManifestError(
            [f"ordinary_test_receipt: {error}" for error in receipt_errors]
        )
    manifest: dict[str, Any] = {
        "schema": CURRENT_RELEASE_MANIFEST_SCHEMA,
        "release": {
            "release_id": release["release_id"],
            "state": "candidate",
            "claim_scope": "ordinary_current_candidate",
            "formal_release_status": "not_run",
            "formal_release_reason": release["formal_release_reason"],
            "current_milestone": release["current_milestone"],
            "manifest_path": release["manifest_path"],
        },
        "repositories": {
            "avengine": {
                "repository": repository["expected_avengine_repository"],
                "implementation_commit": repository["implementation_commit"],
            }
        },
        "runtime_inputs": dict(CURRENT_RUNTIME_INPUT_ROLES),
        "ordinary_test_receipt": receipt_record,
        "ordinary_test_status": receipt["status"],
    }
    schema_errors = validate_current_release_manifest_document(manifest)
    if schema_errors:
        raise ReleaseManifestError(schema_errors)
    return manifest


def prepare_current_release_manifest(
    request_path: str | Path,
    *,
    avengine_root: str | Path,
    runtime_prefix: str | Path,
    rlr_sdk_root: str | Path,
    scene_data_root: str | Path,
    magnum_python_site: str | Path,
) -> Path:
    """Write one no-clobber v2 ordinary candidate manifest on commit A."""

    root = Path(avengine_root).resolve(strict=True)
    policy = _workspace_policy(root)
    try:
        logical_request_path = logical_current_tmp_path(
            root,
            request_path,
            owner="current release build request",
        )
    except CurrentReleaseReceiptError as exc:
        raise ReleaseManifestError(exc.errors) from exc
    request_source = policy.resolve_input(
        root / logical_request_path,
        owner="current release build request",
        kind="file",
    )
    request = load_json_strict(request_source)
    manifest = build_current_release_manifest(
        request,
        avengine_root=root,
        runtime_prefix=runtime_prefix,
        rlr_sdk_root=rlr_sdk_root,
        scene_data_root=scene_data_root,
        magnum_python_site=magnum_python_site,
    )
    destination = _resolve_repository_file(
        manifest["release"]["manifest_path"],
        repository_root=root,
        owner="current release manifest output",
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
    load_current_release_manifest(published)
    return published


def verify_current_release_manifest(
    manifest_path: str | Path,
    *,
    avengine_root: str | Path,
    runtime_prefix: str | Path,
    rlr_sdk_root: str | Path,
    scene_data_root: str | Path,
    magnum_python_site: str | Path,
    schema_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify an ordinary candidate and explicitly retain its formal boundary."""

    checks: list[dict[str, Any]] = []
    observed: dict[str, Any] = {}
    try:
        root = Path(avengine_root).resolve(strict=True)
        source = _resolve_repository_file(
            manifest_path,
            repository_root=root,
            owner="current release manifest input",
            strict=True,
        )
        observed["manifest_file_record"] = build_file_record(
            source,
            root=root,
            root_id="avengine",
        )
    except (OSError, ReleaseManifestError) as exc:
        errors = exc.errors if isinstance(exc, ReleaseManifestError) else [str(exc)]
        _check(checks, "manifest_json", errors)
        return {
            "schema": "avengine_current_release_verification_v2",
            "status": "fail",
            "claim_scope": "ordinary_current_candidate",
            "formal_release_status": "not_run",
            "checks": checks,
            "observed": observed,
        }
    try:
        manifest = load_json_strict(source)
    except ReleaseManifestError as exc:
        _check(checks, "manifest_json", exc.errors)
        return {
            "schema": "avengine_current_release_verification_v2",
            "status": "fail",
            "claim_scope": "ordinary_current_candidate",
            "formal_release_status": "not_run",
            "checks": checks,
            "observed": observed,
        }
    _check(checks, "manifest_json", [])
    schema_errors = validate_current_release_manifest_document(
        manifest,
        schema_path=schema_path,
    )
    actual_manifest_path = source.relative_to(root).as_posix()
    if manifest["release"]["manifest_path"] != actual_manifest_path:
        schema_errors.append(
            "release.manifest_path differs from the current manifest location: "
            f"declared {manifest['release']['manifest_path']!r}, "
            f"actual {actual_manifest_path!r}"
        )
    _check(checks, "manifest_schema", schema_errors)
    if schema_errors:
        return {
            "schema": "avengine_current_release_verification_v2",
            "status": "fail",
            "claim_scope": "ordinary_current_candidate",
            "formal_release_status": "not_run",
            "checks": checks,
            "observed": observed,
        }

    repository = manifest["repositories"]["avengine"]
    identity_errors = _require_repository_identity(
        root,
        implementation_commit=repository["implementation_commit"],
        expected_repository=repository["repository"],
        require_head=False,
        require_clean=False,
    )
    _check(checks, "repository_identity", identity_errors)
    try:
        runtime = validate_current_runtime_inputs(
            runtime_prefix=runtime_prefix,
            rlr_sdk_root=rlr_sdk_root,
            scene_data_root=scene_data_root,
            magnum_python_site=magnum_python_site,
        )
        runtime_errors: list[str] = []
    except CurrentReleaseReceiptError as exc:
        runtime = None
        runtime_errors = exc.errors
    _check(checks, "runtime_inputs", runtime_errors)

    receipt_errors: list[str] = []
    if runtime is not None:
        record = manifest["ordinary_test_receipt"]
        try:
            logical_receipt_path = logical_current_tmp_path(
                root,
                record["path"],
                owner="ordinary_test_receipt.path",
            )
            receipt_path = _resolve_repository_file(
                logical_receipt_path.as_posix(),
                repository_root=root,
                owner="ordinary_test_receipt.path",
                strict=True,
            )
            actual_record = _build_repository_file_record(
                receipt_path,
                repository_root=root,
                owner="ordinary_test_receipt.path",
            )
            if actual_record != record:
                receipt_errors.append(
                    "ordinary_test_receipt file record differs from current bytes"
                )
            receipt, payload_errors = _load_and_validate_receipt(
                receipt_path,
                expected_commit=repository["implementation_commit"],
                expected_runtime=runtime,
            )
            receipt_errors.extend(payload_errors)
            if receipt.get("status") != manifest["ordinary_test_status"]:
                receipt_errors.append(
                    "ordinary_test_status differs from the bound receipt status"
                )
            if receipt.get("status") != "pass":
                receipt_errors.append("ordinary_test_receipt does not record a pass")
        except ReleaseManifestError as exc:
            receipt_errors.extend(exc.errors)
    else:
        receipt_errors.append(
            "cannot verify ordinary_test_receipt without valid explicit runtime inputs"
        )
    _check(checks, "ordinary_test_receipt", receipt_errors)

    claim_errors: list[str] = []
    release = manifest["release"]
    if release.get("claim_scope") != "ordinary_current_candidate":
        claim_errors.append("claim_scope must remain ordinary_current_candidate")
    if release.get("formal_release_status") != "not_run":
        claim_errors.append("formal_release_status must remain not_run")
    _check(checks, "claim_boundary", claim_errors)
    status = "pass" if all(item["status"] == "pass" for item in checks) else "fail"
    return {
        "schema": "avengine_current_release_verification_v2",
        "status": status,
        "claim_scope": "ordinary_current_candidate",
        "formal_release_status": "not_run",
        "checks": checks,
        "observed": observed,
    }


__all__ = [
    "CURRENT_RELEASE_BUILD_REQUEST_SCHEMA",
    "CURRENT_RELEASE_CHECK_IDS",
    "CURRENT_RELEASE_MANIFEST_SCHEMA",
    "CURRENT_RUNTIME_INPUT_ROLES",
    "build_current_release_manifest",
    "current_release_schema_path",
    "load_current_release_manifest",
    "prepare_current_release_manifest",
    "validate_current_release_manifest_document",
    "verify_current_release_manifest",
]
