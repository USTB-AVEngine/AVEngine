"""Write truthful, current-installed ordinary release test receipts.

This module is deliberately separate from avengine.release_receipt. The
historical v1 receipt records a Habitat checkout commit and an RLR submodule
gitlink. An installed AVEngine runtime has neither. Reusing that schema would
therefore force a false checkout identity. The v2 receipt observes only the
explicit external locations supplied for one run and makes no native-RLR or
formal-release claim.
"""

from __future__ import annotations

import base64
import binascii
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence
import xml.etree.ElementTree as ET

from jsonschema import Draft202012Validator

from avengine.backends.rlr.sdk import (
    ExternalRlrSdkError,
    discover_external_rlr_sdk,
    require_outside_git_checkout,
)
from avengine.security.path_policy import (
    WorkspacePathPolicy,
    write_bytes_no_clobber,
)


CURRENT_TEST_EXECUTION_RECEIPT_SCHEMA = (
    "avengine_current_test_execution_receipt_v2"
)
CURRENT_TEST_EXECUTION_RECEIPT_SCHEMA_FILE = (
    "current_test_execution_receipt_v2.schema.json"
)
CURRENT_EXECUTABLE_LAYERS = (
    "fast-unit",
    "slow-hermetic",
)
LEGACY_CHILD_ENVIRONMENT_VARIABLES = frozenset(
    {
        "AVENGINE_CAPTURE_ROOT",
        "AVENGINE_EVIDENCE_ROOT",
        "AVENGINE_HABITAT_RUNTIME_ROOT",
        "AVENGINE_HABITAT_RUNTIME_PREFIX",
        "AVENGINE_HABITAT_MAGNUM_PYTHON_SITE",
        "AVENGINE_HRTF_ROOT",
        "AVENGINE_LEGACY_ROOT",
        "AVENGINE_LEGACY_APARTMENT_ACOUSTIC_PACKAGE_ROOT",
        "AVENGINE_LEGACY_APARTMENT_EXPORT_ROOT",
        "AVENGINE_LEGACY_APARTMENT_PACKAGE_ROOT",
        "AVENGINE_MP3D_ROOT",
        "AVENGINE_MP3D_PROXY_V2_ROOT",
        "AVENGINE_MP3D_SOUNDSPACES2_PACKAGE_ROOT",
        "AVENGINE_REPLICACAD_ACOUSTIC_PACKAGE_ROOT",
        "AVENGINE_REPLICACAD_ROOT",
        "AVENGINE_REPOSITORY_ROOT",
        "AVENGINE_RLR_SDK_ROOT",
        "AVENGINE_SOUNDSPACES_ROOT",
        "AVENGINE_SPEAR_EXECUTABLE",
        "AVENGINE_SPEAR_ROOT",
        "HABITAT_ROOT",
        "HABITAT_SIM_PATH",
        "HABITAT_SIM_ROOT",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "RLR_ROOT",
        "SOUNDSPACES_ROOT",
        "SPEAR_EXECUTABLE",
        "SPEAR_HOME",
        "SPEAR_PATH",
        "SPEAR_ROOT",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
    }
)
CURRENT_CHILD_REPLACED_ENVIRONMENT_VARIABLES = frozenset({"PATH", "PYTHONPATH"})
_STABLE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class CurrentReleaseReceiptError(ValueError):
    """The current-installed receipt request or generated evidence is invalid."""

    def __init__(self, errors: str | Sequence[str]):
        self.errors = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("; ".join(self.errors))


@dataclass(frozen=True)
class CurrentRuntimeInputs:
    """Canonical external inputs observed by one ordinary current receipt."""

    habitat_runtime_prefix: Path
    rlr_sdk_root: Path
    scene_data_root: Path
    magnum_python_site: Path

    def as_document(self) -> dict[str, str]:
        return {
            "habitat_runtime_prefix": str(self.habitat_runtime_prefix),
            "rlr_sdk_root": str(self.rlr_sdk_root),
            "scene_data_root": str(self.scene_data_root),
            "magnum_python_site": str(self.magnum_python_site),
        }


@dataclass(frozen=True)
class CurrentTestReceiptExecution:
    path: Path
    receipt: dict[str, Any]
    exit_code: int


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def current_receipt_schema_path() -> Path:
    source = _repository_root() / "schemas" / CURRENT_TEST_EXECUTION_RECEIPT_SCHEMA_FILE
    if source.is_file():
        return source
    installed = (
        Path(sys.prefix)
        / "share"
        / "avengine"
        / "schemas"
        / CURRENT_TEST_EXECUTION_RECEIPT_SCHEMA_FILE
    )
    if installed.is_file():
        return installed
    raise CurrentReleaseReceiptError(
        f"current release receipt schema is unavailable: {source}"
    )


def _schema_errors(value: Mapping[str, Any]) -> list[str]:
    try:
        schema = json.loads(current_receipt_schema_path().read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
    except (OSError, UnicodeError, ValueError) as exc:
        return [f"could not load current release receipt schema: {exc}"]
    validator = Draft202012Validator(schema)
    return [
        f"{'.'.join(str(part) for part in error.absolute_path) or '$'}: "
        f"{error.message}"
        for error in sorted(
            validator.iter_errors(dict(value)),
            key=lambda item: tuple(str(part) for part in item.absolute_path),
        )
    ]


def validate_current_receipt_document(value: Mapping[str, Any]) -> list[str]:
    """Return deterministic schema errors for a v2 current receipt."""

    return _schema_errors(value)


def _require_stable_id(value: str, *, owner: str) -> str:
    if not isinstance(value, str) or _STABLE_ID.fullmatch(value) is None:
        raise CurrentReleaseReceiptError(
            f"{owner} is not a stable lowercase identifier"
        )
    return value


def _repository_relative_path(value: str, *, owner: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise CurrentReleaseReceiptError(
            f"{owner} must be a POSIX repository-relative path"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise CurrentReleaseReceiptError(
            f"{owner} must be a normalized repository-relative path"
        )
    return path


def _current_git_environment() -> dict[str, str]:
    """Use system Git without caller-selected repository or loader state."""

    environment = dict(os.environ)
    for variable in LEGACY_CHILD_ENVIRONMENT_VARIABLES:
        environment.pop(variable, None)
    for variable in tuple(environment):
        if variable.startswith("GIT_"):
            environment.pop(variable, None)
    environment["PATH"] = os.defpath
    return environment


def _git(root: Path, *arguments: str, owner: str) -> str:
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
        raise CurrentReleaseReceiptError(f"could not inspect {owner}: {exc}") from exc
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "git failed"
        raise CurrentReleaseReceiptError(f"could not inspect {owner}: {message}")
    return completed.stdout.strip()


def _commit(root: Path, *, owner: str) -> str:
    value = _git(root, "rev-parse", "HEAD", owner=owner)
    if _COMMIT.fullmatch(value) is None:
        raise CurrentReleaseReceiptError(f"{owner} HEAD is not a full Git commit")
    return value


def _require_clean_worktree(root: Path) -> None:
    status = _git(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        owner="AVEngine workspace worktree",
    )
    if status:
        raise CurrentReleaseReceiptError(
            "AVEngine workspace must have no tracked changes or non-ignored "
            f"untracked entries for a commit-bound receipt: {status!r}"
        )


def _current_child_index_paths(root: Path) -> tuple[PurePosixPath, ...]:
    """Return only regular, non-temporary files from the clean Git index."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--stage", "-z"],
            check=False,
            capture_output=True,
            env=_current_git_environment(),
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CurrentReleaseReceiptError(
            f"could not inspect current child source index: {exc}"
        ) from exc
    if completed.returncode != 0:
        message = (
            completed.stderr.decode("utf-8", errors="replace").strip()
            or completed.stdout.decode("utf-8", errors="replace").strip()
            or "git ls-files failed"
        )
        raise CurrentReleaseReceiptError(
            f"could not inspect current child source index: {message}"
        )

    paths: list[PurePosixPath] = []
    for record in completed.stdout.split(b"\0"):
        if not record:
            continue
        metadata, separator, raw_path = record.partition(b"\t")
        fields = metadata.split()
        if not separator or len(fields) != 3 or not raw_path:
            raise CurrentReleaseReceiptError(
                "current child source index contains an invalid entry"
            )
        mode, _object_id, stage = fields
        try:
            relative = _repository_relative_path(
                os.fsdecode(raw_path),
                owner="current child source index path",
            )
        except UnicodeError as exc:
            raise CurrentReleaseReceiptError(
                "current child source index contains a non-decodable path"
            ) from exc
        if relative.parts[:1] == ("tmp",):
            raise CurrentReleaseReceiptError(
                "current child source snapshot must not retain tracked logical "
                f"tmp content: {relative.as_posix()}"
            )
        if stage != b"0":
            raise CurrentReleaseReceiptError(
                "current child source index must contain only resolved stage-0 "
                f"entries: {relative.as_posix()}"
            )
        if mode == b"120000":
            raise CurrentReleaseReceiptError(
                "current child source snapshot must not retain a symlink: "
                f"{relative.as_posix()}"
            )
        if mode != b"100644" and mode != b"100755":
            raise CurrentReleaseReceiptError(
                "current child source snapshot must contain only regular files: "
                f"{relative.as_posix()}"
            )
        paths.append(relative)
    return tuple(paths)


def _materialize_current_child_index(root: Path, destination: Path) -> None:
    """Populate *destination* from index blobs, never working-tree byproducts."""

    _current_child_index_paths(root)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "checkout-index",
                "--all",
                f"--prefix={destination}{os.sep}",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=_current_git_environment(),
            timeout=60.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CurrentReleaseReceiptError(
            f"could not materialize current child source index: {exc}"
        ) from exc
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "git checkout-index failed"
        raise CurrentReleaseReceiptError(
            f"could not materialize current child source index: {message}"
        )
    unexpected = next(
        (path for path in destination.rglob("*") if path.is_symlink()),
        None,
    )
    if unexpected is not None:
        raise CurrentReleaseReceiptError(
            "current child source snapshot must not retain a symlink: "
            f"{unexpected}"
        )


def _workspace_policy(root: Path) -> WorkspacePathPolicy:
    """Allow AVEngine's declared tmp compatibility target, and nothing else."""

    roots: list[Path] = [root]
    tmp_link = root / "tmp"
    if tmp_link.is_symlink():
        try:
            tmp_root = tmp_link.resolve(strict=True)
        except OSError as exc:
            raise CurrentReleaseReceiptError(
                f"AVEngine tmp compatibility root cannot be resolved: {exc}"
            ) from exc
        if not tmp_root.is_dir():
            raise CurrentReleaseReceiptError(
                f"AVEngine tmp compatibility root is not a directory: {tmp_root}"
            )
        roots.append(tmp_root)
    return WorkspacePathPolicy.from_roots(roots)


def _path_is_git_ignored(root: Path, relative: PurePosixPath) -> bool:
    """Check the logical root without traversing an allowed tmp symlink."""

    logical_root = relative.parts[0]
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "check-ignore",
                "--quiet",
                "--no-index",
                "--",
                logical_root,
            ],
            check=False,
            capture_output=True,
            text=True,
            env=_current_git_environment(),
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CurrentReleaseReceiptError(
            f"could not inspect ignored tmp path: {exc}"
        ) from exc
    if completed.returncode == 0:
        return True
    if completed.returncode == 1:
        return False
    message = completed.stderr.strip() or completed.stdout.strip() or "git failed"
    raise CurrentReleaseReceiptError(
        f"could not inspect ignored tmp path: {message}"
    )


def logical_current_tmp_path(
    root: Path,
    value: str | Path,
    *,
    owner: str,
) -> PurePosixPath:
    """Normalize one input to the declared, Git-ignored logical tmp root."""

    workspace = root.resolve(strict=True)
    requested = Path(value)
    if requested.is_absolute():
        try:
            relative = requested.relative_to(workspace)
        except ValueError:
            tmp_link = workspace / "tmp"
            try:
                relative = Path("tmp") / requested.resolve(strict=False).relative_to(
                    tmp_link.resolve(strict=False)
                )
            except (OSError, ValueError) as exc:
                raise CurrentReleaseReceiptError(
                    f"{owner} must be under the logical ignored tmp root"
                ) from exc
    else:
        relative = requested
    try:
        normalized = _repository_relative_path(str(relative), owner=owner)
    except CurrentReleaseReceiptError:
        raise
    if normalized.parts[:1] != ("tmp",) or len(normalized.parts) < 2:
        raise CurrentReleaseReceiptError(
            f"{owner} must be under the logical ignored tmp root"
        )
    if not _path_is_git_ignored(workspace, normalized):
        raise CurrentReleaseReceiptError(
            f"{owner} must be Git-ignored beneath the logical tmp root: "
            f"{normalized.as_posix()}"
        )
    return normalized


def display_current_workspace_path(root: Path, path: Path) -> str:
    """Prefer AVEngine's logical tmp path over its resolved external target."""

    tmp_link = root / "tmp"
    if tmp_link.is_symlink():
        try:
            relative = path.resolve(strict=True).relative_to(
                tmp_link.resolve(strict=True)
            )
        except (OSError, ValueError):
            pass
        else:
            return (Path("tmp") / relative).as_posix()
    try:
        return path.resolve(strict=True).relative_to(root).as_posix()
    except (OSError, ValueError):
        return str(path)


@contextmanager
def _isolated_current_child_root(root: Path) -> Iterator[Path]:
    """Yield a fresh source snapshot whose parent has no legacy sibling.

    ``discover_runtime_root()`` in the retained v1 compatibility layer once
    derived a sibling fallback from the imported module's ``__file__``; that
    discovery is retired (a Git-checkout root now fails closed), and this
    isolation is kept as defense in depth for the ordinary child.
    The current-v2 child instead imports from a disposable snapshot materialized
    exactly from the clean Git index. Ignored and untracked source, caches and
    local startup hooks never enter that snapshot. Its randomly created parent
    is checked to have no such sibling. ``tmp`` is the sole shared path so the
    declared JUnit remains under the original logical, Git-ignored evidence
    root.
    """

    with tempfile.TemporaryDirectory(prefix="avengine-current-release-child-") as value:
        container = Path(value)
        execution_root = container / "workspace"
        try:
            _materialize_current_child_index(root, execution_root)
            source_tmp = root / "tmp"
            if not source_tmp.is_dir():
                raise CurrentReleaseReceiptError(
                    "current child source root has no logical tmp directory"
                )
            child_tmp = execution_root / "tmp"
            child_tmp.symlink_to(source_tmp, target_is_directory=True)
        except OSError as exc:
            raise CurrentReleaseReceiptError(
                f"could not create isolated current child source root: {exc}"
            ) from exc
        sibling = execution_root.parent / "habitat-sim-AVEngine"
        if sibling.exists() or sibling.is_symlink():
            raise CurrentReleaseReceiptError(
                "isolated current child parent unexpectedly contains a legacy "
                f"Habitat checkout candidate: {sibling}"
            )
        yield execution_root


def _current_child_environment(*, execution_root: Path) -> dict[str, str]:
    """Drop inherited selectors and replace lookup paths with fixed values."""

    environment = dict(os.environ)
    for variable in LEGACY_CHILD_ENVIRONMENT_VARIABLES:
        environment.pop(variable, None)
    for variable in tuple(environment):
        if variable.startswith("GIT_"):
            environment.pop(variable, None)
    environment["PATH"] = os.defpath
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(execution_root / "src"), str(execution_root))
    )
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONSAFEPATH"] = "1"
    return environment


def _normalize_current_test_command(command: Sequence[str]) -> list[str]:
    """Record and execute one canonical non-checkout executable argv.

    A bare command would be resolved through a caller-controlled ``PATH``.
    Current receipts must instead name the actual absolute executable that ran.
    The child receives a fixed system PATH too, so a script's interpreter
    launcher cannot recover a legacy caller path.
    """

    selected = list(command)
    if not selected or any(
        not isinstance(value, str) or not value for value in selected
    ):
        raise CurrentReleaseReceiptError("command must contain nonempty argv strings")
    raw_executable = Path(selected[0])
    if not raw_executable.is_absolute():
        raise CurrentReleaseReceiptError(
            "command executable must be an absolute path, never a PATH-resolved "
            f"name: {selected[0]!r}"
        )
    try:
        executable = raw_executable.resolve(strict=True)
    except OSError as exc:
        raise CurrentReleaseReceiptError(
            f"command executable cannot be resolved: {raw_executable}: {exc}"
        ) from exc
    try:
        mode = executable.stat().st_mode
    except OSError as exc:
        raise CurrentReleaseReceiptError(
            f"command executable cannot be inspected: {executable}: {exc}"
        ) from exc
    if not stat.S_ISREG(mode) or not os.access(executable, os.X_OK):
        raise CurrentReleaseReceiptError(
            f"command executable must be a regular executable file: {executable}"
        )
    try:
        executable = require_outside_git_checkout(
            executable,
            owner="command executable",
        )
    except (ExternalRlrSdkError, OSError, RuntimeError) as exc:
        raise CurrentReleaseReceiptError(str(exc)) from exc
    selected[0] = str(executable)
    return selected


def _require_external_directory(value: str | Path, *, owner: str) -> Path:
    """Resolve one explicit external root without a checkout or symlink hop."""

    text = str(value)
    raw = Path(text)
    if not text or not raw.is_absolute():
        raise CurrentReleaseReceiptError(f"{owner} must be an absolute path")
    cursor = raw
    while True:
        if cursor.is_symlink():
            raise CurrentReleaseReceiptError(
                f"{owner} must not traverse a symlink: {raw}"
            )
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise CurrentReleaseReceiptError(f"{owner} cannot be resolved: {exc}") from exc
    if str(raw) != str(resolved):
        raise CurrentReleaseReceiptError(
            f"{owner} must use its canonical path without indirection: {raw}"
        )
    if not resolved.is_dir():
        raise CurrentReleaseReceiptError(f"{owner} is not a directory: {resolved}")
    try:
        return require_outside_git_checkout(resolved, owner=owner)
    except (ExternalRlrSdkError, OSError, RuntimeError) as exc:
        raise CurrentReleaseReceiptError(str(exc)) from exc


def validate_current_runtime_inputs(
    *,
    runtime_prefix: str | Path,
    rlr_sdk_root: str | Path,
    scene_data_root: str | Path,
    magnum_python_site: str | Path,
) -> CurrentRuntimeInputs:
    """Validate explicit, non-checkout current-installed inputs.

    This is deliberately topology-only. It proves that the requested roots are
    real, canonical, external inputs and that the SDK has the expected
    header/library layout. It does not import Habitat, load RLR, or claim the
    adapter is enabled; those are native-evidence concerns.
    """

    prefix = _require_external_directory(
        runtime_prefix,
        owner="--runtime-prefix",
    )
    sdk_root = _require_external_directory(
        rlr_sdk_root,
        owner="--rlr-sdk-root",
    )
    data_root = _require_external_directory(
        scene_data_root,
        owner="--scene-data-root",
    )
    magnum_site = _require_external_directory(
        magnum_python_site,
        owner="--magnum-python-site",
    )
    try:
        sdk = discover_external_rlr_sdk(sdk_root)
    except ExternalRlrSdkError as exc:
        raise CurrentReleaseReceiptError(str(exc)) from exc
    if sdk.root != sdk_root:
        raise CurrentReleaseReceiptError(
            "--rlr-sdk-root does not resolve to the discovered SDK root"
        )
    return CurrentRuntimeInputs(
        habitat_runtime_prefix=prefix,
        rlr_sdk_root=sdk.root,
        scene_data_root=data_root,
        magnum_python_site=magnum_site,
    )


def _command_references_junit(command: Sequence[str], declared_path: str) -> bool:
    if declared_path in command:
        return True
    accepted = {
        f"--junitxml={declared_path}",
        f"--junit-xml={declared_path}",
    }
    return any(value in accepted for value in command)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def derive_current_junit_totals(raw_xml: bytes) -> dict[str, int]:
    """Derive test totals from JUnit testcase elements, never suite counters."""

    if not raw_xml:
        raise CurrentReleaseReceiptError("JUnit XML is empty")
    try:
        root = ET.fromstring(raw_xml)
    except ET.ParseError as exc:
        raise CurrentReleaseReceiptError(f"JUnit XML is malformed: {exc}") from exc
    if _local_name(root.tag) not in {"testsuite", "testsuites"}:
        raise CurrentReleaseReceiptError(
            "JUnit XML root must be testsuite or testsuites"
        )
    totals = {"executed": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for testcase in (
        element for element in root.iter() if _local_name(element.tag) == "testcase"
    ):
        outcomes = [
            _local_name(child.tag)
            for child in testcase
            if _local_name(child.tag) in {"failure", "error", "skipped"}
        ]
        if len(outcomes) > 1:
            raise CurrentReleaseReceiptError(
                "JUnit testcase has more than one failure/error/skipped outcome"
            )
        totals["executed"] += 1
        if not outcomes:
            totals["passed"] += 1
        elif outcomes[0] == "failure":
            totals["failed"] += 1
        elif outcomes[0] == "error":
            totals["errors"] += 1
        else:
            totals["skipped"] += 1
    if totals["executed"] < 1:
        raise CurrentReleaseReceiptError("JUnit XML contains no testcase elements")
    return totals


def _derived_status(exit_code: int, totals: Mapping[str, int]) -> str:
    if (
        exit_code == 0
        and totals["passed"] >= 1
        and totals["failed"] == 0
        and totals["errors"] == 0
    ):
        return "pass"
    return "fail"


def _decode_base64(value: Any, *, owner: str, allow_empty: bool) -> bytes:
    if not isinstance(value, str):
        raise CurrentReleaseReceiptError(f"{owner} must be a base64 string")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise CurrentReleaseReceiptError(f"{owner} is not canonical base64") from exc
    if not allow_empty and not decoded:
        raise CurrentReleaseReceiptError(f"{owner} decodes to empty bytes")
    if base64.b64encode(decoded).decode("ascii") != value:
        raise CurrentReleaseReceiptError(f"{owner} is not canonical base64")
    return decoded


def verify_current_receipt_payload(document: Mapping[str, Any]) -> list[str]:
    """Re-derive current receipt test semantics without reading a checkout."""

    errors = validate_current_receipt_document(document)
    if errors:
        return errors
    try:
        command = document["command"]
        normalized_command = _normalize_current_test_command(command)
        if list(command) != normalized_command:
            raise CurrentReleaseReceiptError(
                "recorded command executable must use its canonical absolute "
                "non-checkout path"
            )
        declared_path = document["junit_xml"]["declared_path"]
        declared_relative = _repository_relative_path(
            declared_path,
            owner="junit_xml.declared_path",
        )
        if (
            declared_relative.parts[:1] != ("tmp",)
            or len(declared_relative.parts) < 2
        ):
            raise CurrentReleaseReceiptError(
                "junit_xml.declared_path must remain under logical tmp/"
            )
        if not _command_references_junit(command, declared_path):
            raise CurrentReleaseReceiptError(
                "recorded command does not reference junit_xml.declared_path"
            )
        _decode_base64(
            document["captured_output"]["stdout_base64"],
            owner="captured_output.stdout_base64",
            allow_empty=True,
        )
        _decode_base64(
            document["captured_output"]["stderr_base64"],
            owner="captured_output.stderr_base64",
            allow_empty=True,
        )
        junit = _decode_base64(
            document["junit_xml"]["raw_bytes_base64"],
            owner="junit_xml.raw_bytes_base64",
            allow_empty=False,
        )
        totals = derive_current_junit_totals(junit)
        if document["result_totals"] != totals:
            raise CurrentReleaseReceiptError(
                "result_totals differ from embedded JUnit testcase outcomes: "
                f"expected {totals}, got {document['result_totals']}"
            )
        status = _derived_status(document["exit_code"], totals)
        if document["status"] != status:
            raise CurrentReleaseReceiptError(
                "status differs from exit_code and embedded JUnit outcomes: "
                f"expected {status}, got {document['status']}"
            )
    except (KeyError, TypeError, CurrentReleaseReceiptError) as exc:
        if isinstance(exc, CurrentReleaseReceiptError):
            errors.extend(exc.errors)
        else:
            errors.append(f"receipt payload structure is invalid: {exc}")
    return errors


def execute_current_test_receipt(
    output_path: str | Path,
    *,
    workspace_root: str | Path,
    runtime_prefix: str | Path,
    rlr_sdk_root: str | Path,
    scene_data_root: str | Path,
    magnum_python_site: str | Path,
    receipt_id: str,
    test_layer_id: str,
    junit_xml: str,
    command: Sequence[str],
) -> CurrentTestReceiptExecution:
    """Execute one argv and publish an immutable ordinary current receipt."""

    root = Path(workspace_root).resolve(strict=True)
    if not root.is_dir():
        raise CurrentReleaseReceiptError("workspace root must be a directory")
    runtime = validate_current_runtime_inputs(
        runtime_prefix=runtime_prefix,
        rlr_sdk_root=rlr_sdk_root,
        scene_data_root=scene_data_root,
        magnum_python_site=magnum_python_site,
    )
    selected_receipt_id = _require_stable_id(receipt_id, owner="receipt_id")
    selected_layer_id = _require_stable_id(test_layer_id, owner="test_layer_id")
    if selected_layer_id not in CURRENT_EXECUTABLE_LAYERS:
        raise CurrentReleaseReceiptError(
            "test_layer_id must name a non-release executable layer"
        )
    selected_command = _normalize_current_test_command(command)
    declared_junit = logical_current_tmp_path(
        root,
        junit_xml,
        owner="junit_xml",
    )
    declared_junit_text = declared_junit.as_posix()
    if not _command_references_junit(selected_command, declared_junit_text):
        raise CurrentReleaseReceiptError(
            "execution command must reference the declared --junit-xml path"
        )

    policy = _workspace_policy(root)
    logical_output = logical_current_tmp_path(
        root,
        output_path,
        owner="current test execution receipt",
    )
    receipt_target = policy.resolve_output(
        root / logical_output,
        owner="current test execution receipt",
        create_parent=True,
    )
    junit_target = policy.resolve_output(
        root / declared_junit,
        owner="command-generated JUnit XML",
        create_parent=True,
    )
    if receipt_target == junit_target:
        raise CurrentReleaseReceiptError("receipt output and JUnit XML must differ")

    _require_clean_worktree(root)
    implementation_commit = _commit(root, owner="AVEngine workspace")
    with _isolated_current_child_root(root) as execution_root:
        try:
            completed = subprocess.run(
                selected_command,
                cwd=execution_root,
                check=False,
                capture_output=True,
                env=_current_child_environment(execution_root=execution_root),
            )
        except OSError as exc:
            raise CurrentReleaseReceiptError(
                f"could not execute test command: {exc}"
            ) from exc
    _require_clean_worktree(root)
    if _commit(root, owner="AVEngine workspace") != implementation_commit:
        raise CurrentReleaseReceiptError("test command changed the AVEngine HEAD commit")
    runtime_after = validate_current_runtime_inputs(
        runtime_prefix=runtime_prefix,
        rlr_sdk_root=rlr_sdk_root,
        scene_data_root=scene_data_root,
        magnum_python_site=magnum_python_site,
    )
    if runtime_after != runtime:
        raise CurrentReleaseReceiptError(
            "explicit current-installed runtime inputs changed during test execution"
        )
    if not junit_target.exists():
        raise CurrentReleaseReceiptError(
            "test command did not generate the declared JUnit XML"
        )
    if junit_target.is_symlink() or not junit_target.is_file():
        raise CurrentReleaseReceiptError(
            "generated JUnit XML must be a regular non-symlink file"
        )
    try:
        junit_bytes = junit_target.read_bytes()
    except OSError as exc:
        raise CurrentReleaseReceiptError(
            f"could not read generated JUnit XML: {exc}"
        ) from exc
    totals = derive_current_junit_totals(junit_bytes)
    status = _derived_status(completed.returncode, totals)
    document: dict[str, Any] = {
        "schema": CURRENT_TEST_EXECUTION_RECEIPT_SCHEMA,
        "receipt_id": selected_receipt_id,
        "test_layer_id": selected_layer_id,
        "status": status,
        "claim_scope": "ordinary_current_candidate",
        "runtime_inputs": runtime.as_document(),
        "runtime_observation": {
            "mode": "path_topology_only",
            "native_rlr_execution": False,
            "formal_release_status": "not_run",
        },
        "command": selected_command,
        "execution_cwd": ".",
        "exit_code": completed.returncode,
        "implementation_commit": implementation_commit,
        "captured_output": {
            "encoding": "base64",
            "stdout_base64": base64.b64encode(completed.stdout).decode("ascii"),
            "stderr_base64": base64.b64encode(completed.stderr).decode("ascii"),
        },
        "junit_xml": {
            "declared_path": declared_junit_text,
            "encoding": "base64",
            "raw_bytes_base64": base64.b64encode(junit_bytes).decode("ascii"),
        },
        "result_totals": totals,
    }
    errors = verify_current_receipt_payload(document)
    if errors:
        raise CurrentReleaseReceiptError(errors)
    payload = (
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    published = write_bytes_no_clobber(policy, receipt_target, payload)
    try:
        readback = json.loads(published.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise CurrentReleaseReceiptError(
            f"could not read back published receipt: {exc}"
        ) from exc
    readback_errors = verify_current_receipt_payload(readback)
    if readback_errors:
        raise CurrentReleaseReceiptError(readback_errors)
    try:
        junit_target.unlink()
    except OSError as exc:
        raise CurrentReleaseReceiptError(
            "receipt was published but embedded JUnit could not be removed: "
            f"{exc}"
        ) from exc
    return CurrentTestReceiptExecution(published, readback, completed.returncode)


__all__ = [
    "CURRENT_EXECUTABLE_LAYERS",
    "CURRENT_TEST_EXECUTION_RECEIPT_SCHEMA",
    "CurrentReleaseReceiptError",
    "CurrentRuntimeInputs",
    "CurrentTestReceiptExecution",
    "current_receipt_schema_path",
    "derive_current_junit_totals",
    "display_current_workspace_path",
    "execute_current_test_receipt",
    "LEGACY_CHILD_ENVIRONMENT_VARIABLES",
    "logical_current_tmp_path",
    "validate_current_receipt_document",
    "validate_current_runtime_inputs",
    "verify_current_receipt_payload",
]
