"""Execute one test command and publish a self-derived M6 receipt.

The caller supplies only identity, layer, repository locations, a fresh JUnit
path and the argv to execute.  Exit status, repository commits and test totals
are observations made by this module, never caller assertions.  Raw JUnit,
stdout and stderr bytes are embedded so a verifier can independently derive
the receipt semantics without extra leaf artifacts.

This is evidence for the declared trusted research workspace.  It rejects
tracked changes and non-ignored untracked entries, but it is not a
cryptographic attestation against an operator who controls the workspace and
can fabricate a JSON file.  Adversarial provenance requires an external
signed CI/OIDC attestation.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

from jsonschema import Draft202012Validator

from avengine.security.path_policy import (
    WorkspacePathPolicy,
    write_bytes_no_clobber,
)


RECEIPT_SCHEMA = "avengine_m6_test_execution_receipt_v1"
RECEIPT_SCHEMA_FILE = "m6_test_execution_receipt_v1.schema.json"
EXECUTABLE_LAYERS = (
    "fast-unit",
    "slow-hermetic",
    "native-habitat",
    "rlr-audio",
    "blender-assets",
    "media-readback",
)
DEFAULT_RLR_SUBMODULE_PATH = "src/deps/rlr-audio-propagation"

_STABLE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class TestReceiptError(ValueError):
    """The execution request, generated JUnit, or receipt is invalid."""

    def __init__(self, errors: str | Sequence[str]):
        self.errors = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("; ".join(self.errors))


@dataclass(frozen=True)
class TestReceiptExecution:
    path: Path
    receipt: dict[str, Any]
    exit_code: int


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _schema_path() -> Path:
    source = _repository_root() / "schemas" / RECEIPT_SCHEMA_FILE
    if source.is_file():
        return source
    installed = Path(os.sys.prefix) / "share" / "avengine" / "schemas" / RECEIPT_SCHEMA_FILE
    if installed.is_file():
        return installed
    raise TestReceiptError(f"test execution receipt schema is unavailable: {source}")


def _schema_errors(value: Mapping[str, Any]) -> list[str]:
    try:
        schema = json.loads(_schema_path().read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
    except (OSError, UnicodeError, ValueError) as exc:
        return [f"could not load test execution receipt schema: {exc}"]
    validator = Draft202012Validator(schema)
    return [
        f"{'.'.join(str(part) for part in error.absolute_path) or '$'}: "
        f"{error.message}"
        for error in sorted(
            validator.iter_errors(dict(value)),
            key=lambda item: tuple(str(part) for part in item.absolute_path),
        )
    ]


def _require_stable_id(value: str, *, owner: str) -> str:
    if not isinstance(value, str) or _STABLE_ID.fullmatch(value) is None:
        raise TestReceiptError(f"{owner} is not a stable lowercase identifier")
    return value


def _repository_relative_path(value: str, *, owner: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise TestReceiptError(f"{owner} must be a POSIX repository-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise TestReceiptError(f"{owner} must be a normalized repository-relative path")
    return path


def _git(root: Path, *arguments: str, owner: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TestReceiptError(f"could not inspect {owner}: {exc}") from exc
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "git failed"
        raise TestReceiptError(f"could not inspect {owner}: {message}")
    return completed.stdout.strip()


def _commit(root: Path, *, owner: str) -> str:
    value = _git(root, "rev-parse", "HEAD", owner=owner)
    if _COMMIT.fullmatch(value) is None:
        raise TestReceiptError(f"{owner} HEAD is not a full Git commit")
    return value


def _require_clean_worktree(root: Path, *, owner: str) -> None:
    status = _git(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        owner=f"{owner} worktree",
    )
    if status:
        raise TestReceiptError(
            f"{owner} worktree must have no tracked changes or non-ignored "
            f"untracked entries for a commit-bound receipt: {status!r}"
        )


def _rlr_commit(habitat_root: Path, submodule_path: str) -> str:
    relative = _repository_relative_path(submodule_path, owner="rlr_submodule_path")
    value = _git(
        habitat_root,
        "rev-parse",
        f"HEAD:{relative.as_posix()}",
        owner="Habitat RLR gitlink",
    )
    if _COMMIT.fullmatch(value) is None:
        raise TestReceiptError("Habitat RLR gitlink is not a full Git commit")
    return value


def _rlr_checkout(habitat_root: Path, submodule_path: str) -> Path:
    relative = _repository_relative_path(submodule_path, owner="rlr_submodule_path")
    try:
        checkout = (habitat_root / relative).resolve(strict=True)
        checkout.relative_to(habitat_root)
    except (OSError, ValueError) as exc:
        raise TestReceiptError(f"Habitat RLR checkout is unavailable: {exc}") from exc
    if not checkout.is_dir():
        raise TestReceiptError("Habitat RLR checkout is not a directory")
    return checkout


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


def derive_junit_totals(raw_xml: bytes) -> dict[str, int]:
    """Derive exact outcome totals from testcase elements, not suite counters."""

    if not raw_xml:
        raise TestReceiptError("JUnit XML is empty")
    try:
        root = ET.fromstring(raw_xml)
    except ET.ParseError as exc:
        raise TestReceiptError(f"JUnit XML is malformed: {exc}") from exc
    if _local_name(root.tag) not in {"testsuite", "testsuites"}:
        raise TestReceiptError("JUnit XML root must be testsuite or testsuites")
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
            raise TestReceiptError(
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
        raise TestReceiptError("JUnit XML contains no testcase elements")
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
        raise TestReceiptError(f"{owner} must be a base64 string")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise TestReceiptError(f"{owner} is not canonical base64") from exc
    if not allow_empty and not decoded:
        raise TestReceiptError(f"{owner} decodes to empty bytes")
    if base64.b64encode(decoded).decode("ascii") != value:
        raise TestReceiptError(f"{owner} is not canonical base64")
    return decoded


def verify_receipt_payload(document: Mapping[str, Any]) -> list[str]:
    """Re-decode embedded bytes and recompute totals/status from raw JUnit."""

    errors = _schema_errors(document)
    if errors:
        return errors
    try:
        command = document["command"]
        declared_path = document["junit_xml"]["declared_path"]
        _repository_relative_path(declared_path, owner="junit_xml.declared_path")
        if not _command_references_junit(command, declared_path):
            raise TestReceiptError(
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
        derived_totals = derive_junit_totals(junit)
        if document["result_totals"] != derived_totals:
            raise TestReceiptError(
                "result_totals differ from embedded JUnit testcase outcomes: "
                f"expected {derived_totals}, got {document['result_totals']}"
            )
        derived_status = _derived_status(document["exit_code"], derived_totals)
        if document["status"] != derived_status:
            raise TestReceiptError(
                "status differs from exit_code and embedded JUnit outcomes: "
                f"expected {derived_status}, got {document['status']}"
            )
    except (KeyError, TypeError, TestReceiptError) as exc:
        if isinstance(exc, TestReceiptError):
            errors.extend(exc.errors)
        else:
            errors.append(f"receipt payload structure is invalid: {exc}")
    return errors


def execute_test_receipt(
    output_path: str | Path,
    *,
    workspace_root: str | Path,
    habitat_runtime_root: str | Path,
    receipt_id: str,
    test_layer_id: str,
    junit_xml: str,
    command: Sequence[str],
    rlr_submodule_path: str = DEFAULT_RLR_SUBMODULE_PATH,
) -> TestReceiptExecution:
    """Execute argv directly and atomically publish a self-derived receipt."""

    root = Path(workspace_root).resolve(strict=True)
    habitat = Path(habitat_runtime_root).resolve(strict=True)
    if not root.is_dir() or not habitat.is_dir():
        raise TestReceiptError("workspace and Habitat runtime roots must be directories")
    selected_receipt_id = _require_stable_id(receipt_id, owner="receipt_id")
    selected_layer_id = _require_stable_id(test_layer_id, owner="test_layer_id")
    if selected_layer_id not in EXECUTABLE_LAYERS:
        raise TestReceiptError("test_layer_id must name a non-release executable layer")
    selected_command = list(command)
    if not selected_command or any(
        not isinstance(value, str) or not value for value in selected_command
    ):
        raise TestReceiptError("command must contain nonempty argv strings")
    declared_junit = _repository_relative_path(junit_xml, owner="junit_xml")
    declared_junit_text = declared_junit.as_posix()
    if not _command_references_junit(selected_command, declared_junit_text):
        raise TestReceiptError(
            "execution command must reference the declared --junit-xml path"
        )

    policy = WorkspacePathPolicy.from_roots([root])
    # Preflight receipt first so an expensive command never runs when the
    # immutable output name is already occupied.
    receipt_target = policy.resolve_output(
        output_path, owner="test execution receipt", create_parent=True
    )
    junit_target = policy.resolve_output(
        root / declared_junit,
        owner="command-generated JUnit XML",
        create_parent=True,
    )
    if receipt_target == junit_target:
        raise TestReceiptError("receipt output and JUnit XML must be different paths")

    _require_clean_worktree(root, owner="AVEngine workspace")
    _require_clean_worktree(habitat, owner="Habitat runtime")
    rlr_checkout = _rlr_checkout(habitat, rlr_submodule_path)
    _require_clean_worktree(rlr_checkout, owner="RLR checkout")
    implementation_commit = _commit(root, owner="AVEngine workspace")
    habitat_commit = _commit(habitat, owner="Habitat runtime")
    rlr_commit = _rlr_commit(habitat, rlr_submodule_path)
    if _commit(rlr_checkout, owner="RLR checkout") != rlr_commit:
        raise TestReceiptError("RLR checkout HEAD differs from the Habitat gitlink")
    try:
        completed = subprocess.run(
            selected_command,
            cwd=root,
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise TestReceiptError(f"could not execute test command: {exc}") from exc

    _require_clean_worktree(root, owner="AVEngine workspace")
    _require_clean_worktree(habitat, owner="Habitat runtime")
    _require_clean_worktree(rlr_checkout, owner="RLR checkout")
    if _commit(root, owner="AVEngine workspace") != implementation_commit:
        raise TestReceiptError("test command changed the AVEngine HEAD commit")
    if _commit(habitat, owner="Habitat runtime") != habitat_commit:
        raise TestReceiptError("test command changed the Habitat HEAD commit")
    if _rlr_commit(habitat, rlr_submodule_path) != rlr_commit:
        raise TestReceiptError("test command changed the Habitat RLR gitlink commit")
    if _commit(rlr_checkout, owner="RLR checkout") != rlr_commit:
        raise TestReceiptError("test command changed the RLR checkout commit")
    if not os.path.lexists(junit_target):
        raise TestReceiptError("test command did not generate the declared JUnit XML")
    if junit_target.is_symlink() or not junit_target.is_file():
        raise TestReceiptError("generated JUnit XML must be a regular non-symlink file")
    try:
        junit_bytes = junit_target.read_bytes()
    except OSError as exc:
        raise TestReceiptError(f"could not read generated JUnit XML: {exc}") from exc
    totals = derive_junit_totals(junit_bytes)
    status = _derived_status(completed.returncode, totals)
    document: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "receipt_id": selected_receipt_id,
        "test_layer_id": selected_layer_id,
        "status": status,
        "command": selected_command,
        "execution_cwd": ".",
        "exit_code": completed.returncode,
        "implementation_commit": implementation_commit,
        "habitat_runtime_commit": habitat_commit,
        "rlr_commit": rlr_commit,
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
    errors = verify_receipt_payload(document)
    if errors:
        raise TestReceiptError(errors)
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
        raise TestReceiptError(f"could not read back published receipt: {exc}") from exc
    readback_errors = verify_receipt_payload(readback)
    if readback_errors:
        raise TestReceiptError(readback_errors)
    try:
        junit_target.unlink()
    except OSError as exc:
        raise TestReceiptError(
            f"receipt was published but embedded JUnit could not be removed: {exc}"
        ) from exc
    return TestReceiptExecution(published, readback, completed.returncode)


__all__ = [
    "DEFAULT_RLR_SUBMODULE_PATH",
    "EXECUTABLE_LAYERS",
    "TestReceiptError",
    "TestReceiptExecution",
    "derive_junit_totals",
    "execute_test_receipt",
    "verify_receipt_payload",
]
