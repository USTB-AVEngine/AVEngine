from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys

import pytest

from avengine.release import load_json_strict, validate_test_execution_receipt_document
from avengine.release_receipt import derive_junit_totals, verify_receipt_payload
from tools.release.build_manifest import main as release_tool_main


RLR_URL = "https://github.com/facebookresearch/rlr-audio-propagation.git"
HABITAT_URL = "https://github.com/Eastforward/habitat-sim-AVEngine.git"
AVENGINE_URL = "https://github.com/Eastforward/AVEngine.git"


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _init_repository(path: Path, *, origin: str) -> None:
    path.mkdir(parents=True)
    _git(path, "init", "--quiet")
    _git(path, "config", "user.name", "receipt fixture")
    _git(path, "config", "user.email", "receipt@example.invalid")
    _git(path, "remote", "add", "origin", origin)


def _commit(path: Path, message: str) -> str:
    _git(path, "add", "--all")
    _git(path, "commit", "--quiet", "-m", message)
    return _git(path, "rev-parse", "HEAD")


@dataclass(frozen=True)
class ReceiptRepositories:
    workspace: Path
    habitat: Path
    implementation_commit: str
    habitat_commit: str
    rlr_commit: str


def _repositories(tmp_path: Path) -> ReceiptRepositories:
    rlr = tmp_path / "rlr"
    _init_repository(rlr, origin=RLR_URL)
    (rlr / "README.md").write_text("RLR fixture\n", encoding="utf-8")
    rlr_commit = _commit(rlr, "RLR fixture")

    habitat = tmp_path / "habitat"
    _init_repository(habitat, origin=HABITAT_URL)
    (habitat / "README.md").write_text("Habitat fixture\n", encoding="utf-8")
    _git(
        habitat,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "--quiet",
        str(rlr),
        "src/deps/rlr-audio-propagation",
    )
    habitat_commit = _commit(habitat, "Habitat fixture")

    workspace = tmp_path / "avengine"
    _init_repository(workspace, origin=AVENGINE_URL)
    (workspace / ".gitignore").write_text("tmp/\n", encoding="utf-8")
    (workspace / "README.md").write_text("AVEngine fixture\n", encoding="utf-8")
    implementation_commit = _commit(workspace, "AVEngine fixture")
    return ReceiptRepositories(
        workspace,
        habitat,
        implementation_commit,
        habitat_commit,
        rlr_commit,
    )


def _junit(*, outcome: str = "pass") -> bytes:
    child = {
        "pass": "",
        "fail": '<failure message="expected failure">trace</failure>',
        "error": '<error message="collection error">trace</error>',
        "skip": '<skipped message="not available"/>',
    }[outcome]
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<testsuites><testsuite name="fixture">'
        f'<testcase classname="fixture" name="test_one">{child}</testcase>'
        "</testsuite></testsuites>"
    ).encode("utf-8")


def _command(junit_path: str, junit: bytes, *, exit_code: int) -> list[str]:
    script = (
        "from pathlib import Path; import sys; "
        f"Path(sys.argv[1]).write_bytes({junit!r}); "
        "print('captured stdout'); "
        "sys.stderr.write('captured stderr\\n'); "
        f"raise SystemExit({exit_code})"
    )
    return [sys.executable, "-c", script, junit_path]


def _arguments(
    repositories: ReceiptRepositories,
    *,
    receipt_name: str,
    junit_path: str,
    command: list[str],
) -> list[str]:
    return [
        "receipt",
        "--output",
        str(repositories.workspace / "tmp" / "receipts" / receipt_name),
        "--workspace-root",
        str(repositories.workspace),
        "--habitat-runtime-root",
        str(repositories.habitat),
        "--receipt-id",
        receipt_name.removesuffix(".json"),
        "--layer-id",
        "fast-unit",
        "--junit-xml",
        junit_path,
        "--",
        *command,
    ]


def test_junit_totals_are_derived_from_testcase_outcomes() -> None:
    raw = (
        b"<testsuites><testsuite>"
        b'<testcase name="pass"/>'
        b'<testcase name="fail"><failure/></testcase>'
        b'<testcase name="error"><error/></testcase>'
        b'<testcase name="skip"><skipped/></testcase>'
        b"</testsuite></testsuites>"
    )
    assert derive_junit_totals(raw) == {
        "executed": 4,
        "passed": 1,
        "failed": 1,
        "skipped": 1,
        "errors": 1,
    }


def test_receipt_cli_executes_argv_and_derives_pass_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repositories = _repositories(tmp_path)
    junit_path = "tmp/receipts/pass.junit.xml"
    command = _command(junit_path, _junit(), exit_code=0)
    arguments = _arguments(
        repositories,
        receipt_name="pass-receipt.json",
        junit_path=junit_path,
        command=command,
    )
    assert release_tool_main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "pass"
    assert result["write_status"] == "written"
    receipt = load_json_strict(result["path"])
    assert receipt["command"] == command
    assert receipt["status"] == "pass"
    assert receipt["exit_code"] == 0
    assert receipt["implementation_commit"] == repositories.implementation_commit
    assert receipt["habitat_runtime_commit"] == repositories.habitat_commit
    assert receipt["rlr_commit"] == repositories.rlr_commit
    assert receipt["result_totals"] == {
        "executed": 1,
        "passed": 1,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
    }
    assert base64.b64decode(receipt["captured_output"]["stdout_base64"]) == (
        b"captured stdout\n"
    )
    assert base64.b64decode(receipt["captured_output"]["stderr_base64"]) == (
        b"captured stderr\n"
    )
    assert base64.b64decode(receipt["junit_xml"]["raw_bytes_base64"]) == _junit()
    assert not (repositories.workspace / junit_path).exists()
    assert validate_test_execution_receipt_document(receipt) == []
    assert verify_receipt_payload(receipt) == []

    # Receipt preflight happens before re-execution and remains no-clobber.
    assert release_tool_main(arguments) == 2
    assert "refusing to replace" in capsys.readouterr().out


def test_failed_command_writes_fail_receipt_and_propagates_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repositories = _repositories(tmp_path)
    junit_path = "tmp/receipts/fail.junit.xml"
    command = _command(junit_path, _junit(outcome="fail"), exit_code=7)
    arguments = _arguments(
        repositories,
        receipt_name="fail-receipt.json",
        junit_path=junit_path,
        command=command,
    )
    assert release_tool_main(arguments) == 7
    result = json.loads(capsys.readouterr().out)
    receipt = load_json_strict(result["path"])
    assert receipt["status"] == "fail"
    assert receipt["exit_code"] == 7
    assert receipt["result_totals"]["failed"] == 1
    assert verify_receipt_payload(receipt) == []
    assert not (repositories.workspace / junit_path).exists()


def test_receipt_rejects_preexisting_or_unreferenced_junit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repositories = _repositories(tmp_path)
    junit_path = "tmp/receipts/existing.junit.xml"
    existing = repositories.workspace / junit_path
    existing.parent.mkdir(parents=True)
    existing.write_bytes(_junit())
    command = _command(junit_path, _junit(), exit_code=0)
    arguments = _arguments(
        repositories,
        receipt_name="preexisting.json",
        junit_path=junit_path,
        command=command,
    )
    assert release_tool_main(arguments) == 2
    assert "refusing to replace" in capsys.readouterr().out
    assert not (repositories.workspace / "tmp/receipts/preexisting.json").exists()

    other_path = "tmp/receipts/declared.junit.xml"
    unreferenced = _arguments(
        repositories,
        receipt_name="unreferenced.json",
        junit_path=other_path,
        command=_command("tmp/receipts/other.junit.xml", _junit(), exit_code=0),
    )
    assert release_tool_main(unreferenced) == 2
    assert "must reference" in capsys.readouterr().out


def test_cli_has_no_caller_reported_status_commit_or_total_flags(
    tmp_path: Path,
) -> None:
    repositories = _repositories(tmp_path)
    arguments = _arguments(
        repositories,
        receipt_name="forbidden.json",
        junit_path="tmp/receipts/forbidden.junit.xml",
        command=_command(
            "tmp/receipts/forbidden.junit.xml", _junit(), exit_code=0
        ),
    )
    insertion = arguments.index("--")
    arguments[insertion:insertion] = ["--status", "pass"]
    with pytest.raises(SystemExit):
        release_tool_main(arguments)


def test_cli_requires_explicit_command_separator(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repositories = _repositories(tmp_path)
    junit_path = "tmp/receipts/no-separator.junit.xml"
    arguments = _arguments(
        repositories,
        receipt_name="no-separator.json",
        junit_path=junit_path,
        command=_command(junit_path, _junit(), exit_code=0),
    )
    arguments.remove("--")
    assert release_tool_main(arguments) == 2
    assert "explicit --" in capsys.readouterr().out
    assert not (repositories.workspace / junit_path).exists()


def test_receipt_requires_clean_worktrees_before_and_after_execution(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repositories = _repositories(tmp_path)
    pre_junit = "tmp/receipts/pre-dirty.junit.xml"
    pre_arguments = _arguments(
        repositories,
        receipt_name="pre-dirty.json",
        junit_path=pre_junit,
        command=_command(pre_junit, _junit(), exit_code=0),
    )
    (repositories.workspace / "README.md").write_text(
        "dirty before execution\n", encoding="utf-8"
    )
    assert release_tool_main(pre_arguments) == 2
    assert "no tracked changes or non-ignored untracked entries" in (
        capsys.readouterr().out
    )
    assert not (repositories.workspace / pre_junit).exists()
    _git(repositories.workspace, "checkout", "--", "README.md")

    untracked_junit = "tmp/receipts/untracked-code.junit.xml"
    untracked_arguments = _arguments(
        repositories,
        receipt_name="untracked-code.json",
        junit_path=untracked_junit,
        command=_command(untracked_junit, _junit(), exit_code=0),
    )
    untracked_code = repositories.workspace / "conftest.py"
    untracked_code.write_text(
        "def pytest_collection_modifyitems(items):\n"
        "    items[:] = items[:1]\n",
        encoding="utf-8",
    )
    assert release_tool_main(untracked_arguments) == 2
    untracked_error = capsys.readouterr().out
    assert "no tracked changes or non-ignored untracked entries" in untracked_error
    assert "conftest.py" in untracked_error
    assert not (repositories.workspace / untracked_junit).exists()
    assert not (
        repositories.workspace / "tmp/receipts/untracked-code.json"
    ).exists()
    untracked_code.unlink()

    post_junit = "tmp/receipts/post-dirty.junit.xml"
    script = (
        "from pathlib import Path; import sys; "
        f"Path(sys.argv[1]).write_bytes({_junit()!r}); "
        "Path('README.md').write_text('dirty after execution\\n'); "
        "raise SystemExit(0)"
    )
    post_arguments = _arguments(
        repositories,
        receipt_name="post-dirty.json",
        junit_path=post_junit,
        command=[sys.executable, "-c", script, post_junit],
    )
    assert release_tool_main(post_arguments) == 2
    assert "no tracked changes or non-ignored untracked entries" in (
        capsys.readouterr().out
    )
    assert not (
        repositories.workspace / "tmp" / "receipts" / "post-dirty.json"
    ).exists()


def test_verifier_rederives_embedded_junit_totals_and_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repositories = _repositories(tmp_path)
    junit_path = "tmp/receipts/tamper.junit.xml"
    arguments = _arguments(
        repositories,
        receipt_name="tamper.json",
        junit_path=junit_path,
        command=_command(junit_path, _junit(), exit_code=0),
    )
    assert release_tool_main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    receipt = load_json_strict(result["path"])

    totals_tamper = deepcopy(receipt)
    totals_tamper["result_totals"]["passed"] = 0
    assert any(
        "result_totals differ" in error
        for error in verify_receipt_payload(totals_tamper)
    )

    junit_tamper = deepcopy(receipt)
    junit_tamper["junit_xml"]["raw_bytes_base64"] = base64.b64encode(
        _junit(outcome="fail")
    ).decode("ascii")
    errors = verify_receipt_payload(junit_tamper)
    assert any("result_totals differ" in error for error in errors)

    status_tamper = deepcopy(receipt)
    status_tamper["status"] = "fail"
    assert any(
        "status differs" in error for error in verify_receipt_payload(status_tamper)
    )

    base64_tamper = deepcopy(receipt)
    base64_tamper["captured_output"]["stdout_base64"] = "not base64"
    assert verify_receipt_payload(base64_tamper)
