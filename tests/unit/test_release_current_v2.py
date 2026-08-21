from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from avengine.release import load_json_strict
from avengine.release_current import (
    CURRENT_RELEASE_CHECK_IDS,
    CURRENT_RUNTIME_INPUT_ROLES,
    load_current_release_manifest,
    validate_current_release_manifest_document,
)
from avengine.release_current_receipt import (
    CURRENT_CHILD_REPLACED_ENVIRONMENT_VARIABLES,
    CurrentReleaseReceiptError,
    LEGACY_CHILD_ENVIRONMENT_VARIABLES,
    validate_current_receipt_document,
    validate_current_runtime_inputs,
    verify_current_receipt_payload,
)
from tools.release.build_manifest import main as release_tool_main


AVENGINE_URL = "https://github.com/USTB-AVEngine/AVEngine.git"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _commit_all(root: Path, message: str) -> str:
    _git(root, "add", "--all")
    _git(root, "commit", "--quiet", "-m", message)
    return _git(root, "rev-parse", "HEAD")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _current_build_request(
    fixture: CurrentFixture,
    *,
    receipt_path: str = "tmp/current/fast-unit.json",
) -> dict[str, Any]:
    return {
        "schema": "avengine_current_release_build_request_v2",
        "release": {
            "release_id": "avengine-current-fixture",
            "current_milestone": "integration-refactor",
            "manifest_path": "release/avengine_release_manifest_v2.json",
            "formal_release_reason": (
                "No legal external RLR SDK plus adapter-on native run is "
                "bound by this ordinary candidate."
            ),
        },
        "repositories": {
            "implementation_commit": fixture.implementation_commit,
            "expected_avengine_repository": AVENGINE_URL,
        },
        "ordinary_test_receipt": {
            "root_id": "avengine",
            "path": receipt_path,
        },
    }


def _valid_current_manifest_document() -> dict[str, Any]:
    return {
        "schema": "avengine_release_manifest_v2",
        "release": {
            "release_id": "current-fixture",
            "state": "candidate",
            "claim_scope": "ordinary_current_candidate",
            "formal_release_status": "not_run",
            "formal_release_reason": "Native adapter-on evidence has not run.",
            "current_milestone": "integration-refactor",
            "manifest_path": "release/current-fixture.json",
        },
        "repositories": {
            "avengine": {
                "repository": AVENGINE_URL,
                "implementation_commit": "0" * 40,
            }
        },
        "runtime_inputs": dict(CURRENT_RUNTIME_INPUT_ROLES),
        "ordinary_test_receipt": {
            "root_id": "avengine",
            "path": "tmp/current/fast-unit.json",
            "byte_size": 1,
            "sha256": "0" * 64,
        },
        "ordinary_test_status": "pass",
    }


@dataclass(frozen=True)
class CurrentFixture:
    workspace: Path
    implementation_commit: str
    runtime_prefix: Path
    rlr_sdk_root: Path
    scene_data_root: Path
    magnum_python_site: Path


def _make_fixture(
    tmp_path: Path,
    *,
    include_current_source: bool = False,
) -> CurrentFixture:
    workspace = tmp_path / "avengine"
    workspace.mkdir()
    _git(workspace, "init", "--quiet")
    _git(workspace, "config", "user.name", "current release fixture")
    _git(workspace, "config", "user.email", "current@example.invalid")
    _git(workspace, "remote", "add", "origin", AVENGINE_URL)
    (workspace / ".gitignore").write_text("tmp\n", encoding="utf-8")
    (workspace / "README.md").write_text("fixture\n", encoding="utf-8")
    if include_current_source:
        shutil.copytree(
            REPOSITORY_ROOT / "src",
            workspace / "src",
            symlinks=False,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    evidence_root = tmp_path / "evidence-root"
    evidence_root.mkdir()
    (workspace / "tmp").symlink_to(evidence_root, target_is_directory=True)
    implementation_commit = _commit_all(workspace, "implementation A")

    external = tmp_path / "external"
    runtime_prefix = external / "habitat-prefix"
    runtime_prefix.mkdir(parents=True)
    rlr_sdk_root = external / "rlr-sdk"
    header = rlr_sdk_root / "headers" / "RLRAudioPropagation.h"
    library = (
        rlr_sdk_root
        / "libs"
        / "linux"
        / "x64"
        / "libRLRAudioPropagation.so"
    )
    header.parent.mkdir(parents=True)
    library.parent.mkdir(parents=True)
    header.write_text("fixture header\n", encoding="utf-8")
    library.write_bytes(b"fixture shared object\n")
    scene_data_root = external / "scene-data"
    scene_data_root.mkdir()
    magnum_python_site = external / "magnum-site"
    magnum_python_site.mkdir()
    return CurrentFixture(
        workspace=workspace,
        implementation_commit=implementation_commit,
        runtime_prefix=runtime_prefix,
        rlr_sdk_root=rlr_sdk_root,
        scene_data_root=scene_data_root,
        magnum_python_site=magnum_python_site,
    )


def _junit_command(junit_path: str) -> list[str]:
    payload = (
        b"<testsuite name='fixture'><testcase name='passes'/></testsuite>"
    )
    script = (
        "from pathlib import Path; import sys; "
        f"Path(sys.argv[1]).write_bytes({payload!r}); "
        "raise SystemExit(0)"
    )
    return [sys.executable, "-c", script, junit_path]


def _current_receipt_arguments(
    fixture: CurrentFixture,
    *,
    output: Path,
    junit_path: str,
) -> list[str]:
    return [
        "current-receipt",
        "--output",
        str(output),
        "--workspace-root",
        str(fixture.workspace),
        "--runtime-prefix",
        str(fixture.runtime_prefix),
        "--rlr-sdk-root",
        str(fixture.rlr_sdk_root),
        "--scene-data-root",
        str(fixture.scene_data_root),
        "--magnum-python-site",
        str(fixture.magnum_python_site),
        "--receipt-id",
        "current-fast-unit",
        "--layer-id",
        "fast-unit",
        "--junit-xml",
        junit_path,
        "--",
        *_junit_command(junit_path),
    ]


def _current_prepare_arguments(
    fixture: CurrentFixture,
    *,
    request: Path,
) -> list[str]:
    return [
        "current-prepare",
        "--request",
        str(request),
        "--avengine-root",
        str(fixture.workspace),
        "--runtime-prefix",
        str(fixture.runtime_prefix),
        "--rlr-sdk-root",
        str(fixture.rlr_sdk_root),
        "--scene-data-root",
        str(fixture.scene_data_root),
        "--magnum-python-site",
        str(fixture.magnum_python_site),
    ]


def _current_verify_arguments(
    fixture: CurrentFixture,
    *,
    manifest: Path,
) -> list[str]:
    return [
        "current-verify",
        "--manifest",
        str(manifest),
        "--avengine-root",
        str(fixture.workspace),
        "--runtime-prefix",
        str(fixture.runtime_prefix),
        "--rlr-sdk-root",
        str(fixture.rlr_sdk_root),
        "--scene-data-root",
        str(fixture.scene_data_root),
        "--magnum-python-site",
        str(fixture.magnum_python_site),
    ]


def _prepare_current_candidate(
    fixture: CurrentFixture,
    *,
    capsys: pytest.CaptureFixture[str],
) -> tuple[Path, Path]:
    receipt = fixture.workspace / "tmp" / "current" / "fast-unit.json"
    junit = "tmp/current/fast-unit.junit.xml"
    assert release_tool_main(
        _current_receipt_arguments(
            fixture,
            output=receipt,
            junit_path=junit,
        )
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["formal_release_status"] == "not_run"
    assert result["path"] == "tmp/current/fast-unit.json"
    request = fixture.workspace / "tmp" / "current" / "request.json"
    _write_json(request, _current_build_request(fixture))
    assert release_tool_main(
        _current_prepare_arguments(fixture, request=request)
    ) == 0
    prepared = json.loads(capsys.readouterr().out)
    manifest = Path(prepared["manifest"])
    assert manifest.relative_to(fixture.workspace).as_posix().startswith("release/")
    return manifest, receipt


def test_current_receipt_and_candidate_are_checkout_free_and_ordinary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _make_fixture(tmp_path)
    manifest_path, receipt_path = _prepare_current_candidate(fixture, capsys=capsys)
    manifest = load_current_release_manifest(manifest_path)
    receipt = load_json_strict(receipt_path)

    assert receipt["runtime_inputs"] == {
        "habitat_runtime_prefix": str(fixture.runtime_prefix),
        "rlr_sdk_root": str(fixture.rlr_sdk_root),
        "scene_data_root": str(fixture.scene_data_root),
        "magnum_python_site": str(fixture.magnum_python_site),
    }
    assert "habitat_runtime_commit" not in receipt
    assert "rlr_commit" not in receipt
    assert receipt["runtime_observation"]["native_rlr_execution"] is False
    assert verify_current_receipt_payload(receipt) == []
    assert manifest["release"]["claim_scope"] == "ordinary_current_candidate"
    assert manifest["release"]["formal_release_status"] == "not_run"
    assert manifest["runtime_inputs"]["mode"] == "current-installed"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert str(tmp_path / "external") not in manifest_text

    assert release_tool_main(
        _current_verify_arguments(fixture, manifest=manifest_path)
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "pass"
    assert report["formal_release_status"] == "not_run"
    assert [check["check_id"] for check in report["checks"]] == list(
        CURRENT_RELEASE_CHECK_IDS
    )


def test_current_candidate_detects_replaced_receipt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _make_fixture(tmp_path)
    manifest_path, receipt_path = _prepare_current_candidate(fixture, capsys=capsys)
    receipt = load_json_strict(receipt_path)
    receipt["captured_output"]["stdout_base64"] = "Zm9yZ2Vk"
    _write_json(receipt_path, receipt)

    assert release_tool_main(
        _current_verify_arguments(fixture, manifest=manifest_path)
    ) == 1
    report = json.loads(capsys.readouterr().out)
    checks = {item["check_id"]: item for item in report["checks"]}
    assert checks["ordinary_test_receipt"]["status"] == "fail"
    assert any(
        "file record differs" in error
        for error in checks["ordinary_test_receipt"]["errors"]
    )


def _legacy_environment_probe_command(junit_path: str) -> list[str]:
    script = f"""
from pathlib import Path
import os
import sys

from avengine.m1.habitat_capture import discover_runtime_root

failures = [
    name for name in {sorted(LEGACY_CHILD_ENVIRONMENT_VARIABLES)!r}
    if name in os.environ
    and name not in {sorted(CURRENT_CHILD_REPLACED_ENVIRONMENT_VARIABLES)!r}
]
expected_pythonpath = os.pathsep.join((str(Path.cwd() / "src"), str(Path.cwd())))
if os.environ.get("PYTHONPATH") != expected_pythonpath:
    failures.append("PYTHONPATH did not name only the isolated child source root")
module_path = Path(sys.modules["avengine.m1.habitat_capture"].__file__).resolve()
if not module_path.is_relative_to(Path.cwd()):
    failures.append("legacy Habitat module was not imported from the isolated root")
try:
    discover_runtime_root()
except FileNotFoundError:
    pass
else:
    failures.append("discover_runtime_root resolved a legacy checkout")

junit = Path(sys.argv[1])
if failures:
    junit.write_text(
        "<testsuite name='legacy-env'><testcase name='scrub'>"
        "<failure/></testcase></testsuite>",
        encoding="utf-8",
    )
    raise SystemExit(1)
junit.write_text(
    "<testsuite name='legacy-env'><testcase name='scrub'/></testsuite>",
    encoding="utf-8",
)
"""
    return [sys.executable, "-c", script, junit_path]


def test_current_receipt_isolates_child_from_inherited_and_sibling_legacy_checkout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path, include_current_source=True)
    old_checkout = fixture.workspace.parent / "habitat-sim-AVEngine"
    old_checkout.mkdir()
    _git(old_checkout, "init", "--quiet")

    direct_environment = dict(os.environ)
    for variable in LEGACY_CHILD_ENVIRONMENT_VARIABLES:
        direct_environment.pop(variable, None)
    direct_environment["PYTHONPATH"] = str(fixture.workspace / "src")
    direct_environment["PYTHONNOUSERSITE"] = "1"
    direct_environment["PYTHONSAFEPATH"] = "1"
    direct_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    direct = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from avengine.m1.habitat_capture import discover_runtime_root; "
                "print(discover_runtime_root())"
            ),
        ],
        cwd=fixture.workspace,
        check=False,
        capture_output=True,
        text=True,
        env=direct_environment,
    )
    assert direct.returncode != 0
    assert "FileNotFoundError" in direct.stderr

    for variable in LEGACY_CHILD_ENVIRONMENT_VARIABLES:
        monkeypatch.setenv(variable, str(old_checkout))

    output = fixture.workspace / "tmp" / "current" / "legacy-env.json"
    junit = "tmp/current/legacy-env.junit.xml"
    arguments = _current_receipt_arguments(
        fixture,
        output=output,
        junit_path=junit,
    )
    separator = arguments.index("--")
    arguments[separator + 1 :] = _legacy_environment_probe_command(junit)

    assert release_tool_main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "pass"
    assert load_json_strict(output)["status"] == "pass"


def test_current_receipt_refuses_a_source_snapshot_symlink(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _make_fixture(tmp_path)
    target = tmp_path / "old-source.py"
    target.write_text("old source\n", encoding="utf-8")
    (fixture.workspace / "legacy-source.py").symlink_to(target)
    _commit_all(fixture.workspace, "tracked legacy source link")

    output = fixture.workspace / "tmp" / "current" / "symlink.json"
    assert release_tool_main(
        _current_receipt_arguments(
            fixture,
            output=output,
            junit_path="tmp/current/symlink.junit.xml",
        )
    ) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert "source snapshot must not retain a symlink" in error
    assert not output.exists()


def test_current_receipt_excludes_ignored_source_startup_hook(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _make_fixture(tmp_path, include_current_source=True)
    (fixture.workspace / ".gitignore").write_text(
        "tmp\nsitecustomize.py\n",
        encoding="utf-8",
    )
    _commit_all(fixture.workspace, "ignore local startup hook")
    (fixture.workspace / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "target = Path(sys.argv[-1])\n"
        "target.parent.mkdir(parents=True, exist_ok=True)\n"
        "target.write_text(\"<testsuite><testcase name='forged'/></testsuite>\", "
        "encoding='utf-8')\n",
        encoding="utf-8",
    )

    output = fixture.workspace / "tmp" / "current" / "ignored-source.json"
    junit = "tmp/current/ignored-source.junit.xml"
    arguments = _current_receipt_arguments(
        fixture,
        output=output,
        junit_path=junit,
    )
    separator = arguments.index("--")
    arguments[separator + 1 :] = [
        sys.executable,
        "-c",
        "raise SystemExit(0)",
        junit,
    ]

    assert release_tool_main(arguments) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert "test command did not generate the declared JUnit XML" in error
    assert not output.exists()


def test_current_receipt_rejects_path_resolved_executable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path)
    fake_bin = tmp_path / "legacy-bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin))

    output = fixture.workspace / "tmp" / "current" / "path.json"
    junit = "tmp/current/path.junit.xml"
    arguments = _current_receipt_arguments(
        fixture,
        output=output,
        junit_path=junit,
    )
    separator = arguments.index("--")
    arguments[separator + 1 :] = [
        "python",
        "-c",
        "raise SystemExit(99)",
        junit,
    ]

    assert release_tool_main(arguments) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert "command executable must be an absolute path" in error
    assert not output.exists()


def test_current_receipt_replaces_inherited_path_for_absolute_executable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path)
    fake_bin = tmp_path / "legacy-bin"
    fake_bin.mkdir()
    git_marker = tmp_path / "fake-git-was-run"
    fake_git = fake_bin / "git"
    fake_git.write_text(
        f"#!/bin/sh\nprintf forged > {git_marker}\nexit 99\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.defpath}")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "forged-git-dir"))

    output = fixture.workspace / "tmp" / "current" / "safe-path.json"
    junit = "tmp/current/safe-path.junit.xml"
    payload = "<testsuite><testcase name='safe-path'/></testsuite>"
    command = (
        "from pathlib import Path; import os, sys; "
        "assert sys.argv[2] not in os.environ['PATH']; "
        f"Path(sys.argv[1]).write_text({payload!r}, encoding='utf-8')"
    )
    arguments = _current_receipt_arguments(
        fixture,
        output=output,
        junit_path=junit,
    )
    separator = arguments.index("--")
    arguments[separator + 1 :] = [
        sys.executable,
        "-c",
        command,
        junit,
        str(fake_bin),
    ]

    assert release_tool_main(arguments) == 0
    receipt = load_json_strict(output)
    assert receipt["command"][0] == str(Path(sys.executable).resolve())
    assert not git_marker.exists()


def test_current_receipt_drops_inherited_git_dir_before_child_execution(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path)
    legacy_checkout = tmp_path / "legacy-checkout"
    legacy_checkout.mkdir()
    _git(legacy_checkout, "init", "--quiet")
    _git(legacy_checkout, "config", "user.name", "legacy fixture")
    _git(legacy_checkout, "config", "user.email", "legacy@example.invalid")
    (legacy_checkout / "legacy_only.txt").write_text(
        "old checkout object\n",
        encoding="utf-8",
    )
    _commit_all(legacy_checkout, "legacy-only object")
    assert _git(legacy_checkout, "show", "HEAD:legacy_only.txt") == (
        "old checkout object"
    )
    monkeypatch.setenv("GIT_DIR", str(legacy_checkout / ".git"))

    output = fixture.workspace / "tmp" / "current" / "child-git-dir.json"
    junit = "tmp/current/child-git-dir.junit.xml"
    pass_junit = "<testsuite><testcase name='child-git-dir'/></testsuite>"
    fail_junit = (
        "<testsuite><testcase name='child-git-dir'><failure/>"
        "</testcase></testsuite>"
    )
    probe = (
        "from pathlib import Path; import subprocess, sys; "
        "result = subprocess.run(['git', 'show', 'HEAD:legacy_only.txt'], "
        "check=False, capture_output=True, text=True); "
        "leaked = result.returncode == 0 and result.stdout == 'old checkout object\\n'; "
        f"Path(sys.argv[1]).write_text({pass_junit!r} if leaked else "
        f"{fail_junit!r}, encoding='utf-8'); "
        "raise SystemExit(0 if leaked else 1)"
    )
    arguments = _current_receipt_arguments(
        fixture,
        output=output,
        junit_path=junit,
    )
    separator = arguments.index("--")
    arguments[separator + 1 :] = [
        sys.executable,
        "-c",
        probe,
        junit,
    ]

    assert release_tool_main(arguments) == 1
    receipt = load_json_strict(output)
    assert receipt["status"] == "fail"
    assert receipt["exit_code"] == 1
    assert receipt["result_totals"]["failed"] == 1


def test_current_prepare_ignores_inherited_git_selectors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path)
    receipt = fixture.workspace / "tmp" / "current" / "fast-unit.json"
    junit = "tmp/current/fast-unit.junit.xml"
    assert release_tool_main(
        _current_receipt_arguments(
            fixture,
            output=receipt,
            junit_path=junit,
        )
    ) == 0
    capsys.readouterr()
    request = fixture.workspace / "tmp" / "current" / "request.json"
    _write_json(request, _current_build_request(fixture))
    fake_bin = tmp_path / "legacy-bin"
    fake_bin.mkdir()
    marker = tmp_path / "fake-git-was-run"
    fake_git = fake_bin / "git"
    fake_git.write_text(
        f"#!/bin/sh\nprintf forged > {marker}\nexit 99\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.defpath}")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "forged-git-dir"))

    assert release_tool_main(_current_prepare_arguments(fixture, request=request)) == 0
    assert not marker.exists()


def test_current_receipt_rejects_git_checkout_executable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _make_fixture(tmp_path)
    runner = fixture.workspace / "tracked-runner"
    runner.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runner.chmod(0o755)
    _commit_all(fixture.workspace, "tracked command runner")

    output = fixture.workspace / "tmp" / "current" / "checkout-command.json"
    arguments = _current_receipt_arguments(
        fixture,
        output=output,
        junit_path="tmp/current/checkout-command.junit.xml",
    )
    separator = arguments.index("--")
    arguments[separator + 1 :] = [
        str(runner),
        "tmp/current/checkout-command.junit.xml",
    ]

    assert release_tool_main(arguments) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert "command executable must resolve outside a Git checkout" in error
    assert not output.exists()


@pytest.mark.parametrize(
    ("command_kind", "expected"),
    [
        ("bare-python", "command executable must be an absolute path"),
        (
            "git-checkout",
            "command executable must resolve outside a Git checkout",
        ),
        (
            "noncanonical-symlink",
            "recorded command executable must use its canonical absolute",
        ),
    ],
)
def test_current_prepare_and_verify_reject_schema_valid_manual_receipt_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    command_kind: str,
    expected: str,
) -> None:
    fixture = _make_fixture(tmp_path)
    receipt_path = fixture.workspace / "tmp" / "current" / "fast-unit.json"
    junit_path = "tmp/current/fast-unit.junit.xml"
    assert release_tool_main(
        _current_receipt_arguments(
            fixture,
            output=receipt_path,
            junit_path=junit_path,
        )
    ) == 0
    capsys.readouterr()
    valid_receipt = load_json_strict(receipt_path)
    assert valid_receipt["command"][0] == str(Path(sys.executable).resolve())
    assert validate_current_receipt_document(valid_receipt) == []

    manual_receipt = json.loads(json.dumps(valid_receipt))
    if command_kind == "bare-python":
        manual_receipt["command"][0] = "python"
    elif command_kind == "git-checkout":
        legacy_checkout = tmp_path / "legacy-command-checkout"
        legacy_checkout.mkdir()
        _git(legacy_checkout, "init", "--quiet")
        runner = legacy_checkout / "runner"
        runner.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        runner.chmod(0o755)
        manual_receipt["command"][0] = str(runner)
    else:
        executable_link = tmp_path / "noncanonical-python"
        executable_link.symlink_to(Path(sys.executable))
        manual_receipt["command"][0] = str(executable_link)
    assert validate_current_receipt_document(manual_receipt) == []

    request = fixture.workspace / "tmp" / "current" / "request.json"
    _write_json(request, _current_build_request(fixture))
    _write_json(receipt_path, manual_receipt)
    assert release_tool_main(_current_prepare_arguments(fixture, request=request)) == 2
    assert expected in json.loads(capsys.readouterr().out)["error"]

    _write_json(receipt_path, valid_receipt)
    assert release_tool_main(_current_prepare_arguments(fixture, request=request)) == 0
    manifest_path = Path(json.loads(capsys.readouterr().out)["manifest"])
    assert release_tool_main(
        _current_verify_arguments(fixture, manifest=manifest_path)
    ) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"

    _write_json(receipt_path, manual_receipt)
    assert release_tool_main(
        _current_verify_arguments(fixture, manifest=manifest_path)
    ) == 1
    report = json.loads(capsys.readouterr().out)
    checks = {item["check_id"]: item for item in report["checks"]}
    assert checks["ordinary_test_receipt"]["status"] == "fail"
    assert any(expected in error for error in checks["ordinary_test_receipt"]["errors"])


@pytest.mark.parametrize(
    ("output_relative", "junit_path"),
    [
        ("release/current/receipt.json", "tmp/current/receipt.junit.xml"),
        ("tmp/current/receipt.json", "release/current/receipt.junit.xml"),
    ],
)
def test_current_receipt_requires_logical_ignored_tmp_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    output_relative: str,
    junit_path: str,
) -> None:
    fixture = _make_fixture(tmp_path)
    output = fixture.workspace / output_relative

    assert release_tool_main(
        _current_receipt_arguments(
            fixture,
            output=output,
            junit_path=junit_path,
        )
    ) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert "logical ignored tmp root" in error
    assert not output.exists()


def test_current_receipt_requires_git_ignored_tmp_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _make_fixture(tmp_path)
    (fixture.workspace / ".gitignore").write_text("other\n", encoding="utf-8")
    output = fixture.workspace / "tmp" / "current" / "receipt.json"

    assert release_tool_main(
        _current_receipt_arguments(
            fixture,
            output=output,
            junit_path="tmp/current/receipt.junit.xml",
        )
    ) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert "must be Git-ignored beneath the logical tmp root" in error
    assert not output.exists()


def test_current_prepare_requires_logical_ignored_tmp_request(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _make_fixture(tmp_path)
    request = fixture.workspace / "release" / "current-request.json"
    _write_json(request, _current_build_request(fixture))

    assert (
        release_tool_main(_current_prepare_arguments(fixture, request=request))
        == 2
    )
    error = json.loads(capsys.readouterr().out)["error"]
    assert (
        "current release build request must be under the logical ignored tmp root"
        in error
    )


def test_current_prepare_requires_logical_ignored_tmp_bound_receipt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _make_fixture(tmp_path)
    request = fixture.workspace / "tmp" / "current" / "request.json"
    _write_json(
        request,
        _current_build_request(fixture, receipt_path="release/current/receipt.json"),
    )

    assert (
        release_tool_main(_current_prepare_arguments(fixture, request=request))
        == 2
    )
    error = json.loads(capsys.readouterr().out)["error"]
    assert (
        "ordinary_test_receipt.path must be under the logical ignored tmp root"
        in error
    )


@pytest.mark.parametrize(
    ("section", "field", "value", "expected"),
    [
        (
            "release",
            "formal_release_reason",
            "/",
            "release.formal_release_reason must not contain a filesystem absolute path",
        ),
        (
            "release",
            "current_milestone",
            "//",
            "release.current_milestone must not contain a filesystem absolute path",
        ),
        (
            "release",
            "formal_release_reason",
            "See /private/old-habitat for details.",
            "release.formal_release_reason must not contain a filesystem absolute path",
        ),
        (
            "release",
            "formal_release_reason",
            "See //old-server/private/habitat-checkout for details.",
            "release.formal_release_reason must not contain a filesystem absolute path",
        ),
        (
            "release",
            "current_milestone",
            "file:///private/current-milestone",
            "release.current_milestone must not contain a file URL",
        ),
        (
            "repositories",
            "expected_avengine_repository",
            "/srv/old-AVEngine.git",
            "repositories.expected_avengine_repository must not contain a filesystem absolute path",
        ),
        (
            "repositories",
            "expected_avengine_repository",
            "//",
            "repositories.expected_avengine_repository must not contain a filesystem absolute path",
        ),
        (
            "repositories",
            "expected_avengine_repository",
            "C:/old-AVEngine.git",
            "repositories.expected_avengine_repository must not contain a filesystem absolute path",
        ),
    ],
)
def test_current_prepare_rejects_filesystem_syntax_in_git_bound_free_text(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    section: str,
    field: str,
    value: str,
    expected: str,
) -> None:
    fixture = _make_fixture(tmp_path)
    request = fixture.workspace / "tmp" / "current" / "request.json"
    document = _current_build_request(fixture)
    if section == "release":
        document["release"][field] = value
    else:
        document["repositories"][field] = value
    _write_json(request, document)

    assert (
        release_tool_main(_current_prepare_arguments(fixture, request=request))
        == 2
    )
    assert expected in json.loads(capsys.readouterr().out)["error"]


@pytest.mark.parametrize(
    ("section", "field", "value", "expected"),
    [
        (
            "release",
            "formal_release_reason",
            "/",
            "release.formal_release_reason must not contain a filesystem absolute path",
        ),
        (
            "release",
            "current_milestone",
            "//",
            "release.current_milestone must not contain a filesystem absolute path",
        ),
        (
            "release",
            "formal_release_reason",
            "See /private/old-habitat for details.",
            "release.formal_release_reason must not contain a filesystem absolute path",
        ),
        (
            "release",
            "current_milestone",
            "file:///private/current-milestone",
            "release.current_milestone must not contain a file URL",
        ),
        (
            "repositories",
            "repository",
            "/",
            "repositories.avengine.repository must not contain a filesystem absolute path",
        ),
        (
            "repositories",
            "repository",
            "/srv/old-AVEngine.git",
            "repositories.avengine.repository must not contain a filesystem absolute path",
        ),
        (
            "repositories",
            "repository",
            r"\\old-server\AVEngine.git",
            "repositories.avengine.repository must not contain a filesystem absolute path",
        ),
        (
            "repositories",
            "repository",
            "file:///private/AVEngine.git",
            "repositories.avengine.repository must not contain a file URL",
        ),
    ],
)
def test_current_manifest_rejects_filesystem_syntax_in_git_bound_free_text(
    section: str,
    field: str,
    value: str,
    expected: str,
) -> None:
    document = _valid_current_manifest_document()
    if section == "release":
        document["release"][field] = value
    else:
        document["repositories"]["avengine"][field] = value

    assert any(
        expected in error
        for error in validate_current_release_manifest_document(document)
    )


def test_current_manifest_schema_requires_tmp_receipt_path() -> None:
    document = _valid_current_manifest_document()
    document["ordinary_test_receipt"]["path"] = "release/current/receipt.json"

    errors = validate_current_release_manifest_document(document)
    assert any(
        "ordinary_test_receipt.path" in error and "^tmp/" in error
        for error in errors
    )


def test_current_inputs_reject_checkout_symlink_and_missing_sdk(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _git(checkout, "init", "--quiet")
    with pytest.raises(CurrentReleaseReceiptError, match="outside a Git checkout"):
        validate_current_runtime_inputs(
            runtime_prefix=checkout,
            rlr_sdk_root=fixture.rlr_sdk_root,
            scene_data_root=fixture.scene_data_root,
            magnum_python_site=fixture.magnum_python_site,
        )

    link = tmp_path / "runtime-link"
    link.symlink_to(fixture.runtime_prefix, target_is_directory=True)
    with pytest.raises(CurrentReleaseReceiptError, match="must not traverse a symlink"):
        validate_current_runtime_inputs(
            runtime_prefix=link,
            rlr_sdk_root=fixture.rlr_sdk_root,
            scene_data_root=fixture.scene_data_root,
            magnum_python_site=fixture.magnum_python_site,
        )

    empty_sdk = tmp_path / "empty-sdk"
    empty_sdk.mkdir()
    with pytest.raises(CurrentReleaseReceiptError, match="is missing"):
        validate_current_runtime_inputs(
            runtime_prefix=fixture.runtime_prefix,
            rlr_sdk_root=empty_sdk,
            scene_data_root=fixture.scene_data_root,
            magnum_python_site=fixture.magnum_python_site,
        )


def test_current_cli_has_no_legacy_habitat_root_option(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    arguments = _current_receipt_arguments(
        fixture,
        output=fixture.workspace / "tmp" / "current" / "receipt.json",
        junit_path="tmp/current/receipt.junit.xml",
    )
    insertion = arguments.index("--")
    arguments[insertion:insertion] = [
        "--habitat-runtime-root",
        str(tmp_path / "old-checkout"),
    ]
    with pytest.raises(SystemExit):
        release_tool_main(arguments)


def test_current_cli_refuses_a_native_layer_label(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    arguments = _current_receipt_arguments(
        fixture,
        output=fixture.workspace / "tmp" / "current" / "receipt.json",
        junit_path="tmp/current/receipt.junit.xml",
    )
    layer_index = arguments.index("--layer-id") + 1
    arguments[layer_index] = "rlr-audio"
    with pytest.raises(SystemExit):
        release_tool_main(arguments)


def test_current_verify_requires_the_declared_manifest_location(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _make_fixture(tmp_path)
    manifest_path, _ = _prepare_current_candidate(fixture, capsys=capsys)
    manifest = load_json_strict(manifest_path)
    manifest["release"]["manifest_path"] = "release/other.json"
    _write_json(manifest_path, manifest)

    assert release_tool_main(
        _current_verify_arguments(fixture, manifest=manifest_path)
    ) == 1
    report = json.loads(capsys.readouterr().out)
    checks = {item["check_id"]: item for item in report["checks"]}
    assert checks["manifest_schema"]["status"] == "fail"
    assert any(
        "differs from the current manifest location" in error
        for error in checks["manifest_schema"]["errors"]
    )


def test_current_verify_rejects_double_slash_network_path_in_git_bound_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _make_fixture(tmp_path)
    manifest_path, _ = _prepare_current_candidate(fixture, capsys=capsys)
    manifest = load_json_strict(manifest_path)
    manifest["release"]["formal_release_reason"] = (
        "See //old-server/private/habitat-checkout for details."
    )
    _write_json(manifest_path, manifest)

    assert release_tool_main(
        _current_verify_arguments(fixture, manifest=manifest_path)
    ) == 1
    report = json.loads(capsys.readouterr().out)
    checks = {item["check_id"]: item for item in report["checks"]}
    assert checks["manifest_schema"]["status"] == "fail"
    assert any(
        "release.formal_release_reason must not contain a filesystem absolute path"
        in error
        for error in checks["manifest_schema"]["errors"]
    )


def test_current_prepare_allows_https_url_in_git_bound_free_text(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _make_fixture(tmp_path)
    request = fixture.workspace / "tmp" / "current" / "request.json"
    receipt = fixture.workspace / "tmp" / "current" / "fast-unit.json"
    assert release_tool_main(
        _current_receipt_arguments(
            fixture,
            output=receipt,
            junit_path="tmp/current/fast-unit.junit.xml",
        )
    ) == 0
    capsys.readouterr()
    document = _current_build_request(fixture)
    document["release"]["formal_release_reason"] = (
        "Details are available at https://example.invalid/current-release."
    )
    _write_json(request, document)

    assert release_tool_main(_current_prepare_arguments(fixture, request=request)) == 0
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["formal_release_status"] == "not_run"


def test_current_manifest_schema_is_draft_2020_12_and_nonformal() -> None:
    schema_path = (
        REPOSITORY_ROOT / "schemas" / "avengine_release_manifest_v2.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert validate_current_release_manifest_document(
        {
            "schema": "avengine_release_manifest_v2",
            "release": {
                "release_id": "bad",
                "state": "candidate",
                "claim_scope": "ordinary_current_candidate",
                "formal_release_status": "pass",
                "formal_release_reason": "wrong",
                "current_milestone": "M6",
                "manifest_path": "release/invalid.json",
            },
            "repositories": {
                "avengine": {
                    "repository": AVENGINE_URL,
                    "implementation_commit": "0" * 40,
                }
            },
            "runtime_inputs": {},
            "ordinary_test_receipt": {
                "root_id": "avengine",
                "path": "tmp/receipt.json",
                "byte_size": 1,
                "sha256": "0" * 64,
            },
            "ordinary_test_status": "pass",
        }
    )
