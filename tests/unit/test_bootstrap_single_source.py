from __future__ import annotations

import importlib.util
from pathlib import Path
import shlex
import stat
import subprocess
import sys
from types import ModuleType


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SETUP = REPOSITORY_ROOT / "scripts" / "setup.sh"
LOAD_PATHS = REPOSITORY_ROOT / "scripts" / "load_paths.py"


def _load_paths_module() -> ModuleType:
    # The project runtime requires PyYAML, but this focused unit test also runs
    # in the source-only test environment. The synthetic document exercises the
    # checkout-rejection branch without depending on a shared environment.
    yaml_module = ModuleType("yaml")

    def safe_load(_value: object) -> dict[str, object]:
        return {
            "schema": "avengine_workspace_paths_v2",
            "paths": {
                "AVENGINE_HABITAT_RUNTIME_PREFIX": {
                    "default": None,
                    "kind": "directory",
                    "required_for": ["native_external"],
                    "must_not_be_git_checkout": True,
                },
                "AVENGINE_HABITAT_MAGNUM_PYTHON_SITE": {
                    "default": None,
                    "kind": "directory",
                    "required_for": ["native_external"],
                    "must_not_be_git_checkout": True,
                },
                "AVENGINE_RLR_SDK_ROOT": {
                    "default": None,
                    "kind": "directory",
                    "required_for": ["native_external"],
                    "must_not_be_git_checkout": True,
                },
                "AVENGINE_MP3D_ROOT": {
                    "default": None,
                    "kind": "directory",
                    "required_for": ["native_external"],
                    "must_not_be_git_checkout": True,
                },
            },
        }

    yaml_module.safe_load = safe_load
    previous_yaml = sys.modules.get("yaml")
    sys.modules["yaml"] = yaml_module
    try:
        spec = importlib.util.spec_from_file_location("load_paths", LOAD_PATHS)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_yaml is None:
            del sys.modules["yaml"]
        else:
            sys.modules["yaml"] = previous_yaml


def _fake_conda_python(tmp_path: Path, *, conda: bool) -> Path:
    prefix = tmp_path / "conda-prefix"
    if conda:
        (prefix / "conda-meta").mkdir(parents=True)
    else:
        prefix.mkdir()
    python = prefix / "bin" / "python"
    python.parent.mkdir(exist_ok=True)
    executable = shlex.quote(str(python.resolve()))
    resolved_prefix = shlex.quote(str(prefix.resolve()))
    python.write_text(
        "\n".join(
            (
                "#!/usr/bin/env bash",
                'if [ "$1" = "-c" ]; then',
                f"  printf '%s\n' {executable} {resolved_prefix} {resolved_prefix}",
                "  exit 0",
                "fi",
                'if [ -n "${AVENGINE_TEST_PYTHON_LOG:-}" ]; then',
                '  printf "%s\n" "$*" >> "$AVENGINE_TEST_PYTHON_LOG"',
                "fi",
                'echo "unexpected fake Python invocation: $*" >&2',
                "exit 97",
                "",
            )
        ),
        encoding="utf-8",
    )
    python.chmod(python.stat().st_mode | stat.S_IXUSR)
    return python


def test_bootstrap_manifest_declares_one_source_checkout_and_conda() -> None:
    lines = (REPOSITORY_ROOT / "manifest.yaml").read_text(encoding="utf-8").splitlines()
    assert "default_profile: fast_unit" in lines
    assert "  avengine:" in lines
    assert "  habitat_runtime:" not in lines
    assert "    replacement_profile: native_external" in lines
    assert "  environment_kind: conda" in lines
    assert "  require_conda_meta: true" in lines
    assert "  default_environment: .venv" not in lines


def test_paths_replace_legacy_runtime_root_with_explicit_inputs() -> None:
    lines = (REPOSITORY_ROOT / "paths.yaml").read_text(encoding="utf-8").splitlines()
    assert "  AVENGINE_HABITAT_RUNTIME_ROOT:" not in lines
    for name in (
        "AVENGINE_HABITAT_RUNTIME_PREFIX",
        "AVENGINE_HABITAT_MAGNUM_PYTHON_SITE",
        "AVENGINE_RLR_SDK_ROOT",
        "AVENGINE_MP3D_ROOT",
    ):
        assert f"  {name}:" in lines
    assert lines.count("    must_not_be_git_checkout: true") == 4


def test_native_external_validation_rejects_a_checkout_prefix(
    monkeypatch, tmp_path: Path
) -> None:
    checkout = tmp_path / "habitat-sim-AVEngine"
    (checkout / ".git").mkdir(parents=True)
    prefix = checkout / "installed-prefix"
    prefix.mkdir()
    magnum = tmp_path / "magnum-site"
    rlr = tmp_path / "rlr-sdk"
    mp3d = tmp_path / "mp3d"
    for path in (magnum, rlr, mp3d):
        path.mkdir()
    monkeypatch.setenv("AVENGINE_HABITAT_RUNTIME_PREFIX", str(prefix))
    monkeypatch.setenv("AVENGINE_HABITAT_MAGNUM_PYTHON_SITE", str(magnum))
    monkeypatch.setenv("AVENGINE_RLR_SDK_ROOT", str(rlr))
    monkeypatch.setenv("AVENGINE_MP3D_ROOT", str(mp3d))

    ok, checks = _load_paths_module().validate_paths(["native_external"])
    observed = {check["name"]: check for check in checks}
    assert not ok
    assert observed["AVENGINE_HABITAT_RUNTIME_PREFIX"]["status"] == "fail"
    assert observed["AVENGINE_HABITAT_RUNTIME_PREFIX"]["reason"] == (
        "inside_git_checkout"
    )
    assert observed["AVENGINE_HABITAT_RUNTIME_PREFIX"]["checkout_root"] == str(
        checkout.resolve()
    )


def test_setup_rejects_retired_clone_and_venv_flags_before_any_install() -> None:
    for flag in ("--clone-runtime", "--venv"):
        result = subprocess.run(
            ["bash", str(SETUP), flag],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 2
        assert "removed" in result.stderr


def test_setup_fast_unit_dry_run_uses_explicit_conda_python(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    conda_python = _fake_conda_python(tmp_path, conda=True)
    result = subprocess.run(
        [
            "bash",
            str(SETUP),
            "--dry-run",
            "--skip-tests",
            "--profile",
            "fast_unit",
            "--python",
            str(conda_python),
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "conda_prefix=" in result.stdout
    assert "DRY-RUN" in result.stdout
    assert "-m venv" not in result.stdout
    assert "git clone" not in result.stdout


def test_setup_resolves_an_active_conda_prefix(monkeypatch, tmp_path: Path) -> None:
    conda_python = _fake_conda_python(tmp_path, conda=True)
    monkeypatch.delenv("AVENGINE_CONDA_PYTHON", raising=False)
    monkeypatch.delenv("PYTHON", raising=False)
    monkeypatch.setenv("CONDA_PREFIX", str(conda_python.parents[1]))
    result = subprocess.run(
        ["bash", str(SETUP), "--dry-run", "--skip-tests"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert f"python={conda_python.resolve()}" in result.stdout


def test_setup_rejects_non_conda_python(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    python = _fake_conda_python(tmp_path, conda=False)
    result = subprocess.run(
        ["bash", str(SETUP), "--dry-run", "--skip-tests", "--python", str(python)],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "must resolve to a Conda environment" in result.stderr


def test_setup_native_external_requires_inputs_before_pip(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    for native_variable in (
        "AVENGINE_HABITAT_RUNTIME_PREFIX",
        "AVENGINE_HABITAT_MAGNUM_PYTHON_SITE",
        "AVENGINE_MP3D_ROOT",
        "AVENGINE_RLR_SDK_ROOT",
    ):
        monkeypatch.delenv(native_variable, raising=False)
    conda_python = _fake_conda_python(tmp_path, conda=True)
    result = subprocess.run(
        [
            "bash",
            str(SETUP),
            "--profile",
            "native_external",
            "--python",
            str(conda_python),
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "requires explicit" in result.stderr
    assert "AVENGINE_HABITAT_RUNTIME_PREFIX" in result.stderr
    assert "-m pip" not in result.stdout

def test_setup_rejects_python_from_a_different_active_conda_prefix(
    monkeypatch, tmp_path: Path
) -> None:
    conda_python = _fake_conda_python(tmp_path / "selected", conda=True)
    active_prefix = tmp_path / "active-prefix"
    (active_prefix / "conda-meta").mkdir(parents=True)
    monkeypatch.setenv("CONDA_PREFIX", str(active_prefix))
    result = subprocess.run(
        [
            "bash",
            str(SETUP),
            "--dry-run",
            "--skip-tests",
            "--python",
            str(conda_python),
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "differs from active CONDA_PREFIX" in result.stderr
    assert "-m pip" not in result.stdout

def test_setup_rejects_git_rlr_before_pip_even_in_dry_run(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    conda_python = _fake_conda_python(tmp_path, conda=True)
    prefix = tmp_path / "installed-prefix"
    magnum = tmp_path / "magnum-site"
    mp3d = tmp_path / "mp3d"
    rlr_checkout = tmp_path / "old-habitat"
    rlr = rlr_checkout / "RLRAudioPropagationPkg"
    for path in (prefix, magnum, mp3d / "scene_datasets", rlr):
        path.mkdir(parents=True)
    (rlr_checkout / ".git").mkdir(exist_ok=True)
    log = tmp_path / "python-after-preflight.log"
    monkeypatch.setenv("AVENGINE_HABITAT_RUNTIME_PREFIX", str(prefix))
    monkeypatch.setenv("AVENGINE_HABITAT_MAGNUM_PYTHON_SITE", str(magnum))
    monkeypatch.setenv("AVENGINE_MP3D_ROOT", str(mp3d))
    monkeypatch.setenv("AVENGINE_RLR_SDK_ROOT", str(rlr))
    monkeypatch.setenv("AVENGINE_TEST_PYTHON_LOG", str(log))

    result = subprocess.run(
        [
            "bash",
            str(SETUP),
            "--dry-run",
            "--skip-tests",
            "--profile",
            "native_external",
            "--python",
            str(conda_python),
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "AVENGINE_RLR_SDK_ROOT must resolve outside a Git checkout" in result.stderr
    assert "DRY-RUN" not in result.stdout
    assert not log.exists()
