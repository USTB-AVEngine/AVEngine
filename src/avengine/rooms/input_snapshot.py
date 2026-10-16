"""Small, human-maintainable input snapshot for an M6.x review bundle.

This is not a release-lock system. Git identifies the two code repositories;
the bundle copies the small JSON configuration that selected the room and
programs, and records only the external assets that directly affect pixels or
audio.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Mapping

from avengine.contracts.json_io import sha256_file, write_json


INPUT_SNAPSHOT_SCHEMA = "avengine_m6x_canary_input_snapshot_v1"


class M6XInputSnapshotError(RuntimeError):
    """The bounded canary input snapshot could not be materialized."""


def _git_head(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and len(value) == 40 else None


def write_canary_input_snapshot(
    bundle_root: str | Path,
    *,
    repository_root: str | Path,
    runtime_root: str | Path,
    config_root: str | Path,
    json_inputs: Mapping[str, str | Path],
    external_assets: Mapping[str, str | Path],
    acoustic_identity: Mapping[str, object],
    capture_mode: str,
    acoustics_mode: str,
) -> Path:
    """Copy small configs and write one readable input index.

    Dense room meshes, captures and RIR arrays remain referenced by their
    existing evidence. They are intentionally not duplicated here.
    """

    bundle = Path(bundle_root).resolve()
    repository = Path(repository_root).resolve()
    runtime = Path(runtime_root).resolve()
    config = Path(config_root).resolve()
    root = bundle / "inputs"
    if root.exists():
        raise M6XInputSnapshotError(f"input snapshot already exists: {root}")
    if not config.is_dir():
        raise M6XInputSnapshotError(f"fixed-room config directory is missing: {config}")

    root.mkdir(parents=True)
    config_destination = root / "fixed_apartment_config"
    shutil.copytree(config, config_destination)

    copied_json: dict[str, dict[str, str]] = {}
    json_root = root / "json"
    json_root.mkdir()
    for role, raw_path in sorted(json_inputs.items()):
        source = Path(raw_path).resolve()
        if not source.is_file():
            raise M6XInputSnapshotError(f"{role} JSON input is missing: {source}")
        destination = json_root / f"{role}.json"
        shutil.copy2(source, destination)
        copied_json[role] = {
            "source_path": str(source),
            "bundle_path": destination.relative_to(bundle).as_posix(),
        }

    external_records: dict[str, dict[str, str]] = {}
    for role, raw_path in sorted(external_assets.items()):
        source = Path(raw_path).resolve()
        if not source.is_file():
            raise M6XInputSnapshotError(f"{role} external asset is missing: {source}")
        external_records[role] = {
            "source_path": str(source),
            "sha256": sha256_file(source),
        }

    record = {
        "schema": INPUT_SNAPSHOT_SCHEMA,
        "code_versions": {
            "avengine_git_commit": _git_head(repository),
            "habitat_runtime_git_commit": _git_head(runtime),
            "policy": "Git commits only; no additional release lock",
        },
        "fixed_apartment_config": {
            "source_path": str(config),
            "bundle_path": config_destination.relative_to(bundle).as_posix(),
        },
        "json_inputs": copied_json,
        "external_assets": external_records,
        "acoustic_identity": dict(acoustic_identity),
        "runtime_artifact_modes": {
            "visual_capture": capture_mode,
            "acoustics": acoustics_mode,
        },
        "dense_state_policy": (
            "room meshes, capture arrays and RIR arrays stay in their existing "
            "bundle evidence; this snapshot does not duplicate them"
        ),
    }
    path = root / "input_index.json"
    write_json(path, record)
    return path


__all__ = [
    "INPUT_SNAPSHOT_SCHEMA",
    "M6XInputSnapshotError",
    "write_canary_input_snapshot",
]
