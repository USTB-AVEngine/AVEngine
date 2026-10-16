from __future__ import annotations

import json
from pathlib import Path

from avengine.contracts.json_io import sha256_file
from avengine.rooms.input_snapshot import (
    INPUT_SNAPSHOT_SCHEMA,
    write_canary_input_snapshot,
)


ROOT = Path(__file__).resolve().parents[2]


def test_input_snapshot_copies_small_configs_and_records_direct_assets(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    config = tmp_path / "config"
    config.mkdir()
    (config / "room_capsule.json").write_text("{}\n", encoding="utf-8")
    selected = tmp_path / "selected.json"
    selected.write_text('{"selected": true}\n', encoding="utf-8")
    external = tmp_path / "source.wav"
    external.write_bytes(b"direct-result-changing-input")

    path = write_canary_input_snapshot(
        bundle,
        repository_root=ROOT,
        runtime_root=ROOT.parent / "habitat-sim-AVEngine",
        config_root=config,
        json_inputs={"room_manifest": selected},
        external_assets={"dry_audio": external},
        acoustic_identity={"room_capsule_id": "fixed_room_v1"},
        capture_mode="retained_validated",
        acoustics_mode="retained_validated",
    )

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["schema"] == INPUT_SNAPSHOT_SCHEMA
    assert record["code_versions"]["avengine_git_commit"] is not None
    assert record["code_versions"]["policy"] == (
        "Git commits only; no additional release lock"
    )
    assert record["json_inputs"]["room_manifest"]["bundle_path"] == (
        "inputs/json/room_manifest.json"
    )
    assert (bundle / "inputs/json/room_manifest.json").is_file()
    assert (bundle / "inputs/fixed_apartment_config/room_capsule.json").is_file()
    assert record["external_assets"]["dry_audio"]["sha256"] == sha256_file(external)
    assert record["acoustic_identity"]["room_capsule_id"] == "fixed_room_v1"
