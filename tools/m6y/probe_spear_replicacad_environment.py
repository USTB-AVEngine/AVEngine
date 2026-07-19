#!/usr/bin/env python3
"""Probe whether ReplicaCAD can be imported/cooked without touching old SPEAR.

The probe is read-only.  A writable *isolated* UE project must be supplied for
editor import and cook; pointing it at the user's old dirty SPEAR checkout is
rejected.  Missing tools/assets are emitted as explicit blocked evidence rather
than being mistaken for a rendered canary.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence


REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_SPEAR_ROOT = REPOSITORY.parent / "AVEngine/external/SPEAR"
SCHEMA = "avengine_optional_spear_replicacad_environment_probe_v1"


def _file_probe(path: Path) -> dict[str, Any]:
    exists = path.exists()
    is_file = path.is_file()
    return {
        "path": str(path),
        "exists": exists,
        "is_file": is_file,
        "is_executable": bool(is_file and os.access(path, os.X_OK)),
        "size_bytes": path.stat().st_size if is_file else None,
    }


def _engine_candidates(explicit: Path | None) -> list[Path]:
    values: list[Path] = []
    if explicit is not None:
        values.append(explicit)
    for key in ("UNREAL_ENGINE_DIR", "UE_ENGINE_DIR"):
        if os.environ.get(key):
            values.append(Path(os.environ[key]))
    values.extend(
        (
            Path("/opt/UnrealEngine"),
        )
    )
    result = []
    seen = set()
    for value in values:
        resolved = value.expanduser().resolve()
        if resolved not in seen:
            result.append(resolved)
            seen.add(resolved)
    return result


def probe(args: argparse.Namespace) -> dict[str, Any]:
    spear_root = args.spear_root.resolve()
    old_project = (
        spear_root / "cpp/unreal_projects/SpearSim"
    ).resolve()
    engine_records = []
    usable_engine = None
    for candidate in _engine_candidates(args.unreal_engine_dir):
        editor = candidate / "Engine/Binaries/Linux/UnrealEditor"
        uat = candidate / "Engine/Build/BatchFiles/RunUAT.sh"
        record = {
            "engine_root": str(candidate),
            "editor": _file_probe(editor),
            "run_uat": _file_probe(uat),
        }
        record["usable"] = (
            record["editor"]["is_executable"]
            and record["editor"]["size_bytes"] > 0
            and record["run_uat"]["is_executable"]
            and record["run_uat"]["size_bytes"] > 0
        )
        if record["usable"] and usable_engine is None:
            usable_engine = candidate
        engine_records.append(record)

    project_record: dict[str, Any]
    if args.ue_project_dir is None:
        project_record = {
            "path": None,
            "status": "missing",
            "is_isolated_from_old_dirty_spear": False,
        }
    else:
        project = args.ue_project_dir.resolve()
        uprojects = sorted(project.glob("*.uproject")) if project.is_dir() else []
        isolated = project != old_project and old_project not in project.parents
        project_record = {
            "path": str(project),
            "status": (
                "ready"
                if project.is_dir()
                and len(uprojects) == 1
                and os.access(project, os.W_OK)
                and isolated
                else "invalid"
            ),
            "uprojects": [str(path) for path in uprojects],
            "writable": bool(project.is_dir() and os.access(project, os.W_OK)),
            "is_isolated_from_old_dirty_spear": isolated,
            "rejected_old_project": str(old_project),
        }

    packaged = old_project / "Standalone-Development/Linux/SpearSim.sh"
    content_directory = old_project / "Content/AVEngine/Optional/ReplicaCAD/apt_0"
    manifest = old_project / "Standalone-Development/Linux/Manifest_UFSFiles_Linux.txt"
    manifest_text = (
        manifest.read_text(encoding="utf-8", errors="replace")
        if manifest.is_file()
        else ""
    )
    cooked_reference_present = any(
        token in manifest_text.casefold()
        for token in ("replicacad", "replica_cad", "frl_apartment_stage")
    )
    request_record: dict[str, Any] | None = None
    if args.execution_request is not None:
        request_path = args.execution_request.resolve()
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
            counts = request.get("counts", {})
            preparation = request.get("glb_preparation", {})
            request_record = {
                "path": str(request_path),
                "status": (
                    "ready"
                    if request.get("schema")
                    == "avengine_optional_spear_replicacad_execution_v1"
                    and counts.get("expected_runtime_mesh_actor_count") == 171
                    and counts.get("articulated_visual_occurrence_count") == 31
                    and preparation.get("status") == "pass"
                    and preparation.get("source_glb_count") == 101
                    else "invalid"
                ),
                "counts": counts,
                "glb_preparation": {
                    key: preparation.get(key)
                    for key in (
                        "status",
                        "prepared_directory",
                        "source_glb_count",
                        "prepared_mesh_asset_count",
                    )
                },
            }
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            request_record = {
                "path": str(request_path),
                "status": "invalid",
                "error": str(exc),
            }

    blockers = []
    if usable_engine is None:
        blockers.append("no non-empty executable UnrealEditor plus RunUAT installation")
    if project_record["status"] != "ready":
        blockers.append("no writable isolated SPEAR UE project was supplied")
    if request_record is not None and request_record["status"] != "ready":
        blockers.append("ReplicaCAD prepared execution request is incomplete")
    status = "ready" if not blockers else "blocked"
    return {
        "schema": SCHEMA,
        "status": status,
        "operation": "read_only_environment_probe",
        "unreal_engines": engine_records,
        "known_zero_byte_placeholders": [
            _file_probe(Path("/tmp/UnrealEditor")),
            _file_probe(Path("/tmp/UnrealEditor-Cmd")),
        ],
        "spear": {
            "root": str(spear_root),
            "old_project": str(old_project),
            "old_project_policy": "read_only; editor/cook writes require isolated project",
            "source_uproject": _file_probe(old_project / "SpearSim.uproject"),
            "packaged_runtime": _file_probe(packaged),
            "replicacad_source_content_directory_present": content_directory.is_dir(),
            "packaged_manifest": _file_probe(manifest),
            "replicacad_reference_in_packaged_manifest": cooked_reference_present,
        },
        "isolated_ue_project": project_record,
        "execution_request": request_record,
        "blockers": blockers,
        "claim_boundary": (
            "ready means import/cook may be attempted; blocked means no UE editor result "
            "or ReplicaCAD comparison video has been produced"
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spear-root", type=Path, default=DEFAULT_SPEAR_ROOT)
    parser.add_argument("--unreal-engine-dir", type=Path)
    parser.add_argument("--ue-project-dir", type=Path)
    parser.add_argument("--execution-request", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = probe(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"SPEAR_REPLICACAD_ENV_{result['status'].upper()} output={output}",
        flush=True,
    )
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
