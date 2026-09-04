#!/usr/bin/env python3
"""Assemble a fresh UE package stage for the current Apartment visual route.

Combines the repository's selected UE source slice (``native/spear/unreal``)
with the physical content files named by one variant of a minimal-closure
report (see ``tools/ue/build_minimal_closure_report.py``), producing a new
stage directory ready for UE 5.5 BuildCookRun. The tool never modifies an
existing stage: the output directory must not exist, every copy is
fresh-write, and a failed assembly leaves no stage marked usable.

The resulting layout matches the retained 20260820 stage:
  <stage>/SpearSim/...              project descriptor, Source, Config
  <stage>/plugins/...               Sp* plugins incl. SpContent
  <stage>/SpearSim/Content/...      /Game closure content
  <stage>/plugins/SpContent/Content/...  /SpContent closure content
  <stage>/STAGE_PROVENANCE.json     assembly record (research only)

BuildCookRun is printed, and executed only with --run-buildcookrun.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]

SOURCE_SLICE_MEMBERS = ("SpearSim", "plugins", "PROVENANCE.md", "README.md")
DEFAULT_MAP = "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000"


class StageAssemblyError(RuntimeError):
    """Raised when the stage cannot be assembled fail-closed."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure-report", required=True, type=Path)
    parser.add_argument("--variant-name", required=True)
    parser.add_argument(
        "--source-slice",
        type=Path,
        default=REPOSITORY / "native/spear/unreal",
        help="repository UE source slice (default: native/spear/unreal)",
    )
    parser.add_argument("--stage-root", required=True, type=Path)
    parser.add_argument("--ue-root", type=Path, default=Path("/data/UE_5.5"))
    parser.add_argument("--map", dest="cook_map", default=DEFAULT_MAP)
    parser.add_argument("--source-commit", default=None,
                        help="repository commit recorded in STAGE_PROVENANCE")
    parser.add_argument(
        "--enable-engine-plugin", action="append", default=[],
        help="engine plugin to enable in the fresh staged uproject; repeatable",
    )
    parser.add_argument(
        "--global-definition", action="append", default=[],
        help=("C++ target GlobalDefinitions entry required by an enabled "
              "runtime plugin; repeatable"),
    )
    parser.add_argument("--run-buildcookrun", action="store_true")
    return parser.parse_args(argv)


def _load_variant(report_path: Path, variant_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise StageAssemblyError(f"cannot read closure report: {error}") from error
    variants = report.get("variants")
    if not isinstance(variants, dict) or variant_name not in variants:
        raise StageAssemblyError(f"closure report has no variant {variant_name}")
    variant = variants[variant_name]
    if variant.get("mapping_complete") is not True:
        raise StageAssemblyError(f"variant {variant_name} is not mapping_complete")
    return report, variant


def _stage_path_for(package: str, source_file: Path, stage: Path) -> Path:
    if package.startswith("/Game/"):
        return stage / "SpearSim/Content" / (
            package.removeprefix("/Game/") + source_file.suffix
        )
    if package.startswith("/SpContent/"):
        return stage / "plugins/SpContent/Content" / (
            package.removeprefix("/SpContent/") + source_file.suffix
        )
    raise StageAssemblyError(f"unsupported package root: {package}")


def _enable_plugins(uproject: dict[str, Any], requested: list[str]) -> None:
    plugins = uproject.setdefault("Plugins", [])
    if not isinstance(plugins, list):
        raise StageAssemblyError("staged uproject Plugins must be a list")
    for name in requested:
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_]*", name
        ):
            raise StageAssemblyError(f"invalid engine plugin name: {name!r}")
        matches = [
            item for item in plugins
            if isinstance(item, dict) and item.get("Name") == name
        ]
        if len(matches) > 1:
            raise StageAssemblyError(f"duplicate staged plugin declaration: {name}")
        if matches:
            matches[0]["Enabled"] = True
        else:
            plugins.append({"Name": name, "Enabled": True})


def _add_global_definitions(target_path: Path, definitions: list[str]) -> None:
    if not definitions:
        return
    for value in definitions:
        if not isinstance(value, str) or not re.fullmatch(
            r"[A-Z][A-Z0-9_]*(=[A-Za-z0-9_.-]+)?", value
        ):
            raise StageAssemblyError(f"invalid global definition: {value!r}")
    if not target_path.is_file():
        raise StageAssemblyError(
            f"staged game target is missing for global definitions: {target_path}"
        )
    text = target_path.read_text(encoding="utf-8")
    marker = "        Type = TargetType.Game;\n"
    if marker not in text:
        raise StageAssemblyError("staged game target lacks TargetType.Game marker")
    additions = []
    for value in definitions:
        statement = f'        GlobalDefinitions.Add("{value}");\n'
        if statement not in text:
            additions.append(statement)
    if additions:
        target_path.write_text(
            text.replace(marker, marker + "".join(additions), 1), encoding="utf-8"
        )


def assemble_stage(args: argparse.Namespace) -> dict[str, Any]:
    stage = args.stage_root.expanduser()
    if stage.exists() or stage.is_symlink():
        raise StageAssemblyError(f"refusing to replace existing stage: {stage}")
    source_slice = args.source_slice.expanduser().resolve()
    for member in ("SpearSim/SpearSim.uproject", "plugins/SpContent/SpContent.uplugin"):
        if not (source_slice / member).is_file():
            raise StageAssemblyError(f"source slice is missing {member}: {source_slice}")
    report, variant = _load_variant(args.closure_report, args.variant_name)

    stage.mkdir(parents=True)
    for member in SOURCE_SLICE_MEMBERS:
        source = source_slice / member
        if not source.exists():
            continue
        target = stage / member
        if source.is_dir():
            shutil.copytree(source, target, symlinks=False)
        else:
            shutil.copy2(source, target)

    # The capture validator requires the SpContent plugin to be explicitly
    # enabled in the staged uproject (the retained 20260820 stage carries the
    # same adjustment); the imported source slice only lists it via
    # AdditionalPluginDirectories.
    uproject_path = stage / "SpearSim/SpearSim.uproject"
    uproject = json.loads(uproject_path.read_text(encoding="utf-8"))
    enabled_plugins = ["SpContent", *args.enable_engine_plugin]
    _enable_plugins(uproject, enabled_plugins)
    uproject_path.write_text(
        json.dumps(uproject, indent=4) + "\n", encoding="utf-8"
    )
    _add_global_definitions(
        stage / "SpearSim/Source/SpearSim.Target.cs", args.global_definition
    )

    copied_files = 0
    copied_bytes = 0
    for entry in variant["physical_mappings"]:
        if entry.get("status") != "unique_authorized_external_input":
            raise StageAssemblyError(
                f"package {entry.get('package')} has no unique external source"
            )
        source_file = Path(entry["source_file"])
        if not source_file.is_file() or source_file.is_symlink():
            raise StageAssemblyError(f"missing source file: {source_file}")
        target = _stage_path_for(entry["package"], source_file, stage)
        if target.exists():
            raise StageAssemblyError(f"duplicate stage target: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target)
        copied_files += 1
        copied_bytes += source_file.stat().st_size
        for sidecar in entry.get("source_sidecars", ()):  # travel with primary
            sidecar_path = Path(sidecar)
            sidecar_target = target.with_name(sidecar_path.name)
            shutil.copy2(sidecar_path, sidecar_target)
            copied_files += 1
            copied_bytes += sidecar_path.stat().st_size

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = stage / f"Packaged-Development-{stamp}"
    uat = args.ue_root.expanduser() / "Engine/Build/BatchFiles/RunUAT.sh"
    command = [
        str(uat),
        "BuildCookRun",
        f"-project={stage / 'SpearSim/SpearSim.uproject'}",
        "-target=SpearSim",
        "-targetplatform=Linux",
        "-clientconfig=Development",
        f"-archivedirectory={archive}",
        "-build",
        "-cook",
        "-stage",
        "-package",
        "-archive",
        "-pak",
        f"-map={args.cook_map}",
        "-nop4",
        "-utf8output",
    ]
    provenance = {
        "schema": "avengine_ue_package_stage_provenance_v2",
        "asset_policy": "mutable_external_inputs_selected_by_package_and_registered_identity_no_byte_lock",
        "avengine_source_commit": args.source_commit,
        "avengine_ue_source": str(source_slice),
        "build_command": command,
        "closure_report": str(args.closure_report.resolve()),
        "closure_variant": args.variant_name,
        "content_files_copied": copied_files,
        "content_bytes_copied": copied_bytes,
        "episode_counted": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "qualification_claim": False,
        "research_only": True,
        "enabled_engine_plugins": list(args.enable_engine_plugin),
        "global_definitions": list(args.global_definition),
        "rights_scope": "owner_authorized_internal_research_not_redistribution",
        "status": "assembled_for_build",
    }
    (stage / "STAGE_PROVENANCE.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"stage": str(stage), "archive": str(archive), "command": command,
            "content_files_copied": copied_files,
            "content_gib": round(copied_bytes / 2**30, 3)}


REQUIRED_SDK_VARIABLES = (
    "AVENGINE_SPEAR_BOOST_ROOT",
    "AVENGINE_SPEAR_RPCLIB_ROOT",
    "AVENGINE_SPEAR_YAML_CPP_ROOT",
)


def main() -> int:
    args = parse_args()
    if args.run_buildcookrun:
        import os

        missing = [name for name in REQUIRED_SDK_VARIABLES if not os.environ.get(name)]
        if missing:
            raise StageAssemblyError(
                "SpModuleRules requires these environment variables before "
                f"BuildCookRun: {', '.join(missing)} (see the ue-stages README "
                "for the installed prefixes under /data/avengine_external/ue-sdk)"
            )
    result = assemble_stage(args)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    if args.run_buildcookrun:
        log = Path(result["stage"]) / f"buildcookrun_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
        with open(log, "w") as handle:
            completed = subprocess.run(
                result["command"], stdout=handle, stderr=subprocess.STDOUT
            )
        print(json.dumps({"buildcookrun_exit": completed.returncode, "log": str(log)}))
        return completed.returncode
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StageAssemblyError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
