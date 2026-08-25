"""Unreal Editor-side exporter for the stock approaching_storm TextureCube.

This file is executed by ``UnrealEditor-Cmd -ExecutePythonScript``.  The host
wrapper supplies ``AVENGINE_HDRI_EXPORT_PATH`` and verifies the exported file.
"""

from __future__ import annotations

import os
from pathlib import Path

import unreal


ASSET_PATH = "/HDRIBackdrop/Textures/approaching_storm_4k"
OUTPUT_ENV = "AVENGINE_HDRI_EXPORT_PATH"


def main() -> None:
    raw_output = os.environ.get(OUTPUT_ENV)
    if not raw_output:
        raise RuntimeError(f"{OUTPUT_ENV} is required")
    output = Path(raw_output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite HDRI export: {output}")
    asset = unreal.EditorAssetLibrary.load_asset(ASSET_PATH)
    if asset is None:
        raise RuntimeError(f"could not load Unreal asset {ASSET_PATH}")
    task = unreal.AssetExportTask()
    task.automated = True
    task.object = asset
    task.filename = str(output)
    task.prompt = False
    task.replace_identical = False
    task.write_empty_files = False
    task.selected = False
    task.use_file_archive = False
    task.ignore_object_list = []
    task.errors = []
    if not unreal.Exporter.run_asset_export_task(task):
        raise RuntimeError(f"Unreal exporter rejected {ASSET_PATH}: {list(task.errors)}")
    if task.errors:
        raise RuntimeError(f"Unreal exporter reported errors: {list(task.errors)}")
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Unreal exporter did not create {output}")
    unreal.log(f"AVEngine exported {ASSET_PATH} to {output}")


main()
