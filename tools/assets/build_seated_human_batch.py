"""Run the AVEngine-owned Blender seated-human builder for four assets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

from avengine.assets.seated_humans import load_seated_human_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--blender", required=True, type=Path)
    parser.add_argument("--blender-script", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser.parse_args()


def _asset_args(item: Any, output: Path) -> list[str]:
    args = [
        "--human-glb",
        str(item.source_glb),
        "--output",
        str(output),
        "--asset-id",
        item.asset_id,
        "--display-label",
        item.display_label,
        "--color-name",
        item.color_name,
        "--emitter-offset-blender-m=" + ",".join(str(value) for value in item.emitter_offset_blender_m),
        "--seat-anchor-id",
        item.seat_anchor_id,
        "--seat-top-m",
        str(item.seat_top_m),
        "--chair-center-blender-m",
        ",".join(str(value) for value in item.chair_center_blender_m),
        "--floor-correction-m",
        str(item.floor_correction_m),
    ]
    if item.shirt_color_rgb is not None:
        args.extend(["--shirt-color-rgb", ",".join(str(value) for value in item.shirt_color_rgb)])
    return args


def main() -> None:
    args = parse_args()
    blender = args.blender.expanduser().resolve()
    script = args.blender_script.expanduser().resolve()
    if not blender.is_file() or not script.is_file():
        raise SystemExit("Blender executable or seated builder script is missing")
    specs = load_seated_human_batch(args.spec)
    output = args.output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to replace seated output: {output}")
    output.mkdir(parents=True)
    reports: list[dict[str, Any]] = []
    for item in specs:
        asset_output = output / item.asset_id
        command = [
            str(blender),
            "-b",
            "--factory-startup",
            "--python",
            str(script),
            "--",
            *_asset_args(item, asset_output),
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        (output / f"{item.asset_id}.blender.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise SystemExit(f"Blender seated build failed for {item.asset_id}: {result.returncode}")
        report_path = asset_output / "seated_pose_report.json"
        if not report_path.is_file():
            raise SystemExit(f"seated report is missing: {report_path}")
        reports.append(json.loads(report_path.read_text(encoding="utf-8")))
    manifest = {
        "kind": "avengine_seated_human_skeletal_manifest_v1",
        "status": "research_only",
        "assets": reports,
        "claim_boundary": "four independent skeletal seated-idle actors; no transition or lip-sync claim",
    }
    manifest_path = args.manifest.expanduser().resolve()
    if manifest_path.exists() or manifest_path.is_symlink():
        raise SystemExit(f"refusing to replace manifest: {manifest_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + chr(10), encoding="utf-8")
    print(json.dumps({"status": "pass", "output": str(output), "assets": len(reports)}, indent=2))


if __name__ == "__main__":
    main()
