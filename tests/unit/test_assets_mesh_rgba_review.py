"""Checks for the review-only mesh RGBA preview tool."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[2]
TOOL = REPOSITORY / "tools/assets/render_mesh_rgba_review.py"


def test_preview_tool_is_explicitly_review_only_and_cpu_transparent() -> None:
    source = TOOL.read_text(encoding="utf-8")
    assert '"source_kind": args.source_kind' in source
    assert '"rendered_from_registered_mesh": (' in source
    assert '"rendered_from_generated_candidate": (' in source
    assert '"verified_pixal3d_receipt"' in source
    assert '"original_canonical_input": False' in source
    assert '"canonical_image_replacement": False' in source
    assert '"new_asset_registration": False' in source
    assert '"cpu_only": (' in source
    assert 'scene.render.engine = "CYCLES"' in source
    assert 'scene.cycles.device = "CPU"' in source
    assert 'scene.cycles.samples = int(args.samples)' in source
    assert 'scene.render.film_transparent = True' in source
    assert 'scene.render.image_settings.color_mode = "RGBA"' in source
    assert 'scene.render.engine = "CYCLES"' in source
    assert 'obj.type not in {"MESH", "LIGHT"} and obj is not camera' in source
    assert '"--source-kind"' in source
    assert '"generated_candidate"' in source
    assert '"--pixal-receipt"' in source
    assert '"verified_pixal3d_receipt"' in source
    assert 'receipt_output != input_path' in source
    assert 'if not binding["verified"]:' in source
    assert 'registered_asset source binding failed' in source


def test_preview_tool_compiles() -> None:
    subprocess.run([sys.executable, "-m", "py_compile", str(TOOL)], check=True)


def test_no_clobber_checks_png_and_record_before_render() -> None:
    source = TOOL.read_text(encoding="utf-8")
    assert source.index("if output_path.exists()") < source.index(
        "bpy.ops.wm.read_factory_settings"
    )
    assert source.index("if record_path.exists()") < source.index(
        "bpy.ops.wm.read_factory_settings"
    )
