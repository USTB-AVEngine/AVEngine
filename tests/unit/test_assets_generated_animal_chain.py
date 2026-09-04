"""Checks for the AVEngine-owned generated-animal helper closure."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[2]
ASSET_TOOLS = REPOSITORY / "tools/assets"
CHAIN = ASSET_TOOLS / "run_generated_animal_chain.sh"
LOCAL_HELPERS = (
    "blender_normalize_generated_animal_heading.py",
    "blender_level_generated_animal_support_plane.py",
    "blender_retarget_quaternius_to_generated_quadruped.py",
)
LOCAL_SUPPORT_MODULES = (
    "generated_animal_support_plane.py",
    "generated_animal_support_plane_contract.py",
    "generated_quadruped_semantics.py",
)


def test_chain_invokes_local_animal_helpers_and_ignores_legacy_spear_root() -> None:
    source = CHAIN.read_text(encoding="utf-8")
    assert 'cd "$SPEAR_ROOT"' not in source
    assert "$SPEAR_ROOT/tools/" not in source
    assert "--spear-root is deprecated and ignored" in source
    for name in LOCAL_HELPERS:
        assert f'"$AVENGINE_ROOT/tools/assets/{name}"' in source


def test_migrated_helper_imports_stay_inside_asset_tools_directory() -> None:
    for name in (
        "blender_level_generated_animal_support_plane.py",
        "blender_retarget_quaternius_to_generated_quadruped.py",
        "generated_animal_support_plane.py",
    ):
        source = (ASSET_TOOLS / name).read_text(encoding="utf-8")
        assert "from tools.generated_" not in source
    assert all((ASSET_TOOLS / name).is_file() for name in LOCAL_HELPERS)
    assert all((ASSET_TOOLS / name).is_file() for name in LOCAL_SUPPORT_MODULES)


def test_chain_and_migrated_helpers_are_syntactically_valid() -> None:
    subprocess.run(["bash", "-n", str(CHAIN)], check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            *(str(ASSET_TOOLS / name) for name in (*LOCAL_HELPERS, *LOCAL_SUPPORT_MODULES)),
        ],
        check=True,
    )
