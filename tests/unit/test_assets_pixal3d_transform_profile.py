"""Class-level Pixal3D transform profile: recovered, not per-asset-id."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
PROFILE_PATH = REPOSITORY / "examples/assets/pixal3d_transform_profile_v1.json"
LOADER_PATH = REPOSITORY / "tools/assets/pixal3d_transform_profile.py"
RUNNER_PATH = REPOSITORY / "tools/assets/run_pixal3d_mesh.py"
CHAIN = REPOSITORY / "tools/assets/run_generated_animal_chain.sh"
HEADING = REPOSITORY / "tools/assets/blender_normalize_generated_animal_heading.py"
INFERENCE = REPOSITORY / "src/avengine/assets/pixal3d/inference.py"
POLICY = REPOSITORY / "examples/assets/generated_animal_review_policy_v1.json"
HISTORICAL_EXPORT = [
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]
HISTORICAL_BIPED_ROOT = [
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def _load_loader():
    spec = importlib.util.spec_from_file_location("pixal3d_transform_profile", LOADER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_profile_matches_historical_export_and_biped_root() -> None:
    loader = _load_loader()
    profile = loader.load_profile(PROFILE_PATH)
    assert loader.mesh_export_matrix(profile) == HISTORICAL_EXPORT
    assert loader.body_class_root_matrix(profile, "biped") == HISTORICAL_BIPED_ROOT
    assert loader.body_class_root_matrix(profile, "quadruped") == HISTORICAL_BIPED_ROOT
    assert loader.body_class_target_front_axis(profile, "quadruped") == "positive-x"
    assert loader.body_class_target_front_axis(profile, "biped") == "negative-y"
    blender_import = profile["blender_import"]
    assert blender_import["operator"] == "bpy.ops.import_scene.gltf"
    assert blender_import["gltf_up"] == "+Y"


def test_quadruped_target_front_matches_animal_review_policy() -> None:
    loader = _load_loader()
    profile = loader.load_profile(PROFILE_PATH)
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    assert (
        loader.body_class_target_front_axis(profile, "quadruped")
        == policy["common"]["retarget"]["target_front_axis"]
    )


def test_profile_rejects_asset_id_selectors(tmp_path: Path) -> None:
    loader = _load_loader()
    document = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    document["body_classes"]["asset_id"] = {"generated_siamese_standard_seal_point_research_v1": {}}
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(loader.Pixal3DTransformProfileError, match="asset id"):
        loader.load_profile(bad)


def test_runner_and_inference_consume_profile_export_transform() -> None:
    runner = RUNNER_PATH.read_text(encoding="utf-8")
    inference = INFERENCE.read_text(encoding="utf-8")
    heading = HEADING.read_text(encoding="utf-8")
    chain = CHAIN.read_text(encoding="utf-8")
    assert "export_transform=mesh_export_matrix(profile)" in runner
    assert "--transform-profile" in runner
    assert "export_transform=None" in inference
    assert "HISTORICAL_PIXAL3D_MESH_EXPORT_TRANSFORM" in inference
    assert "--transform-profile" in heading
    assert "--body-class" in heading
    assert "pixal3d_transform_profile_v1.json" in chain
    assert "--body-class quadruped" in chain


def test_loader_and_profile_are_syntactically_valid() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            str(LOADER_PATH),
            str(REPOSITORY / "tools/assets/blender_apply_pixal3d_import_transform.py"),
            str(REPOSITORY / "tools/assets/blender_render_generated_asset_review.py"),
            str(REPOSITORY / "tools/assets/build_generated_asset_visual_review_page.py"),
        ],
        check=True,
    )
    json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
