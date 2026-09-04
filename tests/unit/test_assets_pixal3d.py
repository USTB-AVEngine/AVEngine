"""Focused checks for the local Pixal3D inference entry point."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest
from PIL import Image


REPOSITORY = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPOSITORY / "tools/assets/run_pixal3d_mesh.py"
SPEC = importlib.util.spec_from_file_location("run_pixal3d_mesh", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


PIXAL_SOURCE = REPOSITORY / "src/avengine/assets/pixal3d"
NAF_SOURCE = REPOSITORY / "src/avengine/assets/naf"


def test_rgba_validation_requires_a_real_cutout(tmp_path: Path) -> None:
    cutout = tmp_path / "cutout.png"
    image = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    image.putpixel((1, 1), (255, 255, 255, 255))
    image.save(cutout)
    RUNNER._validate_rgba_cutout(cutout)

    opaque = tmp_path / "opaque.png"
    Image.new("RGBA", (4, 4), (255, 255, 255, 255)).save(opaque)
    with pytest.raises(ValueError, match="non-opaque"):
        RUNNER._validate_rgba_cutout(opaque)

    rgb = tmp_path / "rgb.png"
    Image.new("RGB", (4, 4), (255, 255, 255)).save(rgb)
    with pytest.raises(ValueError, match="RGBA"):
        RUNNER._validate_rgba_cutout(rgb)


def test_model_root_overrides_are_explicit_and_type_checked(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    assert RUNNER._resolve_model_root(
        model_dir,
        name="unused",
        label="model",
        directory=True,
    ) == model_dir.resolve()
    with pytest.raises(ValueError, match="directory is missing"):
        RUNNER._resolve_model_root(
            tmp_path / "missing",
            name="unused",
            label="model",
            directory=True,
        )


def test_legacy_external_code_root_is_rejected() -> None:
    result = RUNNER.main([
        "--pixal3d-root", "/data/jzy/code/Pixal3D-lead-b",
        "--image", "/tmp/missing.png",
        "--output", "/tmp/fresh.glb",
    ])
    assert result == 2


def test_runner_uses_local_inference_module_without_model_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "cutout.png"
    image = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    image.putpixel((1, 1), (255, 255, 255, 255))
    image.save(image_path)
    output_path = tmp_path / "result.glb"
    roots = {
        "pixal3d": tmp_path / "pixal",
        "moge": tmp_path / "moge",
        "dinov3": tmp_path / "dino",
        "naf": tmp_path / "naf.pth",
    }
    for key, path in roots.items():
        if key == "naf":
            path.write_bytes(b"fixture")
        else:
            path.mkdir()
    calls: list[dict] = []

    fake_package = types.ModuleType("pixal3d")
    fake_package.__path__ = []
    fake_inference = types.ModuleType("pixal3d.inference")

    def fake_run_inference(**kwargs):
        calls.append(kwargs)

    fake_inference.run_inference = fake_run_inference
    monkeypatch.setitem(sys.modules, "pixal3d", fake_package)
    monkeypatch.setitem(sys.modules, "pixal3d.inference", fake_inference)
    monkeypatch.setattr(RUNNER, "_resolve_model_roots", lambda args: roots)

    result = RUNNER.main([
        "--image", str(image_path),
        "--output", str(output_path),
        "--fov", "0.2",
    ])
    assert result == 0
    assert len(calls) == 1
    assert calls[0]["model_path"] == roots["pixal3d"]
    assert calls[0]["moge_model_path"] == roots["moge"]
    assert calls[0]["dinov3_model_path"] == roots["dinov3"]
    assert calls[0]["naf_model_path"] == roots["naf"]


def test_local_pixal_source_has_no_external_loader_or_runner() -> None:
    sources = [
        path.read_text(encoding="utf-8")
        for path in (*PIXAL_SOURCE.rglob("*.py"), *NAF_SOURCE.rglob("*.py"))
    ]
    combined = "\n".join(sources)
    assert "runpy" not in combined
    assert "hf_hub_download" not in combined
    assert "load_state_dict_from_url" not in combined
    assert "torch.hub" not in combined


def test_model_registry_declares_all_local_pixal_inputs() -> None:
    registry = json.loads(
        (REPOSITORY / "examples/assets/model_roots_v1.json").read_text(
            encoding="utf-8"
        )
    )
    models = registry["models"]
    assert {"pixal3d", "moge_2_vitl", "dinov3_vitl16", "naf_upsampler"} <= set(models)
