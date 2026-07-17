from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from avengine.m2.glb import load_glb
from avengine.m2.glb_write import build_glb
from avengine.m2.preprocess import preprocess_glb
from avengine.m2.similarity import (
    SimilarityBakeError,
    bake_uniform_skin_ancestor_scale,
)
from tools.m2 import bake_uniform_skin_scale as bake_cli
from tools.m2 import wrap_uniform_scene_scale as wrap_cli


_HORSE = Path("assets/mesh_library/quaternius_farm/Horse.glb")
_DOG = Path("assets/mesh_library/quaternius_animalpack/Dog.glb")


def _wrapped_horse(tmp_path: Path) -> Path:
    wrapped = tmp_path / "wrapped-horse.glb"
    wrap_cli.wrap(_HORSE.resolve(), wrapped, 0.01)
    return wrapped


def _preprocessed_dog(tmp_path: Path) -> Path:
    prepared = tmp_path / "prepared-dog.glb"
    preprocess_glb(
        _DOG,
        prepared,
        action_map=[("Idle", "Idle"), ("Walking", "Walking")],
    )
    return prepared


def _rewrite_document(source_path: Path, output_path: Path, mutate: Any) -> Path:
    source = load_glb(source_path)
    document = copy.deepcopy(source.json)
    mutate(document)
    declared_length = document["buffers"][0]["byteLength"]
    output_path.write_bytes(build_glb(document, source.binary[:declared_length]))
    return output_path


def _skinned_mesh(document: dict[str, Any]) -> tuple[int, int]:
    for node_index, node in enumerate(document["nodes"]):
        if node.get("skin") == 0 and "mesh" in node:
            return node_index, node["mesh"]
    raise AssertionError("fixture has no skinned mesh")


def _wrap_cli_arguments(source: Path, output: Path, report: Path) -> list[str]:
    return [
        "--input",
        str(source),
        "--output",
        str(output),
        "--report",
        str(report),
        "--factor",
        "0.01",
    ]


def _bake_cli_arguments(source: Path, output: Path, report: Path) -> list[str]:
    return [
        "--input",
        str(source),
        "--output",
        str(output),
        "--report",
        str(report),
    ]


def test_similarity_keeps_real_horse_and_golden_rig_routes(tmp_path: Path) -> None:
    horse_report = bake_uniform_skin_ancestor_scale(
        _wrapped_horse(tmp_path), tmp_path / "baked-horse.glb"
    )
    dog_report = bake_uniform_skin_ancestor_scale(
        _preprocessed_dog(tmp_path), tmp_path / "baked-dog.glb"
    )

    assert horse_report["status"] == "pass"
    assert dog_report["status"] == "pass"
    assert dog_report["scale"]["uniform_factor"] == pytest.approx(1.215000987)


def test_similarity_rejects_unskinned_descendant_mesh_payload(
    tmp_path: Path,
) -> None:
    wrapped = _wrapped_horse(tmp_path)

    def add_unskinned_payload(document: dict[str, Any]) -> None:
        _, skinned_mesh = _skinned_mesh(document)
        duplicate_mesh = len(document["meshes"])
        document["meshes"].append(copy.deepcopy(document["meshes"][skinned_mesh]))
        payload_node = len(document["nodes"])
        document["nodes"].append({"name": "unskinned-payload", "mesh": duplicate_mesh})
        wrapper = document["scenes"][0]["nodes"][0]
        document["nodes"][wrapper]["children"].append(payload_node)

    adversarial = _rewrite_document(
        wrapped, tmp_path / "unskinned-descendant.glb", add_unskinned_payload
    )

    with pytest.raises(SimilarityBakeError, match="unskinned mesh payload"):
        bake_uniform_skin_ancestor_scale(adversarial, tmp_path / "baked.glb")

    assert not (tmp_path / "baked.glb").exists()


def test_similarity_rejects_shared_mesh_instancing(tmp_path: Path) -> None:
    wrapped = _wrapped_horse(tmp_path)

    def add_mesh_instance(document: dict[str, Any]) -> None:
        _, skinned_mesh = _skinned_mesh(document)
        instance_node = len(document["nodes"])
        document["nodes"].append(
            {"name": "external-mesh-instance", "mesh": skinned_mesh}
        )
        document["scenes"][0]["nodes"].append(instance_node)

    adversarial = _rewrite_document(
        wrapped, tmp_path / "shared-mesh.glb", add_mesh_instance
    )

    with pytest.raises(SimilarityBakeError, match="shared mesh instancing"):
        bake_uniform_skin_ancestor_scale(adversarial, tmp_path / "baked.glb")

    assert not (tmp_path / "baked.glb").exists()


@pytest.mark.parametrize("animated_target", ["scale_node", "ancestor"])
def test_similarity_rejects_scale_animation_on_bake_chain(
    tmp_path: Path, animated_target: str
) -> None:
    wrapped = _wrapped_horse(tmp_path)

    def animate_bake_chain_scale(document: dict[str, Any]) -> None:
        scale_node = document["scenes"][0]["nodes"][0]
        target_node = scale_node
        if animated_target == "ancestor":
            target_node = len(document["nodes"])
            document["nodes"].append(
                {"name": "animated-unit-ancestor", "children": [scale_node]}
            )
            document["scenes"][0]["nodes"] = [target_node]
        for animation in document["animations"]:
            for channel in animation["channels"]:
                if channel["target"].get("path") == "translation":
                    channel["target"] = {"node": target_node, "path": "scale"}
                    return
        raise AssertionError("fixture has no VEC3 translation channel")

    adversarial = _rewrite_document(
        wrapped,
        tmp_path / f"animated-{animated_target}.glb",
        animate_bake_chain_scale,
    )
    output = tmp_path / "baked.glb"

    with pytest.raises(SimilarityBakeError, match="scale animation targets"):
        bake_uniform_skin_ancestor_scale(adversarial, output)

    assert not output.exists()


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_similarity_library_refuses_existing_or_symlink_output(
    tmp_path: Path, kind: str
) -> None:
    source = _wrapped_horse(tmp_path)
    output = tmp_path / "baked.glb"
    if kind == "file":
        output.write_bytes(b"sentinel")
    else:
        output.symlink_to(tmp_path / "dangling-baked.glb")

    with pytest.raises(SimilarityBakeError, match="refusing to replace output"):
        bake_uniform_skin_ancestor_scale(source, output)

    if kind == "file":
        assert output.read_bytes() == b"sentinel"
    else:
        assert output.is_symlink()
        assert not (tmp_path / "dangling-baked.glb").exists()


@pytest.mark.parametrize("occupied", ["output", "report"])
@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_wrap_cli_preflights_both_paired_outputs(
    tmp_path: Path, occupied: str, kind: str
) -> None:
    output = tmp_path / "wrapped.glb"
    report = tmp_path / "report.json"
    path = output if occupied == "output" else report
    if kind == "file":
        path.write_bytes(b"sentinel")
    else:
        path.symlink_to(tmp_path / f"dangling-{occupied}")

    with pytest.raises(SystemExit):
        wrap_cli.main(_wrap_cli_arguments(_HORSE, output, report))

    counterpart = report if occupied == "output" else output
    assert not counterpart.exists()
    assert not counterpart.is_symlink()
    if kind == "file":
        assert path.read_bytes() == b"sentinel"
    else:
        assert path.is_symlink()


def test_wrap_cli_cleans_output_when_report_creation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "wrapped.glb"
    report = tmp_path / "report.json"
    real_write = wrap_cli._write_exclusive
    calls = 0

    def fail_second_write(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected report failure")
        real_write(path, payload)

    monkeypatch.setattr(wrap_cli, "_write_exclusive", fail_second_write)
    with pytest.raises(SystemExit):
        wrap_cli.main(_wrap_cli_arguments(_HORSE, output, report))

    assert calls == 2
    assert not output.exists()
    assert not report.exists()


@pytest.mark.parametrize("occupied", ["output", "report"])
@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_bake_cli_preflights_both_paired_outputs(
    tmp_path: Path, occupied: str, kind: str
) -> None:
    source = _wrapped_horse(tmp_path)
    output = tmp_path / "baked.glb"
    report = tmp_path / "report.json"
    path = output if occupied == "output" else report
    if kind == "file":
        path.write_bytes(b"sentinel")
    else:
        path.symlink_to(tmp_path / f"dangling-{occupied}")

    with pytest.raises(SystemExit):
        bake_cli.main(_bake_cli_arguments(source, output, report))

    counterpart = report if occupied == "output" else output
    assert not counterpart.exists()
    assert not counterpart.is_symlink()
    if kind == "file":
        assert path.read_bytes() == b"sentinel"
    else:
        assert path.is_symlink()


def test_bake_cli_cleans_output_when_report_creation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _wrapped_horse(tmp_path)
    output = tmp_path / "baked.glb"
    report = tmp_path / "report.json"

    def fail_report(_path: Path, _payload: bytes) -> None:
        raise OSError("injected report failure")

    monkeypatch.setattr(bake_cli, "_write_exclusive", fail_report)
    with pytest.raises(SystemExit):
        bake_cli.main(_bake_cli_arguments(source, output, report))

    assert not output.exists()
    assert not report.exists()
