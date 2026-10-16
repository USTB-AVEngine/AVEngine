from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from avengine.assets.glb import decode_accessor, load_glb
from tools.assets import spike_habitat_local_tr as spike


def _write_glb(tmp_path: Path) -> Path:
    path = tmp_path / "probe.glb"
    path.write_bytes(spike.build_probe_glb())
    return path


def _animation_outputs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    document = load_glb(path)
    animation = document.json["animations"][0]
    by_path = {
        channel["target"]["path"]: animation["samplers"][channel["sampler"]]["output"]
        for channel in animation["channels"]
    }
    translation = np.asarray(
        decode_accessor(document, by_path["translation"]).values[-1]
    )
    rotation = np.asarray(decode_accessor(document, by_path["rotation"]).values[-1])
    rotation /= np.linalg.norm(rotation)
    return translation, rotation


def test_glb_encodes_oblique_dynamic_local_translation_and_rotation(
    tmp_path: Path,
) -> None:
    path = _write_glb(tmp_path)
    document = load_glb(path)
    translation, rotation = _animation_outputs(path)

    skin_names = [
        document.json["nodes"][index]["name"]
        for index in document.json["skins"][0]["joints"]
    ]
    assert skin_names == [spike.ROOT_LINK, spike.CHILD_LINK]
    assert document.json["animations"][0]["name"] == "LocalTRTarget"
    assert translation == pytest.approx(spike.TARGET_TRANSLATION, abs=5.0e-8)
    assert rotation == pytest.approx(spike.target_rotation_xyzw(), abs=5.0e-8)
    assert np.all(np.abs(translation) > 0.1)
    assert len({round(abs(float(value)), 3) for value in translation}) == 3
    assert np.all(np.abs(rotation[:3]) > 0.1)
    assert not math.isclose(float(rotation[3]), 1.0, abs_tol=0.01)


def test_urdf_expands_xyz_prismatic_chain_before_same_named_spherical() -> None:
    root = ET.fromstring(spike.render_probe_urdf())
    joints = root.findall("joint")
    links = [element.attrib["name"] for element in root.findall("link")]

    assert links == [spike.ROOT_LINK, *spike.DUMMY_LINKS, spike.CHILD_LINK]
    assert [joint.attrib["type"] for joint in joints] == [
        "prismatic",
        "prismatic",
        "prismatic",
        "spherical",
    ]
    assert [joint.find("axis").attrib["xyz"] for joint in joints[:3]] == [
        "1 0 0",
        "0 1 0",
        "0 0 1",
    ]
    for joint, target in zip(joints[:3], spike.TARGET_TRANSLATION, strict=True):
        limit = joint.find("limit")
        assert limit is not None
        lower = float(limit.attrib["lower"])
        upper = float(limit.attrib["upper"])
        assert lower == spike.PRISMATIC_LIMIT_LOWER
        assert upper == spike.PRISMATIC_LIMIT_UPPER
        assert lower < target < upper
        assert float(limit.attrib["effort"]) > 0.0
        assert float(limit.attrib["velocity"]) > 0.0
    assert joints[-1].find("child").attrib["link"] == spike.CHILD_LINK
    assert joints[-1].find("parent").attrib["link"] == spike.DUMMY_LINKS[-1]
    assert not set(spike.DUMMY_LINKS) & {spike.ROOT_LINK, spike.CHILD_LINK}


def test_expanded_chain_matrix_is_exact_gltf_tr_in_float32_domain(
    tmp_path: Path,
) -> None:
    translation, rotation = _animation_outputs(_write_glb(tmp_path))

    expected = spike.transform_matrix(translation, rotation)
    expanded = spike.expanded_chain_matrix(translation, rotation)

    assert np.max(np.abs(expanded - expected)) < 1.0e-12
    assert expanded[:3, 3] == pytest.approx(translation, abs=1.0e-12)
    assert not np.allclose(expanded[:3, :3], np.eye(3))


def test_ao_config_opts_into_native_skin_frame_without_qualification() -> None:
    config = spike.probe_ao_config()

    assert config["render_mode"] == "skin"
    assert config["user_defined"] == {"avengine_native_gltf_skin_frame": True}
    assert config["render_asset"] == "local_tr_probe.glb"
    assert config["urdf_filepath"] == "local_tr_probe.urdf"
