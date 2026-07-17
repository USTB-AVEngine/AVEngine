from __future__ import annotations

import copy
from pathlib import Path
import struct
from typing import Any, Sequence

import pytest

from avengine.m2.glb import extract_actions, load_glb
from avengine.m2.glb_write import build_glb
from avengine.m2.timing import ActionTimingError, retime_glb_actions
from tools.m2 import retime_actions as retime_actions_cli


def _append_accessor(
    document: dict[str, Any],
    binary: bytearray,
    *,
    element_type: str,
    values: Sequence[Sequence[float]],
) -> int:
    component_count = {"SCALAR": 1, "VEC3": 3}[element_type]
    binary.extend(b"\0" * ((-len(binary)) % 4))
    offset = len(binary)
    packer = struct.Struct("<" + "f" * component_count)
    for value in values:
        binary.extend(packer.pack(*value))
    view_index = len(document.setdefault("bufferViews", []))
    document["bufferViews"].append(
        {"buffer": 0, "byteOffset": offset, "byteLength": len(binary) - offset}
    )
    accessor_index = len(document.setdefault("accessors", []))
    document["accessors"].append(
        {
            "bufferView": view_index,
            "componentType": 5126,
            "count": len(values),
            "type": element_type,
        }
    )
    return accessor_index


def _write_animation_fixture(tmp_path: Path, interpolation: str = "LINEAR") -> Path:
    document: dict[str, Any] = {
        "asset": {"version": "2.0"},
        "nodes": [{"name": "Mover"}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    binary = bytearray()
    timestamps = _append_accessor(
        document,
        binary,
        element_type="SCALAR",
        values=[(0.0,), (1.0,)],
    )
    if interpolation == "CUBICSPLINE":
        samples = [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ]
    else:
        samples = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    output = _append_accessor(document, binary, element_type="VEC3", values=samples)
    document["animations"] = [
        {
            "name": "Move",
            "samplers": [
                {
                    "input": timestamps,
                    "output": output,
                    "interpolation": interpolation,
                }
            ],
            "channels": [{"sampler": 0, "target": {"node": 0, "path": "translation"}}],
        }
    ]
    document["buffers"] = [{"byteLength": len(binary)}]
    source = tmp_path / f"source-{interpolation}.glb"
    source.write_bytes(build_glb(document, binary))
    return source


def _cli_arguments(source: Path, output: Path, report: Path) -> list[str]:
    return [
        "--input",
        str(source),
        "--output",
        str(output),
        "--report",
        str(report),
        "--duration",
        "Move=2.5",
    ]


@pytest.mark.parametrize("interpolation", ["LINEAR", "STEP"])
def test_retime_accepts_only_real_linear_and_step_routes(
    tmp_path: Path, interpolation: str
) -> None:
    source = _write_animation_fixture(tmp_path, interpolation)
    output = tmp_path / "retimed.glb"

    report = retime_glb_actions(source, output, durations_seconds={"Move": 2.5})

    action = extract_actions(load_glb(output))[0]
    assert action.channels[0].interpolation == interpolation
    assert action.duration_seconds == pytest.approx(2.5)
    assert report["actions"][0]["output_duration_seconds_requested"] == 2.5


def test_retime_rejects_cubic_spline_before_creating_output(tmp_path: Path) -> None:
    source = _write_animation_fixture(tmp_path, "CUBICSPLINE")
    output = tmp_path / "retimed.glb"

    with pytest.raises(ActionTimingError, match="CUBICSPLINE.*only LINEAR and STEP"):
        retime_glb_actions(source, output, durations_seconds={"Move": 2.5})

    assert not output.exists()


def test_retime_rejects_even_unreferenced_cubic_sampler(tmp_path: Path) -> None:
    source_path = _write_animation_fixture(tmp_path)
    source = load_glb(source_path)
    document = copy.deepcopy(source.json)
    original_sampler = document["animations"][0]["samplers"][0]
    document["animations"][0]["samplers"].append(
        {**original_sampler, "interpolation": "CUBICSPLINE"}
    )
    declared_length = document["buffers"][0]["byteLength"]
    adversarial = tmp_path / "unused-cubic.glb"
    adversarial.write_bytes(build_glb(document, source.binary[:declared_length]))

    with pytest.raises(ActionTimingError, match="CUBICSPLINE.*only LINEAR and STEP"):
        retime_glb_actions(
            adversarial,
            tmp_path / "retimed.glb",
            durations_seconds={"Move": 2.5},
        )


def test_retime_rejects_cubic_sampler_in_untargeted_action(tmp_path: Path) -> None:
    source_path = _write_animation_fixture(tmp_path)
    source = load_glb(source_path)
    document = copy.deepcopy(source.json)
    other = copy.deepcopy(document["animations"][0])
    other["name"] = "Other"
    other["samplers"].append({**other["samplers"][0], "interpolation": "CUBICSPLINE"})
    document["animations"].append(other)
    declared_length = document["buffers"][0]["byteLength"]
    adversarial = tmp_path / "untargeted-cubic.glb"
    adversarial.write_bytes(build_glb(document, source.binary[:declared_length]))

    with pytest.raises(ActionTimingError, match="Other.*CUBICSPLINE"):
        retime_glb_actions(
            adversarial,
            tmp_path / "retimed.glb",
            durations_seconds={"Move": 2.5},
        )


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_retime_library_refuses_existing_or_symlink_output(
    tmp_path: Path, kind: str
) -> None:
    source = _write_animation_fixture(tmp_path)
    output = tmp_path / "retimed.glb"
    if kind == "file":
        output.write_bytes(b"sentinel")
    else:
        output.symlink_to(tmp_path / "dangling-output.glb")

    with pytest.raises(ActionTimingError, match="refusing to replace output"):
        retime_glb_actions(source, output, durations_seconds={"Move": 2.5})

    if kind == "file":
        assert output.read_bytes() == b"sentinel"
    else:
        assert output.is_symlink()
        assert not (tmp_path / "dangling-output.glb").exists()


@pytest.mark.parametrize("occupied", ["output", "report"])
@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_retime_cli_preflights_both_paired_outputs(
    tmp_path: Path, occupied: str, kind: str
) -> None:
    source = _write_animation_fixture(tmp_path)
    output = tmp_path / "retimed.glb"
    report = tmp_path / "report.json"
    path = output if occupied == "output" else report
    if kind == "file":
        path.write_bytes(b"sentinel")
    else:
        path.symlink_to(tmp_path / f"dangling-{occupied}")

    with pytest.raises(SystemExit):
        retime_actions_cli.main(_cli_arguments(source, output, report))

    counterpart = report if occupied == "output" else output
    assert not counterpart.exists()
    assert not counterpart.is_symlink()
    if kind == "file":
        assert path.read_bytes() == b"sentinel"
    else:
        assert path.is_symlink()


def test_retime_cli_cleans_output_when_report_creation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_animation_fixture(tmp_path)
    output = tmp_path / "retimed.glb"
    report = tmp_path / "report.json"

    def fail_report(_path: Path, _payload: bytes) -> None:
        raise OSError("injected report failure")

    monkeypatch.setattr(retime_actions_cli, "_write_exclusive", fail_report)
    with pytest.raises(SystemExit):
        retime_actions_cli.main(_cli_arguments(source, output, report))

    assert not output.exists()
    assert not report.exists()
