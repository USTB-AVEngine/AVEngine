from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from avengine.contracts.transforms import compose_transforms
from avengine.acoustics.runtime import RUNTIME_MODE_CURRENT_INSTALLED
from avengine.spatial_audio import current_request_pair_ir
from avengine.spatial_audio.audio import read_float32_wav
from avengine.spatial_audio.runtime import (
    BINAURAL_LAYOUT_ID,
    FOA_LAYOUT_ID,
    MultiSourceRenderResult,
    NamedPairIR,
)


REPOSITORY = Path(__file__).resolve().parents[2]
M1_TEMPLATE = REPOSITORY / "examples/m5_1/legacy_apartment/m1_capture_request.json"
SIMULATION_TEMPLATE = (
    REPOSITORY / "examples/runtime/rir_cache_simulation_request_v2.json"
)
ROOM_ID = "current_m1_pair_ir_room_v1"


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _m1_request() -> dict[str, Any]:
    value = json.loads(M1_TEMPLATE.read_text(encoding="utf-8"))
    value["request_id"] = "current_m1_pair_ir_request_v1"
    value["room_id"] = ROOM_ID
    value["primary_camera_rig"]["world_from_rig"] = {
        "translation_m": [1.0, 2.0, 3.0],
        "rotation_xyzw": [0.0, 0.7071067811865475, 0.0, 0.7071067811865476],
    }
    identity = {
        "translation_m": [0.0, 0.0, 0.0],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    value["primary_camera_rig"]["shared_calibration"]["rig_from_sensor"] = (
        copy.deepcopy(identity)
    )
    value["listener"]["rig_from_listener"] = copy.deepcopy(identity)
    value["sources"] = [
        {
            "source_id": "source0",
            "world_from_source": {
                "translation_m": [-1.0, 0.5, 2.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
        {
            "source_id": "source1",
            "world_from_source": {
                "translation_m": [2.0, 0.5, -1.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
    ]
    return value


def _simulation_request() -> dict[str, Any]:
    return json.loads(SIMULATION_TEMPLATE.read_text(encoding="utf-8"))


def _scene(manifest: Path, *, room_id: str = ROOM_ID) -> SimpleNamespace:
    return SimpleNamespace(
        manifest_path=manifest.resolve(),
        package_id="current_m1_research_package",
        manifest={
            "source_room": {"room_id": room_id},
            "package_mode": "research_candidate",
        },
        qa_reports={
            "geometry_report": {"status": "fail"},
            "ray_report": {"status": "not_run"},
        },
    )


def _runtime_identity(root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    habitat = root / "habitat-prefix"
    magnum = root / "magnum-python-site"
    sdk = root / "rlr-sdk"
    module = habitat / "python" / "habitat_sim" / "__init__.py"
    binding = habitat / "python" / "habitat_sim" / "_ext" / "bindings.so"
    header = sdk / "headers" / "RLRAudioPropagation.h"
    library = sdk / "libs" / "linux" / "x64" / "libRLRAudioPropagation.so"
    for path in (module, binding, header, library):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    magnum.mkdir(parents=True, exist_ok=True)
    identity = {
        "identity_schema": "avengine_current_installed_rlr_runtime_v1",
        "mode": "current-installed",
        "habitat_runtime_prefix": str(habitat.resolve()),
        "habitat_sim_module": str(module.resolve()),
        "habitat_sim_binding": str(binding.resolve()),
        "magnum_python_site": str(magnum.resolve()),
        "rlr_sdk_root": str(sdk.resolve()),
        "rlr_sdk_header": str(header.resolve()),
        "rlr_sdk_library": str(library.resolve()),
        "rlr_adapter_enabled": True,
        "binding_api": "habitat_sim.RLRAcousticContext_v1",
    }
    return identity, {
        "runtime_prefix": habitat,
        "magnum_python_site": magnum,
        "rlr_sdk_root": sdk,
    }


def _render_result(
    *,
    listener_id: str,
    identity: dict[str, Any],
    layout_type: str,
) -> MultiSourceRenderResult:
    binaural = layout_type == "binaural"
    channel_count = 2 if binaural else 4
    layout_id = BINAURAL_LAYOUT_ID if binaural else FOA_LAYOUT_ID
    channel_labels = ("left", "right") if binaural else ("W", "Y", "Z", "X")
    normalization = "not_applicable" if binaural else "N3D"
    coordinate_frame = "listener_local" if binaural else "avengine_world"
    pairs = tuple(
        NamedPairIR(
            listener_id=listener_id,
            source_id=source_id,
            sample_rate_hz=16_000.0,
            samples=np.full((channel_count, 16), index + 1, dtype="<f4"),
            layout_type=layout_type,
            layout_id=layout_id,
            channel_labels=channel_labels,
            normalization=normalization,
            coordinate_frame=coordinate_frame,
        )
        for index, source_id in enumerate(("source0", "source1"))
    )
    return MultiSourceRenderResult(
        pairs=pairs,
        requested_source_order=("source0", "source1"),
        canonical_native_source_order=("source0", "source1"),
        listener_id=listener_id,
        sample_rate_hz=16_000.0,
        layout_type=layout_type,
        layout_id=layout_id,
        runtime={
            "runtime_mode": "current-installed",
            "runtime_identity": copy.deepcopy(identity),
            "binding_api": "habitat_sim.RLRAcousticContext_v1",
        },
        upload_report={},
        endpoint_receipts={},
        indirect_ray_efficiency=0.5,
        timing={},
    )


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    package_manifest: Path,
    identity: dict[str, Any],
) -> dict[str, Any]:
    calls: dict[str, Any] = {"render": []}

    def load_scene(
        path: str | Path, *, allow_nonpassing_research_qa: bool
    ) -> SimpleNamespace:
        calls["package"] = (Path(path).resolve(), allow_nonpassing_research_qa)
        return _scene(package_manifest)

    def render(*_args: Any, **kwargs: Any) -> MultiSourceRenderResult:
        calls["render"].append(kwargs)
        return _render_result(
            listener_id=kwargs["listener"].anchor_id,
            identity=identity,
            layout_type=kwargs["layout_type"],
        )

    monkeypatch.setattr(current_request_pair_ir, "load_compiled_acoustic_scene", load_scene)
    monkeypatch.setattr(current_request_pair_ir, "render_named_sources", render)
    monkeypatch.setattr(
        current_request_pair_ir,
        "_validate_sofa_native_input",
        lambda _path, *, expected_sample_rate_hz: {
            "status": "pass",
            "validator": "fixture",
            "data_ir_shape": [2, 2, 8],
            "data_sampling_rate_hz": expected_sample_rate_hz,
        },
    )
    return calls


def test_sofa_native_validation_rejects_structurally_invalid_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "invalid.sofa"
    path.write_bytes(b"not a netcdf SOFA file")
    fake_sofar = SimpleNamespace(
        __version__="test",
        read_sofa=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("invalid netCDF dimension scales")
        ),
    )
    monkeypatch.setitem(sys.modules, "sofar", fake_sofar)

    with pytest.raises(
        current_request_pair_ir.CurrentM1PairIRError,
        match="not a valid verified SOFA",
    ):
        current_request_pair_ir._validate_sofa_native_input(
            path, expected_sample_rate_hz=16_000
        )


def test_sofa_native_validation_records_verified_binaural_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "valid.sofa"
    path.write_bytes(b"fixture")
    sofa = SimpleNamespace(
        Data_IR=np.zeros((4, 2, 32), dtype=np.float64),
        Data_SamplingRate=16_000.0,
        GLOBAL_SOFAConventions="SimpleFreeFieldHRIR",
        GLOBAL_SOFAConventionsVersion="1.0",
    )
    fake_sofar = SimpleNamespace(
        __version__="1.2.3",
        read_sofa=lambda *_args, **_kwargs: sofa,
    )
    monkeypatch.setitem(sys.modules, "sofar", fake_sofar)

    assert current_request_pair_ir._validate_sofa_native_input(
        path, expected_sample_rate_hz=16_000
    ) == {
        "status": "pass",
        "validator": "sofar",
        "validator_version": "1.2.3",
        "sofa_convention": "SimpleFreeFieldHRIR",
        "sofa_convention_version": "1.0",
        "data_ir_shape": [4, 2, 32],
        "data_sampling_rate_hz": 16_000,
    }


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    m1 = _write_json(tmp_path / "m1.json", _m1_request())
    simulation = _write_json(tmp_path / "simulation.json", _simulation_request())
    package = tmp_path / "package" / "manifest.json"
    return m1, simulation, package


def test_shared_loader_composes_static_m1_endpoints_and_allows_research_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m1, simulation, package = _inputs(tmp_path)
    calls: dict[str, Any] = {}

    def load_scene(
        path: str | Path, *, allow_nonpassing_research_qa: bool
    ) -> SimpleNamespace:
        calls["package"] = (Path(path).resolve(), allow_nonpassing_research_qa)
        return _scene(package)

    monkeypatch.setattr(current_request_pair_ir, "load_compiled_acoustic_scene", load_scene)
    loaded = current_request_pair_ir.load_current_m1_pair_ir_inputs(m1, simulation, package)

    expected_listener = compose_transforms(
        loaded.m1_request["primary_camera_rig"]["world_from_rig"],
        loaded.m1_request["listener"]["rig_from_listener"],
    )
    assert calls["package"] == (package.resolve(), True)
    assert loaded.room_id == ROOM_ID
    assert [source.anchor_id for source in loaded.sources] == ["source0", "source1"]
    assert [source.position_m for source in loaded.sources] == [
        (-1.0, 0.5, 2.0),
        (2.0, 0.5, -1.0),
    ]
    assert loaded.listener.position_m == pytest.approx(
        expected_listener["translation_m"]
    )
    x, y, z, w = expected_listener["rotation_xyzw"]
    assert loaded.listener.orientation_wxyz == pytest.approx((w, x, y, z))
    assert loaded.simulation.sample_rate_hz == 16_000
    assert loaded.simulation_request["qualification_claim"] is False


def test_shared_loader_rejects_malformed_m1_before_package_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _m1_request()
    request["listener"]["listener_id"] = "wrong"
    m1 = _write_json(tmp_path / "m1.json", request)
    simulation = _write_json(tmp_path / "simulation.json", _simulation_request())
    called = False

    def load_scene(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        nonlocal called
        called = True
        return _scene(tmp_path / "package.json")

    monkeypatch.setattr(current_request_pair_ir, "load_compiled_acoustic_scene", load_scene)
    with pytest.raises(
        current_request_pair_ir.CurrentM1PairIRError,
        match="before package load",
    ):
        current_request_pair_ir.load_current_m1_pair_ir_inputs(
            m1, simulation, tmp_path / "package.json"
        )
    assert called is False


def test_shared_loader_rejects_package_room_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m1, simulation, package = _inputs(tmp_path)
    monkeypatch.setattr(
        current_request_pair_ir,
        "load_compiled_acoustic_scene",
        lambda *_args, **_kwargs: _scene(package, room_id="different_room"),
    )
    with pytest.raises(
        current_request_pair_ir.CurrentM1PairIRError,
        match="source_room.room_id",
    ):
        current_request_pair_ir.load_current_m1_pair_ir_inputs(m1, simulation, package)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(schema="wrong"), "schema must be exactly"),
        (
            lambda value: value.update(qualification_claim=True),
            "qualification_claim=false",
        ),
        (
            lambda value: value["simulation"].update(sample_rate_hz=44_100),
            "sample_rate_hz=16000",
        ),
    ],
)
def test_shared_loader_rejects_wrong_simulation_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
    message: str,
) -> None:
    m1 = _write_json(tmp_path / "m1.json", _m1_request())
    value = _simulation_request()
    mutation(value)
    simulation = _write_json(tmp_path / "simulation.json", value)
    monkeypatch.setattr(
        current_request_pair_ir,
        "load_compiled_acoustic_scene",
        lambda *_args, **_kwargs: _scene(tmp_path / "package.json"),
    )
    with pytest.raises(current_request_pair_ir.CurrentM1PairIRError, match=message):
        current_request_pair_ir.load_current_m1_pair_ir_inputs(
            m1, simulation, tmp_path / "package.json"
        )


def test_current_m1_foa_writes_two_raw_pairs_without_hrtf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m1, simulation, package = _inputs(tmp_path)
    identity, runtime_paths = _runtime_identity(tmp_path / "runtime")
    calls = _install_fakes(
        monkeypatch,
        package_manifest=package,
        identity=identity,
    )
    output = tmp_path / "foa-output"

    receipt = current_request_pair_ir.run_current_m1_foa(
        m1,
        simulation,
        package,
        output,
        **runtime_paths,
    )

    assert len(calls["render"]) == 1
    render = calls["render"][0]
    assert render["layout_type"] == "ambisonics"
    assert render["channel_count"] == 4
    assert render["hrtf_file_path"] == ""
    assert render["runtime_mode"] == RUNTIME_MODE_CURRENT_INSTALLED
    assert receipt["status"] == "pass"
    assert receipt["research_only"] is True
    assert receipt["episode_counted"] is False
    assert receipt["formal_dataset_count"] == 0
    assert receipt["qualification"] is False
    assert receipt["qualification_claim"] is False
    assert receipt["hrtf_used"] is False
    assert receipt["authority"]["static_m1_sources"] is True
    assert receipt["authority"]["dynamic_actor_anchor_claim"] is False
    assert receipt["authority"]["m2_anchor_evidence_claim"] is False
    assert receipt["research_package_qa"] == {
        "admission": "nonpassing_research_override",
        "statuses": {
            "geometry_report": "fail",
            "ray_report": "not_run",
        },
        "formal_qualification": False,
        "dataset_admission": False,
    }
    assert len(receipt["pairs"]) == 2
    for record in receipt["pairs"]:
        wav = read_float32_wav(output / record["wav"], verify_sidecar=True)
        assert wav.sample_rate_hz == 16_000
        assert wav.samples.shape == (4, 16)
        assert wav.sidecar is not None
        assert wav.sidecar["metadata"]["hrtf_used"] is False

    with pytest.raises(
        current_request_pair_ir.CurrentM1PairIRError,
        match="refusing to replace",
    ):
        current_request_pair_ir.run_current_m1_foa(
            m1,
            simulation,
            package,
            output,
            **runtime_paths,
        )
    assert len(calls["render"]) == 1


def test_current_m1_binaural_preflights_and_writes_two_lr_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m1, simulation, package = _inputs(tmp_path)
    identity, runtime_paths = _runtime_identity(tmp_path / "runtime")
    calls = _install_fakes(
        monkeypatch,
        package_manifest=package,
        identity=identity,
    )
    hrtf = tmp_path / "hrtf" / "derived_16k.sofa"
    license_path = tmp_path / "hrtf" / "LICENSE.txt"
    hrtf.parent.mkdir()
    hrtf.write_bytes(b"explicit 16 kHz SOFA fixture")
    license_path.write_bytes(b"fixture license")
    output = tmp_path / "binaural-output"

    receipt = current_request_pair_ir.run_current_m1_binaural(
        m1,
        simulation,
        package,
        output,
        **runtime_paths,
        hrtf_path=hrtf,
        expected_hrtf_sha256=hashlib.sha256(hrtf.read_bytes()).hexdigest(),
        hrtf_sample_rate_hz=16_000,
        hrtf_license_path=license_path,
        expected_hrtf_license_sha256=hashlib.sha256(
            license_path.read_bytes()
        ).hexdigest(),
        hrtf_license_id="fixture-license",
        hrtf_citation="Fixture citation",
    )

    assert len(calls["render"]) == 1
    render = calls["render"][0]
    assert render["layout_type"] == "binaural"
    assert render["channel_count"] == 2
    assert render["hrtf_file_path"] == str(hrtf.absolute())
    assert receipt["qualification"] is False
    assert receipt["qualification_claim"] is False
    assert receipt["hrtf_preflight"]["status"] == "pass"
    assert receipt["sofa_native_compatibility"]["status"] == "pass"
    assert receipt["hrtf_preflight"]["native_cardinal_validation"] == "not_run"
    assert receipt["spatial_format"]["channel_layout"] == {
        "type": "binaural",
        "channel_count": 2,
        "channel_order": ["left", "right"],
    }
    for record in receipt["pairs"]:
        wav = read_float32_wav(output / record["wav"], verify_sidecar=True)
        assert wav.sample_rate_hz == 16_000
        assert wav.samples.shape == (2, 16)
        assert wav.sidecar is not None
        assert wav.sidecar["metadata"]["channel_layout"]["channel_order"] == [
            "left",
            "right",
        ]


def test_current_m1_binaural_stops_on_strict_rate_mismatch_before_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m1, simulation, package = _inputs(tmp_path)
    identity, runtime_paths = _runtime_identity(tmp_path / "runtime")
    calls = _install_fakes(
        monkeypatch,
        package_manifest=package,
        identity=identity,
    )
    hrtf = tmp_path / "hrtf.sofa"
    license_path = tmp_path / "LICENSE.txt"
    hrtf.write_bytes(b"44.1 kHz fixture")
    license_path.write_bytes(b"fixture license")
    output = tmp_path / "blocked-output"

    with pytest.raises(
        current_request_pair_ir.CurrentM1PairIRBlockedError,
        match="does not equal render sample rate",
    ):
        current_request_pair_ir.run_current_m1_binaural(
            m1,
            simulation,
            package,
            output,
            **runtime_paths,
            hrtf_path=hrtf,
            expected_hrtf_sha256=hashlib.sha256(hrtf.read_bytes()).hexdigest(),
            hrtf_sample_rate_hz=44_100,
            hrtf_license_path=license_path,
            expected_hrtf_license_sha256=hashlib.sha256(
                license_path.read_bytes()
            ).hexdigest(),
            hrtf_license_id="fixture-license",
            hrtf_citation="Fixture citation",
        )
    assert calls["render"] == []
    assert not output.exists()


def test_current_m1_binaural_rejects_invalid_sofa_before_package_or_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m1, simulation, package = _inputs(tmp_path)
    identity, runtime_paths = _runtime_identity(tmp_path / "runtime")
    calls = _install_fakes(
        monkeypatch,
        package_manifest=package,
        identity=identity,
    )
    hrtf = tmp_path / "invalid.sofa"
    license_path = tmp_path / "LICENSE.txt"
    hrtf.write_bytes(b"invalid SOFA fixture")
    license_path.write_bytes(b"fixture license")
    monkeypatch.setattr(
        current_request_pair_ir,
        "_validate_sofa_native_input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            current_request_pair_ir.CurrentM1PairIRError(
                "declared HRTF is not a valid verified SOFA file"
            )
        ),
    )
    output = tmp_path / "invalid-sofa-output"

    with pytest.raises(
        current_request_pair_ir.CurrentM1PairIRError,
        match="not a valid verified SOFA",
    ):
        current_request_pair_ir.run_current_m1_binaural(
            m1,
            simulation,
            package,
            output,
            **runtime_paths,
            hrtf_path=hrtf,
            expected_hrtf_sha256=hashlib.sha256(hrtf.read_bytes()).hexdigest(),
            hrtf_sample_rate_hz=16_000,
            hrtf_license_path=license_path,
            expected_hrtf_license_sha256=hashlib.sha256(
                license_path.read_bytes()
            ).hexdigest(),
            hrtf_license_id="fixture-license",
            hrtf_citation="Fixture citation",
        )

    assert "package" not in calls
    assert calls["render"] == []
    assert not output.exists()
