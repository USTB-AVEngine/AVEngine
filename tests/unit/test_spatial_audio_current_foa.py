from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from avengine.m3.runtime import RUNTIME_MODE_CURRENT_INSTALLED
from avengine.spatial_audio import current_foa
from avengine.spatial_audio.audio import read_float32_wav
from avengine.spatial_audio.runtime import FOA_LAYOUT_ID, MultiSourceRenderResult, NamedPairIR
from avengine.spatial_audio.spatial import rlr_foa_contract


REPOSITORY = Path(__file__).resolve().parents[2]
REQUEST_PATH = REPOSITORY / "examples/m4/blender_custom/multi_source_canary_request.json"


def _request(*, sample_rate_hz: int = 16_000) -> dict[str, Any]:
    value = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
    value["simulation"]["sample_rate_hz"] = sample_rate_hz
    return value


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
    request: dict[str, Any], identity: dict[str, Any]
) -> MultiSourceRenderResult:
    listener_id = request["listeners"][0]["listener_id"]
    source_ids = tuple(item["source_id"] for item in request["sources"])
    pairs = tuple(
        NamedPairIR(
            listener_id=listener_id,
            source_id=source_id,
            sample_rate_hz=16_000.0,
            samples=np.full((4, 16), index + 1, dtype="<f4"),
            layout_type="ambisonics",
            layout_id=FOA_LAYOUT_ID,
            channel_labels=("W", "Y", "Z", "X"),
            normalization="N3D",
            coordinate_frame="avengine_world",
        )
        for index, source_id in enumerate(source_ids)
    )
    return MultiSourceRenderResult(
        pairs=pairs,
        requested_source_order=source_ids,
        canonical_native_source_order=source_ids,
        listener_id=listener_id,
        sample_rate_hz=16_000.0,
        layout_type="ambisonics",
        layout_id=FOA_LAYOUT_ID,
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
    request: dict[str, Any],
    identity: dict[str, Any],
) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    def load_request(_path: str | Path) -> SimpleNamespace:
        calls["request"] = True
        return SimpleNamespace(
            request=copy.deepcopy(request),
            canonical_source_ids=tuple(
                item["source_id"] for item in request["sources"]
            ),
        )

    def load_scene(
        path: str | Path, *, allow_nonpassing_research_qa: bool
    ) -> SimpleNamespace:
        calls["allow_research_package"] = allow_nonpassing_research_qa
        return SimpleNamespace(
            manifest_path=Path(path).resolve(),
            package_id="current_mp3d_research_package",
            manifest={
                "package_mode": "research_candidate",
                "materials": {
                    "material_semantics": "research_placeholder",
                    "qualification_claim": "unqualified_research_placeholder",
                },
            },
        )

    def render(*_args: Any, **kwargs: Any) -> MultiSourceRenderResult:
        calls["render"] = kwargs
        return _render_result(request, identity)

    monkeypatch.setattr(
        current_foa, "load_and_validate_multi_source_canary_request", load_request
    )
    monkeypatch.setattr(current_foa, "load_compiled_acoustic_scene", load_scene)
    monkeypatch.setattr(current_foa, "render_named_sources", render)
    return calls


def test_current_foa_writes_raw_native_four_channel_wavs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    identity, runtime_paths = _runtime_identity(tmp_path / "runtime")
    calls = _install_fakes(monkeypatch, request=request, identity=identity)
    package_manifest = tmp_path / "package" / "manifest.json"
    package_manifest.parent.mkdir()
    package_manifest.write_text("{}", encoding="utf-8")
    output = tmp_path / "foa-output"

    receipt = current_foa.run_current_foa(
        REQUEST_PATH,
        package_manifest,
        output,
        **runtime_paths,
    )

    render = calls["render"]
    assert render["layout_type"] == "ambisonics"
    assert render["channel_count"] == 4
    assert render["hrtf_file_path"] == ""
    assert render["runtime_mode"] == RUNTIME_MODE_CURRENT_INSTALLED
    assert render["runtime_prefix"] == runtime_paths["runtime_prefix"].resolve()
    assert render["rlr_sdk_root"] == runtime_paths["rlr_sdk_root"].resolve()
    assert render["magnum_python_site"] == runtime_paths["magnum_python_site"].resolve()
    assert calls["allow_research_package"] is True
    assert receipt["status"] == "pass"
    assert receipt["research_status"] == "research_candidate"
    assert receipt["research_only"] is True
    assert receipt["qualification_claim"] is False
    assert receipt["binaural"] == "not_requested"
    assert receipt["spatial_format"] == rlr_foa_contract()
    assert receipt["render"]["mode"] == "native_rlr_foa"
    assert receipt["render"]["propagation"] == {
        "direct": True,
        "indirect": True,
        "diffraction": True,
        "transmission": False,
    }
    assert "hrtf" not in json.dumps(receipt).casefold()
    assert len(receipt["pairs"]) == 2

    for record in receipt["pairs"]:
        wav = read_float32_wav(output / record["wav"], verify_sidecar=True)
        assert wav.sample_rate_hz == 16_000
        assert wav.samples.shape == (4, 16)
        assert wav.sidecar is not None
        assert wav.sidecar["metadata"]["spatial_format"] == rlr_foa_contract()
        assert wav.sidecar["metadata"]["audio_role"] == "native_rlr_foa_pair_ir"
        assert wav.sidecar["metadata"]["binaural"] == "not_requested"

    with pytest.raises(current_foa.CurrentFOAError, match="refusing to replace"):
        current_foa.run_current_foa(
            REQUEST_PATH,
            package_manifest,
            output,
            **runtime_paths,
        )


def test_current_foa_rejects_non_16k_before_render_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(sample_rate_hz=22_050)
    identity, runtime_paths = _runtime_identity(tmp_path / "runtime")
    calls = _install_fakes(monkeypatch, request=request, identity=identity)
    output = tmp_path / "wrong-rate"

    with pytest.raises(current_foa.CurrentFOAError, match="sample_rate_hz=16000"):
        current_foa.run_current_foa(
            REQUEST_PATH,
            tmp_path / "package" / "manifest.json",
            output,
            **runtime_paths,
        )

    assert "allow_research_package" not in calls
    assert "render" not in calls
    assert not output.exists()


def test_current_foa_rejects_git_backed_runtime_before_request_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    identity, runtime_paths = _runtime_identity(tmp_path / "runtime")
    calls = _install_fakes(monkeypatch, request=request, identity=identity)
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    runtime_prefix = checkout / "installed-prefix"
    runtime_prefix.mkdir()
    output = tmp_path / "git-runtime"

    with pytest.raises(current_foa.CurrentFOAError, match="outside a Git checkout"):
        current_foa.run_current_foa(
            REQUEST_PATH,
            tmp_path / "package" / "manifest.json",
            output,
            runtime_prefix=runtime_prefix,
            rlr_sdk_root=runtime_paths["rlr_sdk_root"],
            magnum_python_site=runtime_paths["magnum_python_site"],
        )

    assert not calls
    assert not output.exists()
