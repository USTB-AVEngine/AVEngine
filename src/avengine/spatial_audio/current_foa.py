"""Current installed-prefix writer for raw native RLR FOA pair IRs.

This deliberately small research writer is separate from the retained M4
canary.  The latter proves both the raw FOA and native-binaural paths and its
v1/v2 evidence readers consequently require an explicitly selected HRTF.  A
raw FOA pair IR is already native RLR output, however: it has four channels in
the declared AVEngine world axes and needs neither an HRTF nor a binaural
projection.  Keeping this entrypoint separate means that producing that raw
research artifact cannot relax any of the binaural receipt requirements.
"""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Any, Mapping

from avengine.backends.rlr.sdk import ExternalRlrSdkError, require_outside_git_checkout
from avengine.contracts.json_io import write_json
from avengine.current_installed_runtime import is_current_installed_runtime_identity
from avengine.m3.runtime import (
    RUNTIME_MODE_CURRENT_INSTALLED,
    RuntimeAnchor,
    RuntimeContractError,
    RuntimeUnavailableError,
    load_compiled_acoustic_scene,
)
from avengine.spatial_audio.audio import AudioContractError, write_float32_wav
from avengine.spatial_audio.contracts import (
    M4ContractError,
    load_and_validate_multi_source_canary_request,
)
from avengine.spatial_audio.runtime import (
    FOA_LAYOUT_ID,
    M4SimulationConfig,
    MultiSourceRenderResult,
    render_named_sources,
)
from avengine.spatial_audio.spatial import (
    RLR_FOA_CHANNEL_COUNT,
    RLR_FOA_CHANNEL_ORDER,
    RLR_FOA_COORDINATE_FRAME,
    RLR_FOA_NORMALIZATION,
    SpatialContractError,
    rlr_foa_contract,
    rlr_foa_wav_metadata,
    validate_foa_samples,
)


CURRENT_FOA_SAMPLE_RATE_HZ = 16_000


class CurrentFOAError(ValueError):
    """A raw current RLR FOA research render is not safe to publish."""


def _require_external_directory(value: str | Path | None, *, option: str) -> Path:
    if value is None or not str(value).strip():
        raise CurrentFOAError(f"run-current-foa requires explicit {option}")
    try:
        resolved = require_outside_git_checkout(value, owner=option)
    except (ExternalRlrSdkError, OSError, RuntimeError, ValueError) as error:
        raise CurrentFOAError(str(error)) from error
    if not resolved.is_dir():
        raise CurrentFOAError(f"{option} is not a directory: {resolved}")
    return resolved


def _current_runtime_inputs(
    *,
    runtime_prefix: str | Path | None,
    rlr_sdk_root: str | Path | None,
    magnum_python_site: str | Path | None,
) -> dict[str, Path]:
    """Resolve all current native inputs before a context can be created."""

    return {
        "runtime_prefix": _require_external_directory(
            runtime_prefix, option="--runtime-prefix"
        ),
        "rlr_sdk_root": _require_external_directory(
            rlr_sdk_root, option="--rlr-sdk-root"
        ),
        "magnum_python_site": _require_external_directory(
            magnum_python_site, option="--magnum-python-site"
        ),
    }


def _fresh_output_directory(path: str | Path) -> Path:
    requested = Path(path).expanduser()
    if os.path.lexists(requested):
        raise CurrentFOAError(f"refusing to replace current FOA output: {requested}")
    output = requested.resolve()
    if os.path.lexists(output):
        raise CurrentFOAError(f"refusing to replace current FOA output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir()
    except FileExistsError as error:
        raise CurrentFOAError(f"refusing to replace current FOA output: {output}") from error
    return output


def _anchors(request: Mapping[str, Any]) -> tuple[tuple[RuntimeAnchor, ...], RuntimeAnchor]:
    """Reconstruct the already validated M4 endpoint declarations."""

    try:
        sources = tuple(
            RuntimeAnchor.from_mapping(
                {
                    "id": item["source_id"],
                    "position_m": item["position_m"],
                    "radius_m": item["radius_m"],
                },
                listener=False,
            )
            for item in request["sources"]
        )
        listener_item = request["listeners"][0]
        listener = RuntimeAnchor.from_mapping(
            {
                "id": listener_item["listener_id"],
                "position_m": listener_item["position_m"],
                "radius_m": listener_item["radius_m"],
                "orientation_wxyz": listener_item["orientation_wxyz"],
            },
            listener=True,
        )
    except (IndexError, KeyError, RuntimeContractError, TypeError, ValueError) as error:
        raise CurrentFOAError(f"validated M4 endpoint declarations are unavailable: {error}") from error
    return sources, listener


def _foa_simulation(request: Mapping[str, Any]) -> M4SimulationConfig:
    try:
        simulation = M4SimulationConfig.from_mapping(request["simulation"])
    except (KeyError, RuntimeContractError, TypeError, ValueError) as error:
        raise CurrentFOAError(f"M4 simulation is unavailable: {error}") from error
    if (
        simulation.channel_layout.layout_type != "ambisonics"
        or simulation.channel_layout.channel_count != RLR_FOA_CHANNEL_COUNT
    ):
        raise CurrentFOAError(
            "run-current-foa requires request simulation channel_layout "
            "ambisonics/4"
        )
    if not math.isclose(
        float(simulation.sample_rate_hz),
        float(CURRENT_FOA_SAMPLE_RATE_HZ),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise CurrentFOAError(
            "run-current-foa requires native sample_rate_hz=16000; "
            "it never resamples raw RLR FOA output"
        )
    return simulation


def _validate_raw_foa_result(
    result: MultiSourceRenderResult,
    *,
    source_ids: tuple[str, ...],
    listener_id: str,
) -> None:
    if (
        result.layout_type != "ambisonics"
        or result.layout_id != FOA_LAYOUT_ID
        or not math.isclose(
            float(result.sample_rate_hz),
            float(CURRENT_FOA_SAMPLE_RATE_HZ),
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise CurrentFOAError("native output is not raw 16 kHz RLR FOA")
    if tuple(pair.source_id for pair in result.pairs) != source_ids:
        raise CurrentFOAError("native FOA pair source order differs from M4 request")
    if any(pair.listener_id != listener_id for pair in result.pairs):
        raise CurrentFOAError("native FOA pair listener differs from M4 request")
    for pair in result.pairs:
        if (
            pair.layout_type != "ambisonics"
            or pair.layout_id != FOA_LAYOUT_ID
            or pair.channel_labels != RLR_FOA_CHANNEL_ORDER
            or pair.normalization != RLR_FOA_NORMALIZATION
            or pair.coordinate_frame != RLR_FOA_COORDINATE_FRAME
            or not math.isclose(
                float(pair.sample_rate_hz),
                float(CURRENT_FOA_SAMPLE_RATE_HZ),
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise CurrentFOAError("native FOA pair metadata differs from the M4 contract")
        try:
            validate_foa_samples(pair.samples, channel_axis=0)
        except SpatialContractError as error:
            raise CurrentFOAError(f"native FOA samples are invalid: {error}") from error
    runtime_identity = result.runtime.get("runtime_identity")
    if not is_current_installed_runtime_identity(runtime_identity):
        raise CurrentFOAError(
            "native FOA result lacks one canonical non-Git current-installed "
            "Habitat/RLR adapter identity"
        )


def _raw_foa_receipt(
    *,
    output: Path,
    request: Mapping[str, Any],
    simulation: M4SimulationConfig,
    scene: Any,
    result: MultiSourceRenderResult,
) -> dict[str, Any]:
    """Write only raw native FOA pair IRs and a non-formal research record."""

    pairs: list[dict[str, Any]] = []
    for pair in result.pairs:
        wav = write_float32_wav(
            output / "raw_ir" / "foa" / f"{pair.source_id}.wav",
            pair.samples,
            CURRENT_FOA_SAMPLE_RATE_HZ,
            channel_axis=0,
            metadata={
                **rlr_foa_wav_metadata(),
                "audio_role": "native_rlr_foa_pair_ir",
                "render_mode": "native_rlr_foa",
                "listener_id": pair.listener_id,
                "source_id": pair.source_id,
                "binaural": "not_requested",
                "amplitude_normalization": "not_applied",
                "resampling": "not_applied",
            },
        )
        pairs.append(
            {
                "listener_id": pair.listener_id,
                "source_id": pair.source_id,
                "wav": wav.audio_path.relative_to(output).as_posix(),
                "sidecar": wav.sidecar_path.relative_to(output).as_posix(),
                "sample_rate_hz": wav.sample_rate_hz,
                "channel_count": wav.channel_count,
            }
        )

    identity = copy.deepcopy(result.runtime["runtime_identity"])
    return {
        "status": "pass",
        "research_status": "research_candidate",
        "research_only": True,
        "qualification_claim": False,
        "runtime_mode": RUNTIME_MODE_CURRENT_INSTALLED,
        "request_id": request["request_id"],
        "binaural": "not_requested",
        "spatial_format": rlr_foa_contract(),
        "render": {
            "mode": "native_rlr_foa",
            "sample_rate_hz": CURRENT_FOA_SAMPLE_RATE_HZ,
            "source_order": list(result.canonical_native_source_order),
            "listener_id": result.listener_id,
            "native_pair_count": len(result.pairs),
            "propagation": {
                "direct": simulation.direct,
                "indirect": simulation.indirect,
                "diffraction": simulation.diffraction,
                "transmission": simulation.transmission,
            },
        },
        "acoustic_package": {
            "manifest": str(scene.manifest_path),
            "package_id": scene.package_id,
            "package_mode": scene.manifest.get("package_mode"),
            "material_semantics": scene.manifest.get("materials", {}).get(
                "material_semantics"
            ),
            "qualification_claim": scene.manifest.get("materials", {}).get(
                "qualification_claim"
            ),
        },
        "runtime_identity": identity,
        "pairs": pairs,
    }


def run_current_foa(
    request_path: str | Path,
    package_manifest_path: str | Path,
    output_directory: str | Path,
    *,
    runtime_prefix: str | Path | None,
    rlr_sdk_root: str | Path | None,
    magnum_python_site: str | Path | None,
) -> dict[str, Any]:
    """Render raw native 16 kHz FOA pair IR WAVs for current research only.

    This has no HRTF argument by design.  The retained ``run_m4_canary`` path
    remains the only route that renders native binaural output and therefore
    retains the strict HRTF/license preflight and v1/v2 evidence handling.
    """

    runtime_inputs = _current_runtime_inputs(
        runtime_prefix=runtime_prefix,
        rlr_sdk_root=rlr_sdk_root,
        magnum_python_site=magnum_python_site,
    )
    try:
        validated = load_and_validate_multi_source_canary_request(request_path)
        simulation = _foa_simulation(validated.request)
        sources, listener = _anchors(validated.request)
        scene = load_compiled_acoustic_scene(
            package_manifest_path,
            allow_nonpassing_research_qa=True,
        )
    except (M4ContractError, RuntimeContractError, OSError, ValueError) as error:
        raise CurrentFOAError(str(error)) from error

    output = _fresh_output_directory(output_directory)
    try:
        result = render_named_sources(
            scene,
            simulation,
            sources=sources,
            listener=listener,
            layout_type="ambisonics",
            channel_count=RLR_FOA_CHANNEL_COUNT,
            hrtf_file_path="",
            runtime_mode=RUNTIME_MODE_CURRENT_INSTALLED,
            runtime_prefix=runtime_inputs["runtime_prefix"],
            rlr_sdk_root=runtime_inputs["rlr_sdk_root"],
            magnum_python_site=runtime_inputs["magnum_python_site"],
        )
        _validate_raw_foa_result(
            result,
            source_ids=tuple(validated.canonical_source_ids),
            listener_id=listener.anchor_id,
        )
        receipt = _raw_foa_receipt(
            output=output,
            request=validated.request,
            simulation=simulation,
            scene=scene,
            result=result,
        )
        write_json(output / "research_receipt.json", receipt)
    except (AudioContractError, SpatialContractError, RuntimeUnavailableError) as error:
        raise CurrentFOAError(str(error)) from error
    return receipt


__all__ = [
    "CURRENT_FOA_SAMPLE_RATE_HZ",
    "CurrentFOAError",
    "run_current_foa",
]
