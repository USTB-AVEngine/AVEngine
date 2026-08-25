"""Research-only current-installed RLR pair IRs driven directly by M1.

The retained M4 canary request carries historical M2 identity evidence and is
therefore not the authority for the current static room/listener/source slice.
This module instead validates one existing M1 request, composes its listener
world pose, takes the two static source positions directly from M1, and binds
them to one existing M3 package and RIR-cache simulation request.  It creates
neither an M2 anchor claim nor a dynamic-actor claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from avengine.backends.rlr.sdk import ExternalRlrSdkError, require_outside_git_checkout
from avengine.contracts.json_io import load_json, write_json
from avengine.contracts.transforms import compose_transforms
from avengine.current_installed_runtime import is_current_installed_runtime_identity
from avengine.m1.contracts import (
    CAPTURE_SCHEMA,
    ContractError,
    validate_capture_request,
)
from avengine.acoustics.runtime import (
    RUNTIME_MODE_CURRENT_INSTALLED,
    CompiledAcousticScene,
    RuntimeAnchor,
    RuntimeContractError,
    RuntimeUnavailableError,
    load_compiled_acoustic_scene,
)
from avengine.spatial_audio.audio import AudioContractError, write_float32_wav
from avengine.spatial_audio.binaural import (
    BINAURAL_CHANNEL_ORDER,
    STRICT_SAMPLE_RATE_POLICY,
    BinauralContractError,
    build_rlr_native_binaural_metadata,
    rlr_native_binaural_contract,
    validate_binaural_samples,
)
from avengine.spatial_audio.runtime import (
    BINAURAL_LAYOUT_ID,
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


RIR_CACHE_SIMULATION_REQUEST_SCHEMA = "avengine_rir_cache_simulation_request_v1"
CURRENT_M1_PAIR_IR_SAMPLE_RATE_HZ = 16_000
CURRENT_M1_SOURCE_IDS = ("source0", "source1")


class CurrentM1PairIRError(ValueError):
    """The current M1 pair-IR request is unsafe or internally inconsistent."""


class CurrentM1PairIRBlockedError(CurrentM1PairIRError):
    """A declared external binaural dependency is currently unavailable."""


@dataclass(frozen=True)
class CurrentM1PairIRInputs:
    """Validated static endpoints and propagation inputs shared by both layouts."""

    m1_request_path: Path
    simulation_request_path: Path
    package_manifest_path: Path
    m1_request: dict[str, Any]
    simulation_request: dict[str, Any]
    simulation: M4SimulationConfig
    scene: CompiledAcousticScene
    sources: tuple[RuntimeAnchor, RuntimeAnchor]
    listener: RuntimeAnchor
    room_id: str

    @property
    def canonical_source_ids(self) -> tuple[str, str]:
        return CURRENT_M1_SOURCE_IDS


def _require_external_directory(value: str | Path | None, *, option: str) -> Path:
    if value is None or not str(value).strip():
        raise CurrentM1PairIRError(f"current M1 pair IR requires explicit {option}")
    try:
        resolved = require_outside_git_checkout(value, owner=option)
    except (ExternalRlrSdkError, OSError, RuntimeError, ValueError) as error:
        raise CurrentM1PairIRError(str(error)) from error
    if not resolved.is_dir():
        raise CurrentM1PairIRError(f"{option} is not a directory: {resolved}")
    return resolved


def _current_runtime_inputs(
    *,
    runtime_prefix: str | Path | None,
    rlr_sdk_root: str | Path | None,
    magnum_python_site: str | Path | None,
) -> dict[str, Path]:
    """Resolve every native dependency before an RLR context can be created."""

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


def _fresh_output_path(value: str | Path) -> Path:
    if not str(value).strip():
        raise CurrentM1PairIRError("current M1 pair IR requires explicit --output")
    requested = Path(value).expanduser()
    output = requested.resolve()
    if os.path.lexists(requested) or os.path.lexists(output):
        raise CurrentM1PairIRError(
            f"refusing to replace current M1 pair-IR output: {requested}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _load_simulation_request(path: Path) -> tuple[dict[str, Any], M4SimulationConfig]:
    document = load_json(path)
    if not isinstance(document, dict):
        raise CurrentM1PairIRError("simulation request must contain one JSON object")
    if document.get("schema") != RIR_CACHE_SIMULATION_REQUEST_SCHEMA:
        raise CurrentM1PairIRError(
            "simulation request schema must be exactly "
            f"{RIR_CACHE_SIMULATION_REQUEST_SCHEMA}"
        )
    if document.get("qualification_claim") is not False:
        raise CurrentM1PairIRError(
            "current M1 pair IR requires simulation qualification_claim=false"
        )
    try:
        simulation = M4SimulationConfig.from_mapping(document.get("simulation"))
    except (RuntimeContractError, TypeError, ValueError) as error:
        raise CurrentM1PairIRError(f"simulation request is invalid: {error}") from error
    if not math.isclose(
        float(simulation.sample_rate_hz),
        float(CURRENT_M1_PAIR_IR_SAMPLE_RATE_HZ),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise CurrentM1PairIRError(
            "current M1 pair IR requires native sample_rate_hz=16000; "
            "it never resamples RLR output"
        )
    if (
        simulation.channel_layout.layout_type != "ambisonics"
        or simulation.channel_layout.channel_count != RLR_FOA_CHANNEL_COUNT
    ):
        raise CurrentM1PairIRError(
            "current M1 pair IR requires the existing simulation request to "
            "declare ambisonics/4; the binaural command derives only its output layout"
        )
    return document, simulation


def _validate_sofa_native_input(
    path: Path, *, expected_sample_rate_hz: int
) -> dict[str, Any]:
    """Verify a SOFA file's AES69 structure before RLR can silently fall back."""

    try:
        import sofar
    except ImportError as error:
        raise CurrentM1PairIRBlockedError(
            "native binaural rendering requires the optional 'sofar' package "
            "to validate the declared SOFA structure"
        ) from error
    try:
        sofa = sofar.read_sofa(path, verify=True, verbose=False)
        samples = np.asarray(sofa.Data_IR)
        sample_rate = float(sofa.Data_SamplingRate)
    except Exception as error:
        raise CurrentM1PairIRError(
            f"declared HRTF is not a valid verified SOFA file: {error}"
        ) from error
    if (
        samples.ndim != 3
        or samples.shape[1] != 2
        or samples.shape[2] < 2
        or not np.issubdtype(samples.dtype, np.number)
        or not np.isfinite(samples).all()
    ):
        raise CurrentM1PairIRError(
            "declared binaural SOFA Data.IR must be finite [measurements,2,samples]"
        )
    if not math.isclose(
        sample_rate,
        float(expected_sample_rate_hz),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise CurrentM1PairIRError(
            "verified SOFA Data.SamplingRate differs from the declared render rate"
        )
    return {
        "status": "pass",
        "validator": "sofar",
        "validator_version": str(getattr(sofar, "__version__", "unknown")),
        "sofa_convention": str(getattr(sofa, "GLOBAL_SOFAConventions", "")),
        "sofa_convention_version": str(
            getattr(sofa, "GLOBAL_SOFAConventionsVersion", "")
        ),
        "data_ir_shape": list(samples.shape),
        "data_sampling_rate_hz": int(sample_rate),
    }


def _m1_endpoints(
    request: Mapping[str, Any],
) -> tuple[tuple[RuntimeAnchor, RuntimeAnchor], RuntimeAnchor]:
    raw_sources = request.get("sources")
    if not isinstance(raw_sources, list) or len(raw_sources) != 2:
        raise CurrentM1PairIRError(
            "current M1 pair IR requires exactly two static M1 sources"
        )
    by_id = {
        item.get("source_id"): item for item in raw_sources if isinstance(item, Mapping)
    }
    if set(by_id) != set(CURRENT_M1_SOURCE_IDS) or len(by_id) != 2:
        raise CurrentM1PairIRError(
            "current M1 pair IR requires static M1 source0 and source1"
        )
    try:
        sources = tuple(
            RuntimeAnchor.from_mapping(
                {
                    "id": source_id,
                    "position_m": by_id[source_id]["world_from_source"][
                        "translation_m"
                    ],
                    "radius_m": 0.0,
                },
                listener=False,
            )
            for source_id in CURRENT_M1_SOURCE_IDS
        )
        rig = request["primary_camera_rig"]
        listener_declaration = request["listener"]
        world_from_listener = compose_transforms(
            rig["world_from_rig"], listener_declaration["rig_from_listener"]
        )
        x, y, z, w = world_from_listener["rotation_xyzw"]
        listener = RuntimeAnchor.from_mapping(
            {
                "id": listener_declaration["listener_id"],
                "position_m": world_from_listener["translation_m"],
                "orientation_wxyz": [w, x, y, z],
                "radius_m": 0.0,
            },
            listener=True,
        )
    except (KeyError, RuntimeContractError, TypeError, ValueError) as error:
        raise CurrentM1PairIRError(
            f"M1 listener/source world composition failed: {error}"
        ) from error
    return (sources[0], sources[1]), listener


def load_current_m1_pair_ir_inputs(
    m1_request_path: str | Path,
    simulation_request_path: str | Path,
    package_manifest_path: str | Path,
) -> CurrentM1PairIRInputs:
    """Validate one M1/static-endpoint M4 slice without creating native state."""

    m1_path = Path(m1_request_path).resolve()
    simulation_path = Path(simulation_request_path).resolve()
    package_path = Path(package_manifest_path).resolve()
    try:
        m1_request = load_json(m1_path)
    except (ContractError, OSError, RuntimeContractError, ValueError) as error:
        raise CurrentM1PairIRError(str(error)) from error
    if not isinstance(m1_request, dict):
        raise CurrentM1PairIRError("M1 request must contain one JSON object")
    if m1_request.get("schema") != CAPTURE_SCHEMA:
        raise CurrentM1PairIRError(
            f"M1 request schema must be exactly {CAPTURE_SCHEMA}"
        )
    errors = validate_capture_request(m1_request)
    if errors:
        raise CurrentM1PairIRError(
            "M1 request validation failed before package load: " + "; ".join(errors)
        )
    try:
        simulation_request, simulation = _load_simulation_request(simulation_path)
        scene = load_compiled_acoustic_scene(
            package_path,
            allow_nonpassing_research_qa=True,
        )
    except (ContractError, OSError, RuntimeContractError, ValueError) as error:
        if isinstance(error, CurrentM1PairIRError):
            raise
        raise CurrentM1PairIRError(str(error)) from error
    source_room = scene.manifest.get("source_room")
    room_id = source_room.get("room_id") if isinstance(source_room, Mapping) else None
    if not isinstance(room_id, str) or not room_id:
        raise CurrentM1PairIRError(
            "M3 package must declare a non-empty source_room.room_id"
        )
    if m1_request.get("room_id") != room_id:
        raise CurrentM1PairIRError(
            "M3 package source_room.room_id must equal M1 request room_id"
        )
    errors = validate_capture_request(m1_request, room_id=room_id)
    if errors:
        raise CurrentM1PairIRError("M1 request validation failed: " + "; ".join(errors))
    sources, listener = _m1_endpoints(m1_request)
    return CurrentM1PairIRInputs(
        m1_request_path=m1_path,
        simulation_request_path=simulation_path,
        package_manifest_path=scene.manifest_path,
        m1_request=m1_request,
        simulation_request=simulation_request,
        simulation=simulation,
        scene=scene,
        sources=sources,
        listener=listener,
        room_id=room_id,
    )


def _validate_result(
    result: MultiSourceRenderResult,
    inputs: CurrentM1PairIRInputs,
    *,
    layout_type: str,
) -> None:
    expected = {
        "ambisonics": {
            "layout_id": FOA_LAYOUT_ID,
            "channel_labels": RLR_FOA_CHANNEL_ORDER,
            "normalization": RLR_FOA_NORMALIZATION,
            "coordinate_frame": RLR_FOA_COORDINATE_FRAME,
        },
        "binaural": {
            "layout_id": BINAURAL_LAYOUT_ID,
            "channel_labels": BINAURAL_CHANNEL_ORDER,
            "normalization": "not_applicable",
            "coordinate_frame": "listener_local",
        },
    }[layout_type]
    channel_count = 4 if layout_type == "ambisonics" else 2
    if (
        result.layout_type != layout_type
        or result.layout_id != expected["layout_id"]
        or not math.isclose(
            float(result.sample_rate_hz),
            float(CURRENT_M1_PAIR_IR_SAMPLE_RATE_HZ),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or tuple(result.canonical_native_source_order) != inputs.canonical_source_ids
        or result.listener_id != inputs.listener.anchor_id
    ):
        raise CurrentM1PairIRError(
            f"native output differs from current M1 {layout_type} pair contract"
        )
    if tuple(pair.source_id for pair in result.pairs) != inputs.canonical_source_ids:
        raise CurrentM1PairIRError("native pair source order differs from static M1")
    for pair in result.pairs:
        if (
            pair.listener_id != inputs.listener.anchor_id
            or pair.layout_type != layout_type
            or pair.layout_id != expected["layout_id"]
            or pair.channel_labels != expected["channel_labels"]
            or pair.normalization != expected["normalization"]
            or pair.coordinate_frame != expected["coordinate_frame"]
            or pair.samples.ndim != 2
            or pair.samples.shape[0] != channel_count
            or not math.isclose(
                float(pair.sample_rate_hz),
                float(CURRENT_M1_PAIR_IR_SAMPLE_RATE_HZ),
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise CurrentM1PairIRError(
                f"native {layout_type} pair metadata differs from the M4 contract"
            )
        try:
            if layout_type == "ambisonics":
                validate_foa_samples(pair.samples, channel_axis=0)
            else:
                validate_binaural_samples(pair.samples, channel_axis=0)
        except (BinauralContractError, SpatialContractError) as error:
            raise CurrentM1PairIRError(
                f"native {layout_type} samples are invalid: {error}"
            ) from error
    if not is_current_installed_runtime_identity(
        result.runtime.get("runtime_identity")
    ):
        raise CurrentM1PairIRError(
            "native result lacks one canonical non-Git current-installed "
            "Habitat/RLR adapter identity"
        )


def _authority_receipt(inputs: CurrentM1PairIRInputs) -> dict[str, Any]:
    return {
        "room_id": inputs.room_id,
        "m1_request": str(inputs.m1_request_path),
        "listener": {
            "listener_id": inputs.listener.anchor_id,
            "position_m": list(inputs.listener.position_m),
            "orientation_wxyz": list(inputs.listener.orientation_wxyz),
            "authority": "M1 world_from_rig composed with rig_from_listener",
        },
        "sources": [
            {
                "source_id": source.anchor_id,
                "position_m": list(source.position_m),
                "motion": "static",
                "authority": "M1 world_from_source.translation_m",
            }
            for source in inputs.sources
        ],
        "static_m1_sources": True,
        "dynamic_actor_anchor_claim": False,
        "m2_anchor_evidence_claim": False,
    }


def _base_receipt(
    *,
    inputs: CurrentM1PairIRInputs,
    result: MultiSourceRenderResult,
    render_mode: str,
) -> dict[str, Any]:
    simulation = inputs.simulation
    return {
        "status": "pass",
        "research_status": "research_candidate",
        "research_only": True,
        "episode_counted": False,
        "formal_dataset_count": 0,
        "qualification": False,
        "qualification_claim": False,
        "runtime_mode": RUNTIME_MODE_CURRENT_INSTALLED,
        "request_id": inputs.m1_request["request_id"],
        "inputs": {
            "m1_request": str(inputs.m1_request_path),
            "simulation_request": str(inputs.simulation_request_path),
            "package_manifest": str(inputs.package_manifest_path),
        },
        "authority": _authority_receipt(inputs),
        "claims": {
            "static_m1_source_positions": True,
            "dynamic_actor_motion": False,
            "m2_anchor_evidence": False,
            "formal_m4_qualification": False,
        },
        "render": {
            "mode": render_mode,
            "sample_rate_hz": CURRENT_M1_PAIR_IR_SAMPLE_RATE_HZ,
            "declared_m1_source_order": [
                item["source_id"] for item in inputs.m1_request["sources"]
            ],
            "canonical_native_source_order": list(result.canonical_native_source_order),
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
            "manifest": str(inputs.scene.manifest_path),
            "package_id": inputs.scene.package_id,
            "source_room_id": inputs.room_id,
            "package_mode": inputs.scene.manifest.get("package_mode"),
            "nonpassing_research_qa_allowed": True,
            "qualification_claim": False,
        },
        "research_package_qa": {
            "admission": "nonpassing_research_override",
            "statuses": {
                name: report.get("status")
                for name, report in sorted(inputs.scene.qa_reports.items())
            },
            "formal_qualification": False,
            "dataset_admission": False,
        },
        "runtime_identity": result.runtime["runtime_identity"],
    }


def _write_pairs(
    *,
    output: Path,
    inputs: CurrentM1PairIRInputs,
    result: MultiSourceRenderResult,
    layout_type: str,
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pair in result.pairs:
        wav = write_float32_wav(
            output / "raw_ir" / layout_type / f"{pair.source_id}.wav",
            pair.samples,
            CURRENT_M1_PAIR_IR_SAMPLE_RATE_HZ,
            channel_axis=0,
            metadata={
                **metadata,
                "listener_id": pair.listener_id,
                "source_id": pair.source_id,
                "source_authority": "static_m1_world_from_source",
                "dynamic_actor_anchor_claim": False,
                "m2_anchor_evidence_claim": False,
                "amplitude_normalization": "not_applied",
                "resampling": "not_applied",
            },
        )
        records.append(
            {
                "listener_id": pair.listener_id,
                "source_id": pair.source_id,
                "wav": wav.audio_path.relative_to(output).as_posix(),
                "sidecar": wav.sidecar_path.relative_to(output).as_posix(),
                "sample_rate_hz": wav.sample_rate_hz,
                "channel_count": wav.channel_count,
            }
        )
    if tuple(record["source_id"] for record in records) != inputs.canonical_source_ids:
        raise CurrentM1PairIRError("written pair order differs from static M1 sources")
    return records


def run_current_m1_foa(
    m1_request_path: str | Path,
    simulation_request_path: str | Path,
    package_manifest_path: str | Path,
    output_directory: str | Path,
    *,
    runtime_prefix: str | Path | None,
    rlr_sdk_root: str | Path | None,
    magnum_python_site: str | Path | None,
) -> dict[str, Any]:
    """Render two static-M1 native 16 kHz FOA pair IR WAVs."""

    runtime_inputs = _current_runtime_inputs(
        runtime_prefix=runtime_prefix,
        rlr_sdk_root=rlr_sdk_root,
        magnum_python_site=magnum_python_site,
    )
    inputs = load_current_m1_pair_ir_inputs(
        m1_request_path,
        simulation_request_path,
        package_manifest_path,
    )
    output = _fresh_output_path(output_directory)
    try:
        result = render_named_sources(
            inputs.scene,
            inputs.simulation,
            sources=inputs.sources,
            listener=inputs.listener,
            layout_type="ambisonics",
            channel_count=RLR_FOA_CHANNEL_COUNT,
            hrtf_file_path="",
            runtime_mode=RUNTIME_MODE_CURRENT_INSTALLED,
            runtime_prefix=runtime_inputs["runtime_prefix"],
            rlr_sdk_root=runtime_inputs["rlr_sdk_root"],
            magnum_python_site=runtime_inputs["magnum_python_site"],
        )
        _validate_result(result, inputs, layout_type="ambisonics")
    except RuntimeUnavailableError:
        raise
    except (RuntimeContractError, ValueError) as error:
        raise CurrentM1PairIRError(str(error)) from error
    try:
        output.mkdir()
        receipt = _base_receipt(
            inputs=inputs,
            result=result,
            render_mode="native_rlr_foa_from_static_m1",
        )
        receipt.update(
            {
                "spatial_format": rlr_foa_contract(),
                "hrtf_used": False,
                "pairs": _write_pairs(
                    output=output,
                    inputs=inputs,
                    result=result,
                    layout_type="foa",
                    metadata={
                        **rlr_foa_wav_metadata(),
                        "audio_role": "native_rlr_foa_pair_ir",
                        "render_mode": "native_rlr_foa_from_static_m1",
                        "hrtf_used": False,
                    },
                ),
            }
        )
        write_json(output / "research_receipt.json", receipt)
    except (AudioContractError, OSError, SpatialContractError, ValueError) as error:
        raise CurrentM1PairIRError(str(error)) from error
    return receipt


def run_current_m1_binaural(
    m1_request_path: str | Path,
    simulation_request_path: str | Path,
    package_manifest_path: str | Path,
    output_directory: str | Path,
    *,
    runtime_prefix: str | Path | None,
    rlr_sdk_root: str | Path | None,
    magnum_python_site: str | Path | None,
    hrtf_path: str | Path,
    expected_hrtf_sha256: str,
    hrtf_sample_rate_hz: int,
    hrtf_license_path: str | Path,
    expected_hrtf_license_sha256: str,
    hrtf_license_id: str,
    hrtf_citation: str,
) -> dict[str, Any]:
    """Render two static-M1 native 16 kHz [left, right] pair IR WAVs."""

    runtime_inputs = _current_runtime_inputs(
        runtime_prefix=runtime_prefix,
        rlr_sdk_root=rlr_sdk_root,
        magnum_python_site=magnum_python_site,
    )
    preflight = build_rlr_native_binaural_metadata(
        CURRENT_M1_PAIR_IR_SAMPLE_RATE_HZ,
        hrtf_path=hrtf_path,
        expected_hrtf_sha256=expected_hrtf_sha256,
        hrtf_sample_rate_hz=hrtf_sample_rate_hz,
        license_id=hrtf_license_id,
        citation=hrtf_citation,
        license_text_path=hrtf_license_path,
        expected_license_sha256=expected_hrtf_license_sha256,
        asset_id="current_m1_explicit_sofa_hrtf",
        sample_rate_policy=STRICT_SAMPLE_RATE_POLICY,
    )
    if preflight.get("status") == "blocked":
        raise CurrentM1PairIRBlockedError(
            str(preflight.get("reason", "HRTF dependency preflight blocked"))
        )
    if preflight.get("status") != "pass":
        raise CurrentM1PairIRError(
            str(preflight.get("reason", "HRTF dependency preflight failed"))
        )
    resolved_hrtf = Path(hrtf_path).absolute()
    sofa_validation = _validate_sofa_native_input(
        resolved_hrtf,
        expected_sample_rate_hz=CURRENT_M1_PAIR_IR_SAMPLE_RATE_HZ,
    )
    inputs = load_current_m1_pair_ir_inputs(
        m1_request_path,
        simulation_request_path,
        package_manifest_path,
    )
    output = _fresh_output_path(output_directory)
    try:
        result = render_named_sources(
            inputs.scene,
            inputs.simulation,
            sources=inputs.sources,
            listener=inputs.listener,
            layout_type="binaural",
            channel_count=2,
            hrtf_file_path=str(resolved_hrtf),
            runtime_mode=RUNTIME_MODE_CURRENT_INSTALLED,
            runtime_prefix=runtime_inputs["runtime_prefix"],
            rlr_sdk_root=runtime_inputs["rlr_sdk_root"],
            magnum_python_site=runtime_inputs["magnum_python_site"],
        )
        _validate_result(result, inputs, layout_type="binaural")
    except RuntimeUnavailableError:
        raise
    except (RuntimeContractError, ValueError) as error:
        raise CurrentM1PairIRError(str(error)) from error
    try:
        output.mkdir()
        receipt = _base_receipt(
            inputs=inputs,
            result=result,
            render_mode="native_rlr_binaural_from_static_m1",
        )
        receipt.update(
            {
                "spatial_format": rlr_native_binaural_contract(),
                "hrtf_preflight": preflight,
                "sofa_native_compatibility": sofa_validation,
                "pairs": _write_pairs(
                    output=output,
                    inputs=inputs,
                    result=result,
                    layout_type="binaural",
                    metadata={
                        **rlr_native_binaural_contract(),
                        "audio_role": "native_rlr_binaural_pair_ir",
                        "render_mode": "native_rlr_binaural_from_static_m1",
                        "hrtf_asset_id": preflight["hrtf"]["asset_id"],
                    },
                ),
            }
        )
        write_json(output / "research_receipt.json", receipt)
    except (AudioContractError, OSError, ValueError) as error:
        raise CurrentM1PairIRError(str(error)) from error
    return receipt


__all__ = [
    "CURRENT_M1_PAIR_IR_SAMPLE_RATE_HZ",
    "CURRENT_M1_SOURCE_IDS",
    "RIR_CACHE_SIMULATION_REQUEST_SCHEMA",
    "CurrentM1PairIRBlockedError",
    "CurrentM1PairIRError",
    "CurrentM1PairIRInputs",
    "load_current_m1_pair_ir_inputs",
    "run_current_m1_binaural",
    "run_current_m1_foa",
]
