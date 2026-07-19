"""M6 controlled-room feasibility canary.

This module materializes one deliberately small proof: two persistent named
source endpoints exist, while an exact-sample AudioProgram activates only one
of them.  It reuses a fully verified M5 retained-evidence closure instead of
pretending that a second native simulation happened.  The derived bundle keeps
the authoritative dry bus, per-source RIR sequence, rendered stems, FOA mix,
360-degree binaural mix, videos, timeline and legacy flag semantics auditable.

The result is research evidence.  Its pass status is scoped to semantic
verification of the retained-evidence materialization; current native Habitat
and RLR execution remain ``not_run``.  It never implies room or dataset
admission.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator
import numpy as np

from avengine.contracts.json_io import (
    canonical_json_sha256,
    file_record,
    load_json,
    sha256_file,
    write_json,
)
from avengine.m4.audio import read_float32_wav, write_float32_wav
from avengine.m4.spatial import rlr_foa_contract
from avengine.m5.canary import verify_m5_canary_evidence
from avengine.m5.video import (
    aac_decode_diagnostics,
    mux_binaural_wav,
    mux_qa_binaural_wav,
    probe_episode_video,
    probe_qa_review_video,
    video_packet_sha256,
)
from avengine.m5_1.mp3d_delivery import listener_yaw_degrees
from avengine.m6.audio_program import compile_audio_program, load_audio_program
from avengine.m6.entities import (
    load_animal_template_registry,
    load_entity_asset_registry,
    resolve_entity_asset,
)
from avengine.m6.flags import evaluate_legacy_flags, load_legacy_flag_registry
from avengine.m6.qualification import (
    build_qualification_report,
    load_qualification_report,
    validate_qualification_report,
)
from avengine.m6.rooms import (
    find_room_record,
    load_room_registry,
    validate_room_registry,
)
from avengine.m6.sources import (
    endpoint_index,
    load_sound_asset_registry,
    load_source_endpoint_registry,
    resolve_source_endpoint_bindings,
    sound_index,
)
from avengine.security.path_policy import (
    WorkspacePathPolicy,
    atomic_publish_directory,
)


REQUEST_SCHEMA = "avengine_m6_controlled_canary_request_v1"
EVIDENCE_SCHEMA = "avengine_m6_canary_evidence_v1"
SOURCE_MANIFEST_SCHEMA = "avengine_m6_controlled_source_manifest_v1"
ENTITY_INSTANCES_SCHEMA = "avengine_m6_entity_instances_v1"
RUNTIME_QA_SCHEMA = "avengine_m6_runtime_qa_report_v1"
PROVENANCE_SCHEMA = "avengine_m6_provenance_manifest_v1"
FINAL_STATUS_SCHEMA = "avengine_m6_final_status_v1"

_CONTROLLED_EXECUTION_BASIS = "verified_retained_evidence_materialization"
_CONTROLLED_STATUS_SCOPE = "semantic_materialization_verifier"
_CURRENT_NATIVE_EPISODE_BLOCKER = "m6_current_native_episode_not_run"

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_LAYOUTS: Mapping[str, tuple[int, tuple[str, ...]]] = {
    "binaural": (2, ("left", "right")),
    "foa": (4, ("W", "Y", "Z", "X")),
}
_REGISTRY_FILES: Mapping[str, str] = {
    "animals": "animal_templates_v1.json",
    "entities": "entity_assets_v1.json",
    "endpoints": "source_endpoints_v1.json",
    "sounds": "sound_assets_v1.json",
    "flags": "legacy_m5_1_flags_v1.json",
}
_CANONICAL_REQUEST_PATH = (
    "examples/m6/canary/controlled_one_active_of_two_request.json"
)
_CANONICAL_PROGRAM_PATH = "examples/m6/registries/one_active_of_n_program_v1.json"
_CANONICAL_REGISTRY_ROOT = "examples/m6/registries"
_CANONICAL_ROOM_REGISTRY_PATH = "examples/m6/rooms/room_registry.json"
_CANONICAL_ROOM_QUALIFICATION_PATH = (
    "examples/m6/rooms/qualification/blender_custom_two_zone.json"
)
_RIR_PROJECTION_FIELDS = (
    "layout_id",
    "layout_type",
    "channel_labels",
    "coordinate_frame",
    "normalization",
    "sample_rate_hz",
    "source_ids",
    "listener_id",
    "trajectory_sha256",
    "context_policy",
    "hrtf",
    "endpoint_receipts",
    "ir_sha256_by_frame_source",
    "upload_report",
)


def _controlled_execution_contract() -> dict[str, Any]:
    """Return the explicit non-native scope of the controlled M6 materialization."""

    return {
        "execution_basis": _CONTROLLED_EXECUTION_BASIS,
        "status_scope": _CONTROLLED_STATUS_SCOPE,
        "native_execution": {
            "habitat_sim": "not_run",
            "rlr_audio_propagation": "not_run",
        },
    }


class M6CanaryError(ValueError):
    """The controlled canary request, source evidence, or output is invalid."""

    def __init__(self, errors: str | Sequence[str]):
        self.errors = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("; ".join(self.errors))


def _spatial_format(layout: str) -> dict[str, Any]:
    if layout == "foa":
        return rlr_foa_contract()
    if layout == "binaural":
        return {
            "format_id": "rlr_binaural_lr_v1",
            "channel_count": 2,
            "channel_order": ["left", "right"],
            "normalization": "not_applicable",
            "coordinate_frame": "listener_local",
            "azimuth_domain_deg": [0, 360],
            "raw_array_layout": "channel_major_[channels,samples]",
            "dtype": "float32_le",
        }
    raise M6CanaryError(f"unsupported spatial layout {layout!r}")


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "git failed"
        raise M6CanaryError(f"git {' '.join(arguments)}: {message}")
    return completed.stdout.strip()


def _git_json_at_commit(
    repository: Path,
    implementation_commit: str,
    repository_path: str,
) -> dict[str, Any]:
    """Load one small canonical JSON input directly from an immutable Git commit.

    This is deliberately limited to the formal M6 canary.  Normal development
    requests and registries remain editable; only release evidence must prove
    which committed configuration produced it.
    """

    relative = PurePosixPath(repository_path)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise M6CanaryError(
            f"canonical implementation input path is not repository-relative: "
            f"{repository_path!r}"
        )
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "show",
            f"{implementation_commit}:{repository_path}",
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise M6CanaryError(
            f"implementation input {repository_path!r} is absent from "
            f"{implementation_commit}: {message or 'git show failed'}"
        )
    try:
        value = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise M6CanaryError(
            f"implementation input {repository_path!r} is not valid UTF-8 JSON"
        ) from error
    if not isinstance(value, dict):
        raise M6CanaryError(
            f"implementation input {repository_path!r} must be a JSON object"
        )
    return value


def _canonical_implementation_inputs(
    repository: Path,
    implementation_commit: str,
) -> dict[str, Any]:
    registries = {
        registry_id: _git_json_at_commit(
            repository,
            implementation_commit,
            f"{_CANONICAL_REGISTRY_ROOT}/{filename}",
        )
        for registry_id, filename in _REGISTRY_FILES.items()
    }
    return {
        "request": _git_json_at_commit(
            repository, implementation_commit, _CANONICAL_REQUEST_PATH
        ),
        "program": _git_json_at_commit(
            repository, implementation_commit, _CANONICAL_PROGRAM_PATH
        ),
        "registries": registries,
        "room_registry": _git_json_at_commit(
            repository, implementation_commit, _CANONICAL_ROOM_REGISTRY_PATH
        ),
        "room_qualification": _git_json_at_commit(
            repository,
            implementation_commit,
            _CANONICAL_ROOM_QUALIFICATION_PATH,
        ),
    }


def _implementation_input_differences(
    canonical: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    program: Mapping[str, Any],
    registries: Mapping[str, Mapping[str, Any]],
    room_registry: Mapping[str, Any],
    room_qualification: Mapping[str, Any],
) -> list[str]:
    """Compare formal inputs with the small canonical set committed in A."""

    errors: list[str] = []
    if request != canonical.get("request"):
        errors.append("controlled request differs from implementation commit")
    if program != canonical.get("program"):
        errors.append("AudioProgram differs from implementation commit")
    canonical_registries = canonical.get("registries")
    if not isinstance(canonical_registries, Mapping):
        errors.append("implementation commit has no canonical registry set")
    else:
        for registry_id in _REGISTRY_FILES:
            if registries.get(registry_id) != canonical_registries.get(registry_id):
                errors.append(
                    f"{registry_id} registry differs from implementation commit"
                )
    canonical_room_registry = canonical.get("room_registry")
    room_registry_errors = validate_room_registry(canonical_room_registry)
    if room_registry_errors:
        errors.extend(
            f"committed room registry: {item}" for item in room_registry_errors
        )
    elif room_registry != canonical_room_registry:
        errors.append("room registry differs from implementation commit")
    canonical_qualification = canonical.get("room_qualification")
    qualification_errors = validate_qualification_report(canonical_qualification)
    if qualification_errors:
        errors.extend(
            f"committed room qualification: {item}"
            for item in qualification_errors
        )
    elif room_qualification != canonical_qualification:
        errors.append("room qualification input differs from implementation commit")
    return errors


def _require_current_clean_implementation(
    repository: Path, implementation_commit: str
) -> None:
    _git(repository, "cat-file", "-e", f"{implementation_commit}^{{commit}}")
    head = _git(repository, "rev-parse", "HEAD")
    if head != implementation_commit:
        raise M6CanaryError(
            f"implementation_commit {implementation_commit} differs from current HEAD {head}"
        )
    dirty = _git(repository, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise M6CanaryError(
            "formal M6 canary requires a clean implementation worktree"
        )


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _schema_path(filename: str) -> Path:
    return _repository_root() / "schemas" / filename


def _schema_errors(value: Any, filename: str) -> list[str]:
    schema = load_json(_schema_path(filename))
    validator = Draft202012Validator(schema)
    errors: list[str] = []
    for error in sorted(validator.iter_errors(value), key=lambda item: list(item.path)):
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        errors.append(f"{location}: {error.message}")
    return errors


def bind_controlled_canary_request_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result.pop("request_content_sha256", None)
    result["request_content_sha256"] = canonical_json_sha256(result)
    return result


def validate_controlled_canary_request(value: Any) -> list[str]:
    errors = _schema_errors(value, "m6_controlled_canary_request_v1.schema.json")
    if not isinstance(value, Mapping):
        return errors
    declared = value.get("request_content_sha256")
    core = {key: item for key, item in value.items() if key != "request_content_sha256"}
    if declared != canonical_json_sha256(core):
        errors.append("request_content_sha256 does not match canonical request content")
    endpoints = value.get("source_endpoint_ids")
    upstream = value.get("endpoint_to_upstream_source_id")
    if isinstance(endpoints, list) and isinstance(upstream, Mapping):
        if set(endpoints) != set(upstream):
            errors.append("endpoint_to_upstream_source_id must cover exactly the endpoint IDs")
        mapped = list(upstream.values())
        if len(mapped) != len(set(mapped)):
            errors.append("each endpoint must map to a distinct upstream source")
    instances = value.get("entity_instances")
    if isinstance(instances, list):
        ids = [
            item.get("entity_instance_id")
            for item in instances
            if isinstance(item, Mapping)
        ]
        if all(isinstance(item, str) for item in ids):
            if ids != sorted(set(ids)):
                errors.append(
                    "entity_instances must use unique canonical instance-ID order"
                )
        actor_ids = [
            item.get("timeline_actor_id")
            for item in instances
            if isinstance(item, Mapping)
        ]
        if all(isinstance(item, str) for item in actor_ids) and len(actor_ids) != len(
            set(actor_ids)
        ):
            errors.append("entity_instances must bind distinct timeline actor IDs")
    return errors


def load_controlled_canary_request(path: str | Path) -> dict[str, Any]:
    value = load_json(path)
    errors = validate_controlled_canary_request(value)
    if errors:
        raise M6CanaryError(errors)
    return value


def _retain_file_copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise M6CanaryError(f"retained upstream artifact is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise M6CanaryError(f"refusing to replace retained artifact: {destination}")
    # Evidence bundles must not share a writable inode with their upstream
    # source.  An exclusive byte copy keeps the retained closure independent;
    # the artifact index and semantic verifier authenticate the copied bytes.
    with source.open("rb") as read_handle, destination.open("xb") as write_handle:
        shutil.copyfileobj(read_handle, write_handle, 1024 * 1024)


def _bind_document(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result.pop(field, None)
    result[field] = canonical_json_sha256(result)
    return result


def _rir_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    return {field: deepcopy(value.get(field)) for field in _RIR_PROJECTION_FIELDS}


def _artifact_index(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise M6CanaryError(
                f"controlled bundle staging contains symlink: "
                f"{path.relative_to(root)}"
            )
        if path.is_file() and path != root / "evidence.json":
            relative = path.relative_to(root).as_posix()
            result[relative] = file_record(path, relative_to=root)
    return result


def _json_no_qa_pairs(value: Any) -> bool:
    forbidden = {"qa_pairs", "question", "answer", "natural_language_question"}
    if isinstance(value, Mapping):
        return not forbidden.intersection(value) and all(
            _json_no_qa_pairs(item) for item in value.values()
        )
    if isinstance(value, list):
        return all(_json_no_qa_pairs(item) for item in value)
    return True


def _portable_bundle_paths(value: Any, root: Path) -> Any:
    """Replace generated absolute bundle paths in tool reports with relatives."""

    if isinstance(value, Mapping):
        return {key: _portable_bundle_paths(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_bundle_paths(item, root) for item in value]
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute():
            try:
                return candidate.resolve().relative_to(root.resolve()).as_posix()
            except (OSError, ValueError):
                return value
    return value


def _load_registries(registry_directory: Path) -> dict[str, dict[str, Any]]:
    return {
        "animals": load_animal_template_registry(
            registry_directory / _REGISTRY_FILES["animals"]
        ),
        "entities": load_entity_asset_registry(
            registry_directory / _REGISTRY_FILES["entities"]
        ),
        "endpoints": load_source_endpoint_registry(
            registry_directory / _REGISTRY_FILES["endpoints"]
        ),
        "sounds": load_sound_asset_registry(
            registry_directory / _REGISTRY_FILES["sounds"]
        ),
        "flags": load_legacy_flag_registry(
            registry_directory / _REGISTRY_FILES["flags"]
        ),
    }


def _validate_request_bindings(
    request: Mapping[str, Any],
    registries: Mapping[str, Mapping[str, Any]],
    program: Mapping[str, Any],
) -> None:
    resolve_source_endpoint_bindings(registries["endpoints"], registries["entities"])
    compiled = compile_audio_program(
        program,
        source_endpoint_registry=registries["endpoints"],
        sound_asset_registry=registries["sounds"],
    )
    if tuple(request["source_endpoint_ids"]) != compiled.candidate_source_endpoint_ids:
        raise M6CanaryError("request endpoint order differs from the AudioProgram")
    program_ref = request["audio_program"]
    if (program_ref["program_id"], program_ref["revision"]) != (
        compiled.program_id,
        compiled.revision,
    ):
        raise M6CanaryError("request AudioProgram reference does not resolve")
    flag_ref = request["legacy_flag_registry"]
    if (flag_ref["registry_id"], flag_ref["revision"]) != (
        registries["flags"]["registry_id"],
        registries["flags"]["revision"],
    ):
        raise M6CanaryError("request legacy flag registry reference does not resolve")
    for instance in request["entity_instances"]:
        resolve_entity_asset(
            registries["entities"],
            instance["entity_asset_id"],
            instance["entity_asset_revision"],
        )
    endpoint_records = endpoint_index(registries["endpoints"])
    instance_by_id = {
        item["entity_instance_id"]: item for item in request["entity_instances"]
    }
    for endpoint_id in request["source_endpoint_ids"]:
        endpoint = endpoint_records[endpoint_id]
        binding = endpoint["binding"]
        instance = instance_by_id.get(binding.get("entity_instance_id"))
        if instance is None:
            raise M6CanaryError(f"endpoint {endpoint_id!r} has no request entity instance")
        if (
            binding.get("entity_asset_id") != instance["entity_asset_id"]
            or binding.get("entity_asset_revision")
            != instance["entity_asset_revision"]
        ):
            raise M6CanaryError(
                f"endpoint {endpoint_id!r} asset binding differs from its request instance"
            )
        if endpoint.get("persistent_when_silent") is not True:
            raise M6CanaryError(
                f"controlled endpoint {endpoint_id!r} must persist while silent"
            )


def _validate_entity_visual_authority(
    *,
    request: Mapping[str, Any],
    registries: Mapping[str, Mapping[str, Any]],
    upstream_request: Mapping[str, Any],
) -> None:
    """Bind registry entities to the exact meshes rendered by the M5 episode."""

    upstream_actors = {
        item["actor_id"]: item for item in upstream_request.get("actors", [])
    }
    templates = {
        (item["template_id"], item["revision"]): item
        for item in registries["animals"]["templates"]
    }
    errors: list[str] = []
    for instance in request["entity_instances"]:
        actor_id = instance["timeline_actor_id"]
        actor = upstream_actors.get(actor_id)
        if actor is None:
            errors.append(f"timeline actor {actor_id!r} is absent from retained M5")
            continue
        asset = resolve_entity_asset(
            registries["entities"],
            instance["entity_asset_id"],
            instance["entity_asset_revision"],
        )
        template_ref = asset.get("animal_template_ref", {})
        template = templates.get(
            (template_ref.get("template_id"), template_ref.get("revision"))
        )
        if asset.get("visual_asset", {}).get("sha256") != actor.get(
            "mesh_sha256"
        ):
            errors.append(
                f"entity {instance['entity_instance_id']!r} visual mesh differs "
                "from retained M5 actor"
            )
        if template is None:
            errors.append(
                f"entity {instance['entity_instance_id']!r} animal template is absent"
            )
        elif (
            template.get("template_id") != actor.get("template_id")
            or template.get("body_plan_id") != actor.get("body_plan_id")
        ):
            errors.append(
                f"entity {instance['entity_instance_id']!r} template/body plan differs "
                "from retained M5 actor"
            )
    if errors:
        raise M6CanaryError(errors)


def _validate_program_against_upstream(
    program: Mapping[str, Any],
    request: Mapping[str, Any],
    upstream_root: Path,
    variant: str,
    endpoint_registry: Mapping[str, Any],
    sound_registry: Mapping[str, Any],
) -> tuple[str, str]:
    compiled = compile_audio_program(program)
    active_endpoint = compiled.active_source_endpoint_ids[0]
    silent_endpoint = compiled.silent_source_endpoint_ids[0]
    mapping = request["endpoint_to_upstream_source_id"]
    active_source = mapping[active_endpoint]
    silent_source = mapping[silent_endpoint]
    upstream_request = load_json(upstream_root / "inputs" / "request.json")
    source_ids = [item["source_id"] for item in upstream_request["sources"]]
    if sorted((active_source, silent_source)) != sorted(source_ids):
        raise M6CanaryError("endpoint mapping does not cover the two retained M5 sources")
    upstream_actor_by_source = {
        item["source_id"]: item["actor_id"] for item in upstream_request["sources"]
    }
    request_instance_by_id = {
        item["entity_instance_id"]: item for item in request["entity_instances"]
    }
    endpoints = endpoint_index(endpoint_registry)
    for endpoint_id, upstream_source in mapping.items():
        binding = endpoints[endpoint_id]["binding"]
        instance = request_instance_by_id[binding["entity_instance_id"]]
        if instance["timeline_actor_id"] != upstream_actor_by_source[upstream_source]:
            raise M6CanaryError(
                f"endpoint {endpoint_id!r} timeline actor differs from retained M5 source"
            )
    upstream_program = upstream_request["audio_program"]
    provenance = program.get("source_program_provenance", {})
    if provenance.get("upstream_program_id") != upstream_program.get("program_id"):
        raise M6CanaryError("AudioProgram upstream program identity differs")
    if provenance.get("upstream_source_id") != active_source:
        raise M6CanaryError("AudioProgram upstream source identity differs")
    if provenance.get("upstream_request_sha256") != sha256_file(
        upstream_root / "inputs" / "request.json"
    ):
        raise M6CanaryError("AudioProgram upstream request byte identity differs")

    upstream_event = next(
        item for item in upstream_request["events"] if item["source_id"] == active_source
    )
    sounds = sound_index(sound_registry)
    windows = {
        item["window_id"]: (item["start_sample"], item["end_sample"])
        for item in upstream_program["simultaneous_windows"]
    }
    observed_window_ids: set[str] = set()
    for event in compiled.events:
        raw = next(
            item for item in program["events"] if item["event_id"] == event.event_id
        )
        window_id = raw["upstream_window_id"]
        observed_window_ids.add(window_id)
        expected = windows.get(window_id)
        if expected != (event.start_sample, event.end_sample_exclusive):
            raise M6CanaryError(
                f"AudioProgram event {event.event_id!r} differs from retained M5 window"
            )
        expected_transform = (
            upstream_program["clip_source_interval"]["start_sample"],
            upstream_program["clip_source_interval"]["end_sample"],
            upstream_program["linear_gain"],
            upstream_program["fade_samples"],
        )
        observed_transform = (
            raw["source_start_sample"],
            raw["source_end_sample_exclusive"],
            raw["linear_gain"],
            raw["fade_samples"],
        )
        if observed_transform != expected_transform:
            raise M6CanaryError(
                f"AudioProgram event {event.event_id!r} transform differs from retained M5"
            )
        sound = sounds[raw["sound_asset_id"]]
        if sound["dry_audio"]["sha256"] != upstream_event["dry_audio_asset_sha256"]:
            raise M6CanaryError(
                f"AudioProgram event {event.event_id!r} dry asset differs from retained M5"
            )
    if observed_window_ids != set(windows) or len(compiled.events) != len(windows):
        raise M6CanaryError("AudioProgram does not cover the retained M5 windows exactly")
    variant_root = upstream_root / "episodes" / variant
    if not variant_root.is_dir():
        raise M6CanaryError(f"retained M5 variant is absent: {variant}")
    return active_source, silent_source


def _write_audio(
    *,
    staging: Path,
    upstream_root: Path,
    variant: str,
    active_endpoint: str,
    silent_endpoint: str,
    active_source: str,
    silent_source: str,
    program: Mapping[str, Any],
    upstream_evidence: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    audio_records: dict[str, Any] = {"layouts": {}, "dry_buses": {}}
    source_root = upstream_root / "episodes" / variant / "audio"

    upstream_dry = read_float32_wav(source_root / "dry" / f"{active_source}.wav")
    if upstream_dry.samples.shape != (1, 80_000):
        raise M6CanaryError("retained active dry bus is not mono [1, 80000]")
    active_windows = [
        (item["start_sample"], item["end_sample_exclusive"])
        for item in program["events"]
    ]
    permitted = np.zeros(80_000, dtype=bool)
    for start, end in active_windows:
        permitted[start:end] = True
    if np.any(upstream_dry.samples[0, ~permitted] != 0.0):
        raise M6CanaryError("retained active dry bus contains samples outside AudioProgram")
    if any(not np.any(upstream_dry.samples[0, start:end]) for start, end in active_windows):
        raise M6CanaryError("one or more active AudioProgram windows are silent")

    dry_dir = staging / "audio" / "dry_buses"
    active_dry_path = dry_dir / f"{active_endpoint}.wav"
    silent_dry_path = dry_dir / f"{silent_endpoint}.wav"
    dry_metadata = {
        "role": "scheduled_dry_source_bus",
        "program_id": program["program_id"],
        "program_content_sha256": program["program_content_sha256"],
        "upstream_variant": variant,
    }
    upstream_artifacts = upstream_evidence.get("artifacts")
    if not isinstance(upstream_artifacts, Mapping):
        raise M6CanaryError("retained M5 evidence has no artifact index")
    active_dry_role = f"episodes/{variant}/audio/dry/{active_source}.wav"
    active_dry_upstream_record = upstream_artifacts.get(active_dry_role)
    if not isinstance(active_dry_upstream_record, Mapping):
        raise M6CanaryError("retained M5 active dry bus is absent from artifact index")
    write_float32_wav(
        active_dry_path,
        upstream_dry.samples,
        16_000,
        metadata={
            **dry_metadata,
            "source_endpoint_id": active_endpoint,
            "upstream_source_id": active_source,
            "upstream_artifact": {
                "path": f"upstream://m5/{active_dry_role}",
                "byte_size": active_dry_upstream_record["byte_size"],
                "sha256": active_dry_upstream_record["sha256"],
            },
            "active": True,
        },
    )
    write_float32_wav(
        silent_dry_path,
        np.zeros_like(upstream_dry.samples),
        16_000,
        metadata={
            **dry_metadata,
            "source_endpoint_id": silent_endpoint,
            "upstream_source_id": silent_source,
            "active": False,
        },
    )
    audio_records["dry_buses"] = {
        active_endpoint: file_record(active_dry_path, relative_to=staging),
        silent_endpoint: file_record(silent_dry_path, relative_to=staging),
    }

    for layout, (channel_count, channel_labels) in _LAYOUTS.items():
        source_wet = read_float32_wav(
            source_root / layout / f"{active_source}_stem.wav"
        )
        if source_wet.samples.shape != (channel_count, 80_000):
            raise M6CanaryError(
                f"retained {layout} active stem has unexpected array shape"
            )
        silent = np.zeros_like(source_wet.samples)
        wet_dir = staging / "audio" / "source_stems" / layout
        active_path = wet_dir / f"{active_endpoint}.wav"
        silent_path = wet_dir / f"{silent_endpoint}.wav"
        mixture_path = (
            staging / "audio" / "mixture.wav"
            if layout == "binaural"
            else staging / "audio" / "mixture_foa.wav"
        )
        common = {
            "role": "dynamic_wet_stem",
            "layout_id": (
                "rlr_binaural_lr_v1"
                if layout == "binaural"
                else "rlr_foa_acn_n3d_world_v1"
            ),
            "channel_labels": list(channel_labels),
            "spatial_format": _spatial_format(layout),
            "program_id": program["program_id"],
            "program_content_sha256": program["program_content_sha256"],
            "derivation": "authenticated_m5_stem_selection_v1",
            "upstream_variant": variant,
        }
        active_stem_role = (
            f"episodes/{variant}/audio/{layout}/{active_source}_stem.wav"
        )
        active_stem_upstream_record = upstream_artifacts.get(active_stem_role)
        if not isinstance(active_stem_upstream_record, Mapping):
            raise M6CanaryError(
                f"retained M5 {layout} active stem is absent from artifact index"
            )
        write_float32_wav(
            active_path,
            source_wet.samples,
            16_000,
            metadata={
                **common,
                "source_endpoint_id": active_endpoint,
                "upstream_source_id": active_source,
                "upstream_artifact": {
                    "path": f"upstream://m5/{active_stem_role}",
                    "byte_size": active_stem_upstream_record["byte_size"],
                    "sha256": active_stem_upstream_record["sha256"],
                },
                "active": True,
            },
        )
        write_float32_wav(
            silent_path,
            silent,
            16_000,
            metadata={
                **common,
                "source_endpoint_id": silent_endpoint,
                "upstream_source_id": silent_source,
                "active": False,
            },
        )
        write_float32_wav(
            mixture_path,
            source_wet.samples + silent,
            16_000,
            metadata={
                **common,
                "role": "source_stem_sum_mixture",
                "canonical_source_endpoint_order": [active_endpoint, silent_endpoint],
                "active_source_endpoint_ids": [active_endpoint],
                "silent_source_endpoint_ids": [silent_endpoint],
            },
        )
        audio_records["layouts"][layout] = {
            "channel_count": channel_count,
            "channel_labels": list(channel_labels),
            "active_stem": file_record(active_path, relative_to=staging),
            "silent_stem": file_record(silent_path, relative_to=staging),
            "mixture": file_record(mixture_path, relative_to=staging),
        }

    for layout in _LAYOUTS:
        upstream_rir = upstream_root / "rir" / layout
        retained_rir = staging / "audio" / "rir_or_rir_references" / layout
        retained_rir.mkdir(parents=True, exist_ok=True)
        for filename in ("samples.npy", "lengths.npy"):
            _retain_file_copy(upstream_rir / filename, retained_rir / filename)
        upstream_metadata_path = upstream_rir / "metadata.json"
        upstream_metadata = load_json(upstream_metadata_path)
        projection = _rir_projection(upstream_metadata)
        projection_sha256 = canonical_json_sha256(projection)
        expected_projection_sha256 = request["upstream_evidence"][
            "rir_projection_sha256_by_layout"
        ][layout]
        if projection_sha256 != expected_projection_sha256:
            raise M6CanaryError(
                f"retained M5 {layout} RIR metadata projection differs from request"
            )
        # Runtime import paths in old evidence are machine-local provenance.
        # Keep the independently verified acoustic facts and bind the original
        # metadata bytes, but never republish private absolute paths.
        portable_metadata = _bind_document(
            {
                "schema": "avengine_m6_retained_rir_reference_v1",
                **projection,
                "upstream_projection_sha256": projection_sha256,
                "retained_arrays": {
                    filename: {
                        **file_record(retained_rir / filename, relative_to=staging),
                        "upstream_artifact": deepcopy(
                            upstream_artifacts[f"rir/{layout}/{filename}"]
                        ),
                    }
                    for filename in ("samples.npy", "lengths.npy")
                },
                "upstream_metadata": {
                    "path": f"upstream://m5/rir/{layout}/metadata.json",
                    "sha256": sha256_file(upstream_metadata_path),
                    "verified_by": "upstream_m5_evidence_verification",
                },
                "runtime_paths_republished": False,
            },
            "reference_content_sha256",
        )
        write_json(retained_rir / "metadata.json", portable_metadata)
    return audio_records


def _build_timeline(
    *,
    upstream_timeline: Mapping[str, Any],
    upstream_request: Mapping[str, Any],
    program: Mapping[str, Any],
    active_source: str,
) -> dict[str, Any]:
    result = deepcopy(dict(upstream_timeline))
    source = next(item for item in upstream_request["sources"] if item["source_id"] == active_source)
    actor_id = source["actor_id"]
    route = next(
        item
        for item in upstream_request["events"]
        if item["source_id"] == active_source and item["actor_id"] == actor_id
    )
    result["audio_events"] = [
        {
            "event_id": event["event_id"],
            "actor_id": actor_id,
            "event_type": "vocalization",
            "start_sample": event["start_sample"],
            "end_sample": event["end_sample_exclusive"],
            "emitter_bone": source["emitter_link"],
            "emitter_path_sha256": source["emitter_path_sha256"],
            "audio_asset_sha256": route["dry_audio_asset_sha256"],
            "semantic_sync_required": True,
        }
        for event in program["events"]
    ]
    events = result["audio_events"]
    for frame in result["frames"]:
        sample = frame["sample_start"]
        active_now = any(
            item["start_sample"] <= sample < item["end_sample"] for item in events
        )
        for state in frame["actor_states"]:
            state["mouth_state"]["vocalizing"] = (
                active_now if state["actor_id"] == actor_id else False
            )
            # Geometry was captured with mouth articulation disabled.  The
            # semantic vocalizing bit changes; the visual pose remains closed.
            state["mouth_state"]["open_ratio"] = 0.0
    errors = _schema_errors(result, "avengine_timeline_v2.schema.json")
    if errors:
        raise M6CanaryError([f"derived timeline: {item}" for item in errors])
    return result


def _build_entities_and_sources(
    *,
    request: Mapping[str, Any],
    registries: Mapping[str, Mapping[str, Any]],
    program: Mapping[str, Any],
    trajectory: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    entity_records = []
    for item in request["entity_instances"]:
        asset = resolve_entity_asset(
            registries["entities"],
            item["entity_asset_id"],
            item["entity_asset_revision"],
        )
        template_ref = asset.get("animal_template_ref")
        template = None
        if isinstance(template_ref, Mapping):
            template = next(
                (
                    candidate
                    for candidate in registries["animals"]["templates"]
                    if candidate["template_id"] == template_ref["template_id"]
                    and candidate["revision"] == template_ref["revision"]
                ),
                None,
            )
            if template is None:
                raise M6CanaryError(
                    "entity asset references an unregistered animal template"
                )
        record = {**deepcopy(dict(item)), "asset": deepcopy(dict(asset))}
        if template is not None:
            record["animal_template"] = deepcopy(dict(template))
        entity_records.append(record)
    entities = _bind_document(
        {"schema": ENTITY_INSTANCES_SCHEMA, "instances": entity_records},
        "manifest_content_sha256",
    )

    endpoint_records = endpoint_index(registries["endpoints"])
    sound_records = sound_index(registries["sounds"])
    instances = {
        item["entity_instance_id"]: item for item in request["entity_instances"]
    }
    compiled = compile_audio_program(program)
    routes = []
    for endpoint_id in request["source_endpoint_ids"]:
        upstream_source = request["endpoint_to_upstream_source_id"][endpoint_id]
        instance_id = endpoint_records[endpoint_id]["binding"]["entity_instance_id"]
        routes.append(
            {
                "source_endpoint_id": endpoint_id,
                "source_endpoint_revision": endpoint_records[endpoint_id]["revision"],
                "endpoint": deepcopy(dict(endpoint_records[endpoint_id])),
                "timeline_actor_id": instances[instance_id]["timeline_actor_id"],
                "upstream_source_id": upstream_source,
                "trajectory_content_sha256": trajectory["trajectory_content_sha256"],
                "activation_state": (
                    "active_program_endpoint"
                    if endpoint_id in compiled.active_source_endpoint_ids
                    else "persistent_silent_endpoint"
                ),
            }
        )
    sources = _bind_document(
        {
            "schema": SOURCE_MANIFEST_SCHEMA,
            "source_endpoint_registry": {
                "registry_id": registries["endpoints"]["registry_id"],
                "revision": registries["endpoints"]["revision"],
                "registry_content_sha256": registries["endpoints"]["registry_content_sha256"],
            },
            "sound_asset_registry": {
                "registry_id": registries["sounds"]["registry_id"],
                "revision": registries["sounds"]["revision"],
                "registry_content_sha256": registries["sounds"][
                    "registry_content_sha256"
                ],
            },
            "sound_assets": [
                deepcopy(dict(sound_records[sound_asset_id]))
                for sound_asset_id in sorted(
                    {item["sound_asset_id"] for item in program["events"]}
                )
            ],
            "audio_program": {
                "program_id": program["program_id"],
                "revision": program["revision"],
                "program_content_sha256": program["program_content_sha256"],
            },
            "routes": routes,
            "active_source_endpoint_ids": list(compiled.active_source_endpoint_ids),
            "silent_source_endpoint_ids": list(compiled.silent_source_endpoint_ids),
        },
        "manifest_content_sha256",
    )
    return entities, sources


def _build_flag_report(
    *, request: Mapping[str, Any], trajectory: Mapping[str, Any]
) -> dict[str, Any]:
    mapping = request["endpoint_to_upstream_source_id"]
    keyframes = trajectory["keyframes"]
    first = keyframes[0]
    positions = {
        endpoint_id: [
            frame["source_positions_m"][upstream_source] for frame in keyframes
        ]
        for endpoint_id, upstream_source in sorted(mapping.items())
    }
    orientation_wxyz = first["listener_orientation_wxyz"]
    orientation_xyzw = [
        orientation_wxyz[1],
        orientation_wxyz[2],
        orientation_wxyz[3],
        orientation_wxyz[0],
    ]
    yaw = listener_yaw_degrees(orientation_xyzw)
    requested_yaw = float(request["listener"]["observer_yaw_deg"])
    if abs(((yaw - requested_yaw + 180.0) % 360.0) - 180.0) > 1.0e-9:
        raise M6CanaryError(
            f"request observer yaw {requested_yaw} differs from retained trajectory {yaw}"
        )
    return evaluate_legacy_flags(
        observer_position_m=first["listener_position_m"],
        observer_yaw_deg=yaw,
        fps=15.0,
        positions_by_source=positions,
        visibility_facts_by_source=None,
        evidence_uri="provenance/upstream_emitter_path.json",
    )


def _retained_materialization_room_report(
    *,
    historical_report: Mapping[str, Any],
    upstream_evidence: Path,
    materialization_status: str,
) -> dict[str, Any]:
    """Build a room report without promoting retained bytes to native execution."""

    if materialization_status not in {"pass", "fail"}:
        raise M6CanaryError(
            "controlled materialization status must be either pass or fail"
        )
    dimensions = deepcopy(historical_report["dimensions"])
    dimensions["episode_feasibility_status"] = {
        "status": "not_run",
        "summary": (
            "No current native Habitat episode or native RLR RIR generation ran. "
            "Separately, the controlled retained-evidence one-active-of-two "
            f"materialization semantic verifier status is {materialization_status}."
        ),
        "evidence_refs": [
            "qa/runtime_qa_report.json",
            "provenance/provenance_manifest.json",
            "evidence.json",
        ],
        "blocker_code": _CURRENT_NATIVE_EPISODE_BLOCKER,
    }
    return build_qualification_report(
        report_id="blender_custom_two_zone_m6_controlled_attempt_v1",
        subject=historical_report["subject"],
        evidence_basis=_CONTROLLED_EXECUTION_BASIS,
        evidence_artifacts=[
            {
                "artifact_id": "verified_upstream_m5_evidence",
                "path": "upstream://m5/evidence.json",
                "sha256": sha256_file(upstream_evidence),
            }
        ],
        dimensions=dimensions,
        placement_feasibility=historical_report["placement_feasibility"],
        acoustic_diagnostics=historical_report["acoustic_diagnostics"],
        provenance={
            "source_records": [
                "examples/m6/rooms/qualification/blender_custom_two_zone.json",
                "upstream://m5/evidence.json",
            ],
            "notes": (
                "Controlled retained-evidence materialization semantic verifier "
                f"status is {materialization_status}. Current native Habitat/RLR "
                "episode execution is not_run, and placement remains not_run, so "
                "this report does not create a qualified room revision."
            ),
        },
        promote_if_eligible=False,
    )


def _retained_materialization_claim_errors(
    *,
    evidence: Mapping[str, Any],
    provenance: Mapping[str, Any],
    room_report: Mapping[str, Any],
) -> list[str]:
    """Reject any native-execution promotion of retained materialization bytes."""

    errors: list[str] = []
    expected_execution = _controlled_execution_contract()
    for field, expected in expected_execution.items():
        if evidence.get(field) != expected:
            errors.append(
                f"controlled evidence {field} differs from retained-materialization "
                "scope"
            )

    derivation = provenance.get("derivation", {})
    if derivation.get("native_rir_rerun") is not False:
        errors.append("controlled provenance must record native_rir_rerun=false")
    if derivation.get("current_native_episode_status") != "not_run":
        errors.append(
            "controlled provenance must record current native episode as not_run"
        )
    if (
        derivation.get("semantic_materialization_verifier_status")
        != evidence.get("overall_status")
    ):
        errors.append(
            "controlled materialization verifier status differs from evidence status"
        )

    episode = room_report.get("dimensions", {}).get(
        "episode_feasibility_status", {}
    )
    if room_report.get("evidence_basis") != _CONTROLLED_EXECUTION_BASIS:
        errors.append(
            "controlled room report must use retained-materialization evidence basis"
        )
    if episode.get("status") != "not_run":
        errors.append("controlled current native episode must remain not_run")
    if episode.get("blocker_code") != _CURRENT_NATIVE_EPISODE_BLOCKER:
        errors.append("controlled current native episode blocker differs")
    return errors


def _runtime_checks(
    *,
    upstream_checks: Sequence[Mapping[str, Any]],
    audio_records: Mapping[str, Any],
    mux_reports: Mapping[str, Any],
    aac_diagnostics: Mapping[str, Any],
    active_endpoint_count: int,
    silent_endpoint_count: int,
    event_count: int,
) -> list[dict[str, Any]]:
    checks = [
        {
            "check_id": "upstream_m5_evidence_verification",
            "status": (
                "pass"
                if upstream_checks and all(item.get("status") == "pass" for item in upstream_checks)
                else "fail"
            ),
            "measured": {
                "check_count": len(upstream_checks),
                "failed_check_ids": [
                    item.get("check_id")
                    for item in upstream_checks
                    if item.get("status") != "pass"
                ],
            },
        },
        {
            "check_id": "one_active_of_two_source_program",
            "status": (
                "pass"
                if (active_endpoint_count, silent_endpoint_count) == (1, 1)
                and event_count > 0
                else "fail"
            ),
            "measured": {
                "active_endpoint_count": active_endpoint_count,
                "silent_endpoint_count": silent_endpoint_count,
                "event_count": event_count,
            },
        },
        {
            "check_id": "foa_and_360_binaural_retained",
            "status": (
                "pass"
                if set(audio_records.get("layouts", {})) == set(_LAYOUTS)
                and all(
                    audio_records["layouts"].get(name, {}).get("channel_count")
                    == channel_count
                    and tuple(
                        audio_records["layouts"].get(name, {}).get(
                            "channel_labels", ()
                        )
                    )
                    == channel_labels
                    for name, (channel_count, channel_labels) in _LAYOUTS.items()
                )
                else "fail"
            ),
            "measured": {
                name: {
                    "channel_count": value["channel_count"],
                    "channel_labels": value["channel_labels"],
                }
                for name, value in audio_records["layouts"].items()
            },
        },
        {
            "check_id": "video_mux_packet_copy",
            "status": (
                "pass"
                if all(item.get("video_stream_copy_verified") is True for item in mux_reports.values())
                else "fail"
            ),
            "measured": dict(mux_reports),
        },
        {
            "check_id": "aac_media_readback",
            "status": (
                "pass"
                if aac_diagnostics.get("presentation_sample_count_matches") is True
                and aac_diagnostics.get("lr_swap_suspected") is False
                and float(aac_diagnostics.get("minimum_correlation", -1.0)) >= 0.98
                and float(aac_diagnostics.get("minimum_snr_db", -999.0)) >= 18.0
                else "fail"
            ),
            "measured": dict(aac_diagnostics),
        },
    ]
    return checks


def _materialization_runtime_report(
    checks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind semantic materialization QA without implying a native rerun."""

    materialization_status = (
        "pass" if all(item.get("status") == "pass" for item in checks) else "fail"
    )
    return _bind_document(
        {
            "schema": RUNTIME_QA_SCHEMA,
            "qa_meaning": "quality_assurance_not_natural_language_question_answering",
            **_controlled_execution_contract(),
            "checks": deepcopy(list(checks)),
            "overall_status": materialization_status,
        },
        "report_content_sha256",
    )


def _recompute_runtime_report(
    *,
    root: Path,
    retained_upstream: Mapping[str, Any],
    program: Mapping[str, Any],
    ffmpeg: str | Path,
    ffprobe: str | Path,
) -> dict[str, Any]:
    """Rebuild the retained QA report from independently read bundle facts."""

    compiled = compile_audio_program(program)
    layouts: dict[str, Any] = {}
    for layout in _LAYOUTS:
        mixture_path = (
            root / "audio" / "mixture.wav"
            if layout == "binaural"
            else root / "audio" / "mixture_foa.wav"
        )
        wave = read_float32_wav(mixture_path)
        metadata = (
            wave.sidecar.get("metadata", {})
            if isinstance(wave.sidecar, Mapping)
            else {}
        )
        labels = metadata.get("channel_labels")
        layouts[layout] = {
            "channel_count": int(wave.samples.shape[0]),
            "channel_labels": list(labels) if isinstance(labels, list) else [],
        }

    mixture_path = root / "audio" / "mixture.wav"
    mux_reports: dict[str, Any] = {}
    for delivery_id, delivery_path, upstream_filename, probe in (
        (
            "primary",
            root / "visual" / "primary_view.mp4",
            "view0_base_video_only.mp4",
            probe_episode_video,
        ),
        (
            "topdown",
            root / "visual" / "optional_debug_views" / "topdown_review.mp4",
            "view0_topdown_base_video_only.mp4",
            probe_qa_review_video,
        ),
    ):
        report = probe(delivery_path, ffprobe=ffprobe)
        observed_packets = video_packet_sha256(delivery_path, ffprobe=ffprobe)
        upstream_packets = video_packet_sha256(
            root / "provenance" / f"upstream_{upstream_filename}",
            ffprobe=ffprobe,
        )
        report["video_packet_hash"] = observed_packets
        report["video_stream_copy_verified"] = (
            observed_packets.get("payload_sha256")
            == upstream_packets.get("payload_sha256")
            and observed_packets.get("timeline_sha256")
            == upstream_packets.get("timeline_sha256")
        )
        report["authoritative_wav"] = {
            "path": str(mixture_path),
            "sample_rate_hz": 16_000,
            "channel_count": 2,
            "sample_count": 80_000,
        }
        mux_reports[delivery_id] = report
    mux_reports = _portable_bundle_paths(mux_reports, root)
    aac_report = aac_decode_diagnostics(
        root / "visual" / "primary_view.mp4",
        mixture_path,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
    )
    checks = _runtime_checks(
        upstream_checks=retained_upstream.get("checks", []),
        audio_records={"layouts": layouts},
        mux_reports=mux_reports,
        aac_diagnostics=aac_report,
        active_endpoint_count=len(compiled.active_source_endpoint_ids),
        silent_endpoint_count=len(compiled.silent_source_endpoint_ids),
        event_count=len(compiled.events),
    )
    return _materialization_runtime_report(checks)


def run_controlled_canary(
    *,
    request_path: str | Path,
    upstream_evidence_path: str | Path,
    output_directory: str | Path,
    implementation_commit: str,
    registry_directory: str | Path | None = None,
    room_registry_path: str | Path | None = None,
    room_qualification_path: str | Path | None = None,
    program_path: str | Path | None = None,
    ffmpeg: str | Path = "ffmpeg",
    ffprobe: str | Path = "ffprobe",
) -> Path:
    """Materialize and atomically publish one controlled M6 evidence closure."""

    if _COMMIT.fullmatch(implementation_commit) is None:
        raise M6CanaryError("implementation_commit must be a full lowercase Git commit")
    repository = _repository_root()
    _require_current_clean_implementation(repository, implementation_commit)
    policy = WorkspacePathPolicy.from_roots([repository])
    request_file = policy.resolve_input(request_path, owner="M6 canary request")
    upstream_evidence = policy.resolve_input(
        upstream_evidence_path, owner="verified upstream M5 evidence"
    )
    destination = policy.resolve_output(
        output_directory, owner="M6 controlled canary bundle", create_parent=True
    )
    registry_dir = policy.resolve_input(
        registry_directory or repository / "examples" / "m6" / "registries",
        owner="M6 registry directory",
        kind="directory",
    )
    room_registry_file = policy.resolve_input(
        room_registry_path or repository / "examples" / "m6" / "rooms" / "room_registry.json",
        owner="M6 room registry",
    )
    room_qualification_file = policy.resolve_input(
        room_qualification_path
        or repository / "examples" / "m6" / "rooms" / "qualification" / "blender_custom_two_zone.json",
        owner="M6 controlled room qualification",
    )
    selected_program_file = policy.resolve_input(
        program_path or registry_dir / "one_active_of_n_program_v1.json",
        owner="M6 AudioProgram",
    )

    request = load_controlled_canary_request(request_file)
    upstream_value = load_json(upstream_evidence)
    if upstream_value.get("evidence_content_sha256") != request["upstream_evidence"]["evidence_content_sha256"]:
        raise M6CanaryError("request does not bind the retained M5 evidence content hash")
    upstream_status, upstream_checks = verify_m5_canary_evidence(upstream_evidence)
    if upstream_status != "pass":
        raise M6CanaryError("retained M5 evidence did not pass independent verification")
    upstream_root = upstream_evidence.parent

    registries = _load_registries(registry_dir)
    program = load_audio_program(
        selected_program_file,
        source_endpoint_registry=registries["endpoints"],
        sound_asset_registry=registries["sounds"],
    )
    _validate_request_bindings(request, registries, program)
    upstream_request_value = load_json(upstream_root / "inputs" / "request.json")
    _validate_entity_visual_authority(
        request=request,
        registries=registries,
        upstream_request=upstream_request_value,
    )
    variant = request["upstream_evidence"]["episode_variant"]
    active_source, silent_source = _validate_program_against_upstream(
        program,
        request,
        upstream_root,
        variant,
        registries["endpoints"],
        registries["sounds"],
    )
    compiled = compile_audio_program(program)
    active_endpoint = compiled.active_source_endpoint_ids[0]
    silent_endpoint = compiled.silent_source_endpoint_ids[0]

    room_registry = load_room_registry(room_registry_file)
    room_record = find_room_record(
        room_registry, request["room"]["room_id"], request["room"]["revision"]
    )
    historical_room_report = load_qualification_report(room_qualification_file)
    if historical_room_report["report_id"] != request["room"]["qualification_report_id"]:
        raise M6CanaryError("request room qualification reference does not resolve")
    canonical_inputs = _canonical_implementation_inputs(
        repository, implementation_commit
    )
    authority_errors = _implementation_input_differences(
        canonical_inputs,
        request=request,
        program=program,
        registries=registries,
        room_registry=room_registry,
        room_qualification=historical_room_report,
    )
    if authority_errors:
        raise M6CanaryError(authority_errors)

    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        write_json(staging / "request.json", request)
        release_ref = {
            "schema": "avengine_release_manifest_ref_v1",
            **deepcopy(request["release_manifest_ref"]),
            "implementation_commit": implementation_commit,
        }
        write_json(staging / "release_manifest_ref.json", release_ref)
        write_json(staging / "source_program.json", program)
        retained_registry_paths: dict[str, Path] = {}
        for registry_id, filename in _REGISTRY_FILES.items():
            retained = staging / "provenance" / "registries" / filename
            _retain_file_copy(registry_dir / filename, retained)
            retained_registry_paths[registry_id] = retained

        retained_upstream_evidence = (
            staging / "provenance" / "upstream_m5_evidence.json"
        )
        _retain_file_copy(upstream_evidence, retained_upstream_evidence)
        retained_upstream_request = staging / "provenance" / "upstream_m5_request.json"
        _retain_file_copy(
            upstream_evidence.parent / "inputs" / "request.json",
            retained_upstream_request,
        )
        retained_upstream_timeline = (
            staging / "provenance" / "upstream_m5_timeline.json"
        )
        _retain_file_copy(
            upstream_evidence.parent
            / "episodes"
            / request["upstream_evidence"]["episode_variant"]
            / "timeline.json",
            retained_upstream_timeline,
        )
        retained_upstream_videos: dict[str, Path] = {}
        for delivery_id, filename in (
            ("primary", "view0_base_video_only.mp4"),
            ("topdown", "view0_topdown_base_video_only.mp4"),
        ):
            retained = staging / "provenance" / f"upstream_{filename}"
            _retain_file_copy(upstream_evidence.parent / "videos" / filename, retained)
            retained_upstream_videos[delivery_id] = retained

        trajectory_path = upstream_root / "trajectory" / "emitter_path.json"
        trajectory = load_json(trajectory_path)
        _retain_file_copy(
            trajectory_path, staging / "provenance" / "upstream_emitter_path.json"
        )
        entities, sources = _build_entities_and_sources(
            request=request,
            registries=registries,
            program=program,
            trajectory=trajectory,
        )
        write_json(staging / "entity_instances.json", entities)
        write_json(staging / "source_manifest.json", sources)
        write_json(staging / "room_manifest.json", room_record)

        upstream_request = upstream_request_value
        # Use the requested A/B variant timeline. Both variants share visual
        # truth, but this keeps provenance exact.
        upstream_timeline = load_json(
            upstream_root / "episodes" / variant / "timeline.json"
        )
        timeline = _build_timeline(
            upstream_timeline=upstream_timeline,
            upstream_request=upstream_request,
            program=program,
            active_source=active_source,
        )
        write_json(staging / "timeline.json", timeline)

        audio_records = _write_audio(
            staging=staging,
            upstream_root=upstream_root,
            variant=variant,
            active_endpoint=active_endpoint,
            silent_endpoint=silent_endpoint,
            active_source=active_source,
            silent_source=silent_source,
            program=program,
            upstream_evidence=upstream_value,
            request=request,
        )
        primary = staging / "visual" / "primary_view.mp4"
        topdown = staging / "visual" / "optional_debug_views" / "topdown_review.mp4"
        primary.parent.mkdir(parents=True, exist_ok=True)
        topdown.parent.mkdir(parents=True, exist_ok=True)
        mux_reports = {
            "primary": mux_binaural_wav(
                upstream_root / "videos" / "view0_base_video_only.mp4",
                staging / "audio" / "mixture.wav",
                primary,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
            ),
            "topdown": mux_qa_binaural_wav(
                upstream_root / "videos" / "view0_topdown_base_video_only.mp4",
                staging / "audio" / "mixture.wav",
                topdown,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
            ),
        }
        mux_reports = _portable_bundle_paths(mux_reports, staging)
        probe_episode_video(primary, ffprobe=ffprobe)
        probe_qa_review_video(topdown, ffprobe=ffprobe)
        aac_report = aac_decode_diagnostics(
            primary,
            staging / "audio" / "mixture.wav",
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
        )

        flag_report = _build_flag_report(request=request, trajectory=trajectory)
        write_json(staging / "flags" / "source_event_flag_report.json", flag_report)

        checks = _runtime_checks(
            # The independent verifier above remains a hard generation gate.
            # The retained QA measurement itself uses the authenticated checks
            # stored in the bound M5 evidence so a future verifier can rebuild
            # it without copying the entire upstream evidence directory.
            upstream_checks=upstream_value["checks"],
            audio_records=audio_records,
            mux_reports=mux_reports,
            aac_diagnostics=aac_report,
            active_endpoint_count=len(compiled.active_source_endpoint_ids),
            silent_endpoint_count=len(compiled.silent_source_endpoint_ids),
            event_count=len(compiled.events),
        )
        runtime_report = _materialization_runtime_report(checks)
        write_json(staging / "qa" / "runtime_qa_report.json", runtime_report)

        room_report = _retained_materialization_room_report(
            historical_report=historical_room_report,
            upstream_evidence=upstream_evidence,
            materialization_status=runtime_report["overall_status"],
        )
        write_json(staging / "room_qualification_report.json", room_report)

        provenance = _bind_document(
            {
                "schema": PROVENANCE_SCHEMA,
                "implementation_commit": implementation_commit,
                "upstream_m5_evidence": {
                    "path": "provenance/upstream_m5_evidence.json",
                    "source_repository_path": upstream_evidence.relative_to(
                        repository
                    ).as_posix(),
                    "file_sha256": sha256_file(retained_upstream_evidence),
                    "evidence_content_sha256": upstream_value["evidence_content_sha256"],
                    "verification_status": upstream_status,
                },
                "upstream_m5_request": {
                    "path": "provenance/upstream_m5_request.json",
                    "file_sha256": sha256_file(retained_upstream_request),
                },
                "upstream_m5_timeline": {
                    "path": "provenance/upstream_m5_timeline.json",
                    "file_sha256": sha256_file(retained_upstream_timeline),
                },
                "upstream_m5_base_videos": {
                    delivery_id: {
                        "path": retained.relative_to(staging).as_posix(),
                        "file_sha256": sha256_file(retained),
                    }
                    for delivery_id, retained in retained_upstream_videos.items()
                },
                "derivation": {
                    "kind": "verified_retained_evidence_program_materialization_v1",
                    "native_rir_rerun": False,
                    "current_native_episode_status": "not_run",
                    "semantic_materialization_verifier_status": runtime_report[
                        "overall_status"
                    ],
                    "selected_upstream_variant": variant,
                    "selected_active_upstream_source_id": active_source,
                    "selected_upstream_source_id_for_silenced_endpoint": silent_source,
                    "derived_silent_source_endpoint_id": silent_endpoint,
                    "rir_sequences_retained_for_both_sources": True,
                    "visual_packet_stream_copied_without_reencoding": True,
                },
                "registry_hashes": {
                    key: value.get("registry_content_sha256")
                    for key, value in registries.items()
                },
                "registry_artifacts": {
                    key: {
                        "path": path.relative_to(staging).as_posix(),
                        "file_sha256": sha256_file(path),
                    }
                    for key, path in retained_registry_paths.items()
                },
                "audio_program_content_sha256": program["program_content_sha256"],
                "natural_language_qa_generated": False,
            },
            "provenance_content_sha256",
        )
        write_json(staging / "provenance" / "provenance_manifest.json", provenance)

        final_status = _bind_document(
            {
                "schema": FINAL_STATUS_SCHEMA,
                "run_id": request["run_id"],
                "controlled_canary_status": runtime_report["overall_status"],
                **_controlled_execution_contract(),
                "research_only": True,
                "qualification_claim": False,
                "qualified_room_revision_created": False,
                "dataset_admission": False,
                "active_source_endpoint_ids": [active_endpoint],
                "silent_source_endpoint_ids": [silent_endpoint],
                "spatial_audio": {
                    "binaural_360_degree": True,
                    "foa_authority_retained": True,
                },
                "natural_language_qa_generated": False,
                "room_admission_blockers": room_report["admission_blockers"],
            },
            "status_content_sha256",
        )
        write_json(staging / "final_status.json", final_status)

        evidence: dict[str, Any] = {
            "schema": EVIDENCE_SCHEMA,
            "run_id": request["run_id"],
            "evidence_kind": "controlled_one_active_of_n",
            **_controlled_execution_contract(),
            "research_only": True,
            "qualification_claim": False,
            "dataset_admission": False,
            "implementation_commit": implementation_commit,
            "request": file_record(staging / "request.json", relative_to=staging),
            "release_manifest_ref": file_record(
                staging / "release_manifest_ref.json", relative_to=staging
            ),
            "upstream_evidence": {
                "kind": "verified_m5_controlled_bundle",
                "status": upstream_status,
                "path": "provenance/upstream_m5_evidence.json",
                "sha256": sha256_file(retained_upstream_evidence),
            },
            "artifacts": _artifact_index(staging),
            "checks": checks,
            "overall_status": runtime_report["overall_status"],
        }
        evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)
        schema_errors = _schema_errors(evidence, "m6_canary_evidence_v1.schema.json")
        if schema_errors:
            raise M6CanaryError([f"evidence: {item}" for item in schema_errors])
        if not _json_no_qa_pairs(evidence):
            raise M6CanaryError("natural-language QA content is forbidden in M6")
        write_json(staging / "evidence.json", evidence)
        published = atomic_publish_directory(policy, staging, destination)
        return published / "evidence.json"
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _artifact_errors(root: Path, evidence: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    records = evidence.get("artifacts")
    if not isinstance(records, Mapping):
        return ["artifacts is not an object"]
    for relative, record in records.items():
        if (
            not isinstance(relative, str)
            or not relative
            or "\\" in relative
            or not isinstance(record, Mapping)
        ):
            errors.append(f"artifact key is not a portable record: {relative!r}")
            continue
        portable = PurePosixPath(relative)
        if portable.is_absolute() or any(
            part in {"", ".", ".."} for part in portable.parts
        ):
            errors.append(f"artifact path is not confined: {relative}")
            continue
        if record.get("path") != relative:
            errors.append(f"artifact record path differs from its key: {relative}")
        unresolved_candidate = root / Path(*portable.parts)
        current = root
        traverses_symlink = False
        for part in portable.parts:
            current = current / part
            if current.is_symlink():
                traverses_symlink = True
                break
        if traverses_symlink:
            errors.append(f"artifact must not traverse a symlink: {relative}")
            continue
        candidate = unresolved_candidate.resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            errors.append(f"artifact escapes bundle: {relative}")
            continue
        if not candidate.is_file():
            errors.append(f"artifact is missing: {relative}")
            continue
        if candidate.stat().st_size != record.get("byte_size"):
            errors.append(f"artifact byte size differs: {relative}")
        if sha256_file(candidate) != record.get("sha256"):
            errors.append(f"artifact SHA-256 differs: {relative}")
    entries = list(root.rglob("*"))
    symlinks = sorted(
        path.relative_to(root).as_posix()
        for path in entries
        if path.is_symlink()
    )
    if symlinks:
        errors.append(f"retained bundle contains symlinks: {symlinks}")
    actual = {
        path.relative_to(root).as_posix()
        for path in entries
        if not path.is_symlink()
        and path.is_file()
        and path != root / "evidence.json"
    }
    if actual != set(records):
        errors.append("artifact index is not an exact retained-file closure")
    return errors


def _bound_document_errors(
    value: Mapping[str, Any], field: str, *, owner: str
) -> list[str]:
    core = dict(value)
    declared = core.pop(field, None)
    recomputed = canonical_json_sha256(core)
    return [] if declared == recomputed else [f"{owner} content hash differs"]


def _record_matches_artifact(
    evidence: Mapping[str, Any], relative: str, record: Any
) -> bool:
    artifacts = evidence.get("artifacts")
    return (
        isinstance(record, Mapping)
        and isinstance(artifacts, Mapping)
        and record == artifacts.get(relative)
        and record.get("path") == relative
    )


def verify_controlled_canary_evidence(
    evidence_path: str | Path,
    *,
    ffmpeg: str | Path = "ffmpeg",
    ffprobe: str | Path = "ffprobe",
) -> tuple[str, list[dict[str, Any]]]:
    """Rehash and semantically verify a published controlled M6 bundle."""

    unresolved_path = Path(evidence_path).absolute()
    symlink_components: list[str] = []
    cursor = Path(unresolved_path.anchor)
    for part in unresolved_path.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            symlink_components.append(str(cursor))
    path = unresolved_path.resolve(strict=True)
    root = path.parent
    evidence = load_json(path)
    checks: list[dict[str, Any]] = []

    def add(check_id: str, passed: bool, measured: Any) -> None:
        checks.append(
            {
                "check_id": check_id,
                "status": "pass" if passed else "fail",
                "measured": measured,
            }
        )

    add("entry_path_no_symlink", not symlink_components, symlink_components)

    schema_errors = _schema_errors(evidence, "m6_canary_evidence_v1.schema.json")
    add("evidence_schema", not schema_errors, schema_errors)
    core = dict(evidence)
    declared = core.pop("evidence_content_sha256", None)
    recomputed = canonical_json_sha256(core)
    add(
        "evidence_content_hash",
        declared == recomputed,
        {"declared": declared, "recomputed": recomputed},
    )
    artifact_errors = _artifact_errors(root, evidence)
    add("artifact_closure", not artifact_errors, artifact_errors)

    implementation_errors: list[str] = []
    canonical_inputs: Mapping[str, Any] = {}
    implementation_commit = evidence.get("implementation_commit")
    try:
        if not isinstance(implementation_commit, str) or _COMMIT.fullmatch(
            implementation_commit
        ) is None:
            implementation_errors.append("implementation commit is not canonical")
        else:
            _git(_repository_root(), "cat-file", "-e", f"{implementation_commit}^{{commit}}")
            ancestor = subprocess.run(
                [
                    "git",
                    "-C",
                    str(_repository_root()),
                    "merge-base",
                    "--is-ancestor",
                    implementation_commit,
                    "HEAD",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if ancestor.returncode != 0:
                implementation_errors.append(
                    "implementation commit is not an ancestor of the verifier checkout"
                )
            else:
                canonical_inputs = _canonical_implementation_inputs(
                    _repository_root(), implementation_commit
                )
    except (M6CanaryError, OSError) as error:
        implementation_errors.append(str(error))
    add(
        "implementation_commit_reachability",
        not implementation_errors,
        implementation_errors,
    )

    request: Mapping[str, Any] = {}
    program: Mapping[str, Any] = {}
    source_manifest: Mapping[str, Any] = {}
    entity_manifest: Mapping[str, Any] = {}
    retained_upstream: Mapping[str, Any] = {}
    retained_upstream_request: Mapping[str, Any] = {}
    retained_upstream_timeline: Mapping[str, Any] = {}
    retained_registries: Mapping[str, Mapping[str, Any]] = {}
    runtime_report: Mapping[str, Any] = {}
    final: Mapping[str, Any] = {}
    trajectory: Mapping[str, Any] = {}
    document_errors: list[str] = []
    try:
        request = load_controlled_canary_request(root / "request.json")
        program = load_json(root / "source_program.json")
        source_manifest = load_json(root / "source_manifest.json")
        entity_manifest = load_json(root / "entity_instances.json")
        runtime_report = load_json(root / "qa" / "runtime_qa_report.json")
        final = load_json(root / "final_status.json")
        provenance = load_json(root / "provenance" / "provenance_manifest.json")
        release_ref = load_json(root / "release_manifest_ref.json")
        room_manifest = load_json(root / "room_manifest.json")
        room_report = load_qualification_report(
            root / "room_qualification_report.json"
        )
        trajectory = load_json(root / "provenance" / "upstream_emitter_path.json")
        retained_registry_root = root / "provenance" / "registries"
        retained_registries = _load_registries(retained_registry_root)
        validated_program = load_audio_program(
            root / "source_program.json",
            source_endpoint_registry=retained_registries["endpoints"],
            sound_asset_registry=retained_registries["sounds"],
        )
        if validated_program != program:
            document_errors.append("AudioProgram registry validation projection differs")
        upstream_record = evidence.get("upstream_evidence", {})
        upstream_relative = upstream_record.get("path")
        if upstream_relative != "provenance/upstream_m5_evidence.json":
            document_errors.append("upstream evidence does not name its retained copy")
        retained_upstream_path = root / "provenance" / "upstream_m5_evidence.json"
        retained_upstream = load_json(retained_upstream_path)
        retained_upstream_request_path = (
            root / "provenance" / "upstream_m5_request.json"
        )
        retained_upstream_request = load_json(retained_upstream_request_path)
        retained_upstream_timeline_path = (
            root / "provenance" / "upstream_m5_timeline.json"
        )
        retained_upstream_timeline = load_json(retained_upstream_timeline_path)

        for value, field, owner in (
            (program, "program_content_sha256", "AudioProgram"),
            (source_manifest, "manifest_content_sha256", "source manifest"),
            (entity_manifest, "manifest_content_sha256", "entity manifest"),
            (runtime_report, "report_content_sha256", "runtime report"),
            (final, "status_content_sha256", "final status"),
            (provenance, "provenance_content_sha256", "provenance manifest"),
            (trajectory, "trajectory_content_sha256", "retained trajectory"),
            (retained_upstream, "evidence_content_sha256", "retained M5 evidence"),
        ):
            document_errors.extend(
                _bound_document_errors(value, field, owner=owner)
            )

        if not _record_matches_artifact(evidence, "request.json", evidence.get("request")):
            document_errors.append("top-level request record is not artifact-bound")
        if not _record_matches_artifact(
            evidence,
            "release_manifest_ref.json",
            evidence.get("release_manifest_ref"),
        ):
            document_errors.append(
                "top-level release manifest reference is not artifact-bound"
            )
        if (
            upstream_record.get("sha256") != sha256_file(retained_upstream_path)
            or retained_upstream.get("evidence_content_sha256")
            != request["upstream_evidence"]["evidence_content_sha256"]
            or retained_upstream.get("overall_status") != "pass"
            or not retained_upstream.get("checks")
            or any(
                item.get("status") != "pass"
                for item in retained_upstream.get("checks", [])
            )
        ):
            document_errors.append("retained M5 evidence identity/status differs")
        upstream_request_record = retained_upstream.get("artifacts", {}).get(
            "inputs/request.json", {}
        )
        if (
            sha256_file(retained_upstream_request_path)
            != upstream_request_record.get("sha256")
            or retained_upstream_request_path.stat().st_size
            != upstream_request_record.get("byte_size")
            or program.get("source_program_provenance", {}).get(
                "upstream_request_sha256"
            )
            != upstream_request_record.get("sha256")
            or provenance.get("upstream_m5_request", {}).get("file_sha256")
            != upstream_request_record.get("sha256")
        ):
            document_errors.append("retained M5 request identity differs")
        upstream_variant = request["upstream_evidence"]["episode_variant"]
        upstream_timeline_record = retained_upstream.get("artifacts", {}).get(
            f"episodes/{upstream_variant}/timeline.json", {}
        )
        if (
            sha256_file(retained_upstream_timeline_path)
            != upstream_timeline_record.get("sha256")
            or retained_upstream_timeline_path.stat().st_size
            != upstream_timeline_record.get("byte_size")
            or provenance.get("upstream_m5_timeline", {}).get("file_sha256")
            != upstream_timeline_record.get("sha256")
        ):
            document_errors.append("retained M5 timeline identity differs")
        for delivery_id, filename in (
            ("primary", "view0_base_video_only.mp4"),
            ("topdown", "view0_topdown_base_video_only.mp4"),
        ):
            retained_video_path = root / "provenance" / f"upstream_{filename}"
            upstream_video_record = retained_upstream.get("artifacts", {}).get(
                f"videos/{filename}", {}
            )
            provenance_video = provenance.get("upstream_m5_base_videos", {}).get(
                delivery_id, {}
            )
            if (
                sha256_file(retained_video_path)
                != upstream_video_record.get("sha256")
                or retained_video_path.stat().st_size
                != upstream_video_record.get("byte_size")
                or provenance_video.get("file_sha256")
                != upstream_video_record.get("sha256")
                or provenance_video.get("path")
                != f"provenance/upstream_{filename}"
            ):
                document_errors.append(
                    f"retained M5 {delivery_id} base video identity differs"
                )
        requested_release = request["release_manifest_ref"]
        expected_release_ref = {
            "schema": "avengine_release_manifest_ref_v1",
            "release_id": requested_release["release_id"],
            "repository_path": requested_release["repository_path"],
            "expected_tag": requested_release["expected_tag"],
            "implementation_commit": implementation_commit,
        }
        if release_ref != expected_release_ref:
            document_errors.append("release manifest reference differs from request/evidence")
        if (
            provenance.get("implementation_commit") != implementation_commit
            or provenance.get("audio_program_content_sha256")
            != program.get("program_content_sha256")
            or provenance.get("upstream_m5_evidence", {}).get("file_sha256")
            != sha256_file(retained_upstream_path)
            or provenance.get("upstream_m5_evidence", {}).get(
                "evidence_content_sha256"
            )
            != retained_upstream.get("evidence_content_sha256")
            or provenance.get("natural_language_qa_generated") is not False
        ):
            document_errors.append("provenance manifest does not close over its inputs")
        expected_registry_hashes = {
            key: value["registry_content_sha256"]
            for key, value in retained_registries.items()
        }
        if provenance.get("registry_hashes") != expected_registry_hashes:
            document_errors.append("provenance registry hashes differ")
        for registry_id, filename in _REGISTRY_FILES.items():
            registry_path = retained_registry_root / filename
            registry_artifact = provenance.get("registry_artifacts", {}).get(
                registry_id, {}
            )
            if (
                registry_artifact.get("path")
                != f"provenance/registries/{filename}"
                or registry_artifact.get("file_sha256")
                != sha256_file(registry_path)
            ):
                document_errors.append(
                    f"retained {registry_id} registry artifact binding differs"
                )
        canonical_room_registry = canonical_inputs.get("room_registry", {})
        canonical_room_qualification = canonical_inputs.get(
            "room_qualification", {}
        )
        document_errors.extend(
            _implementation_input_differences(
                canonical_inputs,
                request=request,
                program=program,
                registries=retained_registries,
                room_registry=canonical_room_registry,
                room_qualification=canonical_room_qualification,
            )
        )
        expected_runtime_report = _recompute_runtime_report(
            root=root,
            retained_upstream=retained_upstream,
            program=program,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
        )
        if runtime_report != expected_runtime_report:
            document_errors.append(
                "runtime QA report differs from independent bundle readback"
            )
        committed_room_record = find_room_record(
            canonical_room_registry,
            request["room"]["room_id"],
            request["room"]["revision"],
        )
        if room_manifest != committed_room_record:
            document_errors.append(
                "room manifest differs from implementation-commit room authority"
            )
        expected_room_report = _retained_materialization_room_report(
            historical_report=canonical_room_qualification,
            upstream_evidence=retained_upstream_path,
            materialization_status=expected_runtime_report["overall_status"],
        )
        if room_report != expected_room_report:
            document_errors.append(
                "room qualification report differs from the committed qualification "
                "input plus deterministic retained-materialization result"
            )
        expected_entities, expected_sources = _build_entities_and_sources(
            request=request,
            registries=retained_registries,
            program=program,
            trajectory=trajectory,
        )
        if entity_manifest != expected_entities:
            document_errors.append(
                "entity manifest differs from committed registries and request"
            )
        if source_manifest != expected_sources:
            document_errors.append(
                "source manifest differs from committed registries and AudioProgram"
            )
        compiled_for_documents = compile_audio_program(program)
        active_endpoint_for_documents = (
            compiled_for_documents.active_source_endpoint_ids[0]
        )
        silent_endpoint_for_documents = (
            compiled_for_documents.silent_source_endpoint_ids[0]
        )
        source_repository_path = provenance.get("upstream_m5_evidence", {}).get(
            "source_repository_path"
        )
        source_repository_locator = (
            PurePosixPath(source_repository_path)
            if isinstance(source_repository_path, str)
            else None
        )
        if (
            source_repository_locator is None
            or source_repository_locator.is_absolute()
            or any(
                part in {"", ".", ".."}
                for part in source_repository_locator.parts
            )
        ):
            document_errors.append(
                "upstream M5 source repository path is not portable"
            )
        expected_provenance = _bind_document(
            {
                "schema": PROVENANCE_SCHEMA,
                "implementation_commit": implementation_commit,
                "upstream_m5_evidence": {
                    "path": "provenance/upstream_m5_evidence.json",
                    "source_repository_path": source_repository_path,
                    "file_sha256": sha256_file(retained_upstream_path),
                    "evidence_content_sha256": retained_upstream[
                        "evidence_content_sha256"
                    ],
                    "verification_status": "pass",
                },
                "upstream_m5_request": {
                    "path": "provenance/upstream_m5_request.json",
                    "file_sha256": sha256_file(retained_upstream_request_path),
                },
                "upstream_m5_timeline": {
                    "path": "provenance/upstream_m5_timeline.json",
                    "file_sha256": sha256_file(retained_upstream_timeline_path),
                },
                "upstream_m5_base_videos": {
                    delivery_id: {
                        "path": f"provenance/upstream_{filename}",
                        "file_sha256": sha256_file(
                            root / "provenance" / f"upstream_{filename}"
                        ),
                    }
                    for delivery_id, filename in (
                        ("primary", "view0_base_video_only.mp4"),
                        ("topdown", "view0_topdown_base_video_only.mp4"),
                    )
                },
                "derivation": {
                    "kind": (
                        "verified_retained_evidence_program_materialization_v1"
                    ),
                    "native_rir_rerun": False,
                    "current_native_episode_status": "not_run",
                    "semantic_materialization_verifier_status": (
                        expected_runtime_report["overall_status"]
                    ),
                    "selected_upstream_variant": request["upstream_evidence"][
                        "episode_variant"
                    ],
                    "selected_active_upstream_source_id": request[
                        "endpoint_to_upstream_source_id"
                    ][active_endpoint_for_documents],
                    "selected_upstream_source_id_for_silenced_endpoint": request[
                        "endpoint_to_upstream_source_id"
                    ][silent_endpoint_for_documents],
                    "derived_silent_source_endpoint_id": (
                        silent_endpoint_for_documents
                    ),
                    "rir_sequences_retained_for_both_sources": True,
                    "visual_packet_stream_copied_without_reencoding": True,
                },
                "registry_hashes": {
                    key: value["registry_content_sha256"]
                    for key, value in retained_registries.items()
                },
                "registry_artifacts": {
                    registry_id: {
                        "path": f"provenance/registries/{filename}",
                        "file_sha256": sha256_file(
                            retained_registry_root / filename
                        ),
                    }
                    for registry_id, filename in _REGISTRY_FILES.items()
                },
                "audio_program_content_sha256": program[
                    "program_content_sha256"
                ],
                "natural_language_qa_generated": False,
            },
            "provenance_content_sha256",
        )
        if provenance != expected_provenance:
            document_errors.append(
                "provenance manifest differs from independently reconstructed inputs"
            )
        expected_final = _bind_document(
            {
                "schema": FINAL_STATUS_SCHEMA,
                "run_id": request["run_id"],
                "controlled_canary_status": expected_runtime_report[
                    "overall_status"
                ],
                **_controlled_execution_contract(),
                "research_only": True,
                "qualification_claim": False,
                "qualified_room_revision_created": False,
                "dataset_admission": False,
                "active_source_endpoint_ids": list(
                    compiled_for_documents.active_source_endpoint_ids
                ),
                "silent_source_endpoint_ids": list(
                    compiled_for_documents.silent_source_endpoint_ids
                ),
                "spatial_audio": {
                    "binaural_360_degree": True,
                    "foa_authority_retained": True,
                },
                "natural_language_qa_generated": False,
                "room_admission_blockers": expected_room_report[
                    "admission_blockers"
                ],
            },
            "status_content_sha256",
        )
        if final != expected_final:
            document_errors.append(
                "final status differs from the independently reconstructed result"
            )
        if {
            key: evidence.get(key)
            for key in (
                "run_id",
                "evidence_kind",
                "execution_basis",
                "status_scope",
                "native_execution",
                "research_only",
                "qualification_claim",
                "dataset_admission",
            )
        } != {
            "run_id": request["run_id"],
            "evidence_kind": "controlled_one_active_of_n",
            **_controlled_execution_contract(),
            "research_only": True,
            "qualification_claim": False,
            "dataset_admission": False,
        }:
            document_errors.append(
                "top-level evidence claim boundary differs from the committed request"
            )
        if (
            expected_runtime_report.get("checks") != evidence.get("checks")
            or expected_runtime_report.get("overall_status")
            != evidence.get("overall_status")
            or expected_final.get("controlled_canary_status")
            != evidence.get("overall_status")
            or evidence.get("overall_status") != "pass"
            or any(item.get("status") != "pass" for item in evidence.get("checks", []))
        ):
            document_errors.append("runtime/evidence/final status closure differs or failed")
        document_errors.extend(
            _retained_materialization_claim_errors(
                evidence=evidence,
                provenance=provenance,
                room_report=room_report,
            )
        )
        if (
            (room_manifest.get("room_id"), room_manifest.get("revision"))
            != (request["room"]["room_id"], request["room"]["revision"])
            or room_report.get("report_id")
            != "blender_custom_two_zone_m6_controlled_attempt_v1"
            or room_report.get("dataset_admission") is not False
            or final.get("room_admission_blockers")
            != room_report.get("admission_blockers")
        ):
            document_errors.append("room attempt/report claim boundary differs")
    except (OSError, ValueError, KeyError, TypeError) as error:
        document_errors.append(str(error))
    add("bound_document_closure", not document_errors, document_errors)

    try:
        compiled = compile_audio_program(program)
        one_active = (
            len(compiled.candidate_source_endpoint_ids) == 2
            and len(compiled.active_source_endpoint_ids) == 1
            and len(compiled.silent_source_endpoint_ids) == 1
            and source_manifest.get("active_source_endpoint_ids")
            == list(compiled.active_source_endpoint_ids)
            and source_manifest.get("silent_source_endpoint_ids")
            == list(compiled.silent_source_endpoint_ids)
        )
        add(
            "one_active_of_two_contract",
            one_active,
            {
                "active": list(compiled.active_source_endpoint_ids),
                "silent": list(compiled.silent_source_endpoint_ids),
                "event_count": len(compiled.events),
            },
        )
    except (OSError, ValueError) as error:
        request = {}
        compiled = None
        add("one_active_of_two_contract", False, str(error))

    binding_errors: list[str] = []
    try:
        _validate_entity_visual_authority(
            request=request,
            registries=retained_registries,
            upstream_request=retained_upstream_request,
        )
        authority_assets = {
            (item["entity_asset_id"], item["revision"]): item
            for item in retained_registries["entities"]["entities"]
        }
        authority_templates = {
            (item["template_id"], item["revision"]): item
            for item in retained_registries["animals"]["templates"]
        }
        authority_endpoints = endpoint_index(retained_registries["endpoints"])
        authority_sounds = sound_index(retained_registries["sounds"])
        requested_instances = {
            item["entity_instance_id"]: item for item in request["entity_instances"]
        }
        retained_instances = {
            item["entity_instance_id"]: item
            for item in entity_manifest["instances"]
        }
        if set(retained_instances) != set(requested_instances):
            binding_errors.append("entity manifest instance closure differs from request")
        for instance_id, requested in requested_instances.items():
            retained = retained_instances.get(instance_id, {})
            asset = retained.get("asset", {})
            for key in (
                "timeline_actor_id",
                "entity_asset_id",
                "entity_asset_revision",
            ):
                if retained.get(key) != requested.get(key):
                    binding_errors.append(
                        f"entity {instance_id!r} {key} differs from request"
                    )
            if (
                asset.get("entity_asset_id") != requested["entity_asset_id"]
                or asset.get("revision") != requested["entity_asset_revision"]
            ):
                binding_errors.append(
                    f"entity {instance_id!r} retained asset identity differs"
                )
            authority_asset = authority_assets.get(
                (requested["entity_asset_id"], requested["entity_asset_revision"])
            )
            if asset != authority_asset:
                binding_errors.append(
                    f"entity {instance_id!r} differs from authoritative registry record"
                )
            template_ref = asset.get("animal_template_ref", {})
            authority_template = authority_templates.get(
                (template_ref.get("template_id"), template_ref.get("revision"))
            )
            if retained.get("animal_template") != authority_template:
                binding_errors.append(
                    f"entity {instance_id!r} animal template authority differs"
                )

        routes = source_manifest["routes"]
        if [item.get("source_endpoint_id") for item in routes] != request.get(
            "source_endpoint_ids"
        ):
            binding_errors.append("source route order/closure differs from request")
        for route in routes:
            endpoint_id = route["source_endpoint_id"]
            endpoint = route["endpoint"]
            binding = endpoint["binding"]
            requested = requested_instances.get(binding.get("entity_instance_id"), {})
            retained_instance = retained_instances.get(
                binding.get("entity_instance_id"), {}
            )
            expected_state = (
                "active_program_endpoint"
                if compiled is not None
                and endpoint_id in compiled.active_source_endpoint_ids
                else "persistent_silent_endpoint"
            )
            authority_endpoint = authority_endpoints.get(endpoint_id)
            anchor_ids = {
                item["anchor_id"] for item in retained_instance.get("asset", {}).get(
                    "emitter_anchors", []
                )
            }
            if (
                endpoint.get("source_endpoint_id") != endpoint_id
                or endpoint != authority_endpoint
                or route.get("upstream_source_id")
                != request["endpoint_to_upstream_source_id"].get(endpoint_id)
                or route.get("activation_state") != expected_state
                or endpoint.get("persistent_when_silent") is not True
                or binding.get("entity_asset_id")
                != requested.get("entity_asset_id")
                or binding.get("entity_asset_revision")
                != requested.get("entity_asset_revision")
                or route.get("timeline_actor_id")
                != requested.get("timeline_actor_id")
                or binding.get("emitter_anchor_id") not in anchor_ids
                or route.get("trajectory_content_sha256")
                != trajectory.get("trajectory_content_sha256")
            ):
                binding_errors.append(f"source route {endpoint_id!r} binding differs")
        if (
            source_manifest.get("audio_program", {}).get("program_content_sha256")
            != program.get("program_content_sha256")
        ):
            binding_errors.append("source manifest AudioProgram binding differs")
        sound_assets = {
            item["sound_asset_id"]: item
            for item in source_manifest.get("sound_assets", [])
        }
        program_sound_ids = {item["sound_asset_id"] for item in program["events"]}
        if set(sound_assets) != program_sound_ids:
            binding_errors.append("retained sound asset closure differs from AudioProgram")
        if any(
            sound_assets.get(sound_id) != authority_sounds.get(sound_id)
            for sound_id in program_sound_ids
        ):
            binding_errors.append("retained sound assets differ from registry authority")
        expected_endpoint_registry = {
            key: retained_registries["endpoints"][key]
            for key in ("registry_id", "revision", "registry_content_sha256")
        }
        expected_sound_registry = {
            key: retained_registries["sounds"][key]
            for key in ("registry_id", "revision", "registry_content_sha256")
        }
        if source_manifest.get("source_endpoint_registry") != expected_endpoint_registry:
            binding_errors.append("source endpoint registry authority differs")
        if source_manifest.get("sound_asset_registry") != expected_sound_registry:
            binding_errors.append("sound asset registry authority differs")
        if (
            request.get("legacy_flag_registry", {}).get("registry_id")
            != retained_registries["flags"].get("registry_id")
            or request.get("legacy_flag_registry", {}).get("revision")
            != retained_registries["flags"].get("revision")
        ):
            binding_errors.append("legacy flag registry authority differs")
        active_endpoint = compiled.active_source_endpoint_ids[0]
        active_upstream_source = request["endpoint_to_upstream_source_id"][
            active_endpoint
        ]
        active_upstream_event = next(
            item
            for item in retained_upstream_request["events"]
            if item["source_id"] == active_upstream_source
        )
        upstream_program = retained_upstream_request["audio_program"]
        upstream_windows = {
            item["window_id"]: (item["start_sample"], item["end_sample"])
            for item in upstream_program["simultaneous_windows"]
        }
        if {
            item.get("upstream_window_id") for item in program["events"]
        } != set(upstream_windows):
            binding_errors.append("AudioProgram retained window closure differs")
        for item in program["events"]:
            sound = sound_assets.get(item["sound_asset_id"], {})
            if (
                item["source_endpoint_id"] != active_endpoint
                or (
                    item["start_sample"], item["end_sample_exclusive"]
                )
                != upstream_windows.get(item["upstream_window_id"])
                or (
                    item["source_start_sample"],
                    item["source_end_sample_exclusive"],
                    item["linear_gain"],
                    item["fade_samples"],
                )
                != (
                    upstream_program["clip_source_interval"]["start_sample"],
                    upstream_program["clip_source_interval"]["end_sample"],
                    upstream_program["linear_gain"],
                    upstream_program["fade_samples"],
                )
                or sound.get("dry_audio", {}).get("sha256")
                != active_upstream_event["dry_audio_asset_sha256"]
            ):
                binding_errors.append(
                    f"AudioProgram event {item.get('event_id')!r} differs from retained M5"
                )
    except (AttributeError, KeyError, StopIteration, TypeError, ValueError) as error:
        binding_errors.append(str(error))
    add("entity_source_binding_closure", not binding_errors, binding_errors)

    audio_failures: list[str] = []
    if compiled is not None:
        active = compiled.active_source_endpoint_ids[0]
        silent = compiled.silent_source_endpoint_ids[0]
        for layout, (channels, _) in _LAYOUTS.items():
            try:
                active_wav = read_float32_wav(
                    root / "audio" / "source_stems" / layout / f"{active}.wav"
                )
                silent_wav = read_float32_wav(
                    root / "audio" / "source_stems" / layout / f"{silent}.wav"
                )
                mixture_wav = read_float32_wav(
                    root / "audio" / ("mixture.wav" if layout == "binaural" else "mixture_foa.wav")
                )
                expected_shape = (channels, 80_000)
                if active_wav.samples.shape != expected_shape:
                    audio_failures.append(f"{layout} active shape differs")
                if (
                    active_wav.sample_rate_hz != 16_000
                    or silent_wav.sample_rate_hz != 16_000
                    or mixture_wav.sample_rate_hz != 16_000
                ):
                    audio_failures.append(f"{layout} sample rate differs")
                if silent_wav.samples.shape != expected_shape or np.any(silent_wav.samples != 0.0):
                    audio_failures.append(f"{layout} silent stem is not exact zero")
                if mixture_wav.samples.shape != expected_shape:
                    audio_failures.append(f"{layout} mixture shape differs")
                expected_mix = active_wav.samples + silent_wav.samples
                if not np.array_equal(mixture_wav.samples, expected_mix):
                    audio_failures.append(f"{layout} mixture is not exact canonical stem sum")
                if not np.any(active_wav.samples):
                    audio_failures.append(f"{layout} active stem is silent")
                active_metadata = (
                    active_wav.sidecar.get("metadata", {})
                    if isinstance(active_wav.sidecar, Mapping)
                    else {}
                )
                silent_metadata = (
                    silent_wav.sidecar.get("metadata", {})
                    if isinstance(silent_wav.sidecar, Mapping)
                    else {}
                )
                mixture_metadata = (
                    mixture_wav.sidecar.get("metadata", {})
                    if isinstance(mixture_wav.sidecar, Mapping)
                    else {}
                )
                expected_layout_id = (
                    "rlr_binaural_lr_v1"
                    if layout == "binaural"
                    else "rlr_foa_acn_n3d_world_v1"
                )
                for owner, metadata in (
                    ("active", active_metadata),
                    ("silent", silent_metadata),
                    ("mixture", mixture_metadata),
                ):
                    if (
                        metadata.get("layout_id") != expected_layout_id
                        or metadata.get("channel_labels")
                        != list(_LAYOUTS[layout][1])
                        or metadata.get("spatial_format")
                        != _spatial_format(layout)
                        or metadata.get("program_id") != program.get("program_id")
                        or metadata.get("program_content_sha256")
                        != program.get("program_content_sha256")
                    ):
                        audio_failures.append(
                            f"{layout} {owner} spatial/program sidecar differs"
                        )
                if (
                    active_metadata.get("role") != "dynamic_wet_stem"
                    or active_metadata.get("source_endpoint_id") != active
                    or active_metadata.get("active") is not True
                    or silent_metadata.get("role") != "dynamic_wet_stem"
                    or silent_metadata.get("source_endpoint_id") != silent
                    or silent_metadata.get("active") is not False
                    or mixture_metadata.get("role")
                    != "source_stem_sum_mixture"
                    or mixture_metadata.get("canonical_source_endpoint_order")
                    != [active, silent]
                    or mixture_metadata.get("active_source_endpoint_ids")
                    != [active]
                    or mixture_metadata.get("silent_source_endpoint_ids")
                    != [silent]
                ):
                    audio_failures.append(
                        f"{layout} stem/mixture sidecar endpoint roles differ"
                    )
                upstream_record = (
                    active_wav.sidecar.get("metadata", {}).get("upstream_artifact", {})
                    if active_wav.sidecar is not None
                    else {}
                )
                upstream_source_id = request["endpoint_to_upstream_source_id"][active]
                upstream_role = (
                    f"episodes/{request['upstream_evidence']['episode_variant']}"
                    f"/audio/{layout}/{upstream_source_id}_stem.wav"
                )
                retained_record = retained_upstream.get("artifacts", {}).get(
                    upstream_role, {}
                )
                active_path = (
                    root / "audio" / "source_stems" / layout / f"{active}.wav"
                )
                if (
                    upstream_record.get("sha256") != sha256_file(active_path)
                    or upstream_record.get("byte_size") != active_path.stat().st_size
                    or upstream_record.get("sha256")
                    != retained_record.get("sha256")
                    or upstream_record.get("byte_size")
                    != retained_record.get("byte_size")
                ):
                    audio_failures.append(
                        f"{layout} active stem differs from bound verified M5 artifact"
                    )
            except (OSError, ValueError) as error:
                audio_failures.append(f"{layout}: {error}")
        try:
            active_dry = read_float32_wav(root / "audio" / "dry_buses" / f"{active}.wav")
            silent_dry = read_float32_wav(root / "audio" / "dry_buses" / f"{silent}.wav")
            permitted = np.zeros(80_000, dtype=bool)
            for event in compiled.events:
                permitted[event.start_sample : event.end_sample_exclusive] = True
            if np.any(active_dry.samples[0, ~permitted] != 0.0):
                audio_failures.append("active dry bus escapes declared AudioProgram windows")
            if active_dry.samples.shape != (1, 80_000) or active_dry.sample_rate_hz != 16_000:
                audio_failures.append("active dry bus shape/rate differs")
            for event in compiled.events:
                if not np.any(
                    active_dry.samples[
                        0, event.start_sample : event.end_sample_exclusive
                    ]
                ):
                    audio_failures.append(
                        f"active dry event {event.event_id} contains no nonzero sample"
                    )
            active_dry_path = root / "audio" / "dry_buses" / f"{active}.wav"
            upstream_dry_record = (
                active_dry.sidecar.get("metadata", {}).get("upstream_artifact", {})
                if active_dry.sidecar is not None
                else {}
            )
            upstream_source_id = request["endpoint_to_upstream_source_id"][active]
            upstream_role = (
                f"episodes/{request['upstream_evidence']['episode_variant']}"
                f"/audio/dry/{upstream_source_id}.wav"
            )
            retained_dry_record = retained_upstream.get("artifacts", {}).get(
                upstream_role, {}
            )
            if (
                upstream_dry_record.get("sha256") != sha256_file(active_dry_path)
                or upstream_dry_record.get("byte_size")
                != active_dry_path.stat().st_size
                or upstream_dry_record.get("sha256")
                != retained_dry_record.get("sha256")
                or upstream_dry_record.get("byte_size")
                != retained_dry_record.get("byte_size")
            ):
                audio_failures.append(
                    "active dry bus differs from bound verified M5 artifact"
                )
            if (
                silent_dry.samples.shape != (1, 80_000)
                or silent_dry.sample_rate_hz != 16_000
            ):
                audio_failures.append("silent dry bus shape/rate differs")
            if np.any(silent_dry.samples != 0.0):
                audio_failures.append("silent dry bus is not exact zero")
        except (OSError, ValueError) as error:
            audio_failures.append(f"dry buses: {error}")
    add("source_stem_and_mixture_reconstruction", not audio_failures, audio_failures)

    trajectory_errors: list[str] = []
    try:
        trajectory_path = root / "provenance" / "upstream_emitter_path.json"
        upstream_trajectory_record = retained_upstream.get("artifacts", {}).get(
            "trajectory/emitter_path.json", {}
        )
        if (
            upstream_trajectory_record.get("sha256") != sha256_file(trajectory_path)
            or upstream_trajectory_record.get("byte_size")
            != trajectory_path.stat().st_size
        ):
            trajectory_errors.append("trajectory differs from retained M5 artifact")
        expected_sources = [
            request["endpoint_to_upstream_source_id"][endpoint_id]
            for endpoint_id in request["source_endpoint_ids"]
        ]
        keyframes = trajectory.get("keyframes")
        if trajectory.get("source_ids") != expected_sources:
            trajectory_errors.append("trajectory source order differs from request")
        if not isinstance(keyframes, list) or len(keyframes) != 75:
            trajectory_errors.append("trajectory does not contain 75 keyframes")
        elif [item.get("sample_index") for item in keyframes] != [
            (3_200 * index + 1) // 3 for index in range(75)
        ]:
            trajectory_errors.append("trajectory rational 15 Hz sample grid differs")
        elif any(
            set(item.get("source_positions_m", {})) != set(expected_sources)
            for item in keyframes
        ):
            trajectory_errors.append("trajectory source-position closure differs")
    except (OSError, ValueError, KeyError, TypeError) as error:
        trajectory_errors.append(str(error))
    add("retained_trajectory_authority", not trajectory_errors, trajectory_errors)

    rir_failures: list[str] = []
    rir_trajectory_hashes: set[str] = set()
    expected_rir_metadata = {
        "foa": {
            "layout_id": "rlr_foa_acn_n3d_world_v1",
            "layout_type": "ambisonics",
            "channel_labels": ["W", "Y", "Z", "X"],
            "coordinate_frame": "avengine_world",
            "normalization": "N3D",
        },
        "binaural": {
            "layout_id": "rlr_binaural_lr_v1",
            "layout_type": "binaural",
            "channel_labels": ["left", "right"],
            "coordinate_frame": "listener_local",
            "normalization": "not_applicable",
        },
    }
    for layout, (channels, _) in _LAYOUTS.items():
        rir_root = root / "audio" / "rir_or_rir_references" / layout
        try:
            metadata = load_json(rir_root / "metadata.json")
            samples = np.load(rir_root / "samples.npy", allow_pickle=False)
            lengths = np.load(rir_root / "lengths.npy", allow_pickle=False)
            rir_failures.extend(
                _bound_document_errors(
                    metadata,
                    "reference_content_sha256",
                    owner=f"{layout} retained RIR reference",
                )
            )
            projection_sha256 = canonical_json_sha256(_rir_projection(metadata))
            if (
                projection_sha256
                != request.get("upstream_evidence", {}).get(
                    "rir_projection_sha256_by_layout", {}
                ).get(layout)
                or metadata.get("upstream_projection_sha256")
                != projection_sha256
            ):
                rir_failures.append(
                    f"{layout} RIR metadata projection differs from request authority"
                )
            arrays_valid = True
            if (
                samples.ndim != 4
                or samples.shape[:3] != (75, 2, channels)
                or samples.dtype != np.dtype("<f4")
                or not np.all(np.isfinite(samples))
            ):
                arrays_valid = False
                rir_failures.append(
                    f"{layout} retained RIR samples violate [75,2,{channels},L] float32"
                )
            if (
                lengths.shape != (75, 2)
                or lengths.dtype.kind != "u"
                or (lengths.shape == (75, 2) and np.any(lengths < 2))
                or (
                    samples.ndim == 4
                    and lengths.shape == (75, 2)
                    and np.any(lengths > samples.shape[3])
                )
            ):
                arrays_valid = False
                rir_failures.append(f"{layout} retained RIR lengths are invalid")
            expected_sources = [
                request.get("endpoint_to_upstream_source_id", {}).get(endpoint_id)
                for endpoint_id in request.get("source_endpoint_ids", [])
            ]
            expected_metadata = {
                **expected_rir_metadata[layout],
                "sample_rate_hz": 16_000,
                "source_ids": expected_sources,
                "listener_id": request.get("listener", {}).get("listener_id"),
                "trajectory_sha256": trajectory.get("trajectory_content_sha256"),
            }
            for key, expected_value in expected_metadata.items():
                if metadata.get(key) != expected_value:
                    rir_failures.append(f"{layout} RIR {key} differs")
            trajectory_hash = metadata.get("trajectory_sha256")
            if isinstance(trajectory_hash, str):
                rir_trajectory_hashes.add(trajectory_hash)
            if metadata.get("runtime_paths_republished") is not False:
                rir_failures.append(f"{layout} runtime path redaction is not declared")
            if "/data/" in json.dumps(metadata, sort_keys=True):
                rir_failures.append(f"{layout} metadata republishes a private absolute path")
            hashes = metadata.get("ir_sha256_by_frame_source")
            if not isinstance(hashes, list) or len(hashes) != 75:
                arrays_valid = False
                rir_failures.append(f"{layout} per-RIR hashes are absent")
            if arrays_valid:
                for frame_index in range(75):
                    frame_hashes = hashes[frame_index]
                    if not isinstance(frame_hashes, Mapping):
                        rir_failures.append(
                            f"{layout} RIR hash record {frame_index} is invalid"
                        )
                        continue
                    for source_index, source_id in enumerate(expected_sources):
                        length = int(lengths[frame_index, source_index])
                        unpadded = np.ascontiguousarray(
                            samples[frame_index, source_index, :, :length],
                            dtype="<f4",
                        )
                        observed_hash = hashlib.sha256(
                            unpadded.tobytes(order="C")
                        ).hexdigest()
                        if frame_hashes.get(source_id) != observed_hash:
                            rir_failures.append(
                                f"{layout} RIR hash differs at {frame_index}/{source_id}"
                            )
                        if np.any(
                            samples[frame_index, source_index, :, length:] != 0.0
                        ):
                            rir_failures.append(
                                f"{layout} RIR padding is nonzero at "
                                f"{frame_index}/{source_id}"
                            )
            for filename in ("samples.npy", "lengths.npy"):
                record = metadata.get("retained_arrays", {}).get(filename, {})
                candidate = rir_root / filename
                bundle_relative = candidate.relative_to(root).as_posix()
                upstream_record = record.get("upstream_artifact", {})
                authoritative_record = retained_upstream.get("artifacts", {}).get(
                    f"rir/{layout}/{filename}", {}
                )
                if (
                    record.get("path") != bundle_relative
                    or record.get("sha256") != sha256_file(candidate)
                    or record.get("byte_size") != candidate.stat().st_size
                    or upstream_record != authoritative_record
                    or upstream_record.get("sha256") != sha256_file(candidate)
                ):
                    rir_failures.append(f"{layout} {filename} hash binding differs")
            authoritative_metadata_record = retained_upstream.get("artifacts", {}).get(
                f"rir/{layout}/metadata.json", {}
            )
            if metadata.get("upstream_metadata", {}).get("sha256") != (
                authoritative_metadata_record.get("sha256")
            ):
                rir_failures.append(f"{layout} upstream metadata binding differs")
        except (OSError, ValueError, KeyError) as error:
            rir_failures.append(f"{layout}: {error}")
    if rir_trajectory_hashes != {trajectory.get("trajectory_content_sha256")}:
        rir_failures.append("FOA/binaural RIR trajectory binding differs")
    add("retained_per_source_rir_authority", not rir_failures, rir_failures)

    timeline_errors: list[str] = []
    try:
        timeline = load_json(root / "timeline.json")
        timeline_errors.extend(_schema_errors(timeline, "avengine_timeline_v2.schema.json"))
        if compiled is not None:
            active_endpoint = compiled.active_source_endpoint_ids[0]
            active_instance = next(
                route["endpoint"]["binding"]["entity_instance_id"]
                for route in load_json(root / "source_manifest.json")["routes"]
                if route["source_endpoint_id"] == active_endpoint
            )
            entity_manifest = load_json(root / "entity_instances.json")
            instance_to_actor = {
                item["entity_instance_id"]: item["timeline_actor_id"]
                for item in entity_manifest["instances"]
            }
            active_actor = instance_to_actor.get(active_instance)
            active_upstream_source = request["endpoint_to_upstream_source_id"][
                active_endpoint
            ]
            expected_timeline = _build_timeline(
                upstream_timeline=retained_upstream_timeline,
                upstream_request=retained_upstream_request,
                program=program,
                active_source=active_upstream_source,
            )
            if timeline != expected_timeline:
                timeline_errors.append(
                    "timeline visual/audio projection differs from retained M5 derivation"
                )
            upstream_source = next(
                item
                for item in retained_upstream_request["sources"]
                if item["source_id"] == active_upstream_source
            )
            upstream_event = next(
                item
                for item in retained_upstream_request["events"]
                if item["source_id"] == active_upstream_source
            )
            event_actors = {item["actor_id"] for item in timeline["audio_events"]}
            if event_actors != {active_actor}:
                timeline_errors.append("timeline events are not owned only by active endpoint actor")
            if len(timeline["audio_events"]) != len(compiled.events):
                timeline_errors.append("timeline event count differs from AudioProgram")
            else:
                expected_audio_events = [
                    {
                        "event_id": event.event_id,
                        "actor_id": active_actor,
                        "event_type": "vocalization",
                        "start_sample": event.start_sample,
                        "end_sample": event.end_sample_exclusive,
                        "emitter_bone": upstream_source["emitter_link"],
                        "emitter_path_sha256": upstream_source[
                            "emitter_path_sha256"
                        ],
                        "audio_asset_sha256": upstream_event[
                            "dry_audio_asset_sha256"
                        ],
                        "semantic_sync_required": True,
                    }
                    for event in compiled.events
                ]
                if timeline["audio_events"] != expected_audio_events:
                    timeline_errors.append(
                        "timeline event identity/bounds/emitter/audio binding differs"
                    )
            for frame in timeline["frames"]:
                current = compiled.current_event_by_source(frame["frame_index"])
                expected_active = current[active_endpoint] is not None
                for state in frame["actor_states"]:
                    expected = expected_active if state["actor_id"] == active_actor else False
                    if state["mouth_state"]["vocalizing"] is not expected:
                        timeline_errors.append(
                            f"frame {frame['frame_index']} mouth semantic state differs"
                        )
                        break
    except (KeyError, OSError, StopIteration, TypeError, ValueError) as error:
        timeline_errors.append(str(error))
    add("timeline_program_alignment", not timeline_errors, timeline_errors)

    media_errors: list[str] = []
    try:
        primary = probe_episode_video(root / "visual" / "primary_view.mp4", ffprobe=ffprobe)
        topdown = probe_qa_review_video(
            root / "visual" / "optional_debug_views" / "topdown_review.mp4",
            ffprobe=ffprobe,
        )
        if primary["audio"]["channel_count"] != 2 or topdown["audio"]["channel_count"] != 2:
            media_errors.append("review video audio is not stereo binaural delivery")
        independent_aac = aac_decode_diagnostics(
            root / "visual" / "primary_view.mp4",
            root / "audio" / "mixture.wav",
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
        )
        if (
            independent_aac.get("presentation_sample_count_matches") is not True
            or independent_aac.get("lr_swap_suspected") is not False
            or float(independent_aac.get("minimum_correlation", -1.0)) < 0.98
            or float(independent_aac.get("minimum_snr_db", -999.0)) < 18.0
        ):
            media_errors.append("independent AAC readback failed correlation/SNR/LR gates")
        stored_mux_check = next(
            item
            for item in runtime_report.get("checks", [])
            if item.get("check_id") == "video_mux_packet_copy"
        )
        for delivery_id, delivery_path, upstream_filename in (
            (
                "primary",
                root / "visual" / "primary_view.mp4",
                "view0_base_video_only.mp4",
            ),
            (
                "topdown",
                root / "visual" / "optional_debug_views" / "topdown_review.mp4",
                "view0_topdown_base_video_only.mp4",
            ),
        ):
            declared_packets = stored_mux_check.get("measured", {}).get(
                delivery_id, {}
            ).get("video_packet_hash")
            observed_packets = video_packet_sha256(delivery_path, ffprobe=ffprobe)
            upstream_packets = video_packet_sha256(
                root / "provenance" / f"upstream_{upstream_filename}",
                ffprobe=ffprobe,
            )
            if declared_packets != observed_packets or observed_packets != upstream_packets:
                media_errors.append(
                    f"{delivery_id} video packet stream differs from retained M5/runtime report"
                )
    except (OSError, StopIteration, ValueError) as error:
        media_errors.append(str(error))
    add("media_readback", not media_errors, media_errors)

    flag_errors: list[str] = []
    observed_statuses: list[str] = []
    try:
        flags = load_json(root / "flags" / "source_event_flag_report.json")
        expected_flags = _build_flag_report(request=request, trajectory=trajectory)
        if flags != expected_flags:
            flag_errors.append("legacy flag report differs from trajectory recomputation")
        observed_statuses = [
            assessment["status"]
            for source in flags["source_flags"].values()
            for assessment in source.values()
        ]
        if (
            flags.get("definition_revision") != "m5_1_v1"
            or "not_evaluated" not in observed_statuses
            or not all(
                item in {"present", "absent", "not_evaluated"}
                for item in observed_statuses
            )
        ):
            flag_errors.append("legacy flag tri-state semantics differ")
    except (OSError, ValueError, KeyError, TypeError) as error:
        flag_errors.append(str(error))
    add(
        "legacy_flag_semantics",
        not flag_errors,
        {
            "errors": flag_errors,
            "observed_statuses": sorted(set(observed_statuses)),
        },
    )

    qa_absent = True
    offending: list[str] = []
    for json_path in root.rglob("*.json"):
        try:
            value = load_json(json_path)
        except ValueError:
            continue
        if not _json_no_qa_pairs(value):
            qa_absent = False
            offending.append(json_path.relative_to(root).as_posix())
    add("no_natural_language_qa_pairs", qa_absent, offending)

    try:
        final = load_json(root / "final_status.json")
        claims_ok = (
            evidence.get("research_only") is True
            and evidence.get("qualification_claim") is False
            and evidence.get("dataset_admission") is False
            and final.get("qualified_room_revision_created") is False
            and final.get("dataset_admission") is False
            and final.get("spatial_audio", {}).get("binaural_360_degree") is True
            and final.get("spatial_audio", {}).get("foa_authority_retained") is True
        )
        add("honest_research_only_claim", claims_ok, final)
    except (OSError, ValueError) as error:
        add("honest_research_only_claim", False, str(error))

    status = "pass" if all(item["status"] == "pass" for item in checks) else "fail"
    return status, checks


__all__ = [
    "M6CanaryError",
    "bind_controlled_canary_request_hash",
    "load_controlled_canary_request",
    "run_controlled_canary",
    "validate_controlled_canary_request",
    "verify_controlled_canary_evidence",
]
