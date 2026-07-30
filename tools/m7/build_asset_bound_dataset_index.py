#!/usr/bin/env python3
"""Index 1,000 samples without copying visual, audio, or room media."""

from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence

from avengine.contracts.json_io import (
    canonical_json_sha256,
    load_json,
    sha256_file,
    write_json,
)
from avengine.m7.dataset_index import (
    ApartmentDatasetIndexError,
    assign_episode_splits,
    summarize_split_distribution,
)


SCHEMA = "avengine_m7_apartment_training_index_v1"
SPLIT_SEED = "avengine-apartment-split-v1"
SPLIT_SAMPLE_COUNTS = {"train": 800, "validation": 100, "test": 100}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ACOUSTIC_SELECTION_FIELDS = {
    "schema",
    "selection_mode",
    "registry_selection_applied",
    "room_ref",
    "profile_ref",
    "binding_id",
    "registry_selection_content_sha256",
    "effective_selection_content_sha256",
    "acoustic_package_manifest_sha256",
    "simulation_request_sha256",
    "input_receipt_sha256",
    "binding_content_sha256",
}
_REGISTRY_SELECTION_MODES = {
    "registry",
    "registry_with_verified_equivalent_overrides",
}
_LEGACY_SELECTION_MODES = {
    "explicit_legacy",
    "explicit_legacy_unbound",
}
_SPEAR_RUNTIME_EVIDENCE_SCHEMA = (
    "avengine_optional_spear_apartment_runtime_evidence_v2"
)
_SPEAR_RUNTIME_IDENTITY_SCHEMA = (
    "avengine_spear_acoustic_visual_runtime_identity_v1"
)
_SPEAR_RUNTIME_IDENTITY_FIELDS = {
    "schema",
    "status",
    "verification_status",
    "selection_mode",
    "compatibility",
    "acoustic_selection_binding_sha256",
    "binding_id",
    "profile_ref",
    "visual_room_ref",
    "acoustic_room_ref",
    "runtime_room_ref",
    "runtime_profile_id",
    "runtime_map_id",
    "runtime_map_path",
}
_AUDIO_PROGRAM_SAMPLE_FIELDS = frozenset(
    {
        "audio_program_binding",
        "audio_program_instance_path",
        "audio_program_instance_sha256",
    }
)


def _validated_acoustic_selection_binding(
    value: Any,
) -> tuple[dict[str, Any], str | None]:
    if (
        not isinstance(value, Mapping)
        or value.get("schema")
        != "avengine_rir_cache_acoustic_selection_binding_v1"
        or set(value) != _ACOUSTIC_SELECTION_FIELDS
    ):
        raise ApartmentDatasetIndexError(
            "acoustic_selection_binding is invalid"
        )
    binding = deepcopy(dict(value))
    mode = binding.get("selection_mode")
    binding_sha256 = binding.get("binding_content_sha256")
    if mode == "explicit_legacy_unbound":
        if (
            binding_sha256 is not None
            or binding.get("registry_selection_applied") is not False
            or binding.get("room_ref") is not None
            or binding.get("profile_ref") is not None
            or binding.get("binding_id") is not None
        ):
            raise ApartmentDatasetIndexError(
                "legacy unbound input fabricated an acoustic identity"
            )
        return binding, None
    if (
        mode
        not in {
            "explicit_legacy",
            "registry",
            "registry_with_verified_equivalent_overrides",
        }
        or not isinstance(binding_sha256, str)
        or _SHA256_RE.fullmatch(binding_sha256) is None
        or canonical_json_sha256(
            {
                key: item
                for key, item in binding.items()
                if key != "binding_content_sha256"
            }
        )
        != binding_sha256
    ):
        raise ApartmentDatasetIndexError(
            "acoustic_selection_binding hash is invalid"
        )
    if mode == "explicit_legacy":
        if (
            binding.get("registry_selection_applied") is not False
            or binding.get("room_ref") is not None
            or binding.get("profile_ref") is not None
            or binding.get("binding_id") is not None
        ):
            raise ApartmentDatasetIndexError(
                "explicit legacy input contains a registry identity"
            )
    elif (
        binding.get("registry_selection_applied") is not True
        or not isinstance(binding.get("room_ref"), Mapping)
        or set(binding["room_ref"])
        != {"registry_id", "room_id", "revision"}
        or not isinstance(binding.get("profile_ref"), Mapping)
        or set(binding["profile_ref"]) != {"profile_id", "revision"}
        or not isinstance(binding.get("binding_id"), str)
        or not binding["binding_id"]
    ):
        raise ApartmentDatasetIndexError(
            "registry input lacks its exact room/profile binding"
        )
    return binding, binding_sha256


def _visual_episodes(bundle_root: Path) -> list[dict[str, Any]]:
    manifest = load_json(bundle_root / "manifest.json")
    values = manifest.get("episodes")
    if (
        manifest.get("status") != "pass"
        or not isinstance(values, list)
        or not values
        or manifest.get("episode_count") != len(values)
    ):
        raise ApartmentDatasetIndexError("visual input bundle is invalid")
    result = []
    for value in values:
        if (
            not isinstance(value, Mapping)
            or "acoustic_selection_binding_sha256" not in value
        ):
            raise ApartmentDatasetIndexError("visual episode record is invalid")
        result.append(
            {
                "episode_id": value.get("episode_id"),
                "motion_case": value.get("motion_case"),
                "asset_ids_by_source_slot": value.get("asset_ids_by_source_slot"),
                "acoustic_selection_binding_sha256": value.get(
                    "acoustic_selection_binding_sha256"
                ),
            }
        )
    return result


def _render_evidence(
    render_root: Path,
    *,
    evidence_path: Path | None = None,
) -> dict[str, Mapping[str, Any]]:
    evidence = load_json(
        evidence_path
        if evidence_path is not None
        else render_root / "evidence.json"
    )
    values = evidence.get("scenarios")
    if evidence.get("status") != "pass" or not isinstance(values, list):
        raise ApartmentDatasetIndexError("UE visual evidence did not pass")
    result: dict[str, Mapping[str, Any]] = {}
    for value in values:
        if not isinstance(value, Mapping) or value.get("status") != "pass":
            raise ApartmentDatasetIndexError("UE scenario evidence did not pass")
        episode_id = value.get("scenario_id")
        if not isinstance(episode_id, str) or episode_id in result:
            raise ApartmentDatasetIndexError("UE scenario IDs are invalid")
        result[episode_id] = value
    return result


def _is_exact_room_ref(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {"registry_id", "room_id", "revision"}
        and all(
            isinstance(value.get(field), str) and bool(value[field])
            for field in ("registry_id", "room_id", "revision")
        )
    )


def _validated_spear_runtime_evidence(
    *,
    evidence_path: Path | None,
    acoustic_selection_binding: Mapping[str, Any],
    acoustic_selection_binding_sha256: str | None,
    visual_room_alignment: Mapping[str, Any],
    episode_ids: set[str],
) -> dict[str, Any]:
    """Bind SPEAR/UE readback to the exact audio and visual room identity."""

    mode = acoustic_selection_binding.get("selection_mode")
    if evidence_path is None:
        if mode in _REGISTRY_SELECTION_MODES:
            raise ApartmentDatasetIndexError(
                "registry-bound index requires SPEAR/UE runtime evidence"
            )
        if mode not in _LEGACY_SELECTION_MODES:
            raise ApartmentDatasetIndexError(
                "unsupported acoustic selection mode for runtime evidence"
            )
        return {
            "status": "not_verified",
            "verification_status": "not_verified",
            "path": None,
            "sha256": None,
            "schema": None,
            "acoustic_visual_identity": None,
        }

    evidence_path = evidence_path.resolve()
    if not evidence_path.is_file():
        raise ApartmentDatasetIndexError(
            "SPEAR/UE runtime evidence file is missing"
        )
    evidence = load_json(evidence_path)
    scenarios = evidence.get("scenarios")
    if (
        evidence.get("schema") != _SPEAR_RUNTIME_EVIDENCE_SCHEMA
        or evidence.get("status") != "pass"
        or not isinstance(scenarios, list)
    ):
        raise ApartmentDatasetIndexError(
            "SPEAR/UE runtime evidence is invalid"
        )
    scenario_ids = [
        value.get("scenario_id")
        for value in scenarios
        if isinstance(value, Mapping)
    ]
    if (
        len(scenario_ids) != len(scenarios)
        or not all(
            isinstance(scenario_id, str) and bool(scenario_id)
            for scenario_id in scenario_ids
        )
        or len(set(scenario_ids)) != len(scenario_ids)
        or set(scenario_ids) != episode_ids
    ):
        raise ApartmentDatasetIndexError(
            "SPEAR/UE runtime evidence closure differs from the visual bank"
        )

    identity = evidence.get("acoustic_visual_identity")
    if (
        not isinstance(identity, Mapping)
        or identity.get("schema") != _SPEAR_RUNTIME_IDENTITY_SCHEMA
        or set(identity) != _SPEAR_RUNTIME_IDENTITY_FIELDS
    ):
        raise ApartmentDatasetIndexError(
            "SPEAR/UE acoustic_visual_identity is invalid"
        )
    identity = deepcopy(dict(identity))
    if any(
        not isinstance(value, Mapping)
        or value.get("acoustic_visual_identity") != identity
        for value in scenarios
    ):
        raise ApartmentDatasetIndexError(
            "SPEAR/UE scenario runtime identities differ from the batch"
        )

    visual_room_ref = visual_room_alignment.get("visual_room_ref")
    acoustic_room_ref = acoustic_selection_binding.get("room_ref")
    if (
        not _is_exact_room_ref(visual_room_ref)
        or not _is_exact_room_ref(identity.get("visual_room_ref"))
        or not _is_exact_room_ref(identity.get("runtime_room_ref"))
        or identity.get("selection_mode") != mode
        or identity.get("acoustic_selection_binding_sha256")
        != acoustic_selection_binding_sha256
        or identity.get("binding_id")
        != acoustic_selection_binding.get("binding_id")
        or identity.get("profile_ref")
        != acoustic_selection_binding.get("profile_ref")
        or identity.get("visual_room_ref") != visual_room_ref
        or identity.get("runtime_room_ref") != visual_room_ref
    ):
        raise ApartmentDatasetIndexError(
            "SPEAR/UE runtime identity differs from the audio/visual binding"
        )
    runtime_profile = evidence.get("room_runtime_profile")
    if (
        not isinstance(identity.get("runtime_profile_id"), str)
        or not identity["runtime_profile_id"]
        or not isinstance(identity.get("runtime_map_id"), str)
        or not identity["runtime_map_id"]
        or not isinstance(identity.get("runtime_map_path"), str)
        or not identity["runtime_map_path"]
        or not isinstance(runtime_profile, Mapping)
        or runtime_profile.get("profile_id")
        != identity["runtime_profile_id"]
        or evidence.get("native_map") != identity["runtime_map_path"]
    ):
        raise ApartmentDatasetIndexError(
            "SPEAR/UE runtime map/profile identity is invalid"
        )

    if mode in _REGISTRY_SELECTION_MODES:
        if (
            identity.get("status") != "pass"
            or identity.get("verification_status") != "verified"
            or identity.get("compatibility") is not None
            or not _is_exact_room_ref(acoustic_room_ref)
            or identity.get("acoustic_room_ref") != acoustic_room_ref
            or identity.get("runtime_room_ref") != acoustic_room_ref
        ):
            raise ApartmentDatasetIndexError(
                "registry-bound SPEAR/UE runtime identity was not verified"
            )
    elif mode in _LEGACY_SELECTION_MODES:
        if (
            identity.get("status") != "not_verified"
            or identity.get("verification_status") != "not_verified"
            or identity.get("compatibility")
            != "legacy_acoustic_selection_without_room_ref"
            or acoustic_room_ref is not None
            or identity.get("acoustic_room_ref") is not None
        ):
            raise ApartmentDatasetIndexError(
                "legacy SPEAR/UE runtime identity must remain not_verified"
            )
    else:
        raise ApartmentDatasetIndexError(
            "unsupported acoustic selection mode for runtime evidence"
        )

    return {
        "status": identity["status"],
        "verification_status": identity["verification_status"],
        "path": str(evidence_path),
        "sha256": sha256_file(evidence_path),
        "schema": evidence["schema"],
        "acoustic_visual_identity": identity,
    }


def _runtime_evidence_row_identity(
    runtime_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "status": runtime_evidence["status"],
        "verification_status": runtime_evidence["verification_status"],
        "evidence_schema": runtime_evidence["schema"],
        "evidence_sha256": runtime_evidence["sha256"],
        "acoustic_visual_identity": deepcopy(
            runtime_evidence["acoustic_visual_identity"]
        ),
    }


def _audio_program_index_fields(
    sample: Mapping[str, Any],
    *,
    audio_batch_root: Path,
) -> tuple[dict[str, Any], str | None]:
    """Return optional verified AudioProgram fields and its label path."""

    present = _AUDIO_PROGRAM_SAMPLE_FIELDS.intersection(sample)
    if not present:
        return {}, None
    if present != _AUDIO_PROGRAM_SAMPLE_FIELDS:
        missing = sorted(_AUDIO_PROGRAM_SAMPLE_FIELDS - present)
        raise ApartmentDatasetIndexError(
            f"audio sample has an incomplete AudioProgram binding; missing={missing}"
        )
    binding = sample.get("audio_program_binding")
    raw_path = sample.get("audio_program_instance_path")
    declared_sha256 = sample.get("audio_program_instance_sha256")
    if (
        not isinstance(binding, Mapping)
        or not isinstance(raw_path, str)
        or not raw_path
        or Path(raw_path).is_absolute()
        or not isinstance(declared_sha256, str)
        or _SHA256_RE.fullmatch(declared_sha256) is None
    ):
        raise ApartmentDatasetIndexError(
            "audio sample AudioProgram index fields are invalid"
        )
    root = audio_batch_root.resolve()
    resolved = (root / raw_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ApartmentDatasetIndexError(
            "audio_program_instance_path escapes the audio batch"
        ) from exc
    if not resolved.is_file() or sha256_file(resolved) != declared_sha256:
        raise ApartmentDatasetIndexError(
            "audio sample AudioProgram instance file or hash differs"
        )
    instance = load_json(resolved)
    program = instance.get("materialized_audio_program")
    if (
        instance.get("schema") != "avengine_m7_m6_audio_program_instance_v1"
        or instance.get("status") != "pass"
        or instance.get("audio_program_binding") != binding
        or not isinstance(program, Mapping)
        or program.get("program_content_sha256")
        != binding.get("materialized_program_content_sha256")
    ):
        raise ApartmentDatasetIndexError(
            "audio sample AudioProgram instance content differs"
        )
    return (
        {
            "audio_program_binding": deepcopy(dict(binding)),
            "audio_program_instance_path": raw_path,
            "audio_program_instance_sha256": declared_sha256,
        },
        raw_path,
    )


def build_index(
    *,
    audio_batch_root: Path,
    visual_bundle_root: Path,
    ue_render_root: Path,
    output: Path,
    spear_runtime_evidence: Path | None = None,
) -> Path:
    audio_batch_root = audio_batch_root.resolve()
    visual_bundle_root = visual_bundle_root.resolve()
    ue_render_root = ue_render_root.resolve()
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace output: {output}")
    staging = output.with_name(f".{output.name}.staging.{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"refusing to replace staging output: {staging}")

    visual_manifest = load_json(visual_bundle_root / "manifest.json")
    episodes = _visual_episodes(visual_bundle_root)
    episode_ids = {value["episode_id"] for value in episodes}
    if len(episode_ids) != len(episodes) or not all(
        isinstance(value, str) for value in episode_ids
    ):
        raise ApartmentDatasetIndexError("visual episode IDs are invalid")
    samples_record = load_json(audio_batch_root / "samples.json")
    audio_episodes_record = load_json(audio_batch_root / "episodes.json")
    delivery = load_json(audio_batch_root / "delivery.json")
    samples = samples_record.get("samples")
    if (
        samples_record.get("status") != "pass"
        or samples_record.get("sample_count") != 1000
        or not isinstance(samples, list)
        or len(samples) != 1000
    ):
        raise ApartmentDatasetIndexError("audio batch is not the 1,000-item closure")
    verification = load_json(audio_batch_root / "verification.json")
    if verification.get("status") != "pass":
        raise ApartmentDatasetIndexError("audio batch verification did not pass")
    (
        acoustic_selection_binding,
        acoustic_selection_binding_sha256,
    ) = _validated_acoustic_selection_binding(
        delivery.get("acoustic_selection_binding")
    )
    if (
        samples_record.get("acoustic_selection_binding")
        != acoustic_selection_binding
        or audio_episodes_record.get("acoustic_selection_binding")
        != acoustic_selection_binding
        or verification.get("acoustic_selection_binding")
        != acoustic_selection_binding
        or visual_manifest.get("acoustic_selection_binding")
        != acoustic_selection_binding
    ):
        raise ApartmentDatasetIndexError(
            "audio and visual acoustic selection bindings differ"
        )
    visual_room_alignment = visual_manifest.get(
        "acoustic_visual_room_alignment"
    )
    if not isinstance(visual_room_alignment, Mapping):
        raise ApartmentDatasetIndexError(
            "visual bundle lacks acoustic/visual room alignment"
        )
    acoustic_room_ref = acoustic_selection_binding.get("room_ref")
    if acoustic_room_ref is None:
        if (
            visual_room_alignment.get("status") != "not_verified"
            or visual_room_alignment.get("acoustic_room_ref") is not None
        ):
            raise ApartmentDatasetIndexError(
                "legacy unbound acoustic/visual room alignment is invalid"
            )
    elif (
        visual_room_alignment.get("status") != "pass"
        or visual_room_alignment.get("acoustic_room_ref")
        != acoustic_room_ref
        or visual_room_alignment.get("visual_room_ref")
        != acoustic_room_ref
    ):
        raise ApartmentDatasetIndexError(
            "visual room_ref differs from the acoustic selection"
        )
    runtime_evidence = _validated_spear_runtime_evidence(
        evidence_path=spear_runtime_evidence,
        acoustic_selection_binding=acoustic_selection_binding,
        acoustic_selection_binding_sha256=(
            acoustic_selection_binding_sha256
        ),
        visual_room_alignment=visual_room_alignment,
        episode_ids=episode_ids,
    )
    render_evidence = _render_evidence(
        ue_render_root,
        evidence_path=(
            spear_runtime_evidence.resolve()
            if spear_runtime_evidence is not None
            else None
        ),
    )
    if set(render_evidence) != episode_ids:
        raise ApartmentDatasetIndexError("UE render closure differs from visual bank")
    runtime_row_identity = _runtime_evidence_row_identity(runtime_evidence)
    runtime_identity = runtime_evidence["acoustic_visual_identity"]
    runtime_map_id = (
        runtime_identity["runtime_map_id"]
        if isinstance(runtime_identity, Mapping)
        else None
    )
    program_states: list[bool] = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise ApartmentDatasetIndexError("audio sample is invalid")
        if (
            "acoustic_selection_binding_sha256" not in sample
            or sample.get("acoustic_selection_binding_sha256")
            != acoustic_selection_binding_sha256
        ):
            raise ApartmentDatasetIndexError(
                "audio sample acoustic selection differs from its batch"
            )
        present = _AUDIO_PROGRAM_SAMPLE_FIELDS.intersection(sample)
        if present and present != _AUDIO_PROGRAM_SAMPLE_FIELDS:
            raise ApartmentDatasetIndexError(
                "audio sample has an incomplete AudioProgram binding"
            )
        program_states.append(bool(present))
    has_audio_program_samples = any(program_states)
    if has_audio_program_samples and not all(program_states):
        raise ApartmentDatasetIndexError(
            "legacy and AudioProgram samples may not be mixed"
        )
    variants_per_episode = delivery.get("variants_per_episode")
    if (
        delivery.get("status") != "pass"
        or isinstance(variants_per_episode, bool)
        or not isinstance(variants_per_episode, int)
        or variants_per_episode < 1
        or delivery.get("episode_count") != len(episodes)
        or len(episodes) * variants_per_episode != 1_000
        or (
            has_audio_program_samples
            and variants_per_episode != 1
        )
        or any(
            count % variants_per_episode
            for count in SPLIT_SAMPLE_COUNTS.values()
        )
    ):
        raise ApartmentDatasetIndexError(
            "audio episode/variant layout cannot form the 1,000-item split"
        )
    episode_split_counts = {
        split: count // variants_per_episode
        for split, count in SPLIT_SAMPLE_COUNTS.items()
    }
    assignments = assign_episode_splits(
        episodes,
        train_count=episode_split_counts["train"],
        validation_count=episode_split_counts["validation"],
        test_count=episode_split_counts["test"],
        seed=SPLIT_SEED,
    )
    by_episode: dict[str, list[Mapping[str, Any]]] = {
        episode_id: [] for episode_id in episode_ids
    }
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise ApartmentDatasetIndexError("audio sample is invalid")
        episode_id = sample.get("episode_id")
        if episode_id not in by_episode:
            raise ApartmentDatasetIndexError("audio sample has no visual episode")
        by_episode[str(episode_id)].append(sample)
    if any(
        sorted(value.get("variant_index") for value in rows)
        != list(range(variants_per_episode))
        for rows in by_episode.values()
    ):
        raise ApartmentDatasetIndexError(
            "each visual episode lacks its declared audio variants"
        )

    staging.mkdir(parents=True)
    try:
        rows = []
        visual_rows = []
        for episode in sorted(episodes, key=lambda value: str(value["episode_id"])):
            episode_id = str(episode["episode_id"])
            if (
                episode["acoustic_selection_binding_sha256"]
                != acoustic_selection_binding_sha256
            ):
                raise ApartmentDatasetIndexError(
                    "visual episode acoustic selection differs from its bundle"
                )
            split = assignments[episode_id]
            visual_assets = episode["asset_ids_by_source_slot"]
            scenario_evidence = render_evidence[episode_id]
            media = scenario_evidence.get("media")
            if not isinstance(media, Mapping):
                raise ApartmentDatasetIndexError("UE media evidence is missing")
            required_media = {
                "rgb": "ue_visual_only.mp4",
                "topdown": "ue_topdown_visual_only.mp4",
            }
            media_paths = {}
            for role, filename in required_media.items():
                record = media.get(filename.removesuffix(".mp4"))
                path = ue_render_root / episode_id / filename
                if (
                    not isinstance(record, Mapping)
                    or record.get("status") != "pass"
                    or not path.is_file()
                ):
                    raise ApartmentDatasetIndexError(f"{episode_id} lacks {role} media")
                media_paths[role] = f"{episode_id}/{filename}"
            labels = {
                "timeline": f"episodes/{episode_id}/metadata/timeline.json",
                "source_manifest": (
                    f"episodes/{episode_id}/metadata/source_manifest.json"
                ),
                "flags": f"episodes/{episode_id}/metadata/flags.json",
            }
            if any(not (visual_bundle_root / value).is_file() for value in labels.values()):
                raise ApartmentDatasetIndexError(f"{episode_id} labels are incomplete")
            visual_rows.append(
                {
                    "episode_id": episode_id,
                    "split": split,
                    "motion_case": episode["motion_case"],
                    "asset_ids_by_source_slot": episode["asset_ids_by_source_slot"],
                    "rgb_path": media_paths["rgb"],
                    "topdown_path": media_paths["topdown"],
                    "label_paths": labels,
                    "audio_variant_reuse_count": variants_per_episode,
                    "acoustic_selection_binding_sha256": (
                        acoustic_selection_binding_sha256
                    ),
                    "runtime_map_id": runtime_map_id,
                    "spear_ue_runtime_evidence_identity": deepcopy(
                        runtime_row_identity
                    ),
                }
            )
            for sample in sorted(
                by_episode[episode_id],
                key=lambda value: int(value["variant_index"]),
            ):
                if sample.get("asset_ids_by_source_slot") != visual_assets:
                    raise ApartmentDatasetIndexError(
                        "visual and audio asset bindings differ"
                    )
                mixture = sample.get("audio", {}).get("mixture", {})
                audio_path = audio_batch_root / "audio" / "binaural" / str(
                    mixture.get("path")
                )
                if not audio_path.is_file():
                    raise ApartmentDatasetIndexError("indexed audio file is missing")
                audio_program_fields, audio_program_label = (
                    _audio_program_index_fields(
                        sample,
                        audio_batch_root=audio_batch_root,
                    )
                )
                sample_labels = dict(labels)
                label_path_roots = {
                    "timeline": "visual_bundle_root",
                    "source_manifest": "visual_bundle_root",
                    "flags": "visual_bundle_root",
                }
                if audio_program_label is not None:
                    sample_labels["audio_program_instance"] = audio_program_label
                    label_path_roots[
                        "audio_program_instance"
                    ] = "audio_batch_root"
                row = {
                    "sample_id": sample["sample_id"],
                    "split": split,
                    "episode_id": episode_id,
                    "variant_index": sample["variant_index"],
                    "motion_case": episode["motion_case"],
                    "asset_ids_by_source_slot": sample[
                        "asset_ids_by_source_slot"
                    ],
                    "both_sources_active": sample["both_sources_active"],
                    "audio_path": f"audio/binaural/{mixture['path']}",
                    "audio_sample_rate_hz": sample["audio"]["sample_rate_hz"],
                    "audio_channel_count": sample["audio"]["channel_count"],
                    "rgb_episode_path": media_paths["rgb"],
                    "topdown_episode_path": media_paths["topdown"],
                    "label_paths": sample_labels,
                    "acoustic_selection_binding_sha256": (
                        acoustic_selection_binding_sha256
                    ),
                    "runtime_map_id": runtime_map_id,
                    "spear_ue_runtime_evidence_identity": deepcopy(
                        runtime_row_identity
                    ),
                    **audio_program_fields,
                }
                if audio_program_label is not None:
                    row["label_path_roots"] = label_path_roots
                rows.append(row)

        sample_split_counts = {
            split: sum(value["split"] == split for value in rows)
            for split in SPLIT_SAMPLE_COUNTS
        }
        if sample_split_counts != SPLIT_SAMPLE_COUNTS:
            raise ApartmentDatasetIndexError("sample split counts changed")
        write_json(
            staging / "dataset_index.json",
            {
                "schema": SCHEMA,
                "status": "pass",
                "research_only": True,
                "qualification_claim": False,
                "room_ref": deepcopy(acoustic_room_ref),
                "runtime_map_id": runtime_map_id,
                "sample_count": 1000,
                "visual_episode_count": len(episodes),
                "audio_variants_per_visual_episode": variants_per_episode,
                "split_unit": "visual_episode",
                "split_seed": SPLIT_SEED,
                "split_sample_counts": sample_split_counts,
                "scene_copy_count": 0,
                "acoustic_selection_binding": acoustic_selection_binding,
                "acoustic_visual_room_alignment": deepcopy(
                    dict(visual_room_alignment)
                ),
                "spear_ue_runtime_evidence": deepcopy(runtime_evidence),
                "media_storage_policy": (
                    "one_rgb_and_topdown_pair_per_episode_plus_declared_binaural_variants"
                ),
                "roots": {
                    "audio_batch_root": str(audio_batch_root),
                    "visual_bundle_root": str(visual_bundle_root),
                    "ue_render_root": str(ue_render_root),
                },
                "samples": rows,
            },
        )
        write_json(
            staging / "split_report.json",
            {
                "schema": "avengine_m7_apartment_split_report_v1",
                "status": "pass",
                "split_unit": "visual_episode",
                "episode_distribution": summarize_split_distribution(
                    episodes, assignments
                ),
                "sample_split_counts": sample_split_counts,
                "acoustic_selection_binding": acoustic_selection_binding,
                "room_ref": deepcopy(acoustic_room_ref),
                "runtime_map_id": runtime_map_id,
                "spear_ue_runtime_evidence": deepcopy(runtime_evidence),
                "visual_episodes": visual_rows,
            },
        )
        os.rename(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-batch-root", type=Path, required=True)
    parser.add_argument("--visual-bundle-root", type=Path, required=True)
    parser.add_argument("--ue-render-root", type=Path, required=True)
    parser.add_argument(
        "--spear-runtime-evidence",
        type=Path,
        help=(
            "Final SPEAR/UE OUTPUT/evidence.json; required for registry-bound "
            "formal indexing."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_index(
        audio_batch_root=args.audio_batch_root,
        visual_bundle_root=args.visual_bundle_root,
        ue_render_root=args.ue_render_root,
        output=args.output,
        spear_runtime_evidence=args.spear_runtime_evidence,
    )
    print(f"ASSET_BOUND_DATASET_INDEX_OK output={result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
