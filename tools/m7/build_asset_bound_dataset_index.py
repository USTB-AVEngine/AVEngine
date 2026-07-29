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

from avengine.contracts.json_io import load_json, sha256_file, write_json
from avengine.m7.dataset_index import (
    ApartmentDatasetIndexError,
    assign_episode_splits,
    summarize_split_distribution,
)


SCHEMA = "avengine_m7_apartment_training_index_v1"
SPLIT_SEED = "avengine-apartment-split-v1"
SPLIT_SAMPLE_COUNTS = {"train": 800, "validation": 100, "test": 100}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_AUDIO_PROGRAM_SAMPLE_FIELDS = frozenset(
    {
        "audio_program_binding",
        "audio_program_instance_path",
        "audio_program_instance_sha256",
    }
)


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
        if not isinstance(value, Mapping):
            raise ApartmentDatasetIndexError("visual episode record is invalid")
        result.append(
            {
                "episode_id": value.get("episode_id"),
                "motion_case": value.get("motion_case"),
                "asset_ids_by_source_slot": value.get("asset_ids_by_source_slot"),
            }
        )
    return result


def _render_evidence(render_root: Path) -> dict[str, Mapping[str, Any]]:
    evidence = load_json(render_root / "evidence.json")
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

    episodes = _visual_episodes(visual_bundle_root)
    episode_ids = {value["episode_id"] for value in episodes}
    if len(episode_ids) != len(episodes) or not all(
        isinstance(value, str) for value in episode_ids
    ):
        raise ApartmentDatasetIndexError("visual episode IDs are invalid")
    render_evidence = _render_evidence(ue_render_root)
    if set(render_evidence) != episode_ids:
        raise ApartmentDatasetIndexError("UE render closure differs from visual bank")

    samples_record = load_json(audio_batch_root / "samples.json")
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
    program_states: list[bool] = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise ApartmentDatasetIndexError("audio sample is invalid")
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
    delivery = load_json(audio_batch_root / "delivery.json")
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
                "room_id": "apartment_0000",
                "sample_count": 1000,
                "visual_episode_count": len(episodes),
                "audio_variants_per_visual_episode": variants_per_episode,
                "split_unit": "visual_episode",
                "split_seed": SPLIT_SEED,
                "split_sample_counts": sample_split_counts,
                "scene_copy_count": 0,
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
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_index(
        audio_batch_root=args.audio_batch_root,
        visual_bundle_root=args.visual_bundle_root,
        ue_render_root=args.ue_render_root,
        output=args.output,
    )
    print(f"ASSET_BOUND_DATASET_INDEX_OK output={result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
