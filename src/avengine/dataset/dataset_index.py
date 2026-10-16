"""Deterministic, episode-isolated indexing for the Apartment training closure."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from typing import Any, Mapping, Sequence


class ApartmentDatasetIndexError(ValueError):
    """The 1,000-item Apartment index cannot preserve its split contract."""


def _stable_digest(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode("utf-8")).hexdigest()


def _episode_row(value: Mapping[str, Any]) -> tuple[str, tuple[str, str, str]]:
    episode_id = value.get("episode_id")
    motion_case = value.get("motion_case")
    assets = value.get("asset_ids_by_source_slot")
    if (
        not isinstance(episode_id, str)
        or not episode_id
        or not isinstance(motion_case, str)
        or not motion_case
        or not isinstance(assets, Mapping)
        or not isinstance(assets.get("source1"), str)
        or not assets["source1"]
        or not isinstance(assets.get("source2"), str)
        or not assets["source2"]
    ):
        raise ApartmentDatasetIndexError("episode split input is incomplete")
    return episode_id, (str(assets["source1"]), str(assets["source2"]), motion_case)


def _largest_remainder_quotas(
    *,
    target_count: int,
    capacities: Mapping[tuple[str, str, str], int],
    total_capacity: int,
    seed: str,
    owner: str,
) -> dict[tuple[str, str, str], int]:
    if target_count < 0 or target_count > total_capacity:
        raise ApartmentDatasetIndexError(f"{owner} target is outside capacity")
    exact = {
        key: target_count * capacity / total_capacity
        for key, capacity in capacities.items()
    }
    quotas = {key: int(math.floor(value)) for key, value in exact.items()}
    remaining = target_count - sum(quotas.values())
    order = sorted(
        capacities,
        key=lambda key: (
            -(exact[key] - quotas[key]),
            _stable_digest(seed, owner, *key),
        ),
    )
    while remaining:
        changed = False
        for key in order:
            if quotas[key] >= capacities[key]:
                continue
            quotas[key] += 1
            remaining -= 1
            changed = True
            if not remaining:
                break
        if not changed:
            raise ApartmentDatasetIndexError(f"{owner} quotas cannot be completed")
    return quotas


def _balanced_validation_quotas(
    *,
    target_count: int,
    capacities: Mapping[tuple[str, str, str], int],
    total_capacity: int,
    seed: str,
) -> dict[tuple[str, str, str], int]:
    """Allocate validation residuals while covering motions and asset pairs."""

    exact = {
        key: target_count * capacity / total_capacity
        for key, capacity in capacities.items()
    }
    quotas = {key: int(math.floor(value)) for key, value in exact.items()}
    remaining = target_count - sum(quotas.values())
    while remaining:
        motion_counts: dict[str, int] = defaultdict(int)
        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        for (source1, source2, motion), count in quotas.items():
            motion_counts[motion] += count
            pair_counts[(source1, source2)] += count
        candidates = [
            key for key in capacities if quotas[key] < capacities[key]
        ]
        if not candidates:
            raise ApartmentDatasetIndexError(
                "validation quotas cannot be completed"
            )
        selected = min(
            candidates,
            key=lambda key: (
                motion_counts[key[2]],
                pair_counts[(key[0], key[1])],
                -(exact[key] - quotas[key]),
                _stable_digest(seed, "validation", *key),
            ),
        )
        quotas[selected] += 1
        remaining -= 1
    return quotas


def assign_episode_splits(
    episodes: Sequence[Mapping[str, Any]],
    *,
    train_count: int = 80,
    validation_count: int = 10,
    test_count: int = 10,
    seed: str = "avengine-apartment-split-v1",
) -> dict[str, str]:
    """Assign one split per visual trajectory, stratified by assets and motion.

    All dry-audio variants of an episode inherit this assignment.  The
    two-stage largest-remainder allocation gives exact global counts while
    keeping ordered source assets and motion cases as balanced as the finite
    episode bank permits.
    """

    if not isinstance(seed, str) or not seed:
        raise ApartmentDatasetIndexError("split seed must be a non-empty string")
    grouped: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    seen: set[str] = set()
    for value in episodes:
        if not isinstance(value, Mapping):
            raise ApartmentDatasetIndexError("episode split input is invalid")
        episode_id, stratum = _episode_row(value)
        if episode_id in seen:
            raise ApartmentDatasetIndexError("episode IDs are repeated")
        seen.add(episode_id)
        grouped[stratum].append(episode_id)
    total = len(seen)
    if total == 0 or train_count + validation_count + test_count != total:
        raise ApartmentDatasetIndexError("split counts do not close over episodes")

    capacities = {key: len(values) for key, values in grouped.items()}
    train_quotas = _largest_remainder_quotas(
        target_count=train_count,
        capacities=capacities,
        total_capacity=total,
        seed=seed,
        owner="train",
    )
    residual = {
        key: capacities[key] - train_quotas[key] for key in capacities
    }
    validation_quotas = _balanced_validation_quotas(
        target_count=validation_count,
        capacities=residual,
        total_capacity=validation_count + test_count,
        seed=seed,
    )

    result: dict[str, str] = {}
    for stratum, episode_ids in grouped.items():
        ranked = sorted(
            episode_ids,
            key=lambda episode_id: _stable_digest(seed, "episode", episode_id),
        )
        train_end = train_quotas[stratum]
        validation_end = train_end + validation_quotas[stratum]
        for episode_id in ranked[:train_end]:
            result[episode_id] = "train"
        for episode_id in ranked[train_end:validation_end]:
            result[episode_id] = "validation"
        for episode_id in ranked[validation_end:]:
            result[episode_id] = "test"
    counts = {
        split: sum(value == split for value in result.values())
        for split in ("train", "validation", "test")
    }
    if counts != {
        "train": train_count,
        "validation": validation_count,
        "test": test_count,
    }:
        raise ApartmentDatasetIndexError("episode split counts changed")
    return result


def summarize_split_distribution(
    episodes: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, str],
) -> dict[str, Any]:
    """Return exact split, motion and ordered-asset-pair counts."""

    summary: dict[str, Any] = {
        split: {
            "episode_count": 0,
            "motion_case_counts": {},
            "ordered_asset_pair_counts": {},
        }
        for split in ("train", "validation", "test")
    }
    for value in episodes:
        episode_id, stratum = _episode_row(value)
        split = assignments.get(episode_id)
        if split not in summary:
            raise ApartmentDatasetIndexError("split assignment is incomplete")
        source1, source2, motion_case = stratum
        record = summary[split]
        record["episode_count"] += 1
        record["motion_case_counts"][motion_case] = (
            record["motion_case_counts"].get(motion_case, 0) + 1
        )
        pair = f"{source1} -> {source2}"
        record["ordered_asset_pair_counts"][pair] = (
            record["ordered_asset_pair_counts"].get(pair, 0) + 1
        )
    return summary


__all__ = [
    "ApartmentDatasetIndexError",
    "assign_episode_splits",
    "summarize_split_distribution",
]
