#!/usr/bin/env python3
"""Register split sound-event clips into avengine_m6_sound_asset_registry_v1.

Every dry_audio field, semantic class, taxonomy path and origin is derived
from the event-pool catalog and the splitter manifest. Two policy fields
are not derived and have no code defaults: --permitted-event-usage and
--normalization-policy. Owner 2026-09-03 admitted the whole cut into
research; that sentence is stored in provenance so the grant has a name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from avengine.registry.registry import bind_content_hash  # noqa: E402
from avengine.registry.sources import (  # noqa: E402
    validate_sound_asset_registry,
)
from avengine.assets.sound_harvest import (  # noqa: E402
    speech_metadata_from_mapping,
)
from split_sound_library_events import category_for_class  # noqa: E402

POOL_SCHEMA = "avengine_sound_event_pool_v1"
MANIFEST_SCHEMA = "avengine_sound_event_library_v1"
REGISTRY_SCHEMA = "avengine_m6_sound_asset_registry_v1"
OWNER_ADMISSION_NOTE = "owner 2026-09-03 授权全部准入"
ADMISSIBILITY = "research"
ALLOWED_TRANSFORMS = ("crop", "gain", "zero_pad")
PERMITTED_USAGE = (
    "counterfactual_route_swap",
    "intermittent_events",
    "one_active_of_n",
    "sequential_sources",
    "simultaneous_subset",
)
NORMALIZATION_MODES = ("preserve", "peak_dbfs", "rms_dbfs")


class RegisterError(ValueError):
    """The catalog cannot become a sound-asset registry."""


def _canonical_unique(values: list[str], *, owner: str) -> list[str]:
    if len(values) != len(set(values)):
        raise RegisterError(f"{owner} repeats a value: {values}")
    return sorted(values)


def _wav_fields(path: Path) -> tuple[str, int, int, int]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        frames = handle.getnframes()
    if channels != 1:
        raise RegisterError(f"{path} channel_count={channels}, registry requires 1")
    if frames <= 0 or rate <= 0:
        raise RegisterError(f"{path} is empty")
    return digest, rate, channels, frames


def _path_from_file_uri(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise RegisterError(f"dry_audio.uri is not a file URI: {uri!r}")
    return Path(unquote(parsed.path))


def dry_audio_byte_errors(registry: Mapping[str, Any]) -> list[str]:
    """Hash each file:// dry_audio.uri and compare to the declared sha256."""

    errors: list[str] = []
    assets = registry.get("sound_assets")
    if not isinstance(assets, list):
        return ["sound_assets missing"]
    for index, asset in enumerate(assets):
        if not isinstance(asset, Mapping):
            continue
        dry = asset.get("dry_audio")
        if not isinstance(dry, Mapping):
            errors.append(f"sound_assets[{index}].dry_audio missing")
            continue
        uri = dry.get("uri")
        declared = dry.get("sha256")
        try:
            path = _path_from_file_uri(str(uri))
        except RegisterError as error:
            errors.append(f"sound_assets[{index}]: {error}")
            continue
        if not path.is_file():
            errors.append(f"sound_assets[{index}].dry_audio.uri missing file {path}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != declared:
            errors.append(
                f"sound_assets[{index}].dry_audio.sha256 does not match "
                f"{path} bytes"
            )
    return errors


def _manifest_by_asset_id(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(manifest.get("clips") or []):
        if row.get("status") != "event":
            continue
        asset_id = str(row.get("sound_asset_id") or "")
        if not asset_id:
            raise RegisterError(f"manifest clips[{index}] missing sound_asset_id")
        if asset_id in by_id:
            raise RegisterError(f"manifest repeats sound_asset_id {asset_id}")
        by_id[asset_id] = row
    return by_id


def _origin_for(row: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    splitter = manifest.get("splitter") or "tools/assets/split_sound_library_events.py"
    library_sha = manifest.get("source_library_sha256") or ""
    source = row.get("source") or ""
    source_sha = row.get("source_sha256") or ""
    return (
        f"source={source}; source_sha256={source_sha}; "
        f"splitter={splitter}; source_library_sha256={library_sha}; "
        f"{OWNER_ADMISSION_NOTE}"
    )


def register_sound_event_assets(
    catalog_path: Path,
    output_path: Path,
    *,
    permitted_event_usage: list[str],
    normalization_policy: str,
    registry_id: str,
    revision: str,
    normalization_target_dbfs: float | None = None,
) -> dict[str, Any]:
    if not permitted_event_usage:
        raise RegisterError("--permitted-event-usage is required")
    usage = _canonical_unique(
        list(permitted_event_usage), owner="permitted_event_usage"
    )
    unknown_usage = [item for item in usage if item not in PERMITTED_USAGE]
    if unknown_usage:
        raise RegisterError(f"unknown permitted_event_usage {unknown_usage}")
    if normalization_policy not in NORMALIZATION_MODES:
        raise RegisterError(
            f"unknown --normalization-policy {normalization_policy!r}"
        )
    if normalization_policy == "preserve":
        if normalization_target_dbfs is not None:
            raise RegisterError("preserve forbids --normalization-target-dbfs")
        target_dbfs = None
    else:
        if normalization_target_dbfs is None:
            raise RegisterError(
                f"--normalization-policy {normalization_policy} needs "
                "--normalization-target-dbfs"
            )
        target_dbfs = float(normalization_target_dbfs)

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("schema") != POOL_SCHEMA:
        raise RegisterError(
            f"{catalog_path} schema {catalog.get('schema')!r} is not {POOL_SCHEMA}"
        )
    source_manifest = catalog.get("source_manifest")
    if not source_manifest:
        raise RegisterError(f"{catalog_path} missing source_manifest")
    manifest_path = Path(source_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise RegisterError(
            f"{manifest_path} schema {manifest.get('schema')!r} is not "
            f"{MANIFEST_SCHEMA}"
        )
    output_root = Path(manifest["output_root"])
    by_id = _manifest_by_asset_id(manifest)
    rights_evidence_sha256 = hashlib.sha256(
        OWNER_ADMISSION_NOTE.encode("utf-8")
    ).hexdigest()

    assets: list[dict[str, Any]] = []
    for index, clip in enumerate(catalog.get("clips") or []):
        owner = f"clips[{index}]"
        asset_id = str(clip.get("sound_asset_id") or "")
        event_class = str(clip.get("event_class") or "")
        prepared = str(clip.get("prepared") or "")
        if not asset_id or not event_class or not prepared:
            raise RegisterError(
                f"{owner} needs sound_asset_id, event_class and prepared"
            )
        row = by_id.get(asset_id)
        if row is None:
            raise RegisterError(f"{owner} {asset_id} is not in {manifest_path}")
        wav_path = (output_root / prepared).resolve()
        if not wav_path.is_file():
            raise RegisterError(f"{owner} wav missing: {wav_path}")
        digest, rate, channels, frames = _wav_fields(wav_path)
        if frames != int(clip["duration_samples"]):
            raise RegisterError(
                f"{owner} duration_samples={clip['duration_samples']} != "
                f"wav frames {frames}"
            )
        if rate != int(clip["sample_rate_hz"]):
            raise RegisterError(
                f"{owner} sample_rate_hz={clip['sample_rate_hz']} != wav {rate}"
            )
        category = category_for_class(event_class)
        tags = _canonical_unique([category, event_class], owner=f"{owner}.tags")
        assets.append(
            {
                "sound_asset_id": asset_id,
                "revision": revision,
                "semantic_sound_class": event_class,
                **speech_metadata_from_mapping(row),
                "taxonomy_path": [category, event_class],
                "instance_lineage_id": None,
                "dry_audio": {
                    "uri": wav_path.as_uri(),
                    "sha256": digest,
                    "sample_rate_hz": rate,
                    "channel_count": channels,
                    "sample_count": frames,
                },
                "normalization_policy": {
                    "mode": normalization_policy,
                    "target_dbfs": target_dbfs,
                },
                "allowed_transforms": list(ALLOWED_TRANSFORMS),
                "permitted_event_usage": list(usage),
                "tags": tags,
                "provenance": {
                    "origin": _origin_for(row, manifest),
                    "license": None,
                    "rights_status": "research_use_only",
                    "rights_evidence_sha256": rights_evidence_sha256,
                },
                "admissibility": ADMISSIBILITY,
            }
        )

    if not assets:
        raise RegisterError(f"{catalog_path} has no clips")
    assets.sort(key=lambda item: (item["sound_asset_id"], item["revision"]))
    registry = bind_content_hash(
        {
            "schema": REGISTRY_SCHEMA,
            "registry_id": registry_id,
            "revision": revision,
            "sound_assets": assets,
        }
    )
    errors = validate_sound_asset_registry(registry)
    errors.extend(dry_audio_byte_errors(registry))
    if errors:
        raise RegisterError("; ".join(errors))
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--permitted-event-usage",
        action="append",
        choices=PERMITTED_USAGE,
        required=True,
        help="Repeatable. No default: the owner names the allowed program modes.",
    )
    parser.add_argument(
        "--normalization-policy",
        choices=NORMALIZATION_MODES,
        required=True,
        help="No default. preserve: the cut clip already has its intended level.",
    )
    parser.add_argument(
        "--normalization-target-dbfs",
        type=float,
        default=None,
        help="Required for peak_dbfs/rms_dbfs; forbidden for preserve.",
    )
    args = parser.parse_args(argv)
    try:
        registry = register_sound_event_assets(
            args.catalog.resolve(),
            args.output.resolve(),
            permitted_event_usage=list(args.permitted_event_usage),
            normalization_policy=args.normalization_policy,
            registry_id=args.registry_id,
            revision=args.revision,
            normalization_target_dbfs=args.normalization_target_dbfs,
        )
    except RegisterError as error:
        print(f"refusing to register: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sound_assets": len(registry["sound_assets"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
