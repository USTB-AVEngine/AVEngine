"""Fail-closed static contracts for the M4 named multi-source boundary.

This module validates identity and spatial-audio declarations before a native
Habitat/RLR context is created.  It deliberately does not simulate acoustics,
render audio, or promote missing M2 evidence into a qualification claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator
import numpy as np

from avengine.contracts.json_io import canonical_json_sha256
from avengine.contracts.transforms import (
    compose_transforms,
    normalized_quaternion_xyzw,
    validate_transform,
)
from avengine.m1.contracts import validate_capture_request as validate_m1_capture_request
from avengine.m2.contracts import (
    compute_pose_hash,
    validate_animal_asset_package,
    validate_capture_request as validate_m2_capture_request,
)
from avengine.m3.contracts import validate_canary_request as validate_m3_canary_request


REQUEST_SCHEMA = "avengine_m4_multi_source_canary_request_v1"
IDENTITY_SCHEMA = "avengine_m4_source_identity_manifest_v1"
AUDIO_BUNDLE_SCHEMA = "avengine_m4_audio_bundle_v1"
EVIDENCE_SCHEMA = "avengine_m4_multi_source_canary_evidence_v1"
CURRENT_INSTALLED_EVIDENCE_SCHEMA = "avengine_m4_multi_source_canary_evidence_v2"
FOA_FORMAT_ID = "rlr_foa_acn_n3d_world_v1"

_SCHEMA_FILES = {
    REQUEST_SCHEMA: "m4_multi_source_canary_request_v1.schema.json",
    IDENTITY_SCHEMA: "m4_source_identity_manifest_v1.schema.json",
    AUDIO_BUNDLE_SCHEMA: "m4_audio_bundle_v1.schema.json",
    EVIDENCE_SCHEMA: "m4_multi_source_canary_evidence_v1.schema.json",
    CURRENT_INSTALLED_EVIDENCE_SCHEMA: "m4_multi_source_canary_evidence_v2.schema.json",
}
_STABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTENT_HASH_FIELDS = {
    REQUEST_SCHEMA: "request_content_sha256",
    IDENTITY_SCHEMA: "manifest_content_sha256",
    AUDIO_BUNDLE_SCHEMA: "bundle_content_sha256",
    EVIDENCE_SCHEMA: "evidence_content_sha256",
    CURRENT_INSTALLED_EVIDENCE_SCHEMA: "evidence_content_sha256",
}

FOA_CONTRACT: dict[str, Any] = {
    "format_id": FOA_FORMAT_ID,
    "ambisonic_order": 1,
    "channel_count": 4,
    "raw_channel_order": ["W", "Y", "Z", "X"],
    "acn_indices": [0, 1, 2, 3],
    "normalization": "N3D",
    "coordinate_frame": "avengine_world",
    "handedness": "right",
    "axes": {"right": "+X", "up": "+Y", "back": "+Z", "forward": "-Z"},
    "raw_array_layout": "channel_major_[channels,samples]",
    "dtype": "float32_le",
}


class M4ContractError(ValueError):
    """One or more M4 schema, identity, or file-closure checks failed."""

    def __init__(self, errors: Iterable[str]):
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


@dataclass(frozen=True)
class ImmutableFileSnapshot:
    path: Path
    payload: bytes
    byte_size: int
    sha256: str


@dataclass(frozen=True)
class ValidatedM4CanaryRequest:
    request_path: Path
    repository_root: Path
    request: dict[str, Any]
    m1_capture_request_path: Path
    m1_capture_request: dict[str, Any]
    m3_acoustic_canary_request_path: Path
    m3_acoustic_canary_request: dict[str, Any]
    identity_manifest_path: Path
    identity_manifest: dict[str, Any]
    canonical_source_ids: tuple[str, ...]

    @property
    def listener(self) -> dict[str, Any]:
        return self.request["listeners"][0]

    @property
    def sources(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.request["sources"])

    @property
    def all_m2_anchor_evidence_available(self) -> bool:
        return all(
            source["m2_anchor_evidence"]["status"] == "available"
            for source in self.identity_manifest["sources"]
        )


def _schema_path(schema_name: str) -> Path:
    try:
        filename = _SCHEMA_FILES[schema_name]
    except KeyError as exc:
        raise ValueError(f"unknown M4 schema: {schema_name!r}") from exc
    source = Path(__file__).resolve().parents[3] / "schemas" / filename
    installed = Path(sys.prefix) / "share" / "avengine" / "schemas" / filename
    path = source if source.is_file() else installed
    if not path.is_file():
        raise FileNotFoundError(f"AVEngine schema is unavailable: {filename}")
    return path


def _strict_json(payload: bytes, *, owner: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{owner} must be strict UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{owner} must contain one JSON object")
    return value


def _snapshot(
    path: Path, *, cache: dict[Path, ImmutableFileSnapshot]
) -> ImmutableFileSnapshot:
    resolved = path.resolve()
    if resolved in cache:
        return cache[resolved]
    payload = resolved.read_bytes()
    value = ImmutableFileSnapshot(
        path=resolved,
        payload=payload,
        byte_size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    cache[resolved] = value
    return value


def json_schema_errors(value: Any, schema_name: str) -> list[str]:
    schema = _strict_json(_schema_path(schema_name).read_bytes(), owner=schema_name)
    errors: list[str] = []
    for error in sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    ):
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        errors.append(f"JSON Schema {location}: {error.message}")
    return errors


def _all_numbers_finite(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_all_numbers_finite(item) for item in value)
    if isinstance(value, Mapping):
        return all(_all_numbers_finite(item) for item in value.values())
    return False


def _content_hash_errors(value: Mapping[str, Any], schema_name: str) -> list[str]:
    field = _CONTENT_HASH_FIELDS[schema_name]
    declared = value.get(field)
    try:
        actual = canonical_json_sha256(
            {key: item for key, item in value.items() if key != field}
        )
    except (TypeError, ValueError) as exc:
        return [f"{field} cannot be recomputed: {exc}"]
    if declared != actual:
        return [f"{field} does not match canonical document content"]
    return []


def canonical_source_ids(values: Iterable[str]) -> tuple[str, ...]:
    ids = tuple(values)
    errors: list[str] = []
    for index, source_id in enumerate(ids):
        if not isinstance(source_id, str) or not _STABLE_ID.fullmatch(source_id):
            errors.append(
                f"source_ids[{index}] must use 1..128 portable ASCII characters "
                "matching [A-Za-z0-9][A-Za-z0-9_.-]*"
            )
    if len(set(ids)) != len(ids):
        errors.append("source IDs must be unique")
    if errors:
        raise M4ContractError(errors)
    return tuple(sorted(ids, key=lambda item: item.encode("ascii")))


canonical_source_order = canonical_source_ids


def _discover_repository_root(
    document_path: Path, repository_root: str | Path | None
) -> Path:
    if repository_root is not None:
        root = Path(repository_root).resolve()
        if not root.is_dir():
            raise ValueError("repository_root must be an existing directory")
        return root
    resolved = document_path.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "schemas"
        ).is_dir():
            return candidate
    source_root = Path(__file__).resolve().parents[3]
    if source_root.is_dir():
        return source_root
    raise ValueError("unable to discover AVEngine repository root")


def _path_without_symlinks(root: Path, relative: Path) -> bool:
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            return False
    return True


def _record_snapshot(
    record: Any,
    *,
    owner: str,
    base: Path,
    cache: dict[Path, ImmutableFileSnapshot],
    errors: list[str],
) -> ImmutableFileSnapshot | None:
    if not isinstance(record, Mapping):
        errors.append(f"{owner} must be a file record")
        return None
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        errors.append(f"{owner}.path must be a non-empty relative path")
        return None
    declared = Path(raw_path)
    if (
        declared.is_absolute()
        or raw_path.startswith("~")
        or "$" in raw_path
        or "\\" in raw_path
        or any(part in {"", ".", ".."} for part in declared.parts)
    ):
        errors.append(f"{owner}.path must be a confined POSIX relative path")
        return None
    candidate = base.joinpath(*declared.parts)
    if not _path_without_symlinks(base, declared):
        errors.append(f"{owner}.path must not traverse a symbolic link")
        return None
    resolved = candidate.resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError:
        errors.append(f"{owner}.path escapes its declared root")
        return None
    if not resolved.is_file() or resolved.is_symlink():
        errors.append(f"{owner}.path is not an existing regular file: {raw_path}")
        return None
    try:
        snapshot = _snapshot(resolved, cache=cache)
    except OSError as exc:
        errors.append(f"{owner}.path cannot be read: {exc}")
        return None
    if record.get("byte_size") != snapshot.byte_size:
        errors.append(f"{owner}.byte_size does not match {raw_path}")
    if record.get("sha256") != snapshot.sha256:
        errors.append(f"{owner}.sha256 does not match {raw_path}")
    return snapshot


def _json_record(
    record: Any,
    *,
    owner: str,
    base: Path,
    cache: dict[Path, ImmutableFileSnapshot],
    errors: list[str],
) -> tuple[ImmutableFileSnapshot | None, dict[str, Any] | None]:
    snapshot = _record_snapshot(
        record, owner=owner, base=base, cache=cache, errors=errors
    )
    if snapshot is None:
        return None, None
    try:
        value = _strict_json(snapshot.payload, owner=owner)
    except ValueError as exc:
        errors.append(str(exc))
        return snapshot, None
    return snapshot, value


def _finite_vec(value: Any, length: int) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) != length:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return None
    result = np.asarray(value, dtype=np.float64)
    return result if result.shape == (length,) and np.isfinite(result).all() else None


def _position_error(left: Any, right: Any) -> float:
    left_value = _finite_vec(left, 3)
    right_value = _finite_vec(right, 3)
    if left_value is None or right_value is None:
        return math.inf
    return float(np.linalg.norm(left_value - right_value))


def _child(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, Mapping) else None


def _orientation_wxyz_error(left: Any, right: Any) -> float:
    left_value = _finite_vec(left, 4)
    right_value = _finite_vec(right, 4)
    if left_value is None or right_value is None:
        return math.inf
    left_norm = float(np.linalg.norm(left_value))
    right_norm = float(np.linalg.norm(right_value))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return math.inf
    left_value /= left_norm
    right_value /= right_norm
    return float(min(np.linalg.norm(left_value - right_value), np.linalg.norm(left_value + right_value)))


def _validate_foa_contract(value: Any, *, owner: str) -> list[str]:
    if value != FOA_CONTRACT:
        return [
            f"{owner} must be exactly ACN/N3D world FOA [W,Y,Z,X] with "
            "+X right, +Y up, +Z back, and -Z forward"
        ]
    return []


def _unique_field_errors(
    items: Any, fields: Sequence[str], *, owner: str
) -> list[str]:
    if not isinstance(items, list):
        return [f"{owner} must be an array"]
    errors: list[str] = []
    for field in fields:
        values = [item.get(field) for item in items if isinstance(item, Mapping)]
        if len(values) != len(items) or any(not isinstance(value, str) for value in values):
            errors.append(f"{owner}[].{field} must be present on every item")
        elif len(set(values)) != len(values):
            errors.append(f"{owner}[].{field} must be one-to-one and unique")
    return errors


def _validate_m2_anchor_evidence(
    source: Mapping[str, Any],
    *,
    source_index: int,
    repository_root: Path,
    cache: dict[Path, ImmutableFileSnapshot],
) -> list[str]:
    owner = f"sources[{source_index}].m2_anchor_evidence"
    evidence = source.get("m2_anchor_evidence")
    if not isinstance(evidence, Mapping):
        return [f"{owner} must be an object"]
    if evidence.get("status") == "not_run":
        errors: list[str] = []
        if evidence.get("qualification_claim") is not False:
            errors.append(f"{owner} not_run must not make a qualification claim")
        if not isinstance(evidence.get("reason"), str) or not evidence.get("reason"):
            errors.append(f"{owner}.reason must explain the missing tracked evidence")
        return errors
    if evidence.get("status") != "available":
        return [f"{owner}.status must be available or not_run"]

    errors = []
    asset_snapshot, asset = _json_record(
        evidence.get("asset_manifest"),
        owner=f"{owner}.asset_manifest",
        base=repository_root,
        cache=cache,
        errors=errors,
    )
    capture_snapshot, capture = _json_record(
        evidence.get("capture_request"),
        owner=f"{owner}.capture_request",
        base=repository_root,
        cache=cache,
        errors=errors,
    )
    _, resolution = _json_record(
        evidence.get("resolved_anchor_evidence"),
        owner=f"{owner}.resolved_anchor_evidence",
        base=repository_root,
        cache=cache,
        errors=errors,
    )
    if asset is None or capture is None or asset_snapshot is None or capture_snapshot is None:
        return errors
    errors.extend(
        f"{owner}.asset_manifest: {error}"
        for error in validate_animal_asset_package(
            asset, manifest_path=asset_snapshot.path
        )
    )
    if asset.get("admission_state") != "canary_qualified":
        errors.append(f"{owner}.asset_manifest must be canary_qualified")
    errors.extend(
        f"{owner}.capture_request: {error}"
        for error in validate_m2_capture_request(
            capture,
            asset=asset,
            asset_manifest_sha256=asset_snapshot.sha256,
        )
    )
    if evidence.get("asset_id") != asset.get("asset_id"):
        errors.append(f"{owner}.asset_id does not match asset manifest")
    asset_anchor_id = evidence.get("asset_anchor_id")
    anchors = asset.get("anchors")
    if not isinstance(anchors, list) or asset_anchor_id not in {
        anchor.get("anchor_id") for anchor in anchors if isinstance(anchor, Mapping)
    }:
        errors.append(f"{owner}.asset_anchor_id is not declared by the asset")
    frame_index = evidence.get("frame_index")
    states = capture.get("states")
    state = (
        states[frame_index]
        if isinstance(states, list)
        and isinstance(frame_index, int)
        and not isinstance(frame_index, bool)
        and 0 <= frame_index < len(states)
        and isinstance(states[frame_index], dict)
        else None
    )
    if state is None:
        errors.append(f"{owner}.frame_index is absent from the M2 capture request")
    else:
        try:
            actual_pose_hash = compute_pose_hash(asset, state)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{owner}.pose_hash cannot be recomputed: {exc}")
        else:
            if evidence.get("pose_hash") != actual_pose_hash:
                errors.append(f"{owner}.pose_hash differs from the M2 pose")

    if not isinstance(resolution, Mapping):
        return errors
    required_resolution = {
        "schema",
        "source_id",
        "actor_id",
        "asset_id",
        "asset_manifest_sha256",
        "capture_request_sha256",
        "frame_index",
        "pose_hash",
        "asset_anchor_id",
        "world_from_actor",
        "actor_from_anchor",
        "world_from_anchor",
        "resolver_implementation_sha256",
        "content_sha256",
    }
    if set(resolution) != required_resolution:
        errors.append(
            f"{owner}.resolved_anchor_evidence must contain the exact M4 resolution receipt fields"
        )
        return errors
    if resolution.get("schema") != "avengine_m4_resolved_actor_anchor_v1":
        errors.append(f"{owner}.resolved_anchor_evidence.schema is invalid")
    receipt_bindings = {
        "source_id": source.get("source_id"),
        "actor_id": source.get("actor_id"),
        "asset_id": asset.get("asset_id"),
        "asset_manifest_sha256": asset_snapshot.sha256,
        "capture_request_sha256": capture_snapshot.sha256,
        "frame_index": frame_index,
        "pose_hash": evidence.get("pose_hash"),
        "asset_anchor_id": asset_anchor_id,
    }
    for field, expected in receipt_bindings.items():
        if resolution.get(field) != expected:
            errors.append(f"{owner}.resolved_anchor_evidence.{field} is not bound")
    for field in ("world_from_actor", "actor_from_anchor", "world_from_anchor"):
        errors.extend(
            f"{owner}.resolved_anchor_evidence: {error}"
            for error in validate_transform(
                resolution.get(field), name=field
            )
        )
    try:
        recomputed = compose_transforms(
            resolution["world_from_actor"], resolution["actor_from_anchor"]
        )
    except (KeyError, TypeError, ValueError):
        recomputed = None
    if recomputed is not None:
        if _position_error(
            recomputed.get("translation_m"),
            resolution["world_from_anchor"].get("translation_m"),
        ) > 1e-9:
            errors.append(f"{owner}.world_from_anchor translation is not composed")
        expected_xyzw = recomputed.get("rotation_xyzw")
        observed_xyzw = resolution["world_from_anchor"].get("rotation_xyzw")
        if expected_xyzw is not None and observed_xyzw is not None:
            expected_wxyz = [expected_xyzw[3], *expected_xyzw[:3]]
            observed_wxyz = [observed_xyzw[3], *observed_xyzw[:3]]
            if _orientation_wxyz_error(expected_wxyz, observed_wxyz) > 1e-9:
                errors.append(f"{owner}.world_from_anchor rotation is not composed")
    if _position_error(
        _child(resolution.get("world_from_anchor"), "translation_m"),
        source.get("position_m"),
    ) > 1e-9:
        errors.append(f"{owner}.resolved anchor differs from the M1-bound source")
    if not _SHA256.fullmatch(str(resolution.get("resolver_implementation_sha256", ""))):
        errors.append(f"{owner}.resolver_implementation_sha256 is invalid")
    actual_content = canonical_json_sha256(
        {key: item for key, item in resolution.items() if key != "content_sha256"}
    )
    if resolution.get("content_sha256") != actual_content:
        errors.append(f"{owner}.resolved_anchor_evidence content hash differs")
    return errors


def _validate_identity_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path | None,
    repository_root: Path | None,
    cache: dict[Path, ImmutableFileSnapshot],
) -> tuple[list[str], dict[str, Any] | None, Path | None]:
    errors = json_schema_errors(manifest, IDENTITY_SCHEMA)
    if not _all_numbers_finite(manifest):
        errors.append("source identity manifest contains a non-finite number")
    errors.extend(_content_hash_errors(manifest, IDENTITY_SCHEMA))
    if manifest_path is None:
        errors.append("manifest_path is required for source identity path/hash closure")
        return errors, None, None
    try:
        root = _discover_repository_root(manifest_path, repository_root)
    except ValueError as exc:
        errors.append(str(exc))
        return errors, None, None

    m1_snapshot, m1 = _json_record(
        manifest.get("m1_capture_request"),
        owner="m1_capture_request",
        base=root,
        cache=cache,
        errors=errors,
    )
    if m1 is not None:
        errors.extend(
            f"m1_capture_request: {error}"
            for error in validate_m1_capture_request(m1)
        )

    sources = manifest.get("sources")
    errors.extend(
        _unique_field_errors(
            sources,
            (
                "source_id",
                "actor_id",
                "event_id",
                "anchor_id",
                "m1_source_id",
                "dry_audio_id",
            ),
            owner="sources",
        )
    )
    if isinstance(sources, list):
        source_ids = [
            source.get("source_id")
            for source in sources
            if isinstance(source, Mapping)
        ]
        try:
            canonical = list(canonical_source_ids(source_ids))
        except M4ContractError as exc:
            errors.extend(exc.errors)
            canonical = []
        if manifest.get("canonical_source_order") != canonical:
            errors.append(
                "canonical_source_order must be the bytewise ASCII sort of source IDs"
            )
        if source_ids != canonical:
            errors.append("identity manifest sources must use canonical source order")

        recipes: list[str] = []
        m1_sources = {
            source.get("source_id"): source
            for source in (m1.get("sources", []) if isinstance(m1, Mapping) else [])
            if isinstance(source, Mapping)
        }
        for index, source in enumerate(sources):
            if not isinstance(source, Mapping):
                continue
            m1_source_id = source.get("m1_source_id")
            m1_source = m1_sources.get(m1_source_id)
            if not isinstance(m1_source, Mapping):
                errors.append(
                    f"sources[{index}].m1_source_id is absent from M1 capture request"
                )
            else:
                expected_position = _child(
                    m1_source.get("world_from_source"), "translation_m"
                )
                if _position_error(source.get("position_m"), expected_position) > 1e-9:
                    errors.append(
                        f"sources[{index}].position_m differs from its M1 source pose"
                    )
            recipe = source.get("deterministic_signal")
            try:
                recipes.append(canonical_json_sha256(recipe))
            except (TypeError, ValueError):
                pass
            errors.extend(
                _validate_m2_anchor_evidence(
                    source,
                    source_index=index,
                    repository_root=root,
                    cache=cache,
                )
            )
        if len(recipes) == len(sources) and len(set(recipes)) != len(recipes):
            errors.append(
                "sources[].deterministic_signal must be distinct for routing evidence"
            )
    return errors, m1, m1_snapshot.path if m1_snapshot is not None else None


def validate_source_identity_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> list[str]:
    path = Path(manifest_path).resolve() if manifest_path is not None else None
    errors, _, _ = _validate_identity_manifest(
        manifest,
        manifest_path=path,
        repository_root=(
            Path(repository_root).resolve() if repository_root is not None else None
        ),
        cache={},
    )
    return errors


def _validate_request(
    request: Mapping[str, Any],
    *,
    request_path: Path | None,
    repository_root: Path | None,
    cache: dict[Path, ImmutableFileSnapshot],
) -> tuple[
    list[str],
    Path | None,
    dict[str, Any] | None,
    Path | None,
    dict[str, Any] | None,
    Path | None,
    dict[str, Any] | None,
]:
    errors = json_schema_errors(request, REQUEST_SCHEMA)
    if not _all_numbers_finite(request):
        errors.append("M4 canary request contains a non-finite number")
    errors.extend(_content_hash_errors(request, REQUEST_SCHEMA))
    errors.extend(_validate_foa_contract(request.get("spatial_audio"), owner="spatial_audio"))
    if request_path is None:
        errors.append("request_path is required for M4 request path/hash closure")
        return errors, None, None, None, None, None, None
    try:
        root = _discover_repository_root(request_path, repository_root)
    except ValueError as exc:
        errors.append(str(exc))
        return errors, None, None, None, None, None, None

    inputs = request.get("inputs")
    inputs = inputs if isinstance(inputs, Mapping) else {}
    m1_snapshot, m1 = _json_record(
        inputs.get("m1_capture_request"),
        owner="inputs.m1_capture_request",
        base=root,
        cache=cache,
        errors=errors,
    )
    m3_snapshot, m3 = _json_record(
        inputs.get("m3_acoustic_canary_request"),
        owner="inputs.m3_acoustic_canary_request",
        base=root,
        cache=cache,
        errors=errors,
    )
    identity_snapshot, identity = _json_record(
        inputs.get("source_identity_manifest"),
        owner="inputs.source_identity_manifest",
        base=root,
        cache=cache,
        errors=errors,
    )
    if m1 is not None:
        errors.extend(
            f"inputs.m1_capture_request: {error}"
            for error in validate_m1_capture_request(m1)
        )
    if m3 is not None:
        errors.extend(
            f"inputs.m3_acoustic_canary_request: {error}"
            for error in validate_m3_canary_request(m3)
        )
    if identity is not None and identity_snapshot is not None:
        identity_errors, identity_m1, identity_m1_path = _validate_identity_manifest(
            identity,
            manifest_path=identity_snapshot.path,
            repository_root=root,
            cache=cache,
        )
        errors.extend(
            f"inputs.source_identity_manifest: {error}"
            for error in identity_errors
        )
        if (
            m1_snapshot is not None
            and identity_m1_path is not None
            and identity_m1_path != m1_snapshot.path
        ):
            errors.append(
                "request and identity manifest must bind the same M1 capture request"
            )
        if m1 is not None and identity_m1 is not None and m1 != identity_m1:
            errors.append(
                "request and identity manifest M1 snapshots differ"
            )

    listeners = request.get("listeners")
    if not isinstance(listeners, list) or len(listeners) != 1:
        errors.append("M4 MVP requires exactly one listener")
        listener = None
    else:
        listener = listeners[0] if isinstance(listeners[0], Mapping) else None
    sources = request.get("sources")
    if not isinstance(sources, list) or len(sources) < 2:
        errors.append("M4 requires at least two named sources")
        sources = []
    errors.extend(_unique_field_errors(sources, ("source_id",), owner="sources"))
    source_ids = [
        source.get("source_id")
        for source in sources
        if isinstance(source, Mapping)
    ]
    try:
        canonical = list(canonical_source_ids(source_ids))
    except M4ContractError as exc:
        errors.extend(exc.errors)
        canonical = []
    if request.get("canonical_source_order") != canonical:
        errors.append(
            "canonical_source_order must be the bytewise ASCII sort of source IDs"
        )
    if source_ids != canonical:
        errors.append("request sources must use canonical source order")

    registration_orders = request.get("registration_orders")
    if isinstance(registration_orders, list):
        order_ids: list[Any] = []
        permutations: list[tuple[Any, ...]] = []
        for index, order in enumerate(registration_orders):
            if not isinstance(order, Mapping):
                continue
            order_ids.append(order.get("order_id"))
            order_sources = order.get("source_ids")
            if (
                not isinstance(order_sources, list)
                or not all(isinstance(item, str) for item in order_sources)
                or len(order_sources) != len(canonical)
                or set(order_sources) != set(canonical)
            ):
                errors.append(
                    f"registration_orders[{index}].source_ids must be an exact source permutation"
                )
            else:
                permutations.append(tuple(order_sources))
        if len(set(order_ids)) != len(order_ids):
            errors.append("registration_orders[].order_id must be unique")
        if len(set(permutations)) != len(permutations):
            errors.append("registration_orders must contain distinct permutations")
        if tuple(canonical) not in permutations:
            errors.append("registration_orders must include canonical source order")
        if tuple(reversed(canonical)) not in permutations:
            errors.append("registration_orders must include reversed source order")

    threshold = request.get("thresholds", {}).get(
        "maximum_anchor_transform_error", 1e-9
    ) if isinstance(request.get("thresholds"), Mapping) else 1e-9
    tolerance = float(threshold) if isinstance(threshold, (int, float)) else 1e-9
    if listener is not None and m1 is not None:
        rig = m1.get("primary_camera_rig")
        m1_listener = m1.get("listener")
        if isinstance(rig, Mapping) and isinstance(m1_listener, Mapping):
            try:
                world_from_listener = compose_transforms(
                    rig["world_from_rig"], m1_listener["rig_from_listener"]
                )
                expected_xyzw = normalized_quaternion_xyzw(
                    world_from_listener["rotation_xyzw"]
                )
                expected_wxyz = [
                    float(expected_xyzw[3]),
                    float(expected_xyzw[0]),
                    float(expected_xyzw[1]),
                    float(expected_xyzw[2]),
                ]
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"unable to compose M1 listener transform: {exc}")
            else:
                if listener.get("listener_id") != m1_listener.get("listener_id"):
                    errors.append("listener_id differs from the M1 formal listener")
                if listener.get("camera_rig_id") != rig.get("rig_id"):
                    errors.append("listener camera_rig_id differs from the M1 formal rig")
                if listener.get("view_id") != rig.get("view_id"):
                    errors.append("listener view_id differs from the M1 formal view")
                if _position_error(
                    listener.get("position_m"), world_from_listener["translation_m"]
                ) > tolerance:
                    errors.append("listener position differs from the M1 formal camera rig")
                if _orientation_wxyz_error(
                    listener.get("orientation_wxyz"), expected_wxyz
                ) > tolerance:
                    errors.append("listener orientation differs from the M1 formal camera rig")
                orientation = _finite_vec(listener.get("orientation_wxyz"), 4)
                if orientation is None or not math.isclose(
                    float(np.linalg.norm(orientation)),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    errors.append("listener orientation_wxyz must already be unit normalized")
        else:
            errors.append("M1 formal camera rig/listener declaration is missing")

    m1_sources = {
        item.get("source_id"): item
        for item in (m1.get("sources", []) if isinstance(m1, Mapping) else [])
        if isinstance(item, Mapping)
    }
    identity_sources = {
        item.get("source_id"): item
        for item in (
            identity.get("sources", []) if isinstance(identity, Mapping) else []
        )
        if isinstance(item, Mapping)
    }
    if identity is not None and set(identity_sources) != set(source_ids):
        errors.append("request sources differ from source identity manifest")
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            continue
        source_id = source.get("source_id")
        m1_source = m1_sources.get(source_id)
        if not isinstance(m1_source, Mapping):
            errors.append(f"sources[{index}].source_id is absent from M1 capture request")
        else:
            expected = _child(m1_source.get("world_from_source"), "translation_m")
            if _position_error(source.get("position_m"), expected) > tolerance:
                errors.append(f"sources[{index}].position_m differs from M1 source pose")
        identity_source = identity_sources.get(source_id)
        if isinstance(identity_source, Mapping) and _position_error(
            source.get("position_m"), identity_source.get("position_m")
        ) > tolerance:
            errors.append(
                f"sources[{index}].position_m differs from source identity manifest"
            )
    for left_index, left in enumerate(sources):
        if not isinstance(left, Mapping):
            continue
        for right_index in range(left_index + 1, len(sources)):
            right = sources[right_index]
            if isinstance(right, Mapping) and _position_error(
                left.get("position_m"), right.get("position_m")
            ) <= tolerance:
                errors.append(
                    f"sources[{left_index}] and sources[{right_index}] must have distinct positions"
                )
        if listener is not None and _position_error(
            left.get("position_m"), listener.get("position_m")
        ) <= tolerance:
            errors.append(f"sources[{left_index}] must be distinct from the listener")

    sample_rate = _child(request.get("simulation"), "sample_rate_hz")
    if isinstance(sample_rate, (int, float)) and not isinstance(sample_rate, bool):
        for source_id, identity_source in identity_sources.items():
            signal = identity_source.get("deterministic_signal")
            frequency = _child(signal, "frequency_hz")
            if isinstance(frequency, (int, float)) and not isinstance(frequency, bool):
                if not 0.0 < float(frequency) < float(sample_rate) / 2.0:
                    errors.append(
                        f"identity source {source_id!r} dry frequency must be below Nyquist"
                    )

    return (
        errors,
        root,
        m1,
        m1_snapshot.path if m1_snapshot is not None else None,
        m3,
        m3_snapshot.path if m3_snapshot is not None else None,
        identity,
    )


def validate_multi_source_canary_request(
    request: Mapping[str, Any],
    *,
    request_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> list[str]:
    path = Path(request_path).resolve() if request_path is not None else None
    root = Path(repository_root).resolve() if repository_root is not None else None
    errors, *_ = _validate_request(
        request,
        request_path=path,
        repository_root=root,
        cache={},
    )
    return errors


validate_canary_request = validate_multi_source_canary_request


def load_and_validate_multi_source_canary_request(
    request_path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> ValidatedM4CanaryRequest:
    path = Path(request_path).resolve()
    cache: dict[Path, ImmutableFileSnapshot] = {}
    try:
        request_snapshot = _snapshot(path, cache=cache)
        request = _strict_json(request_snapshot.payload, owner="M4 canary request")
    except (OSError, ValueError) as exc:
        raise M4ContractError([str(exc)]) from exc
    (
        errors,
        root,
        m1,
        m1_path,
        m3,
        m3_path,
        identity,
    ) = _validate_request(
        request,
        request_path=path,
        repository_root=(
            Path(repository_root).resolve() if repository_root is not None else None
        ),
        cache=cache,
    )
    if errors:
        raise M4ContractError(errors)
    assert root is not None and m1 is not None and m1_path is not None
    assert m3 is not None and m3_path is not None and identity is not None
    identity_record = request["inputs"]["source_identity_manifest"]
    identity_path = root / identity_record["path"]
    return ValidatedM4CanaryRequest(
        request_path=path,
        repository_root=root,
        request=dict(request),
        m1_capture_request_path=m1_path,
        m1_capture_request=dict(m1),
        m3_acoustic_canary_request_path=m3_path,
        m3_acoustic_canary_request=dict(m3),
        identity_manifest_path=identity_path.resolve(),
        identity_manifest=dict(identity),
        canonical_source_ids=tuple(request["canonical_source_order"]),
    )


load_and_validate_request = load_and_validate_multi_source_canary_request


def _artifact_records_from_audio_bundle(bundle: Mapping[str, Any]) -> list[tuple[str, Any]]:
    records: list[tuple[str, Any]] = []
    identity = bundle.get("source_identity_manifest")
    records.append(("source_identity_manifest", identity))
    pairs = bundle.get("pairs")
    if isinstance(pairs, list):
        for index, pair in enumerate(pairs):
            if not isinstance(pair, Mapping):
                continue
            for role in ("rir", "stem"):
                artifact = pair.get(role)
                if isinstance(artifact, Mapping) and artifact.get("status") == "available":
                    records.append((f"pairs[{index}].{role}.file", artifact.get("file")))
                    if "sidecar" in artifact:
                        records.append((f"pairs[{index}].{role}.sidecar", artifact.get("sidecar")))
    binaural = bundle.get("binaural_decoder")
    if isinstance(binaural, Mapping) and binaural.get("status") == "pass":
        hrtf = binaural.get("hrtf")
        if isinstance(hrtf, Mapping):
            records.append(("binaural_decoder.hrtf.file", hrtf.get("file")))
        rights = binaural.get("rights")
        if isinstance(rights, Mapping):
            records.append(
                ("binaural_decoder.rights.license_file", rights.get("license_file"))
            )
        outputs = binaural.get("outputs")
        if isinstance(outputs, list):
            for index, output in enumerate(outputs):
                if isinstance(output, Mapping):
                    records.append((f"binaural_decoder.outputs[{index}].file", output.get("file")))
                    records.append((f"binaural_decoder.outputs[{index}].sidecar", output.get("sidecar")))
    return records


def validate_audio_bundle(
    bundle: Mapping[str, Any],
    *,
    bundle_path: str | Path | None = None,
) -> list[str]:
    errors = json_schema_errors(bundle, AUDIO_BUNDLE_SCHEMA)
    if not _all_numbers_finite(bundle):
        errors.append("M4 audio bundle contains a non-finite number")
    errors.extend(_content_hash_errors(bundle, AUDIO_BUNDLE_SCHEMA))
    errors.extend(_validate_foa_contract(bundle.get("spatial_audio"), owner="spatial_audio"))
    canonical = bundle.get("canonical_source_order")
    try:
        expected_canonical = list(canonical_source_ids(canonical if isinstance(canonical, list) else []))
    except M4ContractError as exc:
        errors.extend(exc.errors)
        expected_canonical = []
    if canonical != expected_canonical:
        errors.append("audio bundle canonical_source_order is not canonical")
    pairs = bundle.get("pairs")
    pairs = pairs if isinstance(pairs, list) else []
    pair_source_ids = [
        pair.get("source_id") for pair in pairs if isinstance(pair, Mapping)
    ]
    if pair_source_ids != expected_canonical:
        errors.append("audio bundle must contain one canonically ordered pair per source")
    listener_id = bundle.get("listener_id")
    if any(
        isinstance(pair, Mapping) and pair.get("listener_id") != listener_id
        for pair in pairs
    ):
        errors.append("audio bundle pairs must use the declared listener_id")
    dry_ids = [
        pair.get("dry", {}).get("dry_audio_id")
        for pair in pairs
        if isinstance(pair, Mapping) and isinstance(pair.get("dry"), Mapping)
    ]
    if len(dry_ids) != len(set(dry_ids)):
        errors.append("audio bundle dry_audio_id values must be one-to-one")
    for index, pair in enumerate(pairs):
        if not isinstance(pair, Mapping):
            continue
        dry = pair.get("dry")
        rir = pair.get("rir")
        stem = pair.get("stem")
        if (
            isinstance(dry, Mapping)
            and isinstance(rir, Mapping)
            and isinstance(stem, Mapping)
            and rir.get("status") == "available"
            and stem.get("status") == "available"
        ):
            expected_samples = dry.get("sample_count", 0) + rir.get("sample_count", 0) - 1
            if stem.get("sample_count") != expected_samples:
                errors.append(
                    f"pairs[{index}].stem.sample_count must equal full convolution length"
                )
    binaural = bundle.get("binaural_decoder")
    if isinstance(binaural, Mapping) and binaural.get("status") == "pass":
        output_ids = [
            output.get("source_id")
            for output in binaural.get("outputs", [])
            if isinstance(output, Mapping)
        ]
        if output_ids != expected_canonical:
            errors.append("binaural outputs must contain each source in canonical order")
        hrtf = binaural.get("hrtf")
        binding = binaural.get("sample_rate_binding")
        if isinstance(binding, Mapping):
            if binding.get("render_sample_rate_hz") != bundle.get("sample_rate_hz"):
                errors.append("binaural sample-rate binding differs from the bundle")
            if isinstance(hrtf, Mapping) and binding.get(
                "hrtf_input_sample_rate_hz"
            ) != hrtf.get("sample_rate_hz"):
                errors.append("binaural sample-rate binding differs from the HRTF")
            if binding.get("policy") == "strict_match" and (
                binding.get("render_sample_rate_hz")
                != binding.get("hrtf_input_sample_rate_hz")
                or binding.get("native_rate_adaptation") != "not_required"
            ):
                errors.append("strict binaural sample-rate policy must not adapt rates")
    if bundle.get("overall_status") == "pass":
        if bundle.get("failure_reasons"):
            errors.append("passing audio bundle cannot contain failure reasons")
        if any(
            not isinstance(pair, Mapping)
            or _child(pair.get("rir"), "status") != "available"
            or _child(pair.get("stem"), "status") != "available"
            for pair in pairs
        ):
            errors.append("passing audio bundle requires every RIR and stem")
        if not isinstance(binaural, Mapping) or binaural.get("status") != "pass":
            errors.append(
                "passing M4 audio bundle requires native binaural cardinal validation"
            )
    elif not bundle.get("failure_reasons"):
        errors.append("non-passing audio bundle must explain why it did not pass")

    if bundle_path is None:
        errors.append("bundle_path is required for audio bundle path/hash closure")
        return errors
    base = Path(bundle_path).resolve().parent
    cache: dict[Path, ImmutableFileSnapshot] = {}
    resolved_paths: set[Path] = set()
    for owner, record in _artifact_records_from_audio_bundle(bundle):
        snapshot = _record_snapshot(
            record, owner=owner, base=base, cache=cache, errors=errors
        )
        if snapshot is not None:
            if snapshot.path in resolved_paths:
                errors.append(f"{owner} aliases another audio bundle artifact")
            resolved_paths.add(snapshot.path)
    return errors


def _artifact_role_references(value: Any, *, path: str = "$") -> list[tuple[str, Any]]:
    references: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            item_path = f"{path}.{key}"
            if key == "role" or key.endswith("_role"):
                references.append((item_path, item))
            elif key == "roles" or key.endswith("_roles"):
                if isinstance(item, list):
                    references.extend((f"{item_path}[{index}]", role) for index, role in enumerate(item))
                elif isinstance(item, Mapping):
                    references.extend(
                        (f"{item_path}.{name}", role)
                        for name, role in item.items()
                    )
                else:
                    references.append((item_path, item))
            else:
                references.extend(_artifact_role_references(item, path=item_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            references.extend(_artifact_role_references(item, path=f"{path}[{index}]"))
    return references


def validate_multi_source_canary_evidence(
    evidence: Mapping[str, Any],
    *,
    evidence_path: str | Path | None = None,
    _schema_name: str = EVIDENCE_SCHEMA,
) -> list[str]:
    errors = json_schema_errors(evidence, _schema_name)
    if not _all_numbers_finite(evidence):
        errors.append("M4 canary evidence contains a non-finite number")
    errors.extend(_content_hash_errors(evidence, _schema_name))
    status = evidence.get("overall_status")
    checks = evidence.get("checks")
    checks = checks if isinstance(checks, list) else []
    check_ids = [
        check.get("check_id") for check in checks if isinstance(check, Mapping)
    ]
    if len(check_ids) != len(set(check_ids)):
        errors.append("evidence checks must have unique check_id values")
    if status == "pass":
        if _schema_name == EVIDENCE_SCHEMA and evidence.get(
            "qualification_claim"
        ) is not True:
            errors.append("passing M4 evidence must explicitly make its bounded claim")
        if _schema_name == CURRENT_INSTALLED_EVIDENCE_SCHEMA and evidence.get(
            "qualification_claim"
        ) is not False:
            errors.append(
                "current-installed evidence must not claim historical qualification"
            )
        if evidence.get("failure_reasons"):
            errors.append("passing M4 evidence cannot contain failure reasons")
        if any(
            isinstance(check, Mapping)
            and check.get("required") is True
            and check.get("status") != "pass"
            for check in checks
        ):
            errors.append("passing M4 evidence requires every required check to pass")
    elif not evidence.get("failure_reasons"):
        errors.append("non-passing M4 evidence must contain a failure reason")

    identity = evidence.get("identity")
    identity = identity if isinstance(identity, Mapping) else {}
    declared_source_ids = identity.get("canonical_source_ids")
    source_ids = declared_source_ids if isinstance(declared_source_ids, list) else []
    try:
        canonical_ids = list(canonical_source_ids(source_ids))
    except M4ContractError as exc:
        errors.extend(f"identity: {error}" for error in exc.errors)
        canonical_ids = []
    if source_ids != canonical_ids:
        errors.append("identity.canonical_source_ids must use canonical byte order")
    source_identities = identity.get("source_identities")
    source_identities = (
        source_identities if isinstance(source_identities, Mapping) else {}
    )
    if identity.get("source_count") != len(canonical_ids):
        errors.append("identity.source_count differs from canonical_source_ids")
    if set(source_identities) != set(canonical_ids):
        errors.append("identity.source_identities keys differ from canonical_source_ids")
    errors.extend(
        _unique_field_errors(
            list(source_identities.values()),
            ("actor_id", "event_id", "anchor_id", "m1_source_id", "dry_audio_id"),
            owner="identity.source_identities",
        )
    )

    pairs = evidence.get("pairs")
    pairs = pairs if isinstance(pairs, Mapping) else {}
    if set(pairs) != set(canonical_ids):
        errors.append("pairs keys differ from identity.canonical_source_ids")
    listener_id = identity.get("listener_id")
    identity_fields = (
        "actor_id",
        "event_id",
        "anchor_id",
        "semantic_anchor_id",
        "dry_audio_id",
    )
    for source_id in canonical_ids:
        pair = pairs.get(source_id)
        source_identity = source_identities.get(source_id)
        if not isinstance(pair, Mapping) or not isinstance(source_identity, Mapping):
            continue
        if pair.get("source_id") != source_id:
            errors.append(f"pairs.{source_id}.source_id differs from its map key")
        if pair.get("listener_id") != listener_id:
            errors.append(f"pairs.{source_id}.listener_id differs from identity")
        for field in identity_fields:
            if pair.get(field) != source_identity.get(field):
                errors.append(f"pairs.{source_id}.{field} differs from identity")
        direct_arrival = pair.get("direct_arrival")
        if isinstance(direct_arrival, Mapping):
            try:
                expected = (
                    float(direct_arrival["distance_m"])
                    / float(direct_arrival["speed_of_sound_m_s"])
                    * float(
                        evidence["audio_contracts"]["binaural"][
                            "render_sample_rate_hz"
                        ]
                    )
                )
                absolute_error = abs(
                    float(direct_arrival["detected_sample"]) - expected
                )
                declared_expected = float(direct_arrival["expected_sample"])
                declared_absolute_error = float(
                    direct_arrival["absolute_error_samples"]
                )
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                pass
            else:
                if not math.isclose(
                    declared_expected,
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    errors.append(
                        f"pairs.{source_id}.direct_arrival.expected_sample is not geometric"
                    )
                if not math.isclose(
                    declared_absolute_error,
                    absolute_error,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    errors.append(
                        f"pairs.{source_id}.direct_arrival.absolute_error_samples differs"
                    )

    execution = evidence.get("execution")
    execution = execution if isinstance(execution, Mapping) else {}
    if execution.get("canonical_native_source_order") != canonical_ids:
        errors.append("execution canonical native order differs from identity")
    registration_orders = execution.get("requested_registration_orders")
    permutations: list[tuple[Any, ...]] = []
    if isinstance(registration_orders, list):
        for index, order in enumerate(registration_orders):
            order_sources = _child(order, "source_ids")
            if (
                not isinstance(order_sources, list)
                or len(order_sources) != len(canonical_ids)
                or not all(isinstance(item, str) for item in order_sources)
                or set(order_sources) != set(canonical_ids)
            ):
                errors.append(
                    f"execution.requested_registration_orders[{index}] is not a source permutation"
                )
            else:
                permutations.append(tuple(order_sources))
    if len(permutations) != len(set(permutations)):
        errors.append("execution requested registration orders must be distinct")

    mixtures = evidence.get("mixtures")
    mixtures = mixtures if isinstance(mixtures, Mapping) else {}
    if mixtures.get("source_ids") != canonical_ids:
        errors.append("mixtures.source_ids differs from identity")
    if mixtures.get("summation_order") != canonical_ids:
        errors.append("mixtures.summation_order must be canonical")

    lifecycle = evidence.get("lifecycle")
    lifecycle = lifecycle if isinstance(lifecycle, Mapping) else {}
    for field in ("fresh_first_roles", "updated_roles", "reset_first_roles"):
        role_map = lifecycle.get(field)
        if not isinstance(role_map, Mapping) or set(role_map) != set(canonical_ids):
            errors.append(f"lifecycle.{field} keys differ from identity")
    if lifecycle.get("moved_source_id") not in canonical_ids:
        errors.append("lifecycle.moved_source_id is absent from identity")
    source_receipts = lifecycle.get("source_registration_receipts_after_update")
    if isinstance(source_receipts, list):
        receipt_ids = [
            receipt.get("source_id")
            for receipt in source_receipts
            if isinstance(receipt, Mapping)
        ]
        if (
            not all(isinstance(item, str) for item in receipt_ids)
            or set(receipt_ids) != set(canonical_ids)
            or len(receipt_ids) != len(canonical_ids)
        ):
            errors.append("lifecycle source registration receipts differ from identity")

    runtime = evidence.get("runtime")
    runtime = runtime if isinstance(runtime, Mapping) else {}
    for layout in ("foa", "binaural"):
        endpoint_receipts = runtime.get(f"{layout}_endpoint_receipts")
        endpoint_receipts = (
            endpoint_receipts if isinstance(endpoint_receipts, Mapping) else {}
        )
        if endpoint_receipts.get("authority") != "native_registration_readback":
            errors.append(
                f"runtime.{layout}_endpoint_receipts authority is not native readback"
            )
        endpoint_sources = endpoint_receipts.get("sources")
        endpoint_ids = [
            source.get("source_id")
            for source in endpoint_sources
            if isinstance(source, Mapping)
        ] if isinstance(endpoint_sources, list) else []
        if endpoint_ids != canonical_ids:
            errors.append(f"runtime.{layout}_endpoint_receipts source order differs")
        if isinstance(endpoint_sources, list) and any(
            not isinstance(source, Mapping)
            or source.get("native_realized") is not True
            for source in endpoint_sources
        ):
            errors.append(
                f"runtime.{layout}_endpoint_receipts sources are not native-realized"
            )
        endpoint_listener = endpoint_receipts.get("listener")
        if _child(endpoint_listener, "listener_id") != listener_id:
            errors.append(f"runtime.{layout}_endpoint_receipts listener differs")
        if _child(endpoint_listener, "native_realized") is not True:
            errors.append(
                f"runtime.{layout}_endpoint_receipts listener is not native-realized"
            )
    audio_contracts = evidence.get("audio_contracts")
    audio_contracts = audio_contracts if isinstance(audio_contracts, Mapping) else {}
    errors.extend(
        _validate_foa_contract(audio_contracts.get("foa"), owner="audio_contracts.foa")
    )
    binaural = audio_contracts.get("binaural")
    binaural = binaural if isinstance(binaural, Mapping) else {}
    if audio_contracts.get("native_rate_adaptation") != binaural.get(
        "sample_rate_binding"
    ):
        errors.append("audio_contracts native rate binding differs from binaural")
    hrtf = binaural.get("hrtf")
    if isinstance(hrtf, Mapping) and hrtf.get("sha256") != hrtf.get(
        "expected_sha256"
    ):
        errors.append("audio_contracts binaural HRTF hash binding differs")
    rights = binaural.get("rights")
    if isinstance(rights, Mapping) and rights.get(
        "license_text_sha256"
    ) != rights.get("expected_license_sha256"):
        errors.append("audio_contracts binaural license hash binding differs")
    rate_binding = binaural.get("sample_rate_binding")
    if _schema_name == EVIDENCE_SCHEMA:
        rlr_binary = _child(
            runtime.get("native_binaries"), "rlr_audio_propagation"
        )
        if (
            isinstance(rate_binding, Mapping)
            and rate_binding.get("policy") == "rlr_native_internal_bound_to_binary"
            and rate_binding.get("rlr_binary_sha256")
            != _child(rlr_binary, "sha256")
        ):
            errors.append(
                "binaural rate adaptation is not bound to the runtime RLR binary"
            )
    elif (
        isinstance(rate_binding, Mapping)
        and rate_binding.get("policy") == "rlr_native_internal_bound_to_binary"
    ):
        errors.append(
            "current-installed evidence must not claim hash-bound native rate adaptation"
        )

    artifacts = evidence.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    sections = {
        key: value
        for key, value in evidence.items()
        if key not in {"artifacts", "evidence_content_sha256"}
    }
    for location, role in _artifact_role_references(sections):
        if not isinstance(role, str) or role not in artifacts:
            errors.append(f"{location} does not resolve to evidence.artifacts")

    if evidence_path is None:
        errors.append("evidence_path is required for evidence path/hash closure")
        return errors
    base = Path(evidence_path).resolve().parent
    cache: dict[Path, ImmutableFileSnapshot] = {}
    resolved_paths: set[Path] = set()
    for role, record in artifacts.items():
        snapshot = _record_snapshot(
            record,
            owner=f"artifacts.{role}",
            base=base,
            cache=cache,
            errors=errors,
        )
        if snapshot is not None:
            if snapshot.path in resolved_paths:
                errors.append(f"artifacts.{role} aliases another artifact role")
            resolved_paths.add(snapshot.path)
    return errors



def validate_current_installed_multi_source_canary_evidence(
    evidence: Mapping[str, Any],
    *,
    evidence_path: str | Path | None = None,
) -> list[str]:
    """Validate a one-time current-installed receipt without a v1 runtime lock."""

    return validate_multi_source_canary_evidence(
        evidence,
        evidence_path=evidence_path,
        _schema_name=CURRENT_INSTALLED_EVIDENCE_SCHEMA,
    )

validate_evidence = validate_multi_source_canary_evidence


__all__ = [
    "AUDIO_BUNDLE_SCHEMA",
    "CURRENT_INSTALLED_EVIDENCE_SCHEMA",
    "EVIDENCE_SCHEMA",
    "FOA_CONTRACT",
    "FOA_FORMAT_ID",
    "IDENTITY_SCHEMA",
    "ImmutableFileSnapshot",
    "M4ContractError",
    "REQUEST_SCHEMA",
    "ValidatedM4CanaryRequest",
    "canonical_source_ids",
    "canonical_source_order",
    "json_schema_errors",
    "load_and_validate_multi_source_canary_request",
    "load_and_validate_request",
    "validate_audio_bundle",
    "validate_canary_request",
    "validate_current_installed_multi_source_canary_evidence",
    "validate_evidence",
    "validate_multi_source_canary_evidence",
    "validate_multi_source_canary_request",
    "validate_source_identity_manifest",
]
