"""Resolve Habitat bindings for static source assets.

The runtime source registry and the external sound-source index intentionally
remain separate inputs. This module joins them by asset_id and exposes a small
renderer-facing binding object. It does not mutate either registry. External
animal entries are kept in the inventory and are never classified as rigid
objects; their articulated Habitat packages are owned by P12.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HABITAT_BINDING_SCHEMA = "avengine_habitat_runtime_binding_v1"
HABITAT_BINDING_DELTA_SCHEMA = "avengine_habitat_runtime_binding_delta_v1"
RIGID_ENTITY_CLASSES = frozenset({"rigid_object", "rigid_static_object"})
ARTICULATED_ENTITY_CLASSES = frozenset({"articulated_animal", "articulated_human"})


class HabitatStaticAssetError(ValueError):
    """A Habitat source asset binding is absent or inconsistent."""


def _regular_json(path: str | Path, *, owner: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise HabitatStaticAssetError(f"{owner} must be a regular file: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HabitatStaticAssetError(f"cannot read {owner}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise HabitatStaticAssetError(f"{owner} must be a JSON object")
    return resolved, dict(value)


def _asset_list(value: Mapping[str, Any], *, owner: str) -> tuple[dict[str, Any], ...]:
    assets = value.get("assets")
    if not isinstance(assets, list):
        raise HabitatStaticAssetError(f"{owner} has no assets list")
    records: list[dict[str, Any]] = []
    for ordinal, asset in enumerate(assets):
        if not isinstance(asset, Mapping):
            raise HabitatStaticAssetError(f"{owner}.assets[{ordinal}] must be an object")
        asset_id = asset.get("asset_id")
        if not isinstance(asset_id, str) or not asset_id:
            raise HabitatStaticAssetError(
                f"{owner}.assets[{ordinal}].asset_id must be non-empty"
            )
        records.append(deepcopy(dict(asset)))
    return tuple(records)


def _non_empty_text(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HabitatStaticAssetError(f"{owner} must be non-empty text")
    return value


def _entity_class(asset: Mapping[str, Any], *, owner: str) -> str:
    value = asset.get("entity_class")
    return _non_empty_text(value, owner=f"{owner}.entity_class")


def _asset_category(asset: Mapping[str, Any], *, owner: str) -> str | None:
    value = asset.get("category")
    if value is None:
        identity = asset.get("identity")
        if isinstance(identity, Mapping):
            value = identity.get("category") or identity.get("species_id")
            if value is None:
                value = identity.get("object_type")
    if value is None:
        return None
    return _non_empty_text(value, owner=f"{owner}.category")


def _normalized_entity_class(value: str) -> str:
    if value in RIGID_ENTITY_CLASSES:
        return "rigid_object"
    if value in ARTICULATED_ENTITY_CLASSES:
        return value
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_vec(value: Any, *, owner: str, length: int = 3) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise HabitatStaticAssetError(f"{owner} must be a finite vector")
    if len(value) != length:
        raise HabitatStaticAssetError(f"{owner} must have length {length}")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HabitatStaticAssetError(f"{owner} must be a finite vector") from exc
    if any(not item == item or item in (float("inf"), float("-inf")) for item in result):
        raise HabitatStaticAssetError(f"{owner} must be a finite vector")
    return result


def _normalize_emitter(
    value: Any,
    *,
    owner: str,
    fallback_anchor_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HabitatStaticAssetError(f"{owner} must be an object")
    anchor_id = value.get("anchor_id") or fallback_anchor_id
    anchor_id = _non_empty_text(anchor_id, owner=f"{owner}.anchor_id")
    offset = value.get("offset_m")
    if offset is None:
        offset = value.get("translation_m")
    if offset is None:
        offset = [0.0, 0.0, 0.0]
    normalized = deepcopy(dict(value))
    normalized["anchor_id"] = anchor_id
    normalized["offset_m"] = list(_finite_vec(offset, owner=f"{owner}.offset_m"))
    normalized.setdefault("offset_space", "final_scaled_asset_root")
    return normalized


def _normalize_resting_pose(value: Any, *, owner: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HabitatStaticAssetError(f"{owner} must be an object")
    normalized = deepcopy(dict(value))
    attachment = normalized.get("attachment_surface")
    if not isinstance(attachment, str) or attachment not in {"floor", "wall", "ceiling"}:
        raise HabitatStaticAssetError(
            f"{owner}.attachment_surface must be floor, wall or ceiling"
        )
    height = normalized.get("height_m")
    if height is not None:
        try:
            height = float(height)
        except (TypeError, ValueError, OverflowError) as exc:
            raise HabitatStaticAssetError(f"{owner}.height_m must be finite") from exc
        if height < 0.0 or height != height or height in (float("inf"), float("-inf")):
            raise HabitatStaticAssetError(f"{owner}.height_m must be finite and nonnegative")
        normalized["height_m"] = height
    offset = normalized.get("base_plane_offset_m", 0.0)
    try:
        offset = float(offset)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HabitatStaticAssetError(
            f"{owner}.base_plane_offset_m must be finite"
        ) from exc
    if offset != offset or offset in (float("inf"), float("-inf")):
        raise HabitatStaticAssetError(f"{owner}.base_plane_offset_m must be finite")
    normalized["base_plane_offset_m"] = offset
    return normalized


def _resolve_external_glb(
    *,
    index_path: Path,
    asset: Mapping[str, Any],
    geometry: Mapping[str, Any],
    owner: str,
) -> tuple[Path, str]:
    raw = geometry.get("finalized_glb")
    if not isinstance(raw, str) or not raw:
        raise HabitatStaticAssetError(
            f"{owner} has no geometry.finalized_glb; articulated assets require P12"
        )
    asset_path = asset.get("path")
    if not isinstance(asset_path, str) or not asset_path:
        raise HabitatStaticAssetError(f"{owner}.path is required")
    relative = Path(asset_path) / raw
    resolved = (index_path.parent / relative).resolve()
    try:
        resolved.relative_to(index_path.parent.resolve())
    except ValueError as exc:
        raise HabitatStaticAssetError(f"{owner} GLB escapes external asset root") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise HabitatStaticAssetError(f"{owner} finalized GLB is missing: {resolved}")
    declared_sha = geometry.get("finalized_glb_sha256")
    if isinstance(declared_sha, str) and declared_sha and _sha256(resolved) != declared_sha:
        raise HabitatStaticAssetError(
            f"{owner} finalized GLB sha256 differs from index: {resolved}"
        )
    return resolved, relative.as_posix()


@dataclass(frozen=True)
class HabitatAssetBinding:
    """Renderer-facing identity and placement binding for one source asset."""

    asset_id: str
    entity_class: str
    category: str | None
    asset_kind: str
    glb_path: Path
    glb_relative_path: str
    semantic_template: Mapping[str, Any]
    resting_pose: Mapping[str, Any]
    emitter: Mapping[str, Any]
    source: str
    revision: str | None = None
    asset_manifest_path: Path | None = None
    base_m2_request_path: Path | None = None

    @property
    def normalized_entity_class(self) -> str:
        return _normalized_entity_class(self.entity_class)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": HABITAT_BINDING_SCHEMA,
            "asset_id": self.asset_id,
            "entity_class": self.entity_class,
            "normalized_entity_class": self.normalized_entity_class,
            "category": self.category,
            "asset_kind": self.asset_kind,
            "glb_path": str(self.glb_path),
            "glb_relative_path": self.glb_relative_path,
            "semantic_template": deepcopy(dict(self.semantic_template)),
            "resting_pose": deepcopy(dict(self.resting_pose)),
            "emitter": deepcopy(dict(self.emitter)),
            "source": self.source,
            "revision": self.revision,
            "asset_manifest_path": (
                None
                if self.asset_manifest_path is None
                else str(self.asset_manifest_path)
            ),
            "base_m2_request_path": (
                None
                if self.base_m2_request_path is None
                else str(self.base_m2_request_path)
            ),
        }


def _binding_from_records(
    *,
    asset_id: str,
    runtime_asset: Mapping[str, Any] | None,
    external_asset: Mapping[str, Any] | None,
    runtime_registry_path: Path | None,
    external_index_path: Path | None,
    habitat_backend: Mapping[str, Any] | None,
    source: str,
) -> HabitatAssetBinding:
    runtime_owner = f"runtime asset {asset_id!r}"
    external_owner = f"external asset {asset_id!r}"
    runtime_class = (
        _entity_class(runtime_asset, owner=runtime_owner)
        if runtime_asset is not None
        else None
    )
    external_class = (
        _entity_class(external_asset, owner=external_owner)
        if external_asset is not None
        else None
    )
    if runtime_class and external_class:
        if _normalized_entity_class(runtime_class) != _normalized_entity_class(external_class):
            raise HabitatStaticAssetError(
                f"{asset_id!r} runtime/external entity classes disagree: "
                f"{runtime_class!r} vs {external_class!r}"
            )
    entity_class = runtime_class or external_class
    if entity_class is None:
        raise HabitatStaticAssetError(f"{asset_id!r} has no entity_class")
    category = _asset_category(
        runtime_asset or external_asset or {},
        owner=runtime_owner if runtime_asset is not None else external_owner,
    )
    if runtime_asset is not None and external_asset is not None:
        external_category = _asset_category(external_asset, owner=external_owner)
        if category is not None and external_category is not None and category != external_category:
            raise HabitatStaticAssetError(
                f"{asset_id!r} runtime/external categories disagree: "
                f"{category!r} vs {external_category!r}"
            )
    backend = deepcopy(dict(habitat_backend or {}))
    glb_path_raw = backend.get("glb_path")
    glb_relative_path = backend.get("glb_relative_path")
    if glb_path_raw is not None:
        glb_path = Path(_non_empty_text(glb_path_raw, owner=f"{asset_id}.glb_path")).expanduser().resolve()
        if not glb_path.is_file() or glb_path.is_symlink():
            raise HabitatStaticAssetError(f"{asset_id!r} Habitat GLB is missing: {glb_path}")
        if not isinstance(glb_relative_path, str) or not glb_relative_path:
            glb_relative_path = glb_path.name
    elif external_asset is not None and external_index_path is not None:
        geometry = external_asset.get("geometry")
        if not isinstance(geometry, Mapping):
            raise HabitatStaticAssetError(f"{external_owner}.geometry is required")
        glb_path, glb_relative_path = _resolve_external_glb(
            index_path=external_index_path,
            asset=external_asset,
            geometry=geometry,
            owner=external_owner,
        )
    else:
        raise HabitatStaticAssetError(
            f"{asset_id!r} lacks a Habitat GLB binding; articulated packages are provided by P12"
        )
    semantic_template = backend.get("semantic_template")
    if not isinstance(semantic_template, Mapping):
        semantic_template = {
            "template_kind": "rigid_object"
            if _normalized_entity_class(entity_class) == "rigid_object"
            else "articulated_package",
            "semantic_id_source": "episode_binding",
        }
    declared_asset_kind = backend.get("asset_kind")
    if declared_asset_kind is not None:
        declared_asset_kind = _non_empty_text(
            declared_asset_kind, owner=f"{asset_id}.asset_kind"
        )
    asset_kind = declared_asset_kind or (
        "rigid_static_object"
        if _normalized_entity_class(entity_class) == "rigid_object"
        else "articulated_asset"
    )
    resting = backend.get("resting_pose")
    if resting is None and external_asset is not None:
        geometry = external_asset.get("geometry")
        if isinstance(geometry, Mapping):
            resting = geometry.get("resting_pose")
    if resting is None:
        resting = {"attachment_surface": "floor", "base_plane_offset_m": 0.0}
    resting_pose = _normalize_resting_pose(resting, owner=f"{asset_id}.resting_pose")
    emitter = backend.get("emitter")
    if emitter is None and external_asset is not None:
        emitter = external_asset.get("emitter")
    if emitter is None and runtime_asset is not None:
        fallback = runtime_asset.get("default_emitter_anchor_id")
        anchors = runtime_asset.get("emitter_anchors")
        if isinstance(anchors, list):
            matches = [
                item for item in anchors
                if isinstance(item, Mapping) and item.get("anchor_id") == fallback
            ]
            emitter = matches[0] if len(matches) == 1 else None
    if emitter is None:
        raise HabitatStaticAssetError(f"{asset_id!r} has no emitter binding")
    emitter_record = _normalize_emitter(
        emitter,
        owner=f"{asset_id}.emitter",
        fallback_anchor_id=(
            runtime_asset.get("default_emitter_anchor_id")
            if runtime_asset is not None
            else None
        ),
    )
    revision = None
    if runtime_asset is not None:
        revision = runtime_asset.get("revision")
    if not isinstance(revision, str) or not revision:
        revision = None
    asset_manifest_path = backend.get("asset_manifest_path")
    if asset_manifest_path is not None:
        asset_manifest_path = Path(
            _non_empty_text(
                asset_manifest_path, owner=f"{asset_id}.asset_manifest_path"
            )
        ).expanduser().resolve()
        if asset_manifest_path.is_symlink() or not asset_manifest_path.is_file():
            raise HabitatStaticAssetError(
                f"{asset_id!r} asset manifest is missing: {asset_manifest_path}"
            )
    base_m2_request_path = backend.get("base_m2_request_path")
    if base_m2_request_path is not None:
        base_m2_request_path = Path(
            _non_empty_text(
                base_m2_request_path, owner=f"{asset_id}.base_m2_request_path"
            )
        ).expanduser().resolve()
        if base_m2_request_path.is_symlink() or not base_m2_request_path.is_file():
            raise HabitatStaticAssetError(
                f"{asset_id!r} M2 request is missing: {base_m2_request_path}"
            )
    return HabitatAssetBinding(
        asset_id=asset_id,
        entity_class=entity_class,
        category=category,
        asset_kind=asset_kind,
        glb_path=glb_path,
        glb_relative_path=str(glb_relative_path),
        semantic_template=semantic_template,
        resting_pose=resting_pose,
        emitter=emitter_record,
        source=source,
        revision=revision,
        asset_manifest_path=asset_manifest_path,
        base_m2_request_path=base_m2_request_path,
    )


def _load_delta(
    path: str | Path | None,
) -> tuple[Path | None, dict[str, Mapping[str, Any]]]:
    if path is None:
        return None, {}
    resolved, value = _regular_json(path, owner="Habitat binding delta")
    if value.get("schema") != HABITAT_BINDING_DELTA_SCHEMA:
        raise HabitatStaticAssetError(
            f"Habitat binding delta schema must be {HABITAT_BINDING_DELTA_SCHEMA!r}"
        )
    raw_bindings = value.get("bindings")
    if not isinstance(raw_bindings, list):
        raise HabitatStaticAssetError("Habitat binding delta has no bindings list")
    bindings: dict[str, Mapping[str, Any]] = {}
    for ordinal, item in enumerate(raw_bindings):
        if not isinstance(item, Mapping):
            raise HabitatStaticAssetError(f"binding delta bindings[{ordinal}] must be an object")
        asset_id = _non_empty_text(item.get("asset_id"), owner=f"bindings[{ordinal}].asset_id")
        if asset_id in bindings:
            raise HabitatStaticAssetError(f"binding delta repeats asset {asset_id!r}")
        backend = item.get("runtime_backend") or item.get("habitat")
        if not isinstance(backend, Mapping):
            raise HabitatStaticAssetError(f"bindings[{ordinal}] has no runtime_backend")
        bindings[asset_id] = deepcopy(dict(backend))
    return resolved, bindings


def load_habitat_asset_bindings(
    asset_ids: Iterable[str],
    *,
    runtime_registry_path: str | Path | None = None,
    external_index_path: str | Path | None = None,
    binding_delta_path: str | Path | None = None,
) -> dict[str, HabitatAssetBinding]:
    """Resolve exact Habitat bindings for asset_ids."""

    requested = tuple(_non_empty_text(item, owner="asset_ids[]") for item in asset_ids)
    if len(set(requested)) != len(requested):
        raise HabitatStaticAssetError("asset_ids must be unique")
    runtime_path = runtime_data = None
    external_path = external_data = None
    if runtime_registry_path is not None:
        runtime_path, runtime_data = _regular_json(
            runtime_registry_path, owner="source asset runtime registry"
        )
    if external_index_path is not None:
        external_path, external_data = _regular_json(
            external_index_path, owner="sound-source asset index"
        )
    _delta_path, delta_bindings = _load_delta(binding_delta_path)
    runtime_assets = {
        item["asset_id"]: item
        for item in _asset_list(runtime_data, owner="source asset runtime registry")
    } if runtime_data is not None else {}
    external_assets = {
        item["asset_id"]: item
        for item in _asset_list(external_data, owner="sound-source asset index")
    } if external_data is not None else {}
    result: dict[str, HabitatAssetBinding] = {}
    for asset_id in requested:
        runtime_asset = runtime_assets.get(asset_id)
        external_asset = external_assets.get(asset_id)
        if runtime_asset is None and external_asset is None:
            raise HabitatStaticAssetError(f"asset {asset_id!r} is absent from supplied registries")
        backend = None
        if runtime_asset is not None:
            backends = runtime_asset.get("runtime_backends")
            if isinstance(backends, Mapping):
                candidate = backends.get("habitat")
                if candidate is not None and not isinstance(candidate, Mapping):
                    raise HabitatStaticAssetError(
                        f"runtime asset {asset_id!r} habitat backend must be an object"
                    )
                backend = candidate
        if backend is None:
            backend = delta_bindings.get(asset_id)
        result[asset_id] = _binding_from_records(
            asset_id=asset_id,
            runtime_asset=runtime_asset,
            external_asset=external_asset,
            runtime_registry_path=runtime_path,
            external_index_path=external_path,
            habitat_backend=backend,
            source=(
                "runtime_registry"
                if runtime_asset is not None and backend is not None
                else "external_index"
                if external_asset is not None
                else "binding_delta"
            ),
        )
    return result


def bind_assets(
    asset_ids: Iterable[str],
    renderer: str = "habitat",
    **kwargs: Any,
) -> dict[str, HabitatAssetBinding]:
    """Renderer-neutral binding entrypoint used by Habitat executors."""

    if renderer != "habitat":
        raise HabitatStaticAssetError(
            f"unsupported renderer {renderer!r}; this adapter resolves habitat bindings"
        )
    return load_habitat_asset_bindings(asset_ids, **kwargs)


def summarize_asset_inventory(
    *,
    runtime_registry_path: str | Path,
    external_index_path: str | Path,
) -> dict[str, Any]:
    """Return the deduplicated external/runtime count used by P4 reports."""

    runtime_path, runtime_data = _regular_json(
        runtime_registry_path, owner="source asset runtime registry"
    )
    external_path, external_data = _regular_json(
        external_index_path, owner="sound-source asset index"
    )
    del runtime_path, external_path
    runtime_assets = {item["asset_id"]: item for item in _asset_list(runtime_data, owner="runtime registry")}
    external_assets = {item["asset_id"]: item for item in _asset_list(external_data, owner="external index")}
    overlap = sorted(set(runtime_assets) & set(external_assets))
    union = {**external_assets, **runtime_assets}

    def counts(records: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
        out: dict[str, int] = {}
        for asset in records.values():
            normalized = _normalized_entity_class(_entity_class(asset, owner="asset"))
            out[normalized] = out.get(normalized, 0) + 1
        return dict(sorted(out.items()))

    return {
        "runtime_total": len(runtime_assets),
        "runtime_by_entity_class": counts(runtime_assets),
        "external_total": len(external_assets),
        "external_by_entity_class": counts(external_assets),
        "overlap_asset_ids": overlap,
        "overlap_count": len(overlap),
        "union_total": len(union),
        "union_by_entity_class": counts(union),
        "external_rigid_count": sum(
            _normalized_entity_class(_entity_class(asset, owner="external asset"))
            == "rigid_object"
            for asset in external_assets.values()
        ),
        "external_articulated_animal_count": sum(
            _entity_class(asset, owner="external asset") == "articulated_animal"
            for asset in external_assets.values()
        ),
        "runtime_rigid_overlap_count": sum(
            asset_id in overlap
            and _normalized_entity_class(_entity_class(runtime_assets[asset_id], owner="runtime asset"))
            == "rigid_object"
            for asset_id in overlap
        ),
    }


def make_binding_delta(
    *,
    runtime_registry_path: str | Path,
    external_index_path: str | Path,
    beagle_asset_manifest_path: str | Path | None = None,
    beagle_m2_request_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a reviewable machine-readable Habitat binding delta."""

    runtime_path, runtime_data = _regular_json(
        runtime_registry_path, owner="source asset runtime registry"
    )
    external_path, external_data = _regular_json(
        external_index_path, owner="sound-source asset index"
    )
    runtime_assets = {item["asset_id"]: item for item in _asset_list(runtime_data, owner="runtime registry")}
    external_assets = {item["asset_id"]: item for item in _asset_list(external_data, owner="external index")}
    bindings: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for asset_id in sorted(external_assets):
        ext = external_assets[asset_id]
        normalized = _normalized_entity_class(_entity_class(ext, owner="external asset"))
        if normalized != "rigid_object":
            excluded.append(
                {
                    "asset_id": asset_id,
                    "entity_class": ext["entity_class"],
                    "category": _asset_category(ext, owner="external asset"),
                    "reason": "articulated_asset_requires_P12_habitat_package",
                }
            )
            continue
        runtime_asset = runtime_assets.get(asset_id)
        binding = _binding_from_records(
            asset_id=asset_id,
            runtime_asset=runtime_asset,
            external_asset=ext,
            runtime_registry_path=runtime_path,
            external_index_path=external_path,
            habitat_backend=None,
            source="external_index" if runtime_asset is None else "runtime_registry+external_index",
        )
        payload = {
            "asset_id": asset_id,
            "runtime_backend": {
                "asset_kind": binding.asset_kind,
                "glb_relative_path": binding.glb_relative_path,
                "semantic_template": deepcopy(dict(binding.semantic_template)),
                "resting_pose": deepcopy(dict(binding.resting_pose)),
                "emitter": deepcopy(dict(binding.emitter)),
                "category": binding.category,
                "entity_class": "rigid_static_object",
            },
        }
        bindings.append(payload)
    if (beagle_asset_manifest_path is None) != (beagle_m2_request_path is None):
        raise HabitatStaticAssetError(
            "beagle asset manifest and M2 request must be supplied together"
        )
    if beagle_asset_manifest_path is not None:
        beagle_id = "rocketbox_dog_beagle_01_m2_v7_world_contact_candidate"
        beagle = runtime_assets.get(beagle_id)
        if beagle is None:
            raise HabitatStaticAssetError(
                f"runtime registry lacks the canonical beagle asset {beagle_id!r}"
            )
        manifest_path, manifest = _regular_json(
            beagle_asset_manifest_path, owner="beagle M2 asset manifest"
        )
        request_path, request = _regular_json(
            beagle_m2_request_path, owner="beagle M2 request"
        )
        if manifest.get("asset_id") != beagle_id or request.get("asset_id") != beagle_id:
            raise HabitatStaticAssetError(
                "beagle M2 manifest/request asset IDs do not match the runtime registry"
            )
        files = manifest.get("files")
        if not isinstance(files, list):
            raise HabitatStaticAssetError("beagle M2 manifest has no files list")
        visual_records = [
            item
            for item in files
            if isinstance(item, Mapping) and item.get("role") == "visual"
        ]
        if len(visual_records) != 1 or not isinstance(visual_records[0].get("path"), str):
            raise HabitatStaticAssetError("beagle M2 manifest has no unique visual role")
        visual_path = (manifest_path.parent / visual_records[0]["path"]).resolve()
        if visual_path.is_symlink() or not visual_path.is_file():
            raise HabitatStaticAssetError(f"beagle visual GLB is missing: {visual_path}")
        anchors = beagle.get("emitter_anchors")
        if not isinstance(anchors, list) or not anchors:
            raise HabitatStaticAssetError("beagle runtime asset has no emitter anchors")
        emitter = anchors[0]
        bindings.append(
            {
                "asset_id": beagle_id,
                "runtime_backend": {
                    "asset_kind": "articulated_m2_package",
                    "glb_path": str(visual_path),
                    "glb_relative_path": visual_path.name,
                    "asset_manifest_path": str(manifest_path),
                    "base_m2_request_path": str(request_path),
                    "semantic_template": {
                        "template_kind": "articulated_m2",
                        "semantic_id_source": "episode_binding",
                        "asset_manifest_path": str(manifest_path),
                    },
                    "resting_pose": {
                        "attachment_surface": "floor",
                        "base_plane_offset_m": 0.0,
                        "measured_from": "M2 package contact/rest calibration",
                        "source_manifest": str(manifest_path),
                    },
                    "emitter": deepcopy(dict(emitter)),
                    "category": "dog",
                    "entity_class": "articulated_animal",
                },
            }
        )
    summary = summarize_asset_inventory(
        runtime_registry_path=runtime_path,
        external_index_path=external_path,
    )
    return {
        "schema": HABITAT_BINDING_DELTA_SCHEMA,
        "renderer": "habitat",
        "source": {
            "runtime_registry": str(runtime_path),
            "external_index": str(external_path),
        },
        "inventory": summary,
        "bindings": bindings,
        "excluded": excluded,
        "claim_boundary": (
            "Bindings are parsed from the existing registries and external "
            "finalized GLBs. They do not claim native load, collision or pixel "
            "qualification; articulated external animals remain P12 inputs."
        ),
    }


__all__ = [
    "ARTICULATED_ENTITY_CLASSES",
    "bind_assets",
    "HABITAT_BINDING_DELTA_SCHEMA",
    "HABITAT_BINDING_SCHEMA",
    "HabitatAssetBinding",
    "HabitatStaticAssetError",
    "RIGID_ENTITY_CLASSES",
    "load_habitat_asset_bindings",
    "make_binding_delta",
    "summarize_asset_inventory",
]
