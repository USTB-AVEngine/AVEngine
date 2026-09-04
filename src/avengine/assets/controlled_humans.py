"""Load catalog-described controlled-human runtime inputs.

The catalog is the only source of asset-specific expectations. This module
does not embed a colour, avatar, material, animation, skeleton, or geometry
allow-list. The optional artifact metadata is checked only when a catalog
entry declares it.


Adapted from the retained Eastforward/SPEAR asset tooling at commit
7b4d2cd3 (2026-09-04). See docs/provenance/UPSTREAM_ADAPTATIONS.md;
the retained SPEAR MIT notice is in LICENSES/SPEAR-MIT.txt.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

CATALOG_SCHEMA = "avengine_controlled_humans_catalog_v1"
CATALOG_RELATIVE_PATH = "examples/assets/controlled_humans_v1.json"
_ARTIFACT_NAMES = ("source_asset", "normalization_manifest", "runtime_glb")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ControlledHumanError(ValueError):
    """A catalog or described controlled-human artifact is invalid."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ControlledHumanError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ControlledHumanError(f"JSON contains non-finite number {value}")


def _load_json(path: Path, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ControlledHumanError(f"{description} is not a regular file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except ControlledHumanError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ControlledHumanError(f"invalid {description}: {path}") from error
    if not isinstance(value, dict):
        raise ControlledHumanError(f"{description} must be one JSON object")
    return value


def repository_root(path: str | Path | None = None) -> Path:
    if path is None:
        return Path(__file__).resolve().parents[3]
    root = Path(path).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    return root.resolve()


def default_catalog_path(repo_root: str | Path | None = None) -> Path:
    return repository_root(repo_root).joinpath(
        *PurePosixPath(CATALOG_RELATIVE_PATH).parts
    )


def _catalog_path(
    repo_root: str | Path | None = None,
    catalog_path: str | Path | None = None,
) -> Path:
    selected = catalog_path or os.environ.get("AVENGINE_CONTROLLED_HUMAN_CATALOG")
    path = Path(selected).expanduser() if selected else default_catalog_path(repo_root)
    if not path.is_absolute():
        path = repository_root(repo_root) / path
    return path.resolve()


def _text(value: object, description: str, *, required: bool = False) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ControlledHumanError(f"{description} must be a non-empty string")
    return value


def _finite(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ControlledHumanError(f"{description} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ControlledHumanError(f"{description} must be finite")
    return result


def _strings(value: object, description: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ControlledHumanError(f"{description} must be a non-empty list")
    result = []
    for index, item in enumerate(value):
        text = _text(item, f"{description}[{index}]", required=True)
        assert text is not None
        result.append(text)
    if len(result) != len(set(result)):
        raise ControlledHumanError(f"{description} contains duplicates")
    return result


def _required_strings(value: object, description: str) -> list[str]:
    result = _strings(value, description)
    if result is None:
        raise ControlledHumanError(f"{description} must be declared in the catalog")
    return result


def _range(value: object, description: str) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ControlledHumanError(f"{description} must contain two numbers")
    result = [_finite(item, f"{description}[{i}]") for i, item in enumerate(value)]
    if result[0] > result[1]:
        raise ControlledHumanError(f"{description} lower bound exceeds upper bound")
    return result


def _rgb(value: object, description: str) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 3:
        raise ControlledHumanError(f"{description} must contain three channels")
    result = []
    for index, item in enumerate(value):
        number = _finite(item, f"{description}[{index}]")
        if number < 0 or number > 255 or number != int(number):
            raise ControlledHumanError(
                f"{description}[{index}] must be an integer in [0, 255]"
            )
        result.append(int(number))
    return result


def _artifact(value: object, description: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = {"path": value}
    if not isinstance(value, dict):
        raise ControlledHumanError(f"{description} must be a path or object")
    unknown = set(value) - {"path", "sha256", "size_bytes"}
    if unknown:
        raise ControlledHumanError(
            f"{description} has unexpected keys: {sorted(unknown)}"
        )
    path = value.get("path")
    if not isinstance(path, str) or not path:
        raise ControlledHumanError(f"{description}.path must be non-empty")
    if not Path(path).expanduser().is_absolute():
        relative = PurePosixPath(path)
        if relative.is_absolute() or any(part in {".", ".."} for part in relative.parts):
            raise ControlledHumanError(f"unsafe {description}.path: {path!r}")
    digest = value.get("sha256")
    if digest is not None and (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ControlledHumanError(f"{description}.sha256 must be lowercase SHA-256")
    size = value.get("size_bytes")
    if size is not None and (
        isinstance(size, bool) or not isinstance(size, int) or size < 0
    ):
        raise ControlledHumanError(f"{description}.size_bytes must be non-negative")
    return dict(value)


def _validate(document: Mapping[str, Any], path: Path) -> dict[str, Any]:
    if document.get("schema") != CATALOG_SCHEMA:
        raise ControlledHumanError(
            f"catalog schema must be {CATALOG_SCHEMA!r}: {path}"
        )
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise ControlledHumanError("catalog.entries must be a list")
    for name in ("defaults", "ue_import"):
        value = document.get(name)
        if value is not None and not isinstance(value, Mapping):
            raise ControlledHumanError(f"catalog.{name} must be an object")
    env_name = document.get("artifact_root_env")
    if env_name is not None:
        env_name = _text(env_name, "catalog.artifact_root_env", required=True)
        assert env_name is not None
        if _ENV_NAME.fullmatch(env_name) is None:
            raise ControlledHumanError("catalog.artifact_root_env is not a safe environment name")
    if document.get("producer_root") is not None:
        _text(document["producer_root"], "catalog.producer_root", required=True)

    for field in ("material_names", "image_names", "animation_names",
                  "required_animation_names", "required_primitive_attributes"):
        _strings(document.get(field), f"catalog.{field}")
    for field in ("height_range_cm", "bottom_range_cm"):
        _range(document.get(field), f"catalog.{field}")

    seen: set[str] = set()
    for index, entry in enumerate(entries):
        label = f"entries[{index}]"
        if not isinstance(entry, dict):
            raise ControlledHumanError(f"{label} must be an object")
        tag = _text(entry.get("tag"), f"{label}.tag", required=True)
        _text(entry.get("asset_id"), f"{label}.asset_id", required=True)
        assert tag is not None
        if tag in seen:
            raise ControlledHumanError(f"controlled human tag is duplicated: {tag}")
        seen.add(tag)
        for field in (
            "top_color", "source_tag", "variant_id", "source_asset_id",
            "base_avatar_id", "usage_scope", "source_manifest_schema",
            "normalization_schema", "ue_manifest_schema", "skeleton_family",
            "preview_animation_name", "ue_manifest_relative_path", "producer_root",
        ):
            if field in entry and entry[field] is not None:
                _text(entry[field], f"{label}.{field}", required=True)
        _rgb(entry.get("rgb"), f"{label}.rgb")
        for field in ("height_range_cm", "bottom_range_cm"):
            _range(entry.get(field), f"{label}.{field}")
        for field in (
            "material_names", "image_names", "animation_names",
            "required_animation_names", "required_primitive_attributes",
        ):
            _strings(entry.get(field), f"{label}.{field}")
        for field in _ARTIFACT_NAMES:
            _artifact(entry.get(field), f"{label}.{field}")
        if "expected_bone_count" in entry and (
            isinstance(entry["expected_bone_count"], bool)
            or not isinstance(entry["expected_bone_count"], int)
            or entry["expected_bone_count"] <= 0
        ):
            raise ControlledHumanError(f"{label}.expected_bone_count must be positive")
        for field in ("expected_primitive_count", "expected_texture_count"):
            if field in entry and (
                isinstance(entry[field], bool)
                or not isinstance(entry[field], int)
                or entry[field] <= 0
            ):
                raise ControlledHumanError(f"{label}.{field} must be positive")
        for field in ("requires_in_place_actions",):
            if field in entry and not isinstance(entry[field], bool):
                raise ControlledHumanError(f"{label}.{field} must be boolean")
        if "actor_scale" in entry:
            _finite(entry["actor_scale"], f"{label}.actor_scale")
        nested = entry.get("ue_import")
        if nested is not None and not isinstance(nested, Mapping):
            raise ControlledHumanError(f"{label}.ue_import must be an object")
    return deepcopy(dict(document))


def load_catalog(
    repo_root: str | Path | None = None,
    catalog_path: str | Path | None = None,
) -> dict[str, Any]:
    return _validate(_load_json(_catalog_path(repo_root, catalog_path), "controlled human catalog"),
                     _catalog_path(repo_root, catalog_path))


def document(
    repo_root: str | Path | None = None,
    catalog_path: str | Path | None = None,
) -> dict[str, Any]:
    return load_catalog(repo_root, catalog_path)


def entries(
    repo_root: str | Path | None = None,
    catalog_path: str | Path | None = None,
) -> tuple[dict[str, Any], ...]:
    return tuple(load_catalog(repo_root, catalog_path)["entries"])


def entry_for_tag(
    tag: str,
    repo_root: str | Path | None = None,
    catalog_path: str | Path | None = None,
) -> dict[str, Any]:
    if not isinstance(tag, str) or not tag:
        raise ControlledHumanError("tag must be a non-empty string")
    for entry in entries(repo_root, catalog_path):
        if entry["tag"] == tag:
            return deepcopy(entry)
    raise ControlledHumanError(f"controlled human tag is not described: {tag!r}")


def _settings(entry: Mapping[str, Any], catalog: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("defaults", "ue_import"):
        value = catalog.get(name)
        if isinstance(value, Mapping):
            result.update(value)
    nested = entry.get("ue_import")
    if isinstance(nested, Mapping):
        result.update(nested)
    result.update(entry)
    return result


def _catalog_base(
    catalog: Mapping[str, Any],
    *,
    catalog_path: str | Path | None,
    repo_root: str | Path | None,
) -> Path:
    env_name = catalog.get("artifact_root_env")
    if env_name is not None:
        value = os.environ.get(str(env_name))
        if not value:
            raise ControlledHumanError(
                f"environment {env_name!r} must name the external controlled-human data root"
            )
        base = Path(value).expanduser()
        if not base.is_absolute():
            base = Path.cwd() / base
        return base.resolve()
    if catalog_path is not None:
        path = Path(catalog_path).expanduser()
        if not path.is_absolute():
            path = repository_root(repo_root) / path
        return path.resolve().parent
    return repository_root(repo_root)


def entry_producer_root(
    entry: Mapping[str, Any],
    catalog: Mapping[str, Any],
    *,
    catalog_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> Path:
    value = entry.get("producer_root", catalog.get("producer_root"))
    base = _catalog_base(catalog, catalog_path=catalog_path, repo_root=repo_root)
    if value is None:
        return base
    root = Path(str(value)).expanduser()
    if not root.is_absolute():
        root = base / root
    return root.resolve()


def _check_symlink_components(path: Path, description: str) -> None:
    cursor = Path(path.anchor)
    for part in path.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ControlledHumanError(
                f"{description} contains a symbolic link: {cursor}"
            )


def _artifact_path(root: Path, record: Mapping[str, Any], description: str) -> Path:
    raw = record.get("path")
    if not isinstance(raw, str) or not raw:
        raise ControlledHumanError(f"{description}.path must be non-empty")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        relative = PurePosixPath(raw)
        if relative.is_absolute() or any(part in {".", ".."} for part in relative.parts):
            raise ControlledHumanError(f"unsafe {description}.path: {raw!r}")
        path = root.joinpath(*relative.parts)
    _check_symlink_components(path, description)
    return path.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ControlledHumanError(f"unable to read artifact: {path}") from error
    return digest.hexdigest()


def resolve_artifact(
    entry: Mapping[str, Any],
    catalog: Mapping[str, Any],
    name: str,
    *,
    catalog_path: str | Path | None = None,
    repo_root: str | Path | None = None,
    validate: bool = True,
) -> tuple[Path, dict[str, Any]] | None:
    if name not in _ARTIFACT_NAMES:
        raise ControlledHumanError(f"unknown controlled-human artifact: {name}")
    value = entry.get(name)
    if value is None:
        return None
    record = {"path": value} if isinstance(value, str) else value
    if not isinstance(record, Mapping):
        raise ControlledHumanError(f"{name} must be a path or object")
    root = entry_producer_root(
        entry, catalog, catalog_path=catalog_path, repo_root=repo_root
    )
    path = _artifact_path(root, record, name)
    if not validate:
        return path, dict(record)
    if path.is_symlink() or not path.is_file():
        raise ControlledHumanError(f"{name} is not a direct regular file: {path}")
    size = path.stat().st_size
    digest = _sha256(path)
    if record.get("size_bytes") is not None and record["size_bytes"] != size:
        raise ControlledHumanError(
            f"{name} size differs from its descriptive record: "
            f"observed {size}, recorded {record['size_bytes']}"
        )
    if record.get("sha256") is not None and record["sha256"] != digest:
        raise ControlledHumanError(
            f"{name} hash differs from its descriptive record: "
            f"observed {digest}, recorded {record['sha256']}"
        )
    return path, {"path": str(path), "sha256": digest, "size_bytes": size}


def validate_artifacts(
    entry: Mapping[str, Any],
    catalog: Mapping[str, Any],
    *,
    catalog_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, tuple[Path, dict[str, Any]]]:
    result: dict[str, tuple[Path, dict[str, Any]]] = {}
    for name in _ARTIFACT_NAMES:
        value = resolve_artifact(
            entry,
            catalog,
            name,
            catalog_path=catalog_path,
            repo_root=repo_root,
        )
        if value is not None:
            result[name] = value
    return result


def _setting(settings: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if settings.get(name) is not None:
            return settings[name]
    return None


def _required_int(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ControlledHumanError(f"{description} must be a positive integer in the catalog")
    return value


def _required_text(value: object, description: str) -> str:
    result = _text(value, description, required=True)
    assert result is not None
    return result


def _manifest_path(value: object, tag: str) -> str:
    path = Path(_required_text(value, f"{tag}.ue_manifest_relative_path")).expanduser()
    if path.is_absolute():
        return str(path)
    relative = PurePosixPath(str(path))
    if relative.is_absolute() or any(part in {".", ".."} for part in relative.parts):
        raise ControlledHumanError(f"unsafe {tag} UE manifest path: {value!r}")
    return relative.as_posix()


def load_importer_contracts(
    repo_root: str | Path | None = None,
    catalog_path: str | Path | None = None,
    *,
    validate_artifacts: bool = False,
) -> dict[str, dict[str, Any]]:
    """Build import descriptions from JSON, with no Python asset defaults."""

    root = repository_root(repo_root)
    path = _catalog_path(root, catalog_path)
    catalog = load_catalog(root, path)
    result: dict[str, dict[str, Any]] = {}
    for raw in catalog["entries"]:
        tag = _required_text(raw["tag"], "entry.tag")
        setting = _settings(raw, catalog)
        checked = (
            globals()["validate_artifacts"](
                raw, catalog, catalog_path=path, repo_root=root
            )
            if validate_artifacts
            else {}
        )
        entry_root = entry_producer_root(
            raw, catalog, catalog_path=path, repo_root=root
        )
        artifact_paths: dict[str, str] = {}
        artifact_records: dict[str, dict[str, Any]] = {}
        for name in _ARTIFACT_NAMES:
            value = raw.get(name)
            if value is None:
                continue
            record = {"path": value} if isinstance(value, str) else value
            if not isinstance(record, Mapping):
                raise ControlledHumanError(f"{tag}.{name} must be a path or object")
            if name in checked:
                artifact_paths[name] = str(checked[name][0])
                artifact_records[name] = checked[name][1]
            else:
                artifact_paths[name] = str(_artifact_path(entry_root, record, name))
                artifact_records[name] = dict(record)

        materials = _required_strings(
            _setting(setting, "material_names"), f"{tag}.material_names"
        )
        images = _required_strings(
            _setting(setting, "image_names"), f"{tag}.image_names"
        )
        animations = _required_strings(
            _setting(
                setting, "animation_names", "required_animation_names"
            ),
            f"{tag}.animation_names",
        )
        attributes = _required_strings(
            _setting(setting, "required_primitive_attributes"),
            f"{tag}.required_primitive_attributes",
        )
        expected_bones = _required_int(
            _setting(setting, "expected_bone_count", "skin_joint_count"),
            f"{tag}.expected_bone_count",
        )
        expected_primitives = _required_int(
            _setting(setting, "expected_primitive_count"),
            f"{tag}.expected_primitive_count",
        )
        expected_textures = _required_int(
            _setting(setting, "expected_texture_count"),
            f"{tag}.expected_texture_count",
        )
        ue_manifest_schema = _required_text(
            _setting(setting, "ue_manifest_schema"),
            f"{tag}.ue_manifest_schema",
        )
        preview_animation_name = _required_text(
            _setting(setting, "preview_animation_name"),
            f"{tag}.preview_animation_name",
        )
        if preview_animation_name not in animations:
            raise ControlledHumanError(
                f"{tag}.preview_animation_name must be one of animation_names"
            )
        actor_scale = _finite(
            _setting(setting, "actor_scale"),
            f"{tag}.actor_scale",
        )
        requires_in_place_actions = _setting(
            setting, "requires_in_place_actions"
        )
        if requires_in_place_actions is not None and not isinstance(
            requires_in_place_actions, bool
        ):
            raise ControlledHumanError(
                f"{tag}.requires_in_place_actions must be boolean"
            )

        root_value = entry_root
        result[tag] = {
            "tag": tag,
            "asset_id": _required_text(setting.get("asset_id"), f"{tag}.asset_id"),
            "top_color": deepcopy(setting.get("top_color")),
            "rgb": deepcopy(setting.get("rgb")),
            "source_tag": setting.get("source_tag"),
            "variant_id": setting.get("variant_id"),
            "source_asset_id": setting.get("source_asset_id"),
            "base_avatar_id": setting.get("base_avatar_id"),
            "producer_root": str(root_value),
            "catalog_path": str(path),
            "catalog_schema": CATALOG_SCHEMA,
            "artifact_paths": artifact_paths,
            "artifact_records": artifact_records,
            "runtime_glb": (
                Path(artifact_paths["runtime_glb"]).name
                if "runtime_glb" in artifact_paths
                else "runtime.glb"
            ),
            "source_manifest": (
                Path(artifact_paths["normalization_manifest"]).name
                if "normalization_manifest" in artifact_paths
                else "normalization_manifest.json"
            ),
            "runtime_glb_path": artifact_paths.get("runtime_glb"),
            "source_manifest_path": artifact_paths.get("normalization_manifest"),
            "source_asset_path": artifact_paths.get("source_asset"),
            "expected_bone_count": expected_bones,
            "expected_material_names": materials,
            "expected_image_names": images,
            "required_animation_names": animations,
            "required_primitive_attributes": attributes,
            "expected_primitive_count": expected_primitives,
            "expected_texture_count": expected_textures,
            "preview_animation_name": preview_animation_name,
            "actor_scale": actor_scale,
            "expected_skeleton_family": _setting(
                setting, "skeleton_family", "expected_skeleton_family"
            ),
            "height_range_cm": _range(
                _setting(setting, "height_range_cm"), f"{tag}.height_range_cm"
            ),
            "bottom_range_cm": _range(
                _setting(setting, "bottom_range_cm"), f"{tag}.bottom_range_cm"
            ),
            "authored_height_cm": _setting(setting, "authored_height_cm"),
            "source_manifest_schema": _setting(
                setting, "source_manifest_schema"
            ),
            "normalization_schema": _setting(setting, "normalization_schema"),
            "ue_manifest_schema": ue_manifest_schema,
            "usage_scope": _setting(setting, "usage_scope"),
            "requires_in_place_actions": requires_in_place_actions,
            "ue_manifest_relative_path": _manifest_path(
                _setting(setting, "ue_manifest_relative_path"),
                tag,
            ),
            "raw_entry": deepcopy(raw),
        }
        for name in (
            "expected_skeleton_family",
            "source_manifest_schema",
            "normalization_schema",
            "usage_scope",
        ):
            value = result[tag][name]
            if value is not None:
                result[tag][name] = _required_text(value, f"{tag}.{name}")
        for name in ("authored_height_cm",):
            value = result[tag][name]
            if value is not None:
                result[tag][name] = _finite(value, f"{tag}.{name}")
        walking = result[tag]["requires_in_place_actions"]
        if walking is not None and not isinstance(walking, bool):
            raise ControlledHumanError(f"{tag}.requires_in_place_actions must be boolean")
    return result


def resolve_import_contract(
    tag: str,
    repo_root: str | Path | None = None,
    catalog_path: str | Path | None = None,
    *,
    validate_artifacts: bool = False,
) -> dict[str, Any]:
    contracts = load_importer_contracts(
        repo_root, catalog_path, validate_artifacts=validate_artifacts
    )
    try:
        return contracts[tag]
    except KeyError as error:
        raise ControlledHumanError(
            f"controlled human tag is not described: {tag!r}"
        ) from error


__all__ = [
    "CATALOG_RELATIVE_PATH",
    "CATALOG_SCHEMA",
    "ControlledHumanError",
    "default_catalog_path",
    "document",
    "entries",
    "entry_for_tag",
    "entry_producer_root",
    "load_catalog",
    "load_importer_contracts",
    "repository_root",
    "resolve_artifact",
    "resolve_import_contract",
    "validate_artifacts",
]
