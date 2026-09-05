"""Studio server configuration.

Everything the server may touch — repository, review root, task state root,
and the per-template default input paths — is declared in one JSON config so
the HTTP layer itself holds no deployment policy. The server binds loopback
addresses only; owners preview through an ``ssh -L`` tunnel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

STUDIO_CONFIG_SCHEMA = "avengine_studio_config_v1"

# ThreadingHTTPServer binds an IPv4 socket, so the IPv6 loopback is not
# accepted here even though it is also local-only.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})


class StudioConfigError(ValueError):
    """Raised when a Studio server configuration is invalid."""


@dataclass(frozen=True)
class RoomPolicy:
    """Owner room rules the Studio surfaces on the room catalog."""

    banned_provider_ids: frozenset[str] = frozenset()
    excluded_room_id_substrings: frozenset[str] = frozenset()


@dataclass(frozen=True)
class StudioConfig:
    repository_root: Path
    python_executable: Path
    review_root: Path
    tasks_root: Path
    room_registry_path: Path
    registry_paths: dict[str, Path]
    room_policy: RoomPolicy
    task_templates: dict[str, dict[str, object]]
    host: str = "127.0.0.1"
    port: int = 8765
    main_branch: str = "main"
    scenes_root: Path | None = None
    external_sound_assets: dict[str, str] | None = None
    # index.json of the published sound-source asset tree; its parent
    # directory is the root the asset deck reads and serves files from
    sound_asset_index: Path | None = None
    # root directory of the dry-audio clip library the sounds page reads
    sound_library_root: Path | None = None


def _sound_asset_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "=" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise StudioConfigError(
            "external sound asset IDs must be non-empty text without '=', "
            "surrounding whitespace, or control characters"
        )
    return value


def _resolved_path(value: object, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _required(payload: dict, key: str) -> object:
    if key not in payload:
        raise StudioConfigError(f"studio config is missing required key {key!r}")
    return payload[key]


def load_studio_config(config_path: str | Path) -> StudioConfig:
    config_file = Path(config_path).resolve()
    if not config_file.is_file():
        raise StudioConfigError(f"studio config not found: {config_file}")
    payload = json.loads(config_file.read_text(encoding="utf-8"))
    if payload.get("schema") != STUDIO_CONFIG_SCHEMA:
        raise StudioConfigError(
            f"studio config schema must be {STUDIO_CONFIG_SCHEMA!r}, "
            f"got {payload.get('schema')!r}"
        )

    repository_root = _resolved_path(
        _required(payload, "repository_root"), config_file.parent
    )
    if not repository_root.is_dir():
        raise StudioConfigError(f"repository_root is not a directory: {repository_root}")

    python_executable = _resolved_path(
        _required(payload, "python_executable"), repository_root
    )
    if not python_executable.is_file():
        raise StudioConfigError(f"python_executable not found: {python_executable}")

    review_root = _resolved_path(_required(payload, "review_root"), repository_root)
    if not review_root.is_dir():
        raise StudioConfigError(f"review_root is not a directory: {review_root}")

    tasks_root = _resolved_path(_required(payload, "tasks_root"), repository_root)

    room_registry_path = _resolved_path(
        _required(payload, "room_registry"), repository_root
    )
    if not room_registry_path.is_file():
        raise StudioConfigError(f"room registry not found: {room_registry_path}")

    registries_payload = _required(payload, "registries")
    if not isinstance(registries_payload, dict) or not registries_payload:
        raise StudioConfigError("registries must be a non-empty object of name→path")
    registry_paths: dict[str, Path] = {}
    for name, value in registries_payload.items():
        registry_path = _resolved_path(value, repository_root)
        if not registry_path.is_file():
            raise StudioConfigError(f"registry {name!r} not found: {registry_path}")
        registry_paths[str(name)] = registry_path

    policy_payload = payload.get("room_policy", {})
    if not isinstance(policy_payload, dict):
        raise StudioConfigError("room_policy must be an object")
    room_policy = RoomPolicy(
        banned_provider_ids=frozenset(
            str(item) for item in policy_payload.get("banned_provider_ids", [])
        ),
        excluded_room_id_substrings=frozenset(
            str(item).lower()
            for item in policy_payload.get("excluded_room_id_substrings", [])
        ),
    )

    templates_payload = payload.get("task_templates", {})
    if not isinstance(templates_payload, dict):
        raise StudioConfigError("task_templates must be an object of name→defaults")
    task_templates: dict[str, dict[str, object]] = {}
    for name, defaults in templates_payload.items():
        if not isinstance(defaults, dict):
            raise StudioConfigError(f"task template {name!r} defaults must be an object")
        task_templates[str(name)] = dict(defaults)

    host = str(payload.get("host", "127.0.0.1"))
    if host not in LOOPBACK_HOSTS:
        raise StudioConfigError(
            f"studio host must be loopback ({sorted(LOOPBACK_HOSTS)}); got {host!r}. "
            "Preview remotely through an ssh -L tunnel instead of a public bind."
        )
    port = int(payload.get("port", 8765))
    if not 0 <= port <= 65535:
        raise StudioConfigError(f"port out of range: {port}")

    scenes_root: Path | None = None
    if payload.get("scenes_root") is not None:
        scenes_root = _resolved_path(payload["scenes_root"], repository_root)

    external_sound_assets = None
    if payload.get("external_sound_assets") is not None:
        raw_sound_assets = payload["external_sound_assets"]
        if not isinstance(raw_sound_assets, dict):
            raise StudioConfigError("external_sound_assets must be an object")
        external_sound_assets = {}
        for raw_id, raw_path in raw_sound_assets.items():
            sound_id = _sound_asset_id(raw_id)
            sound_path = _resolved_path(raw_path, repository_root)
            if not sound_path.is_file():
                raise StudioConfigError(
                    f"external sound asset {sound_id!r} not found: {sound_path}"
                )
            external_sound_assets[sound_id] = str(sound_path)

    sound_asset_index: Path | None = None
    if payload.get("sound_asset_index") is not None:
        sound_asset_index = _resolved_path(
            payload["sound_asset_index"], repository_root
        )
        if not sound_asset_index.is_file():
            raise StudioConfigError(
                f"sound_asset_index not found: {sound_asset_index}"
            )

    sound_library_root: Path | None = None
    if payload.get("sound_library_root") is not None:
        sound_library_root = _resolved_path(
            payload["sound_library_root"], repository_root
        )
        if not sound_library_root.is_dir():
            raise StudioConfigError(
                f"sound_library_root is not a directory: {sound_library_root}"
            )

    return StudioConfig(
        scenes_root=scenes_root,
        external_sound_assets=external_sound_assets,
        sound_asset_index=sound_asset_index,
        sound_library_root=sound_library_root,
        repository_root=repository_root,
        python_executable=python_executable,
        review_root=review_root,
        tasks_root=tasks_root,
        room_registry_path=room_registry_path,
        registry_paths=registry_paths,
        room_policy=room_policy,
        task_templates=task_templates,
        host=host,
        port=port,
        main_branch=str(payload.get("main_branch", "main")),
    )
