from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.machinery import EXTENSION_SUFFIXES
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
import uuid

import numpy as np
from PIL import __version__ as pillow_version
from PIL import Image

from avengine.contracts.json_io import (
    canonical_json_sha256,
    file_record,
    load_json,
    resolve_declared_path,
    sha256_file,
    write_json,
)
from avengine.contracts.transforms import (
    compose_transforms,
    invert_transform,
    round_trip_via_parent,
    transform_error,
)
from avengine.m1.contracts import (
    EVIDENCE_SCHEMA_V2,
    ValidatedM1Inputs,
    validate_loaded_scene_asset_graph,
    validate_scene_asset_graph,
)
from avengine.m1.evidence import (
    array_sha256,
    finalize_evidence,
    make_check,
    save_observations,
    verify_evidence_artifacts,
)
from avengine.runtime_lock import RuntimeLockError, resolve_runtime_profile


VISUAL_SENSOR_TYPES = {
    "rgb": "COLOR",
    "depth": "DEPTH",
    "semantic": "SEMANTIC",
}

PROCESS_INSTANCE_ID = str(uuid.uuid4())
PROCESS_STARTED_AT_UTC = datetime.now(timezone.utc).isoformat()
PROCESS_INITIAL_PID = os.getpid()


@dataclass(frozen=True)
class InstalledHabitatRuntime:
    """One explicitly selected installed Habitat runtime and its external inputs."""

    prefix: Path
    mp3d_root: Path | None
    pbr_asset_root: Path | None
    magnum_python_site: Path
    physics_config_path: Path
    quaternion: Any
    habitat_sim: Any
    magnum: Any
    quat_to_coeffs: Any


@dataclass(frozen=True)
class _PreparedInstalledHabitatImport:
    """Activated prefix/Magnum state before the Habitat extension import."""

    prefix: Path
    magnum_python_site: Path


_HABITAT_BINDING_MODULE_NAME = "habitat_sim._ext.habitat_sim_bindings"
_RLR_LIBRARY_BASENAME = "libRLRAudioPropagation.so"
PBR_CONFIG_FILENAME = "brown_photostudio.pbr_config.json"
PBR_BRDF_LUT_RELATIVE_PATH = Path("bluts/brdflut_ldr_512x512.png")
PBR_ENVIRONMENT_MAP_RELATIVE_PATH = Path(
    "env_maps/brown_photostudio_02_1k.hdr"
)


def _producer_process_identity() -> dict[str, Any]:
    return {
        "process_instance_id": PROCESS_INSTANCE_ID,
        "pid": os.getpid(),
        "initial_pid": PROCESS_INITIAL_PID,
        "started_at_utc": PROCESS_STARTED_AT_UTC,
    }


def discover_runtime_root(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).resolve()
    configured = os.environ.get("AVENGINE_HABITAT_RUNTIME_ROOT")
    if configured:
        return Path(configured).resolve()
    repository_root = Path(__file__).resolve().parents[3]
    sibling = repository_root.parent / "habitat-sim-AVEngine"
    if sibling.is_dir():
        return sibling.resolve()
    raise FileNotFoundError("Set AVENGINE_HABITAT_RUNTIME_ROOT or pass --runtime-root")


def _git_checkout_ancestor(path: Path) -> Path | None:
    """Return a containing Git worktree marker without invoking Git itself."""

    candidate = path.resolve()
    while True:
        if os.path.lexists(candidate / ".git"):
            return candidate
        parent = candidate.parent
        if parent == candidate:
            return None
        candidate = parent


def discover_runtime_prefix(explicit: str | Path | None = None) -> Path:
    configured = explicit if explicit is not None else os.environ.get(
        "AVENGINE_HABITAT_RUNTIME_PREFIX"
    )
    if configured is None:
        raise FileNotFoundError(
            "Set AVENGINE_HABITAT_RUNTIME_PREFIX or pass --runtime-prefix"
        )
    prefix = Path(configured).resolve()
    if not prefix.is_dir():
        raise FileNotFoundError(f"Habitat installed runtime prefix is missing: {prefix}")
    checkout_root = _git_checkout_ancestor(prefix)
    if checkout_root is not None:
        raise ValueError(
            "Habitat installed runtime prefix must not be inside a Git checkout: "
            f"{prefix} (found .git at {checkout_root})"
        )
    return prefix


def resolve_installed_runtime_prefix(
    runtime_prefix: str | Path | None = None,
    *,
    runtime_root: str | Path | None = None,
) -> Path:
    """Resolve a non-Git installed prefix; ``runtime_root`` is an alias only."""

    if runtime_prefix is not None and runtime_root is not None:
        raise ValueError(
            "Specify only one of runtime_prefix or runtime_root; runtime_root "
            "is an installed-prefix compatibility alias"
        )
    selected = runtime_prefix if runtime_prefix is not None else runtime_root
    return discover_runtime_prefix(selected)


def _required_magnum_site_file(
    site: Path, relative_path: str, *, description: str
) -> Path:
    candidate = site / relative_path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(site)
    except ValueError as error:
        raise RuntimeError(
            "AVENGINE_HABITAT_MAGNUM_PYTHON_SITE must keep "
            f"{description} under its site root: {candidate}"
        ) from error
    if not resolved.is_file():
        raise FileNotFoundError(
            "AVENGINE_HABITAT_MAGNUM_PYTHON_SITE is missing "
            f"{description}: {candidate}"
        )
    return resolved


def _required_magnum_extension(site: Path, module_name: str) -> Path:
    for suffix in EXTENSION_SUFFIXES:
        candidate = site / f"{module_name}{suffix}"
        if candidate.exists() or candidate.is_symlink():
            return _required_magnum_site_file(
                site,
                candidate.name,
                description=(
                    f"the current interpreter ABI extension for {module_name}"
                ),
            )
    expected = ", ".join(f"{module_name}{suffix}" for suffix in EXTENSION_SUFFIXES)
    raise FileNotFoundError(
        "AVENGINE_HABITAT_MAGNUM_PYTHON_SITE must contain one current "
        f"interpreter ABI extension for {module_name}; expected one of: {expected}"
    )


def discover_magnum_python_site(explicit: str | Path | None = None) -> Path:
    configured = explicit if explicit is not None else os.environ.get(
        "AVENGINE_HABITAT_MAGNUM_PYTHON_SITE"
    )
    if not configured:
        raise FileNotFoundError(
            "Set AVENGINE_HABITAT_MAGNUM_PYTHON_SITE to an external "
            "Corrade/Magnum Python site before importing Habitat"
        )
    site = Path(configured).resolve()
    if not site.is_dir():
        raise FileNotFoundError(
            "AVENGINE_HABITAT_MAGNUM_PYTHON_SITE is not a directory: "
            f"{site}"
        )
    _required_magnum_site_file(
        site, "corrade/__init__.py", description="corrade package"
    )
    _required_magnum_site_file(site, "magnum/__init__.py", description="magnum package")
    _required_magnum_extension(site, "_corrade")
    _required_magnum_extension(site, "_magnum")
    return site


def discover_mp3d_root(
    explicit: str | Path | None = None,
    *,
    allow_environment: bool = True,
) -> Path | None:
    configured = explicit
    if configured is None and allow_environment:
        configured = os.environ.get("AVENGINE_MP3D_ROOT")
    if configured is None:
        return None
    root = Path(configured).resolve()
    if not (root / "scene_datasets").is_dir():
        raise FileNotFoundError(
            f"AVENGINE_MP3D_ROOT must contain scene_datasets: {root}"
        )
    return root


def discover_pbr_asset_root(
    explicit: str | Path | None = None,
) -> Path | None:
    """Resolve an explicitly selected non-Git PBR IBL asset directory."""

    if explicit is None:
        return None
    root = Path(explicit).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"PBR asset root is not a directory: {root}")
    checkout_root = _git_checkout_ancestor(root)
    if checkout_root is not None:
        raise ValueError(
            "PBR asset root must not be inside a Git checkout: "
            f"{root} (found .git at {checkout_root})"
        )
    for relative_path in (
        PBR_BRDF_LUT_RELATIVE_PATH,
        PBR_ENVIRONMENT_MAP_RELATIVE_PATH,
        Path("license.txt"),
    ):
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                "PBR asset root must keep every required file below its root: "
                f"{root / relative_path}"
            ) from error
        if not candidate.is_file():
            raise FileNotFoundError(
                f"PBR asset root is missing {relative_path.as_posix()}: {root}"
            )
    return root


def _activate_runtime_prefix(
    prefix: Path, *, magnum_python_site: Path | None = None
) -> None:
    activated_roots = [prefix.resolve()]
    if magnum_python_site is not None:
        activated_roots.append(magnum_python_site.resolve())
    activated_root_strings = [str(root) for root in activated_roots]

    def is_activated_root(entry: object) -> bool:
        if not isinstance(entry, str):
            return False
        try:
            return Path(entry).resolve() in activated_roots
        except OSError:
            return entry in activated_root_strings

    sys.path[:] = [
        *activated_root_strings,
        *(entry for entry in sys.path if not is_activated_root(entry)),
    ]


def _is_editable_habitat_sim_meta_finder(finder: object) -> bool:
    finder_class = finder if isinstance(finder, type) else type(finder)
    class_module = getattr(finder_class, "__module__", "")
    if not isinstance(class_module, str):
        return False
    normalized_module = class_module.casefold()
    return "editable" in normalized_module and "habitat_sim" in normalized_module


def _remove_editable_habitat_sim_meta_finders() -> None:
    """Keep an installed-prefix import from being redirected to an editable checkout."""
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if not _is_editable_habitat_sim_meta_finder(finder)
    ]


def _originless_native_habitat_child_parent(
    module: Any, *, module_name: str
) -> tuple[str, Any] | None:
    parent_name, separator, child_name = module_name.rpartition(".")
    if not separator or not child_name:
        return None
    parent = sys.modules.get(parent_name)
    if parent is None:
        return None
    parent_origin = getattr(parent, "__file__", None)
    if not isinstance(parent_origin, (str, os.PathLike)):
        return None
    if not any(str(parent_origin).endswith(suffix) for suffix in EXTENSION_SUFFIXES):
        return None
    try:
        parent_child = vars(parent).get(child_name)
    except TypeError:
        return None
    if parent_child is not module:
        return None
    return parent_name, parent


def _habitat_sim_module_origin_under(
    module: Any, prefix: Path, *, module_name: str
) -> Path:
    origin = getattr(module, "__file__", None)
    if not origin:
        module_spec = getattr(module, "__spec__", None)
        origin = getattr(module_spec, "origin", None)
    if not origin:
        native_parent = _originless_native_habitat_child_parent(
            module, module_name=module_name
        )
        if native_parent is not None:
            parent_name, parent = native_parent
            return _habitat_sim_module_origin_under(
                parent, prefix, module_name=parent_name
            )
    if not isinstance(origin, (str, os.PathLike)):
        raise RuntimeError(
            "Loaded Habitat module has no filesystem origin to validate against "
            f"--runtime-prefix: {module_name}"
        )
    resolved = Path(origin).resolve()
    try:
        resolved.relative_to(prefix)
    except ValueError as error:
        raise RuntimeError(
            "Loaded Habitat module is outside the required --runtime-prefix "
            f"{prefix}: {module_name} -> {resolved}"
        ) from error
    if not resolved.is_file():
        raise FileNotFoundError(
            "Loaded Habitat module file is missing under --runtime-prefix "
            f"{prefix}: {module_name} -> {resolved}"
        )
    return resolved


def _validate_loaded_habitat_sim_origins(prefix: Path) -> None:
    for module_name, module in tuple(sys.modules.items()):
        if module_name == "habitat_sim" or module_name.startswith("habitat_sim."):
            _habitat_sim_module_origin_under(
                module, prefix, module_name=module_name
            )


def _module_file_under(module: Any, root: Path, *, module_name: str) -> Path:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError(
            f"Imported {module_name} has no file to validate against {root}"
        )
    resolved = Path(module_file).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            f"Imported {module_name} is outside the required external site {root}: "
            f"{resolved}"
        ) from error
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Imported {module_name} file is missing under {root}: {resolved}"
        )
    return resolved


def _validate_preloaded_magnum_python_origins(site: Path) -> None:
    for module_name in ("corrade", "magnum", "_corrade", "_magnum"):
        module = sys.modules.get(module_name)
        if module is not None:
            _module_file_under(module, site, module_name=module_name)


def _validate_magnum_python_origins(site: Path, magnum: Any) -> None:
    import _corrade
    import _magnum
    import corrade

    modules = {
        "corrade": corrade,
        "magnum": magnum,
        "_corrade": _corrade,
        "_magnum": _magnum,
    }
    for module_name, module in modules.items():
        _module_file_under(module, site, module_name=module_name)


def _prepare_installed_habitat_import(
    prefix: Path, *, magnum_python_site: str | Path | None = None
) -> _PreparedInstalledHabitatImport:
    """Isolate and activate an installed runtime without importing Habitat."""
    _remove_editable_habitat_sim_meta_finders()
    _validate_loaded_habitat_sim_origins(prefix)
    selected_magnum_site = (
        discover_magnum_python_site()
        if magnum_python_site is None
        else discover_magnum_python_site(magnum_python_site)
    )
    _activate_runtime_prefix(prefix, magnum_python_site=selected_magnum_site)
    _validate_preloaded_magnum_python_origins(selected_magnum_site)
    return _PreparedInstalledHabitatImport(
        prefix=prefix,
        magnum_python_site=selected_magnum_site,
    )


def _import_prepared_installed_habitat(
    prepared: _PreparedInstalledHabitatImport,
) -> tuple[Any, Any, Any, Any]:
    """Import Habitat after process-local native dependencies are selected."""
    try:
        imported = _import_habitat()
    except (ImportError, OSError) as error:
        if _RLR_LIBRARY_BASENAME in str(error):
            raise RuntimeError(
                "An adapter-linked installed Habitat prefix built without an "
                "RLR RUNPATH requires explicit rlr_sdk_root before import; use "
                "AVENGINE_HABITAT_BUILD_RLR_ADAPTER=OFF for visual-only "
                "M1/M2/M5 runtimes"
            ) from error
        raise
    _validate_loaded_habitat_sim_origins(prepared.prefix)
    _validate_magnum_python_origins(
        prepared.magnum_python_site,
        imported[2],
    )
    return imported


def _habitat_binding_is_loaded() -> bool:
    return any(
        module_name == _HABITAT_BINDING_MODULE_NAME
        or module_name.startswith(_HABITAT_BINDING_MODULE_NAME + ".")
        for module_name in sys.modules
    )


def _import_prepared_installed_habitat_dependencies(
    prepared: _PreparedInstalledHabitatImport,
) -> tuple[Any, Any]:
    """Load numerical/Magnum dependencies before process-global RLR symbols."""

    import quaternion as qt
    import magnum as mn

    _validate_magnum_python_origins(prepared.magnum_python_site, mn)
    return qt, mn


def _import_prepared_installed_habitat_with_rlr(
    prepared: _PreparedInstalledHabitatImport,
    *,
    rlr_sdk_root: str | Path,
) -> tuple[tuple[Any, Any, Any, Any], Any]:
    """Select one explicit external RLR SDK before importing the binding."""

    from avengine.backends.rlr import sdk as rlr_sdk_module

    # RLR exports process-global C++ symbols. Import quaternion/llvmlite and
    # Magnum first so their one-time native initialization cannot bind against
    # the RLR SDK's C++ runtime symbols.
    _import_prepared_installed_habitat_dependencies(prepared)

    sdk = rlr_sdk_module.discover_external_rlr_sdk(rlr_sdk_root)
    # Repeats may reuse an imported binding only when its process mapping is
    # already the exact newly declared SDK. A mismatch stops before another
    # process-global CDLL load can make selection ambiguous.
    if _habitat_binding_is_loaded():
        rlr_sdk_module.validate_loaded_external_rlr_sdk(sdk)
    rlr_sdk_module.preload_external_rlr_sdk(sdk)
    imported = _import_prepared_installed_habitat(prepared)
    rlr_sdk_module.validate_loaded_external_rlr_sdk(sdk)
    return imported, sdk


def _import_installed_habitat(
    prefix: Path, *, magnum_python_site: str | Path | None = None
) -> tuple[Any, Any, Any, Any]:
    """Compose installed-runtime preparation and import for existing M1 callers."""

    prepared = _prepare_installed_habitat_import(
        prefix, magnum_python_site=magnum_python_site
    )
    return _import_prepared_installed_habitat(prepared)


def _installed_runtime_paths(
    prefix: Path, habitat_sim: Any, habitat_sim_bindings: Any
) -> tuple[Path, Path, Path]:
    module_path = Path(habitat_sim.__file__).resolve()
    binding_path = Path(habitat_sim_bindings.__file__).resolve()
    physics_path = (prefix / "config" / "default.physics_config.json").resolve()
    try:
        module_path.relative_to(prefix)
        binding_path.relative_to(prefix)
        physics_path.relative_to(prefix)
    except ValueError as error:
        raise RuntimeError(
            "Installed Habitat module, native binding, or physics config is not "
            "from --runtime-prefix"
        ) from error
    if not module_path.is_file() or not binding_path.is_file() or not physics_path.is_file():
        raise FileNotFoundError(
            "Installed Habitat runtime must provide module, native binding, and "
            f"config/default.physics_config.json under {prefix}"
        )
    return module_path, binding_path, physics_path


def prepare_installed_habitat_runtime(
    *,
    runtime_prefix: str | Path | None = None,
    runtime_root: str | Path | None = None,
    mp3d_root: str | Path | None = None,
    pbr_asset_root: str | Path | None = None,
    magnum_python_site: str | Path | None = None,
    rlr_sdk_root: str | Path | None = None,
    allow_mp3d_environment: bool = True,
) -> InstalledHabitatRuntime:
    """Activate one non-Git installed prefix for a new Habitat writer.

    The runtime library, binding, and physics config live in ``prefix``. MP3D
    assets remain separately supplied through ``mp3d_root`` and Magnum remains
    an explicit external Python site. Optional PBR IBL images come only from
    ``pbr_asset_root``. No sibling checkout fallback is involved.
    """

    if rlr_sdk_root is not None and not str(rlr_sdk_root).strip():
        raise ValueError("rlr_sdk_root must be a non-empty explicit path")
    prefix = resolve_installed_runtime_prefix(
        runtime_prefix, runtime_root=runtime_root
    )
    selected_pbr_asset_root = discover_pbr_asset_root(pbr_asset_root)
    selected_magnum_site = discover_magnum_python_site(magnum_python_site)
    selected_mp3d_root = (
        discover_mp3d_root(mp3d_root)
        if allow_mp3d_environment
        else discover_mp3d_root(mp3d_root, allow_environment=False)
    )
    prepared = _prepare_installed_habitat_import(
        prefix, magnum_python_site=selected_magnum_site
    )
    if rlr_sdk_root is None:
        imported = _import_prepared_installed_habitat(prepared)
    else:
        imported, _sdk = _import_prepared_installed_habitat_with_rlr(
            prepared,
            rlr_sdk_root=rlr_sdk_root,
        )
    qt, habitat_sim, mn, quat_to_coeffs = imported
    from habitat_sim._ext import habitat_sim_bindings

    _module_path, _binding_path, physics_config_path = _installed_runtime_paths(
        prefix, habitat_sim, habitat_sim_bindings
    )
    return InstalledHabitatRuntime(
        prefix=prefix,
        mp3d_root=selected_mp3d_root,
        pbr_asset_root=selected_pbr_asset_root,
        magnum_python_site=selected_magnum_site,
        physics_config_path=physics_config_path,
        quaternion=qt,
        habitat_sim=habitat_sim,
        magnum=mn,
        quat_to_coeffs=quat_to_coeffs,
    )


def _git_value(repository: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _runtime_lock_commit(repository_root: Path) -> str | None:
    try:
        lock_path = resolve_runtime_profile(repository_root, "m1")
    except RuntimeLockError:
        return None
    text = lock_path.read_text(encoding="utf-8")
    match = re.search(
        r"^habitat_runtime:\s*$.*?^\s+fork_governance_commit:\s+([0-9a-f]{40})\s*$",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    return match.group(1) if match else None


def _import_habitat() -> tuple[Any, Any, Any, Any]:
    # The pinned audio-enabled build aborts if habitat_sim is imported before
    # numpy-quaternion. Keep this order local and explicit.
    import quaternion as qt

    import habitat_sim
    import magnum as mn
    from habitat_sim.utils.common import quat_to_coeffs

    return qt, habitat_sim, mn, quat_to_coeffs


def _environment_for_paths(
    runtime_root: Path | None = None,
    *,
    mp3d_root: Path | None = None,
) -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("AVENGINE_HABITAT_RUNTIME_ROOT", None)
    environment.pop("AVENGINE_MP3D_ROOT", None)
    if runtime_root is not None:
        environment["AVENGINE_HABITAT_RUNTIME_ROOT"] = str(runtime_root)
    if mp3d_root is not None:
        environment["AVENGINE_MP3D_ROOT"] = str(mp3d_root)
    return environment


def _resolved_scene(
    inputs: ValidatedM1Inputs,
    runtime_root: Path | None,
    *,
    mp3d_root: Path | None = None,
) -> dict[str, Any]:
    scene = inputs.room["scene"]
    environment = _environment_for_paths(runtime_root, mp3d_root=mp3d_root)
    manifest_dir = inputs.room_path.parent

    dataset_raw = scene["dataset_config_path"]
    dataset_config: str | Path
    if dataset_raw == "default":
        dataset_config = "default"
    else:
        dataset_config = resolve_declared_path(
            dataset_raw, manifest_dir=manifest_dir, environment=environment
        )

    if scene["scene_id_kind"] == "path":
        scene_id: str | Path = resolve_declared_path(
            scene["scene_id"], manifest_dir=manifest_dir, environment=environment
        )
    else:
        scene_id = scene["scene_id"]

    navmesh = None
    if scene.get("navmesh_path"):
        navmesh = resolve_declared_path(
            scene["navmesh_path"],
            manifest_dir=manifest_dir,
            environment=environment,
        )
    return {
        "dataset_config": dataset_config,
        "scene_id": scene_id,
        "navmesh": navmesh,
        "navmesh_policy": scene["navmesh_policy"],
        "load_semantic_mesh": bool(scene.get("load_semantic_mesh", False)),
        "enable_physics": bool(scene.get("enable_physics", False)),
    }


def _resolved_assets(
    inputs: ValidatedM1Inputs,
    runtime_root: Path | None,
    *,
    mp3d_root: Path | None = None,
) -> list[dict[str, Any]]:
    environment = _environment_for_paths(runtime_root, mp3d_root=mp3d_root)
    records: list[dict[str, Any]] = []
    for asset in inputs.room["assets"]:
        path = resolve_declared_path(
            asset["path"],
            manifest_dir=inputs.room_path.parent,
            environment=environment,
        )
        record = {
            "role": asset["role"],
            "declared_path": asset["path"],
            "resolved_path": str(path),
            "exists": path.is_file(),
        }
        if path.is_file():
            record.update(
                {
                    "byte_size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        records.append(record)
    return records


def _asset_by_role(records: list[dict[str, Any]], role: str) -> dict[str, Any] | None:
    return next((record for record in records if record.get("role") == role), None)


def _positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _ue_project_asset_package_closure(
    report: dict[str, Any], source_root: Path | None
) -> tuple[bool, dict[str, Any]]:
    records = report.get("selected_project_asset_packages")
    measured: dict[str, Any] = {
        "record_count": len(records) if isinstance(records, list) else None,
        "declared_count": report.get("selected_project_asset_package_count"),
        "errors": [],
    }
    errors: list[str] = measured["errors"]
    if source_root is None or not source_root.is_dir():
        errors.append("SPEAR source root is unavailable")
        return False, measured
    if not isinstance(records, list) or not records:
        errors.append("selected project package closure is missing")
        return False, measured
    if report.get("selected_project_asset_package_count") != len(records):
        errors.append("selected project package count differs from records")
    tracked_result = subprocess.run(
        ["git", "-C", str(source_root), "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    if tracked_result.returncode != 0:
        errors.append("unable to enumerate SPEAR tracked files")
        return False, measured
    tracked = {
        value.decode("utf-8") for value in tracked_result.stdout.split(b"\0") if value
    }
    actor_project_paths: set[str] = set()
    actor_engine_paths: set[str] = set()
    for actor in report.get("actors", []):
        for component in actor.get("static_mesh_components", []):
            for value in [
                component.get("static_mesh_asset"),
                *component.get("material_assets", []),
            ]:
                if not isinstance(value, str):
                    continue
                if value.startswith("/Game/"):
                    actor_project_paths.add(value)
                elif value.startswith("/Engine/"):
                    actor_engine_paths.add(value)
    recorded_paths: set[str] = set()
    package_names: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"package record {index} is not an object")
            continue
        package_name = record.get("package_name")
        if not isinstance(package_name, str) or not package_name.startswith("/Game/"):
            errors.append(f"package record {index} has an invalid package name")
            continue
        if package_name in package_names:
            errors.append(f"duplicate package record: {package_name}")
        package_names.add(package_name)
        expected_relative = (
            Path("cpp/unreal_projects/SpearSim/Content")
            / f"{package_name.removeprefix('/Game/')}.uasset"
        ).as_posix()
        relative = record.get("repository_relative_path")
        if relative != expected_relative or relative not in tracked:
            errors.append(
                f"package path is not the tracked expected file: {package_name}"
            )
            continue
        path = (source_root / relative).resolve()
        try:
            path.relative_to(source_root)
        except ValueError:
            errors.append(f"package path escapes SPEAR: {package_name}")
            continue
        if (
            not path.is_file()
            or record.get("resolved_path") != str(path)
            or record.get("git_tracked") is not True
            or record.get("byte_size") != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
        ):
            errors.append(f"package bytes or tracking changed: {package_name}")
        object_paths = record.get("asset_object_paths")
        if not isinstance(object_paths, list) or not object_paths:
            errors.append(f"package has no object paths: {package_name}")
            continue
        if not all(
            isinstance(value, str) and value.split(".", 1)[0] == package_name
            for value in object_paths
        ):
            errors.append(f"package object path mismatch: {package_name}")
        recorded_paths.update(value for value in object_paths if isinstance(value, str))
    if recorded_paths != actor_project_paths:
        errors.append("selected /Game actor assets differ from the package closure")
    if sorted(actor_engine_paths) != report.get("selected_engine_asset_references"):
        errors.append("selected /Engine asset reference set changed")
    measured.update(
        {
            "selected_project_object_count": len(actor_project_paths),
            "recorded_project_object_count": len(recorded_paths),
            "selected_engine_reference_count": len(actor_engine_paths),
        }
    )
    return not errors, measured


def _load_record_json(
    records: list[dict[str, Any]], role: str
) -> tuple[dict[str, Any] | None, str | None]:
    record = _asset_by_role(records, role)
    if record is None:
        return None, f"Missing asset role: {role}"
    try:
        value = load_json(record["resolved_path"])
    except (OSError, ValueError) as error:
        return None, f"Unable to load {role}: {type(error).__name__}: {error}"
    return value, None


def _surface_provenance_check(
    inputs: ValidatedM1Inputs, asset_records: list[dict[str, Any]]
) -> dict[str, Any] | None:
    room_kind = inputs.room["room_kind"]
    render_record = _asset_by_role(asset_records, "render_surface_mesh")
    surface_audit = inputs.room.get("surface_audit", {})
    if room_kind == "blender_custom":
        report, error = _load_record_json(asset_records, "blender_build_report")
        opening_ids = sorted(
            opening["opening_id"] for opening in inputs.room.get("openings", [])
        )
        measured: dict[str, Any] = {
            "load_error": error,
            "aabb_proxy": surface_audit.get("aabb_proxy"),
            "declared_openings": opening_ids,
        }
        passed = error is None and report is not None and render_record is not None
        if report is not None and render_record is not None:
            stage_output = report.get("outputs", {}).get(
                "stages/m1_custom_room.glb", {}
            )
            measured.update(
                {
                    "schema": report.get("schema"),
                    "geometry_representation": report.get("geometry_representation"),
                    "stage_object_count": report.get("stage_object_count"),
                    "stage_triangle_count": report.get("stage_triangle_count"),
                    "report_openings": sorted(report.get("openings", [])),
                    "report_mesh_sha256": stage_output.get("sha256"),
                    "actual_mesh_sha256": render_record.get("sha256"),
                }
            )
            passed = bool(
                passed
                and report.get("schema") == "avengine_blender_room_build_report_v1"
                and report.get("geometry_representation") == "real_surface_mesh"
                and isinstance(report.get("stage_object_count"), int)
                and report["stage_object_count"] > 0
                and isinstance(report.get("stage_triangle_count"), int)
                and report["stage_triangle_count"] > 0
                and sorted(report.get("openings", [])) == opening_ids
                and stage_output.get("sha256") == render_record.get("sha256")
                and surface_audit.get("aabb_proxy") is False
            )
        return make_check(
            "blender_authored_surface_provenance",
            "pass" if passed else "fail",
            measured=measured,
            threshold={
                "tracked_build_report_matches_mesh": True,
                "modeled_openings_match": True,
                "aabb_proxy": False,
            },
            failure_reason=None
            if passed
            else "Blender authored-surface provenance did not validate",
        )

    if room_kind != "legacy_ue_real_surface_export":
        return None

    ue_report, ue_error = _load_record_json(asset_records, "ue_export_manifest")
    mesh_report, mesh_error = _load_record_json(
        asset_records, "real_surface_mesh_audit"
    )
    source_map_record = _asset_by_role(asset_records, "legacy_source_map_package")
    provenance = inputs.room.get("provenance", {})
    source_root_raw = provenance.get("source_repository_root")
    source_root = (
        Path(source_root_raw).resolve()
        if isinstance(source_root_raw, str) and source_root_raw
        else None
    )
    expected_source_project = (
        (source_root / "cpp" / "unreal_projects" / "SpearSim").resolve()
        if source_root is not None
        else None
    )
    current_source_commit = (
        _git_value(source_root, "rev-parse", "HEAD")
        if source_root is not None and source_root.is_dir()
        else None
    )
    current_source_tracked_status = (
        _git_value(
            source_root,
            "status",
            "--porcelain",
            "--untracked-files=no",
        )
        if source_root is not None and source_root.is_dir()
        else None
    )
    measured = {
        "ue_manifest_error": ue_error,
        "mesh_audit_error": mesh_error,
        "manifest_surface_audit": surface_audit,
        "source_repository_root": source_root_raw,
        "current_source_commit": current_source_commit,
        "current_source_tracked_status": current_source_tracked_status,
        "source_map_record": source_map_record,
    }
    passed = (
        ue_error is None
        and mesh_error is None
        and ue_report is not None
        and mesh_report is not None
        and render_record is not None
        and source_map_record is not None
    )
    if ue_report is not None and mesh_report is not None and render_record is not None:
        export_output = ue_report.get("output", {})
        export_messages = ue_report.get("export_messages", {})
        export_source_snapshot = ue_report.get("source_snapshot", {})
        export_source_snapshot_after = ue_report.get("source_snapshot_after_export", {})
        dirty_packages = ue_report.get("dirty_packages", {})
        gate = mesh_report.get("real_surface_gate", {})
        indicators = mesh_report.get("aabb_proxy_indicators", {})
        package_closure_passed, package_closure = _ue_project_asset_package_closure(
            ue_report, source_root
        )
        measured.update(
            {
                "ue_schema": ue_report.get("schema"),
                "loaded_editor_world": ue_report.get("loaded_editor_world"),
                "engine_version": ue_report.get("engine_version"),
                "gltf_exporter_plugin": ue_report.get("gltf_exporter_plugin"),
                "geometry_source": ue_report.get("geometry_source"),
                "uses_actor_bounds_as_geometry": ue_report.get(
                    "uses_actor_bounds_as_geometry"
                ),
                "selected_actor_count": ue_report.get("selected_actor_count"),
                "static_mesh_component_count": ue_report.get(
                    "static_mesh_component_count"
                ),
                "unique_static_mesh_asset_count": ue_report.get(
                    "unique_static_mesh_asset_count"
                ),
                "option_warnings": ue_report.get("option_warnings"),
                "export_errors": export_messages.get("errors"),
                "export_source_snapshot": export_source_snapshot,
                "export_source_snapshot_after": export_source_snapshot_after,
                "actual_project_dir": ue_report.get("actual_project_dir"),
                "dirty_packages": dirty_packages,
                "triangles": mesh_report.get("triangles"),
                "meshes": mesh_report.get("meshes"),
                "materials": mesh_report.get("materials"),
                "mesh_gate": gate,
                "aabb_indicators": indicators,
                "actual_mesh_sha256": render_record.get("sha256"),
                "ue_mesh_sha256": export_output.get("sha256"),
                "audit_mesh_sha256": mesh_report.get("sha256"),
                "selected_project_asset_package_closure": package_closure,
            }
        )
        triangles = mesh_report.get("triangles")
        passed = bool(
            passed
            and ue_report.get("schema") == "avengine_legacy_ue_apartment_export_v1"
            and ue_report.get("status") == "pass"
            and ue_report.get("source_map_asset")
            == "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000"
            and export_source_snapshot.get("schema")
            == "avengine_spear_source_snapshot_v1"
            and export_source_snapshot.get("capture_phase") == "before_ue_gltf_export"
            and export_source_snapshot_after
            == {
                **export_source_snapshot,
                "capture_phase": "after_ue_gltf_export",
            }
            and export_source_snapshot.get("map_asset")
            == ue_report.get("source_map_asset")
            and str(ue_report.get("loaded_editor_world", "")).startswith(
                "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000."
            )
            and isinstance(ue_report.get("engine_version"), str)
            and ue_report["engine_version"].startswith("5.5.")
            and ue_report.get("gltf_exporter_plugin", {}).get("version_name") == "1.3.1"
            and package_closure_passed
            and ue_report.get("geometry_source") == "UE StaticMesh render data LOD0"
            and ue_report.get("uses_actor_bounds_as_geometry") is False
            and ue_report.get("option_warnings") == []
            and export_messages.get("errors") == []
            and _positive_integer(ue_report.get("selected_actor_count"))
            and _positive_integer(ue_report.get("static_mesh_component_count"))
            and _positive_integer(ue_report.get("unique_static_mesh_asset_count"))
            and mesh_report.get("schema") == "avengine_real_surface_mesh_audit_v1"
            and gate.get("status") == "pass"
            and isinstance(triangles, int)
            and triangles > 252
            and _positive_integer(mesh_report.get("meshes"))
            and _positive_integer(mesh_report.get("materials"))
            and indicators.get("known_legacy_triangle_signature") is False
            and indicators.get("all_mesh_nodes_are_simple_boxes") is False
            and export_output.get("sha256") == render_record.get("sha256")
            and mesh_report.get("sha256") == render_record.get("sha256")
            and export_output.get("byte_size") == render_record.get("byte_size")
            and mesh_report.get("bytes") == render_record.get("byte_size")
            and surface_audit.get("aabb_proxy") is False
            and surface_audit.get("triangle_count") == triangles
            and surface_audit.get("mesh_sha256") == render_record.get("sha256")
            and surface_audit.get("real_surface_gate_status") == "pass"
            and export_source_snapshot.get("repository_root") == source_root_raw
            and ue_report.get("actual_project_dir")
            == export_source_snapshot.get("actual_project_dir")
            == str(expected_source_project)
            and dirty_packages.get("before_reload") == {"content": [], "maps": []}
            and dirty_packages.get("after_reload") == {"content": [], "maps": []}
            and dirty_packages.get("after_export") == {"content": [], "maps": []}
            and export_source_snapshot.get("commit")
            == provenance.get("source_revision")
            == current_source_commit
            and export_source_snapshot.get("tracked_worktree_dirty") is False
            and provenance.get("source_repository_tracked_dirty") is False
            and current_source_tracked_status == ""
            and export_source_snapshot.get("map_package_path")
            == provenance.get("source_map_package_path")
            == source_map_record.get("resolved_path")
            and export_source_snapshot.get("map_package_sha256")
            == provenance.get("source_map_package_sha256")
            == source_map_record.get("sha256")
            and provenance.get("exported_scene_sha256") == render_record.get("sha256")
        )
    return make_check(
        "legacy_real_surface_provenance",
        "pass" if passed else "fail",
        measured=measured,
        threshold={
            "source": "UE StaticMesh render data LOD0",
            "minimum_triangles": 253,
            "known_252_triangle_proxy": False,
            "all_simple_boxes": False,
            "hash_chain_matches": True,
            "pre_export_clean_spear_commit_and_map_package_hash_match": True,
        },
        failure_reason=None
        if passed
        else "Legacy real-surface export/audit hash chain did not validate",
    )


def _make_configuration(
    inputs: ValidatedM1Inputs,
    runtime_root: Path | None,
    output_dir: Path,
    *,
    mp3d_root: Path | None = None,
    include_audio_sensor: bool = True,
    physics_config_path: Path | None = None,
) -> tuple[Any, dict[str, str], str, dict[str, Any]]:
    qt, habitat_sim, mn, _ = _import_habitat()
    del qt
    resolved = _resolved_scene(inputs, runtime_root, mp3d_root=mp3d_root)
    rig = inputs.request["primary_camera_rig"]
    calibration = rig["shared_calibration"]
    height, width = calibration["resolution_hw"]
    local = calibration["rig_from_sensor"]
    local_position = mn.Vector3(local["translation_m"])
    local_orientation = mn.Vector3(0.0, 0.0, 0.0)

    modality_to_uuid = {
        item["modality"]: item["sensor_uuid"] for item in rig["modalities"]
    }
    sensor_specs: list[Any] = []
    for modality in ("rgb", "depth", "semantic"):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = modality_to_uuid[modality]
        spec.sensor_type = getattr(
            habitat_sim.SensorType, VISUAL_SENSOR_TYPES[modality]
        )
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        spec.resolution = mn.Vector2i([height, width])
        spec.position = local_position
        spec.orientation = local_orientation
        spec.hfov = float(calibration["hfov_degrees"])
        spec.near = float(calibration["near_m"])
        spec.far = float(calibration["far_m"])
        spec.gpu2gpu_transfer = False
        spec.noise_model = "None"
        if modality != "rgb":
            spec.channels = 1
        if modality == "semantic" and hasattr(spec, "semantic_target"):
            from habitat_sim._ext import habitat_sim_bindings

            spec.semantic_target = habitat_sim_bindings.SemanticSensorTarget.SEMANTIC_ID
        sensor_specs.append(spec)

    listener = inputs.request["listener"]
    if include_audio_sensor:
        audio_spec = habitat_sim.AudioSensorSpec()
        audio_spec.uuid = listener["listener_id"]
        audio_spec.position = local_position
        audio_spec.orientation = local_orientation
        # Kept for downstream M2/M5 callers that still require an audio sensor.
        audio_spec.outputDirectory = str(output_dir / "audio_not_run")
        sensor_specs.append(audio_spec)

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(resolved["scene_id"])
    sim_cfg.scene_dataset_config_file = str(resolved["dataset_config"])
    sim_cfg.load_semantic_mesh = resolved["load_semantic_mesh"]
    sim_cfg.enable_physics = resolved["enable_physics"]
    if physics_config_path is not None:
        sim_cfg.physics_config_file = str(physics_config_path.resolve())
    sim_cfg.random_seed = int(inputs.request["seed"])
    sim_cfg.gpu_device_id = 0

    agent_cfg = habitat_sim.AgentConfiguration()
    navigation = inputs.room.get("navigation", {})
    agent_cfg.height = float(navigation.get("agent_height_m", 1.5))
    agent_cfg.radius = float(navigation.get("agent_radius_m", 0.2))
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {}

    nav_settings = habitat_sim.NavMeshSettings()
    nav_settings.set_defaults()
    nav_settings.agent_height = agent_cfg.height
    nav_settings.agent_radius = agent_cfg.radius
    nav_settings.include_static_objects = bool(
        navigation.get("include_static_objects", False)
    )
    sim_cfg.navmesh_settings = nav_settings
    return (
        habitat_sim.Configuration(sim_cfg, [agent_cfg]),
        modality_to_uuid,
        listener["listener_id"],
        resolved,
    )


def _numpy_quaternion(xyzw: list[float], qt: Any) -> Any:
    x, y, z, w = xyzw
    return qt.quaternion(w, x, y, z)


def _pose_dict(pose: Any, quat_to_coeffs: Any) -> dict[str, list[float]]:
    return {
        "translation_m": np.asarray(pose.position, dtype=np.float64).tolist(),
        "rotation_xyzw": np.asarray(
            quat_to_coeffs(pose.rotation), dtype=np.float64
        ).tolist(),
    }


def _state_snapshot(
    sim: Any,
    agent: Any,
    sensor_uuids: list[str],
    quat_to_coeffs: Any,
) -> dict[str, Any]:
    state = agent.get_state()
    return {
        "world_time_seconds": float(sim.get_world_time()),
        "agent": _pose_dict(state, quat_to_coeffs),
        "sensors": {
            uuid: _pose_dict(state.sensor_states[uuid], quat_to_coeffs)
            for uuid in sorted(sensor_uuids)
        },
    }


def _logical_listener_pose(snapshot: dict[str, Any], listener: dict[str, Any]) -> dict[str, Any]:
    """Derive the M1 logical listener from the actual agent/rig state."""

    return compose_transforms(snapshot["agent"], listener["rig_from_listener"])


def _repeat_hashes(
    captures: list[dict[str, np.ndarray]], modality_to_uuid: dict[str, str]
) -> list[dict[str, str]]:
    return [
        {
            modality: array_sha256(uuid, capture[uuid])
            for modality, uuid in sorted(modality_to_uuid.items())
        }
        for capture in captures
    ]


def _connectivity_checks(
    sim: Any, inputs: ValidatedM1Inputs
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _, habitat_sim, _, _ = _import_habitat()
    reports: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    pairs = inputs.room.get("connectivity_pairs", [])
    if not pairs:
        return reports, checks
    if not sim.pathfinder.is_loaded:
        for pair in pairs:
            checks.append(
                make_check(
                    f"connectivity_{pair['pair_id']}",
                    "blocked",
                    measured={"pathfinder_loaded": False},
                    threshold={"path_found": True},
                    failure_reason="No navmesh is loaded",
                )
            )
        return reports, checks

    for pair in pairs:
        start = np.asarray(pair["start_m"], dtype=np.float32)
        end = np.asarray(pair["end_m"], dtype=np.float32)
        snapped_start = np.asarray(sim.pathfinder.snap_point(start), dtype=np.float64)
        snapped_end = np.asarray(sim.pathfinder.snap_point(end), dtype=np.float64)
        start_snap_distance = float(
            np.linalg.norm(snapped_start - start.astype(np.float64))
        )
        end_snap_distance = float(np.linalg.norm(snapped_end - end.astype(np.float64)))
        maximum_snap_distance = 0.30
        query = habitat_sim.ShortestPath()
        query.requested_start = snapped_start
        query.requested_end = snapped_end
        found = bool(sim.pathfinder.find_path(query))
        report = {
            "pair_id": pair["pair_id"],
            "requested_start_m": start.astype(np.float64).tolist(),
            "requested_end_m": end.astype(np.float64).tolist(),
            "snapped_start_m": snapped_start.tolist(),
            "snapped_end_m": snapped_end.tolist(),
            "start_snap_distance_m": start_snap_distance,
            "end_snap_distance_m": end_snap_distance,
            "found": found,
            "geodesic_distance_m": float(query.geodesic_distance) if found else None,
            "path_point_count": len(query.points) if found else 0,
        }
        reports.append(report)
        passed = bool(
            found
            and np.isfinite(query.geodesic_distance)
            and np.isfinite(start_snap_distance)
            and np.isfinite(end_snap_distance)
            and start_snap_distance <= maximum_snap_distance
            and end_snap_distance <= maximum_snap_distance
        )
        checks.append(
            make_check(
                f"connectivity_{pair['pair_id']}",
                "pass" if passed else "fail",
                measured=report,
                threshold={
                    "path_found": True,
                    "finite_distance": True,
                    "maximum_snap_distance_m": maximum_snap_distance,
                },
                failure_reason=None
                if passed
                else "ShortestPath did not connect the pair",
            )
        )
    return reports, checks


def _ray_checks(
    sim: Any, inputs: ValidatedM1Inputs
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _, habitat_sim, mn, _ = _import_habitat()
    reports: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for declared in inputs.room.get("ray_checks", []):
        ray = habitat_sim.geo.Ray(
            mn.Vector3(declared["origin_m"]), mn.Vector3(declared["direction"])
        )
        results = sim.cast_ray(ray, buffer_distance=0.0)
        nearest = float(results.hits[0].ray_distance) if results.has_hits() else None
        distance = float(declared["distance_m"])
        if declared["expectation"] == "clear_until_m":
            passed = nearest is None or nearest > distance
        else:
            passed = nearest is not None and nearest <= distance
        report = {
            "check_id": declared["check_id"],
            "expectation": declared["expectation"],
            "distance_m": distance,
            "nearest_hit_m": nearest,
            "passed": passed,
        }
        reports.append(report)
        checks.append(
            make_check(
                f"ray_{declared['check_id']}",
                "pass" if passed else "fail",
                measured=report,
                threshold={
                    "expectation": declared["expectation"],
                    "distance_m": distance,
                },
                failure_reason=None if passed else "Opening/control ray did not match",
            )
        )
    return reports, checks


def _save_topdown(
    sim: Any, inputs: ValidatedM1Inputs, output_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reports: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for view in inputs.request.get("qa_views", []):
        if view.get("kind") != "topdown":
            continue
        if not sim.pathfinder.is_loaded:
            checks.append(
                make_check(
                    f"qa_{view['qa_id']}",
                    "blocked",
                    measured={"pathfinder_loaded": False},
                    threshold={"artifact_written": True},
                    failure_reason="No navmesh is loaded",
                )
            )
            continue
        meters_per_pixel = float(view.get("meters_per_pixel", 0.05))
        height = float(view.get("height_m", 0.1))
        topdown = np.asarray(
            sim.pathfinder.get_topdown_view(meters_per_pixel, height), dtype=np.uint8
        )
        qa_dir = output_dir / "qa"
        qa_dir.mkdir(parents=True, exist_ok=True)
        path = qa_dir / f"{view['qa_id']}.png"
        Image.fromarray(topdown * 255, mode="L").save(path)
        report = {
            "qa_id": view["qa_id"],
            "kind": "topdown",
            "formal_view": False,
            "meters_per_pixel": meters_per_pixel,
            "height_m": height,
            "shape": list(topdown.shape),
            "navigable_pixel_count": int(np.count_nonzero(topdown)),
            "artifact": file_record(path, relative_to=output_dir),
        }
        reports.append(report)
        passed = topdown.size > 0 and report["navigable_pixel_count"] > 0
        checks.append(
            make_check(
                f"qa_{view['qa_id']}",
                "pass" if passed else "fail",
                measured={
                    "shape": list(topdown.shape),
                    "formal_view": False,
                    "navigable_pixel_count": report["navigable_pixel_count"],
                },
                threshold={
                    "nonempty": True,
                    "minimum_navigable_pixel_count": 1,
                    "formal_view": False,
                },
                artifact=report["artifact"]["path"],
                failure_reason=None
                if passed
                else "Topdown QA artifact has no navigable pixels",
            )
        )
    return reports, checks


def _source_roundtrip(inputs: ValidatedM1Inputs) -> tuple[list[dict[str, Any]], float]:
    world_from_rig = inputs.request["primary_camera_rig"]["world_from_rig"]
    rig_from_world = invert_transform(world_from_rig)
    reports: list[dict[str, Any]] = []
    maximum = 0.0
    for source in inputs.request["sources"]:
        world_from_source = source["world_from_source"]
        rig_from_source = compose_transforms(rig_from_world, world_from_source)
        recovered, error = round_trip_via_parent(world_from_rig, world_from_source)
        maximum = max(maximum, error)
        reports.append(
            {
                "source_id": source["source_id"],
                "world_from_source": world_from_source,
                "rig_from_source": rig_from_source,
                "recovered_world_from_source": recovered,
                "roundtrip_max_error": error,
            }
        )
    return reports, maximum


def _independent_process_repeatability_check(
    *,
    reference_path: str | Path | None,
    inputs: ValidatedM1Inputs,
    observation_records: dict[str, dict[str, Any]],
    state_hash: str,
    runtime_prefix: Path,
    mp3d_root: Path | None,
    habitat_module_path: Path,
    native_binding_path: Path,
    physics_config_path: Path,
    asset_records: list[dict[str, Any]],
    avengine_commit: str | None,
    repository_clean: bool,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if reference_path is None:
        return (
            make_check(
                "independent_process_repeatability",
                "not_run",
                measured={"reference_evidence": None},
                threshold={"independent_reference_matches": True},
                failure_reason=(
                    "Run capture once, then rerun in a fresh process with "
                    "--reference-evidence pointing to the first evidence.json"
                ),
            ),
            None,
        )

    resolved = Path(reference_path).resolve()
    measured: dict[str, Any] = {"reference_evidence": str(resolved)}
    try:
        reference_verification_status, reference_verification_checks = (
            verify_evidence_artifacts(resolved, _allow_reference=True)
        )
        reference = load_json(resolved)
        expected_hash = reference.get("evidence_content_sha256")
        actual_hash = canonical_json_sha256(
            {
                key: value
                for key, value in reference.items()
                if key != "evidence_content_sha256"
            }
        )
        reference_observation_hashes = {
            modality: reference.get("observations", {})
            .get(modality, {})
            .get("raw_array_sha256")
            for modality in ("rgb", "depth", "semantic")
        }
        current_observation_hashes = {
            modality: observation_records[modality]["raw_array_sha256"]
            for modality in ("rgb", "depth", "semantic")
        }
        reference_assets = {
            record.get("role"): (record.get("sha256"), record.get("byte_size"))
            for record in reference.get("scene_assets", [])
            if isinstance(record, dict)
        }
        current_assets = {
            record.get("role"): (record.get("sha256"), record.get("byte_size"))
            for record in asset_records
        }
        comparisons = {
            "reference_fully_verified": reference_verification_status == "pass",
            "content_hash_valid": expected_hash == actual_hash,
            "schema_matches": reference.get("schema") == EVIDENCE_SCHEMA_V2,
            "evidence_kind_matches": reference.get("evidence_kind")
            == "completed_capture",
            "reference_is_first_run": reference.get("overall_status") == "not_run"
            and reference.get("independent_reference") is None,
            "room_id_matches": reference.get("room_id") == inputs.room["room_id"],
            "request_id_matches": reference.get("request_id")
            == inputs.request["request_id"],
            "room_manifest_hash_matches": reference.get("room_manifest", {}).get(
                "sha256"
            )
            == sha256_file(inputs.room_path),
            "capture_request_hash_matches": reference.get("capture_request", {}).get(
                "sha256"
            )
            == sha256_file(inputs.request_path),
            "runtime_prefix_matches": reference.get("runtime", {}).get(
                "habitat_runtime_prefix"
            )
            == str(runtime_prefix),
            "mp3d_root_matches": reference.get("runtime", {}).get("mp3d_root")
            == (str(mp3d_root) if mp3d_root is not None else None),
            "module_path_matches": reference.get("runtime", {}).get(
                "habitat_module_path"
            )
            == str(habitat_module_path),
            "native_binding_path_matches": reference.get("runtime", {}).get(
                "native_binding_path"
            )
            == str(native_binding_path),
            "physics_config_path_matches": reference.get("runtime", {}).get(
                "physics_config_path"
            )
            == str(physics_config_path),
            "avengine_commit_matches": reference.get("runtime", {}).get(
                "avengine_commit"
            )
            == avengine_commit,
            "avengine_was_and_is_clean": reference.get("runtime", {}).get(
                "avengine_worktree_dirty"
            )
            is False
            and repository_clean,
            "scene_assets_match": reference_assets == current_assets,
            "initial_state_matches": reference.get("capture_state", {}).get(
                "before_sha256"
            )
            == state_hash,
            "raw_observation_hashes_match": reference_observation_hashes
            == current_observation_hashes,
            "fresh_process_instance": reference.get("producer_process", {}).get(
                "process_instance_id"
            )
            != PROCESS_INSTANCE_ID,
        }
        passed = all(comparisons.values())
        measured.update(
            {
                "reference_content_sha256": expected_hash,
                "reference_overall_status": reference.get("overall_status"),
                "reference_verification_status": reference_verification_status,
                "reference_verification_checks": reference_verification_checks,
                "comparisons": comparisons,
                "reference_observation_hashes": reference_observation_hashes,
                "current_observation_hashes": current_observation_hashes,
                "reference_process": reference.get("producer_process"),
                "current_process": _producer_process_identity(),
            }
        )
        if passed:
            copied_root = output_dir / "independent_reference"
            if copied_root.is_dir():
                shutil.rmtree(copied_root)
            copied_root.mkdir(parents=True, exist_ok=True)
            copied_evidence = copied_root / "evidence.json"
            shutil.copy2(resolved, copied_evidence)
            for directory_name in ("observations", "qa"):
                source_directory = resolved.parent / directory_name
                if source_directory.is_dir():
                    shutil.copytree(source_directory, copied_root / directory_name)
            reference_record: dict[str, Any] | None = {
                "path": copied_evidence.relative_to(output_dir).as_posix(),
                "evidence_content_sha256": reference.get("evidence_content_sha256"),
                "artifact": file_record(copied_evidence, relative_to=output_dir),
            }
        else:
            reference_record = None
    except (OSError, ValueError, TypeError) as error:
        reference = None
        passed = False
        reference_record = None
        measured["exception"] = f"{type(error).__name__}: {error}"

    return (
        make_check(
            "independent_process_repeatability",
            "pass" if passed else "fail",
            measured=measured,
            threshold={"all_identity_state_asset_and_observation_comparisons": True},
            failure_reason=None
            if passed
            else "Fresh-process evidence does not match the reference run",
        ),
        reference_record,
    )


def capture_m1(
    inputs: ValidatedM1Inputs,
    output_dir: str | Path,
    *,
    runtime_prefix: str | Path | None = None,
    repeat_count: int = 3,
    reference_evidence: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for stale in (
        output / "observations",
        output / "qa",
        output / "independent_reference",
    ):
        if stale.is_dir():
            shutil.rmtree(stale)
    (output / "evidence.json").unlink(missing_ok=True)
    prefix = discover_runtime_prefix(runtime_prefix)
    mp3d_root = discover_mp3d_root()
    repository_root = Path(__file__).resolve().parents[3]
    qt, habitat_sim, _, quat_to_coeffs = _import_installed_habitat(prefix)
    from habitat_sim._ext import habitat_sim_bindings

    habitat_module_path, native_binding_path, physics_config_path = _installed_runtime_paths(
        prefix, habitat_sim, habitat_sim_bindings
    )
    configuration, modality_to_uuid, listener_uuid, resolved_scene = (
        _make_configuration(
            inputs,
            None,
            output,
            mp3d_root=mp3d_root,
            include_audio_sensor=False,
            physics_config_path=physics_config_path,
        )
    )
    repeat_count = max(2, int(repeat_count))
    asset_records = _resolved_assets(inputs, None, mp3d_root=mp3d_root)
    missing_assets = [record for record in asset_records if not record["exists"]]
    if missing_assets:
        raise FileNotFoundError(
            "Required room assets are missing: "
            + ", ".join(record["declared_path"] for record in missing_assets)
        )
    scene_graph_errors = validate_scene_asset_graph(
        inputs, None, mp3d_root=mp3d_root
    )

    formal_view_passed = (
        inputs.request["primary_camera_rig"]["rig_id"] == "camera_rig_0"
        and inputs.request["primary_camera_rig"]["view_id"] == "view0"
        and set(modality_to_uuid) == set(VISUAL_SENSOR_TYPES)
    )
    checks: list[dict[str, Any]] = [
        make_check(
            "single_formal_view",
            "pass" if formal_view_passed else "fail",
            measured={
                "view_ids": [inputs.request["primary_camera_rig"]["view_id"]],
                "modality_count": len(modality_to_uuid),
            },
            threshold={
                "formal_view_count": 1,
                "modalities": sorted(VISUAL_SENSOR_TYPES),
            },
            failure_reason=None
            if formal_view_passed
            else "M1 requires camera_rig_0/view0 and exactly three shared modalities",
        )
    ]
    scene_graph_check = make_check(
        "scene_load_graph_closure",
        "pass" if not scene_graph_errors else "fail",
        measured={
            "errors": scene_graph_errors,
            "static_errors": scene_graph_errors,
            "loaded_errors": ["Simulator has not been constructed"],
            "loaded_graph": None,
        },
        threshold={
            "errors": [],
            "actual_habitat_scene_resolves_to_declared_assets": True,
        },
        failure_reason=None
        if not scene_graph_errors
        else "Habitat scene selection diverges from the declared asset closure",
    )
    checks.append(scene_graph_check)
    repository_status = _git_value(repository_root, "status", "--porcelain")
    repository_clean = repository_status == ""
    avengine_commit = _git_value(repository_root, "rev-parse", "HEAD")
    checks.append(
        make_check(
            "avengine_worktree_clean",
            "pass" if repository_clean else "fail",
            measured={"git_status": repository_status},
            threshold={"dirty": False},
            failure_reason=None
            if repository_clean
            else "AVEngine worktree is dirty; final evidence must bind a clean commit",
        )
    )
    checks.append(
        make_check(
            "runtime_binary_origin",
            "pass",
            measured={
                "habitat_module": str(habitat_module_path),
                "native_binding": str(native_binding_path),
                "physics_config": str(physics_config_path),
            },
            threshold={"all_paths_within_runtime_prefix": str(prefix)},
        )
    )
    checks.append(
        make_check(
            "scene_asset_closure",
            "pass",
            measured={
                "asset_count": len(asset_records),
                "roles": sorted(record["role"] for record in asset_records),
                "all_exist": True,
            },
            threshold={"all_declared_assets_exist_and_are_hashed": True},
        )
    )
    surface_check = _surface_provenance_check(inputs, asset_records)
    if surface_check is not None:
        checks.append(surface_check)

    rig = inputs.request["primary_camera_rig"]
    world_from_rig = rig["world_from_rig"]
    calibration = rig["shared_calibration"]
    expected_world_from_sensor = compose_transforms(
        world_from_rig, calibration["rig_from_sensor"]
    )
    state = habitat_sim.AgentState()
    state.position = np.asarray(world_from_rig["translation_m"], dtype=np.float64)
    state.rotation = _numpy_quaternion(world_from_rig["rotation_xyzw"], qt)

    visual_uuids = [modality_to_uuid[key] for key in ("rgb", "depth", "semantic")]
    expected_world_from_listener = compose_transforms(
        world_from_rig, inputs.request["listener"]["rig_from_listener"]
    )
    with habitat_sim.Simulator(configuration) as sim:
        navmesh_path = resolved_scene["navmesh"]
        declared_navmesh_loaded = False
        if navmesh_path is not None and Path(navmesh_path).is_file():
            declared_navmesh_loaded = bool(
                sim.pathfinder.load_nav_mesh(str(navmesh_path))
            )
            if not declared_navmesh_loaded:
                raise RuntimeError("Habitat failed to load the declared navmesh")

        loaded_graph_errors, loaded_graph = validate_loaded_scene_asset_graph(
            inputs,
            None,
            sim,
            declared_navmesh_loaded=declared_navmesh_loaded,
            mp3d_root=mp3d_root,
        )
        combined_scene_graph_errors = scene_graph_errors + loaded_graph_errors
        scene_graph_check.clear()
        scene_graph_check.update(
            make_check(
                "scene_load_graph_closure",
                "pass" if not combined_scene_graph_errors else "fail",
                measured={
                    "errors": combined_scene_graph_errors,
                    "static_errors": scene_graph_errors,
                    "loaded_errors": loaded_graph_errors,
                    "loaded_graph": loaded_graph,
                },
                threshold={
                    "errors": [],
                    "actual_habitat_scene_resolves_to_declared_assets": True,
                    "declared_navmesh_explicitly_loaded_and_fingerprinted": True,
                },
                failure_reason=None
                if not combined_scene_graph_errors
                else "Habitat loaded scene graph diverges from declared assets",
            )
        )

        requires_navigation_evidence = bool(
            inputs.room.get("connectivity_pairs")
            or any(
                view.get("kind") == "topdown"
                for view in inputs.request.get("qa_views", [])
            )
        )
        if requires_navigation_evidence and not sim.pathfinder.is_loaded:
            raise RuntimeError(
                "M1 navigation evidence requires a loaded navmesh; capture is blocked"
            )

        sim.seed(int(inputs.request["seed"]))
        agent = sim.initialize_agent(0, state)
        before = _state_snapshot(sim, agent, visual_uuids, quat_to_coeffs)
        before["listener_pose"] = _logical_listener_pose(
            before, inputs.request["listener"]
        )
        before_hash = canonical_json_sha256(before)

        captures: list[dict[str, np.ndarray]] = []
        wrappers = [sim.sensors[uuid] for uuid in visual_uuids]
        for _ in range(repeat_count):
            observation = sim.render_sensors(wrappers)
            captures.append(
                {
                    uuid: np.ascontiguousarray(observation[uuid]).copy()
                    for uuid in visual_uuids
                }
            )
        after = _state_snapshot(sim, agent, visual_uuids, quat_to_coeffs)
        after["listener_pose"] = _logical_listener_pose(
            after, inputs.request["listener"]
        )
        after_hash = canonical_json_sha256(after)

        observation_records = save_observations(captures[0], modality_to_uuid, output)
        repeated_hashes = _repeat_hashes(captures, modality_to_uuid)
        repeat_passed = all(item == repeated_hashes[0] for item in repeated_hashes[1:])

        state_unchanged = before_hash == after_hash
        checks.append(
            make_check(
                "capture_state_unchanged",
                "pass" if state_unchanged else "fail",
                measured={"before": before_hash, "after": after_hash},
                threshold={"hashes_equal": True, "world_time_advance_seconds": 0.0},
                failure_reason=None
                if state_unchanged
                else "Capture advanced or changed state",
            )
        )
        checks.append(
            make_check(
                "repeatability_same_process",
                "pass" if repeat_passed else "fail",
                measured=repeated_hashes,
                threshold={"all_repeats_identical": True, "repeat_count": repeat_count},
                failure_reason=None if repeat_passed else "Observation bytes changed",
            )
        )

        readback_poses = before["sensors"]
        pose_errors = {
            uuid: transform_error(readback_poses[uuid], expected_world_from_sensor)
            for uuid in visual_uuids
        }
        pose_errors[listener_uuid] = transform_error(
            before["listener_pose"], expected_world_from_listener
        )
        maximum_pose_error = max(pose_errors.values())
        alignment_passed = maximum_pose_error <= 1e-7
        checks.append(
            make_check(
                "rig_visual_listener_alignment",
                "pass" if alignment_passed else "fail",
                measured={"pose_errors": pose_errors, "maximum": maximum_pose_error},
                threshold={"maximum_transform_error": 1e-7},
                failure_reason=None
                if alignment_passed
                else "Visual modalities or listener diverged from shared rig pose",
            )
        )

        rgb_stats = observation_records["rgb"]["statistics"]
        rgb_passed = (
            rgb_stats["color_standard_deviation"] > 1.0
            and max(rgb_stats["per_channel_standard_deviation"]) > 1.0
        )
        checks.append(
            make_check(
                "rgb_nonconstant",
                "pass" if rgb_passed else "fail",
                measured=rgb_stats,
                threshold={
                    "minimum_color_standard_deviation": 1.0,
                    "minimum_one_channel_standard_deviation": 1.0,
                    "alpha_channel_excluded": True,
                },
                artifact=observation_records["rgb"]["artifact"]["path"],
                failure_reason=None if rgb_passed else "RGB observation is constant",
            )
        )

        depth_stats = observation_records["depth"]["statistics"]
        depth_max = depth_stats["maximum_finite_m"]
        depth_passed = (
            depth_stats["finite_positive_fraction"] > 0.05
            and depth_max is not None
            and depth_max <= float(calibration["far_m"]) + 1e-4
        )
        checks.append(
            make_check(
                "depth_valid",
                "pass" if depth_passed else "fail",
                measured=depth_stats,
                threshold={
                    "minimum_finite_positive_fraction": 0.05,
                    "maximum_depth_m": calibration["far_m"],
                },
                artifact=observation_records["depth"]["artifact"]["path"],
                failure_reason=None if depth_passed else "Depth observation is invalid",
            )
        )

        semantic_stats = observation_records["semantic"]["statistics"]
        declared_semantic_ids = {
            int(value)
            for value in inputs.room.get("semantics", {}).get("id_to_label", {})
            if str(value).lstrip("-").isdigit()
        }
        visible_semantic_ids = set(semantic_stats["unique_ids"])
        expected_nonzero_ids = declared_semantic_ids - {0}
        declared_marker_visible = not expected_nonzero_ids or bool(
            expected_nonzero_ids & visible_semantic_ids
        )
        semantic_passed = (
            semantic_stats["unique_id_count"] > 1
            and declared_marker_visible
            and all(value >= 0 for value in visible_semantic_ids)
        )
        checks.append(
            make_check(
                "semantic_nontrivial_raw_ids",
                "pass" if semantic_passed else "fail",
                measured={
                    **semantic_stats,
                    "declared_nonzero_ids": sorted(expected_nonzero_ids),
                    "declared_marker_visible": declared_marker_visible,
                },
                threshold={
                    "minimum_unique_id_count": 2,
                    "at_least_one_declared_nonzero_id_visible": bool(
                        expected_nonzero_ids
                    ),
                    "nonnegative_raw_ids": True,
                },
                artifact=observation_records["semantic"]["artifact"]["path"],
                failure_reason=None
                if semantic_passed
                else "Semantic IDs were trivial, invalid, or missed declared markers",
            )
        )

        source_reports, source_error = _source_roundtrip(inputs)
        source_passed = source_error <= 1e-9
        checks.append(
            make_check(
                "named_source_transform_roundtrip",
                "pass" if source_passed else "fail",
                measured={
                    "source_count": len(source_reports),
                    "maximum_transform_error": source_error,
                },
                threshold={
                    "minimum_source_count_for_canary": 2,
                    "maximum_transform_error": 1e-9,
                },
                failure_reason=None
                if source_passed and len(source_reports) >= 2
                else "Need two named sources with stable transform round-trip",
            )
        )
        if len(source_reports) < 2:
            checks[-1]["status"] = "fail"

        connectivity, connectivity_checks = _connectivity_checks(sim, inputs)
        checks.extend(connectivity_checks)
        rays, ray_checks = _ray_checks(sim, inputs)
        checks.extend(ray_checks)
        qa_observations, qa_checks = _save_topdown(sim, inputs, output)
        checks.extend(qa_checks)

    independent_check, reference_record = _independent_process_repeatability_check(
        reference_path=reference_evidence,
        inputs=inputs,
        observation_records=observation_records,
        state_hash=before_hash,
        runtime_prefix=prefix,
        mp3d_root=mp3d_root,
        habitat_module_path=habitat_module_path,
        native_binding_path=native_binding_path,
        physics_config_path=physics_config_path,
        asset_records=asset_records,
        avengine_commit=avengine_commit,
        repository_clean=repository_clean,
        output_dir=output,
    )
    checks.append(independent_check)

    evidence: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA_V2,
        "evidence_kind": "completed_capture",
        "room_id": inputs.room["room_id"],
        "room_kind": inputs.room["room_kind"],
        "request_id": inputs.request["request_id"],
        "producer_process": _producer_process_identity(),
        "capture_batch_id": canonical_json_sha256(
            {
                "room_manifest_sha256": sha256_file(inputs.room_path),
                "capture_request_sha256": sha256_file(inputs.request_path),
                "scene_assets": asset_records,
                "avengine_commit": avengine_commit,
                "habitat_runtime_prefix": str(prefix),
                "mp3d_root": str(mp3d_root) if mp3d_root is not None else None,
                "habitat_module_path": str(habitat_module_path),
                "native_binding_path": str(native_binding_path),
                "physics_config_path": str(physics_config_path),
                "state": before,
                "repeat_count": repeat_count,
            }
        ),
        "formal_view_ids": [rig["view_id"]],
        "room_manifest": {
            "path": str(inputs.room_path),
            "sha256": sha256_file(inputs.room_path),
        },
        "capture_request": {
            "path": str(inputs.request_path),
            "sha256": sha256_file(inputs.request_path),
        },
        "runtime": {
            "avengine_commit": avengine_commit,
            "avengine_worktree_dirty": not repository_clean,
            "habitat_runtime_prefix": str(prefix),
            "mp3d_root": str(mp3d_root) if mp3d_root is not None else None,
            "habitat_module_path": str(habitat_module_path),
            "native_binding_path": str(native_binding_path),
            "physics_config_path": str(physics_config_path),
            "habitat_python_version": getattr(habitat_sim, "__version__", None),
            "habitat_audio_enabled": bool(habitat_sim.audio_enabled),
            "habitat_bullet_enabled": bool(habitat_sim.built_with_bullet),
            "habitat_cuda_enabled": bool(habitat_sim.cuda_enabled),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pillow": pillow_version,
        },
        "scene_assets": asset_records,
        "sensor_contract": {
            "rig_id": rig["rig_id"],
            "view_id": rig["view_id"],
            "world_from_rig": world_from_rig,
            "shared_calibration": calibration,
            "modalities": rig["modalities"],
            "listener": inputs.request["listener"],
            "audio_propagation_status": "not_run",
            "audio_propagation_reason": (
                "M1 listener is derived from agent/rig state; multi-source RLR is M4"
            ),
        },
        "capture_state": {
            "before": before,
            "after": after,
            "before_sha256": before_hash,
            "after_sha256": after_hash,
        },
        "repeat_observation_hashes": repeated_hashes,
        "observations": observation_records,
        "sources": source_reports,
        "connectivity": connectivity,
        "ray_checks": rays,
        "qa_observations": qa_observations,
        "independent_reference": reference_record,
        "known_runtime_failures_carried_forward": [
            {
                "check_id": "habitat_direct_import",
                "status": "fail",
                "reason": "fresh direct import aborts unless quaternion is imported first",
            },
            {
                "check_id": "habitat_greedy_follower_binding_cases",
                "status": "fail",
                "reason": "21 PyCapsule iterator cases remain; M1 uses ShortestPath only",
            },
        ],
        "checks": checks,
    }
    finalize_evidence(evidence)
    write_json(output / "evidence.json", evidence)
    return evidence


def build_navmesh(
    inputs: ValidatedM1Inputs,
    *,
    runtime_prefix: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    prefix = discover_runtime_prefix(runtime_prefix)
    mp3d_root = discover_mp3d_root()
    temporary_output = inputs.room_path.parent / "tmp" / "m1_navmesh_build"
    _, habitat_sim, _, _ = _import_installed_habitat(prefix)
    from habitat_sim._ext import habitat_sim_bindings

    _, _, physics_config_path = _installed_runtime_paths(
        prefix, habitat_sim, habitat_sim_bindings
    )
    configuration, _, _, resolved_scene = _make_configuration(
        inputs,
        None,
        temporary_output,
        mp3d_root=mp3d_root,
        include_audio_sensor=False,
        physics_config_path=physics_config_path,
    )
    destination = (
        Path(output_path).resolve()
        if output_path is not None
        else resolved_scene["navmesh"]
    )
    if destination is None:
        raise ValueError("Room manifest does not declare scene.navmesh_path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with habitat_sim.Simulator(configuration) as sim:
        settings = configuration.sim_cfg.navmesh_settings
        success = bool(sim.recompute_navmesh(sim.pathfinder, settings))
        if not success or not sim.pathfinder.is_loaded:
            raise RuntimeError("Habitat navmesh recomputation failed")
        sim.pathfinder.save_nav_mesh(str(destination))
        return {
            "status": "pass",
            "navigable_area_m2": float(sim.pathfinder.navigable_area),
            "num_islands": int(sim.pathfinder.num_islands),
            "artifact": {
                "path": str(destination),
                "byte_size": destination.stat().st_size,
                "sha256": sha256_file(destination),
            },
        }
