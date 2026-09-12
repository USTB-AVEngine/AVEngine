from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from avengine.rooms.rooms import (
    ResourceResolution,
    find_acoustic_representation,
    resolve_room_resources,
    room_revision_key,
)


@dataclass(frozen=True)
class AcousticRepresentationResolution:
    representation_id: str
    status: str
    path: Path | None
    build_mode: str
    producer: str | None
    reason: str | None
    input_resources: tuple[str, ...]


@dataclass(frozen=True)
class ProviderRoomResolution:
    room_key: str
    provider_id: str
    status: str
    resources: Mapping[str, ResourceResolution]
    dimension_statuses: Mapping[str, str]
    blockers: tuple[str, ...]


def _aggregate_status(statuses: list[str]) -> str:
    if not statuses:
        return "not_run"
    for candidate in ("fail", "blocked", "not_run"):
        if candidate in statuses:
            return candidate
    return "pass"


class RoomProvider:
    """Data-driven room source adapter.

    Providers resolve declared dependencies and expose build plans. They do not
    silently execute a native runtime or compiler, so an unavailable historical
    package remains blocked/not_run instead of becoming an inferred pass.
    """

    provider_id = ""

    def records(self, registry: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            record
            for record in registry["records"]
            if record["provider_id"] == self.provider_id
        )

    def enumerate_local_rooms(
        self,
        registry: Mapping[str, Any],
        *,
        repository_root: str | Path,
        environment: Mapping[str, str] | None = None,
        verify_hash: bool = True,
    ) -> tuple[ProviderRoomResolution, ...]:
        return tuple(
            self.resolve_room(
                record,
                repository_root=repository_root,
                environment=environment,
                verify_hash=verify_hash,
            )
            for record in self.records(registry)
        )

    def resolve_room(
        self,
        record: Mapping[str, Any],
        *,
        repository_root: str | Path,
        environment: Mapping[str, str] | None = None,
        verify_hash: bool = True,
    ) -> ProviderRoomResolution:
        if record["provider_id"] != self.provider_id:
            raise ValueError(
                f"provider {self.provider_id!r} cannot resolve record owned by "
                f"{record['provider_id']!r}"
            )
        resolutions = resolve_room_resources(
            record,
            repository_root=repository_root,
            environment=environment,
            verify_hash=verify_hash,
        )

        dimension_statuses: dict[str, str] = {}
        all_required_statuses: list[str] = []
        blockers: list[str] = []
        dimensions = {
            dimension
            for resource in record["resources"]
            for dimension in resource["required_for"]
        }
        for dimension in sorted(dimensions):
            resource_ids = [
                resource["resource_id"]
                for resource in record["resources"]
                if dimension in resource["required_for"]
            ]
            statuses = [resolutions[resource_id].status for resource_id in resource_ids]
            dimension_statuses[dimension] = _aggregate_status(statuses)
            all_required_statuses.extend(statuses)
            for resource_id in resource_ids:
                resolution = resolutions[resource_id]
                if resolution.status != "pass":
                    blockers.append(
                        f"{dimension}:{resource_id}:{resolution.status}:"
                        f"{resolution.reason or 'no reason recorded'}"
                    )

        return ProviderRoomResolution(
            room_key=room_revision_key(record),
            provider_id=self.provider_id,
            status=_aggregate_status(all_required_statuses),
            resources=resolutions,
            dimension_statuses=dimension_statuses,
            blockers=tuple(blockers),
        )

    def acoustic_representation(
        self,
        record: Mapping[str, Any],
        representation_id: str,
        *,
        repository_root: str | Path,
        environment: Mapping[str, str] | None = None,
        verify_hash: bool = True,
    ) -> AcousticRepresentationResolution:
        if record["provider_id"] != self.provider_id:
            raise ValueError(
                f"provider {self.provider_id!r} cannot inspect record owned by "
                f"{record['provider_id']!r}"
            )
        representation = find_acoustic_representation(record, representation_id)
        if (
            representation["geometry_kind"] == "debug_aabb_proxy"
            and representation["role"] != "diagnostic_only"
        ):
            return AcousticRepresentationResolution(
                representation_id,
                "fail",
                None,
                representation["build_mode"],
                representation.get("producer"),
                "AABB geometry cannot be an acoustic authority",
                tuple(representation["input_resource_ids"]),
            )

        resolutions = resolve_room_resources(
            record,
            repository_root=repository_root,
            environment=environment,
            verify_hash=verify_hash,
        )
        input_ids = tuple(representation["input_resource_ids"])
        input_statuses = [resolutions[resource_id].status for resource_id in input_ids]
        input_status = _aggregate_status(input_statuses)
        if input_status in {"fail", "blocked", "not_run"}:
            reasons = [
                f"{resource_id}: {resolutions[resource_id].reason}"
                for resource_id in input_ids
                if resolutions[resource_id].status != "pass"
            ]
            return AcousticRepresentationResolution(
                representation_id,
                input_status,
                None,
                representation["build_mode"],
                representation.get("producer"),
                "; ".join(reasons),
                input_ids,
            )

        resource_id = representation.get("resource_id")
        output_resolution = resolutions.get(resource_id) if resource_id else None
        build_mode = representation["build_mode"]
        if build_mode == "reference":
            if output_resolution is None:
                return AcousticRepresentationResolution(
                    representation_id,
                    "fail",
                    None,
                    build_mode,
                    None,
                    "reference mode requires a resource_id",
                    input_ids,
                )
            return AcousticRepresentationResolution(
                representation_id,
                output_resolution.status,
                output_resolution.path,
                build_mode,
                None,
                output_resolution.reason,
                input_ids,
            )
        if build_mode in {"compile", "derive"}:
            producer = representation.get("producer")
            if not producer:
                return AcousticRepresentationResolution(
                    representation_id,
                    "fail",
                    None,
                    build_mode,
                    None,
                    f"{build_mode} mode requires an explicit producer",
                    input_ids,
                )
            if output_resolution is not None:
                if output_resolution.status == "pass":
                    return AcousticRepresentationResolution(
                        representation_id,
                        "pass",
                        output_resolution.path,
                        build_mode,
                        producer,
                        None,
                        input_ids,
                    )
                if output_resolution.status in {"fail", "blocked"}:
                    return AcousticRepresentationResolution(
                        representation_id,
                        output_resolution.status,
                        None,
                        build_mode,
                        producer,
                        output_resolution.reason,
                        input_ids,
                    )
            return AcousticRepresentationResolution(
                representation_id,
                "not_run",
                None,
                build_mode,
                producer,
                f"declared producer has not materialized a verified output: {producer}",
                input_ids,
            )
        return AcousticRepresentationResolution(
            representation_id,
            "blocked",
            None,
            build_mode,
            representation.get("producer"),
            representation.get("blocker", "acoustic representation is unavailable"),
            input_ids,
        )


class BlenderCustomRoomProvider(RoomProvider):
    provider_id = "blender_custom"


class ReplicaCADRoomProvider(RoomProvider):
    provider_id = "replica_cad"


class LegacyUEApartmentRoomProvider(RoomProvider):
    provider_id = "legacy_ue_apartment"


class Matterport3DRoomProvider(RoomProvider):
    provider_id = "matterport3d"


PROVIDER_TYPES = {
    provider_type.provider_id: provider_type
    for provider_type in (
        BlenderCustomRoomProvider,
        ReplicaCADRoomProvider,
        LegacyUEApartmentRoomProvider,
        Matterport3DRoomProvider,
    )
}


def provider_for_id(provider_id: str) -> RoomProvider:
    try:
        return PROVIDER_TYPES[provider_id]()
    except KeyError as error:
        raise KeyError(f"no M6 room provider registered for {provider_id!r}") from error


def providers_from_registry(registry: Mapping[str, Any]) -> tuple[RoomProvider, ...]:
    provider_ids = sorted({record["provider_id"] for record in registry["records"]})
    return tuple(provider_for_id(provider_id) for provider_id in provider_ids)


# ---------------------------------------------------------------------------
# V1 room-package catalog: one entry point for every production route
# ---------------------------------------------------------------------------
#
# The provider classes above resolve the M6 room registry, which is a
# different registration than the RoomPackage catalog the V1 routes use. This
# section is the RoomPackage side, and it is the entry point planning and
# capture should ask: give me this room_id's route, its resources and the
# reason if it cannot run.
#
# Every dispatch below keys off declared facts - family, renderer, and
# walkable_space.kind - never off a room_id. Registering another scene on a
# backend that already has an adapter is a catalog entry plus its package;
# it is not a code change here. A genuinely new backend is a new renderer in
# PLANNING_ADAPTERS and CAPTURE_ENTRYPOINTS with its own adapter, which is the
# place that boundary is allowed to appear.

from avengine.rooms.room_package import (
    RENDERERS,
    host_runtime_path_bindings,
    package_from_catalog_entry,
    renderer_for_room,
    resolve_room_runtime,
    room_capability_report,
    resolve_catalog_room_package_path,
    runtime_isolation_key,
)

V1_CATALOG_SCHEMA = "avengine_qa_room_package_catalog_v1"

# walkable_space.kind -> the planning adapter that consumes it. The names are
# the branches load_planning_resources already implements; naming them makes
# the route reportable instead of implicit in an if-chain.
PLANNING_ADAPTERS = {
    "route_bank": "native_spear_route_bank",
    "walkable_grid": "retained_ue_walkable_grid",
    "habitat_navmesh": "habitat_native_navmesh",
    "furniture_grid": "furnished_manifest_raster",
}

# renderer -> the executor that renders the selected room.
CAPTURE_ENTRYPOINTS = {
    "ue_spear": "tools/rooms/run_spear_residential_episode.py",
    "habitat": "tools/capture/capture_mp3d_multi_actor.py",
}

# The families that are current production output. `authored` packages stay
# resolvable - they are retained comparison material - but they are not a
# production route and must not be counted as one.
PRODUCTION_FAMILIES = ("apartment", "kujiale", "mp3d", "hm3d")


class RoomRouteError(ValueError):
    """Raised when a requested room cannot be routed, with the exact reason."""


@dataclass(frozen=True)
class RoomRoute:
    room_id: str
    family: str
    renderer: str
    walkable_kind: str
    planning_adapter: str
    capture_entrypoint: str
    production_family: bool


@dataclass(frozen=True)
class RoomResolution:
    room_id: str
    status: str
    resource_status: str
    route: RoomRoute | None
    package: Mapping[str, Any] | None
    planning_room: Mapping[str, Any] | None
    capabilities: Mapping[str, Any] | None
    runtime: Mapping[str, Any] | None
    blockers: tuple[str, ...]
    reason: str | None
    profile: Mapping[str, Any] | None = None
    render: Mapping[str, Any] | None = None

    def as_report(self) -> dict[str, Any]:
        """A JSON-writable record of what this room resolved to and why."""
        route = self.route
        return {
            "schema": "avengine_qa_room_dispatch_v1",
            "room_id": self.room_id,
            "status": self.status,
            "resource_status": self.resource_status,
            "route": None if route is None else {
                "family": route.family,
                "renderer": route.renderer,
                "walkable_kind": route.walkable_kind,
                "planning_adapter": route.planning_adapter,
                "capture_entrypoint": route.capture_entrypoint,
                "production_family": route.production_family,
            },
            "capabilities": dict(self.capabilities) if self.capabilities else None,
            "runtime": dict(self.runtime) if self.runtime else None,
            "room_runtime_profile_id": (
                None if not self.profile else str(self.profile.get("profile_id"))),
            "render": dict(self.render) if self.render else None,
            "blockers": list(self.blockers),
            "reason": self.reason,
            "native_execution": "not_run",
        }


def load_room_catalog(path: str | Path) -> dict[str, Any]:
    """Read a RoomPackage catalog and reject a shape nothing can dispatch."""
    source = Path(path).expanduser()
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise RoomRouteError(f"room catalog must be a mapping: {source}")
    if raw.get("schema") != V1_CATALOG_SCHEMA:
        raise RoomRouteError(
            f"room catalog schema must be {V1_CATALOG_SCHEMA}, got "
            f"{raw.get('schema')!r}: {source}"
        )
    rooms = raw.get("rooms")
    if not isinstance(rooms, list) or not rooms:
        raise RoomRouteError(f"room catalog declares no rooms: {source}")
    seen: set[str] = set()
    for entry in rooms:
        if not isinstance(entry, Mapping) or not entry.get("room_id"):
            raise RoomRouteError(f"every catalog room needs a room_id: {source}")
        room_id = str(entry["room_id"])
        if room_id in seen:
            raise RoomRouteError(f"duplicate room_id {room_id!r} in {source}")
        seen.add(room_id)
    return dict(raw)


def catalog_room_ids(catalog: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(entry["room_id"]) for entry in catalog["rooms"])


def catalog_room_entry(catalog: Mapping[str, Any], room_id: str) -> Mapping[str, Any]:
    for entry in catalog["rooms"]:
        if str(entry["room_id"]) == str(room_id):
            return entry
    raise RoomRouteError(
        f"no room {room_id!r} is registered in this catalog; registered rooms "
        f"are {list(catalog_room_ids(catalog))}"
    )


def catalog_runtime(
    catalog: Mapping[str, Any],
    runtime: Mapping[str, Any] | None = None,
    *,
    host_config: Mapping[str, Any] | None = None,
    renderer: str | None = None,
    room_id: str | None = None,
) -> dict[str, Any]:
    """Merge the declared path roots, most specific source last.

    Order is catalog, then the run-local host config, then the caller. So a
    shipped catalog can declare defaults, a host can retarget them for its own
    filesystem, and one run can still override a single root without
    restating the rest. Shell environment variables are not a source here.
    """
    catalog_bindings = dict(catalog.get("path_bindings") or {})
    host_bindings = host_runtime_path_bindings(
        host_config, renderer or "", room_id) if host_config else {}
    supplied = dict(runtime or {})
    bindings = {
        **catalog_bindings,
        **host_bindings,
        **dict(supplied.get("path_bindings") or {}),
    }
    return {**supplied, "path_bindings": bindings}


def catalog_binding_requirements(catalog: Mapping[str, Any]) -> tuple[str, ...]:
    """Every ${ROOT} name the catalog's rooms and packages actually need.

    Reported so a host config can be checked for completeness before a run,
    instead of a template expanding to a literal ${...} deep inside a plan.
    """
    declared = catalog.get("path_binding_requirements")
    if isinstance(declared, list) and declared:
        return tuple(sorted({str(item) for item in declared}))
    return tuple(sorted(set(catalog.get("path_bindings") or {})))


def room_route(package: Mapping[str, Any]) -> RoomRoute:
    """Derive the route from declared family/renderer/walkable kind only."""
    renderer = renderer_for_room(package)
    if renderer not in CAPTURE_ENTRYPOINTS:
        raise RoomRouteError(
            f"renderer {renderer!r} has no capture adapter; supported renderers "
            f"are {sorted(CAPTURE_ENTRYPOINTS)}. A new backend needs its own "
            f"entry in CAPTURE_ENTRYPOINTS and PLANNING_ADAPTERS, not a room_id branch."
        )
    walkable = package.get("walkable_space") or {}
    kind = str(walkable.get("kind") or "")
    if kind not in PLANNING_ADAPTERS:
        raise RoomRouteError(
            f"walkable_space.kind {kind!r} has no planning adapter; supported "
            f"kinds are {sorted(PLANNING_ADAPTERS)}"
        )
    family = str(package.get("family"))
    return RoomRoute(
        room_id=str(package.get("room_id")),
        family=family,
        renderer=renderer,
        walkable_kind=kind,
        planning_adapter=PLANNING_ADAPTERS[kind],
        capture_entrypoint=CAPTURE_ENTRYPOINTS[renderer],
        production_family=family in PRODUCTION_FAMILIES,
    )


def planning_room_mapping(package: Mapping[str, Any]) -> dict[str, Any]:
    """Shape the expanded package into the mapping planning adapters read.

    ``planning_inputs`` is flattened to the top level because that is where
    the adapters look for ``room_manifest``, ``route_bank``, ``manifest`` and
    the rest; a retained legacy entry keeps priority so an older catalog row
    still resolves exactly as it did.
    """
    base = package.get("legacy_catalog_entry")
    if not isinstance(base, Mapping):
        base = package.get("planning_inputs") or {}
    room = {**deepcopy(dict(base)), "room_id": package.get("room_id")}
    room["room_package"] = deepcopy(dict(package))
    return room


def resolve_catalog_room(
    catalog: Mapping[str, Any],
    room_id: str,
    *,
    catalog_path: str | Path,
    runtime: Mapping[str, Any] | None = None,
    profile_registry: Mapping[str, Any] | None = None,
    profile_id: str | None = None,
    host_config: Mapping[str, Any] | None = None,
    request: Mapping[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
    allow_environment: bool = False,
) -> RoomResolution:
    """Resolve one registered room into its route, resources and reasons.

    A data problem comes back as a ``RoomResolution`` whose status and reason
    say what is wrong; only an unknown ``room_id`` or an undispatchable
    catalog raises. That split exists so enumerating a catalog reports every
    room's real state instead of stopping at the first blocked one.
    """
    entry = catalog_room_entry(catalog, room_id)
    effective_runtime = catalog_runtime(
        catalog, runtime, host_config=host_config,
        renderer=entry.get("renderer"), room_id=str(room_id))
    try:
        package = package_from_catalog_entry(
            entry, runtime=effective_runtime, catalog_path=catalog_path
        )
    except (ValueError, OSError) as error:
        return RoomResolution(
            room_id=str(room_id), status="blocked", resource_status="blocked",
            route=None, package=None,
            planning_room=None, capabilities=None, runtime=None,
            blockers=(f"{type(error).__name__}: {error}",),
            reason=f"room package did not resolve: {error}",
        )
    declared = (entry.get("family"), entry.get("renderer"))
    if declared != (None, None) and declared != (package.get("family"), package.get("renderer")):
        return RoomResolution(
            room_id=str(room_id), status="fail", resource_status="fail",
            route=None, package=package,
            planning_room=None, capabilities=None, runtime=None,
            blockers=(
                f"catalog declares {declared} but the package declares "
                f"{(package.get('family'), package.get('renderer'))}",
            ),
            reason=(
                "catalog row and room package disagree about the production "
                "route; they must name the same family and renderer"
            ),
        )
    try:
        route = room_route(package)
    except (RoomRouteError, ValueError) as error:
        return RoomResolution(
            room_id=str(room_id), status="blocked", resource_status="blocked",
            route=None, package=package,
            planning_room=None, capabilities=None, runtime=None,
            blockers=(str(error),), reason=str(error),
        )
    relative_roots = [
        resolve_catalog_room_package_path(
            entry["room_package"], catalog_path=catalog_path,
            runtime=effective_runtime,
        ).parent
    ] if isinstance(entry.get("room_package"), str) else None
    capabilities = room_capability_report(
        package, relative_roots=relative_roots, runtime=effective_runtime
    )
    runtime_report = resolve_room_runtime(
        package, effective_runtime, host_config=host_config,
        room_id=str(package["room_id"]), environment=environment,
        allow_environment=allow_environment,
    )
    capabilities["runtime"] = runtime_report
    try:
        profile = resolve_room_profile(
            profile_registry, str(package["room_id"]), profile_id=profile_id)
    except RoomRouteError as error:
        return RoomResolution(
            room_id=str(room_id), status="fail", resource_status="fail",
            route=None, package=package,
            planning_room=None, capabilities=None, runtime=None,
            blockers=(str(error),), reason=str(error),
        )
    render = room_render_parameters(profile, request)
    blockers = tuple(
        f"{dimension}:{entry_report['status']}:{entry_report['reason']}"
        for dimension, entry_report in capabilities["dimensions"].items()
        if entry_report["status"] != "pass" and entry_report.get("reason")
    )
    if runtime_report["status"] != "pass":
        blockers += (f"runtime:{runtime_report['status']}:{runtime_report['reason']}",)
    blockers += tuple(
        f"room_runtime_profile:fail:{item}"
        for item in profile_route_conflicts(profile, package)
    )
    validation = package.get("validation_errors") or []
    if validation:
        blockers += tuple(f"package_validation:{item}" for item in validation)
    resource_blockers = tuple(
        blocker for blocker in blockers if not blocker.startswith("runtime:")
    )
    resource_status = "pass" if not resource_blockers else (
        "fail" if any(
            item.startswith(("package_validation:", "room_runtime_profile:"))
            for item in resource_blockers)
        else "blocked"
    )
    status = resource_status if resource_status != "pass" else (
        "pass" if runtime_report["status"] == "pass" else "blocked"
    )
    return RoomResolution(
        room_id=str(package["room_id"]), status=status,
        resource_status=resource_status, route=route,
        package=package, planning_room=planning_room_mapping(package),
        capabilities=capabilities, runtime=runtime_report, blockers=blockers,
        reason=None if status == "pass" else "; ".join(blockers),
        profile=profile, render=render,
    )


def enumerate_catalog_rooms(
    catalog: Mapping[str, Any],
    *,
    catalog_path: str | Path,
    runtime: Mapping[str, Any] | None = None,
    production_only: bool = False,
    profile_registry: Mapping[str, Any] | None = None,
    host_config: Mapping[str, Any] | None = None,
    request: Mapping[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
    allow_environment: bool = False,
) -> tuple[RoomResolution, ...]:
    """Resolve every registered room, keeping each room's real status."""
    resolutions = tuple(
        resolve_catalog_room(
            catalog, room_id, catalog_path=catalog_path, runtime=runtime,
            profile_registry=profile_registry, host_config=host_config,
            request=request, environment=environment,
            allow_environment=allow_environment,
        )
        for room_id in catalog_room_ids(catalog)
    )
    if production_only:
        resolutions = tuple(
            item for item in resolutions
            if item.route is not None and item.route.production_family
        )
    return resolutions


def require_catalog_room(
    catalog: Mapping[str, Any],
    room_id: str,
    *,
    catalog_path: str | Path,
    runtime: Mapping[str, Any] | None = None,
    require_runtime: bool = True,
    profile_registry: Mapping[str, Any] | None = None,
    profile_id: str | None = None,
    host_config: Mapping[str, Any] | None = None,
    request: Mapping[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
    allow_environment: bool = False,
) -> RoomResolution:
    """Resolve a room for execution, raising the exact reason when it cannot.

    ``require_runtime`` is False for a plan-only query: planning reads the
    room's declared resources and does not need the renderer's executor
    parameters, so a missing ``uproject`` must not block a plan.
    """
    resolution = resolve_catalog_room(
        catalog, room_id, catalog_path=catalog_path, runtime=runtime,
        profile_registry=profile_registry, profile_id=profile_id,
        host_config=host_config, request=request, environment=environment,
        allow_environment=allow_environment,
    )
    if resolution.route is None:
        raise RoomRouteError(
            f"room {room_id!r} cannot be routed: {resolution.reason}"
        )
    fatal = [
        blocker for blocker in resolution.blockers
        if require_runtime or not blocker.startswith("runtime:")
    ]
    if fatal:
        raise RoomRouteError(
            f"room {room_id!r} is not ready to execute on route "
            f"{resolution.route.family}/{resolution.route.renderer}: "
            + "; ".join(fatal)
        )
    return resolution


# ---------------------------------------------------------------------------
# Room runtime profile reference and process isolation grouping
# ---------------------------------------------------------------------------
#
# The profile registry carries the registered render transport for a room -
# resolution, frame count, frame rate, warmups - which is a different fact
# from the room's resources (the package) and from the host's runtime
# (the machine). Referencing it is optional: a room with no registered
# profile still routes on its package alone, so a room can be added by
# registration before anyone writes a profile for it.

from avengine.runtime_profiles import (
    RuntimeProfileError,
    load_room_runtime_profile_registry,
    validate_room_runtime_profile_registry,
)

# The V1 transport registers its own adapters. spear_apartment_v1 pins the
# older 75-frame transport and validates its render block exactly, so V1
# profiles must not claim it.
V1_PROFILE_ADAPTERS = frozenset({
    "avengine_qa_v1_spear_room_v1",
    "avengine_qa_v1_habitat_room_v1",
})

# profile backend_id -> RoomPackage renderer.
BACKEND_RENDERERS = {
    "spear_unreal": "ue_spear",
    "habitat_native": "habitat",
}

_RENDER_KEYS = (
    "width", "height", "frame_count", "frame_rate_hz", "horizontal_fov_deg",
    "streaming_warmup_frames", "camera_warmup_frames",
)


def load_profile_registry(path: str | Path | None = None) -> dict[str, Any]:
    """Load the ordinary room runtime profile registry, validated as usual."""
    if path is None:
        from avengine.runtime_profiles import load_default_room_runtime_profile_registry

        return load_default_room_runtime_profile_registry()
    return load_room_runtime_profile_registry(path)


def resolve_room_profile(
    registry: Mapping[str, Any] | None,
    room_id: str,
    *,
    profile_id: str | None = None,
    v1_only: bool = True,
) -> Mapping[str, Any] | None:
    """Find the registered render transport for one room, or None.

    None is a legitimate answer: it means nobody has registered a profile for
    this room yet, and the package alone drives the route.
    """
    if not registry:
        return None
    errors = validate_room_runtime_profile_registry(registry)
    if errors:
        raise RoomRouteError("room runtime profile registry is invalid: "
                             + "; ".join(errors))
    profiles = list(registry.get("profiles", ()))
    if profile_id is not None:
        for profile in profiles:
            if str(profile.get("profile_id")) == str(profile_id):
                return profile
        raise RoomRouteError(
            f"no room runtime profile {profile_id!r}; registered profiles are "
            f"{[str(item.get('profile_id')) for item in profiles]}"
        )
    matches = [
        profile for profile in profiles
        if str((profile.get("room_ref") or {}).get("room_id")) == str(room_id)
    ]
    if v1_only:
        preferred = [
            profile for profile in matches
            if str(profile.get("adapter_id")) in V1_PROFILE_ADAPTERS
        ]
        if preferred:
            matches = preferred
        else:
            return None
    if not matches:
        return None
    if len(matches) > 1:
        raise RoomRouteError(
            f"room {room_id!r} resolves to more than one runtime profile: "
            f"{[str(item.get('profile_id')) for item in matches]}; name one "
            f"with profile_id"
        )
    return matches[0]


def profile_route_conflicts(
    profile: Mapping[str, Any] | None, package: Mapping[str, Any]
) -> tuple[str, ...]:
    """Report a profile that disagrees with the package about the route."""
    if not profile:
        return ()
    problems: list[str] = []
    backend = str(profile.get("backend_id"))
    expected = BACKEND_RENDERERS.get(backend)
    renderer = renderer_for_room(package)
    if expected is None:
        problems.append(
            f"profile backend_id {backend!r} has no renderer mapping; known "
            f"backends are {sorted(BACKEND_RENDERERS)}"
        )
    elif expected != renderer:
        problems.append(
            f"profile backend_id {backend!r} implies renderer {expected!r} but "
            f"the room package declares {renderer!r}"
        )
    scene = profile.get("scene") or {}
    if renderer == "ue_spear":
        declared = (package.get("visual_scene") or {}).get("map_path")
        if declared and scene.get("map_path") and declared != scene["map_path"]:
            problems.append(
                f"profile map_path {scene['map_path']!r} differs from the room "
                f"package map_path {declared!r}"
            )
    return tuple(problems)


def _request_render_request(request: Mapping[str, Any] | None) -> dict[str, Any]:
    """The render facts a request states, in profile vocabulary."""
    if not request:
        return {}
    stated: dict[str, Any] = {}
    for source, target in (("frame_count", "frame_count"),
                           ("frame_rate_hz", "frame_rate_hz")):
        if request.get(source) is not None:
            stated[target] = request[source]
    camera = request.get("camera")
    if isinstance(camera, Mapping):
        if camera.get("fov_deg") is not None:
            stated["horizontal_fov_deg"] = camera["fov_deg"]
        resolution = camera.get("resolution_hw")
        if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
            stated["height"], stated["width"] = resolution[0], resolution[1]
    return stated


def room_render_parameters(
    profile: Mapping[str, Any] | None,
    request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge the registered transport with what this request states.

    The request wins, because it is the run's decision, but every divergence
    is reported. A silently dropped frame count or field of view is the
    failure this exists to prevent.
    """
    registered = dict((profile or {}).get("render") or {})
    stated = _request_render_request(request)
    effective: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    conflicts: list[str] = []
    for key in _RENDER_KEYS:
        if key in stated:
            effective[key] = stated[key]
            provenance[key] = "request"
            if key in registered and registered[key] != stated[key]:
                conflicts.append(
                    f"{key}: request {stated[key]!r} overrides registered "
                    f"profile {registered[key]!r}"
                )
        elif key in registered:
            effective[key] = registered[key]
            provenance[key] = "room_runtime_profile"
    return {
        "schema": "avengine_qa_room_render_parameters_v1",
        "profile_id": None if not profile else str(profile.get("profile_id")),
        "profile_revision": None if not profile else str(profile.get("revision")),
        "effective": effective,
        "provenance": provenance,
        "conflicts": tuple(conflicts),
        "registered": registered,
        "requested": stated,
    }


def effective_camera_request(
    render: Mapping[str, Any] | None,
    request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """State the camera transport explicitly, so no downstream default decides.

    ``conditioned_sampler.select_camera_and_schedule`` falls back to 85 deg
    when the request's camera block has no ``fov_deg``. That default is right
    for V1, but it is the sampler's, not the room's: a room whose registered
    transport says something else would silently render at 85. Writing the
    resolved value into the request removes the fallback from the decision.

    A field of view the request stated explicitly always survives: it is
    already the winning source inside ``room_render_parameters``.
    """
    camera = dict((request or {}).get("camera") or {})
    effective = dict((render or {}).get("effective") or {})
    fov = effective.get("horizontal_fov_deg")
    if fov is not None:
        camera["fov_deg"] = float(fov)
    height = effective.get("height")
    width = effective.get("width")
    if "resolution_hw" not in camera and height and width:
        camera["resolution_hw"] = [int(height), int(width)]
    camera.setdefault("motion", "static")
    return camera


def request_with_effective_camera(
    request: Mapping[str, Any],
    render: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """A copy of the request whose camera block states the resolved transport."""
    updated = deepcopy(dict(request))
    updated["camera"] = effective_camera_request(render, request)
    fov = updated["camera"].get("fov_deg")
    if fov is not None:
        # The older flat spelling some callers still read.
        updated["camera_fov_deg"] = float(fov)
    return updated


def runtime_isolation_groups(
    resolutions: Sequence[RoomResolution],
) -> tuple[dict[str, Any], ...]:
    """Group rooms that may share one worker process.

    P18 needs this because MP3D and HM3D pin different Habitat prefixes, and
    Habitat binds its native modules once per interpreter. Rooms in different
    groups must run in different processes, and for the renderers flagged
    below the new process has to be a fresh interpreter - a plain fork from a
    parent that already imported Habitat still carries the parent's prefix.
    """
    from avengine.rooms.room_package import runtime_isolation_key

    grouped: dict[tuple, dict[str, Any]] = {}
    for resolution in resolutions:
        if resolution.route is None or not resolution.runtime:
            continue
        key = runtime_isolation_key(resolution.runtime)
        entry = grouped.setdefault(key, {
            "group_key": key,
            "renderer": resolution.route.renderer,
            "isolation_keys": list(resolution.runtime["isolation_keys"]),
            "isolation_values": {
                name: resolution.runtime["effective"].get(name)
                for name in resolution.runtime["isolation_keys"]
            },
            "requires_fresh_interpreter": bool(
                resolution.runtime["requires_fresh_interpreter"]),
            "room_ids": [],
        })
        entry["room_ids"].append(resolution.room_id)
    for entry in grouped.values():
        entry["room_ids"] = sorted(entry["room_ids"])
        entry["reason"] = (
            "Habitat binds its native modules to one runtime prefix per "
            "interpreter, so this group needs its own freshly started process "
            "(spawn or subprocess), not a fork from a parent that already "
            "imported Habitat."
            if entry["requires_fresh_interpreter"] else
            "These rooms share one renderer runtime and may run in one process."
        )
    return tuple(
        entry for _key, entry in sorted(
            grouped.items(), key=lambda item: (item[1]["renderer"], item[0]))
    )
