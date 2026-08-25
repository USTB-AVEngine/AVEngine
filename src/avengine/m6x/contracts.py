"""Small, reference-only contracts for the fixed-room M6.x canary.

These documents select an existing M6 room revision and describe a compact
anchor/template/scenario plan.  They intentionally do not embed room meshes,
furniture state, AudioProgram events, Timeline frames, or legacy flags.  Those
remain owned by their existing registries and versioned contracts.
"""

from __future__ import annotations

from pathlib import Path
import math
import sys
from typing import Any, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator

from avengine.contracts.json_io import load_json
from avengine.timeline.audio_program import compile_audio_program, validate_audio_program
from avengine.m6.rooms import validate_room_registry


ROOM_CAPSULE_SCHEMA = "avengine_m6x_room_capsule_v1"
ANCHOR_LIBRARY_SCHEMA = "avengine_m6x_anchor_library_v1"
TRAJECTORY_TEMPLATE_SET_SCHEMA = "avengine_m6x_trajectory_template_set_v1"
SCENARIO_SUITE_SCHEMA = "avengine_m6x_scenario_suite_v1"

SCHEMA_FILES = {
    ROOM_CAPSULE_SCHEMA: "m6x_room_capsule_v1.schema.json",
    ANCHOR_LIBRARY_SCHEMA: "m6x_anchor_library_v1.schema.json",
    TRAJECTORY_TEMPLATE_SET_SCHEMA: "m6x_trajectory_template_set_v1.schema.json",
    SCENARIO_SUITE_SCHEMA: "m6x_scenario_suite_v1.schema.json",
}

SCENARIO_CONTRACT = {
    "S0": ("routing_sanity", "one_active_of_n"),
    "S1": ("rear_source", "counterfactual_route_swap"),
    "S2": ("visible_silent_distractor", "one_active_of_n"),
    "S3": ("moving_source", "intermittent_events"),
    "S4": ("overlapping_sources", "simultaneous_subset"),
    "S5": ("los_nlos_contrast", "sequential_sources"),
}

_REQUIRED_RESOURCE_ROLES = {
    "scene_dataset",
    "scene_instance",
    "visual_scene",
    "navmesh",
}
_RESOURCE_TYPE_BY_ROLE = {
    "scene_dataset": "scene_dataset_config",
    "scene_instance": "scene_instance_config",
    "visual_scene": "visual_scene",
    "navmesh": "navmesh",
    "semantic_resource": "semantic_resource",
}
_FRONT_SECTORS = {"front", "front_left", "front_right"}
_REAR_SECTORS = {"rear", "rear_left", "rear_right"}


class M6XContractError(ValueError):
    """One or more M6.x contract invariants failed."""

    def __init__(self, errors: Iterable[str]):
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


def schema_path(schema_name: str) -> Path:
    """Return a source-tree or installed M6.x schema path."""

    try:
        filename = SCHEMA_FILES[schema_name]
    except KeyError as exc:
        raise ValueError(f"unknown M6.x schema: {schema_name!r}") from exc
    source = Path(__file__).resolve().parents[3] / "schemas" / filename
    installed = Path(sys.prefix) / "share" / "avengine" / "schemas" / filename
    path = source if source.is_file() else installed
    if not path.is_file():
        raise FileNotFoundError(f"AVEngine schema is unavailable: {filename}")
    return path


def json_schema_errors(value: Any, schema_name: str) -> list[str]:
    schema = load_json(schema_path(schema_name))
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
    if isinstance(value, Mapping):
        return all(_all_numbers_finite(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_all_numbers_finite(item) for item in value)
    return False


def _base_errors(value: Any, schema_name: str) -> list[str]:
    errors = json_schema_errors(value, schema_name)
    if not isinstance(value, Mapping):
        return errors
    if not _all_numbers_finite(value):
        errors.append("contract must contain only finite JSON numbers")
    return errors


def _canonical_ids(
    records: Any, *, field: str, owner: str, require_sorted: bool = True
) -> list[str]:
    if not isinstance(records, list):
        return []
    ids = [item.get(field) for item in records if isinstance(item, Mapping)]
    errors: list[str] = []
    if len(ids) != len(set(ids)):
        errors.append(f"{owner} must use unique {field} values")
    if require_sorted and ids != sorted(ids):
        errors.append(f"{owner} must use canonical bytewise {field} order")
    return errors


def _matching_room_record(
    capsule: Mapping[str, Any], room_registry: Mapping[str, Any]
) -> tuple[Mapping[str, Any] | None, list[str]]:
    errors = validate_room_registry(room_registry)
    if errors:
        return None, [f"room registry: {item}" for item in errors]
    reference = capsule["room_registry_ref"]
    if reference["registry_id"] != room_registry["registry_id"]:
        return None, ["room_registry_ref.registry_id does not resolve"]
    matches = [
        record
        for record in room_registry["records"]
        if record["room_id"] == reference["room_id"]
        and record["revision"] == reference["room_revision"]
    ]
    if len(matches) != 1:
        return None, ["room_registry_ref must resolve exactly one room revision"]
    return matches[0], []


def validate_room_capsule(
    value: Any, *, room_registry: Mapping[str, Any] | None = None
) -> list[str]:
    """Validate one reference-only frozen-room capsule."""

    errors = _base_errors(value, ROOM_CAPSULE_SCHEMA)
    if errors or not isinstance(value, Mapping):
        return errors
    resource_refs = value["resource_refs"]
    errors.extend(
        _canonical_ids(
            resource_refs, field="role", owner="resource_refs", require_sorted=True
        )
    )
    roles = {item["role"] for item in resource_refs}
    if not _REQUIRED_RESOURCE_ROLES.issubset(roles):
        errors.append(
            "resource_refs must include scene_dataset, scene_instance, visual_scene, "
            "and navmesh"
        )

    resources_by_role = {item["role"]: item["resource_id"] for item in resource_refs}
    scene_instance_resource_id = resources_by_role.get("scene_instance")
    if (
        value["fixed_visual_object_set"]["authority_resource_id"]
        != scene_instance_resource_id
    ):
        errors.append(
            "fixed_visual_object_set.authority_resource_id must equal the "
            "scene_instance resource reference"
        )

    lower, upper = value["operating_area"]["bounds_m"]
    if any(float(low) >= float(high) for low, high in zip(lower, upper)):
        errors.append(
            "operating_area.bounds_m must contain increasing lower/upper bounds"
        )
    floor_height = float(value["operating_area"]["floor_height_m"])
    if not float(lower[1]) <= floor_height <= float(upper[1]):
        errors.append("operating_area.floor_height_m must lie within bounds_m")

    required_forbidden_zones = {
        "loaded_rigid_collision_obbs",
        "navmesh_non_navigable",
    }
    if not required_forbidden_zones.issubset(value["forbidden_zones"]):
        errors.append(
            "forbidden_zones must include navmesh_non_navigable and "
            "loaded_rigid_collision_obbs"
        )

    if room_registry is None:
        return errors
    record, room_errors = _matching_room_record(value, room_registry)
    errors.extend(room_errors)
    if record is None:
        return errors
    resources = {item["resource_id"]: item for item in record["resources"]}
    for reference in resource_refs:
        resource = resources.get(reference["resource_id"])
        if resource is None:
            errors.append(
                f"resource_refs[{reference['role']!r}] does not resolve in room record"
            )
            continue
        expected_type = _RESOURCE_TYPE_BY_ROLE[reference["role"]]
        if resource["resource_type"] != expected_type:
            errors.append(
                f"resource_refs[{reference['role']!r}] requires resource_type "
                f"{expected_type!r}"
            )

    acoustic = value["acoustic_package_ref"]
    representations = {
        item["representation_id"]: item for item in record["acoustic_representations"]
    }
    representation = representations.get(acoustic["acoustic_representation_id"])
    if representation is None:
        errors.append("acoustic_package_ref representation does not resolve")
    else:
        if representation.get("resource_id") != acoustic["resource_id"]:
            errors.append(
                "acoustic_package_ref resource differs from room representation"
            )
        if (
            representation["role"] == "diagnostic_only"
            or representation["geometry_kind"] == "debug_aabb_proxy"
        ):
            errors.append("acoustic_package_ref cannot select diagnostic AABB geometry")
    resource = resources.get(acoustic["resource_id"])
    if resource is None or resource["resource_type"] != "acoustic_package":
        errors.append("acoustic_package_ref must resolve an acoustic_package resource")

    qualification_id = value["qualification_report_id"]
    reports = [
        item
        for item in record["qualification_reports"]
        if item["report_id"] == qualification_id
    ]
    if len(reports) != 1:
        errors.append("qualification_report_id does not resolve exactly once")
    elif (
        reports[0].get("acoustic_representation_id")
        != acoustic["acoustic_representation_id"]
    ):
        errors.append(
            "qualification report does not bind the selected acoustic representation"
        )
    return errors


def validate_anchor_library(
    value: Any, *, room_capsule: Mapping[str, Any] | None = None
) -> list[str]:
    """Validate compact authored poses without treating them as dense room state."""

    errors = _base_errors(value, ANCHOR_LIBRARY_SCHEMA)
    if errors or not isinstance(value, Mapping):
        return errors
    anchors = value["anchors"]
    errors.extend(_canonical_ids(anchors, field="anchor_id", owner="anchors"))
    camera_anchors = [
        item for item in anchors if item["kind"] == "camera_listener_pose"
    ]
    if not camera_anchors:
        errors.append("anchors must contain at least one camera_listener_pose")
    for anchor in camera_anchors:
        if "los_probe_height_m" in anchor:
            errors.append(
                f"camera anchor {anchor['anchor_id']!r} cannot define "
                "los_probe_height_m"
            )
    if room_capsule is not None:
        reference = value["room_capsule_ref"]
        if (
            reference["room_capsule_id"],
            reference["revision"],
        ) != (
            room_capsule.get("room_capsule_id"),
            room_capsule.get("revision"),
        ):
            errors.append("anchor library room_capsule_ref does not resolve")
        expected = room_capsule.get("anchor_library_ref", {})
        if (value["anchor_library_id"], value["revision"]) != (
            expected.get("anchor_library_id"),
            expected.get("revision"),
        ):
            errors.append("room capsule anchor_library_ref differs")
        declared_camera_ids = set(
            room_capsule.get("camera_listener_rig", {}).get("pose_anchor_ids", [])
        )
        observed_camera_ids = {item["anchor_id"] for item in camera_anchors}
        if declared_camera_ids != observed_camera_ids:
            errors.append(
                "camera_listener_rig.pose_anchor_ids must equal camera pose anchors"
            )
    return errors


def validate_trajectory_template_set(
    value: Any,
    *,
    anchor_library: Mapping[str, Any] | None = None,
    room_capsule: Mapping[str, Any] | None = None,
) -> list[str]:
    """Validate sparse anchor-referenced trajectories, never dense motion labels."""

    errors = _base_errors(value, TRAJECTORY_TEMPLATE_SET_SCHEMA)
    if errors or not isinstance(value, Mapping):
        return errors
    templates = value["templates"]
    frame_count = int(value["frame_count"])
    errors.extend(_canonical_ids(templates, field="template_id", owner="templates"))
    for template_index, template in enumerate(templates):
        owner = f"templates[{template_index}]"
        routes = template["routes"]
        errors.extend(_canonical_ids(routes, field="route_id", owner=f"{owner}.routes"))
        kind = template["kind"]
        route_count = len(routes)
        lengths = [len(route["anchor_ids"]) for route in routes]
        interpolations = [route["interpolation"] for route in routes]
        for route in routes:
            indices = route["anchor_frame_indices"]
            if len(indices) != len(route["anchor_ids"]):
                errors.append(
                    f"{owner}.{route['route_id']} anchor_frame_indices must align "
                    "one-to-one with anchor_ids"
                )
            elif (
                indices[0] != 0
                or indices != sorted(set(indices))
                or indices[-1] >= frame_count
            ):
                errors.append(
                    f"{owner}.{route['route_id']} frame indices must start at zero, "
                    "increase strictly, and stay within frame_count"
                )
        if kind == "static" and not (
            route_count == 1 and lengths == [1] and interpolations == ["hold"]
        ):
            errors.append("static template requires one one-anchor hold route")
        elif kind == "linear" and not (
            route_count == 1
            and lengths[0] >= 2
            and interpolations == ["piecewise_linear"]
        ):
            errors.append(
                "linear template requires one multi-anchor piecewise_linear route"
            )
        elif kind == "crossing_pair" and not (
            route_count == 2
            and all(length >= 2 for length in lengths)
            and all(item == "piecewise_linear" for item in interpolations)
        ):
            errors.append(
                "crossing_pair template requires two multi-anchor piecewise_linear routes"
            )
        elif kind == "navmesh_follow" and not (
            route_count == 1
            and lengths[0] >= 2
            and interpolations == ["navmesh_follow"]
        ):
            errors.append(
                "navmesh_follow template requires one multi-anchor navmesh_follow route"
            )

    if anchor_library is not None:
        reference = value["anchor_library_ref"]
        if (reference["anchor_library_id"], reference["revision"]) != (
            anchor_library.get("anchor_library_id"),
            anchor_library.get("revision"),
        ):
            errors.append("trajectory anchor_library_ref does not resolve")
        anchor_index = {item["anchor_id"]: item for item in anchor_library["anchors"]}
        for template in templates:
            for route in template["routes"]:
                for anchor_id in route["anchor_ids"]:
                    anchor = anchor_index.get(anchor_id)
                    if anchor is None:
                        errors.append(
                            f"trajectory route references unknown anchor {anchor_id!r}"
                        )
                    elif anchor["kind"] == "camera_listener_pose":
                        errors.append(
                            f"trajectory route cannot use camera anchor {anchor_id!r}"
                        )
    if room_capsule is not None:
        reference = value["room_capsule_ref"]
        if (reference["room_capsule_id"], reference["revision"]) != (
            room_capsule.get("room_capsule_id"),
            room_capsule.get("revision"),
        ):
            errors.append("trajectory room_capsule_ref does not resolve")
        expected = room_capsule.get("trajectory_template_set_ref", {})
        if (value["trajectory_template_set_id"], value["revision"]) != (
            expected.get("trajectory_template_set_id"),
            expected.get("revision"),
        ):
            errors.append("room capsule trajectory_template_set_ref differs")
    return errors


def _program_index(
    programs: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], Mapping[str, Any]], list[str]]:
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    errors: list[str] = []
    for index, program in enumerate(programs):
        program_errors = validate_audio_program(program)
        errors.extend(f"audio_programs[{index}]: {item}" for item in program_errors)
        key = (str(program.get("program_id", "")), str(program.get("revision", "")))
        if key in result:
            errors.append(f"audio_programs repeats {key[0]}@{key[1]}")
        result[key] = program
    return result, errors


def validate_scenario_suite(
    value: Any,
    *,
    room_capsule: Mapping[str, Any] | None = None,
    anchor_library: Mapping[str, Any] | None = None,
    trajectory_templates: Mapping[str, Any] | None = None,
    audio_programs: Sequence[Mapping[str, Any]] | None = None,
) -> list[str]:
    """Validate S0--S5 while retaining existing AudioProgram/flag/timeline authority."""

    errors = _base_errors(value, SCENARIO_SUITE_SCHEMA)
    if errors or not isinstance(value, Mapping):
        return errors
    scenarios = value["scenarios"]
    scenario_ids = [item["scenario_id"] for item in scenarios]
    if scenario_ids != list(SCENARIO_CONTRACT):
        errors.append("scenarios must contain canonical S0..S5 order exactly once")
    for scenario in scenarios:
        expected = SCENARIO_CONTRACT.get(scenario["scenario_id"])
        if expected is None:
            continue
        if (
            scenario["purpose"],
            scenario["audio_program_ref"]["expected_mode"],
        ) != expected:
            errors.append(
                f"{scenario['scenario_id']} must use purpose={expected[0]!r} and "
                f"AudioProgram mode={expected[1]!r}"
            )
        errors.extend(
            _canonical_ids(
                scenario["source_bindings"],
                field="source_endpoint_id",
                owner=f"{scenario['scenario_id']}.source_bindings",
            )
        )

    if room_capsule is not None:
        reference = value["room_capsule_ref"]
        if (reference["room_capsule_id"], reference["revision"]) != (
            room_capsule.get("room_capsule_id"),
            room_capsule.get("revision"),
        ):
            errors.append("scenario suite room_capsule_ref does not resolve")
    if anchor_library is not None:
        reference = value["anchor_library_ref"]
        if (reference["anchor_library_id"], reference["revision"]) != (
            anchor_library.get("anchor_library_id"),
            anchor_library.get("revision"),
        ):
            errors.append("scenario suite anchor_library_ref does not resolve")
    if trajectory_templates is not None:
        reference = value["trajectory_template_set_ref"]
        if (
            reference["trajectory_template_set_id"],
            reference["revision"],
        ) != (
            trajectory_templates.get("trajectory_template_set_id"),
            trajectory_templates.get("revision"),
        ):
            errors.append("scenario suite trajectory_template_set_ref does not resolve")

    anchor_index = (
        None
        if anchor_library is None
        else {item["anchor_id"]: item for item in anchor_library["anchors"]}
    )
    template_index = (
        None
        if trajectory_templates is None
        else {item["template_id"]: item for item in trajectory_templates["templates"]}
    )
    bindings_by_scenario: dict[str, list[Mapping[str, Any]]] = {}
    for scenario in scenarios:
        bindings = scenario["source_bindings"]
        bindings_by_scenario[scenario["scenario_id"]] = bindings
        window = scenario["capture_frame_window"]
        if window["start_frame"] >= window["end_frame_exclusive"]:
            errors.append(
                f"{scenario['scenario_id']} capture_frame_window must be nonempty"
            )
        if trajectory_templates is not None and window["end_frame_exclusive"] > int(
            trajectory_templates["frame_count"]
        ):
            errors.append(
                f"{scenario['scenario_id']} capture_frame_window exceeds trajectory frame_count"
            )
        if anchor_index is not None:
            listener = anchor_index.get(scenario["listener_anchor_id"])
            if listener is None or listener["kind"] != "camera_listener_pose":
                errors.append(
                    f"{scenario['scenario_id']} listener_anchor_id is not a camera pose"
                )
        for binding in bindings:
            anchor = (
                None
                if anchor_index is None
                else anchor_index.get(binding["spawn_anchor_id"])
            )
            if anchor_index is not None and (
                anchor is None or anchor["kind"] == "camera_listener_pose"
            ):
                errors.append(
                    f"{scenario['scenario_id']} source {binding['source_endpoint_id']!r} "
                    "has an invalid spawn anchor"
                )
            template = (
                None
                if template_index is None
                else template_index.get(binding["trajectory_template_id"])
            )
            if template_index is not None and template is None:
                errors.append(
                    f"{scenario['scenario_id']} references unknown trajectory template"
                )
            elif template is not None:
                routes = {item["route_id"]: item for item in template["routes"]}
                route = routes.get(binding["trajectory_route_id"])
                if route is None:
                    errors.append(
                        f"{scenario['scenario_id']} trajectory_route_id does not resolve"
                    )
                elif route["anchor_ids"][0] != binding["spawn_anchor_id"]:
                    errors.append(
                        f"{scenario['scenario_id']} trajectory must start at spawn anchor"
                    )

    complete_scenario_set = len(scenarios) == len(SCENARIO_CONTRACT) and set(
        bindings_by_scenario
    ) == set(SCENARIO_CONTRACT)
    if anchor_index is not None and complete_scenario_set:

        def source_anchors(scenario_id: str) -> list[Mapping[str, Any]]:
            return [
                anchor_index[item["spawn_anchor_id"]]
                for item in bindings_by_scenario[scenario_id]
                if item["spawn_anchor_id"] in anchor_index
            ]

        s0 = source_anchors("S0")
        if s0 and (
            any(item["expected_acoustic_path"] != "los" for item in s0)
            or len({item["listener_relative_sector"] for item in s0}) < 2
        ):
            errors.append("S0 requires distinct clear-LOS source sectors")
        s1_sectors = {item["listener_relative_sector"] for item in source_anchors("S1")}
        if s1_sectors and not (
            s1_sectors.intersection(_FRONT_SECTORS)
            and s1_sectors.intersection(_REAR_SECTORS)
        ):
            errors.append("S1 requires both front and rear candidate anchors")
        s2 = source_anchors("S2")
        if s2 and any(item["expected_camera_fov"] != "in_fov" for item in s2):
            errors.append("S2 requires all candidate entities to be visible")
        s5_paths = {item["expected_acoustic_path"] for item in source_anchors("S5")}
        if s5_paths and not {"los", "nlos"}.issubset(s5_paths):
            errors.append("S5 requires both LOS and NLOS anchors")

    if template_index is not None and complete_scenario_set:
        s3_kinds = {
            template_index[item["trajectory_template_id"]]["kind"]
            for item in bindings_by_scenario["S3"]
            if item["trajectory_template_id"] in template_index
        }
        if "static" not in s3_kinds or s3_kinds == {"static"}:
            errors.append("S3 requires one moving route and one static distractor")

    if audio_programs is not None:
        programs, program_errors = _program_index(audio_programs)
        errors.extend(program_errors)
        for scenario in scenarios:
            reference = scenario["audio_program_ref"]
            key = (reference["program_id"], reference["revision"])
            program = programs.get(key)
            if program is None:
                errors.append(
                    f"{scenario['scenario_id']} AudioProgram reference does not resolve"
                )
                continue
            if program.get("mode") != reference["expected_mode"]:
                errors.append(
                    f"{scenario['scenario_id']} AudioProgram mode differs from reference"
                )
            candidate_ids = [
                item["source_endpoint_id"] for item in scenario["source_bindings"]
            ]
            if program.get("candidate_source_endpoint_ids") != candidate_ids:
                errors.append(
                    f"{scenario['scenario_id']} candidate endpoints differ from AudioProgram"
                )
                continue
            window = scenario["capture_frame_window"]
            if program.get("timeline", {}).get("frame_count") != (
                window["end_frame_exclusive"] - window["start_frame"]
            ):
                errors.append(
                    f"{scenario['scenario_id']} AudioProgram frame_count differs from capture window"
                )
            if scenario["scenario_id"] == "S2":
                silent_reference = scenario["silent_negative_program_ref"]
                silent = programs.get(
                    (silent_reference["program_id"], silent_reference["revision"])
                )
                if silent is None:
                    errors.append(
                        "S2 silent-negative AudioProgram reference does not resolve"
                    )
                else:
                    if (
                        silent_reference["expected_mode"] != "silent_negative"
                        or silent.get("mode") != "silent_negative"
                    ):
                        errors.append("S2 control must use silent_negative mode")
                    if silent.get("candidate_source_endpoint_ids") != candidate_ids:
                        errors.append(
                            "S2 silent-negative candidates differ from primary"
                        )
                    if silent.get("timeline", {}).get("frame_count") != (
                        window["end_frame_exclusive"] - window["start_frame"]
                    ):
                        errors.append(
                            "S2 silent-negative frame_count differs from capture window"
                        )
            if validate_audio_program(program):
                continue
            compiled = compile_audio_program(program)
            if scenario["scenario_id"] == "S3" and template_index is not None:
                moving = {
                    item["source_endpoint_id"]
                    for item in scenario["source_bindings"]
                    if template_index[item["trajectory_template_id"]]["kind"]
                    != "static"
                }
                static = set(candidate_ids) - moving
                if not moving.intersection(compiled.active_source_endpoint_ids):
                    errors.append("S3 moving endpoint must be acoustically active")
                if not static.intersection(compiled.silent_source_endpoint_ids):
                    errors.append("S3 static distractor must remain silent")
    return errors


def _load(path: str | Path, validator: Any) -> dict[str, Any]:
    value = load_json(path)
    errors = validator(value)
    if errors:
        raise M6XContractError(errors)
    return value


def load_room_capsule(path: str | Path) -> dict[str, Any]:
    return _load(path, validate_room_capsule)


def load_anchor_library(path: str | Path) -> dict[str, Any]:
    return _load(path, validate_anchor_library)


def load_trajectory_template_set(path: str | Path) -> dict[str, Any]:
    return _load(path, validate_trajectory_template_set)


def load_scenario_suite(path: str | Path) -> dict[str, Any]:
    return _load(path, validate_scenario_suite)


__all__ = [
    "ANCHOR_LIBRARY_SCHEMA",
    "M6XContractError",
    "ROOM_CAPSULE_SCHEMA",
    "SCENARIO_CONTRACT",
    "SCENARIO_SUITE_SCHEMA",
    "TRAJECTORY_TEMPLATE_SET_SCHEMA",
    "json_schema_errors",
    "load_anchor_library",
    "load_room_capsule",
    "load_scenario_suite",
    "load_trajectory_template_set",
    "schema_path",
    "validate_anchor_library",
    "validate_room_capsule",
    "validate_scenario_suite",
    "validate_trajectory_template_set",
]
