"""Build a complete ReplicaCAD import/spawn plan for the optional UE backend.

This module deliberately has no SPEAR, Unreal, or Habitat-Sim dependency.  It
turns Habitat metadata into a validated data plan that an optional renderer
may execute later.  In particular, a furnished ReplicaCAD scene is represented
by its stage *and* every rigid and articulated instance; loading only the stage
is treated as an incomplete scene, not as a successful conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
import glob
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET


PLAN_SCHEMA = "avengine_optional_spear_replicacad_plan_v1"
COORDINATE_CONVENTION = "habitat_y_up_m_to_unreal_z_up_cm_v1"


class ReplicaCADPlanError(RuntimeError):
    """The supplied ReplicaCAD metadata does not form a complete scene."""


@dataclass(frozen=True)
class HabitatTransform:
    """A source transform exactly in Habitat's scene-instance convention."""

    translation_m: tuple[float, float, float]
    rotation_wxyz: tuple[float, float, float, float]
    scale_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class UnrealTransform:
    """The same transform expressed in UE centimeters and an XYZW quaternion."""

    translation_cm: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    scale_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class ArticulatedJointDefault:
    joint_name: str
    joint_type: str
    position: float
    source: str
    clamped_to_limit: bool


@dataclass(frozen=True)
class ArticulatedVisual:
    """One URDF visual occurrence at the instance's declared joint pose.

    ``mesh_path`` is intentionally not unique: a cabinet can use the same door
    mesh for multiple links.  Keeping every occurrence is required for an
    actual UE scene closure; the de-duplicated import list alone is not a
    drawable-instance list.
    """

    visual_id: str
    link_name: str
    mesh_path: Path
    root_from_visual_matrix: tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class ReplicaCADImport:
    import_id: str
    asset_kind: str
    template_name: str
    template_config_path: Path
    pbr_mesh_paths: tuple[Path, ...]
    urdf_path: Path | None = None


@dataclass(frozen=True)
class ReplicaCADSpawn:
    spawn_id: str
    asset_kind: str
    source_index: int
    import_id: str
    template_name: str
    habitat_transform: HabitatTransform
    unreal_transform: UnrealTransform
    motion_type: str | None
    translation_origin: str | None
    fixed_base: bool | None = None
    auto_clamp_joint_limits: bool | None = None
    joint_defaults: tuple[ArticulatedJointDefault, ...] = ()
    articulated_visuals: tuple[ArticulatedVisual, ...] = ()


@dataclass(frozen=True)
class ReplicaCADScenePlan:
    schema: str
    coordinate_convention: str
    dataset_config_path: Path
    scene_instance_path: Path
    default_lighting: str | None
    imports: tuple[ReplicaCADImport, ...]
    spawns: tuple[ReplicaCADSpawn, ...]
    source_stage_count: int
    source_rigid_count: int
    source_articulated_count: int

    @property
    def stage_spawns(self) -> tuple[ReplicaCADSpawn, ...]:
        return tuple(spawn for spawn in self.spawns if spawn.asset_kind == "stage")

    @property
    def rigid_spawns(self) -> tuple[ReplicaCADSpawn, ...]:
        return tuple(spawn for spawn in self.spawns if spawn.asset_kind == "rigid")

    @property
    def articulated_spawns(self) -> tuple[ReplicaCADSpawn, ...]:
        return tuple(
            spawn for spawn in self.spawns if spawn.asset_kind == "articulated"
        )

    def assert_closed(self) -> None:
        """Reject any producer bug that silently drops a source instance."""

        actual = (
            len(self.stage_spawns),
            len(self.rigid_spawns),
            len(self.articulated_spawns),
        )
        expected = (
            self.source_stage_count,
            self.source_rigid_count,
            self.source_articulated_count,
        )
        if actual != expected:
            raise ReplicaCADPlanError(
                "ReplicaCAD plan is not count-closed: "
                f"source(stage, rigid, articulated)={expected}, planned={actual}"
            )

        import_ids = {item.import_id for item in self.imports}
        missing = sorted(
            {spawn.import_id for spawn in self.spawns}.difference(import_ids)
        )
        if missing:
            raise ReplicaCADPlanError(
                f"ReplicaCAD spawn plan references missing imports: {missing}"
            )


TemplateInputs = Mapping[str, str | Path] | Iterable[str | Path]


def _finite_number(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplicaCADPlanError(f"{owner} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ReplicaCADPlanError(f"{owner} must be a finite number")
    return result


def _vector(value: Any, length: int, *, owner: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ReplicaCADPlanError(f"{owner} must contain {length} finite numbers")
    return tuple(
        _finite_number(item, owner=f"{owner}[{index}]")
        for index, item in enumerate(value)
    )


def habitat_position_to_unreal_cm(
    position_m: Sequence[float],
) -> tuple[float, float, float]:
    """Map Habitat ``[X right, Y up, Z back]`` to UE centimeters.

    The explicit basis is ``U = [100*Hx, 100*Hz, 100*Hy]``.  The axis swap has
    determinant -1, which also changes the quaternion-vector sign below when
    converting Habitat's right-handed basis to UE's left-handed basis.
    """

    x, y, z = _vector(position_m, 3, owner="Habitat position")
    return (100.0 * x, 100.0 * z, 100.0 * y)


def habitat_scale_to_unreal(scale_xyz: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = _vector(scale_xyz, 3, owner="Habitat scale")
    return (x, z, y)


def habitat_quaternion_wxyz_to_unreal_xyzw(
    rotation_wxyz: Sequence[float],
) -> tuple[float, float, float, float]:
    """Conjugate a Habitat rotation by the Habitat-to-UE basis change.

    For ``P = [[1,0,0],[0,0,1],[0,1,0]]``, ``R_ue = P R_h P^-1``.  Since P is
    a reflection, its action on the quaternion vector is ``det(P) * P``.  A
    Habitat ``[w,x,y,z]`` quaternion therefore becomes UE XYZW
    ``[-x,-z,-y,w]``.
    """

    w, x, y, z = _vector(rotation_wxyz, 4, owner="Habitat rotation")
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-12:
        raise ReplicaCADPlanError("Habitat rotation must have non-zero norm")
    return (-x / norm, -z / norm, -y / norm, w / norm)


def _required_file(value: str | Path, *, owner: str) -> Path:
    path = Path(value).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ReplicaCADPlanError(f"{owner} does not exist: {path}") from exc
    if not resolved.is_file():
        raise ReplicaCADPlanError(f"{owner} is not a file: {resolved}")
    return resolved


def _load_json(value: str | Path, *, owner: str) -> tuple[Path, Mapping[str, Any]]:
    path = _required_file(value, owner=owner)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReplicaCADPlanError(f"{owner} cannot be read as JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ReplicaCADPlanError(f"{owner} must contain a JSON object: {path}")
    return path, payload


def _resolve_reference(config_path: Path, value: Any, *, owner: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ReplicaCADPlanError(f"{owner} must be a non-empty path string")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    return _required_file(candidate, owner=owner)


def _config_stem(path: Path) -> str:
    name = path.name
    for suffix in (
        ".stage_config.json",
        ".object_config.json",
        ".ao_config.json",
        ".urdf",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _aliases(handle: str, path: Path, kind: str) -> tuple[str, ...]:
    stem = _config_stem(path)
    prefix = {"stage": "stages", "rigid": "objects"}.get(kind)
    values = {handle, Path(handle).name, stem}
    if prefix:
        values.add(f"{prefix}/{stem}")
    return tuple(sorted(value for value in values if value))


def _template_index(inputs: TemplateInputs, *, kind: str) -> dict[str, Path]:
    if isinstance(inputs, Mapping):
        entries = [(str(handle), Path(path)) for handle, path in inputs.items()]
    else:
        entries = [(_config_stem(Path(path)), Path(path)) for path in inputs]

    index: dict[str, Path] = {}
    for handle, unresolved in entries:
        path = _required_file(unresolved, owner=f"{kind} template {handle!r}")
        for alias in _aliases(handle, path, kind):
            previous = index.get(alias)
            if previous is not None and previous != path:
                raise ReplicaCADPlanError(
                    f"ambiguous {kind} template alias {alias!r}: "
                    f"{previous} and {path}"
                )
            index[alias] = path
    return index


def _declared_template_files(
    dataset_root: Path,
    dataset: Mapping[str, Any],
    *,
    section: str,
    extension: str,
) -> tuple[Path, ...]:
    section_payload = dataset.get(section)
    if not isinstance(section_payload, Mapping):
        raise ReplicaCADPlanError(f"dataset config lacks {section}.paths")
    paths = section_payload.get("paths")
    if not isinstance(paths, Mapping):
        raise ReplicaCADPlanError(f"dataset config lacks {section}.paths")
    declarations = paths.get(extension)
    if not isinstance(declarations, list) or not declarations:
        raise ReplicaCADPlanError(
            f"dataset config lacks {section}.paths[{extension!r}]"
        )

    discovered: set[Path] = set()
    for index, declaration in enumerate(declarations):
        if not isinstance(declaration, str) or not declaration.strip():
            raise ReplicaCADPlanError(
                f"dataset config {section} path {index} must be a string"
            )
        pattern_path = Path(declaration).expanduser()
        if not pattern_path.is_absolute():
            pattern_path = dataset_root / pattern_path
        matches = [Path(match) for match in glob.glob(str(pattern_path))]
        if not matches and pattern_path.exists():
            matches = [pattern_path]
        for match in matches:
            if match.is_dir():
                discovered.update(match.glob(f"*{extension}"))
            elif match.is_file() and match.name.endswith(extension):
                discovered.add(match)
    return tuple(sorted(path.resolve() for path in discovered))


def _lookup_template(
    template_name: Any,
    index: Mapping[str, Path],
    *,
    kind: str,
) -> tuple[str, Path]:
    if not isinstance(template_name, str) or not template_name.strip():
        raise ReplicaCADPlanError(f"{kind} instance template_name is missing")
    candidates = (template_name, Path(template_name).name)
    for candidate in candidates:
        if candidate in index:
            return template_name, index[candidate]
    raise ReplicaCADPlanError(
        f"{kind} instance references unknown template {template_name!r}"
    )


def _scale(mapping: Mapping[str, Any], *, owner: str) -> tuple[float, float, float]:
    if "scale" in mapping and "non_uniform_scale" in mapping:
        raise ReplicaCADPlanError(
            f"{owner} cannot declare both scale and non_uniform_scale"
        )
    non_uniform = mapping.get("non_uniform_scale", mapping.get("scale", [1, 1, 1]))
    x, y, z = _vector(non_uniform, 3, owner=f"{owner} non-uniform scale")
    uniform = _finite_number(mapping.get("uniform_scale", 1.0), owner=f"{owner} uniform scale")
    result = (uniform * x, uniform * y, uniform * z)
    if any(value <= 0.0 for value in result):
        raise ReplicaCADPlanError(f"{owner} scale components must be positive")
    return result


def _multiply_scale(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float]:
    values = tuple(
        float(a) * float(b) for a, b in zip(left, right, strict=True)
    )
    return (values[0], values[1], values[2])


def _transforms(
    instance: Mapping[str, Any],
    template: Mapping[str, Any],
    *,
    owner: str,
) -> tuple[HabitatTransform, UnrealTransform]:
    translation = _vector(
        instance.get("translation", [0, 0, 0]),
        3,
        owner=f"{owner} translation",
    )
    rotation = _vector(
        instance.get("rotation", [1, 0, 0, 0]),
        4,
        owner=f"{owner} rotation",
    )
    final_scale = _multiply_scale(
        _scale(template, owner=f"{owner} template"),
        _scale(instance, owner=f"{owner} instance"),
    )
    habitat = HabitatTransform(
        translation_m=translation,  # type: ignore[arg-type]
        rotation_wxyz=rotation,  # type: ignore[arg-type]
        scale_xyz=final_scale,
    )
    unreal = UnrealTransform(
        translation_cm=habitat_position_to_unreal_cm(translation),
        rotation_xyzw=habitat_quaternion_wxyz_to_unreal_xyzw(rotation),
        scale_xyz=habitat_scale_to_unreal(final_scale),
    )
    return habitat, unreal


def _instance_list(scene: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = scene.get(key, [])
    if not isinstance(value, list):
        raise ReplicaCADPlanError(f"scene instance {key} must be an array")
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ReplicaCADPlanError(f"scene instance {key}[{index}] must be an object")
        result.append(item)
    return result


def _template_render_mesh(
    config_path: Path, config: Mapping[str, Any], *, owner: str
) -> Path:
    return _resolve_reference(
        config_path,
        config.get("render_asset"),
        owner=f"{owner} render_asset",
    )


def _articulated_urdf(
    template_path: Path, *, template_name: str
) -> tuple[Path, Mapping[str, Any]]:
    if template_path.suffix.lower() == ".urdf":
        return template_path, {}
    config_path, config = _load_json(
        template_path, owner=f"articulated template {template_name!r}"
    )
    declared = config.get("urdf_filepath")
    if declared is not None:
        return (
            _resolve_reference(
                config_path,
                declared,
                owner=f"articulated template {template_name!r} urdf_filepath",
            ),
            config,
        )
    inferred = config_path.parent / f"{_config_stem(config_path)}.urdf"
    return (
        _required_file(
            inferred, owner=f"articulated template {template_name!r} inferred URDF"
        ),
        config,
    )


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


@dataclass(frozen=True)
class _URDFJoint:
    name: str
    joint_type: str
    parent_link: str
    child_link: str
    origin_matrix: tuple[tuple[float, float, float, float], ...]
    axis_xyz: tuple[float, float, float]
    lower: float | None
    upper: float | None


@dataclass(frozen=True)
class _URDFVisualSpec:
    visual_id: str
    link_name: str
    mesh_path: Path
    link_from_visual_matrix: tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class _URDFModel:
    mesh_paths: tuple[Path, ...]
    joints: tuple[_URDFJoint, ...]
    links: tuple[str, ...]
    visuals: tuple[_URDFVisualSpec, ...]


_Matrix4 = tuple[tuple[float, float, float, float], ...]


def _identity_matrix() -> _Matrix4:
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _matrix_multiply(left: _Matrix4, right: _Matrix4) -> _Matrix4:
    return tuple(
        tuple(
            sum(left[row][inner] * right[inner][column] for inner in range(4))
            for column in range(4)
        )
        for row in range(4)
    )


def _numbers_from_attribute(
    value: str | None,
    *,
    owner: str,
    default: Sequence[float],
) -> tuple[float, ...]:
    if value is None:
        return tuple(float(item) for item in default)
    tokens = value.replace(",", " ").split()
    if len(tokens) != len(default):
        raise ReplicaCADPlanError(
            f"{owner} must contain {len(default)} finite numbers"
        )
    try:
        result = tuple(float(token) for token in tokens)
    except ValueError as exc:
        raise ReplicaCADPlanError(f"{owner} contains a non-number") from exc
    if not all(math.isfinite(item) for item in result):
        raise ReplicaCADPlanError(f"{owner} contains a non-finite number")
    return result


def _rpy_matrix(rpy: Sequence[float]) -> _Matrix4:
    roll, pitch, yaw = (float(item) for item in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # URDF uses fixed-axis roll, pitch, yaw: Rz(yaw) Ry(pitch) Rx(roll).
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, 0.0),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, 0.0),
        (-sp, cp * sr, cp * cr, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _origin_matrix(node: ET.Element | None, *, owner: str) -> _Matrix4:
    if node is None:
        return _identity_matrix()
    xyz = _numbers_from_attribute(
        node.get("xyz"), owner=f"{owner} xyz", default=(0.0, 0.0, 0.0)
    )
    rpy = _numbers_from_attribute(
        node.get("rpy"), owner=f"{owner} rpy", default=(0.0, 0.0, 0.0)
    )
    rotation = [list(row) for row in _rpy_matrix(rpy)]
    for axis in range(3):
        rotation[axis][3] = xyz[axis]
    return tuple(tuple(float(item) for item in row) for row in rotation)


def _axis_motion_matrix(joint: _URDFJoint, position: float) -> _Matrix4:
    if joint.joint_type == "fixed":
        return _identity_matrix()
    axis = joint.axis_xyz
    norm = math.sqrt(sum(item * item for item in axis))
    if norm <= 1.0e-12:
        raise ReplicaCADPlanError(f"URDF joint {joint.name!r} has a zero axis")
    x, y, z = (item / norm for item in axis)
    if joint.joint_type == "prismatic":
        return (
            (1.0, 0.0, 0.0, x * position),
            (0.0, 1.0, 0.0, y * position),
            (0.0, 0.0, 1.0, z * position),
            (0.0, 0.0, 0.0, 1.0),
        )
    cosine, sine = math.cos(position), math.sin(position)
    one_minus = 1.0 - cosine
    return (
        (
            cosine + x * x * one_minus,
            x * y * one_minus - z * sine,
            x * z * one_minus + y * sine,
            0.0,
        ),
        (
            y * x * one_minus + z * sine,
            cosine + y * y * one_minus,
            y * z * one_minus - x * sine,
            0.0,
        ),
        (
            z * x * one_minus - y * sine,
            z * y * one_minus + x * sine,
            cosine + z * z * one_minus,
            0.0,
        ),
        (0.0, 0.0, 0.0, 1.0),
    )


def _read_urdf(
    path: Path, *, template_name: str
) -> _URDFModel:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ReplicaCADPlanError(
            f"articulated template {template_name!r} has unreadable URDF: {path}"
        ) from exc

    links: list[str] = []
    visuals: list[_URDFVisualSpec] = []
    meshes: list[Path] = []
    for link in (node for node in root if _local_name(node.tag) == "link"):
        link_name = link.get("name")
        if not link_name or link_name in links:
            raise ReplicaCADPlanError(
                f"articulated template {template_name!r} has an invalid link name"
            )
        links.append(link_name)
        link_visuals = [
            node for node in link if _local_name(node.tag) == "visual"
        ]
        for visual_index, visual in enumerate(link_visuals):
            geometry = next(
                (node for node in visual if _local_name(node.tag) == "geometry"),
                None,
            )
            mesh = (
                next(
                    (
                        node
                        for node in geometry
                        if _local_name(node.tag) == "mesh"
                    ),
                    None,
                )
                if geometry is not None
                else None
            )
            if mesh is None:
                raise ReplicaCADPlanError(
                    f"articulated template {template_name!r} link {link_name!r} "
                    "has a non-mesh visual"
                )
            filename = mesh.get("filename")
            if not filename or filename.startswith(("package://", "file://")):
                raise ReplicaCADPlanError(
                    f"articulated template {template_name!r} has unsupported visual mesh "
                    f"reference {filename!r}"
                )
            resolved = _required_file(
                path.parent / filename,
                owner=f"articulated template {template_name!r} visual mesh",
            )
            if resolved not in meshes:
                meshes.append(resolved)
            origin = next(
                (node for node in visual if _local_name(node.tag) == "origin"),
                None,
            )
            transform = _origin_matrix(
                origin,
                owner=(
                    f"articulated template {template_name!r} link "
                    f"{link_name!r} visual {visual_index} origin"
                ),
            )
            scale = _numbers_from_attribute(
                mesh.get("scale"),
                owner=(
                    f"articulated template {template_name!r} link "
                    f"{link_name!r} visual {visual_index} scale"
                ),
                default=(1.0, 1.0, 1.0),
            )
            if any(item <= 0.0 for item in scale):
                raise ReplicaCADPlanError("URDF visual mesh scale must be positive")
            scale_matrix: _Matrix4 = (
                (scale[0], 0.0, 0.0, 0.0),
                (0.0, scale[1], 0.0, 0.0),
                (0.0, 0.0, scale[2], 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
            visuals.append(
                _URDFVisualSpec(
                    visual_id=f"{link_name}:visual:{visual_index:03d}",
                    link_name=link_name,
                    mesh_path=resolved,
                    link_from_visual_matrix=_matrix_multiply(transform, scale_matrix),
                )
            )
    if not meshes:
        raise ReplicaCADPlanError(
            f"articulated template {template_name!r} URDF has no PBR visual meshes"
        )

    joints: list[_URDFJoint] = []
    for joint in (node for node in root if _local_name(node.tag) == "joint"):
        joint_type = joint.get("type", "")
        if joint_type == "floating" or joint_type not in {
            "fixed",
            "continuous",
            "prismatic",
            "revolute",
        }:
            raise ReplicaCADPlanError(
                f"articulated template {template_name!r} uses unsupported joint type "
                f"{joint_type!r}"
            )
        name = joint.get("name")
        if not name:
            raise ReplicaCADPlanError(
                f"articulated template {template_name!r} has an unnamed movable joint"
            )
        parent = next(
            (node for node in joint if _local_name(node.tag) == "parent"), None
        )
        child = next(
            (node for node in joint if _local_name(node.tag) == "child"), None
        )
        parent_link = parent.get("link") if parent is not None else None
        child_link = child.get("link") if child is not None else None
        if parent_link not in links or child_link not in links:
            raise ReplicaCADPlanError(
                f"articulated template {template_name!r} joint {name!r} "
                "references an unknown link"
            )
        origin = next(
            (node for node in joint if _local_name(node.tag) == "origin"), None
        )
        axis_node = next(
            (node for node in joint if _local_name(node.tag) == "axis"), None
        )
        axis = _numbers_from_attribute(
            axis_node.get("xyz") if axis_node is not None else None,
            owner=f"URDF joint {name} axis",
            default=(1.0, 0.0, 0.0),
        )
        limit = next(
            (node for node in joint if _local_name(node.tag) == "limit"), None
        )
        lower = upper = None
        if limit is not None and joint_type not in {"continuous", "fixed"}:
            if limit.get("lower") is not None:
                lower = _finite_number(
                    float(limit.get("lower", "")), owner=f"URDF joint {name} lower"
                )
            if limit.get("upper") is not None:
                upper = _finite_number(
                    float(limit.get("upper", "")), owner=f"URDF joint {name} upper"
                )
        joints.append(
            _URDFJoint(
                name=name,
                joint_type=joint_type,
                parent_link=str(parent_link),
                child_link=str(child_link),
                origin_matrix=_origin_matrix(
                    origin, owner=f"URDF joint {name} origin"
                ),
                axis_xyz=(axis[0], axis[1], axis[2]),
                lower=lower,
                upper=upper,
            )
        )
    child_links = [joint.child_link for joint in joints]
    if len(child_links) != len(set(child_links)):
        raise ReplicaCADPlanError(
            f"articulated template {template_name!r} has a multiply-parented link"
        )
    roots = [link for link in links if link not in child_links]
    if len(roots) != 1:
        raise ReplicaCADPlanError(
            f"articulated template {template_name!r} must have one root link"
        )
    return _URDFModel(
        mesh_paths=tuple(meshes),
        joints=tuple(joints),
        links=tuple(links),
        visuals=tuple(visuals),
    )


def _joint_defaults(
    instance: Mapping[str, Any],
    joints: Sequence[_URDFJoint],
    *,
    owner: str,
) -> tuple[ArticulatedJointDefault, ...]:
    raw = instance.get("initial_joint_pose")
    values = [0.0] * len(joints)
    sources = ["urdf_zero"] * len(joints)
    if isinstance(raw, list):
        if len(raw) != len(joints):
            raise ReplicaCADPlanError(
                f"{owner} initial_joint_pose has {len(raw)} values for "
                f"{len(joints)} movable joints"
            )
        values = [
            _finite_number(value, owner=f"{owner} initial_joint_pose[{index}]")
            for index, value in enumerate(raw)
        ]
        sources = ["scene_array"] * len(joints)
    elif isinstance(raw, Mapping):
        by_name = {joint.name: index for index, joint in enumerate(joints)}
        for key, value in raw.items():
            key_string = str(key)
            if key_string in by_name:
                index = by_name[key_string]
            elif key_string.isdecimal() and int(key_string) < len(joints):
                index = int(key_string)
            else:
                raise ReplicaCADPlanError(
                    f"{owner} initial_joint_pose references unknown joint {key_string!r}"
                )
            values[index] = _finite_number(
                value, owner=f"{owner} initial_joint_pose[{key_string!r}]"
            )
            sources[index] = "scene_mapping"
    elif raw is not None:
        raise ReplicaCADPlanError(f"{owner} initial_joint_pose must be an array or object")

    clamp = bool(instance.get("auto_clamp_joint_limits", False))
    result: list[ArticulatedJointDefault] = []
    for joint, value, source in zip(joints, values, sources, strict=True):
        realized = value
        if clamp and joint.lower is not None:
            realized = max(realized, joint.lower)
        if clamp and joint.upper is not None:
            realized = min(realized, joint.upper)
        result.append(
            ArticulatedJointDefault(
                joint_name=joint.name,
                joint_type=joint.joint_type,
                position=realized,
                source=source,
                clamped_to_limit=not math.isclose(realized, value, abs_tol=1.0e-12),
            )
        )
    return tuple(result)


def _articulated_visuals(
    model: _URDFModel,
    defaults: Sequence[ArticulatedJointDefault],
    *,
    owner: str,
) -> tuple[ArticulatedVisual, ...]:
    """Evaluate every URDF visual at the declared instance joint pose."""

    positions = {item.joint_name: item.position for item in defaults}
    children: dict[str, list[_URDFJoint]] = {link: [] for link in model.links}
    child_links: set[str] = set()
    for joint in model.joints:
        children[joint.parent_link].append(joint)
        child_links.add(joint.child_link)
    roots = [link for link in model.links if link not in child_links]
    if len(roots) != 1:
        raise ReplicaCADPlanError(f"{owner} URDF does not have one root")

    root_from_link: dict[str, _Matrix4] = {roots[0]: _identity_matrix()}
    pending = [roots[0]]
    while pending:
        parent = pending.pop()
        for joint in children[parent]:
            if joint.child_link in root_from_link:
                raise ReplicaCADPlanError(f"{owner} URDF joint graph contains a cycle")
            position = 0.0 if joint.joint_type == "fixed" else positions[joint.name]
            parent_from_child = _matrix_multiply(
                joint.origin_matrix, _axis_motion_matrix(joint, position)
            )
            root_from_link[joint.child_link] = _matrix_multiply(
                root_from_link[parent], parent_from_child
            )
            pending.append(joint.child_link)
    if set(root_from_link) != set(model.links):
        missing = sorted(set(model.links).difference(root_from_link))
        raise ReplicaCADPlanError(f"{owner} URDF has unreachable links: {missing}")

    return tuple(
        ArticulatedVisual(
            visual_id=visual.visual_id,
            link_name=visual.link_name,
            mesh_path=visual.mesh_path,
            root_from_visual_matrix=_matrix_multiply(
                root_from_link[visual.link_name], visual.link_from_visual_matrix
            ),
        )
        for visual in model.visuals
    )


def _optional_string(value: Any, *, owner: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ReplicaCADPlanError(f"{owner} must be a non-empty string")
    return value


def _fixed_base(instance: Mapping[str, Any], *, owner: str) -> bool | None:
    if "fixed_base" in instance:
        value = instance["fixed_base"]
        if not isinstance(value, bool):
            raise ReplicaCADPlanError(f"{owner} fixed_base must be boolean")
        return value
    base_type = instance.get("base_type")
    if base_type is None:
        return None
    if base_type not in {"fixed", "free"}:
        raise ReplicaCADPlanError(f"{owner} base_type must be 'fixed' or 'free'")
    return base_type == "fixed"


def build_replicacad_scene_plan(
    dataset_config_path: str | Path,
    scene_instance_path: str | Path,
    *,
    object_template_configs: TemplateInputs | None = None,
    articulated_template_configs: TemplateInputs | None = None,
    stage_template_configs: TemplateInputs | None = None,
) -> ReplicaCADScenePlan:
    """Build a deterministic, count-closed ReplicaCAD-to-UE data plan.

    Template inputs may be either ``{template_handle: config_path}`` mappings or
    iterables of config paths.  When omitted, paths are discovered only through
    the corresponding declarations in the supplied scene-dataset config.
    Articulated inputs may point to a URDF or an ``*.ao_config.json`` whose
    ``urdf_filepath`` is declared (or whose sibling URDF has the same stem).
    """

    dataset_path, dataset = _load_json(
        dataset_config_path, owner="ReplicaCAD scene dataset config"
    )
    scene_path, scene = _load_json(
        scene_instance_path, owner="ReplicaCAD scene instance"
    )
    dataset_root = dataset_path.parent

    if stage_template_configs is None:
        stage_template_configs = _declared_template_files(
            dataset_root, dataset, section="stages", extension=".json"
        )
    if object_template_configs is None:
        object_template_configs = _declared_template_files(
            dataset_root, dataset, section="objects", extension=".json"
        )
    if articulated_template_configs is None:
        articulated_template_configs = _declared_template_files(
            dataset_root,
            dataset,
            section="articulated_objects",
            extension=".urdf",
        )

    stage_index = _template_index(stage_template_configs, kind="stage")
    object_index = _template_index(object_template_configs, kind="rigid")
    articulated_index = _template_index(
        articulated_template_configs, kind="articulated"
    )

    stage_instance = scene.get("stage_instance")
    if not isinstance(stage_instance, Mapping):
        raise ReplicaCADPlanError("scene instance must contain one stage_instance")
    rigid_instances = _instance_list(scene, "object_instances")
    articulated_instances = _instance_list(scene, "articulated_object_instances")
    default_lighting = _optional_string(
        scene.get("default_lighting"), owner="scene default_lighting"
    )

    imports_by_id: dict[str, ReplicaCADImport] = {}
    spawns: list[ReplicaCADSpawn] = []

    stage_name, stage_config_path = _lookup_template(
        stage_instance.get("template_name"), stage_index, kind="stage"
    )
    _, stage_config = _load_json(stage_config_path, owner=f"stage template {stage_name!r}")
    stage_import_id = f"stage:{stage_name}"
    imports_by_id[stage_import_id] = ReplicaCADImport(
        import_id=stage_import_id,
        asset_kind="stage",
        template_name=stage_name,
        template_config_path=stage_config_path,
        pbr_mesh_paths=(
            _template_render_mesh(
                stage_config_path, stage_config, owner=f"stage template {stage_name!r}"
            ),
        ),
    )
    habitat, unreal = _transforms(
        stage_instance, stage_config, owner=f"stage instance {stage_name!r}"
    )
    spawns.append(
        ReplicaCADSpawn(
            spawn_id="stage:000000",
            asset_kind="stage",
            source_index=0,
            import_id=stage_import_id,
            template_name=stage_name,
            habitat_transform=habitat,
            unreal_transform=unreal,
            motion_type=_optional_string(
                stage_instance.get("motion_type"), owner="stage motion_type"
            ),
            translation_origin=_optional_string(
                stage_instance.get("translation_origin"),
                owner="stage translation_origin",
            ),
        )
    )

    rigid_cache: dict[Path, Mapping[str, Any]] = {}
    for source_index, instance in enumerate(rigid_instances):
        name, config_path = _lookup_template(
            instance.get("template_name"), object_index, kind="rigid"
        )
        config = rigid_cache.get(config_path)
        if config is None:
            _, config = _load_json(config_path, owner=f"rigid template {name!r}")
            rigid_cache[config_path] = config
        import_id = f"rigid:{name}"
        candidate_import = ReplicaCADImport(
            import_id=import_id,
            asset_kind="rigid",
            template_name=name,
            template_config_path=config_path,
            pbr_mesh_paths=(
                _template_render_mesh(
                    config_path, config, owner=f"rigid template {name!r}"
                ),
            ),
        )
        previous = imports_by_id.setdefault(import_id, candidate_import)
        if previous != candidate_import:
            raise ReplicaCADPlanError(f"rigid import identity collision: {import_id}")
        habitat, unreal = _transforms(
            instance, config, owner=f"rigid instance {source_index} ({name!r})"
        )
        spawns.append(
            ReplicaCADSpawn(
                spawn_id=f"rigid:{source_index:06d}",
                asset_kind="rigid",
                source_index=source_index,
                import_id=import_id,
                template_name=name,
                habitat_transform=habitat,
                unreal_transform=unreal,
                motion_type=_optional_string(
                    instance.get("motion_type"),
                    owner=f"rigid instance {source_index} motion_type",
                ),
                translation_origin=_optional_string(
                    instance.get("translation_origin"),
                    owner=f"rigid instance {source_index} translation_origin",
                ),
            )
        )

    articulated_cache: dict[
        Path, tuple[Mapping[str, Any], Path, _URDFModel]
    ] = {}
    for source_index, instance in enumerate(articulated_instances):
        name, template_path = _lookup_template(
            instance.get("template_name"),
            articulated_index,
            kind="articulated",
        )
        cached = articulated_cache.get(template_path)
        if cached is None:
            urdf_path, config = _articulated_urdf(template_path, template_name=name)
            model = _read_urdf(urdf_path, template_name=name)
            cached = (config, urdf_path, model)
            articulated_cache[template_path] = cached
        config, urdf_path, model = cached
        import_id = f"articulated:{name}"
        candidate_import = ReplicaCADImport(
            import_id=import_id,
            asset_kind="articulated",
            template_name=name,
            template_config_path=template_path,
            pbr_mesh_paths=model.mesh_paths,
            urdf_path=urdf_path,
        )
        previous = imports_by_id.setdefault(import_id, candidate_import)
        if previous != candidate_import:
            raise ReplicaCADPlanError(f"articulated import identity collision: {import_id}")
        owner = f"articulated instance {source_index} ({name!r})"
        habitat, unreal = _transforms(instance, config, owner=owner)
        auto_clamp = instance.get("auto_clamp_joint_limits", False)
        if not isinstance(auto_clamp, bool):
            raise ReplicaCADPlanError(f"{owner} auto_clamp_joint_limits must be boolean")
        movable_joints = tuple(
            joint for joint in model.joints if joint.joint_type != "fixed"
        )
        defaults = _joint_defaults(instance, movable_joints, owner=owner)
        spawns.append(
            ReplicaCADSpawn(
                spawn_id=f"articulated:{source_index:06d}",
                asset_kind="articulated",
                source_index=source_index,
                import_id=import_id,
                template_name=name,
                habitat_transform=habitat,
                unreal_transform=unreal,
                motion_type=_optional_string(
                    instance.get("motion_type"), owner=f"{owner} motion_type"
                ),
                translation_origin=_optional_string(
                    instance.get("translation_origin"),
                    owner=f"{owner} translation_origin",
                ),
                fixed_base=_fixed_base(instance, owner=owner),
                auto_clamp_joint_limits=auto_clamp,
                joint_defaults=defaults,
                articulated_visuals=_articulated_visuals(
                    model, defaults, owner=owner
                ),
            )
        )

    kind_order = {"stage": 0, "rigid": 1, "articulated": 2}
    imports = tuple(
        sorted(
            imports_by_id.values(),
            key=lambda item: (kind_order[item.asset_kind], item.template_name),
        )
    )
    plan = ReplicaCADScenePlan(
        schema=PLAN_SCHEMA,
        coordinate_convention=COORDINATE_CONVENTION,
        dataset_config_path=dataset_path,
        scene_instance_path=scene_path,
        default_lighting=default_lighting,
        imports=imports,
        spawns=tuple(spawns),
        source_stage_count=1,
        source_rigid_count=len(rigid_instances),
        source_articulated_count=len(articulated_instances),
    )
    plan.assert_closed()
    return plan


# Explicit alias for callers that want the optional backend in the function name.
build_spear_replicacad_plan = build_replicacad_scene_plan


__all__ = [
    "ArticulatedJointDefault",
    "ArticulatedVisual",
    "COORDINATE_CONVENTION",
    "HabitatTransform",
    "PLAN_SCHEMA",
    "ReplicaCADImport",
    "ReplicaCADPlanError",
    "ReplicaCADScenePlan",
    "ReplicaCADSpawn",
    "UnrealTransform",
    "build_replicacad_scene_plan",
    "build_spear_replicacad_plan",
    "habitat_position_to_unreal_cm",
    "habitat_quaternion_wxyz_to_unreal_xyzw",
    "habitat_scale_to_unreal",
]
