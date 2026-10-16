#!/usr/bin/env python3
"""Let the static finalizer stand an object up, not just turn it.

The finalizer applied a yaw and nothing else, so a reconstruction that arrived
pitched stayed pitched: measured 5.3 to 18.7 degrees across every published
static asset, which both leans the object visibly and costs it height, because
the uniform scale targets a bounding box that a leaning object inflates.

Heading evidence v2 carries the measured upright correction beside the reviewed
yaw. v1 is still accepted unchanged - another agent has authorities in flight -
and an asset without the correction behaves exactly as before.

Order matters. The reviewed yaw is an azimuth read off the tilted mesh, so
standing the object up first moves it. The fix rotates the yaw's own direction
vector by the upright correction and re-reads its azimuth, which lands the
front on +X exactly rather than approximately.
"""

from __future__ import annotations

from pathlib import Path

TOOL = Path(
    "/data/jzy/code/SPEAR-lead-b/tools/blender_finalize_generated_static_object.py"
)

OLD_SCHEMA = 'HEADING_EVIDENCE_SCHEMA = "avengine_static_heading_review_v1"\n'
NEW_SCHEMA = (
    'HEADING_EVIDENCE_SCHEMA = "avengine_static_heading_review_v1"\n'
    'HEADING_EVIDENCE_SCHEMA_V2 = "avengine_static_heading_review_v2"\n'
    "# A stably resting product view cannot arrive past this, so a larger\n"
    "# correction means a face was mis-selected upstream rather than measured.\n"
    "MAXIMUM_UPRIGHT_CORRECTION_DEG = 30.0\n"
)

OLD_VALIDATE = '''    payload = contract.load_json_object(path, "static heading review evidence")
    required = {
        "schema",
        "instance_id",
        "request_sha256",
        "profile_sha256",
        "input_glb_sha256",
        "review_artifact",
        "reviewed_source_front_yaw_deg",
        "target_front_axis",
        "decision",
        "formal_dataset_registration_authorized",
    }
    if set(payload) != required:
        raise contract.EmitterContractError("static heading evidence fields are invalid")
    if (
        payload["schema"] != HEADING_EVIDENCE_SCHEMA
        or payload["target_front_axis"] != "positive-x"
        or payload["decision"] != "approved_for_positive_x_normalization"
        or payload["formal_dataset_registration_authorized"] is not False
    ):
        raise contract.EmitterContractError("static heading evidence is not approved")
'''

NEW_VALIDATE = '''    payload = contract.load_json_object(path, "static heading review evidence")
    required = {
        "schema",
        "instance_id",
        "request_sha256",
        "profile_sha256",
        "input_glb_sha256",
        "review_artifact",
        "reviewed_source_front_yaw_deg",
        "target_front_axis",
        "decision",
        "formal_dataset_registration_authorized",
    }
    schema = payload.get("schema")
    if schema == HEADING_EVIDENCE_SCHEMA_V2:
        required = required | {"reviewed_upright_correction"}
    elif schema != HEADING_EVIDENCE_SCHEMA:
        raise contract.EmitterContractError("static heading evidence is not approved")
    if set(payload) != required:
        raise contract.EmitterContractError("static heading evidence fields are invalid")
    if (
        payload["target_front_axis"] != "positive-x"
        or payload["decision"] != "approved_for_positive_x_normalization"
        or payload["formal_dataset_registration_authorized"] is not False
    ):
        raise contract.EmitterContractError("static heading evidence is not approved")
    if schema == HEADING_EVIDENCE_SCHEMA_V2:
        validate_upright_correction(payload["reviewed_upright_correction"])
'''

VALIDATE_FN = '''def validate_upright_correction(value: Any) -> Vector:
    """Check the reviewed upright correction and return its up direction.

    The vector is the object's own up as measured on the raw reconstruction,
    in glTF axes. Everything else in the record is evidence for a reviewer;
    only this vector and the bound below change what the tool does.
    """

    if not isinstance(value, Mapping) or set(value) != {
        "measured_up_gltf",
        "tilt_from_upright_deg",
        "authority_disagreement_deg",
        "measurement_tool",
        "decision",
    }:
        raise contract.EmitterContractError(
            "reviewed upright correction fields are invalid"
        )
    if value["decision"] != "approved_for_upright_normalization":
        raise contract.EmitterContractError(
            "reviewed upright correction is not approved"
        )
    up_gltf = contract.require_finite_vector(
        value["measured_up_gltf"], 3, "measured up"
    )
    # glTF is y-up, Blender is z-up: (x, y, z) -> (x, -z, y).
    up = Vector((up_gltf[0], -up_gltf[2], up_gltf[1]))
    if up.length <= 1.0e-9:
        raise contract.EmitterContractError("measured up is degenerate")
    up.normalize()
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, up.z))))
    declared = contract.require_finite_vector(
        [value["tilt_from_upright_deg"]], 1, "declared tilt"
    )[0]
    if abs(tilt - declared) > 0.5:
        raise contract.EmitterContractError(
            "declared tilt does not match the measured up direction"
        )
    if tilt > MAXIMUM_UPRIGHT_CORRECTION_DEG:
        raise contract.EmitterContractError(
            "reviewed upright correction exceeds what a resting object can be"
        )
    return up


def real_meshes() -> list[Any]:'''

OLD_APPLY = '''    source_yaw = float(heading["reviewed_source_front_yaw_deg"])
    delta_yaw = -source_yaw
    transform_roots(Matrix.Rotation(math.radians(delta_yaw), 4, "Z"))
'''

NEW_APPLY = '''    source_yaw = float(heading["reviewed_source_front_yaw_deg"])
    correction = heading.get("reviewed_upright_correction")
    if correction is None:
        applied_upright_deg = 0.0
        delta_yaw = -source_yaw
        transform_roots(Matrix.Rotation(math.radians(delta_yaw), 4, "Z"))
    else:
        up = validate_upright_correction(correction)
        upright = up.rotation_difference(Vector((0.0, 0.0, 1.0))).to_matrix()
        applied_upright_deg = math.degrees(
            math.acos(max(-1.0, min(1.0, up.z)))
        )
        # The reviewed yaw is an azimuth in the tilted frame, so carry its own
        # direction through the upright rotation and re-read the azimuth there.
        front = Vector(
            (
                math.cos(math.radians(source_yaw)),
                math.sin(math.radians(source_yaw)),
                0.0,
            )
        )
        levelled = upright @ front
        delta_yaw = -math.degrees(math.atan2(levelled.y, levelled.x))
        transform_roots(
            Matrix.Rotation(math.radians(delta_yaw), 4, "Z")
            @ upright.to_4x4()
        )
'''

OLD_MANIFEST = '''            "reviewed_source_front_yaw_deg": source_yaw,
            "target_front_axis": "positive-x",
            "applied_world_z_yaw_deg": delta_yaw,
'''
NEW_MANIFEST = '''            "reviewed_source_front_yaw_deg": source_yaw,
            "target_front_axis": "positive-x",
            "applied_world_z_yaw_deg": delta_yaw,
            "applied_upright_correction_deg": applied_upright_deg,
'''

text = TOOL.read_text(encoding="utf-8")
for old, new in (
    (OLD_SCHEMA, NEW_SCHEMA),
    (OLD_VALIDATE, NEW_VALIDATE),
    ("def real_meshes() -> list[Any]:", VALIDATE_FN),
    (OLD_APPLY, NEW_APPLY),
    (OLD_MANIFEST, NEW_MANIFEST),
):
    if text.count(old) != 1:
        raise SystemExit(f"anchor matched {text.count(old)} times: {old[:60]!r}")
    text = text.replace(old, new)
TOOL.write_text(text, encoding="utf-8")
print("finalizer now applies a reviewed upright correction")
