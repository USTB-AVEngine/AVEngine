"""Independent byte-level verification of an appearance realization.

The Blender report is evidence, not an oracle.  This module derives the
expected geometry and texture directly from the authenticated source GLB and
the selected instance request, then compares those expectations with the
published GLB and standalone PNG.
"""

from __future__ import annotations

from copy import deepcopy
import io
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

from avengine.m2.action_rebind import (
    ActionRebindError,
    _accessor_array,
    _skinned_primitive,
    verify_appearance_glb_compatibility,
)
from avengine.m2.glb import GlbDocument, GlbError, extract_skins, parse_glb


_FLOAT_TOLERANCE = 5.0e-5
_PIXEL_CHANNEL_TOLERANCE = 6.0e-3
_LUMINANCE_WEIGHTS = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)


class AppearanceRealizationError(RuntimeError):
    """Published appearance bytes do not realize the authenticated request."""


def _operation(request: Mapping[str, Any], axis: str) -> Mapping[str, Any]:
    operations = request.get("realization_operations")
    if not isinstance(operations, Sequence) or isinstance(operations, (str, bytes)):
        raise AppearanceRealizationError("instance request operations are invalid")
    matches = [
        item
        for item in operations
        if isinstance(item, Mapping) and item.get("attribute") == axis
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("parameters"), Mapping):
        raise AppearanceRealizationError(
            f"instance request must contain one {axis!r} operation"
        )
    return matches[0]["parameters"]


def _finite_parameter(parameters: Mapping[str, Any], name: str) -> float:
    value = parameters.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise AppearanceRealizationError(f"requested {name} is not finite")
    return float(value)


def _document(payload: bytes, owner: str) -> GlbDocument:
    try:
        return parse_glb(payload)
    except GlbError as exc:
        raise AppearanceRealizationError(f"{owner} is not a valid GLB: {exc}") from exc


def _substantive_bytes_changed(source: GlbDocument, output: GlbDocument) -> None:
    """Reject an output whose only edit is disposable asset metadata."""

    source_json = deepcopy(source.json)
    output_json = deepcopy(output.json)
    for value in (source_json, output_json):
        asset = value.get("asset")
        if isinstance(asset, dict):
            asset.pop("generator", None)
            asset.pop("copyright", None)
    if source_json == output_json and source.binary == output.binary:
        raise AppearanceRealizationError(
            "appearance output only changes asset metadata; no realization was emitted"
        )


def _joint_names(document: GlbDocument) -> tuple[str, ...]:
    try:
        skins = extract_skins(document)
    except GlbError as exc:
        raise AppearanceRealizationError(f"source skin is invalid: {exc}") from exc
    if len(skins) != 1:
        raise AppearanceRealizationError("appearance source must contain one skin")
    names = tuple(joint.name for joint in skins[0].joints)
    if any(not isinstance(name, str) or not name for name in names):
        raise AppearanceRealizationError("source skin joint names must be non-empty")
    if len(set(names)) != len(names):
        raise AppearanceRealizationError("source skin joint names must be unique")
    return names  # type: ignore[return-value]


def _resolve_semantic_names(
    names: tuple[str, ...], requested: Any, *, owner: str
) -> tuple[str, ...]:
    if (
        not isinstance(requested, Sequence)
        or isinstance(requested, (str, bytes))
        or not requested
    ):
        raise AppearanceRealizationError(f"{owner} must be a non-empty list")
    resolved: list[str] = []
    for value in requested:
        if not isinstance(value, str) or not value:
            raise AppearanceRealizationError(f"{owner} contains an invalid name")
        folded = value.casefold()
        matches = [
            name
            for name in names
            if name.casefold() == folded or name.casefold().endswith(f" {folded}")
        ]
        if len(matches) != 1:
            raise AppearanceRealizationError(
                f"{owner} semantic name {value!r} resolves to {matches!r}"
            )
        resolved.append(matches[0])
    if len(set(resolved)) != len(resolved):
        raise AppearanceRealizationError(f"{owner} resolves duplicate joints")
    return tuple(resolved)


def _semantic_weights(
    joint_ordinals: np.ndarray,
    weights: np.ndarray,
    names: tuple[str, ...],
    selected: tuple[str, ...],
) -> np.ndarray:
    ordinals = {index for index, name in enumerate(names) if name in selected}
    if len(ordinals) != len(selected):
        raise AppearanceRealizationError("semantic joints differ from source skin")
    mask = np.isin(joint_ordinals, list(ordinals))
    result = np.clip(np.sum(np.where(mask, weights, 0.0), axis=1), 0.0, 1.0)
    if float(result.sum()) <= 1.0e-12:
        raise AppearanceRealizationError("semantic skin weights sum to zero")
    return result


def _weighted_rms(
    coordinates: np.ndarray, weights: np.ndarray, axes: tuple[int, ...]
) -> float:
    total = float(weights.sum())
    center = np.sum(coordinates * weights[:, None], axis=0) / total
    offsets = coordinates[:, list(axes)] - center[list(axes)]
    return float(
        np.sqrt(np.sum(weights * np.sum(np.square(offsets), axis=1)) / total)
    )


def _geometry_audit(
    source: GlbDocument,
    output: GlbDocument,
    *,
    body: Mapping[str, Any],
    life: Mapping[str, Any],
    size: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    try:
        _source_primitive, source_attributes = _skinned_primitive(
            source, owner="source GLB"
        )
        _output_primitive, output_attributes = _skinned_primitive(
            output, owner="output GLB"
        )
        source_positions = _accessor_array(
            source, source_attributes["POSITION"], owner="source POSITION"
        ).astype(np.float64)
        output_positions = _accessor_array(
            output, output_attributes["POSITION"], owner="output POSITION"
        ).astype(np.float64)
        joint_ordinals = _accessor_array(
            source, source_attributes["JOINTS_0"], owner="source JOINTS_0"
        ).astype(np.int64)
        weights = _accessor_array(
            source, source_attributes["WEIGHTS_0"], owner="source WEIGHTS_0"
        ).astype(np.float64)
        output_weights = _accessor_array(
            output, output_attributes["WEIGHTS_0"], owner="output WEIGHTS_0"
        ).astype(np.float64)
        normals = _accessor_array(
            output, output_attributes["NORMAL"], owner="output NORMAL"
        ).astype(np.float64)
    except (ActionRebindError, KeyError) as exc:
        raise AppearanceRealizationError(
            f"appearance geometry accessor contract is invalid: {exc}"
        ) from exc
    if joint_ordinals.shape != weights.shape or joint_ordinals.shape[1:] != (4,):
        raise AppearanceRealizationError("JOINTS_0/WEIGHTS_0 must be VEC4")
    names = _joint_names(source)
    if np.any(joint_ordinals < 0) or np.any(joint_ordinals >= len(names)):
        raise AppearanceRealizationError("JOINTS_0 contains an invalid ordinal")
    torso_names = _resolve_semantic_names(
        names, body.get("semantic_joint_names"), owner="body semantic joints"
    )
    head_names = _resolve_semantic_names(
        names, life.get("semantic_joint_names"), owner="life-stage semantic joints"
    )
    torso = _semantic_weights(joint_ordinals, weights, names, torso_names)
    head = _semantic_weights(joint_ordinals, weights, names, head_names)
    torso_count = int(np.count_nonzero(torso > 0.05))
    head_count = int(np.count_nonzero(head > 0.05))
    if torso_count < 50 or head_count < 20:
        raise AppearanceRealizationError(
            f"semantic skin coverage is too small: torso={torso_count}, head={head_count}"
        )

    torso_scale = _finite_parameter(body, "torso_girth_scale")
    head_scale = _finite_parameter(life, "head_scale")
    size_scale = _finite_parameter(size, "scale_ratio")
    blender_before = np.column_stack(
        (source_positions[:, 0], -source_positions[:, 2], source_positions[:, 1])
    )
    transformed = blender_before.copy()
    torso_before = _weighted_rms(blender_before, torso, (1, 2))
    head_before = _weighted_rms(blender_before, head, (0, 1, 2))
    torso_center = np.sum(transformed * torso[:, None], axis=0) / float(torso.sum())
    transformed[:, 1] = torso_center[1] + (
        transformed[:, 1] - torso_center[1]
    ) * (1.0 + (torso_scale - 1.0) * torso)
    transformed[:, 2] = torso_center[2] + (
        transformed[:, 2] - torso_center[2]
    ) * (1.0 + (torso_scale - 1.0) * 0.55 * torso)
    head_center = np.sum(transformed * head[:, None], axis=0) / float(head.sum())
    head_factor = 1.0 + (head_scale - 1.0) * head
    transformed = head_center + (transformed - head_center) * head_factor[:, None]
    torso_after = _weighted_rms(transformed, torso, (1, 2))
    head_after = _weighted_rms(transformed, head, (0, 1, 2))
    scaled = transformed * size_scale
    expected_positions = np.column_stack((scaled[:, 0], scaled[:, 2], -scaled[:, 1]))
    if expected_positions.shape != output_positions.shape:
        raise AppearanceRealizationError("appearance output POSITION shape changed")
    position_error = float(np.max(np.abs(expected_positions - output_positions)))
    if position_error > _FLOAT_TOLERANCE:
        raise AppearanceRealizationError(
            "appearance output POSITION does not realize source+request: "
            f"maximum error={position_error:.9g}"
        )
    normal_error = float(np.max(np.abs(np.linalg.norm(normals, axis=1) - 1.0)))
    if normal_error > 5.0e-4:
        raise AppearanceRealizationError("appearance output normals are not normalized")
    return (
        {
            "torso_group_names": list(torso_names),
            "head_group_names": list(head_names),
            "torso_selected_vertices": torso_count,
            "head_selected_vertices": head_count,
            "requested_torso_girth_scale": torso_scale,
            "requested_head_scale": head_scale,
            "torso_weighted_yz_rms_before": torso_before,
            "torso_weighted_yz_rms_after": torso_after,
            "torso_weighted_yz_rms_ratio": torso_after / torso_before,
            "head_weighted_radius_rms_before": head_before,
            "head_weighted_radius_rms_after": head_after,
            "head_weighted_radius_rms_ratio": head_after / head_before,
            "maximum_expected_position_error_m": position_error,
            "maximum_output_normal_norm_error": normal_error,
            "maximum_raw_weights_0_error": float(
                np.max(np.abs(weights - output_weights))
            ),
            "torso_weight_sum": float(torso.sum()),
            "head_weight_sum": float(head.sum()),
        },
        transformed,
        torso,
        head,
    )


def _image_payload(document: GlbDocument, image_index: Any, *, owner: str) -> bytes:
    images = document.json.get("images")
    views = document.json.get("bufferViews")
    if (
        not isinstance(images, list)
        or not isinstance(views, list)
        or isinstance(image_index, bool)
        or not isinstance(image_index, int)
        or not 0 <= image_index < len(images)
        or not isinstance(images[image_index], Mapping)
    ):
        raise AppearanceRealizationError(f"{owner} image index is invalid")
    image = images[image_index]
    view_index = image.get("bufferView")
    if "uri" in image or not isinstance(view_index, int) or not 0 <= view_index < len(views):
        raise AppearanceRealizationError(f"{owner} must be an embedded bufferView image")
    view = views[view_index]
    if not isinstance(view, Mapping) or view.get("buffer", 0) != 0:
        raise AppearanceRealizationError(f"{owner} image bufferView is invalid")
    start = view.get("byteOffset", 0)
    length = view.get("byteLength")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or start < 0
        or isinstance(length, bool)
        or not isinstance(length, int)
        or length <= 0
        or start + length > len(document.binary)
    ):
        raise AppearanceRealizationError(f"{owner} image escapes the GLB BIN chunk")
    return document.binary[start : start + length]


def _role_image(document: GlbDocument, role: str) -> bytes:
    materials = document.json.get("materials")
    textures = document.json.get("textures")
    if not isinstance(materials, list) or len(materials) != 1 or not isinstance(textures, list):
        raise AppearanceRealizationError("appearance GLB must contain one material")
    material = materials[0]
    if not isinstance(material, Mapping):
        raise AppearanceRealizationError("appearance material is invalid")
    pbr = material.get("pbrMetallicRoughness")
    if not isinstance(pbr, Mapping):
        raise AppearanceRealizationError("appearance PBR material is invalid")
    if role == "base_color":
        info = pbr.get("baseColorTexture")
    elif role == "normal":
        info = material.get("normalTexture")
    elif role == "specular":
        extensions = material.get("extensions")
        specular = (
            extensions.get("KHR_materials_specular")
            if isinstance(extensions, Mapping)
            else None
        )
        info = specular.get("specularTexture") if isinstance(specular, Mapping) else None
    else:  # pragma: no cover - internal caller owns the role vocabulary
        raise AssertionError(role)
    texture_index = info.get("index") if isinstance(info, Mapping) else None
    if (
        isinstance(texture_index, bool)
        or not isinstance(texture_index, int)
        or not 0 <= texture_index < len(textures)
        or not isinstance(textures[texture_index], Mapping)
    ):
        raise AppearanceRealizationError(f"appearance {role} texture is invalid")
    image_index = textures[texture_index].get("source")
    return _image_payload(document, image_index, owner=role)


def _muzzle_mask(
    source: GlbDocument,
    transformed: np.ndarray,
    head: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        primitive, attributes = _skinned_primitive(source, owner="source GLB")
        indices = _accessor_array(
            source, primitive["indices"], owner="source indices"
        ).astype(np.int64).reshape((-1, 3))
        uv = _accessor_array(
            source, attributes["TEXCOORD_0"], owner="source TEXCOORD_0"
        ).astype(np.float64)
    except (ActionRebindError, KeyError, ValueError) as exc:
        raise AppearanceRealizationError(f"muzzle raster input is invalid: {exc}") from exc
    uv_minimum = float(np.min(uv))
    uv_maximum = float(np.max(uv))
    if uv_minimum < -1.0e-6 or uv_maximum > 1.0 + 1.0e-6:
        raise AppearanceRealizationError("muzzle rasterizer requires UVs in [0, 1]")
    selected = np.flatnonzero(head > 0.05)
    head_x = transformed[selected, 0]
    threshold = float(np.quantile(head_x, 0.58))
    span = max(float(head_x.max()) - threshold, 1.0e-9)
    vertex_mask = np.clip(
        head * np.clip((transformed[:, 0] - threshold) / span, 0.0, 1.0),
        0.0,
        1.0,
    )
    mask = np.zeros((height, width), dtype=np.float32)
    for vertices in indices:
        values = vertex_mask[vertices]
        if float(values.max()) <= 0.01:
            continue
        triangle_uv = np.clip(uv[vertices], 0.0, 1.0)
        points = np.column_stack(
            (triangle_uv[:, 0] * (width - 1), triangle_uv[:, 1] * (height - 1))
        )
        x0 = max(0, int(math.floor(points[:, 0].min())))
        x1 = min(width - 1, int(math.ceil(points[:, 0].max())))
        y0 = max(0, int(math.floor(points[:, 1].min())))
        y1 = min(height - 1, int(math.ceil(points[:, 1].max())))
        denominator = (points[1, 1] - points[2, 1]) * (
            points[0, 0] - points[2, 0]
        ) + (points[2, 0] - points[1, 0]) * (points[0, 1] - points[2, 1])
        if abs(float(denominator)) < 1.0e-8:
            continue
        yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        b0 = (
            (points[1, 1] - points[2, 1]) * (xx - points[2, 0])
            + (points[2, 0] - points[1, 0]) * (yy - points[2, 1])
        ) / denominator
        b1 = (
            (points[2, 1] - points[0, 1]) * (xx - points[2, 0])
            + (points[0, 0] - points[2, 0]) * (yy - points[2, 1])
        ) / denominator
        b2 = 1.0 - b0 - b1
        inside = (b0 >= -1.0e-5) & (b1 >= -1.0e-5) & (b2 >= -1.0e-5)
        interpolated = b0 * values[0] + b1 * values[1] + b2 * values[2]
        region = mask[y0 : y1 + 1, x0 : x1 + 1]
        np.maximum(region, np.where(inside, interpolated, 0.0), out=region)
    count = int(np.count_nonzero(mask > 0.01))
    if count < 32:
        raise AppearanceRealizationError("semantic muzzle mask is unexpectedly empty")
    return mask, {
        "muzzle_forward_quantile": 0.58,
        "muzzle_mask_nonzero_pixels": count,
        "muzzle_mask_max": float(mask.max()),
        "uv_minimum": uv_minimum,
        "uv_maximum": uv_maximum,
        "uv_addressing_assumption": "non_tiled_clamp_0_1",
    }


def _texture_audit(
    source: GlbDocument,
    output: GlbDocument,
    standalone: bytes,
    *,
    coat: Mapping[str, Any],
    life: Mapping[str, Any],
    transformed: np.ndarray,
    head: np.ndarray,
) -> dict[str, Any]:
    source_base = _role_image(source, "base_color")
    output_base = _role_image(output, "base_color")
    if output_base != standalone:
        raise AppearanceRealizationError(
            "embedded base-color image bytes differ from standalone PNG"
        )
    for role in ("normal", "specular"):
        if _role_image(source, role) != _role_image(output, role):
            raise AppearanceRealizationError(f"appearance {role} image bytes changed")
    try:
        source_image = Image.open(io.BytesIO(source_base)).convert("RGBA")
        output_image = Image.open(io.BytesIO(output_base)).convert("RGBA")
    except (OSError, UnidentifiedImageError) as exc:
        raise AppearanceRealizationError(f"base-color PNG is invalid: {exc}") from exc
    if source_image.size != output_image.size:
        raise AppearanceRealizationError("base-color texture resolution changed")
    width, height = source_image.size
    mask, mask_record = _muzzle_mask(
        source, transformed, head, width=width, height=height
    )
    gain = _finite_parameter(coat, "luminance_gain")
    desaturation = _finite_parameter(life, "coat_desaturation")
    gray_mix = _finite_parameter(life, "muzzle_gray_mix")
    gray_target = _finite_parameter(life, "muzzle_gray_target")
    preserve_pattern = coat.get("preserve_pattern")
    if preserve_pattern != "tricolor":
        raise AppearanceRealizationError(
            "registered Beagle realizer requires preserve_pattern='tricolor'"
        )

    pigmented_count = 0
    white_count = 0
    dark_count = 0
    warm_count = 0
    before_sum = 0.0
    expected_after_sum = 0.0
    maximum_pixel_error = 0.0
    chunk_rows = 64
    for y0 in range(0, height, chunk_rows):
        y1 = min(height, y0 + chunk_rows)
        source_srgb = np.asarray(source_image.crop((0, y0, width, y1)), dtype=np.float32)
        output_srgb = np.asarray(output_image.crop((0, y0, width, y1)), dtype=np.float32)
        if not np.array_equal(source_srgb[:, :, 3], output_srgb[:, :, 3]):
            raise AppearanceRealizationError(
                "base-color alpha channel changed across appearance realization"
            )
        source_srgb = source_srgb[:, :, :3]
        output_srgb = output_srgb[:, :, :3]
        # Blender's ``Image.pixels`` API exposes the same normalized channel
        # values that are stored in this source PNG for this admitted sRGB
        # workflow.  The producer applies its operations in that channel
        # domain, so reproduce it directly rather than applying a second
        # transfer function here.
        source_rgb = source_srgb / 255.0
        actual_rgb = output_srgb / 255.0
        maximum = source_rgb.max(axis=2)
        minimum = source_rgb.min(axis=2)
        saturation = (maximum - minimum) / np.maximum(maximum, 1.0e-6)
        luminance = np.sum(source_rgb * _LUMINANCE_WEIGHTS, axis=2)
        white = (luminance > 0.52) & (saturation < 0.20)
        pigmented = (~white) & (maximum > 0.01)
        # The source PNG is quantized to 8-bit while Blender classifies its
        # decoded float pixels.  A handful of texels can land on the opposite
        # side of the 0.52/0.20/0.01 boundaries after that round-trip.  Admit
        # either classification only inside one channel quantum of a boundary;
        # all other pixels have one deterministic expected value.
        ambiguous = (
            (np.abs(luminance - 0.52) <= 1.0 / 255.0)
            | (np.abs(saturation - 0.20) <= 1.0 / 255.0)
            | (np.abs(maximum - 0.01) <= 1.0 / 255.0)
        )

        def coat_transform(classification: np.ndarray) -> np.ndarray:
            result = np.clip(
                source_rgb
                * (
                    1.0
                    + (gain - 1.0)
                    * classification[:, :, None].astype(np.float32)
                ),
                0.0,
                1.0,
            )
            if desaturation > 0.0:
                adjusted = np.sum(result * _LUMINANCE_WEIGHTS, axis=2)
                amount = (
                    classification[:, :, None].astype(np.float32) * desaturation
                )
                result = np.clip(
                    result * (1.0 - amount) + adjusted[:, :, None] * amount,
                    0.0,
                    1.0,
                )
            return result

        expected = coat_transform(pigmented)
        alternate = coat_transform(~pigmented)
        if gray_mix > 0.0:
            amount = np.clip(mask[y0:y1] * gray_mix, 0.0, 1.0)[:, :, None]
            expected = np.clip(
                expected * (1.0 - amount) + gray_target * amount,
                0.0,
                1.0,
            )
            alternate = np.clip(
                alternate * (1.0 - amount) + gray_target * amount,
                0.0,
                1.0,
            )
        error = np.max(np.abs(expected - actual_rgb), axis=2)
        alternate_error = np.max(np.abs(alternate - actual_rgb), axis=2)
        error = np.where(ambiguous, np.minimum(error, alternate_error), error)
        maximum_pixel_error = max(maximum_pixel_error, float(np.max(error)))
        expected_luminance = np.sum(expected * _LUMINANCE_WEIGHTS, axis=2)
        pigmented_count += int(np.count_nonzero(pigmented))
        white_count += int(np.count_nonzero(white))
        dark_count += int(np.count_nonzero(pigmented & (luminance < 0.12)))
        warm_count += int(
            np.count_nonzero(
                pigmented
                & (source_rgb[:, :, 0] > source_rgb[:, :, 2] * 1.15)
                & (source_rgb[:, :, 0] > source_rgb[:, :, 1] * 1.03)
            )
        )
        before_sum += float(np.sum(luminance[pigmented], dtype=np.float64))
        expected_after_sum += float(
            np.sum(expected_luminance[pigmented], dtype=np.float64)
        )
    if maximum_pixel_error > _PIXEL_CHANNEL_TOLERANCE:
        raise AppearanceRealizationError(
            "base-color pixels do not realize source+request: "
            f"maximum channel error={maximum_pixel_error:.9g}"
        )
    if min(pigmented_count, white_count, dark_count, warm_count) < 100:
        raise AppearanceRealizationError(
            "source texture does not contain measurable reviewed tricolor regions"
        )
    return {
        "resolution": [width, height],
        "luminance_gain": gain,
        "coat_desaturation": desaturation,
        "muzzle_gray_mix": gray_mix,
        "muzzle_gray_target": gray_target,
        "preserve_pattern": preserve_pattern,
        "pigmented_pixel_count": pigmented_count,
        "white_pixel_count": white_count,
        "dark_pixel_count": dark_count,
        "warm_pixel_count": warm_count,
        "mean_pigmented_luminance_before": before_sum / pigmented_count,
        "mean_pigmented_luminance_after": expected_after_sum / pigmented_count,
        "maximum_expected_pixel_channel_error": maximum_pixel_error,
        **mask_record,
    }


def verify_appearance_realization(
    *,
    source_path: str | Path,
    output_path: str | Path,
    source_payload: bytes,
    output_payload: bytes,
    standalone_texture_payload: bytes,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Return independently derived metrics or fail on any byte mismatch."""

    size = _operation(request, "size")
    body = _operation(request, "body_build")
    coat = _operation(request, "coat_profile")
    life = _operation(request, "life_stage")
    size_scale = _finite_parameter(size, "scale_ratio")
    try:
        compatibility = verify_appearance_glb_compatibility(
            source_path,
            output_path,
            requested_size_scale=size_scale,
            source_payload=source_payload,
            output_payload=output_payload,
        )
    except ActionRebindError as exc:
        raise AppearanceRealizationError(
            f"appearance topology/skin/action compatibility failed: {exc}"
        ) from exc
    source = _document(source_payload, "appearance source")
    output = _document(output_payload, "appearance output")
    _substantive_bytes_changed(source, output)
    geometry, transformed, _torso, head = _geometry_audit(
        source, output, body=body, life=life, size=size
    )
    texture = _texture_audit(
        source,
        output,
        standalone_texture_payload,
        coat=coat,
        life=life,
        transformed=transformed,
        head=head,
    )
    return {
        "compatibility": compatibility,
        "shape": geometry,
        "texture": texture,
    }
