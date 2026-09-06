"""Small P12 helpers for preparing real skinned assets for Habitat M2.

The existing M2 spherical action contract intentionally rejects animated root
translations: root motion belongs to the actor route.  Several reviewed
Pixel3D exports still carry a small root translation channel, however.  This
module performs the narrow, explicit conversion that keeps the target mesh,
skin, and joint rotations intact while making that route ownership visible in
the evidence.  It never changes a source file in place and never infers a
breed or a semantic anchor.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
from typing import Any, Mapping

import numpy as np

from avengine.assets.glb import decode_accessor, extract_actions, load_glb, parse_glb
from avengine.assets.glb_write import build_glb


class P12HabitatAssetError(ValueError):
    """A P12 preparation input or output is not safe to use."""


def _record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "byte_size": resolved.stat().st_size,
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def _accessor_layout(document: Mapping[str, Any], accessor_index: int) -> tuple[int, int, int, int]:
    accessors = document.get("accessors")
    views = document.get("bufferViews")
    if not isinstance(accessors, list) or not isinstance(views, list):
        raise P12HabitatAssetError("GLB accessors/bufferViews are missing")
    try:
        accessor = accessors[accessor_index]
    except (IndexError, TypeError):
        raise P12HabitatAssetError(
            f"animation output accessor is out of range: {accessor_index}"
        ) from None
    if not isinstance(accessor, Mapping) or accessor.get("componentType") != 5126:
        raise P12HabitatAssetError("animation output must use FLOAT components")
    element_type = accessor.get("type")
    width = {"VEC3": 3, "VEC4": 4}.get(element_type)
    if width is None:
        raise P12HabitatAssetError(
            f"translation/rotation output type is unsupported: {element_type!r}"
        )
    view_index = accessor.get("bufferView")
    if isinstance(view_index, bool) or not isinstance(view_index, int):
        raise P12HabitatAssetError("animation output lacks a bufferView")
    try:
        view = views[view_index]
    except (IndexError, TypeError):
        raise P12HabitatAssetError("animation output bufferView is out of range") from None
    if not isinstance(view, Mapping) or view.get("buffer") != 0:
        raise P12HabitatAssetError("animation output must use embedded buffer 0")
    count = accessor.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise P12HabitatAssetError("animation output count is invalid")
    view_offset = view.get("byteOffset", 0)
    accessor_offset = accessor.get("byteOffset", 0)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (view_offset, accessor_offset)
    ):
        raise P12HabitatAssetError("animation output offsets are invalid")
    element_bytes = 4 * width
    stride = view.get("byteStride", element_bytes)
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < element_bytes:
        raise P12HabitatAssetError("animation output stride is invalid")
    view_length = view.get("byteLength")
    if isinstance(view_length, bool) or not isinstance(view_length, int) or view_length < 0:
        raise P12HabitatAssetError("animation output view length is invalid")
    first = int(view_offset) + int(accessor_offset)
    required = int(accessor_offset) + (int(count) - 1) * int(stride) + element_bytes
    if required > int(view_length):
        raise P12HabitatAssetError("animation output extends beyond its bufferView")
    return first, int(stride), int(count), width


def _write_float_accessor(
    document: Mapping[str, Any],
    binary: bytearray,
    accessor_index: int,
    values: np.ndarray,
) -> None:
    first, stride, count, width = _accessor_layout(document, accessor_index)
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (count, width) or not np.all(np.isfinite(array)):
        raise P12HabitatAssetError("replacement animation values have an invalid shape")
    packer = struct.Struct("<" + "f" * width)
    for index, row in enumerate(array):
        try:
            packer.pack_into(binary, first + index * stride, *row.tolist())
        except struct.error as exc:
            raise P12HabitatAssetError("replacement animation values exceed BIN") from exc


def normalize_dynamic_root_translations(
    source_glb: str | Path,
    output_glb: str | Path,
    report_path: str | Path,
) -> Path:
    """Freeze only dynamic translation channels on skin joints.

    The operation is intended for exports whose route owns actor translation.
    Rotation channels, mesh bytes, skin weights, and all source geometry remain
    unchanged.  A report records every changed channel and its measured range.
    """

    source = Path(source_glb).resolve()
    output = Path(output_glb).resolve()
    report = Path(report_path).resolve()
    if not source.is_file() or source.is_symlink():
        raise P12HabitatAssetError(f"source GLB is not a regular file: {source}")
    if output.exists() or output.is_symlink() or report.exists() or report.is_symlink():
        raise P12HabitatAssetError("P12 normalization refuses existing output paths")
    document = load_glb(source)
    actions = extract_actions(document)
    raw_nodes = document.json.get("nodes")
    if not isinstance(raw_nodes, list):
        raise P12HabitatAssetError("GLB nodes are missing")
    root = deepcopy(document.json)
    binary = bytearray(document.binary)
    changed: list[dict[str, Any]] = []
    seen_accessors: set[int] = set()
    for action in actions:
        for channel in action.channels:
            if channel.target_path != "translation":
                continue
            values = np.asarray(channel.values, dtype=np.float64)
            if values.ndim != 2 or values.shape[1] != 3:
                raise P12HabitatAssetError(
                    f"{action.name}/{channel.target_node_name} translation is not VEC3"
                )
            node = raw_nodes[channel.target_node_index]
            if not isinstance(node, Mapping):
                raise P12HabitatAssetError("animation target node is invalid")
            rest = np.asarray(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
            if rest.shape != (3,) or not np.all(np.isfinite(rest)):
                raise P12HabitatAssetError("animation target node translation is invalid")
            maximum_error = float(np.max(np.abs(values - rest)))
            if maximum_error <= 5.0e-5:
                continue
            accessor = channel.output_accessor_index
            if accessor in seen_accessors:
                # Shared output accessors must be rewritten once, but their
                # target defaults must agree or the input is ambiguous.
                continue
            _write_float_accessor(root, binary, accessor, np.repeat(rest[None, :], len(values), axis=0))
            seen_accessors.add(accessor)
            changed.append(
                {
                    "action": action.name,
                    "target_node_index": channel.target_node_index,
                    "target_node_name": channel.target_node_name,
                    "output_accessor_index": accessor,
                    "sample_count": len(values),
                    "rest_translation_m": rest.tolist(),
                    "maximum_original_delta_m": maximum_error,
                }
            )
    root["buffers"][0]["byteLength"] = len(binary)
    payload = build_glb(root, binary)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise P12HabitatAssetError(f"unable to publish normalized GLB: {output}") from exc
    readback = load_glb(output)
    for action in extract_actions(readback):
        for channel in action.channels:
            if channel.target_path != "translation":
                continue
            node = readback.json["nodes"][channel.target_node_index]
            rest = np.asarray(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
            values = np.asarray(channel.values, dtype=np.float64)
            if float(np.max(np.abs(values - rest))) > 5.0e-5:
                raise P12HabitatAssetError(
                    f"translation channel remains dynamic after normalization: {action.name}"
                )
    output_record = _record(output)
    value = {
        "schema": "avengine_m2_habitat_root_translation_normalization_v1",
        "status": "pass",
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "route_translation_authority": "actor_root_trajectory",
        "source": _record(source),
        "output": output_record,
        "changed_channels": changed,
        "changed_channel_count": len(changed),
        "notes": [
            "Only dynamic translation channel samples were replaced with their authored rest value.",
            "Mesh, skin weights, joint rotations, and source bytes remain the asset authority.",
            "The route planner must provide actor root translation for walking episodes.",
        ],
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return output


__all__ = ["P12HabitatAssetError", "normalize_dynamic_root_translations"]
