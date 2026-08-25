#!/usr/bin/env python3
"""Append one 15 Hz return-to-start sample to each GLB action channel.

The compiler preserves every source byte and key, appends new accessors, and
points animation samplers at those accessors.  It is a research-only repair for
clips whose authored final state is not their initial loop state.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from avengine.contracts.json_io import canonical_json_sha256, sha256_file
from avengine.assets.glb import decode_accessor, extract_actions, load_glb, parse_glb
from avengine.assets.glb_write import build_glb


SCHEMA = "avengine_m2_loop_closure_append_v1"
_TIME_BASE_HZ = 48_000
_TICKS_PER_SAMPLE = 3_200
_TICK_TOLERANCE = 9.9e-3


def _objects(value: Any, *, owner: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{owner} must be an array of objects")
    return value


def _append_accessor(
    *,
    binary: bytearray,
    views: list[dict[str, Any]],
    accessors: list[dict[str, Any]],
    values: np.ndarray,
    element_type: str,
    include_bounds: bool,
) -> int:
    array = np.ascontiguousarray(values, dtype="<f4")
    offset = len(binary)
    payload = array.tobytes(order="C")
    binary.extend(payload)
    view_index = len(views)
    views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(payload)})
    accessor: dict[str, Any] = {
        "bufferView": view_index,
        "componentType": 5126,
        "count": int(array.shape[0]),
        "type": element_type,
    }
    if include_bounds:
        accessor["min"] = [float(array.min())]
        accessor["max"] = [float(array.max())]
    index = len(accessors)
    accessors.append(accessor)
    return index


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        if created:
            try:
                path.unlink()
            except OSError:
                pass
        raise


def _compatible_appended_time(start: float, end: float) -> np.float32:
    source_ticks = int(round((end - start) * _TIME_BASE_HZ))
    if source_ticks <= 0 or source_ticks % _TICKS_PER_SAMPLE:
        raise ValueError("source action duration is not aligned to the 15 Hz clock")
    for interval_count in range(1, 33):
        target_ticks = source_ticks + interval_count * _TICKS_PER_SAMPLE
        candidate = np.float32(start + target_ticks / _TIME_BASE_HZ)
        exact_ticks = (float(candidate) - start) * _TIME_BASE_HZ
        rounded_ticks = int(round(exact_ticks))
        if (
            abs(exact_ticks - rounded_ticks) <= _TICK_TOLERANCE
            and rounded_ticks == target_ticks
        ):
            return candidate
    raise ValueError("no float32 loop endpoint aligns to the 15 Hz clock")


def compile_loop_closure(source_path: Path, output_path: Path) -> dict[str, Any]:
    source = load_glb(source_path)
    document = copy.deepcopy(source.json)
    buffers = _objects(document.get("buffers"), owner="buffers")
    views = _objects(document.get("bufferViews"), owner="bufferViews")
    accessors = _objects(document.get("accessors"), owner="accessors")
    animations = _objects(document.get("animations"), owner="animations")
    if len(buffers) != 1 or buffers[0].get("uri") is not None:
        raise ValueError("input must use one embedded GLB buffer")
    if {animation.get("name") for animation in animations} != {"Idle", "Walking"}:
        raise ValueError("input must contain exactly Idle and Walking")

    binary = bytearray(source.binary)
    action_reports: list[dict[str, Any]] = []
    for animation_index, animation in enumerate(animations):
        samplers = _objects(
            animation.get("samplers"), owner=f"animations[{animation_index}].samplers"
        )
        if not samplers:
            raise ValueError("animation contains no samplers")
        input_arrays: list[np.ndarray] = []
        sampler_values: list[tuple[np.ndarray, str]] = []
        for sampler_index, sampler in enumerate(samplers):
            if sampler.get("interpolation", "LINEAR") not in {"LINEAR", "STEP"}:
                raise ValueError("CUBICSPLINE loop augmentation is unsupported")
            input_index = sampler.get("input")
            output_index = sampler.get("output")
            if not isinstance(input_index, int) or not isinstance(output_index, int):
                raise ValueError("animation sampler accessor indices are invalid")
            input_meta = accessors[input_index]
            output_meta = accessors[output_index]
            if (
                input_meta.get("componentType") != 5126
                or input_meta.get("type") != "SCALAR"
                or output_meta.get("componentType") != 5126
                or output_meta.get("type") not in {"VEC3", "VEC4"}
                or "sparse" in input_meta
                or "sparse" in output_meta
            ):
                raise ValueError(
                    "loop augmentation requires dense float SCALAR/VEC3/VEC4"
                )
            times = np.asarray(
                decode_accessor(source, input_index).values, dtype=np.float32
            ).reshape(-1)
            values = np.asarray(
                decode_accessor(source, output_index).values, dtype=np.float32
            )
            if (
                len(times) != len(values)
                or len(times) < 2
                or np.any(np.diff(times) <= 0)
            ):
                raise ValueError(
                    "animation sampler values are not strictly time ordered"
                )
            input_arrays.append(times)
            sampler_values.append((values, str(output_meta["type"])))
        starts = [float(values[0]) for values in input_arrays]
        ends = [float(values[-1]) for values in input_arrays]
        if max(starts) - min(starts) > 1.0e-6 or max(ends) - min(ends) > 1.0e-6:
            raise ValueError("all channels in an action must share exact clip bounds")
        appended_time = _compatible_appended_time(min(starts), max(ends))
        for sampler, times, (values, element_type) in zip(
            samplers, input_arrays, sampler_values, strict=True
        ):
            extended_times = np.concatenate(
                [times, np.asarray([appended_time], dtype=np.float32)]
            )
            extended_values = np.concatenate([values, values[0:1]], axis=0)
            sampler["input"] = _append_accessor(
                binary=binary,
                views=views,
                accessors=accessors,
                values=extended_times,
                element_type="SCALAR",
                include_bounds=True,
            )
            sampler["output"] = _append_accessor(
                binary=binary,
                views=views,
                accessors=accessors,
                values=extended_values,
                element_type=element_type,
                include_bounds=False,
            )
        action_reports.append(
            {
                "action_name": animation["name"],
                "source_start_seconds": min(starts),
                "source_end_seconds": max(ends),
                "output_end_seconds": float(appended_time),
                "return_interval_seconds": float(appended_time) - max(ends),
                "sampler_count": len(samplers),
                "all_source_keys_preserved": True,
                "appended_value_equals_first_value": True,
            }
        )
    buffers[0]["byteLength"] = len(binary)
    payload = build_glb(document, binary)
    readback = parse_glb(payload)
    if readback.binary[: len(source.binary)] != source.binary:
        raise ValueError("source binary prefix changed")
    source_other = {
        key: value
        for key, value in source.json.items()
        if key not in {"buffers", "bufferViews", "accessors", "animations"}
    }
    output_other = {
        key: value
        for key, value in readback.json.items()
        if key not in {"buffers", "bufferViews", "accessors", "animations"}
    }
    if source_other != output_other:
        raise ValueError("non-animation glTF JSON changed")
    parsed_actions = extract_actions(readback)
    for action in parsed_actions:
        for channel in action.channels:
            first = np.asarray(channel.values[0], dtype=np.float64)
            last = np.asarray(channel.values[-1], dtype=np.float64)
            if not np.array_equal(first, last):
                raise ValueError("output action endpoint does not equal its start")
    _write_exclusive(output_path, payload)
    return {
        "schema": SCHEMA,
        "status": "pass",
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "source": {
            "path": str(source_path),
            "sha256": source.sha256,
            "byte_size": source.byte_length,
        },
        "output": {
            "path": str(output_path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_size": len(payload),
        },
        "policy": {
            "return_sample_rate_hz": 15.0,
            "source_binary_prefix_preserved": True,
            "source_keys_preserved": True,
            "interpolation_allowed": ["LINEAR", "STEP"],
        },
        "actions": action_reports,
        "unchanged_non_animation_json_sha256": canonical_json_sha256(source_other),
        "notes": [
            "The appended segment returns each local joint transform to its exact first key.",
            "This changes the diagnostic clip duration and does not prove equivalence to the authored source motion.",
            "The existing source-to-projection deformation failure remains an admission blocker.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    for label, path in (("output", args.output), ("report", args.report)):
        if path.exists() or path.is_symlink():
            parser.error(f"refusing to replace {label}: {path}")
    source, output, report = (
        args.input.resolve(),
        args.output.resolve(),
        args.report.resolve(),
    )
    if len({source, output, report}) != 3:
        parser.error("input, output, and report must differ")
    output_created = False
    try:
        value = compile_loop_closure(source, output)
        output_created = True
        value["report_content_sha256"] = canonical_json_sha256(value)
        _write_exclusive(
            report,
            (
                json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
            ).encode(),
        )
    except (OSError, ValueError) as exc:
        message = str(exc)
        if output_created:
            try:
                output.unlink()
            except OSError as cleanup_exc:
                message += f"; failed to clean newly created output: {cleanup_exc}"
        parser.error(message)
    print(
        json.dumps(
            {
                "status": "pass",
                "output_sha256": sha256_file(output),
                "report_sha256": sha256_file(report),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
