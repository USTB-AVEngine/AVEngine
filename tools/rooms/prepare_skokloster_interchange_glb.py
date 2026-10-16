#!/usr/bin/env python3
"""Bake Skokloster's legacy source axes into a canonical glTF for UE import."""

from __future__ import annotations

import argparse
import json
import os
import struct
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

GLB_MAGIC = 0x46546C67
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942
FLOAT_COMPONENT = 5126


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read_glb(path: Path) -> tuple[dict[str, object], bytearray]:
    payload = path.read_bytes()
    _require(len(payload) >= 20, "GLB is truncated")
    magic, version, declared_length = struct.unpack_from("<III", payload, 0)
    _require(magic == GLB_MAGIC and version == 2, "expected a GLB 2.0 container")
    _require(declared_length == len(payload), "GLB declared length differs")
    offset = 12
    chunks: list[tuple[int, bytes]] = []
    while offset < len(payload):
        _require(offset + 8 <= len(payload), "GLB chunk header is truncated")
        length, kind = struct.unpack_from("<II", payload, offset)
        offset += 8
        _require(offset + length <= len(payload), "GLB chunk payload is truncated")
        chunks.append((kind, payload[offset : offset + length]))
        offset += length
    _require(offset == len(payload), "GLB trailing-byte drift")
    _require(
        [kind for kind, _ in chunks] == [JSON_CHUNK, BIN_CHUNK],
        "expected one JSON and one BIN chunk",
    )
    document = json.loads(chunks[0][1].rstrip(b" \t\r\n\x00"))
    _require(isinstance(document, dict), "GLB JSON root must be an object")
    return document, bytearray(chunks[1][1])


def _json_chunk(document: dict[str, object]) -> bytes:
    payload = json.dumps(
        document, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return payload + b" " * ((-len(payload)) % 4)


def _write_glb(path: Path, document: dict[str, object], binary: bytearray) -> None:
    json_payload = _json_chunk(document)
    binary_payload = bytes(binary) + b"\x00" * ((-len(binary)) % 4)
    length = 12 + 8 + len(json_payload) + 8 + len(binary_payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(struct.pack("<III", GLB_MAGIC, 2, length))
            stream.write(struct.pack("<II", len(json_payload), JSON_CHUNK))
            stream.write(json_payload)
            stream.write(struct.pack("<II", len(binary_payload), BIN_CHUNK))
            stream.write(binary_payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def prepare(source: Path, output: Path) -> dict[str, object]:
    source = source.resolve()
    output = output.resolve()
    _require(
        source.is_file() and not source.is_symlink(), f"source GLB is missing: {source}"
    )
    _require(
        not output.exists() and not output.is_symlink(),
        f"refusing to replace output: {output}",
    )
    document, binary = _read_glb(source)
    meshes = document.get("meshes")
    nodes = document.get("nodes")
    _require(isinstance(meshes, list) and len(meshes) == 1, "expected exactly one mesh")
    _require(
        isinstance(nodes, list) and len(nodes) == 2,
        "expected exact two-node source graph",
    )
    primitives = meshes[0].get("primitives")
    _require(
        isinstance(primitives, list) and len(primitives) == 1, "expected one primitive"
    )
    position_accessor_index = int(primitives[0]["attributes"]["POSITION"])
    accessors = document["accessors"]
    buffer_views = document["bufferViews"]
    accessor = accessors[position_accessor_index]
    _require(
        accessor.get("componentType") == FLOAT_COMPONENT, "POSITION must be float32"
    )
    _require(
        accessor.get("type") == "VEC3" and "sparse" not in accessor,
        "POSITION layout drift",
    )
    view = buffer_views[int(accessor["bufferView"])]
    _require(int(view.get("buffer", 0)) == 0, "POSITION must use buffer 0")
    _require("byteStride" not in view, "interleaved POSITION is not supported")
    count = int(accessor["count"])
    _require(count == 800936, "Skokloster POSITION count drift")
    byte_offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    byte_length = count * 3 * 4
    _require(byte_offset + byte_length <= len(binary), "POSITION exceeds BIN chunk")
    positions = np.ndarray(
        shape=(count, 3), dtype="<f4", buffer=binary, offset=byte_offset
    )
    source_positions = positions.copy()
    positions[:, 0] = source_positions[:, 0]
    positions[:, 1] = source_positions[:, 2]
    positions[:, 2] = -source_positions[:, 1]
    canonical_min = positions.min(axis=0).astype(float)
    canonical_max = positions.max(axis=0).astype(float)
    accessor["min"] = canonical_min.tolist()
    accessor["max"] = canonical_max.tolist()
    extras = document.setdefault("extras", {})
    _require(isinstance(extras, dict), "root extras must be an object")
    extras["avengine_coordinate_preparation"] = {
        "source": "legacy Habitat test-scene POSITION, Z up and +Y front",
        "source_to_habitat": "H=(S.x,S.z,-S.y)",
        "prepared": "canonical glTF metres, +Y up and -Z forward",
        "winding_preserved": True,
    }
    _write_glb(output, document, binary)
    reloaded, _ = _read_glb(output)
    reloaded_accessor = reloaded["accessors"][position_accessor_index]
    _require(reloaded_accessor["min"] == accessor["min"], "prepared min bound drift")
    _require(reloaded_accessor["max"] == accessor["max"], "prepared max bound drift")
    return {
        "schema": "avengine_skokloster_interchange_glb_preparation_v1",
        "status": "pass",
        "source_glb": str(source),
        "prepared_glb": str(output),
        "coordinate_contract": {
            "source_to_habitat": "H=(S.x,S.z,-S.y)",
            "matrix_row_major": [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            "winding_preserved": True,
        },
        "position_count": count,
        "source_bounds_m": {
            "min": source_positions.min(axis=0).astype(float).tolist(),
            "max": source_positions.max(axis=0).astype(float).tolist(),
        },
        "habitat_bounds_m": {
            "min": canonical_min.tolist(),
            "max": canonical_max.tolist(),
        },
        "material_and_texture_payload_preserved": True,
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _require(not args.report.exists(), f"refusing to replace report: {args.report}")
    result = prepare(args.source, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "SKOKLOSTER_INTERCHANGE_GLB_OK "
        f"positions={result['position_count']} output={args.output.resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
