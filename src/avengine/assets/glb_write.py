"""Deterministic GLB 2.0 serialization for bounded M2 compilers.

The strict parser in :mod:`avengine.assets.glb` owns input validation.  This
module only serializes an already validated/mutated glTF JSON object and its
single embedded BIN payload.  Callers are responsible for updating
``buffers[0].byteLength`` before serialization and must parse the result again
before treating it as compiler output.
"""

from __future__ import annotations

import json
import struct
from typing import Any


_JSON_CHUNK_TYPE = 0x4E4F534A
_BIN_CHUNK_TYPE = 0x004E4942


def build_glb(document: dict[str, Any], binary: bytes | bytearray) -> bytes:
    """Serialize one JSON chunk and one embedded BIN chunk as GLB 2.0."""

    json_bytes = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    binary_bytes = bytes(binary)
    binary_bytes += b"\0" * ((-len(binary_bytes)) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(binary_bytes)
    return b"".join(
        [
            struct.pack("<4sII", b"glTF", 2, total),
            struct.pack("<II", len(json_bytes), _JSON_CHUNK_TYPE),
            json_bytes,
            struct.pack("<II", len(binary_bytes), _BIN_CHUNK_TYPE),
            binary_bytes,
        ]
    )
