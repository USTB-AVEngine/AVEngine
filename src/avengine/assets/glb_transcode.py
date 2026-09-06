"""Lossless embedded-WebP to PNG conversion for Habitat-readable GLBs."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from avengine.assets.glb import load_glb, parse_glb
from avengine.assets.glb_write import build_glb


class GlbTextureTranscodeError(ValueError):
    """An embedded texture graph cannot be converted without changing geometry."""


def _record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    payload = resolved.read_bytes()
    return {
        "path": str(resolved),
        "byte_size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _append_aligned(binary: bytearray, payload: bytes) -> int:
    binary.extend(b"\x00" * ((-len(binary)) % 4))
    offset = len(binary)
    binary.extend(payload)
    return offset


def transcode_embedded_webp(
    source_glb: str | Path,
    output_glb: str | Path,
    manifest_path: str | Path,
) -> Path:
    """Write a PNG-backed GLB while preserving the structural glTF graph."""

    source = Path(source_glb).resolve()
    output = Path(output_glb).resolve()
    manifest = Path(manifest_path).resolve()
    if not source.is_file() or source.is_symlink():
        raise GlbTextureTranscodeError(f"source GLB is not a regular file: {source}")
    if output.exists() or output.is_symlink() or manifest.exists() or manifest.is_symlink():
        raise GlbTextureTranscodeError("transcode refuses existing output paths")
    parsed = load_glb(source)
    before = parsed.json
    document = deepcopy(before)
    binary = bytearray(parsed.binary)
    images = document.get("images")
    textures = document.get("textures")
    views = document.get("bufferViews")
    if not isinstance(images, list) or not isinstance(textures, list) or not isinstance(views, list):
        raise GlbTextureTranscodeError("GLB lacks an embedded texture graph")
    converted: list[dict[str, Any]] = []
    converted_indices: set[int] = set()
    for index, image in enumerate(images):
        if not isinstance(image, Mapping) or image.get("mimeType") != "image/webp":
            continue
        view_index = image.get("bufferView")
        if isinstance(view_index, bool) or not isinstance(view_index, int) or not 0 <= view_index < len(views):
            raise GlbTextureTranscodeError(f"image {index} has an invalid bufferView")
        view = views[view_index]
        if not isinstance(view, Mapping) or view.get("buffer", 0) != 0:
            raise GlbTextureTranscodeError("WebP image must use embedded buffer 0")
        start = int(view.get("byteOffset", 0))
        length = int(view.get("byteLength", 0))
        source_bytes = bytes(parsed.binary[start : start + length])
        try:
            with Image.open(io.BytesIO(source_bytes)) as opened:
                opened.load()
                rgba = opened.convert("RGBA")
                pixels = rgba.tobytes()
                encoded = io.BytesIO()
                rgba.save(encoded, format="PNG", optimize=False, compress_level=6)
                png = encoded.getvalue()
                pixel_size = list(rgba.size)
        except (OSError, ValueError) as exc:
            raise GlbTextureTranscodeError(f"cannot decode WebP image {index}") from exc
        offset = _append_aligned(binary, png)
        image = dict(image)
        image["bufferView"] = len(views)
        image["mimeType"] = "image/png"
        images[index] = image
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(png)})
        converted_indices.add(index)
        converted.append(
            {
                "image_index": index,
                "pixel_size": pixel_size,
                "rgba_sha256": hashlib.sha256(pixels).hexdigest(),
                "source_webp_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "source_size_bytes": len(source_bytes),
                "png_sha256": hashlib.sha256(png).hexdigest(),
                "png_size_bytes": len(png),
            }
        )
    if not converted_indices:
        raise GlbTextureTranscodeError("GLB contains no embedded WebP images")
    for texture_index, texture in enumerate(textures):
        if not isinstance(texture, Mapping):
            raise GlbTextureTranscodeError("texture entries must be objects")
        extensions = texture.get("extensions")
        webp = extensions.get("EXT_texture_webp") if isinstance(extensions, Mapping) else None
        if isinstance(webp, Mapping):
            image_index = webp.get("source")
            if image_index not in converted_indices:
                raise GlbTextureTranscodeError("texture WebP source was not converted")
            texture = dict(texture)
            texture["source"] = image_index
            extensions = dict(extensions)
            del extensions["EXT_texture_webp"]
            if extensions:
                texture["extensions"] = extensions
            else:
                texture.pop("extensions", None)
            textures[texture_index] = texture
    for key in ("extensionsUsed", "extensionsRequired"):
        values = [value for value in document.get(key, []) if value != "EXT_texture_webp"]
        if values:
            document[key] = values
        else:
            document.pop(key, None)
    document["buffers"][0]["byteLength"] = len(binary)
    payload = build_glb(document, binary)
    readback = parse_glb(payload)
    for key in ("meshes", "skins", "nodes", "accessors", "animations", "scenes", "scene"):
        if readback.json.get(key) != before.get(key):
            raise GlbTextureTranscodeError(f"transcode changed structural graph key {key}")
    if "EXT_texture_webp" in readback.json.get("extensionsRequired", []):
        raise GlbTextureTranscodeError("WebP remains required after transcode")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise GlbTextureTranscodeError(f"unable to publish transcoded GLB: {output}") from exc
    value = {
        "schema": "glb_embedded_webp_to_png_transcode_v1",
        "status": "pass",
        "geometry_skin_animation_graph_changed": False,
        "source": _record(source),
        "output": _record(output),
        "images": converted,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return output


__all__ = ["GlbTextureTranscodeError", "transcode_embedded_webp"]
