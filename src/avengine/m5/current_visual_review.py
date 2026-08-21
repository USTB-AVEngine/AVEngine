"""Offline, research-only reviewer for current M5 visual captures.

The generated page is a local viewing aid for one already completed current
visual research output. It never changes the capture receipt, produces no
verdict, and has no role in admission or formal evidence readers.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
import math
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
from PIL import Image


TIMELINE_TICKS_PER_SECOND = 48_000
_REQUIRED_ARTIFACTS = ("rgb", "depth", "semantic", "frame_records")


class CurrentVisualReviewError(ValueError):
    """A current visual research output cannot be rendered for local review."""


@dataclass(frozen=True)
class CurrentVisualReview:
    """Location and frame count of one local, non-authoritative review page."""

    html_path: Path
    review_directory: Path
    frame_count: int


def _load_json_object(path: Path, *, owner: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise CurrentVisualReviewError(
            f"{owner} is missing or is not a regular file: {path}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CurrentVisualReviewError(f"could not read {owner}: {exc}") from exc
    if not isinstance(value, dict):
        raise CurrentVisualReviewError(f"{owner} must contain a JSON object")
    return value


def _require_completed_receipt(receipt: Mapping[str, Any]) -> int:
    status = receipt.get("status")
    if status != "research_only":
        raise CurrentVisualReviewError(
            f"review requires a completed research_only receipt, got {status!r}"
        )
    if (
        receipt.get("research_only") is not True
        or receipt.get("episode_counted") is not False
    ):
        raise CurrentVisualReviewError(
            "review requires an explicitly research-only, non-counted receipt"
        )
    capture = receipt.get("capture")
    if (
        not isinstance(capture, Mapping)
        or capture.get("native_habitat_started") is not True
    ):
        raise CurrentVisualReviewError(
            "review requires a completed native Habitat current visual capture"
        )
    frame_count = capture.get("frame_count")
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count < 1
    ):
        raise CurrentVisualReviewError(
            "receipt capture.frame_count must be a positive integer"
        )
    return frame_count


def _artifact_path(
    output_root: Path,
    artifacts: Mapping[str, Any],
    *,
    name: str,
    required: bool = False,
) -> Path | None:
    raw = artifacts.get(name)
    if raw is None and not required:
        return None
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise CurrentVisualReviewError(
            f"receipt artifact {name!r} must be a POSIX relative path"
        )
    relative = PurePosixPath(raw)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise CurrentVisualReviewError(
            f"receipt artifact {name!r} is not a normalized relative path"
        )
    try:
        value = (output_root / relative).resolve(strict=True)
        value.relative_to(output_root)
    except (OSError, ValueError) as exc:
        raise CurrentVisualReviewError(
            f"receipt artifact {name!r} is missing or escapes the research output"
        ) from exc
    if not value.is_file() or value.is_symlink():
        raise CurrentVisualReviewError(
            f"receipt artifact {name!r} is missing or is not a regular file: {value}"
        )
    return value


def _load_npy(path: Path, *, owner: str) -> np.ndarray:
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise CurrentVisualReviewError(f"could not load {owner}: {exc}") from exc
    if not isinstance(value, np.ndarray):
        raise CurrentVisualReviewError(f"{owner} is not a NumPy array")
    return value


def _require_rgb(value: np.ndarray, *, frame_count: int) -> np.ndarray:
    if (
        value.ndim != 4
        or value.shape[0] != frame_count
        or value.shape[-1] != 3
        or value.dtype != np.uint8
    ):
        raise CurrentVisualReviewError(
            "RGB array must be uint8 [frame,height,width,3] and match receipt frame_count"
        )
    if value.shape[1] < 1 or value.shape[2] < 1:
        raise CurrentVisualReviewError(
            "RGB array must have nonempty spatial dimensions"
        )
    return value


def _require_depth(value: np.ndarray, *, rgb: np.ndarray) -> np.ndarray:
    if (
        value.ndim != 3
        or value.shape != rgb.shape[:3]
        or not np.issubdtype(value.dtype, np.number)
        or np.issubdtype(value.dtype, np.complexfloating)
    ):
        raise CurrentVisualReviewError(
            "depth array must be real numeric [frame,height,width] and co-registered with RGB"
        )
    return value


def _require_semantic(value: np.ndarray, *, rgb: np.ndarray) -> np.ndarray:
    if (
        value.ndim != 3
        or value.shape != rgb.shape[:3]
        or not np.issubdtype(value.dtype, np.integer)
    ):
        raise CurrentVisualReviewError(
            "semantic array must be integer [frame,height,width] and co-registered with RGB"
        )
    return value


def _optional_frame_array(
    path: Path | None,
    *,
    owner: str,
    frame_count: int,
    suffix_shape: tuple[int, ...],
    dtype_kind: str | None = None,
) -> np.ndarray | None:
    if path is None:
        return None
    value = _load_npy(path, owner=owner)
    if value.shape != (frame_count, *suffix_shape):
        raise CurrentVisualReviewError(
            f"{owner} must have shape {(frame_count, *suffix_shape)}, got {value.shape}"
        )
    if dtype_kind == "integer" and not np.issubdtype(value.dtype, np.integer):
        raise CurrentVisualReviewError(f"{owner} must contain integer values")
    if dtype_kind == "finite" and (
        not np.issubdtype(value.dtype, np.number)
        or np.issubdtype(value.dtype, np.complexfloating)
        or not np.all(np.isfinite(value))
    ):
        raise CurrentVisualReviewError(
            f"{owner} must contain finite real numeric values"
        )
    return value


def _frame_records(path: Path, *, frame_count: int) -> list[dict[str, Any]]:
    document = _load_json_object(path, owner="frame_records artifact")
    frames = document.get("frames")
    if not isinstance(frames, list) or len(frames) != frame_count:
        raise CurrentVisualReviewError(
            "frame_records artifact must contain one frame object for every captured frame"
        )
    values: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise CurrentVisualReviewError(
                f"frame_records frame {index} must be an object"
            )
        frame_index = frame.get("frame_index")
        ticks = frame.get("pts_ticks")
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or isinstance(ticks, bool)
            or not isinstance(ticks, int)
        ):
            raise CurrentVisualReviewError(
                f"frame_records frame {index} must provide integer frame_index and pts_ticks"
            )
        if frame_index != index:
            raise CurrentVisualReviewError(
                f"frame_records frame {index} frame_index must equal its array index"
            )
        values.append(frame)
    return values


def _depth_range(depth: np.ndarray) -> tuple[float, float]:
    minimum = math.inf
    maximum = -math.inf
    for frame_index in range(depth.shape[0]):
        values = np.asarray(depth[frame_index], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size:
            minimum = min(minimum, float(finite.min()))
            maximum = max(maximum, float(finite.max()))
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise CurrentVisualReviewError(
            "depth array contains no finite values for review"
        )
    return minimum, maximum


def _depth_preview(frame: np.ndarray, *, minimum: float, maximum: float) -> np.ndarray:
    values = np.asarray(frame, dtype=np.float64)
    preview = np.zeros(values.shape, dtype=np.uint8)
    finite = np.isfinite(values)
    if maximum > minimum:
        normalized = (values[finite] - minimum) / (maximum - minimum)
        preview[finite] = np.clip(np.rint(normalized * 255.0), 0, 255).astype(np.uint8)
    elif finite.any():
        preview[finite] = 255
    return np.repeat(preview[..., None], 3, axis=2)


def _semantic_preview(frame: np.ndarray) -> np.ndarray:
    values = np.asarray(frame, dtype=np.int64)
    preview = np.empty((*values.shape, 3), dtype=np.uint8)
    preview[..., 0] = np.mod(values * 37 + 17, 256).astype(np.uint8)
    preview[..., 1] = np.mod(values * 67 + 43, 256).astype(np.uint8)
    preview[..., 2] = np.mod(values * 97 + 89, 256).astype(np.uint8)
    preview[values == 0] = (0, 0, 0)
    preview[values == 210] = (255, 216, 0)
    preview[values == 211] = (255, 79, 160)
    return preview


def _save_png(path: Path, frame: np.ndarray, *, owner: str) -> None:
    try:
        Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB").save(path)
    except (OSError, ValueError) as exc:
        raise CurrentVisualReviewError(f"could not write {owner}: {exc}") from exc


def _frame_metadata(
    records: list[dict[str, Any]],
    *,
    actor_matrices: np.ndarray | None,
    source_positions: np.ndarray | None,
    visibility: np.ndarray | None,
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        frame_index = int(record["frame_index"])
        ticks = int(record["pts_ticks"])
        entry: dict[str, Any] = {
            "frame_index": frame_index,
            "pts_ticks": ticks,
            "time_seconds": ticks / TIMELINE_TICKS_PER_SECOND,
        }
        for field in ("action_id", "action_sample_index"):
            if field in record:
                entry[field] = record[field]
        planned = record.get("actor_world_positions_m")
        if planned is not None:
            entry["planned_actor_world_positions_m"] = planned
        elif actor_matrices is not None:
            entry["planned_actor_world_positions_m"] = actor_matrices[
                index, :, :3, 3
            ].tolist()
        observed = record.get("source_positions_m")
        if observed is not None:
            entry["runtime_source_readback_positions_m"] = observed
        elif source_positions is not None:
            entry["runtime_source_readback_positions_m"] = source_positions[
                index
            ].tolist()
        pixels = record.get("semantic_visibility_pixels")
        if pixels is not None:
            entry["semantic_visibility_pixels"] = pixels
        elif visibility is not None:
            entry["semantic_visibility_pixels"] = visibility[index].tolist()
        values.append(entry)
    return values


def _json_for_script(value: Any) -> str:
    """Encode JSON so no data string can terminate the enclosing script tag."""

    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return (
        encoded.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _review_html(
    *, title: str, images: Mapping[str, list[str]], frames: list[dict[str, Any]]
) -> str:
    payload = _json_for_script({"images": images, "frames": frames})
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
:root {{ color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }}
body {{ margin: 0; background: #121518; color: #edf2f7; }}
main {{ max-width: 1500px; margin: auto; padding: 22px; }}
h1 {{ margin: 0 0 6px; font-size: 1.5rem; }}
.warning {{ border-left: 4px solid #e5a323; background: #2d2517; padding: 12px; margin: 16px 0; }}
.controls {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin: 16px 0; }}
#frame-slider {{ min-width: 360px; flex: 1; }}
button {{ cursor: pointer; background: #273544; color: #edf2f7; border: 1px solid #52677c; border-radius: 5px; padding: 7px 11px; }}
.panels {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
figure {{ margin: 0; background: #1b2229; border: 1px solid #33414e; border-radius: 6px; overflow: hidden; }}
figcaption {{ padding: 8px 10px; font-weight: 600; }}
img {{ width: 100%; height: auto; display: block; image-rendering: auto; background: #050607; }}
#frame-details {{ background: #0c1014; border: 1px solid #33414e; border-radius: 6px; padding: 12px; overflow: auto; white-space: pre-wrap; }}
textarea {{ width: 100%; box-sizing: border-box; min-height: 150px; resize: vertical; background: #0c1014; color: #edf2f7; border: 1px solid #52677c; border-radius: 6px; padding: 10px; }}
.small {{ color: #aebdcc; font-size: .9rem; }}
@media (max-width: 900px) {{ .panels {{ grid-template-columns: 1fr; }} #frame-slider {{ min-width: 180px; }} }}
</style>
</head>
<body>
<main>
  <h1>{escape(title)}</h1>
  <p class="small">Offline synchronized RGB / depth / semantic inspection.</p>
  <section class="warning"><strong>Research-only visual review.</strong> This page does not alter the capture receipt, create a verdict, count an episode, or establish formal/admission evidence.</section>
  <section class="controls" aria-label="Synchronized frame selector">
    <button id="previous" type="button">Previous frame</button>
    <input id="frame-slider" type="range" min="0" step="1" aria-label="Frame index">
    <button id="next" type="button">Next frame</button>
    <strong id="frame-label"></strong>
  </section>
  <section class="panels">
    <figure><figcaption>RGB</figcaption><img id="rgb-panel" alt="RGB frame"></figure>
    <figure><figcaption>Depth (global grayscale preview)</figcaption><img id="depth-panel" alt="Depth frame"></figure>
    <figure><figcaption>Semantic (false color; actor 210 yellow, 211 pink)</figcaption><img id="semantic-panel" alt="Semantic frame"></figure>
  </section>
  <h2>Frame state</h2>
  <pre id="frame-details"></pre>
  <h2>Reviewer notes</h2>
  <p class="small">Notes stay in this browser's localStorage unless you explicitly download them. They are not written into the research output, receipt, schema, or evidence reader.</p>
  <textarea id="reviewer-notes" placeholder="Free-text observations for this local review"></textarea>
  <section class="controls"><button id="clear-notes" type="button">Clear local notes</button><button id="download-notes" type="button">Download local notes</button></section>
</main>
<script>
const review = {payload};
const slider = document.getElementById("frame-slider");
const rgb = document.getElementById("rgb-panel");
const depth = document.getElementById("depth-panel");
const semantic = document.getElementById("semantic-panel");
const label = document.getElementById("frame-label");
const details = document.getElementById("frame-details");
const notes = document.getElementById("reviewer-notes");
const notesKey = "avengine-m5-current-visual-review:" + location.pathname;
slider.max = String(review.frames.length - 1);
const clamp = value => Math.max(0, Math.min(review.frames.length - 1, Number(value) || 0));
function render(value) {{
  const index = clamp(value);
  slider.value = String(index);
  const frame = review.frames[index];
  rgb.src = review.images.rgb[index];
  depth.src = review.images.depth[index];
  semantic.src = review.images.semantic[index];
  label.textContent = `frame ${{frame.frame_index}} · ${{frame.time_seconds.toFixed(6)}} s · ${{frame.pts_ticks}} ticks`;
  details.textContent = JSON.stringify(frame, null, 2);
}}
slider.addEventListener("input", event => render(event.target.value));
document.getElementById("previous").addEventListener("click", () => render(Number(slider.value) - 1));
document.getElementById("next").addEventListener("click", () => render(Number(slider.value) + 1));
window.addEventListener("keydown", event => {{
  if (event.target && event.target.tagName === "TEXTAREA") return;
  if (event.key === "ArrowLeft") {{ event.preventDefault(); render(Number(slider.value) - 1); }}
  if (event.key === "ArrowRight") {{ event.preventDefault(); render(Number(slider.value) + 1); }}
}});
function readLocalNotes() {{
  try {{ return localStorage.getItem(notesKey) || ""; }} catch (_error) {{ return ""; }}
}}
function writeLocalNotes(value) {{
  try {{ localStorage.setItem(notesKey, value); }} catch (_error) {{ return; }}
}}
function clearLocalNotes() {{
  try {{ localStorage.removeItem(notesKey); }} catch (_error) {{ return; }}
}}
notes.value = readLocalNotes();
notes.addEventListener("input", () => writeLocalNotes(notes.value));
document.getElementById("clear-notes").addEventListener("click", () => {{
  notes.value = ""; clearLocalNotes(); notes.focus();
}});
document.getElementById("download-notes").addEventListener("click", () => {{
  const payload = {{review_kind: "m5_current_visual_local_notes", frame_index: Number(slider.value), notes: notes.value}};
  const blob = new Blob([JSON.stringify(payload, null, 2) + "\\n"], {{type: "application/json"}});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob); link.download = "current_visual_local_notes.json"; link.click();
  URL.revokeObjectURL(link.href);
}});
render(0);
</script>
</body>
</html>
"""


def _review_directory(output_root: Path, requested: str | Path | None) -> Path:
    raw = output_root / "review" if requested is None else Path(requested)
    if not raw.is_absolute():
        raw = output_root / raw
    destination = raw.resolve(strict=False)
    arrays_root = (output_root / "arrays").resolve(strict=False)
    try:
        destination.relative_to(arrays_root)
    except ValueError:
        pass
    else:
        raise CurrentVisualReviewError(
            "review output must not be written inside arrays/"
        )
    if destination == output_root or os.path.lexists(destination):
        raise CurrentVisualReviewError(
            f"refusing to replace local review output: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    return destination


def generate_current_visual_review(
    research_output: str | Path,
    *,
    review_directory: str | Path | None = None,
) -> CurrentVisualReview:
    """Render one completed research-only M5 output into a local static review page."""

    try:
        output_root = Path(research_output).resolve(strict=True)
    except OSError as exc:
        raise CurrentVisualReviewError(
            f"current visual research output is unavailable: {research_output}"
        ) from exc
    if not output_root.is_dir():
        raise CurrentVisualReviewError(
            "current visual research output must be a directory"
        )
    receipt = _load_json_object(
        output_root / "research_receipt.json", owner="research receipt"
    )
    frame_count = _require_completed_receipt(receipt)
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise CurrentVisualReviewError(
            "completed research receipt has no artifacts object"
        )
    missing = [name for name in _REQUIRED_ARTIFACTS if name not in artifacts]
    if missing:
        raise CurrentVisualReviewError(
            "completed research receipt is missing required review artifacts: "
            + ", ".join(missing)
        )
    rgb_path = _artifact_path(output_root, artifacts, name="rgb", required=True)
    depth_path = _artifact_path(output_root, artifacts, name="depth", required=True)
    semantic_path = _artifact_path(
        output_root, artifacts, name="semantic", required=True
    )
    records_path = _artifact_path(
        output_root, artifacts, name="frame_records", required=True
    )
    assert rgb_path is not None and depth_path is not None and semantic_path is not None
    assert records_path is not None
    rgb = _require_rgb(
        _load_npy(rgb_path, owner="RGB artifact"), frame_count=frame_count
    )
    depth = _require_depth(_load_npy(depth_path, owner="depth artifact"), rgb=rgb)
    semantic = _require_semantic(
        _load_npy(semantic_path, owner="semantic artifact"), rgb=rgb
    )
    records = _frame_records(records_path, frame_count=frame_count)
    actor_matrices = _optional_frame_array(
        _artifact_path(output_root, artifacts, name="actor_world_matrices"),
        owner="actor_world_matrices artifact",
        frame_count=frame_count,
        suffix_shape=(2, 4, 4),
        dtype_kind="finite",
    )
    source_positions = _optional_frame_array(
        _artifact_path(output_root, artifacts, name="source_positions_m"),
        owner="source_positions_m artifact",
        frame_count=frame_count,
        suffix_shape=(2, 3),
        dtype_kind="finite",
    )
    visibility = _optional_frame_array(
        _artifact_path(output_root, artifacts, name="semantic_visibility_pixels"),
        owner="semantic_visibility_pixels artifact",
        frame_count=frame_count,
        suffix_shape=(2,),
        dtype_kind="integer",
    )
    depth_minimum, depth_maximum = _depth_range(depth)
    metadata = _frame_metadata(
        records,
        actor_matrices=actor_matrices,
        source_positions=source_positions,
        visibility=visibility,
    )
    review_root = _review_directory(output_root, review_directory)
    frames_root = review_root / "frames"
    images = {"rgb": [], "depth": [], "semantic": []}
    for modality in images:
        (frames_root / modality).mkdir(parents=True)
    for index in range(frame_count):
        filename = f"frame_{index:04d}.png"
        rgb_relative = Path("frames") / "rgb" / filename
        depth_relative = Path("frames") / "depth" / filename
        semantic_relative = Path("frames") / "semantic" / filename
        _save_png(review_root / rgb_relative, rgb[index], owner="RGB preview")
        _save_png(
            review_root / depth_relative,
            _depth_preview(depth[index], minimum=depth_minimum, maximum=depth_maximum),
            owner="depth preview",
        )
        _save_png(
            review_root / semantic_relative,
            _semantic_preview(semantic[index]),
            owner="semantic preview",
        )
        images["rgb"].append(rgb_relative.as_posix())
        images["depth"].append(depth_relative.as_posix())
        images["semantic"].append(semantic_relative.as_posix())
    html_path = review_root / "index.html"
    html_path.write_text(
        _review_html(
            title=f"M5 current visual research review · {output_root.name}",
            images=images,
            frames=metadata,
        ),
        encoding="utf-8",
        newline="\n",
    )
    return CurrentVisualReview(
        html_path=html_path,
        review_directory=review_root,
        frame_count=frame_count,
    )


__all__ = [
    "CurrentVisualReview",
    "CurrentVisualReviewError",
    "generate_current_visual_review",
]
