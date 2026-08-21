#!/usr/bin/env python3
"""Build five hermetic modal/target-only pixel-visibility canaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.qa.pixel_visibility import (  # noqa: E402
    PixelVisibilityError,
    compile_pixel_visibility_truth,
)


HEIGHT = 12
WIDTH = 16
TARGET_ID = 17
SCALE = 12


def _context(
    pass_kind: str,
    *,
    renderer_backend: str = "hermetic_same_renderer_canary",
    rgb_renderer_backend: str | None = None,
    target_instance_id: str | None = None,
) -> dict:
    result = {
        "pass_kind": pass_kind,
        "renderer_backend": renderer_backend,
        "rgb_renderer_backend": rgb_renderer_backend or renderer_backend,
        "camera_contract_id": "pixel_visibility_canary_camera_v1",
        "semantic_id_namespace": "pixel_visibility_canary_semantics_v1",
        "resolution_hw": [HEIGHT, WIDTH],
        "frame_indices": [0],
        "camera_pose_ids": ["pixel_visibility_canary_pose_000"],
    }
    if target_instance_id is not None:
        result["target_instance_id"] = target_instance_id
    return result


def _compile(
    modal: np.ndarray,
    target_only: np.ndarray,
    *,
    normal_context: dict | None = None,
) -> dict:
    return compile_pixel_visibility_truth(
        normal_semantic_masks=[modal],
        target_only_semantic_masks_by_instance={"source1": [target_only]},
        semantic_ids_by_instance={"source1": TARGET_ID},
        normal_context=normal_context or _context("modal_scene"),
        target_only_contexts_by_instance={
            "source1": _context("target_only", target_instance_id="source1")
        },
    )


def _mask_image(mask: np.ndarray, *, color: tuple[int, int, int]) -> Image.Image:
    pixels = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    pixels[mask == TARGET_ID] = color
    image = Image.fromarray(pixels, mode="RGB")
    return image.resize((WIDTH * SCALE, HEIGHT * SCALE), Image.Resampling.NEAREST)


def _overlay_image(modal: np.ndarray, target_only: np.ndarray) -> Image.Image:
    pixels = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    target = target_only == TARGET_ID
    visible = modal == TARGET_ID
    pixels[target & ~visible] = (232, 94, 94)
    pixels[visible] = (74, 222, 128)
    image = Image.fromarray(pixels, mode="RGB")
    return image.resize((WIDTH * SCALE, HEIGHT * SCALE), Image.Resampling.NEAREST)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    target = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    target[3:9, 4:12] = TARGET_ID
    absent = np.zeros_like(target)
    partial = np.zeros_like(target)
    partial[3:9, 4:8] = TARGET_ID

    cases = [
        ("clear", "清晰可见", target.copy(), target.copy(), "visible_clear"),
        (
            "partial_occlusion",
            "部分遮挡",
            partial,
            target.copy(),
            "visible_occluded",
        ),
        (
            "fully_occluded",
            "完全遮挡",
            absent.copy(),
            target.copy(),
            "fully_occluded",
        ),
        (
            "out_of_view",
            "画外",
            absent.copy(),
            absent.copy(),
            "out_of_view",
        ),
    ]
    records: list[dict] = []
    rendered_rows: list[tuple[str, np.ndarray, np.ndarray, str]] = []
    for canary_id, label, modal, target_only, expected_state in cases:
        truth = _compile(modal, target_only)
        frame = truth["per_instance"]["source1"]["frames"][0]
        if frame["state"] != expected_state:
            raise RuntimeError(
                f"{canary_id}: expected {expected_state}, got {frame['state']}"
            )
        canary_dir = output / canary_id
        canary_dir.mkdir(parents=True, exist_ok=True)
        _write_json(canary_dir / "truth.json", truth)
        records.append(
            {
                "canary_id": canary_id,
                "label_zh": label,
                "expected": expected_state,
                "status": "pass",
                "truth_path": f"{canary_id}/truth.json",
                "visible_pixels": frame["visible_pixels"],
                "target_pixels": frame["target_pixels"],
                "visible_fraction": frame["visible_fraction"],
                "occlusion_fraction": frame["occlusion_fraction"],
            }
        )
        rendered_rows.append((canary_id, modal, target_only, expected_state))

    rejection_message = ""
    try:
        _compile(
            target.copy(),
            target.copy(),
            normal_context=_context(
                "modal_scene",
                renderer_backend="habitat_sim",
                rgb_renderer_backend="spear_ue",
            ),
        )
    except PixelVisibilityError as error:
        rejection_message = str(error)
    if "Habitat labels" not in rejection_message:
        raise RuntimeError("cross-backend canary was not rejected")
    records.append(
        {
            "canary_id": "reject_habitat_labels_for_ue_rgb",
            "label_zh": "拒绝跨后端伪真值",
            "expected": "PixelVisibilityError",
            "status": "pass",
            "rejection": rejection_message,
        }
    )

    cell_width = WIDTH * SCALE
    cell_height = HEIGHT * SCALE
    label_width = 180
    header_height = 34
    sheet = Image.new(
        "RGB",
        (label_width + 3 * cell_width, header_height + 5 * cell_height),
        (18, 22, 30),
    )
    draw = ImageDraw.Draw(sheet)
    for column, title in enumerate(("Modal", "Target-only", "Visible / Occluded")):
        draw.text(
            (label_width + column * cell_width + 8, 10),
            title,
            fill=(225, 232, 240),
        )
    for row, (label, modal, target_only, state) in enumerate(rendered_rows):
        top = header_height + row * cell_height
        draw.text((10, top + 12), label, fill=(225, 232, 240))
        draw.text((10, top + 32), state, fill=(145, 164, 184))
        sheet.paste(
            _mask_image(modal, color=(74, 222, 128)),
            (label_width, top),
        )
        sheet.paste(
            _mask_image(target_only, color=(95, 155, 255)),
            (label_width + cell_width, top),
        )
        sheet.paste(
            _overlay_image(modal, target_only),
            (label_width + 2 * cell_width, top),
        )
    rejection_top = header_height + 4 * cell_height
    draw.text((10, rejection_top + 12), "cross-backend", fill=(225, 232, 240))
    draw.text((10, rejection_top + 32), "expected rejection", fill=(232, 94, 94))
    draw.rectangle(
        (
            label_width + 12,
            rejection_top + 18,
            label_width + 3 * cell_width - 12,
            rejection_top + cell_height - 18,
        ),
        outline=(232, 94, 94),
        width=3,
    )
    draw.text(
        (label_width + 24, rejection_top + 42),
        "Habitat semantic labels != UE RGB",
        fill=(232, 94, 94),
    )
    sheet.save(output / "contact_sheet.png")

    manifest = {
        "schema": "avengine_lead_a_pixel_visibility_canaries_v1",
        "status": "pass",
        "claim_boundary": (
            "Hermetic semantic-mask contract canaries; no native renderer or "
            "dataset admission claim"
        ),
        "canary_count": len(records),
        "canaries": records,
        "contact_sheet": "contact_sheet.png",
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = build(args.output.resolve())
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
