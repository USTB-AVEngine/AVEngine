"""Audit native UE scene context, matte appearance, bounds and emitter attachment."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def audit_native_images(scene_rgb, isolated_rgb):
    """Reject captures made before the native room mesh becomes visible."""
    mask = isolated_rgb.max(2) > 12
    visible = int(mask.sum())
    saturated = float(np.mean(np.all(isolated_rgb[mask] >= 245, axis=1))) if visible else None
    delta = np.abs(scene_rgb.astype(np.int16) - isolated_rgb.astype(np.int16)).max(2)
    room_pixels = int(np.sum((~mask) & (delta > 3) & (scene_rgb.max(2) > 12)))
    passed = visible >= 128 and saturated is not None and saturated <= .02 and room_pixels >= 1000
    return {
        "isolated_visible_proxy_pixels": visible,
        "saturated_rgb_ge245_ratio": saturated,
        "native_room_context_pixels": room_pixels,
        "status": "pass" if passed else "fail",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", type=Path, required=True)
    args = parser.parse_args()
    task = args.task_root
    capture = json.loads((task / "evidence/ue_capture_results.json").read_text())
    rows = []
    for entry in capture["assets"]:
        images = []
        for image in entry["images"]:
            scene = np.asarray(Image.open(image["rgb"]).convert("RGB"))
            isolated = np.asarray(Image.open(image["isolated"]).convert("RGB"))
            images.append({**image, **audit_native_images(scene, isolated)})
        passed = (
            entry["status"] == "pass_capture"
            and len(images) == 2
            and all(image["status"] == "pass" for image in images)
        )
        rows.append({
            **entry,
            "images": images,
            "qa_status": "pass" if passed else "fail",
            "qa_method": "native unannotated JPEG; isolated max>12, saturated >=245 <=2%; >=1000 room pixels differing from isolated; true mesh bounds and emitter readback/follow",
            "texture_color_comparison": "not_run; Habitat production appearance metrics are separate",
        })
    passed = capture["status"] == "pass_capture" and len(rows) == 2 and all(row["qa_status"] == "pass" for row in rows)
    receipt = {"status": "pass" if passed else "fail", "assets": rows}
    (task / "evidence/ue_qa.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({
        "status": receipt["status"],
        "assets": [{
            "kind": row["kind"],
            "status": row["qa_status"],
            "height_m": row["height_m"],
            "floor_cm": row["base_floor_delta_cm"],
            "readback_cm": row["emit_readback_error_cm"],
            "follow_cm": row["translation_follow_error_cm"],
            "highlights": [image["saturated_rgb_ge245_ratio"] for image in row["images"]],
            "room_pixels": [image["native_room_context_pixels"] for image in row["images"]],
        } for row in rows],
    }))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
