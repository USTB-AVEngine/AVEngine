#!/usr/bin/env python3
"""Make the I23D review camera work for centimetre-scale rigid assets.

Blender cameras default to a 0.1 m near clipping plane.  The review renderer
frames its camera at 1.7 times the asset extent, so a 3 cm floor drain puts the
camera only about 5.4 cm away and the complete asset disappears.  This exact
failure was reproduced in raw and clay marker reviews and fixed by tying the
near plane to the measured review radius.

SPEAR-lead-b has broken Git metadata, so this replayable exact-string patch is
kept in AVEngine.  A mismatch or second application fails instead of guessing.
"""

from __future__ import annotations

import argparse
from pathlib import Path


OLD_CAMERA = '''    camera_data = bpy.data.cameras.new("Camera")
    camera_data.lens = 58
    camera = bpy.data.objects.new("Camera", camera_data)
'''

NEW_CAMERA = '''    camera_data = bpy.data.cameras.new("Camera")
    camera_data.lens = 58
    # Blender's 0.1 m default clips an entire centimetre-scale asset when the
    # framing camera is necessarily closer than that. Tie the near plane to
    # the measured review radius while retaining a nonzero numerical floor.
    camera_data.clip_start = max(1.0e-4, radius * 0.02)
    camera_data.clip_end = max(10.0, radius * 20.0)
    camera = bpy.data.objects.new("Camera", camera_data)
'''

OLD_MANIFEST = '''        "resolution": [args.width, args.height],
        "samples": args.samples,
        "material_preview": material_preview,
'''

NEW_MANIFEST = '''        "resolution": [args.width, args.height],
        "samples": args.samples,
        "camera_clip": [camera_data.clip_start, camera_data.clip_end],
        "material_preview": material_preview,
'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one exact match, found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spear-root", required=True, type=Path)
    args = parser.parse_args()
    path = args.spear_root / "tools/blender_render_i23d_review.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(text, OLD_CAMERA, NEW_CAMERA, "camera near/far clip")
    text = replace_once(text, OLD_MANIFEST, NEW_MANIFEST, "camera clip evidence")
    path.write_text(text, encoding="utf-8")
    print(f"I23D_REVIEW_CAMERA_CLIP_FIX_OK {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
