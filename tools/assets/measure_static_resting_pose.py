"""Measure how a published static asset sits on a floor, and record it in the asset.

The question this answers is the only one a room placer needs: put this object's
origin at floor height, and does it stand up straight. Recording the answer in
the asset means no room ever re-derives it.

Why this replaces long_axis_elevation_deg as the gate. That measure takes the
principal axis of the area-weighted covariance and reports its elevation, which
for a 16:9 flat panel is the width - a horizontal axis. A television lying on
its back still has a level width axis, so it scored 1.0 degrees and passed while
being 26 degrees off. The lean is a rotation about the very axis being measured,
so that measure integrates it out. Base normal tilt cannot miss it: the base is
what touches the floor.

It also needs only one authority. measure_static_upright_correction.py requires
two independent estimates to agree before it will act, which is right when the
output is a rotation applied to somebody's mesh - one bad guess would corrupt
the asset. It is wrong as an acceptance check: it refuses on every finalized
mesh here, including ones its own surviving authority puts within 1.05 degrees
of upright, because no flat downward face holds 0.5 percent of the area once
the correction has been baked in. After the fact you do not need consensus, you
need a number.

The bands are deliberately loose. Reconstructed 3D is noisy and a placement
that is a degree or two off, or floating a few millimetres, looks correct and
behaves correctly. Only a lean somebody would notice is worth failing.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_static_upright_correction import load_glb

# A lean this small is invisible at furniture scale; between the two it is
# visible if you look for it and still sits stably; beyond the second it reads
# as fallen over rather than placed.
LEVEL_DEG = 3.0
ACCEPTABLE_DEG = 8.0

# The base is whatever faces the floor in the bottom slice of the object. The
# slice is a fraction of height rather than a fixed distance so a 7 cm soundbar
# and a 1 m tower are treated alike.
BASE_SLICE_FRACTION = 0.05
# Faces are selected by being roughly vertical-facing, either way up, and then
# flipped to point down. Trusting the winding instead would report "no base"
# on any mesh whose normals are inverted, which is a silent pass rather than a
# failure. 0.5 is cos 60 degrees, so a lean is still measurable well past the
# point where it is already rejected.
VERTICAL_COSINE = 0.5
CONTACT_BAND_M = 0.002


def measure(glb: Path) -> dict:
    vertices, faces = load_glb(glb)
    y = vertices[:, 1]
    height = float(y.max() - y.min())

    corners = vertices[faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    areas = np.linalg.norm(normals, axis=1) / 2.0
    total_area = float(areas.sum())
    unit = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)

    # Selected by the face's lowest vertex, not its centroid. A face that would
    # rest on a floor has a vertex at the contact point, whereas its centroid
    # rises out of a thin slice as soon as the object leans: a 33 cm cabinet
    # tipped 26 degrees puts its base centroid 5.5 cm up, which a slice of 5
    # percent of height would miss entirely - and missing it reports "no base"
    # rather than "leaning", which is a silent pass.
    lowest_y = corners[:, :, 1].min(axis=1)
    base = (np.abs(unit[:, 1]) >= VERTICAL_COSINE) & (
        lowest_y < y.min() + BASE_SLICE_FRACTION * height
    )
    # Point every selected face downward so the mean is not cancelled by
    # whichever way the mesh happens to be wound.
    downward = np.where(unit[:, 1] > 0.0, -1.0, 1.0)[:, None] * unit

    result = {
        "measured_from": glb.name,
        "height_m": round(height, 4),
        "base_plane_offset_m": round(float(y.min()), 4),
        "tolerance": {
            "level_deg": LEVEL_DEG,
            "acceptable_deg": ACCEPTABLE_DEG,
            "note": (
                "loose on purpose; reconstructed geometry is noisy and a degree "
                "or two of lean, or a few millimetres of float, is fine"
            ),
        },
        "how_to_place": (
            "put the asset origin at floor height and apply yaw only; the base "
            "offset above is already zero for assets the finalizer grounded, so "
            "no room needs to re-derive this"
        ),
    }

    if not base.any():
        result["base_normal_tilt_deg"] = None
        result["verdict"] = "no_base_found"
        result["verdict_reason"] = (
            "no face in the bottom "
            f"{BASE_SLICE_FRACTION:.0%} of the height is within 60 degrees of "
            "horizontal, so there is nothing that would rest on a floor"
        )
        return result

    weights = areas[base]
    mean_normal = (downward[base] * weights[:, None]).sum(axis=0)
    mean_normal /= np.linalg.norm(mean_normal)
    tilt = math.degrees(math.acos(min(1.0, abs(float(mean_normal[1])))))

    contact = vertices[y < y.min() + CONTACT_BAND_M]
    def extent(points):
        return [
            round(float(points[:, 0].max() - points[:, 0].min()), 4),
            round(float(points[:, 2].max() - points[:, 2].min()), 4),
        ]

    result["base_normal_tilt_deg"] = round(tilt, 2)
    result["base_area_share"] = round(float(weights.sum() / total_area), 4)
    # Reported, never gated. A domed base contacts a floor over a small patch
    # even when it is perfectly level, so a small number here is not a defect.
    result["contact_extent_m"] = extent(contact) if len(contact) > 2 else [0.0, 0.0]
    result["footprint_extent_m"] = extent(vertices)
    result["verdict"] = (
        "level" if tilt <= LEVEL_DEG
        else "acceptable" if tilt <= ACCEPTABLE_DEG
        else "leaning"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "write the block into each asset.json under geometry.resting_pose. "
            "Safe on published assets: the index carries no hash of asset.json "
            "and the mesh is not touched"
        ),
    )
    args = parser.parse_args()

    records = []
    for asset_json in sorted(args.asset_root.rglob("asset.json")):
        directory = asset_json.parent
        glb = directory / "finalized.glb"
        if not glb.is_file():
            continue
        pose = measure(glb)
        relative = directory.relative_to(args.asset_root).as_posix()
        records.append({"asset": relative, **pose})

        marker = {"level": "  ", "acceptable": " ~", "leaning": " !"}.get(
            pose["verdict"], " ?"
        )
        tilt = pose["base_normal_tilt_deg"]
        print(
            f"{marker} {relative:<56} "
            f"{'  n/a' if tilt is None else f'{tilt:5.2f}'} deg  {pose['verdict']}"
        )

        if args.apply:
            record = json.loads(asset_json.read_text(encoding="utf-8"))
            record.setdefault("geometry", {})["resting_pose"] = pose
            record.setdefault("acceptance", {})["resting_pose_verdict"] = pose["verdict"]
            asset_json.write_text(
                json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8"
            )

    leaning = [r for r in records if r["verdict"] == "leaning"]
    unknown = [r for r in records if r["verdict"] == "no_base_found"]
    print(
        f"\n{len(records)} assets: "
        f"{sum(r['verdict'] == 'level' for r in records)} level, "
        f"{sum(r['verdict'] == 'acceptable' for r in records)} acceptable, "
        f"{len(leaning)} leaning, {len(unknown)} with no base found"
    )
    for r in leaning + unknown:
        print(f"  {r['verdict']}: {r['asset']} ({r['base_normal_tilt_deg']} deg)")

    if args.report:
        args.report.write_text(
            json.dumps(
                {
                    "schema": "avengine_static_resting_pose_v1_batch",
                    "level_deg": LEVEL_DEG,
                    "acceptable_deg": ACCEPTABLE_DEG,
                    "assets": records,
                },
                ensure_ascii=False,
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.report}")
    if args.apply:
        print("applied geometry.resting_pose to each asset.json")
    return 1 if leaning else 0


if __name__ == "__main__":
    raise SystemExit(main())
