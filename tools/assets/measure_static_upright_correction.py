#!/usr/bin/env python3
"""Measure how far a rigid reconstruction is from standing straight.

Image-to-3D output arrives pitched: the camera in the product view looks
slightly down, and the reconstruction inherits that. Nine meshes measured 5.7
to 22.5 degrees off upright, which costs a visibly leaning object and about 12
percent of its height, because the finalizer scales the bounding box height and
a leaning box has a taller bounding box than the object.

Nothing in the static chain corrects it. This tool does not correct it either -
it measures, so that whichever fix is chosen has a number to apply and a number
to verify against.

Two independent authorities estimate the object's own up direction, following
the pattern the animal support-plane leveller established:

  * the dominant horizontal panel - cluster the area-weighted face normals and
    take the strongest cluster within 45 degrees of vertical, which is a
    cabinet top, a base plate or a speaker's top disc;
  * the base panel - extract flat faces as dense cells in (normal, plane
    offset) space and take the one furthest along the downward direction, which
    is the surface the object rests on.

Both are read off flat faces of the object itself, so neither assumes the
object is already standing straight. An earlier version fitted a plane to the
lowest slab by height; on a leaning object that slab is a sliver along the
contact edge and its fit is noise, which showed up as the two authorities
disagreeing by up to 51 degrees on meshes whose base is perfectly flat.

They still fail differently - the panel cluster is unreliable on a shape with
no flat horizontal face, the base panel on an object standing on a few small
feet - so requiring them to agree is what makes the answer trustworthy, and a
disagreement is reported as a refusal rather than averaged away.

The correction is the shortest rotation taking the measured up onto +Y in glTF
coordinates. Yaw is deliberately not part of it: yaw is a reviewed decision
about which face is the front, not something geometry alone can settle.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

SCHEMA = "avengine_static_upright_correction_v1"
GLTF_UP = np.array([0.0, 1.0, 0.0])


def load_glb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and triangles with every node transform applied."""

    data = path.read_bytes()
    json_length = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20 : 20 + json_length].decode("utf-8"))
    offset = 20 + json_length + 8

    def accessor(index: int) -> np.ndarray:
        item = gltf["accessors"][index]
        view = gltf["bufferViews"][item["bufferView"]]
        start = offset + view.get("byteOffset", 0) + item.get("byteOffset", 0)
        dtype = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}[
            item["componentType"]
        ]
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[item["type"]]
        flat = np.frombuffer(data, dtype=dtype, count=item["count"] * width, offset=start)
        return flat.reshape(-1, width) if width > 1 else flat

    def matrix_of(node) -> np.ndarray:
        if "matrix" in node:
            return np.array(node["matrix"], dtype=float).reshape(4, 4).T
        result = np.eye(4)
        if "scale" in node:
            result = np.diag(list(node["scale"]) + [1.0]) @ result
        if "rotation" in node:
            x, y, z, w = node["rotation"]
            block = np.eye(4)
            block[:3, :3] = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            )
            result = block @ result
        if "translation" in node:
            block = np.eye(4)
            block[:3, 3] = node["translation"]
            result = block @ result
        return result

    chunks: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    total = 0

    def walk(index: int, parent: np.ndarray) -> None:
        nonlocal total
        node = gltf["nodes"][index]
        world = parent @ matrix_of(node)
        if "mesh" in node:
            for primitive in gltf["meshes"][node["mesh"]]["primitives"]:
                local = accessor(primitive["attributes"]["POSITION"]).astype(float)
                chunks.append((world[:3, :3] @ local.T).T + world[:3, 3])
                faces.append(
                    accessor(primitive["indices"]).astype(np.int64).reshape(-1, 3) + total
                )
                total += len(local)
        for child in node.get("children", []):
            walk(child, world)

    for root in gltf["scenes"][gltf.get("scene", 0)]["nodes"]:
        walk(root, np.eye(4))
    return np.concatenate(chunks), np.concatenate(faces)


def main_shell(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Faces of the largest connected component, by area.

    Reconstructions carry inner shells and specks; both would drag a plane fit
    or a normal cluster away from the surface a reviewer is looking at.
    """

    keys = np.round(points / 1.0e-6).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    welded = inverse[faces]
    parent = np.arange(len(first))

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != root:
            parent[index], index = root, parent[index]
        return root

    for a, b, c in welded:
        for left, right in ((a, b), (b, c)):
            ra, rb = find(left), find(right)
            if ra != rb:
                parent[ra] = rb
    label = np.array([find(i) for i in range(len(first))])
    area = triangle_area(points, faces)
    roots, inverse_root = np.unique(label[welded[:, 0]], return_inverse=True)
    biggest = roots[int(np.argmax(np.bincount(inverse_root, weights=area)))]
    return faces[label[welded[:, 0]] == biggest]


def triangle_area(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    cross = np.cross(
        points[faces[:, 1]] - points[faces[:, 0]],
        points[faces[:, 2]] - points[faces[:, 0]],
    )
    return 0.5 * np.linalg.norm(cross, axis=1)


def planar_panels(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    bucket_deg: float,
    offset_buckets: int,
) -> list[dict]:
    """Flat faces of the object, as dense cells in (normal, plane offset) space.

    A manufactured object is mostly flat panels, so faces that share a normal
    and a plane offset are one real face. Nothing here refers to the world
    axes, so the result does not change when the object leans.
    """

    cross = np.cross(
        points[faces[:, 1]] - points[faces[:, 0]],
        points[faces[:, 2]] - points[faces[:, 0]],
    )
    area = 0.5 * np.linalg.norm(cross, axis=1)
    keep = area > 0
    normals = cross[keep] / (2.0 * area[keep, None])
    area = area[keep]
    centroid = ((points[faces[:, 0]] + points[faces[:, 1]] + points[faces[:, 2]]) / 3.0)[
        keep
    ]
    offset = np.einsum("ij,ij->i", normals, centroid)

    diagonal = float(np.linalg.norm(points.max(0) - points.min(0)))
    step = np.deg2rad(bucket_deg)
    theta = np.arccos(np.clip(normals[:, 1], -1.0, 1.0))
    phi = np.arctan2(normals[:, 2], normals[:, 0])
    offset_step = diagonal / max(offset_buckets, 1)
    key = (
        (theta / step).astype(np.int64) * 10_000_000
        + ((phi + np.pi) / step).astype(np.int64) * 10_000
        + np.round(offset / max(offset_step, 1e-9)).astype(np.int64)
    )
    order = np.argsort(key)
    key, area, normals, offset = key[order], area[order], normals[order], offset[order]
    total = float(area.sum())
    panels = []
    for group in np.split(np.arange(len(key)), np.flatnonzero(np.diff(key)) + 1):
        weight = float(area[group].sum())
        direction = (normals[group] * area[group, None]).sum(0)
        norm = float(np.linalg.norm(direction))
        if norm <= 0:
            continue
        panels.append(
            {
                "normal": direction / norm,
                "offset": float(np.average(offset[group], weights=area[group])),
                "area_share": weight / max(total, 1e-12),
            }
        )
    panels.sort(key=lambda item: -item["area_share"])
    return panels


def base_panel_up(
    panels: list[dict], down: np.ndarray, *, minimum_area_share: float
) -> dict:
    """Up as the negated normal of the panel the object rests on.

    ``down`` only selects which hemisphere to look in; the estimate itself is
    the panel's own normal, so the two authorities remain independent
    measurements rather than one measurement used twice.
    """

    candidates = [
        panel
        for panel in panels
        if float(np.dot(panel["normal"], down)) > np.cos(np.deg2rad(45.0))
        and panel["area_share"] >= minimum_area_share
    ]
    if not candidates:
        return {
            "available": False,
            "reason": (
                "no flat face within 45 degrees of down holds at least "
                f"{minimum_area_share:.3f} of the area"
            ),
        }
    # Furthest along its own normal in the down direction: the lowest flat face.
    base = max(candidates, key=lambda panel: panel["offset"])
    up = -np.asarray(base["normal"])
    return {
        "available": True,
        "up": (up / np.linalg.norm(up)).tolist(),
        "area_share": base["area_share"],
        "candidates": len(candidates),
    }


def panel_cluster_up(points: np.ndarray, faces: np.ndarray, bucket_deg: float) -> dict:
    cross = np.cross(
        points[faces[:, 1]] - points[faces[:, 0]],
        points[faces[:, 2]] - points[faces[:, 0]],
    )
    area = 0.5 * np.linalg.norm(cross, axis=1)
    keep = area > 0
    normals = cross[keep] / (2.0 * area[keep, None])
    area = area[keep]
    # Fold the two sides together: a top and a bottom panel of the same slab
    # both measure the same up direction.
    folded = np.where(normals[:, 1:2] < 0, -normals, normals)
    near_vertical = folded[:, 1] > np.cos(np.deg2rad(45.0))
    if not near_vertical.any():
        return {"available": False, "reason": "no face within 45 degrees of horizontal"}
    folded, weights = folded[near_vertical], area[near_vertical]

    step = np.deg2rad(bucket_deg)
    theta = np.arccos(np.clip(folded[:, 1], -1.0, 1.0))
    phi = np.arctan2(folded[:, 2], folded[:, 0])
    key = (theta / step).astype(np.int64) * 100000 + (
        (phi + np.pi) / step
    ).astype(np.int64)
    order = np.argsort(key)
    key, folded, weights = key[order], folded[order], weights[order]
    best_weight, best_up = 0.0, None
    for group in np.split(np.arange(len(key)), np.flatnonzero(np.diff(key)) + 1):
        weight = float(weights[group].sum())
        if weight > best_weight:
            best_weight = weight
            direction = (folded[group] * weights[group, None]).sum(0)
            best_up = direction / np.linalg.norm(direction)
    return {
        "available": True,
        "up": best_up.tolist(),
        "area_share": float(best_weight / triangle_area(points, faces).sum()),
    }


def rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    axis = np.cross(source, target)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.dot(source, target))
    if sine < 1e-12:
        return np.eye(3) if cosine > 0 else -np.eye(3)
    axis = axis / sine
    skew = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    angle = float(np.arctan2(sine, cosine))
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    cosine = float(
        np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)
    )
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def measure(
    path: Path,
    *,
    slab: float,
    bucket_deg: float,
    tolerance_deg: float,
    maximum_tilt_deg: float,
) -> dict:
    points, faces = load_glb(path)
    shell = main_shell(points, faces)
    panel = panel_cluster_up(points, shell, bucket_deg)
    if panel.get("available"):
        panels = planar_panels(
            points, shell, bucket_deg=bucket_deg, offset_buckets=200
        )
        plane = base_panel_up(
            panels, -np.asarray(panel["up"]), minimum_area_share=slab
        )
    else:
        plane = {"available": False, "reason": "no horizontal panel to orient the search"}

    authorities = [item for item in (plane, panel) if item.get("available")]
    result: dict = {
        "schema": SCHEMA,
        "input": str(path.resolve()),
        "base_panel": plane,
        "dominant_horizontal_panel": panel,
        "tolerance_deg": tolerance_deg,
    }
    if len(authorities) < 2:
        result["agreed"] = False
        result["refusal"] = "both authorities must be available"
        return result

    disagreement = angle_between(plane["up"], panel["up"])
    result["authority_disagreement_deg"] = round(disagreement, 3)
    if disagreement > tolerance_deg:
        result["agreed"] = False
        result["refusal"] = (
            "the base panel and the dominant horizontal panel disagree by "
            f"{disagreement:.1f} degrees, more than the {tolerance_deg} allowed"
        )
        return result

    up = np.asarray(plane["up"]) + np.asarray(panel["up"])
    up = up / np.linalg.norm(up)
    tilt = angle_between(up, GLTF_UP)
    if tilt > maximum_tilt_deg:
        result["agreed"] = False
        result["measured_up_gltf"] = [round(float(value), 6) for value in up]
        result["tilt_from_upright_deg"] = round(tilt, 3)
        result["refusal"] = (
            f"both authorities agree on {tilt:.1f} degrees, past the "
            f"{maximum_tilt_deg} a stably resting object can plausibly be; they "
            "have most likely selected the same wrong face"
        )
        return result
    rotation = rotation_between(up, GLTF_UP)
    axis = np.array([rotation[2, 1] - rotation[1, 2],
                     rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]])
    norm = float(np.linalg.norm(axis))
    result.update(
        {
            "agreed": True,
            "measured_up_gltf": [round(float(value), 6) for value in up],
            "tilt_from_upright_deg": round(tilt, 3),
            "correction": {
                "convention": "rotate the mesh by this to bring its own up onto +Y in glTF",
                "axis_gltf": [round(float(value), 6) for value in (axis / norm)]
                if norm > 1e-12
                else [1.0, 0.0, 0.0],
                "angle_deg": round(tilt, 3),
                "matrix_gltf": [[round(float(v), 6) for v in row] for row in rotation],
            },
            "height_error_if_uncorrected": _height_error(points, shell, rotation),
        }
    )
    return result


def _height_error(points: np.ndarray, faces: np.ndarray, rotation: np.ndarray) -> dict:
    """What the tilt costs, in the units the finalizer actually uses.

    The finalizer scales so the bounding-box height hits the profile target. A
    leaning object has a taller bounding box than itself, so the object ends up
    shorter than the target by exactly this ratio.
    """

    used = points[np.unique(faces)]
    before = float(used[:, 1].max() - used[:, 1].min())
    after = (rotation @ used.T).T
    upright = float(after[:, 1].max() - after[:, 1].min())
    return {
        "bounding_box_height_now": round(before, 6),
        "bounding_box_height_upright": round(upright, 6),
        "object_is_shorter_than_target_by": round(1.0 - upright / max(before, 1e-12), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--report", type=Path, help="fresh path; never overwritten")
    parser.add_argument(
        "--minimum-base-area-share",
        type=float,
        default=0.005,
        dest="base_slab_fraction",
        help=(
            "smallest area share a face may hold and still count as the base. "
            "0.005 is where a smart speaker's narrow base ring is found while "
            "a television's splayed feet still refuse; going lower makes both "
            "authorities land on the same slanted foot face and agree on a "
            "wrong answer"
        ),
    )
    parser.add_argument("--normal-bucket-deg", type=float, default=5.0)
    parser.add_argument(
        "--authority-tolerance-deg",
        type=float,
        default=1.5,
        help=(
            "how far the two independent up estimates may disagree. Calibrated "
            "by applying the correction and measuring again: every asset whose "
            "authorities agreed within 1.3 degrees came back within 0.63, while "
            "the one that disagreed by 3.0 was still 11 degrees off after its "
            "own correction. The disagreement is the reliability signal"
        ),
    )
    parser.add_argument(
        "--maximum-plausible-tilt-deg",
        type=float,
        default=30.0,
        help=(
            "refuse beyond this. The pose guard asks for an object resting "
            "stably, so a larger answer means a face was mis-selected, and two "
            "authorities can agree on the same wrong face"
        ),
    )
    args = parser.parse_args()

    reports = []
    for path in args.input:
        report = measure(
            path,
            slab=args.base_slab_fraction,
            bucket_deg=args.normal_bucket_deg,
            tolerance_deg=args.authority_tolerance_deg,
            maximum_tilt_deg=args.maximum_plausible_tilt_deg,
        )
        reports.append(report)
        if report.get("agreed"):
            cost = report["height_error_if_uncorrected"][
                "object_is_shorter_than_target_by"
            ]
            print(
                f"{path.parent.name:<46} tilt={report['tilt_from_upright_deg']:6.2f}deg "
                f"disagreement={report['authority_disagreement_deg']:5.2f}deg "
                f"height_loss={cost * 100:5.2f}%"
            )
        else:
            print(f"{path.parent.name:<46} REFUSED  {report['refusal']}")

    if args.report is not None:
        if args.report.exists():
            raise SystemExit(f"{args.report} exists; give a fresh path")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {"schema": SCHEMA + "_batch", "reports": reports},
                ensure_ascii=False,
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
