"""Per-paw contact readbacks from admitted static native geometry."""
from __future__ import annotations

import numpy as np

from avengine.qa.kujiale_asset_geometry import rotation


def landmark_contacts(geometry, row, vertices, placement):
    if row["category"] not in ["dog", "cat"]:
        return {"status": "not_applicable", "reason": "whole declared attachment surface is checked"}
    landmarks = row.get("landmarks_local_m", {})
    feet = [(key, value) for key, value in sorted(landmarks.items())
            if key.startswith("foot_")]
    if len(feet) != 4:
        return {"status": "deferred_to_native_skeletal_motion",
                "reason": "legacy walker registration has no four static sole landmarks"}
    root = np.asarray(placement["root_m"])
    matrix = rotation(placement["yaw_rad"])
    checks = []
    vertices = np.asarray(vertices)
    for name, point in feet:
        point = np.asarray(point)
        distance = np.linalg.norm(vertices-point, axis=1)
        index = int(distance.argmin())
        centers=np.asarray([value for _,value in feet])
        horizontal=np.linalg.norm(vertices[:,[0,2]][:,None,:]-centers[:,[0,2]][None,:,:],axis=2)
        owner=np.argmin(horizontal,axis=1)
        foot_index=[key for key,_ in feet].index(name)
        region=(owner==foot_index)&(horizontal[:,foot_index]<=.04)
        if not region.any():
            checks.append({'foot':name,'status':'fail','reason':'no actual native sole geometry in the registered paw region'})
            continue
        candidates=vertices[region]
        native=candidates[np.argmin(candidates[:,1])]
        world = root + matrix @ native
        hit = geometry.ray(world+[0, .10, 0], [0, -1, 0], .22)
        gap = None if hit is None else float(world[1]-hit["point_m"][1])
        is_floor = hit is not None and "/Meshes/floor/" in hit["material_path"]
        passed = is_floor and abs(gap) <= .012
        checks.append({"foot": name, "declared_landmark_local_m": point.tolist(),
                       "native_lod0_vertex_local_m": native.tolist(),
                       "landmark_to_nearest_native_vertex_m": float(distance[index]),
                       "sole_region_radius_m": .04,
                       "world_m": world.tolist(), "native_floor_ray": hit,
                       "surface_gap_m": gap, "tolerance_m": .012,
                       "status": "pass" if passed else "fail"})
    return {"status": "pass" if all(c["status"]=="pass" for c in checks) else "fail",
            "checks": checks, "method": "Each registered paw, lowest actual UE LOD0 sole vertex in its nonoverlapping 4 cm paw region, "
            "bound native floor triangle; 12 mm gate unchanged"}
