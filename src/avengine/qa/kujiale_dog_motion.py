"""Calibrate imported UE bone axes against retained authored Kujiale motion."""
from __future__ import annotations

import math
import numpy as np

from avengine.qa.kujiale_asset_geometry import rotation


def expected_local_bones(frame, yaw):
    root = np.asarray(frame["world_root_m"])
    result = {}
    for label in ["FL", "FR", "HL", "HR", "mouth"]:
        world = (frame["mouth_world_m"] if label=="mouth"
                 else frame["feet"][label]["ankle_world_m"])
        canonical = (np.asarray(world)-root) @ rotation(yaw)
        result[label] = (canonical[[0, 2, 1]]*100).tolist()
    return result


def calibrate(native, authored):
    labels = ["FL", "FR", "HL", "HR", "mouth"]
    source = np.asarray([native["frames"][0]["native_component_bones_cm"][k]
                         for k in labels])
    target_first = expected_local_bones(authored["frames"][0], authored["yaw_rad"][0])
    target = np.asarray([target_first[k] for k in labels])
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    x, y = source-source_mean, target-target_mean
    u, singular, vt = np.linalg.svd(x.T@y)
    sign = np.eye(3)
    sign[-1, -1] = np.linalg.det(vt.T@u.T)
    matrix = vt.T@sign@u.T
    scale = float(np.sum(singular*np.diag(sign))/np.sum(x*x))
    translation = target_mean-scale*matrix@source_mean
    errors = []
    per_frame = []
    for index, frame in enumerate(native["frames"]):
        actual = np.asarray([frame["native_component_bones_cm"][k] for k in labels])
        normalized = actual @ matrix.T*scale+translation
        expected_map = expected_local_bones(authored["frames"][index],
                                            authored["yaw_rad"][index])
        expected = np.asarray([expected_map[k] for k in labels])
        delta = np.linalg.norm(normalized-expected, axis=1)/100.
        errors.extend(delta.tolist())
        per_frame.append({"frame": index,
                          "maximum_bone_error_m": float(delta.max())})
    maximum = max(errors)
    return {"status": "pass" if maximum <= .003 and scale > 0 else "fail",
            "rotation_ue": matrix.tolist(), "translation_ue_cm": translation.tolist(),
            "uniform_scale": scale, "maximum_all_key_bone_error_m": maximum,
            "tolerance_m": .003, "frames": per_frame,
            "method": "One proper rigid axis/unit calibration from frame zero; "
                      "same transform independently checked on every authored key, "
                      "four paws and mouth. No per-frame fitted corrections."}


def ue_rotator(matrix):
    """Return Roll/Pitch/Yaw for UE's right/forward/up Euler convention."""
    m = np.asarray(matrix)
    pitch = math.asin(float(np.clip(m[2, 0], -1, 1)))
    if abs(math.cos(pitch)) < 1e-8:
        raise ValueError("rig calibration has an ambiguous gimbal-lock rotation")
    yaw = math.atan2(m[1, 0], m[0, 0])
    roll = math.atan2(-m[2, 1], m[2, 2])
    return {"Roll": math.degrees(roll), "Pitch": math.degrees(pitch),
            "Yaw": math.degrees(yaw)}

def fixed_floor_calibration(frames,tolerance=.012,margin=.0015):
    """One global skin/root offset from complete native four-paw evidence.

    Contact/penetration gates stay fixed. This never fits individual frames.
    """
    lower=-float('inf');upper=float('inf')
    for frame in frames:
        rows=frame["sole_depth"]["four_paws"]
        if len(rows)!=4 or any(row.get("floor_gap_m") is None for row in rows):
            raise ValueError("incomplete native four-paw evidence")
        gaps=[row["floor_gap_m"] for row in rows]
        lower=max(lower,-tolerance-min(gaps))
        upper=min(upper,tolerance-min(gaps))
        for row in rows:
            if row.get("expected_in_contact"):
                lower=max(lower,-tolerance-row["floor_gap_m"])
                upper=min(upper,tolerance-row["floor_gap_m"])
    if lower>upper:raise ValueError("no single native skin/floor correction can satisfy all keys")
    lo,hi=lower+margin,upper-margin
    shift=(lower+upper)/2 if lo>hi else min(max(0.,lo),hi)
    if abs(shift)>.012:raise ValueError("fixed root correction exceeds 12 mm; repair derived rig")
    return {"fixed_root_y_offset_m":float(shift),"feasible_interval_m":[float(lower),float(upper)],
            "tolerance_m":tolerance,"reference_frame_count":len(frames),
            "method":"one fixed native skin/root calibration from complete measured clips; no per-frame correction"}
