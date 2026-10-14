"""Bind approved physical environmental assets to registered UE scenes."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

def environment_bindings(scene, house_id, registration, human_endpoints):
    """Require a native visual and a matching physical sound point for each source.

    A capture can opt in with inputs.kujiale_ue_asset_admission. Scene sources
    then declare native_asset_id. Existing recordings are never rewritten.
    """
    path=Path(registration)
    inventory=json.loads(path.read_text())
    assets={r["asset_id"]:r for r in inventory["assets"]}
    result={}
    for endpoint,spec in scene["sources"].items():
        if endpoint in human_endpoints:
            continue
        asset_id=spec.get("native_asset_id")
        if not asset_id or asset_id not in assets:
            raise ValueError(f"{endpoint}: physical native_asset_id is required")
        row=assets[asset_id]
        if row.get("ue_native_physical_status")!="pass":
            raise ValueError(f"{endpoint}: individual native physical admission is incomplete")
        witnesses=row.get("native_house_admissions",{})
        native=witnesses.get(house_id)
        plan=spec.get("native_scene_plan")
        if plan is not None:
            if not scene.get("require_current_native_environment") or plan.get("status")!="pass_geometry_requires_current_native_scene":
                raise ValueError(f"{endpoint}: current joint scene requires explicit native requalification")
            if plan.get("asset_id")!=asset_id or plan.get("silent_decor"):
                raise ValueError(f"{endpoint}: current geometry plan identity or emitter kind differs")
            if plan.get("sound_class") not in (row.get("allowed_v7_classes_candidate") or []):
                raise ValueError(f"{endpoint}: requested sound is outside the registered asset classes")
            native={**plan,"emitter":{"actual_m":plan["expected_emitter_m"]}}
        elif not native or native.get("status")!="pass_static_native":
            raise ValueError(f"{endpoint}: no passed native visual in {house_id}")
        if not native.get("ue_asset_paths"):
            raise ValueError(f"{endpoint}: native StaticMesh paths are missing")
        if row.get("silent_decor"):
            raise ValueError(f"{endpoint}: silent decoration cannot be an audio source")
        actual=np.asarray(native["emitter"]["actual_m"],float)
        trajectory=spec.get("trajectory",[])
        points=[np.asarray(t["position_m"],float) for t in trajectory]
        if (actual.shape!=(3,) or not np.isfinite(actual).all() or not points or
            any(point.shape!=(3,) or not np.isfinite(point).all() or np.linalg.norm(point-actual)>.0005 for point in points)):
            raise ValueError(f"{endpoint}: declared sound trajectory differs from the native physical emitter")
        # Moving animals require the separate skeletal trajectory admission.
        if spec.get("moving") or spec.get("native_motion"):
            raise ValueError(f"{endpoint}: spatial skeletal motion requires its own native trajectory admission")
        result[endpoint]={"asset":row,"native":native,"source_id":spec["actor_id"],
                          "registration":str(path)}
    return result


def require_binding_project(bindings, ue):
    """Refuse to render admitted package paths from a different UE project."""
    actual=Path(ue["uproject"]).resolve()
    for endpoint,binding in bindings.items():
        expected=binding["asset"].get("ue_project")
        if not expected or Path(expected).resolve()!=actual:
            raise ValueError(f"{endpoint}: runtime UE project differs from native asset admission")
