"""Read every Kujiale contact-animation key from each imported UE dog."""
from pathlib import Path
import json
import os
import traceback
import time

import unreal

ROOT = Path(os.environ["V7_KUJIALE_ASSET_ROOT"])
MANIFEST = Path(os.environ["V7_KUJIALE_IMPORT_INPUTS"])
SOURCE = ROOT.parent.parent/"v7_human_roster_20261004"
inputs = json.loads(MANIFEST.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    temporary.replace(path)


def normal(name):
    return str(name).lower().replace("_", "").replace(" ", "").replace("-", "")


def xyz(value):
    return [value.x, value.y, value.z]


def run():
    results = []
    for row in inputs["assets"]:
        if not row["walk_allowed"]:
            continue
        receipt = json.loads((ROOT/"evidence/import"/inputs["revision"]/
                              (row["asset_id"]+".json")).read_text())
        for family in ["Idle", "Walking"]:
            result = {"asset_id": row["asset_id"], "family": family, "status": "fail",
                      "frames": [], "method": "Native UE AnimPose every authored 30 Hz key"}
            try:
                plan_path = SOURCE/"round2/assets/dogs"/row["body_id"]/("kujiale_"+family+".json")
                plan = json.loads(plan_path.read_text())
                mesh_path = next(x["path"] for x in receipt["motion"]
                                 if x["class"] == "SkeletalMesh")
                clip_path = next(x["path"] for x in receipt["motion"]
                                 if x["class"] == "AnimSequence" and x["name"] == plan["clip"])
                mesh, clip = unreal.load_asset(mesh_path), unreal.load_asset(clip_path)
                result["skeletal_materials"] = [{"interface":slot.material_interface.get_path_name(), "base":slot.material_interface.get_base_material().get_path_name() if isinstance(slot.material_interface, unreal.MaterialInstanceConstant) else slot.material_interface.get_path_name(), "used_with_skeletal_mesh":bool((slot.material_interface.get_base_material() if isinstance(slot.material_interface, unreal.MaterialInstanceConstant) else slot.material_interface).get_editor_property("used_with_skeletal_mesh"))} for slot in mesh.get_editor_property("materials")]
                options = unreal.AnimPoseEvaluationOptions()
                options.optional_skeletal_mesh = mesh
                first = unreal.AnimPoseExtensions.get_anim_pose_at_time(clip, 0., options)
                names = [str(x) for x in unreal.AnimPoseExtensions.get_bone_names(first)]
                selected = {}
                for side, suffix in [("FL", "lhand"), ("FR", "rhand"),
                                     ("HL", "lfoot"), ("HR", "rfoot"),
                                     ("mouth", "xtramouth")]:
                    selected[side] = next(n for n in names if normal(n).endswith(suffix))
                for label, suffix in [("pelvis", "pelvis"), ("neck", "neck"), ("spine", "spine")]:
                    matches = [n for n in names if normal(n).endswith(suffix)]
                    if matches:
                        selected[label] = matches[0]
                for index, frame in enumerate(plan["frames"]):
                    time_s = index/30.
                    pose = unreal.AnimPoseExtensions.get_anim_pose_at_time(clip, time_s, options)
                    bones = {
                        key: xyz(unreal.AnimPoseExtensions.get_bone_pose(
                            pose, name, unreal.AnimPoseSpaces.WORLD).translation)
                        for key, name in selected.items()
                    }
                    result["frames"].append({"frame": index, "time_s": time_s,
                                             "native_component_bones_cm": bones})
                result.update(status="pass_native_pose_readback", skeletal_mesh=mesh_path,
                              animation=clip_path, bone_names=names, selected_bones=selected,
                              authored_contact_plan=str(plan_path),
                              frame_count=len(result["frames"]),
                              animation_length_s=float(clip.get_editor_property("sequence_length")))
            except Exception:
                result["error"] = traceback.format_exc()
            write(ROOT/"evidence/native_motion_editor"/
                  (row["asset_id"]+"_"+family+".json"), result)
            results.append(result)
            unreal.log("KUJIALE_DOG_KEYS "+row["asset_id"]+" "+family+" "+result["status"])
    write(ROOT/"evidence/native_motion_editor_results.json", results)


run()
# FShaderCompilingManager implements IAssetCompilingManager. The editor's
# registered command drains shader, texture and mesh compilation before Python
# returns; quitting with background SubmitJobs still active caused SIGSEGV.
started=time.monotonic()
unreal.log("KUJIALE_SHADER_DRAIN_BEGIN")
unreal.SystemLibrary.execute_console_command(None, "Editor.AsyncAssetCompilationFinishAll")
write(ROOT/"evidence/native_motion_editor_shader_drain.json", {"status":"completed", "console_command":"Editor.AsyncAssetCompilationFinishAll", "seconds":time.monotonic()-started})
unreal.log("KUJIALE_SHADER_DRAIN_COMPLETE")
