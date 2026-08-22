"""Headless UE editor export: apartment_0000 level -> textured glb.

Exports only the actors whose bounds sit inside the apartment volume (the
level also carries a ~150m sky dome / backdrop that would swamp the model),
with glTF material baking enabled.
"""
import unreal

OUTPUT = "/data/jzy/tmp/apartment_textured_export/apartment_0000.glb"
MAP_PATH = "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000"

unreal.log("[gltf-export] loading map " + MAP_PATH)
unreal.EditorLoadingAndSavingUtils.load_map(MAP_PATH)
world = unreal.UnrealEditorSubsystem().get_editor_world()

selected = set()
skipped = []
actors = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.StaticMeshActor)
for actor in actors:
    origin, extent = actor.get_actor_bounds(False)
    if (
        max(extent.x, extent.y) < 3000.0
        and extent.z < 1000.0
        and abs(origin.x) < 1500.0
        and abs(origin.y) < 1500.0
        and -200.0 < origin.z < 600.0
    ):
        selected.add(actor)
    else:
        skipped.append(f"{actor.get_actor_label()} extent=({extent.x:.0f},{extent.y:.0f},{extent.z:.0f})")
unreal.log(f"[gltf-export] selected {len(selected)} static mesh actors, skipped {len(skipped)}")
for line in skipped[:10]:
    unreal.log("[gltf-export] skipped: " + line)

options = unreal.GLTFExportOptions()


def try_set(name, value):
    try:
        setattr(options, name, value)
        unreal.log(f"[gltf-export] set {name}={value}")
    except Exception as exc:  # noqa: BLE001
        unreal.log(f"[gltf-export] skip option {name}: {exc}")


try_set("export_uniform_scale", 0.01)
try:
    bake_size = unreal.GLTFMaterialBakeSize()
    bake_size.set_editor_property("x", 512)
    bake_size.set_editor_property("y", 512)
    bake_size.set_editor_property("auto_detect", False)
    try_set("default_material_bake_size", bake_size)
    try_set("texture_image_format", unreal.GLTFTextureImageFormat.JPEG)
    try_set("texture_image_quality", 82)
except Exception as exc:  # noqa: BLE001
    unreal.log(f"[gltf-export] bake size tuning failed: {exc}")
try:
    try_set("bake_material_inputs", unreal.GLTFMaterialBakeMode.USE_MESH_DATA)
except Exception as exc:  # noqa: BLE001
    unreal.log(f"[gltf-export] bake enum unavailable: {exc}")
try_set("export_lights", False)
try_set("export_cameras", False)
try_set("export_hidden_in_game", False)

result = unreal.GLTFExporter.export_to_gltf(world, OUTPUT, options, selected)
unreal.log("[gltf-export] result: " + str(result))
if not result:
    raise RuntimeError("glTF export returned false")
unreal.log("[gltf-export] DONE " + OUTPUT)
