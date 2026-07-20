#!/usr/bin/env python3
"""Prepare an external InteriorAgent USD stage for UE's runtime USD importer.

The output stage contains references to the user-downloaded source scene.  It
does not copy the source dataset or make it redistributable.  NVIDIA MDL inputs
that UE cannot evaluate at runtime are mirrored to portable USD PreviewSurface
nodes while the original source remains untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.optional_backends.interioragent_kujiale import (  # noqa: E402
    build_kujiale_review_plan,
    load_profile,
    load_room_metadata,
    preview_material_parameters,
)

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
except ImportError as exc:  # pragma: no cover - depends on optional USD install
    raise SystemExit(
        "Pixar USD Python bindings are required for this optional tool"
    ) from exc


def _input_values(shader: Any) -> dict[str, Any]:
    return {item.GetBaseName(): item.Get() for item in shader.GetInputs()}


def _texture_path(value: Any) -> Path | None:
    if not isinstance(value, Sdf.AssetPath) or not value.resolvedPath:
        return None
    candidate = Path(value.resolvedPath)
    return candidate if candidate.is_file() else None


def _connect_uv_texture(
    *,
    stage: Any,
    material_path: Any,
    name: str,
    texture: Path,
    color: tuple[float, float, float],
    uva: Any,
    raw: bool = False,
) -> Any:
    reader = UsdShade.Shader.Define(
        stage, material_path.AppendChild(f"{name}PrimvarReader")
    )
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    uv_source = reader
    uv_output = "result"
    if uva is not None and len(uva) >= 4:
        scale = (float(uva[0]), float(uva[1]))
        translation = (float(uva[2]), float(uva[3]))
        if scale != (1.0, 1.0) or translation != (0.0, 0.0):
            transform = UsdShade.Shader.Define(
                stage, material_path.AppendChild(f"{name}UVTransform")
            )
            transform.CreateIdAttr("UsdTransform2d")
            transform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(
                reader.ConnectableAPI(), "result"
            )
            transform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(
                Gf.Vec2f(*scale)
            )
            transform.CreateInput("translation", Sdf.ValueTypeNames.Float2).Set(
                Gf.Vec2f(*translation)
            )
            transform.CreateInput("rotation", Sdf.ValueTypeNames.Float).Set(0.0)
            transform.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            uv_source = transform

    node = UsdShade.Shader.Define(stage, material_path.AppendChild(name))
    node.CreateIdAttr("UsdUVTexture")
    node.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(texture.resolve()))
    )
    node.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(
        "raw" if raw else "sRGB"
    )
    node.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    node.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    node.CreateInput("fallback", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f(float(color[0]), float(color[1]), float(color[2]), 1.0)
    )
    node.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f(float(color[0]), float(color[1]), float(color[2]), 1.0)
    )
    node.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        uv_source.ConnectableAPI(), uv_output
    )
    node.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    return node


def _create_reference_stage(
    *, source: Path, output: Path, plan: dict[str, Any]
) -> Any:
    stage = Usd.Stage.CreateNew(str(output))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Scope.Define(stage, "/Root/Meshes")
    for scope in plan["selected_scopes"]:
        prim = UsdGeom.Scope.Define(stage, f"/Root/Meshes/{scope}").GetPrim()
        prim.GetReferences().AddReference(
            str(source), f"/Root/Meshes/{scope}"
        )
    if plan["include_rendering_scope"]:
        rendering = stage.DefinePrim("/Root/Rendering")
        rendering.GetReferences().AddReference(str(source), "/Root/Rendering")
    stage.GetRootLayer().customLayerData = {
        "avengine:adapter": "interioragent_mdl_to_usd_preview_surface_v1",
        "avengine:backendRole": plan["backend_role"],
        "avengine:source": str(source),
        "avengine:selectedScopes": ",".join(plan["selected_scopes"]),
    }
    stage.GetRootLayer().Save()
    return Usd.Stage.Open(str(output))


def prepare_adapter(
    *,
    source: Path,
    output: Path,
    profile_path: Path,
    result_path: Path,
    replace: bool,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    result_path = result_path.expanduser().resolve()
    if not source.is_file():
        raise RuntimeError(f"source USD stage does not exist: {source}")
    if source == output:
        raise RuntimeError("source and output USD stages must differ")
    if output.exists() and not replace:
        raise FileExistsError(f"refusing to replace output: {output}")
    if result_path.exists() and not replace:
        raise FileExistsError(f"refusing to replace result: {result_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if replace:
        output.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)

    profile = load_profile(profile_path)
    rooms_path = source.parent / "rooms.json"
    rooms = load_room_metadata(rooms_path) if rooms_path.is_file() else None
    plan = build_kujiale_review_plan(
        profile, source_stage=source, rooms=rooms
    )
    stage = _create_reference_stage(source=source, output=output, plan=plan)
    stage.SetEditTarget(stage.GetRootLayer())

    fixed_primvars = 0
    fixed_material_binding_apis = 0
    adapted_materials = 0
    textured_materials = 0
    glass_materials = 0
    unadapted_materials = 0
    mesh_count = 0

    for prim in list(stage.Traverse()):
        has_material_binding = any(
            relationship.GetName().startswith("material:binding")
            for relationship in prim.GetRelationships()
        )
        if has_material_binding and not prim.HasAPI(UsdShade.MaterialBindingAPI):
            UsdShade.MaterialBindingAPI.Apply(prim)
            fixed_material_binding_apis += 1

        if prim.IsA(UsdGeom.Mesh):
            mesh_count += 1
            api = UsdGeom.PrimvarsAPI(prim)
            for name in ("displayColor", "displayOpacity"):
                primvar = api.GetPrimvar(name)
                if not primvar:
                    continue
                value = primvar.Get()
                if value is not None and len(value) == 1:
                    primvar.SetInterpolation(UsdGeom.Tokens.constant)
                    primvar.SetIndices([0])
                    fixed_primvars += 1

        if not prim.IsA(UsdShade.Material):
            continue
        material = UsdShade.Material(prim)
        source_shader = material.ComputeSurfaceSource()
        if (
            not source_shader
            or not source_shader[0]
            or not source_shader[0].GetPrim().IsValid()
        ):
            unadapted_materials += 1
            continue

        shader = source_shader[0]
        values = _input_values(shader)
        try:
            mdl_source = str(shader.GetSourceAsset("mdl"))
        except RuntimeError:
            mdl_source = ""
        translated = preview_material_parameters(values, mdl_source=mdl_source)
        if translated["is_glass"]:
            glass_materials += 1

        preview = UsdShade.Shader.Define(
            stage, prim.GetPath().AppendChild("AVEnginePreviewSurface")
        )
        preview.CreateIdAttr("UsdPreviewSurface")
        preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            translated["roughness"]
        )
        preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
            translated["metallic"]
        )
        preview.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(
            translated["opacity"]
        )
        preview.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(translated["ior"])

        base_color = translated["base_color"]
        base_texture = _texture_path(values.get("BaseColor_Tex"))
        if translated["use_base_texture"] and base_texture is not None:
            texture = _connect_uv_texture(
                stage=stage,
                material_path=prim.GetPath(),
                name="AVEngineBaseColorTexture",
                texture=base_texture,
                color=base_color,
                uva=translated["base_uv"],
            )
            preview.CreateInput(
                "diffuseColor", Sdf.ValueTypeNames.Color3f
            ).ConnectToSource(texture.ConnectableAPI(), "rgb")
            textured_materials += 1
        else:
            preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*base_color)
            )

        normal_texture = _texture_path(values.get("Normal_Tex"))
        if translated["use_normal_texture"] and normal_texture is not None:
            texture = _connect_uv_texture(
                stage=stage,
                material_path=prim.GetPath(),
                name="AVEngineNormalTexture",
                texture=normal_texture,
                color=(1.0, 1.0, 1.0),
                uva=translated["normal_uv"],
                raw=True,
            )
            preview.CreateInput(
                "normal", Sdf.ValueTypeNames.Normal3f
            ).ConnectToSource(texture.ConnectableAPI(), "rgb")

        if translated["emissive_color"] is not None:
            preview.CreateInput(
                "emissiveColor", Sdf.ValueTypeNames.Color3f
            ).Set(Gf.Vec3f(*translated["emissive_color"]))

        preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(
            preview.ConnectableAPI(), "surface"
        )
        adapted_materials += 1

    stage.GetRootLayer().Save()
    result = {
        "status": "pass",
        "backend_role": plan["backend_role"],
        "dataset_id": plan["dataset_id"],
        "scene_id": plan["scene_id"],
        "source": str(source),
        "output": str(output),
        "selected_scopes": plan["selected_scopes"],
        "include_rendering_scope": plan["include_rendering_scope"],
        "mesh_count": mesh_count,
        "fixed_primvars": fixed_primvars,
        "fixed_material_binding_apis": fixed_material_binding_apis,
        "adapted_materials": adapted_materials,
        "textured_materials": textured_materials,
        "glass_materials": glass_materials,
        "unadapted_materials": unadapted_materials,
        "external_asset_policy": plan["external_asset_policy"],
        "claim_boundary": (
            "Derived USD stage references user-downloaded external data; "
            "it is local visual-adapter evidence and is not redistributable data."
        ),
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        type=Path,
        default=(
            REPOSITORY
            / "examples/m6z/interioragent_kujiale_0020_visual_profile.json"
        ),
    )
    parser.add_argument("--result", type=Path)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_path = args.result or args.output.with_suffix(".adapter.json")
    result = prepare_adapter(
        source=args.source,
        output=args.output,
        profile_path=args.profile,
        result_path=result_path,
        replace=args.replace,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
