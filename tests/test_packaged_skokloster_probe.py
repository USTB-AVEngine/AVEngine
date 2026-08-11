from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _probe_module():
    root = Path(__file__).resolve().parents[1]
    staging_path = root / "tools/probe_packaged_skokloster_room.py"
    path = (
        staging_path
        if staging_path.is_file()
        else root / "tools/qa/probe_packaged_skokloster_room.py"
    )
    spec = importlib.util.spec_from_file_location("packaged_skokloster_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_noncallable_mesh_and_material_proxies_use_property_fallbacks() -> None:
    probe = _probe_module()

    class FakeComponent:
        GetStaticMesh = object()
        GetMaterial = object()
        GetNumMaterials = object()

        @staticmethod
        def get_property_value(*, property_name: str, as_handle: bool):
            assert as_handle is True
            if property_name == "StaticMesh":
                return 101
            if property_name == "OverrideMaterials":
                return []
            raise AssertionError(property_name)

    class FakeStaticMesh:
        @staticmethod
        def get_property_value(
            *, property_name: str, as_handle: bool = False, as_value: bool = False
        ):
            if property_name == "StaticMaterials[0].MaterialInterface":
                assert as_handle is True
                return 202
            if property_name == "StaticMaterials":
                assert as_value is True
                return [{"MaterialInterface": 202}]
            raise AssertionError(property_name)

    mesh_handle, mesh_method = probe._component_mesh_handle(FakeComponent())
    material_handle, material_method = probe._component_material_handle(
        FakeComponent(), FakeStaticMesh()
    )
    slot_count, slot_count_method = probe._material_slot_count(
        FakeComponent(), FakeStaticMesh()
    )
    assert (mesh_handle, mesh_method) == (
        101,
        "UStaticMeshComponent.StaticMesh_property",
    )
    assert (material_handle, material_method) == (
        202,
        "UStaticMesh.StaticMaterials[0].MaterialInterface_property",
    )
    assert (slot_count, slot_count_method) == (
        1,
        "UStaticMesh.StaticMaterials_property",
    )


def test_noncallable_bounds_proxy_uses_live_component_property() -> None:
    probe = _probe_module()

    class FakeActor:
        GetActorBounds = object()

    class FakeComponent:
        @staticmethod
        def get_property_value(*, property_name: str, as_value: bool):
            assert property_name == "Bounds"
            assert as_value is True
            return {
                "Origin": {"X": 1.0, "Y": 2.0, "Z": 3.0},
                "BoxExtent": {"X": 4.0, "Y": 5.0, "Z": 6.0},
            }

    bounds, method = probe._actor_bounds(FakeActor(), FakeComponent())
    assert method == "USceneComponent.Bounds_property"
    assert bounds["minimum_cm"] == [-3.0, -3.0, -3.0]
    assert bounds["maximum_cm"] == [5.0, 7.0, 9.0]
