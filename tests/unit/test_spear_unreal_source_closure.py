"""Static checks for the selected, source-only SPEAR UE integration slice."""

from __future__ import annotations

import json
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
UE_ROOT = REPOSITORY / "native/spear/unreal"
PLUGIN_ROOT = UE_ROOT / "plugins"
PROJECT_ROOT = UE_ROOT / "SpearSim"


def _descriptor(relative_path: str) -> dict[str, object]:
    return json.loads((UE_ROOT / relative_path).read_text(encoding="utf-8"))


def _module_names(relative_path: str) -> list[str]:
    descriptor = _descriptor(relative_path)
    modules = descriptor.get("Modules")
    assert isinstance(modules, list)
    return [module["Name"] for module in modules if isinstance(module, dict)]


def test_runtime_source_closure_contains_the_current_frame_path() -> None:
    required = (
        "plugins/SpCore/Source/SpCore/SharedMemory.cpp",
        "plugins/SpCore/Source/SpCore/SpFuncComponent.cpp",
        "plugins/SpServices/Source/SpServices/SpServices.cpp",
        "plugins/SpServices/Source/SpServices/EngineService.h",
        "plugins/SpServices/Source/SpServices/SharedMemoryService.h",
        "plugins/SpUnrealTypes/Source/SpUnrealTypes/SpSceneCaptureComponent2D.cpp",
        "plugins/SpUnrealTypes/Source/SpUnrealTypes/SpMeshProxyComponentManager.h",
        "plugins/SpModuleRules/Source/SpModuleRules/SpModuleRules.Build.cs",
        "SpearSim/Source/SpearSim/SpearSim.cpp",
        "SpearSim/Config/DefaultEngine.ini",
        "SpearSim/Config/DefaultSpear.ini",
    )
    assert all((UE_ROOT / relative_path).is_file() for relative_path in required)


def test_only_the_editor_game_launch_bridge_is_declared() -> None:
    assert _module_names("plugins/SpCore/SpCore.uplugin") == ["SpCore"]
    assert _module_names("plugins/SpServices/SpServices.uplugin") == ["SpServices"]
    assert _module_names("plugins/SpUnrealTypes/SpUnrealTypes.uplugin") == [
        "SpUnrealTypes",
        "SpUnrealTypesEditor",
    ]
    assert _module_names("plugins/SpModuleRules/SpModuleRules.uplugin") == [
        "SpModuleRules",
        "SpModuleRulesEditor",
    ]
    assert (
        PLUGIN_ROOT
        / "SpUnrealTypes/Source/SpUnrealTypesEditor/SpUnrealEdEngine.h"
    ).is_file()
    assert (
        PLUGIN_ROOT
        / "SpModuleRules/Source/SpModuleRulesEditor/SpModuleRulesEditor.Build.cs"
    ).is_file()
    assert not (PLUGIN_ROOT / "SpCore/Source/SpCoreEditor").exists()
    assert not (PLUGIN_ROOT / "SpServices/Source/SpServicesEditor").exists()


def test_build_wiring_uses_only_explicit_installed_sdk_roots() -> None:
    build_rules = (
        PLUGIN_ROOT / "SpModuleRules/Source/SpModuleRules/SpModuleRules.Build.cs"
    ).read_text(encoding="utf-8")
    for environment_variable in (
        "AVENGINE_SPEAR_BOOST_ROOT",
        "AVENGINE_SPEAR_RPCLIB_ROOT",
        "AVENGINE_SPEAR_YAML_CPP_ROOT",
    ):
        assert environment_variable in build_rules
    assert "third_party" not in build_rules
    assert "external/SPEAR" not in build_rules
    # RequireStaticLibrary logs through the instance SP_LOG_GET_PREFIX helper;
    # declaring it static makes UBT's C# rules compilation fail with CS0120.
    assert "private string RequireStaticLibrary(" in build_rules
    assert "private static string RequireStaticLibrary(" not in build_rules


def test_project_and_asset_boundary_are_source_only() -> None:
    project = _descriptor("SpearSim/SpearSim.uproject")
    assert project["AdditionalPluginDirectories"] == ["../plugins"]
    assert (PLUGIN_ROOT / "SpContent/SpContent.uplugin").is_file()
    assert not (PROJECT_ROOT / "Content").exists()
    assert not (PLUGIN_ROOT / "SpContent/Content").exists()
    for generated_directory in ("Binaries", "Intermediate", "Saved"):
        assert not any(
            path.is_dir() and path.name == generated_directory
            for path in UE_ROOT.rglob(generated_directory)
        )
