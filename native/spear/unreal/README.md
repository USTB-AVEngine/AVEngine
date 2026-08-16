# AVEngine-owned SPEAR UE source slice

This directory is the source-only UE closure selected for the current
Apartment and InteriorAgent/Kujiale visual routes. It is not a packaged SPEAR
runtime, a content distribution, or a full Editor integration.

plugins/ contains four retained game-side modules:

| Module | Retained runtime responsibility |
| --- | --- |
| SpCore | -sp-config-file loading, RPC/shared-memory identifiers, POSIX shared-memory lifetime, callable-function registration and UE object utilities |
| SpServices | rpclib/MessagePack RPC, engine initialization/termination, serial begin_frame/execute_frame/end_frame work queue, shared-memory and Unreal-object services |
| SpUnrealTypes | SPEAR game engine/mode, SceneCapture GPU readback and shared-memory pixel arrays, object-ID proxy/segmentation types and world initialization |
| SpModuleRules | common UE Build.cs policy and explicit external SDK resolution for the retained modules |

SpearSim/ carries the project descriptor, Game/Editor target source and small
Config/*.ini inputs. SpUnrealTypesEditor plus SpModuleRulesEditor are retained
only because DefaultEngine.ini selects SpUnrealTypesEditor.SpUnrealEdEngine when
the existing Kujiale route starts UnrealEditor with -game.

## Deliberate Editor boundary

This is not a full Editor support claim. SpCoreEditor and SpServicesEditor are
intentionally omitted, so editor-world RPC such as
get_editor_subsystem_by_class, Editor UI automation, and SPEAR Editor/Python
workflows are unsupported. The current Kujiale runner uses the game-side
Instance.get_game() service after an UnrealEditor -game launch; it does not need
those omitted editor-world services.

## External SDKs

The copied SpModuleRules.Build.cs reads only the following explicit absolute
user-installed prefixes. It never walks to a SPEAR checkout, a sibling source
repository, or a vendored dependency tree.

| Variable | Required installed layout |
| --- | --- |
| AVENGINE_SPEAR_BOOST_ROOT | include/boost/predef.h |
| AVENGINE_SPEAR_RPCLIB_ROOT | include/rpc/client.h, lib/librpc.a on Linux/macOS or lib/rpc.lib on Windows |
| AVENGINE_SPEAR_YAML_CPP_ROOT | include/yaml-cpp/yaml.h, lib/libyaml-cpp.a on Linux/macOS or lib/yaml-cpp.lib on Windows |

Boost is retained as an external headers SDK because this selected closure uses
Boost headers/inline facilities and has no selected boost::filesystem or
Boost.UnitTest link use. rpclib and yaml-cpp must be installed as static
archives for the current UBT rule. The currently exercised production platform
is Linux; Windows/macOS path branches are source support, not validation
evidence.

With an external UE 5.5 installation, first copy this source-only directory
into a fresh external project stage and assemble legal assets there. A normal
source build then begins with an explicit Game or Editor target, for example:

    stage_root=/path/to/fresh/avengine-spear-ue-stage
    cp -a "$AVENGINE_ROOT/native/spear/unreal/." "$stage_root/"
    "$UE_ROOT/Engine/Build/BatchFiles/Linux/Build.sh" \
      SpearSimEditor Linux Development \
      -Project="$stage_root/SpearSim/SpearSim.uproject" -WaitMutex

The stage must be newly created for its build; do not point UBT at the Git
source tree. This stage intentionally makes no UE build, package, or GPU
execution claim.

## External asset and runtime boundary

No Content/ directory is present here. The retained
plugins/SpContent/SpContent.uplugin is only the plugin mount descriptor; the
camera Blueprint, segmentation materials, maps, Apartment content, Kujiale
USD/MDL scene data, textures, meshes and all other .uasset/.umap bytes must
come from separately authorized user-provided data. In particular,
DefaultEngine.ini still names /Game/SPEAR/Scenes/apartment_0000/... and the
Apartment runner dynamically loads /SpContent/Blueprints/BP_CameraSensor; they
cannot work until those legal assets are supplied outside Git.

Likewise Unreal Engine/UAT, the USDImporter engine plugin used by the Kujiale
runner, cooked PAKs, executables, plugin/project binaries, Intermediate,
Saved, shader/DDC outputs and generated media remain external. A deployment
may assemble these source files and authorized content into an external
staging/project directory, but AVEngine ships neither an asset synchronizer nor
a packaged executable in this path.

PROVENANCE.md records every selected upstream source group and the local
configuration adaptation. Most retained source is SPEAR/Intel MIT under
LICENSES/SPEAR-MIT.txt; Assert.{h,cpp} additionally retain the adapted
PPK_ASSERT WTFPLv2 notice at LICENSES/PPK_ASSERT-WTFPLv2.txt, and
SuppressCompilerWarnings.h additionally retains the Microsoft MIT notice at
LICENSES/Microsoft-MIT.txt. Those source notices do not grant any right to
redistribute UE or room assets.
