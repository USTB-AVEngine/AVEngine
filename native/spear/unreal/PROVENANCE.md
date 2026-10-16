# SPEAR UE source provenance

This is a selective source import, not a repository-history merge and not a
statement that AVEngine distributes a SPEAR/UE runtime.

## Source revision and treatment

The public source baseline for every selected UE source/configuration path is
spear-sim/spear@2d4575587a9be39a40ba89c4259836a85ccd3f3f.
The bytes were selected from the maintained transition checkout
/data/jzy/code/AVEngine/external/SPEAR@684789cc71eaaf39e362d9d7a7b4b0b7f0af8568.
That path is a migration source only, never a build or runtime lookup used by
this AVEngine tree.

For the selected paths, the transition source is identical to the local
SPEAR-lead-b@251bd5e0d3d1e7297ec072bb9b0df9ef63f864b7 carrier. Relative to
the public baseline, the only selected-path difference is
cpp/unreal_projects/SpearSim/Config/DefaultGame.ini: local commit
a14f44a48bb084ed4906d530cdc4e051872428ce adds
/Game/MyAssets/Audioset to DirectoriesToAlwaysCook. This is an adapted local
configuration behavior, not a public spear-sim/spear revision.

To satisfy AVEngine's normal diff hygiene, the import removes only inherited
trailing whitespace and extra terminal blank lines from selected text files.
That mechanical normalization changes no token, source selection, build
policy, or runtime behavior.

All selected files retain their original copyright and license headers. Most
retain the SPEAR/Intel MIT notice, whose full text is at
LICENSES/SPEAR-MIT.txt at the repository root. The two
plugins/SpCore/Source/SpCore/Assert.{h,cpp} files additionally retain the
Gregory Pakosz PPK_ASSERT attribution and WTFPLv2 terms named by their
preserved source URL; the full text is at LICENSES/PPK_ASSERT-WTFPLv2.txt.
plugins/SpCore/Source/SpCore/SuppressCompilerWarnings.h additionally retains
the Microsoft Corporation MIT copyright notice; its MIT text is at
LICENSES/Microsoft-MIT.txt. THIRD_PARTY_NOTICES.md records these limited
embedded origins without claiming a separate PPK_ASSERT or Microsoft source
revision that the carried source does not identify.

## Per-module map

| AVEngine target | Original source path | Relationship and retained purpose |
| --- | --- | --- |
| plugins/SpCore/SpCore.uplugin, plugins/SpCore/Source/SpCore/** | cpp/unreal_plugins/SpCore/SpCore.uplugin, cpp/unreal_plugins/SpCore/Source/SpCore/** | adapted, 35 source/build files: boot config, shared memory, callable functions, object/config utilities. The descriptor is narrowed to the game-side Runtime module because the selected route does not use SpCoreEditor. |
| plugins/SpServices/SpServices.uplugin, plugins/SpServices/Source/SpServices/** | matching cpp/unreal_plugins/SpServices/** paths | adapted, 26 source/build files: rpclib/MessagePack RPC and engine/frame/process lifecycle services. The descriptor omits SpServicesEditor, which only supplies editor-world RPC/subsystem behavior. |
| plugins/SpUnrealTypes/SpUnrealTypes.uplugin, plugins/SpUnrealTypes/Source/SpUnrealTypes/** | matching cpp/unreal_plugins/SpUnrealTypes/** paths | adapted, 52 game-side files: custom engine/game mode, SceneCapture readback, shared-memory frame data and proxy/segmentation types. |
| plugins/SpUnrealTypes/Source/SpUnrealTypesEditor/** | matching cpp/unreal_plugins/SpUnrealTypes/Source/SpUnrealTypesEditor/** | adapted, 8-file narrow Editor-game launch bridge required by USpUnrealEdEngine in the retained project config. It is not a general Editor API claim. |
| plugins/SpModuleRules/SpModuleRules.uplugin, plugins/SpModuleRules/Source/SpModuleRules/** | matching cpp/unreal_plugins/SpModuleRules/** paths | adapted, 2 game-side files. SpModuleRules.Build.cs keeps its UBT policy but is changed to require explicit external Boost/rpclib/yaml-cpp prefixes instead of the original checkout-relative dependency tree. |
| plugins/SpModuleRules/Source/SpModuleRulesEditor/** | matching cpp/unreal_plugins/SpModuleRules/Source/SpModuleRulesEditor/** | adapted, 2 files required as the base UBT rule for the retained SpUnrealTypesEditor bridge. |
| SpearSim/SpearSim.uproject, SpearSim/Source/**, SpearSim/Config/*.ini | matching cpp/unreal_projects/SpearSim/** paths | adapted, 15 descriptor/source/config files. SpearSim.uproject changes only AdditionalPluginDirectories to AVEngine's adjacent ../plugins; DefaultGame.ini retains the recorded local AlwaysCook adaptation. |
| plugins/SpContent/SpContent.uplugin | cpp/unreal_plugins/SpContent/SpContent.uplugin | adapted, one content-mount descriptor only. Its Content/** is deliberately absent and must be supplied as legal external data. |

## Explicit exclusions

The selection excludes every Content/** and map/Blueprint/material binary, UE
installation, Engine source/content, SPEAR third_party/** submodules
(Boost/rpclib/yaml-cpp), upstream examples/tests/build/package metadata,
Binaries, Intermediate, Saved, PAKs, cooked package, generated project files,
datasets, USD/MDL scene inputs, and all compiled extension/library bytes.

Boost (BSL-1.0), rpclib (MIT), and yaml-cpp (MIT) remain user-installed build
SDKs. Their source, submodules, headers, libraries, CMake exports and
transitive dependencies are not imported into AVEngine. Users that redistribute
a resulting build must independently satisfy the applicable engine, SDK and
asset licenses.
