# AVEngine-owned static build wiring for selected MIT Habitat-Sim source.
cmake_minimum_required(VERSION 3.16)
project(AVEngineHabitat LANGUAGES CXX)

option(
  AVENGINE_HABITAT_GFX_BATCH_WITH_CUDA
  "Enable CUDA-GL interop after its helper source is separately selected"
  OFF
)
if(AVENGINE_HABITAT_GFX_BATCH_WITH_CUDA)
  message(
    FATAL_ERROR
    "CUDA-GL interop requires a separately reviewed helper-source integration; "
    "this isolated gfx_batch slice contains no CUDA helper headers."
  )
endif()

option(
  AVENGINE_HABITAT_BUILD_RLR_ADAPTER
  "Build the optional static RLR acoustic-context adapter with a user-installed SDK"
  OFF
)
option(
  AVENGINE_HABITAT_BUILD_LEGACY_AUDIO_SENSOR
  "Build the optional legacy Habitat AudioSensor with a user-installed RLR SDK"
  OFF
)

# Generate the four staged template inputs in the include layout used by
# Habitat sources. This target only consumes esp/core/configure.h.
set(ESP_BUILD_ASSIMP_SUPPORT OFF)
set(ESP_BUILD_WITH_AUDIO ${AVENGINE_HABITAT_BUILD_LEGACY_AUDIO_SENSOR})
set(ESP_BUILD_WITH_BACKGROUND_RENDERER OFF)
set(ESP_BUILD_WITH_BULLET OFF)
set(ESP_BUILD_WITH_CUDA OFF)
set(ESP_DEFAULT_PBRSHADER_CONFIG_REL_PATH "")
set(ESP_DEFAULT_PBRSHADER_CONFIG "")
set(ESP_DEFAULT_PHYSICS_CONFIG_REL_PATH "./config/default.physics_config.json")
set(ESP_DEFAULT_PHYSICS_CONFIG "${ESP_DEFAULT_PHYSICS_CONFIG_REL_PATH}")
foreach(_avengine_habitat_config_dir core gfx physics sensor)
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/esp/${_avengine_habitat_config_dir}")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/esp/${_avengine_habitat_config_dir}/configure.h.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/esp/${_avengine_habitat_config_dir}/configure.h"
  )
endforeach()

find_package(Corrade REQUIRED PluginManager Utility)
find_package(
  Magnum
  REQUIRED
  AnyImageConverter
  AnyImageImporter
  AnySceneImporter
  DebugTools
  GL
  MaterialTools
  MeshTools
  Primitives
  SceneGraph
  SceneTools
  Shaders
  Trade
)
find_package(
  MagnumPlugins
  REQUIRED
  GltfImporter
  PrimitiveImporter
  StanfordImporter
  StbImageImporter
  StbImageConverter
)
find_package(RapidJSON CONFIG REQUIRED)
find_package(tinyxml2 CONFIG REQUIRED)
find_package(recastnavigation CONFIG REQUIRED)
find_package(OpenMP QUIET)

if(
  AVENGINE_HABITAT_BUILD_RLR_ADAPTER
  OR AVENGINE_HABITAT_BUILD_LEGACY_AUDIO_SENSOR
)
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/RlrSdk.cmake")
endif()
if(AVENGINE_HABITAT_BUILD_RLR_ADAPTER)
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/RlrAudioSources.cmake")
endif()

if(MAGNUM_TARGET_EGL)
  find_package(Magnum REQUIRED WindowlessEglApplication)
elseif(CORRADE_TARGET_APPLE)
  find_package(Magnum REQUIRED WindowlessCglApplication)
elseif(CORRADE_TARGET_UNIX)
  find_package(Magnum REQUIRED WindowlessGlxApplication)
elseif(CORRADE_TARGET_WINDOWS)
  find_package(Magnum REQUIRED WindowlessWglApplication)
else()
  message(FATAL_ERROR "Unsupported platform for AVEngine Habitat gfx_batch")
endif()

corrade_add_resource(
  GfxBatchShaderResources
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/gfx_batch/Shaders.conf"
)
corrade_add_resource(
  GfxShaderResources
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/gfx/Shaders.conf"
)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/GfxBatchSources.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CoreSources.cmake")

add_library(
  avengine_habitat_gfx_batch
  STATIC
  ${AVENGINE_HABITAT_GFX_BATCH_SOURCES}
  ${GfxBatchShaderResources}
)
add_library(AVEngine::HabitatGfxBatch ALIAS avengine_habitat_gfx_batch)
set_target_properties(
  avengine_habitat_gfx_batch
  PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(
  avengine_habitat_gfx_batch
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
)
target_link_libraries(
  avengine_habitat_gfx_batch
  PUBLIC
    Corrade::Utility
    Magnum::Magnum
    Magnum::GL
    Magnum::MeshTools
    Magnum::SceneTools
    Magnum::Shaders
    Magnum::Trade
    Magnum::WindowlessApplication
)

# An installed EGL application target can exist without defining this macro in
# Magnum/configure.h. Keep the source platform branch aligned with the selected
# CMake platform target and propagate it to HabitatCore through this public
# dependency.
if(MAGNUM_TARGET_EGL)
  target_compile_definitions(
    avengine_habitat_gfx_batch
    PUBLIC
      MAGNUM_TARGET_EGL
  )
endif()

# Keep static importer registration in the final consuming executable. Under a
# static Magnum/MagnumPlugins installation, each exact plugin target provides
# its importStaticPlugin.cpp through INTERFACE_SOURCES. These targets belong to
# this static core's usage interface, rather than its archive implementation:
# a binding or consumer that links AVEngine::HabitatCore compiles the
# registration units itself. No importer source checkout or handwritten
# CORRADE_PLUGIN_IMPORT is required.
add_library(
  avengine_habitat_core
  STATIC
  ${AVENGINE_HABITAT_CORE_SOURCES}
  ${GfxShaderResources}
)
add_library(AVEngine::HabitatCore ALIAS avengine_habitat_core)
set_target_properties(
  avengine_habitat_core
  PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(
  avengine_habitat_core
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
)
# RapidJSON is header-only and appears in selected public Habitat headers.
target_include_directories(
  avengine_habitat_core
  SYSTEM
  PUBLIC
    "${RapidJSON_INCLUDE_DIRS}"
)
target_link_libraries(
  avengine_habitat_core
  PUBLIC
    AVEngine::HabitatGfxBatch
    Corrade::PluginManager
    Corrade::Utility
    Magnum::DebugTools
    Magnum::GL
    Magnum::Magnum
    Magnum::MaterialTools
    Magnum::MeshTools
    Magnum::Primitives
    Magnum::SceneGraph
    Magnum::SceneTools
    Magnum::Shaders
    Magnum::Trade
    Magnum::WindowlessApplication
  PRIVATE
    tinyxml2_static
    RecastNavigation::Detour
    RecastNavigation::Recast
)
if(AVENGINE_HABITAT_BUILD_LEGACY_AUDIO_SENSOR)
  target_link_libraries(avengine_habitat_core PUBLIC AVEngine::RlrSdk)
endif()
# These targets must stay in the static core's consumer interface. Placing
# their registration translation units inside this archive would allow a final
# static link to omit them as unreferenced archive members.
target_link_libraries(
  avengine_habitat_core
  INTERFACE
    Magnum::AnyImageConverter
    Magnum::AnyImageImporter
    Magnum::AnySceneImporter
    MagnumPlugins::GltfImporter
    MagnumPlugins::PrimitiveImporter
    MagnumPlugins::StanfordImporter
    MagnumPlugins::StbImageImporter
    MagnumPlugins::StbImageConverter
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(avengine_habitat_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(AVENGINE_HABITAT_BUILD_RLR_ADAPTER)
  add_library(
    avengine_habitat_rlr_audio
    STATIC
    ${AVENGINE_HABITAT_RLR_AUDIO_SOURCES}
  )
  add_library(AVEngine::HabitatRlrAudio ALIAS avengine_habitat_rlr_audio)
  set_target_properties(
    avengine_habitat_rlr_audio
    PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED YES
      CXX_EXTENSIONS NO
      POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(
    avengine_habitat_rlr_audio
    PUBLIC
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
  )
  target_include_directories(
    avengine_habitat_rlr_audio
    SYSTEM
    PRIVATE
      "${RapidJSON_INCLUDE_DIRS}"
  )
  target_link_libraries(
    avengine_habitat_rlr_audio
    PUBLIC
      AVEngine::RlrSdk
    PRIVATE
      Corrade::Utility
  )
endif()
