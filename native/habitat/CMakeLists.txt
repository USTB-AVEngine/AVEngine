# AVEngine-owned isolated build wiring for selected MIT Habitat-Sim gfx_batch.
cmake_minimum_required(VERSION 3.16)
project(AVEngineHabitatGfxBatch LANGUAGES CXX)

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

# Generate the four staged template inputs in the include layout used by
# Habitat sources. This target only consumes esp/core/configure.h.
set(ESP_BUILD_ASSIMP_SUPPORT OFF)
set(ESP_BUILD_WITH_AUDIO OFF)
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

find_package(Corrade REQUIRED Utility)
find_package(Magnum REQUIRED GL MeshTools SceneTools Shaders Trade)
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
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/GfxBatchSources.cmake")

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
