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
option(
  AVENGINE_HABITAT_BUILD_BULLET
  "Build optional Habitat Bullet physics with externally installed dependencies"
  OFF
)

option(
  AVENGINE_HABITAT_BUILD_PYTHON_BINDINGS
  "Build optional Habitat Python 3.12 bindings into an explicit staging root"
  OFF
)
set(
  AVENGINE_HABITAT_PYTHON_OUTPUT_DIR
  ""
  CACHE PATH
  "Explicit non-source package staging root for optional habitat_sim_bindings"
)
# Generate the four staged template inputs in the include layout used by
# Habitat sources. This target only consumes esp/core/configure.h.
set(ESP_BUILD_ASSIMP_SUPPORT OFF)
set(ESP_BUILD_WITH_AUDIO ${AVENGINE_HABITAT_BUILD_LEGACY_AUDIO_SENSOR})
set(ESP_BUILD_WITH_BACKGROUND_RENDERER OFF)
set(ESP_BUILD_WITH_BULLET ${AVENGINE_HABITAT_BUILD_BULLET})
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
if(AVENGINE_HABITAT_BUILD_BULLET)
  find_package(MagnumIntegration CONFIG REQUIRED COMPONENTS Bullet)
  find_package(Bullet REQUIRED Dynamics)
  target_include_directories(
    MagnumIntegration::Bullet
    SYSTEM
    INTERFACE
      "${MAGNUMINTEGRATION_INCLUDE_DIR}"
  )
endif()

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
if(AVENGINE_HABITAT_BUILD_BULLET)
  target_link_libraries(
    avengine_habitat_core
    PUBLIC
      MagnumIntegration::Bullet
      Bullet::Dynamics
  )
endif()
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

if(AVENGINE_HABITAT_BUILD_PYTHON_BINDINGS)
  if(NOT AVENGINE_HABITAT_PYTHON_OUTPUT_DIR)
    message(
      FATAL_ERROR
      "AVENGINE_HABITAT_PYTHON_OUTPUT_DIR is required when "
      "AVENGINE_HABITAT_BUILD_PYTHON_BINDINGS=ON. Set it to an explicit "
      "non-source package staging root."
    )
  endif()

  get_filename_component(
    _avengine_habitat_python_output_requested
    "${AVENGINE_HABITAT_PYTHON_OUTPUT_DIR}"
    ABSOLUTE
    BASE_DIR "${CMAKE_BINARY_DIR}"
  )
  # Resolve every existing output-path component before any output directory is
  # created. This makes an existing symlink back into native/habitat visible to
  # the containment check while still allowing a new output leaf.
  set(
    _avengine_habitat_python_output_existing_parent
    "${_avengine_habitat_python_output_requested}"
  )
  set(_avengine_habitat_python_output_missing_components)
  while(NOT EXISTS "${_avengine_habitat_python_output_existing_parent}")
    get_filename_component(
      _avengine_habitat_python_output_component
      "${_avengine_habitat_python_output_existing_parent}"
      NAME
    )
    get_filename_component(
      _avengine_habitat_python_output_parent
      "${_avengine_habitat_python_output_existing_parent}"
      DIRECTORY
    )
    if(
      "${_avengine_habitat_python_output_parent}" STREQUAL
      "${_avengine_habitat_python_output_existing_parent}"
      OR "${_avengine_habitat_python_output_component}" STREQUAL ""
    )
      message(
        FATAL_ERROR
        "Could not resolve an existing parent for "
        "AVENGINE_HABITAT_PYTHON_OUTPUT_DIR: "
        "${_avengine_habitat_python_output_requested}"
      )
    endif()
    list(
      PREPEND
      _avengine_habitat_python_output_missing_components
      "${_avengine_habitat_python_output_component}"
    )
    set(
      _avengine_habitat_python_output_existing_parent
      "${_avengine_habitat_python_output_parent}"
    )
  endwhile()
  get_filename_component(
    _avengine_habitat_python_output_root
    "${_avengine_habitat_python_output_existing_parent}"
    REALPATH
  )
  foreach(
    _avengine_habitat_python_output_component
    IN LISTS _avengine_habitat_python_output_missing_components
  )
    set(
      _avengine_habitat_python_output_root
      "${_avengine_habitat_python_output_root}/${_avengine_habitat_python_output_component}"
    )
  endforeach()
  get_filename_component(
    _avengine_habitat_source_root
    "${CMAKE_CURRENT_SOURCE_DIR}"
    REALPATH
  )
  file(
    RELATIVE_PATH
    _avengine_habitat_python_output_relative
    "${_avengine_habitat_source_root}"
    "${_avengine_habitat_python_output_root}"
  )
  # The output is outside only if its first relative-path component is the
  # exact parent component `..`. A source child named `..output` must not be
  # mistaken for an outside path by a string-prefix check.
  string(
    REPLACE "/"
    ";"
    _avengine_habitat_python_output_relative_components
    "${_avengine_habitat_python_output_relative}"
  )
  list(
    GET
    _avengine_habitat_python_output_relative_components
    0
    _avengine_habitat_python_output_first_component
  )
  if(NOT "${_avengine_habitat_python_output_first_component}" STREQUAL "..")
    message(
      FATAL_ERROR
      "AVENGINE_HABITAT_PYTHON_OUTPUT_DIR must be outside "
      "native/habitat source; refusing source-tree extension output: "
      "${_avengine_habitat_python_output_root}"
    )
  endif()

  find_package(
    Python
    3.12
    EXACT
    REQUIRED
    COMPONENTS Interpreter Development.Module
  )
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 CONFIG REQUIRED)
  find_package(MagnumBindings CONFIG REQUIRED)
  find_path(
    AVENGINE_MAGNUM_BINDINGS_PYTHON_INCLUDE_DIR
    NAMES Magnum/PythonBindings.h
    HINTS "${MAGNUMBINDINGS_INCLUDE_DIR}"
    NO_DEFAULT_PATH
  )
  if(NOT AVENGINE_MAGNUM_BINDINGS_PYTHON_INCLUDE_DIR)
    message(
      FATAL_ERROR
      "Installed MagnumBindings is missing Magnum/PythonBindings.h below "
      "its configured include root: ${MAGNUMBINDINGS_INCLUDE_DIR}"
    )
  endif()
  add_library(avengine_magnum_bindings_python INTERFACE)
  add_library(
    AVEngine::MagnumBindingsPython
    ALIAS avengine_magnum_bindings_python
  )
  target_include_directories(
    avengine_magnum_bindings_python
    SYSTEM
    INTERFACE
      "${AVENGINE_MAGNUM_BINDINGS_PYTHON_INCLUDE_DIR}"
  )
  target_link_libraries(
    avengine_magnum_bindings_python
    INTERFACE
      Magnum::Magnum
  )
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/PythonBindingSources.cmake")

  set(
    _avengine_habitat_python_module_dir
    "${_avengine_habitat_python_output_root}/habitat_sim/_ext"
  )
  file(MAKE_DIRECTORY "${_avengine_habitat_python_module_dir}")
  pybind11_add_module(
    avengine_habitat_python_bindings
    MODULE
    ${AVENGINE_HABITAT_PYTHON_BINDING_SOURCES}
  )
  add_library(
    AVEngine::HabitatPythonBindings
    ALIAS avengine_habitat_python_bindings
  )
  set_target_properties(
    avengine_habitat_python_bindings
    PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED YES
      CXX_EXTENSIONS NO
      POSITION_INDEPENDENT_CODE ON
      OUTPUT_NAME habitat_sim_bindings
      LIBRARY_OUTPUT_DIRECTORY "${_avengine_habitat_python_module_dir}"
      RUNTIME_OUTPUT_DIRECTORY "${_avengine_habitat_python_module_dir}"
  )
  # HabitatCore carries the eight selected static importer registration sources
  # through its usage interface. The final Python extension is the consumer.
  target_link_libraries(
    avengine_habitat_python_bindings
    PRIVATE
      AVEngine::HabitatCore
      AVEngine::MagnumBindingsPython
  )
endif()
