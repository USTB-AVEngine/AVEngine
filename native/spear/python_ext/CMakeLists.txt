# AVEngine-owned build wiring for the selected MIT SPEAR python_ext source.
cmake_minimum_required(VERSION 3.18)
project(AVEngineSpearPythonExtension LANGUAGES CXX)

# This target is deliberately installed only into an explicit external staging
# prefix. It must never default to /usr/local or place a compiled extension in
# the selected source directory.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  message(
    FATAL_ERROR
    "Set CMAKE_INSTALL_PREFIX to an explicit existing external staging "
    "directory before configuring AVEngine's SPEAR extension."
  )
endif()
if(NOT IS_DIRECTORY "${CMAKE_INSTALL_PREFIX}")
  message(
    FATAL_ERROR
    "CMAKE_INSTALL_PREFIX must be an existing external staging directory: "
    "${CMAKE_INSTALL_PREFIX}"
  )
endif()

get_filename_component(
  _avengine_spear_install_root
  "${CMAKE_INSTALL_PREFIX}"
  REALPATH
)
get_filename_component(
  _avengine_spear_source_root
  "${CMAKE_CURRENT_SOURCE_DIR}"
  REALPATH
)
file(
  RELATIVE_PATH
  _avengine_spear_install_relative
  "${_avengine_spear_source_root}"
  "${_avengine_spear_install_root}"
)
string(
  REPLACE "/"
  ";"
  _avengine_spear_install_relative_components
  "${_avengine_spear_install_relative}"
)
list(
  GET
  _avengine_spear_install_relative_components
  0
  _avengine_spear_install_first_component
)
if(NOT "${_avengine_spear_install_first_component}" STREQUAL "..")
  message(
    FATAL_ERROR
    "CMAKE_INSTALL_PREFIX must resolve outside native/spear/python_ext; "
    "refusing source-tree extension output: "
    "${_avengine_spear_install_root}"
  )
endif()

set(
  AVENGINE_SPEAR_RPCLIB_ROOT
  ""
  CACHE PATH
  "External user-installed rpclib SDK prefix for AVEngine's SPEAR extension"
)
if(NOT AVENGINE_SPEAR_RPCLIB_ROOT)
  message(
    FATAL_ERROR
    "AVENGINE_SPEAR_RPCLIB_ROOT is required. It must name an external "
    "rpclib SDK prefix, not a SPEAR source checkout."
  )
endif()
get_filename_component(
  _avengine_spear_rpclib_root
  "${AVENGINE_SPEAR_RPCLIB_ROOT}"
  REALPATH
)
set(
  _avengine_spear_rpclib_cmake_dir
  "${_avengine_spear_rpclib_root}/lib/cmake/rpclib"
)
if(
  NOT EXISTS
  "${_avengine_spear_rpclib_cmake_dir}/rpclibConfig.cmake"
)
  message(
    FATAL_ERROR
    "AVENGINE_SPEAR_RPCLIB_ROOT lacks the installed rpclib CMake export: "
    "${_avengine_spear_rpclib_cmake_dir}/rpclibConfig.cmake"
  )
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(
  Python
  3.11
  EXACT
  REQUIRED
  COMPONENTS Interpreter Development.Module
)
set(
  AVENGINE_SPEAR_NANOBIND_DIR
  ""
  CACHE PATH
  "External nanobind CMake config directory for AVEngine's SPEAR extension"
)
if(NOT AVENGINE_SPEAR_NANOBIND_DIR)
  message(
    FATAL_ERROR
    "AVENGINE_SPEAR_NANOBIND_DIR is required. Derive it from the explicit "
    "Python interpreter with -m nanobind --cmake_dir."
  )
endif()
get_filename_component(
  _avengine_spear_nanobind_dir
  "${AVENGINE_SPEAR_NANOBIND_DIR}"
  REALPATH
)
if(
  NOT EXISTS
  "${_avengine_spear_nanobind_dir}/nanobind-config.cmake"
)
  message(
    FATAL_ERROR
    "AVENGINE_SPEAR_NANOBIND_DIR lacks nanobind-config.cmake: "
    "${_avengine_spear_nanobind_dir}/nanobind-config.cmake"
  )
endif()
find_package(
  nanobind
  CONFIG
  REQUIRED
  PATHS "${_avengine_spear_nanobind_dir}"
  NO_DEFAULT_PATH
)
find_package(Threads REQUIRED)
find_package(
  rpclib
  CONFIG
  REQUIRED
  PATHS "${_avengine_spear_rpclib_cmake_dir}"
  NO_DEFAULT_PATH
)

nanobind_add_module(
  avengine_spear_ext
  NB_STATIC
  cpp/spear_ext.cpp
)
add_library(
  AVEngine::SpearPythonExtension
  ALIAS avengine_spear_ext
)
set_target_properties(
  avengine_spear_ext
  PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(
  avengine_spear_ext
  PRIVATE
    rpclib::rpc
    Threads::Threads
)

# The AVEngine host/game client imports this exact top-level module name. No
# spear package or spear_ext compatibility package is installed.
install(
  TARGETS avengine_spear_ext
  LIBRARY DESTINATION "."
  RUNTIME DESTINATION "."
)
