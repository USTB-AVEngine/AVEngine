# AVEngine-owned discovery for a user-installed RLR Audio Propagation SDK.
#
# AVENGINE_RLR_SDK_ROOT names the official RLRAudioPropagationPkg directory,
# which contains headers/RLRAudioPropagation.h and
# libs/linux/x64/libRLRAudioPropagation.so. This module deliberately does not
# search a Habitat checkout, a submodule, or arbitrary system paths.

include_guard(GLOBAL)

set(
  AVENGINE_RLR_SDK_ROOT
  "$ENV{AVENGINE_RLR_SDK_ROOT}"
  CACHE PATH
  "Official user-installed RLRAudioPropagationPkg root"
)

if(NOT AVENGINE_RLR_SDK_ROOT)
  message(
    FATAL_ERROR
    "AVENGINE_RLR_SDK_ROOT is required when either "
    "AVENGINE_HABITAT_BUILD_RLR_ADAPTER or "
    "AVENGINE_HABITAT_BUILD_LEGACY_AUDIO_SENSOR is ON. "
    "Set it to the official RLRAudioPropagationPkg directory containing headers/ "
    "and libs/linux/x64/."
  )
endif()

get_filename_component(
  _avengine_rlr_sdk_root
  "${AVENGINE_RLR_SDK_ROOT}"
  REALPATH
)
if(NOT IS_DIRECTORY "${_avengine_rlr_sdk_root}")
  message(
    FATAL_ERROR
    "AVENGINE_RLR_SDK_ROOT is not a directory: ${_avengine_rlr_sdk_root}"
  )
endif()

function(_avengine_rlr_sdk_require_outside_git_checkout candidate_path)
  set(_avengine_rlr_sdk_ancestor "${candidate_path}")
  while(TRUE)
    if(
      EXISTS "${_avengine_rlr_sdk_ancestor}/.git"
      OR IS_SYMLINK "${_avengine_rlr_sdk_ancestor}/.git"
    )
      message(
        FATAL_ERROR
        "AVENGINE_RLR_SDK_ROOT must resolve outside a Git checkout: "
        "${candidate_path}"
      )
    endif()
    get_filename_component(
      _avengine_rlr_sdk_parent
      "${_avengine_rlr_sdk_ancestor}"
      DIRECTORY
    )
    if("${_avengine_rlr_sdk_parent}" STREQUAL "${_avengine_rlr_sdk_ancestor}")
      break()
    endif()
    set(_avengine_rlr_sdk_ancestor "${_avengine_rlr_sdk_parent}")
  endwhile()
endfunction()

function(
  _avengine_rlr_sdk_require_within_root
  candidate_path
  candidate_name
  result_variable
)
  get_filename_component(
    _avengine_rlr_sdk_candidate
    "${candidate_path}"
    REALPATH
  )
  file(
    RELATIVE_PATH
    _avengine_rlr_sdk_relative
    "${_avengine_rlr_sdk_root}"
    "${_avengine_rlr_sdk_candidate}"
  )
  if(
    "${_avengine_rlr_sdk_relative}" STREQUAL ".."
    OR "${_avengine_rlr_sdk_relative}" MATCHES "^\.\./"
  )
    message(
      FATAL_ERROR
      "${candidate_name} must resolve inside AVENGINE_RLR_SDK_ROOT: "
      "${_avengine_rlr_sdk_candidate}"
    )
  endif()
  set("${result_variable}" "${_avengine_rlr_sdk_candidate}" PARENT_SCOPE)
endfunction()

_avengine_rlr_sdk_require_outside_git_checkout(
  "${_avengine_rlr_sdk_root}"
)

set(
  _avengine_rlr_sdk_header
  "${_avengine_rlr_sdk_root}/headers/RLRAudioPropagation.h"
)
set(
  _avengine_rlr_sdk_library
  "${_avengine_rlr_sdk_root}/libs/linux/x64/libRLRAudioPropagation.so"
)
if(NOT EXISTS "${_avengine_rlr_sdk_header}")
  message(
    FATAL_ERROR
    "AVENGINE_RLR_SDK_ROOT is missing headers/RLRAudioPropagation.h: "
    "${_avengine_rlr_sdk_header}"
  )
endif()
if(NOT EXISTS "${_avengine_rlr_sdk_library}")
  message(
    FATAL_ERROR
    "AVENGINE_RLR_SDK_ROOT is missing libs/linux/x64/libRLRAudioPropagation.so: "
    "${_avengine_rlr_sdk_library}"
  )
endif()

_avengine_rlr_sdk_require_within_root(
  "${_avengine_rlr_sdk_header}"
  "AVENGINE_RLR_SDK_ROOT headers/RLRAudioPropagation.h"
  _avengine_rlr_sdk_header
)
_avengine_rlr_sdk_require_within_root(
  "${_avengine_rlr_sdk_library}"
  "AVENGINE_RLR_SDK_ROOT libs/linux/x64/libRLRAudioPropagation.so"
  _avengine_rlr_sdk_library
)
get_filename_component(
  _avengine_rlr_sdk_include_dir
  "${_avengine_rlr_sdk_header}"
  DIRECTORY
)

add_library(avengine_rlr_sdk SHARED IMPORTED GLOBAL)
add_library(AVEngine::RlrSdk ALIAS avengine_rlr_sdk)
set_target_properties(
  avengine_rlr_sdk
  PROPERTIES
    IMPORTED_LOCATION "${_avengine_rlr_sdk_library}"
    INTERFACE_INCLUDE_DIRECTORIES "${_avengine_rlr_sdk_include_dir}"
)
