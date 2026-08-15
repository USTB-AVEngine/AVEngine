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
  ABSOLUTE
)
if(NOT IS_DIRECTORY "${_avengine_rlr_sdk_root}")
  message(
    FATAL_ERROR
    "AVENGINE_RLR_SDK_ROOT is not a directory: ${_avengine_rlr_sdk_root}"
  )
endif()

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

add_library(avengine_rlr_sdk SHARED IMPORTED GLOBAL)
add_library(AVEngine::RlrSdk ALIAS avengine_rlr_sdk)
set_target_properties(
  avengine_rlr_sdk
  PROPERTIES
    IMPORTED_LOCATION "${_avengine_rlr_sdk_library}"
    INTERFACE_INCLUDE_DIRECTORIES "${_avengine_rlr_sdk_root}/headers"
)
