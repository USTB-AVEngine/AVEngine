# AVEngine-owned source selection for the optional modern RLR C API adapter.
# The RLR engine, headers, material data, and shared library remain external.

set(
  AVENGINE_HABITAT_RLR_AUDIO_SOURCES
  esp/audio/RLRAcousticContext.cpp
  esp/audio/RLRAcousticContext.h
  esp/audio/RLRAcousticContextTestAccess.h
)
