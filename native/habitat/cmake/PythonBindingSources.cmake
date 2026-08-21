# Exact selected binding translation units. Keep this list source-only: package
# staging and module output are controlled by AVENGINE_HABITAT_PYTHON_OUTPUT_DIR.
set(
  AVENGINE_HABITAT_PYTHON_BINDING_SOURCES
  esp/bindings/Bindings.cpp
  esp/bindings/AudioPropagationBindings.cpp
  esp/bindings/AttributesBindings.cpp
  esp/bindings/AttributesManagersBindings.cpp
  esp/bindings/ConfigBindings.cpp
  esp/bindings/CoreBindings.cpp
  esp/bindings/GeoBindings.cpp
  esp/bindings/GfxBindings.cpp
  esp/bindings/MetadataMediatorBindings.cpp
  esp/bindings/GfxReplayBindings.cpp
  esp/bindings/PhysicsBindings.cpp
  esp/bindings/PhysicsObjectBindings.cpp
  esp/bindings/PhysicsWrapperManagerBindings.cpp
  esp/bindings/SceneBindings.cpp
  esp/bindings/SensorBindings.cpp
  esp/bindings/ShortestPathBindings.cpp
  esp/bindings/SimBindings.cpp
)
