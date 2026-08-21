# Explicit non-binding Habitat core source closure. Keep this list source-only:
# 112 C++ translation units across core (6), geo (3), gfx (27), assets (7),
# metadata (32), io (5), scene (8), physics (7), nav (2), sensor (8), and
# sim (7). Audio, bindings, BackgroundRenderer, CUDA noise, and PBR image
# resources are intentionally not part of this static compilation slice.
# The seven existing Bullet translation units are appended only by its opt-in.
set(
  AVENGINE_HABITAT_CORE_SOURCES
  esp/core/Buffer.cpp
  esp/core/Check.cpp
  esp/core/Configuration.cpp
  esp/core/Esp.cpp
  esp/core/Logging.cpp
  esp/core/managedContainers/ManagedContainerBase.cpp

  esp/geo/CoordinateFrame.cpp
  esp/geo/Geo.cpp
  esp/geo/OBB.cpp

  esp/gfx/CubeMap.cpp
  esp/gfx/Drawable.cpp
  esp/gfx/DrawableConfiguration.cpp
  esp/gfx/DrawableGroup.cpp
  esp/gfx/GenericDrawable.cpp
  esp/gfx/MeshVisualizerDrawable.cpp
  esp/gfx/LightSetup.cpp
  esp/gfx/RenderCamera.cpp
  esp/gfx/CubeMapCamera.cpp
  esp/gfx/DebugLineRender.cpp
  esp/gfx/Renderer.cpp
  esp/gfx/replay/Player.cpp
  esp/gfx/replay/Recorder.cpp
  esp/gfx/replay/ReplayManager.cpp
  esp/gfx/WindowlessContext.cpp
  esp/gfx/RenderTarget.cpp
  esp/gfx/ShaderManager.cpp
  esp/gfx/PbrShader.cpp
  esp/gfx/PbrDrawable.cpp
  esp/gfx/TextureVisualizerShader.cpp
  esp/gfx/CubeMapShaderBase.cpp
  esp/gfx/DoubleSphereCameraShader.cpp
  esp/gfx/EquirectangularShader.cpp
  esp/gfx/PbrIBLHelper.cpp
  esp/gfx/PbrEquiRectangularToCubeMapShader.cpp
  esp/gfx/PbrPrecomputedMapShader.cpp
  esp/gfx/GaussianFilterShader.cpp

  esp/assets/Asset.cpp
  esp/assets/BaseMesh.cpp
  esp/assets/GenericSemanticMeshData.cpp
  esp/assets/GenericMeshData.cpp
  esp/assets/RenderAssetInstanceCreationInfo.cpp
  esp/assets/ResourceManager.cpp
  esp/assets/RigManager.cpp

  esp/metadata/MetadataMediator.cpp
  esp/metadata/URDFParser.cpp
  esp/metadata/attributes/AbstractAttributes.cpp
  esp/metadata/attributes/AbstractObjectAttributes.cpp
  esp/metadata/attributes/AbstractSensorAttributes.cpp
  esp/metadata/attributes/AbstractVisualSensorAttributes.cpp
  esp/metadata/attributes/ArticulatedObjectAttributes.cpp
  esp/metadata/attributes/AttributesEnumMaps.cpp
  esp/metadata/attributes/AudioSensorAttributes.cpp
  esp/metadata/attributes/CameraSensorAttributes.cpp
  esp/metadata/attributes/CubeMapSensorAttributes.cpp
  esp/metadata/attributes/CustomSensorAttributes.cpp
  esp/metadata/attributes/LightLayoutAttributes.cpp
  esp/metadata/attributes/ObjectAttributes.cpp
  esp/metadata/attributes/PbrShaderAttributes.cpp
  esp/metadata/attributes/PhysicsManagerAttributes.cpp
  esp/metadata/attributes/PrimitiveAssetAttributes.cpp
  esp/metadata/attributes/SceneDatasetAttributes.cpp
  esp/metadata/attributes/SceneInstanceAttributes.cpp
  esp/metadata/attributes/SemanticAttributes.cpp
  esp/metadata/attributes/StageAttributes.cpp
  esp/metadata/managers/AOAttributesManager.cpp
  esp/metadata/managers/AssetAttributesManager.cpp
  esp/metadata/managers/LightLayoutAttributesManager.cpp
  esp/metadata/managers/ObjectAttributesManager.cpp
  esp/metadata/managers/PbrShaderAttributesManager.cpp
  esp/metadata/managers/PhysicsAttributesManager.cpp
  esp/metadata/managers/SceneDatasetAttributesManager.cpp
  esp/metadata/managers/SceneInstanceAttributesManager.cpp
  esp/metadata/managers/SemanticAttributesManager.cpp
  esp/metadata/managers/SensorAttributesManager.cpp
  esp/metadata/managers/StageAttributesManager.cpp

  esp/io/Io.cpp
  esp/io/Json.cpp
  esp/io/JsonEspTypes.cpp
  esp/io/JsonMagnumTypes.cpp
  esp/io/JsonStlTypes.cpp

  esp/scene/GibsonSemanticScene.cpp
  esp/scene/HM3DSemanticScene.cpp
  esp/scene/Mp3dSemanticScene.cpp
  esp/scene/ReplicaSemanticScene.cpp
  esp/scene/SceneGraph.cpp
  esp/scene/SceneManager.cpp
  esp/scene/SceneNode.cpp
  esp/scene/SemanticScene.cpp

  esp/physics/CollisionGroupHelper.cpp
  esp/physics/objectManagers/ArticulatedObjectManager.cpp
  esp/physics/objectManagers/RigidObjectManager.cpp
  esp/physics/PhysicsManager.cpp
  esp/physics/RigidObject.cpp
  esp/physics/RigidStage.cpp
  esp/physics/URDFImporter.cpp

  esp/nav/GreedyFollower.cpp
  esp/nav/PathFinder.cpp

  esp/sensor/CameraSensor.cpp
  esp/sensor/CubeMapSensorBase.cpp
  esp/sensor/Sensor.cpp
  esp/sensor/SensorFactory.cpp
  esp/sensor/VisualSensor.cpp
  esp/sensor/FisheyeSensor.cpp
  esp/sensor/EquirectangularSensor.cpp
  esp/sensor/AudioSensor.cpp

  esp/sim/AbstractReplayRenderer.cpp
  esp/sim/BatchPlayerImplementation.cpp
  esp/sim/BatchReplayRenderer.cpp
  esp/sim/ClassicReplayRenderer.cpp
  esp/sim/Simulator.cpp
  esp/sim/SimulatorConfiguration.cpp
  esp/sim/RenderInstanceHelper.cpp
)

if(AVENGINE_HABITAT_BUILD_BULLET)
  list(
    APPEND
    AVENGINE_HABITAT_CORE_SOURCES
    esp/physics/bullet/BulletArticulatedObject.cpp
    esp/physics/bullet/BulletBase.cpp
    esp/physics/bullet/BulletCollisionHelper.cpp
    esp/physics/bullet/BulletPhysicsManager.cpp
    esp/physics/bullet/BulletRigidObject.cpp
    esp/physics/bullet/BulletRigidStage.cpp
    esp/physics/bullet/BulletURDFImporter.cpp
  )
endif()
