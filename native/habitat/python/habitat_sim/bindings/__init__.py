#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# TODO: this whole thing needs to get removed, kept just for compatibility
#   with existing code

from habitat_sim._ext import habitat_sim_bindings as _habitat_sim_bindings
from habitat_sim._ext.habitat_sim_bindings import (
    AudioSensor,
    AudioSensorSpec,
    CameraSensor,
    CameraSensorSpec,
    Configuration,
    ConfigValType,
    CubeMapSensorBase,
    CubeMapSensorBaseSpec,
    EquirectangularSensor,
    EquirectangularSensorSpec,
    FisheyeSensor,
    FisheyeSensorDoubleSphereSpec,
    FisheyeSensorModelType,
    FisheyeSensorSpec,
    GreedyFollowerCodes,
    GreedyGeodesicFollowerImpl,
    MultiGoalShortestPath,
    PathFinder,
    ReplayRenderer,
    ReplayRendererConfiguration,
    RigidState,
    RLRAudioPropagationChannelLayout,
    RLRAudioPropagationChannelLayoutType,
    RLRAudioPropagationConfiguration,
    SceneGraph,
    SceneNode,
    SceneNodeType,
    Sensor,
    SensorFactory,
    SensorSpec,
    SensorSubType,
    SensorType,
    ShortestPath,
)
from habitat_sim._ext.habitat_sim_bindings import Simulator as SimulatorBackend
from habitat_sim._ext.habitat_sim_bindings import (
    SimulatorConfiguration,
    VisualSensorSpec,
    audio_enabled,
    built_with_bullet,
    cuda_enabled,
    stage_id,
)

_RLR_ADAPTER_SYMBOLS = (
    "RLRAcousticContext",
    "RLRChannelLayoutType",
    "RLRContextConfiguration",
    "RLRListenerReceipt",
    "RLRMaterialUploadReceipt",
    "RLROwnedIR",
    "RLRRayResult",
    "RLRSceneReadbackReport",
    "RLRSceneUploadReport",
    "RLRSourceReceipt",
)

# Do not catch extension-import errors here. A loaded extension has the adapter
# only when every modern RLR symbol is a concrete binding rather than its
# default-off ``None`` compatibility stub.
RLR_ADAPTER_ENABLED = all(
    getattr(_habitat_sim_bindings, symbol, None) is not None
    for symbol in _RLR_ADAPTER_SYMBOLS
)

if RLR_ADAPTER_ENABLED:
    RLRAcousticContext = _habitat_sim_bindings.RLRAcousticContext
    RLRChannelLayoutType = _habitat_sim_bindings.RLRChannelLayoutType
    RLRContextConfiguration = _habitat_sim_bindings.RLRContextConfiguration
    RLRListenerReceipt = _habitat_sim_bindings.RLRListenerReceipt
    RLRMaterialUploadReceipt = _habitat_sim_bindings.RLRMaterialUploadReceipt
    RLROwnedIR = _habitat_sim_bindings.RLROwnedIR
    RLRRayResult = _habitat_sim_bindings.RLRRayResult
    RLRSceneReadbackReport = _habitat_sim_bindings.RLRSceneReadbackReport
    RLRSceneUploadReport = _habitat_sim_bindings.RLRSceneUploadReport
    RLRSourceReceipt = _habitat_sim_bindings.RLRSourceReceipt
else:
    RLRAcousticContext = None
    RLRChannelLayoutType = None
    RLRContextConfiguration = None
    RLRListenerReceipt = None
    RLRMaterialUploadReceipt = None
    RLROwnedIR = None
    RLRRayResult = None
    RLRSceneReadbackReport = None
    RLRSceneUploadReport = None
    RLRSourceReceipt = None
