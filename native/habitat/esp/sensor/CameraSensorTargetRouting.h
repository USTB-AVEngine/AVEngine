// Copyright (c) Meta Platforms, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "esp/sensor/VisualSensor.h"

namespace esp::sensor {

enum class CameraSemanticRenderRoute {
  SemanticSceneThenActiveObjects,
  FullActiveScene,
};

struct CameraSemanticRenderPolicy {
  CameraSemanticRenderRoute route;
  bool drawSemanticSceneGraph;
  bool drawActiveSceneGraph;
  bool activeSceneObjectsOnly;
};

constexpr CameraSemanticRenderPolicy cameraSemanticRenderPolicy(
    SemanticSensorTarget target) {
  if (target == SemanticSensorTarget::SemanticID) {
    return {
        CameraSemanticRenderRoute::SemanticSceneThenActiveObjects,
        true,
        true,
        true,
    };
  }
  return {
      CameraSemanticRenderRoute::FullActiveScene,
      false,
      true,
      false,
  };
}

static_assert(
    cameraSemanticRenderPolicy(SemanticSensorTarget::SemanticID)
        .activeSceneObjectsOnly);
static_assert(
    !cameraSemanticRenderPolicy(SemanticSensorTarget::ObjectID)
         .activeSceneObjectsOnly);
static_assert(
    !cameraSemanticRenderPolicy(SemanticSensorTarget::DrawableID)
         .activeSceneObjectsOnly);

}  // namespace esp::sensor
