// Copyright (c) Meta Platforms, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <Corrade/TestSuite/Tester.h>

#include "esp/sensor/CameraSensorTargetRouting.h"

namespace esp::sensor::test {

class CameraSensorTargetRoutingTest : public Corrade::TestSuite::Tester {
 public:
  explicit CameraSensorTargetRoutingTest() {
    addTests({
        &CameraSensorTargetRoutingTest::semanticIdKeepsDualGraphRoute,
        &CameraSensorTargetRoutingTest::objectIdUsesFullActiveScene,
        &CameraSensorTargetRoutingTest::drawableIdUsesFullActiveScene,
    });
  }

 private:
  void semanticIdKeepsDualGraphRoute();
  void objectIdUsesFullActiveScene();
  void drawableIdUsesFullActiveScene();
};

void CameraSensorTargetRoutingTest::semanticIdKeepsDualGraphRoute() {
  constexpr auto policy =
      cameraSemanticRenderPolicy(SemanticSensorTarget::SemanticID);
  CORRADE_VERIFY(
      policy.route ==
      CameraSemanticRenderRoute::SemanticSceneThenActiveObjects);
  CORRADE_VERIFY(policy.drawSemanticSceneGraph);
  CORRADE_VERIFY(policy.drawActiveSceneGraph);
  CORRADE_VERIFY(policy.activeSceneObjectsOnly);
}

void CameraSensorTargetRoutingTest::objectIdUsesFullActiveScene() {
  constexpr auto policy =
      cameraSemanticRenderPolicy(SemanticSensorTarget::ObjectID);
  CORRADE_VERIFY(
      policy.route == CameraSemanticRenderRoute::FullActiveScene);
  CORRADE_VERIFY(!policy.drawSemanticSceneGraph);
  CORRADE_VERIFY(policy.drawActiveSceneGraph);
  CORRADE_VERIFY(!policy.activeSceneObjectsOnly);
}

void CameraSensorTargetRoutingTest::drawableIdUsesFullActiveScene() {
  constexpr auto policy =
      cameraSemanticRenderPolicy(SemanticSensorTarget::DrawableID);
  CORRADE_VERIFY(
      policy.route == CameraSemanticRenderRoute::FullActiveScene);
  CORRADE_VERIFY(!policy.drawSemanticSceneGraph);
  CORRADE_VERIFY(policy.drawActiveSceneGraph);
  CORRADE_VERIFY(!policy.activeSceneObjectsOnly);
}

}  // namespace esp::sensor::test

CORRADE_TEST_MAIN(esp::sensor::test::CameraSensorTargetRoutingTest)
