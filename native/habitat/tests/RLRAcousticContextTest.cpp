// Copyright (c) Meta Platforms, Inc. and its affiliates.
// Modifications Copyright (c) AVEngine contributors.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <Corrade/TestSuite/Tester.h>
#include <Corrade/Utility/DebugStl.h>
#include <Corrade/Utility/Path.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "configure.h"
#include "esp/audio/RLRAcousticContext.h"

namespace Cr = Corrade;
using esp::audio::RLRAcousticContext;
using esp::audio::RLRChannelLayoutType;
using esp::audio::RLRContextConfiguration;
using esp::audio::RLRMeshObject;
using esp::audio::RLROwnedIR;

namespace {

const std::vector<std::string> kCategories{
    "avengine_m3_hard_surface_7f12",
    "avengine_m3_absorptive_surface_b946",
};

std::string materialDatabasePath() {
  return Cr::Utility::Path::join(TEST_ASSETS,
                                 "audio/avengine_m3_materials.json");
}

RLRMeshObject makeCube() {
  RLRMeshObject object;
  object.objectId = "controlled_room";
  object.vertices = {
      -2.0f, -2.0f, -2.0f, 2.0f,  -2.0f, -2.0f,
      2.0f,  2.0f,  -2.0f, -2.0f, 2.0f,  -2.0f,
      -2.0f, -2.0f, 2.0f,  2.0f,  -2.0f, 2.0f,
      2.0f,  2.0f,  2.0f,  -2.0f, 2.0f,  2.0f,
  };
  object.triangles = {
      0, 1, 5, 0, 5, 4,  // floor
      4, 5, 6, 4, 6, 7,  // ceiling
      0, 4, 7, 0, 7, 3,  // -x
      1, 2, 6, 1, 6, 5,  // +x
      3, 7, 6, 3, 6, 2,  // +y
      0, 3, 2, 0, 2, 1,  // -z
  };
  object.triangleMaterialIds = {1, 1, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0};
  return object;
}

RLRContextConfiguration quickConfiguration() {
  RLRContextConfiguration configuration;
  configuration.threadCount = 1;
  configuration.directRayCount = 32;
  configuration.indirectRayCount = 256;
  configuration.indirectRayDepth = 8;
  configuration.sourceRayCount = 32;
  configuration.sourceRayDepth = 4;
  configuration.maxDiffractionOrder = 0;
  configuration.diffraction = false;
  configuration.transmission = false;
  configuration.maxIRLength = 0.25f;
  return configuration;
}

RLRContextConfiguration directOnlyConfiguration() {
  RLRContextConfiguration configuration = quickConfiguration();
  configuration.indirect = false;
  configuration.temporalCoherence = false;
  return configuration;
}

const RLROwnedIR& findIR(const std::vector<RLROwnedIR>& responses,
                         const std::string& sourceId) {
  const auto iterator =
      std::find_if(responses.begin(), responses.end(),
                   [&sourceId](const RLROwnedIR& response) {
                     return response.sourceId == sourceId;
                   });
  if (iterator == responses.end()) {
    throw std::runtime_error("test could not find source IR: " + sourceId);
  }
  return *iterator;
}

std::vector<RLROwnedIR> simulateTwoSourceFOA(const bool reverseAddOrder) {
  RLRAcousticContext context{quickConfiguration()};
  context.loadAcousticScene(materialDatabasePath(), kCategories, {makeCube()});
  if (reverseAddOrder) {
    context.addSource("source_b", {{-0.65f, 0.15f, 0.2f}});
    context.addSource("source_a", {{0.75f, -0.1f, -0.15f}});
  } else {
    context.addSource("source_a", {{0.75f, -0.1f, -0.15f}});
    context.addSource("source_b", {{-0.65f, 0.15f, 0.2f}});
  }
  context.addListener("listener", {{0.0f, 0.0f, 0.0f}},
                      {{1.0f, 0.0f, 0.0f, 0.0f}},
                      RLRChannelLayoutType::Ambisonics, 4);
  return context.simulateOwned();
}

std::size_t countOccurrences(const std::string& text,
                             const std::string& needle) {
  std::size_t count = 0;
  for (std::size_t offset = text.find(needle); offset != std::string::npos;
       offset = text.find(needle, offset + needle.size())) {
    ++count;
  }
  return count;
}

struct RLRAcousticContextTest : Cr::TestSuite::Tester {
  explicit RLRAcousticContextTest();

  void strictUploadProbeAndReadback();
  void transformedReadbackEvidence();
  void multiObjectReadbackBlocks();
  void strictConfigurationAndMaterialDatabase();
  void strictMaterialLabels();
  void multiSourceFOAAndCanonicalReceipts();
  void fullIndirectSourceOrderInvariant();
  void endpointUpdatesPreserveIdentity();
  void resetRestoresFreshSimulationSequence();
  void foaAxisConvention();
  void concurrentResetAndNativeOperations();
  void rejectUnknownMaterial();
  void rejectIncompleteTriangleCoverage();
};

RLRAcousticContextTest::RLRAcousticContextTest() {
  addTests({&RLRAcousticContextTest::strictUploadProbeAndReadback,
            &RLRAcousticContextTest::transformedReadbackEvidence,
            &RLRAcousticContextTest::multiObjectReadbackBlocks,
            &RLRAcousticContextTest::strictConfigurationAndMaterialDatabase,
            &RLRAcousticContextTest::strictMaterialLabels,
            &RLRAcousticContextTest::multiSourceFOAAndCanonicalReceipts,
            &RLRAcousticContextTest::fullIndirectSourceOrderInvariant,
            &RLRAcousticContextTest::endpointUpdatesPreserveIdentity,
            &RLRAcousticContextTest::resetRestoresFreshSimulationSequence,
            &RLRAcousticContextTest::foaAxisConvention,
            &RLRAcousticContextTest::concurrentResetAndNativeOperations,
            &RLRAcousticContextTest::rejectUnknownMaterial,
            &RLRAcousticContextTest::rejectIncompleteTriangleCoverage});
}

void RLRAcousticContextTest::strictUploadProbeAndReadback() {
  RLRAcousticContext context{quickConfiguration()};
  const auto report = context.loadAcousticScene(
      materialDatabasePath(), kCategories, {makeCube()});
  CORRADE_COMPARE(report.objectCount, 1);
  CORRADE_COMPARE(report.vertexCount, 8);
  CORRADE_COMPARE(report.triangleCount, 12);
  CORRADE_COMPARE(report.materialCategoryCount, 2);
  CORRADE_COMPARE(report.objectIds, std::vector<std::string>{"controlled_room"});
  CORRADE_COMPARE(report.triangleCountByMaterial.at(kCategories[0]), 10);
  CORRADE_COMPARE(report.triangleCountByMaterial.at(kCategories[1]), 2);
  CORRADE_COMPARE(report.materialUploadCallCount.at(kCategories[0]), 1);
  CORRADE_COMPARE(report.materialUploadCallCount.at(kCategories[1]), 1);
  CORRADE_COMPARE(report.resolvedMaterialNameByCategory.at(kCategories[0]),
                  "AVEngine M3 hard surface");
  CORRADE_COMPARE(report.resolvedMaterialNameByCategory.at(kCategories[1]),
                  "AVEngine M3 absorptive surface");
  CORRADE_COMPARE(report.resolvedMaterialIndexByCategory.at(kCategories[0]),
                  0);
  CORRADE_COMPARE(report.resolvedMaterialIndexByCategory.at(kCategories[1]),
                  1);
  CORRADE_COMPARE(report.expectedMaterialBlockCount, 2);
  CORRADE_VERIFY(report.expectedCanonicalByteCount > 0);
  CORRADE_VERIFY(report.expectedMaterialCoefficientByteCount > 0);
  CORRADE_COMPARE(report.materialUploadReceipts.size(), 2);
  CORRADE_COMPARE(report.materialUploadReceipts[0].materialCategory,
                  kCategories[0]);
  CORRADE_COMPARE(report.materialUploadReceipts[0].triangleCount, 10);
  CORRADE_COMPARE(report.materialUploadReceipts[0].indexCount, 30);
  CORRADE_COMPARE(report.materialUploadReceipts[1].materialCategory,
                  kCategories[1]);
  CORRADE_COMPARE(report.materialUploadReceipts[1].triangleCount, 2);
  CORRADE_COMPARE(report.materialUploadReceipts[1].indexCount, 6);
  CORRADE_COMPARE(context.objectCount(), 1);

  const std::string debugPath = Cr::Utility::Path::join(
      MAGNUMRENDERERTEST_OUTPUT_DIR, "rlr_acoustic_context_scene.obj");
  const auto readback = context.writeSceneMeshOBJ(debugPath);
  CORRADE_VERIFY(readback.verificationPassed);
  CORRADE_VERIFY(readback.worldGeometryMatches);
  CORRADE_VERIFY(readback.materialEvidenceMatches);
  CORRADE_VERIFY(!readback.perFaceMaterialIdsAvailable);
  CORRADE_COMPARE(readback.materialEvidenceMode,
                  "exact_upload_receipt_plus_resolved_coefficient_blocks");
  CORRADE_COMPARE(readback.vertexCount, 8);
  CORRADE_COMPARE(readback.triangleCount, 12);
  CORRADE_COMPARE(readback.materialBlockCount, 2);
  CORRADE_COMPARE(readback.materialAssignmentCount, 12);
  CORRADE_COMPARE(readback.canonicalByteCount,
                  readback.expectedCanonicalByteCount);
  CORRADE_VERIFY(readback.materialCoefficientsMatch);
  CORRADE_COMPARE(readback.materialCoefficientByteCount,
                  readback.expectedMaterialCoefficientByteCount);
  std::ifstream debugStream{debugPath};
  const std::string debugText{std::istreambuf_iterator<char>{debugStream}, {}};
  CORRADE_VERIFY(debugText.find("# Vertex Count: 8") != std::string::npos);
  CORRADE_VERIFY(debugText.find("# Triangle Count: 12") != std::string::npos);
  CORRADE_COMPARE(countOccurrences(debugText, "# Material Index :"), 2);
  CORRADE_COMPARE(countOccurrences(debugText, "# Vertex Count:"), 1);
  const std::size_t materialZero =
      debugText.find("# Material Index : 0");
  const std::size_t materialOne = debugText.find("# Material Index : 1");
  const std::size_t geometryHeader = debugText.find("# Vertex Count: 8");
  CORRADE_VERIFY(materialZero < materialOne && materialOne < geometryHeader);

  CORRADE_COMPARE(context.addSource("source0", {{0.0f, 0.0f, 0.0f}}), 0);
  CORRADE_COMPARE(context.addListener(
                      "listener0", {{1.0f, 0.0f, 0.0f}},
                      {{1.0f, 0.0f, 0.0f, 0.0f}},
                      RLRChannelLayoutType::Mono, 1),
                  0);
  const auto impulseResponses = context.simulateOwned();
  CORRADE_COMPARE(impulseResponses.size(), 1);
  const auto& impulseResponse = impulseResponses.front();
  CORRADE_COMPARE(impulseResponse.listenerId, "listener0");
  CORRADE_COMPARE(impulseResponse.sourceId, "source0");
  CORRADE_COMPARE(impulseResponse.channelCount, 1);
  CORRADE_COMPARE(impulseResponse.normalization, "not_applicable");
  CORRADE_COMPARE(impulseResponse.coordinateFrame, "not_directional");
  CORRADE_COMPARE(impulseResponse.layoutId, "rlr_mono_v1");
  CORRADE_COMPARE(impulseResponse.channelLabels,
                  (std::vector<std::string>{"mono"}));
  CORRADE_VERIFY(impulseResponse.sampleCount > 0);
  CORRADE_COMPARE(impulseResponse.samples.size(),
                  impulseResponse.sampleCount);
  CORRADE_VERIFY(std::all_of(impulseResponse.samples.begin(),
                             impulseResponse.samples.end(),
                             [](const float value) {
                               return std::isfinite(value);
                             }));
  CORRADE_VERIFY(std::any_of(impulseResponse.samples.begin(),
                             impulseResponse.samples.end(),
                             [](const float value) { return value != 0.0f; }));

  const auto anyHit = context.traceRayAnyHit(
      {{0.0f, 0.0f, 0.0f}}, {{1.0f, 0.0f, 0.0f}}, 0.01f, 10.0f);
  CORRADE_VERIFY(anyHit.hit);
  CORRADE_VERIFY(!anyHit.hasHitDetails);
  const auto firstHit = context.traceRayFirstHit(
      {{0.0f, 0.0f, 0.0f}}, {{1.0f, 0.0f, 0.0f}}, 0.01f, 10.0f);
  CORRADE_VERIFY(firstHit.hit);
  CORRADE_VERIFY(firstHit.hasHitDetails);
  CORRADE_VERIFY(std::abs(firstHit.distance - 2.0f) < 1.0e-3f);
  CORRADE_VERIFY(std::isfinite(firstHit.distance));
  CORRADE_VERIFY(std::all_of(firstHit.normal.begin(), firstHit.normal.end(),
                             [](const float value) {
                               return std::isfinite(value);
                             }));
  CORRADE_VERIFY(std::any_of(firstHit.normal.begin(), firstHit.normal.end(),
                             [](const float value) {
                               return value != 0.0f;
                             }));
  const auto miss = context.traceRayFirstHit(
      {{0.0f, 0.0f, 0.0f}}, {{1.0f, 0.0f, 0.0f}}, 0.01f, 0.5f);
  CORRADE_VERIFY(!miss.hit);
  CORRADE_VERIFY(!miss.hasHitDetails);
  CORRADE_VERIFY(std::isfinite(miss.distance));
  CORRADE_COMPARE(miss.distance, 0.0f);
  CORRADE_VERIFY(std::all_of(miss.normal.begin(), miss.normal.end(),
                             [](const float value) {
                               return std::isfinite(value);
                             }));
  const float efficiency = context.indirectRayEfficiency();
  CORRADE_VERIFY(efficiency >= 0.0f && efficiency <= 1.0f);

  const std::string metricsPrefix = Cr::Utility::Path::join(
      MAGNUMRENDERERTEST_OUTPUT_DIR, "rlr_acoustic_context_metrics.txt");
  const auto metricFiles =
      context.writeIRMetrics("listener0", "source0", metricsPrefix);
  CORRADE_COMPARE(metricFiles.size(), 1);
  CORRADE_VERIFY(Cr::Utility::Path::exists(metricFiles.front()));

  Cr::Utility::Path::remove(debugPath);
  Cr::Utility::Path::remove(metricFiles.front());
  Cr::Utility::Path::remove(metricsPrefix + " ch0 etc.txt");
  context.reset();
  CORRADE_VERIFY(!context.sceneLoaded());
  CORRADE_COMPARE(context.objectCount(), 0);
  CORRADE_COMPARE(context.sourceCount(), 0);
  CORRADE_COMPARE(context.listenerCount(), 0);
}

void RLRAcousticContextTest::transformedReadbackEvidence() {
  RLRAcousticContext context{quickConfiguration()};
  RLRMeshObject object = makeCube();
  object.position = {{1.0f, 2.0f, 3.0f}};
  const float halfSqrt = std::sqrt(0.5f);
  object.orientationWxyz = {{halfSqrt, 0.0f, halfSqrt, 0.0f}};
  const auto upload = context.loadAcousticScene(
      materialDatabasePath(), kCategories, {object});
  CORRADE_COMPARE(upload.objectCount, 1);
  CORRADE_COMPARE(upload.vertexCount, 8);
  CORRADE_COMPARE(upload.triangleCount, 12);
  CORRADE_COMPARE(upload.expectedMaterialBlockCount, 2);

  const std::string outputPath = Cr::Utility::Path::join(
      MAGNUMRENDERERTEST_OUTPUT_DIR, "rlr_transformed_scene.obj");
  const auto readback = context.writeSceneMeshOBJ(outputPath);
  CORRADE_VERIFY(readback.verificationPassed);
  CORRADE_VERIFY(readback.worldGeometryMatches);
  CORRADE_VERIFY(readback.materialEvidenceMatches);
  CORRADE_COMPARE(readback.vertexCount, upload.vertexCount);
  CORRADE_COMPARE(readback.triangleCount, upload.triangleCount);
  CORRADE_COMPARE(readback.materialBlockCount,
                  upload.expectedMaterialBlockCount);
  CORRADE_COMPARE(readback.materialAssignmentCount, upload.triangleCount);
  CORRADE_COMPARE(readback.canonicalByteCount,
                  readback.expectedCanonicalByteCount);
  Cr::Utility::Path::remove(outputPath);
}

void RLRAcousticContextTest::multiObjectReadbackBlocks() {
  RLRAcousticContext context{quickConfiguration()};
  RLRMeshObject hardObject = makeCube();
  hardObject.objectId = "hard_room_object";
  hardObject.position = {{-5.0f, 0.0f, 0.0f}};
  std::fill(hardObject.triangleMaterialIds.begin(),
            hardObject.triangleMaterialIds.end(), 0);
  RLRMeshObject absorptiveObject = makeCube();
  absorptiveObject.objectId = "absorptive_room_object";
  absorptiveObject.position = {{5.0f, 0.0f, 0.0f}};
  std::fill(absorptiveObject.triangleMaterialIds.begin(),
            absorptiveObject.triangleMaterialIds.end(), 1);

  const auto upload = context.loadAcousticScene(
      materialDatabasePath(), kCategories, {hardObject, absorptiveObject});
  CORRADE_COMPARE(upload.materialUploadReceipts.size(), 2);
  CORRADE_COMPARE(upload.expectedMaterialBlockCount, 2);

  const std::string outputPath = Cr::Utility::Path::join(
      MAGNUMRENDERERTEST_OUTPUT_DIR, "rlr_multi_object_scene.obj");
  const auto readback = context.writeSceneMeshOBJ(outputPath);
  CORRADE_VERIFY(readback.verificationPassed);
  CORRADE_COMPARE(readback.vertexCount, 16);
  CORRADE_COMPARE(readback.triangleCount, 24);
  CORRADE_COMPARE(readback.materialBlockCount, 2);
  CORRADE_COMPARE(readback.materialAssignmentCount, 24);
  CORRADE_COMPARE(readback.materialBlockCount,
                  upload.materialUploadReceipts.size());
  std::ifstream debugStream{outputPath};
  const std::string debugText{std::istreambuf_iterator<char>{debugStream}, {}};
  CORRADE_COMPARE(countOccurrences(debugText, "# Material Index : 0"), 2);
  CORRADE_COMPARE(countOccurrences(debugText, "# Vertex Count: 8"), 2);
  CORRADE_COMPARE(countOccurrences(debugText, "# Triangle Count: 12"), 2);
  Cr::Utility::Path::remove(outputPath);
}

void RLRAcousticContextTest::strictConfigurationAndMaterialDatabase() {
  RLRContextConfiguration invalid = quickConfiguration();
  invalid.frequencyBands = 6;
  bool rejected = false;
  try {
    RLRAcousticContext context{invalid};
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CORRADE_VERIFY(rejected);

  invalid = quickConfiguration();
  invalid.directSHOrder = 6;
  rejected = false;
  try {
    RLRAcousticContext context{invalid};
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CORRADE_VERIFY(rejected);

  invalid = quickConfiguration();
  invalid.indirectRayDepth = 0;
  rejected = false;
  try {
    RLRAcousticContext context{invalid};
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CORRADE_VERIFY(rejected);

  const std::string missingSpeedPath = Cr::Utility::Path::join(
      MAGNUMRENDERERTEST_OUTPUT_DIR, "rlr_material_missing_speed.json");
  {
    std::ofstream output{missingSpeedPath};
    output << R"({"materials":[{"name":"bad","labels":["bad"],"absorption":[20,0,20000,0],"scattering":[20,0,20000,0],"transmission":[20,0,20000,0],"damping":[20,0,20000,0]}]})";
  }
  RLRMeshObject object = makeCube();
  std::fill(object.triangleMaterialIds.begin(),
            object.triangleMaterialIds.end(), 0);
  RLRAcousticContext context{quickConfiguration()};
  rejected = false;
  try {
    context.loadAcousticScene(missingSpeedPath, {"bad"}, {object});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CORRADE_VERIFY(rejected);
  Cr::Utility::Path::remove(missingSpeedPath);

  const std::string missingDampingPath = Cr::Utility::Path::join(
      MAGNUMRENDERERTEST_OUTPUT_DIR, "rlr_material_missing_damping.json");
  {
    std::ofstream output{missingDampingPath};
    output << R"({"materials":[{"name":"bad","labels":["bad"],"absorption":[20,0,20000,0],"scattering":[20,0,20000,0],"transmission":[20,0,20000,0],"speed":343}]})";
  }
  rejected = false;
  try {
    context.loadAcousticScene(missingDampingPath, {"bad"}, {object});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CORRADE_VERIFY(rejected);
  Cr::Utility::Path::remove(missingDampingPath);

  context.loadAcousticScene(materialDatabasePath(), kCategories, {makeCube()});
  rejected = false;
  try {
    context.addListener("bad_binaural", {{0.0f, 0.0f, 0.0f}},
                        {{1.0f, 0.0f, 0.0f, 0.0f}},
                        RLRChannelLayoutType::Binaural, 1);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CORRADE_VERIFY(rejected);
  rejected = false;
  try {
    context.addListener("bad_ambisonics", {{0.0f, 0.0f, 0.0f}},
                        {{1.0f, 0.0f, 0.0f, 0.0f}},
                        RLRChannelLayoutType::Ambisonics, 2);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CORRADE_VERIFY(rejected);

  context.addSource("binaural_source", {{0.5f, 0.0f, 0.0f}});
  context.addListener("binaural_listener", {{0.0f, 0.0f, 0.0f}},
                      {{1.0f, 0.0f, 0.0f, 0.0f}},
                      RLRChannelLayoutType::Binaural, 2);
  const auto binaural = context.simulateOwned();
  CORRADE_COMPARE(binaural.size(), 1);
  CORRADE_COMPARE(binaural[0].normalization, "not_applicable");
  CORRADE_COMPARE(binaural[0].coordinateFrame, "listener_local");
  CORRADE_COMPARE(binaural[0].layoutId, "rlr_binaural_lr_v1");
  CORRADE_COMPARE(binaural[0].channelLabels,
                  (std::vector<std::string>{"left", "right"}));
}

void RLRAcousticContextTest::strictMaterialLabels() {
  const auto material = [](const std::string& name,
                           const std::string& labels) {
    return "{\"name\":\"" + name + "\",\"labels\":" + labels +
           R"(,"absorption":[20,0.1,20000,0.1],"scattering":[20,0.1,20000,0.1],"transmission":[20,0,20000,0],"damping":[20,0,20000,0],"speed":343})";
  };
  const auto database = [&material](const std::string& firstLabels,
                                    const std::string& secondLabels) {
    std::string result = "{\"materials\":[" +
                         material("first", firstLabels);
    if (!secondLabels.empty()) {
      result += "," + material("second", secondLabels);
    }
    return result + "]}";
  };
  const auto rejectedWith = [](const std::string& path,
                               const std::string& contents,
                               const std::string& expectedMessage) {
    {
      std::ofstream output{path};
      output << contents;
    }
    RLRMeshObject object = makeCube();
    std::fill(object.triangleMaterialIds.begin(),
              object.triangleMaterialIds.end(), 0);
    RLRAcousticContext context{quickConfiguration()};
    bool rejected = false;
    try {
      context.loadAcousticScene(path, {"wall"}, {object});
    } catch (const std::invalid_argument& error) {
      rejected = std::string{error.what()}.find(expectedMessage) !=
                 std::string::npos;
    }
    Cr::Utility::Path::remove(path);
    return rejected && context.objectCount() == 0;
  };

  const std::string duplicatePath = Cr::Utility::Path::join(
      MAGNUMRENDERERTEST_OUTPUT_DIR, "rlr_material_duplicate_label.json");
  CORRADE_VERIFY(rejectedWith(
      duplicatePath, database("[\"wall\",\"all\"]", "[\"wall\"]"),
      "globally unique case-insensitively"));

  const std::string caseDuplicatePath = Cr::Utility::Path::join(
      MAGNUMRENDERERTEST_OUTPUT_DIR,
      "rlr_material_case_duplicate_label.json");
  CORRADE_VERIFY(rejectedWith(
      caseDuplicatePath, database("[\"wall\"]", "[\"WALL\"]"),
      "globally unique case-insensitively"));

  const std::string uppercasePath = Cr::Utility::Path::join(
      MAGNUMRENDERERTEST_OUTPUT_DIR, "rlr_material_uppercase_label.json");
  CORRADE_VERIFY(rejectedWith(uppercasePath,
                              database("[\"WALL\"]", ""),
                              "lowercase"));
}

void RLRAcousticContextTest::multiSourceFOAAndCanonicalReceipts() {
  RLRAcousticContext context{quickConfiguration()};
  context.loadAcousticScene(materialDatabasePath(), kCategories, {makeCube()});
  context.addSource("source_b", {{-0.65f, 0.15f, 0.2f}}, 0.02f);
  context.addSource("source_a", {{0.75f, -0.1f, -0.15f}});
  context.addListener("listener", {{0.0f, 0.0f, 0.0f}},
                      {{1.0f, 0.0f, 0.0f, 0.0f}},
                      RLRChannelLayoutType::Ambisonics, 4, 0.1f);

  CORRADE_COMPARE(context.sourceCount(), 2);
  CORRADE_COMPARE(context.listenerCount(), 1);
  CORRADE_COMPARE(context.sourceIndex("source_a"), 0);
  CORRADE_COMPARE(context.sourceIndex("source_b"), 1);
  const auto pendingSources = context.sourceReceipts();
  CORRADE_COMPARE(pendingSources.size(), 2);
  CORRADE_COMPARE(pendingSources[0].sourceId, "source_a");
  CORRADE_COMPARE(pendingSources[0].nativeIndex, 0);
  CORRADE_VERIFY(!pendingSources[0].nativeRealized);
  CORRADE_COMPARE(pendingSources[1].sourceId, "source_b");
  CORRADE_COMPARE(pendingSources[1].nativeIndex, 1);
  CORRADE_VERIFY(!pendingSources[1].nativeRealized);
  const auto pendingListeners = context.listenerReceipts();
  CORRADE_COMPARE(pendingListeners.size(), 1);
  CORRADE_COMPARE(pendingListeners[0].listenerId, "listener");
  CORRADE_VERIFY(!pendingListeners[0].nativeRealized);

  const auto responses = context.simulateOwned();
  CORRADE_COMPARE(responses.size(), 2);
  CORRADE_COMPARE(responses[0].sourceId, "source_a");
  CORRADE_COMPARE(responses[1].sourceId, "source_b");
  for (std::size_t sourceIndex = 0; sourceIndex < responses.size();
       ++sourceIndex) {
    const auto& response = responses[sourceIndex];
    CORRADE_COMPARE(response.listenerId, "listener");
    CORRADE_COMPARE(response.listenerNativeIndex, 0);
    CORRADE_COMPARE(response.sourceNativeIndex, sourceIndex);
    CORRADE_VERIFY(response.layoutType ==
                   RLRChannelLayoutType::Ambisonics);
    CORRADE_COMPARE(response.channelCount, 4);
    CORRADE_COMPARE(response.ambisonicOrder, 1);
    CORRADE_COMPARE(response.channelOrdering, "ACN");
    CORRADE_COMPARE(response.normalization, "N3D");
    CORRADE_COMPARE(response.coordinateFrame, "avengine_world");
    CORRADE_COMPARE(response.layoutId, "rlr_foa_acn_n3d_world_v1");
    CORRADE_COMPARE(response.channelLabels,
                    (std::vector<std::string>{"W", "Y", "Z", "X"}));
    CORRADE_COMPARE(response.samples.size(), 4 * response.sampleCount);
    CORRADE_VERIFY(std::all_of(response.samples.begin(), response.samples.end(),
                               [](const float value) {
                                 return std::isfinite(value);
                               }));
    CORRADE_VERIFY(std::any_of(response.samples.begin(), response.samples.end(),
                               [](const float value) {
                                 return value != 0.0f;
                               }));
  }
  const auto realizedSources = context.sourceReceipts();
  CORRADE_VERIFY(realizedSources[0].nativeRealized);
  CORRADE_VERIFY(realizedSources[1].nativeRealized);
  CORRADE_VERIFY(context.listenerReceipts()[0].nativeRealized);
}

void RLRAcousticContextTest::fullIndirectSourceOrderInvariant() {
  const auto reverseOrder = simulateTwoSourceFOA(true);
  const auto canonicalOrder = simulateTwoSourceFOA(false);
  CORRADE_COMPARE(reverseOrder.size(), canonicalOrder.size());
  for (const std::string sourceId : {"source_a", "source_b"}) {
    const auto& reverse = findIR(reverseOrder, sourceId);
    const auto& canonical = findIR(canonicalOrder, sourceId);
    CORRADE_COMPARE(reverse.sampleCount, canonical.sampleCount);
    CORRADE_COMPARE(reverse.channelCount, canonical.channelCount);
    CORRADE_COMPARE(reverse.samples, canonical.samples);
  }
}

void RLRAcousticContextTest::endpointUpdatesPreserveIdentity() {
  RLRAcousticContext context{directOnlyConfiguration()};
  context.loadAcousticScene(materialDatabasePath(), kCategories, {makeCube()});
  context.addSource("source_b", {{-1.0f, 0.0f, 0.0f}});
  context.addSource("source_a", {{1.0f, 0.0f, 0.0f}});
  context.addListener("listener", {{0.0f, 0.0f, 0.0f}},
                      {{1.0f, 0.0f, 0.0f, 0.0f}},
                      RLRChannelLayoutType::Ambisonics, 4);
  const auto baseline = context.simulateOwned();

  context.setSourcePosition("source_a", {{0.0f, 1.0f, 0.0f}});
  context.setSourceRadius("source_a", 0.025f);
  const auto sourceReceipts = context.sourceReceipts();
  CORRADE_COMPARE(sourceReceipts[0].sourceId, "source_a");
  CORRADE_COMPARE(sourceReceipts[0].nativeIndex, 0);
  CORRADE_VERIFY(sourceReceipts[0].nativeRealized);
  CORRADE_COMPARE(sourceReceipts[0].position,
                  (std::array<float, 3>{{0.0f, 1.0f, 0.0f}}));
  CORRADE_COMPARE(sourceReceipts[0].radius, 0.025f);
  const auto moved = context.simulateOwned();
  CORRADE_VERIFY(findIR(baseline, "source_a").samples !=
                 findIR(moved, "source_a").samples);
  CORRADE_COMPARE(findIR(baseline, "source_b").samples,
                  findIR(moved, "source_b").samples);

  context.setListenerPose("listener", {{0.0f, 0.0f, 0.0f}},
                          {{0.0f, 0.0f, 1.0f, 0.0f}});
  context.setListenerRadius("listener", 0.075f);
  const auto listenerReceipts = context.listenerReceipts();
  CORRADE_COMPARE(listenerReceipts[0].nativeIndex, 0);
  CORRADE_VERIFY(listenerReceipts[0].nativeRealized);
  CORRADE_COMPARE(listenerReceipts[0].orientationWxyz,
                  (std::array<float, 4>{{0.0f, 0.0f, 1.0f, 0.0f}}));
  CORRADE_COMPARE(listenerReceipts[0].radius, 0.075f);
  CORRADE_COMPARE(context.simulateOwned().size(), 2);
}

void RLRAcousticContextTest::resetRestoresFreshSimulationSequence() {
  for (const bool temporalCoherence : {false, true}) {
    RLRContextConfiguration configuration = quickConfiguration();
    configuration.temporalCoherence = temporalCoherence;
    RLRAcousticContext context{configuration};
    const auto initialize = [&context]() {
      context.loadAcousticScene(materialDatabasePath(), kCategories,
                                {makeCube()});
      context.addSource("source", {{0.6f, 0.1f, -0.2f}});
      context.addListener("listener", {{-0.4f, 0.0f, 0.25f}},
                          {{1.0f, 0.0f, 0.0f, 0.0f}},
                          RLRChannelLayoutType::Mono, 1);
    };
    initialize();
    const auto first = context.simulateOwned();
    const auto second = context.simulateOwned();
    CORRADE_VERIFY(first[0].samples != second[0].samples);
    context.reset();
    initialize();
    const auto resetFirst = context.simulateOwned();
    CORRADE_COMPARE(first[0].sampleCount, resetFirst[0].sampleCount);
    CORRADE_COMPARE(first[0].samples, resetFirst[0].samples);
  }
}

void RLRAcousticContextTest::foaAxisConvention() {
  const auto simulateAxis = [](const std::array<float, 3>& position) {
    RLRAcousticContext context{directOnlyConfiguration()};
    context.loadAcousticScene(materialDatabasePath(), kCategories,
                              {makeCube()});
    context.addSource("source", position);
    context.addListener("listener", {{0.0f, 0.0f, 0.0f}},
                        {{1.0f, 0.0f, 0.0f, 0.0f}},
                        RLRChannelLayoutType::Ambisonics, 4);
    return context.simulateOwned().front();
  };
  const auto atPeak = [](const RLROwnedIR& response,
                         const std::size_t channelIndex,
                         const std::size_t peakIndex) {
    return response.samples[channelIndex * response.sampleCount + peakIndex];
  };
  const auto peakIndex = [](const RLROwnedIR& response) {
    const auto begin = response.samples.begin();
    return static_cast<std::size_t>(std::distance(
        begin, std::max_element(begin, begin + response.sampleCount,
                                [](const float left, const float right) {
                                  return std::abs(left) < std::abs(right);
                                })));
  };

  const RLROwnedIR positiveX = simulateAxis({{1.0f, 0.0f, 0.0f}});
  const RLROwnedIR negativeX = simulateAxis({{-1.0f, 0.0f, 0.0f}});
  const RLROwnedIR positiveY = simulateAxis({{0.0f, 1.0f, 0.0f}});
  const RLROwnedIR positiveZ = simulateAxis({{0.0f, 0.0f, 1.0f}});
  CORRADE_COMPARE(positiveX.channelLabels,
                  (std::vector<std::string>{"W", "Y", "Z", "X"}));
  const std::size_t positiveXPeak = peakIndex(positiveX);
  const std::size_t negativeXPeak = peakIndex(negativeX);
  const std::size_t positiveYPeak = peakIndex(positiveY);
  const std::size_t positiveZPeak = peakIndex(positiveZ);
  const float sqrtThree = std::sqrt(3.0f);
  const auto verifyDirectionalChannel =
      [this, &atPeak, sqrtThree](const RLROwnedIR& response,
                                const std::size_t peak,
                                const std::size_t directionalChannel) {
        const float w = atPeak(response, 0, peak);
        const float directional =
            atPeak(response, directionalChannel, peak);
        CORRADE_VERIFY(std::abs(w) > 1.0e-6f);
        CORRADE_VERIFY(std::abs(std::abs(directional / w) - sqrtThree) <
                       1.0e-4f);
      };
  verifyDirectionalChannel(positiveX, positiveXPeak, 3);
  verifyDirectionalChannel(negativeX, negativeXPeak, 3);
  verifyDirectionalChannel(positiveY, positiveYPeak, 1);
  verifyDirectionalChannel(positiveZ, positiveZPeak, 2);
  CORRADE_VERIFY(atPeak(positiveX, 3, positiveXPeak) *
                     atPeak(negativeX, 3, negativeXPeak) <
                 0.0f);
}

void RLRAcousticContextTest::concurrentResetAndNativeOperations() {
  RLRAcousticContext context{quickConfiguration()};
  context.loadAcousticScene(materialDatabasePath(), kCategories, {makeCube()});
  context.addSource("source0", {{0.0f, 0.0f, 0.0f}});
  context.addListener("listener0", {{1.0f, 0.0f, 0.0f}},
                      {{1.0f, 0.0f, 0.0f, 0.0f}},
                      RLRChannelLayoutType::Mono, 1);

  std::atomic<bool> start{false};
  std::atomic<bool> resetFailed{false};
  std::atomic<bool> unexpectedSimulationFailure{false};
  std::thread simulation{[&]() {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    try {
      context.simulateOwned();
    } catch (const std::logic_error&) {
      // Reset won the mutex first, which is an allowed serialized outcome.
    } catch (...) {
      unexpectedSimulationFailure.store(true, std::memory_order_release);
    }
  }};
  std::thread reset{[&]() {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    try {
      context.reset();
    } catch (...) {
      resetFailed.store(true, std::memory_order_release);
    }
  }};
  start.store(true, std::memory_order_release);
  simulation.join();
  reset.join();
  CORRADE_VERIFY(!resetFailed.load());
  CORRADE_VERIFY(!unexpectedSimulationFailure.load());
  CORRADE_VERIFY(!context.sceneLoaded());
  CORRADE_COMPARE(context.objectCount(), 0);

  std::atomic<bool> stopReaders{false};
  std::atomic<bool> invalidCount{false};
  std::thread reader{[&]() {
    while (!stopReaders.load(std::memory_order_acquire)) {
      const std::size_t objectCount = context.objectCount();
      if (objectCount > 1) {
        invalidCount.store(true, std::memory_order_release);
      }
      static_cast<void>(context.sceneLoaded());
      static_cast<void>(context.configuration());
    }
  }};
  for (std::size_t iteration = 0; iteration < 25; ++iteration) {
    context.loadAcousticScene(materialDatabasePath(), kCategories,
                              {makeCube()});
    context.reset();
  }
  stopReaders.store(true, std::memory_order_release);
  reader.join();
  CORRADE_VERIFY(!invalidCount.load());
}

void RLRAcousticContextTest::rejectUnknownMaterial() {
  RLRAcousticContext context{quickConfiguration()};
  bool rejected = false;
  try {
    context.loadAcousticScene(materialDatabasePath(),
                              {"avengine_missing_material_31ac"},
                              {makeCube()});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CORRADE_VERIFY(rejected);
  CORRADE_COMPARE(context.objectCount(), 0);
}

void RLRAcousticContextTest::rejectIncompleteTriangleCoverage() {
  RLRAcousticContext context{quickConfiguration()};
  RLRMeshObject object = makeCube();
  object.triangleMaterialIds.pop_back();
  bool rejected = false;
  try {
    context.loadAcousticScene(materialDatabasePath(), kCategories, {object});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CORRADE_VERIFY(rejected);
  CORRADE_COMPARE(context.objectCount(), 0);
}

}  // namespace

CORRADE_TEST_MAIN(RLRAcousticContextTest)
