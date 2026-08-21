// Copyright (c) Meta Platforms, Inc. and its affiliates.
// Modifications Copyright (c) AVEngine contributors.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#ifndef ESP_AUDIO_RLRACOUSTICCONTEXT_H_
#define ESP_AUDIO_RLRACOUSTICCONTEXT_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <RLRAudioPropagation.h>

namespace esp {
namespace audio {

/**
 * @brief AVEngine-owned configuration for a modern RLR context.
 *
 * This deliberately does not expose @ref RLRA_ContextConfiguration across the
 * Python ABI boundary. The native structure is always initialized through
 * @ref RLRA_ContextConfigurationDefault before these values are copied into
 * it.
 */
struct RLRContextConfiguration {
  std::size_t frequencyBands = 4;
  std::size_t directSHOrder = 3;
  std::size_t indirectSHOrder = 1;
  std::size_t directRayCount = 500;
  std::size_t indirectRayCount = 5000;
  std::size_t indirectRayDepth = 200;
  std::size_t sourceRayCount = 200;
  std::size_t sourceRayDepth = 10;
  std::size_t maxDiffractionOrder = 10;
  std::size_t threadCount = 1;
  float sampleRate = 44100.0f;
  float maxIRLength = 4.0f;
  float unitScale = 1.0f;
  float globalVolume = 1.0f;
  std::array<float, 3> hrtfRight{{1.0f, 0.0f, 0.0f}};
  std::array<float, 3> hrtfUp{{0.0f, 1.0f, 0.0f}};
  std::array<float, 3> hrtfBack{{0.0f, 0.0f, 1.0f}};
  bool direct = true;
  bool indirect = true;
  bool diffraction = true;
  bool transmission = true;
  bool meshSimplification = false;
  bool temporalCoherence = false;
};

/** A strict, already-owned acoustic mesh object. */
struct RLRMeshObject {
  std::string objectId;
  //! Packed xyz float32 values, three entries per vertex.
  std::vector<float> vertices;
  //! Packed uint32 triangle indices, three entries per triangle.
  std::vector<std::uint32_t> triangles;
  //! One material-category index for every triangle.
  std::vector<std::uint32_t> triangleMaterialIds;
  std::array<float, 3> position{{0.0f, 0.0f, 0.0f}};
  //! RLR order is explicitly [w, x, y, z].
  std::array<float, 4> orientationWxyz{{1.0f, 0.0f, 0.0f, 0.0f}};
};

/** Exact evidence for one successful @c RLRA_AddMeshIndices call. */
struct RLRMaterialUploadReceipt {
  std::string objectId;
  std::string materialCategory;
  std::size_t triangleCount = 0;
  std::size_t indexCount = 0;
  std::size_t canonicalPayloadByteCount = 0;
  std::string canonicalPayloadSha1;
};

/** Evidence returned after a complete, transactional scene upload. */
struct RLRSceneUploadReport {
  std::size_t objectCount = 0;
  std::size_t vertexCount = 0;
  std::size_t triangleCount = 0;
  std::size_t materialCategoryCount = 0;
  std::vector<std::string> objectIds;
  std::map<std::string, std::size_t> triangleCountByMaterial;
  std::map<std::string, std::size_t> materialUploadCallCount;
  std::map<std::string, std::string> resolvedMaterialNameByCategory;
  std::map<std::string, std::size_t> resolvedMaterialIndexByCategory;
  std::vector<RLRMaterialUploadReceipt> materialUploadReceipts;
  std::size_t expectedMaterialBlockCount = 0;
  std::string materialDatabaseSha1;
  std::string expectedWorldGeometrySha1;
  std::size_t expectedCanonicalByteCount = 0;
  std::string expectedMaterialCoefficientSha1;
  std::size_t expectedMaterialCoefficientByteCount = 0;
};

/** Parsed evidence from RLR's exported, world-space debug OBJ. */
struct RLRSceneReadbackReport {
  std::string outputPath;
  std::size_t vertexCount = 0;
  std::size_t triangleCount = 0;
  std::size_t materialBlockCount = 0;
  std::size_t materialAssignmentCount = 0;
  std::size_t expectedVertexCount = 0;
  std::size_t expectedTriangleCount = 0;
  std::size_t expectedMaterialBlockCount = 0;
  std::size_t canonicalByteCount = 0;
  std::size_t expectedCanonicalByteCount = 0;
  std::string worldGeometrySha1;
  std::string expectedWorldGeometrySha1;
  std::size_t materialCoefficientByteCount = 0;
  std::size_t expectedMaterialCoefficientByteCount = 0;
  std::string materialCoefficientSha1;
  std::string expectedMaterialCoefficientSha1;
  bool worldGeometryMatches = false;
  bool materialCoefficientsMatch = false;
  bool materialEvidenceMatches = false;
  bool verificationPassed = false;
  //! RLR OBJ uses random vertex colors and has no per-face category IDs.
  bool perFaceMaterialIdsAvailable = false;
  std::string materialEvidenceMode;
};

enum class RLRChannelLayoutType : int {
  Mono = RLRA_ChannelLayoutType_Mono,
  Binaural = RLRA_ChannelLayoutType_Binaural,
  Ambisonics = RLRA_ChannelLayoutType_Ambisonics,
};

/** Canonical logical/native registration evidence for one source. */
struct RLRSourceReceipt {
  std::string sourceId;
  std::array<float, 3> position{{0.0f, 0.0f, 0.0f}};
  float radius = 0.0f;
  //! Canonical bytewise-ID index used by the next/current native realization.
  std::size_t nativeIndex = 0;
  //! False while a newly-added endpoint is waiting for lazy native rebuild.
  bool nativeRealized = false;
};

/** Canonical logical/native registration evidence for one listener. */
struct RLRListenerReceipt {
  std::string listenerId;
  std::array<float, 3> position{{0.0f, 0.0f, 0.0f}};
  std::array<float, 4> orientationWxyz{{1.0f, 0.0f, 0.0f, 0.0f}};
  RLRChannelLayoutType layoutType = RLRChannelLayoutType::Mono;
  std::size_t channelCount = 1;
  float radius = 0.1f;
  std::string hrtfFilePath;
  //! Canonical bytewise-ID index used by the next/current native realization.
  std::size_t nativeIndex = 0;
  //! False while a newly-added endpoint is waiting for lazy native rebuild.
  bool nativeRealized = false;
};

/** A copied IR whose storage is independent of the RLR simulation buffer. */
struct RLROwnedIR {
  std::string listenerId;
  std::string sourceId;
  std::size_t listenerNativeIndex = 0;
  std::size_t sourceNativeIndex = 0;
  float sampleRate = 0.0f;
  std::size_t channelCount = 0;
  std::size_t sampleCount = 0;
  RLRChannelLayoutType layoutType = RLRChannelLayoutType::Mono;
  std::size_t ambisonicOrder = 0;
  //! "ACN" for ambisonics; "native" for mono/binaural output.
  std::string channelOrdering;
  //! "N3D" for ambisonics; "none" otherwise.
  std::string normalization;
  //! AVEngine's exact coordinate-frame contract identifier.
  std::string coordinateFrame;
  //! Stable machine-readable layout contract identifier.
  std::string layoutId;
  //! FOA is explicitly ["W", "Y", "Z", "X"].
  std::vector<std::string> channelLabels;
  //! Channel-major samples: [channel][sample].
  std::vector<float> samples;
};

/** Result of an RLR acoustic-geometry ray query. */
struct RLRRayResult {
  bool hit = false;
  //! Any-hit queries intentionally do not provide distance or normal details.
  bool hasHitDetails = false;
  float distance = 0.0f;
  std::array<float, 3> normal{{0.0f, 0.0f, 0.0f}};
};

/**
 * @brief RAII adapter over the modern, context-based @c RLRA_* C API.
 *
 * This class is intentionally independent of Habitat's stock AudioSensor,
 * which uses RLR's deprecated single-source/single-listener C++ wrapper.
 * Acoustic geometry is uploaded transactionally: a failed upload leaves the
 * previously admitted context untouched.
 */
class RLRAcousticContext {
 public:
  explicit RLRAcousticContext(
      const RLRContextConfiguration& configuration = {});
  ~RLRAcousticContext();

  RLRAcousticContext(const RLRAcousticContext&) = delete;
  RLRAcousticContext& operator=(const RLRAcousticContext&) = delete;
  RLRAcousticContext(RLRAcousticContext&&) = delete;
  RLRAcousticContext& operator=(RLRAcousticContext&&) = delete;

  /**
   * @brief Validate and atomically upload an explicit acoustic scene.
   *
   * Every triangle must have exactly one valid material-category ID. Every
   * declared category must be used and must select exactly one RLR material
   * through an exact label in the supplied database. This prevents RLR's
   * otherwise silent default-material fallback.
   */
  RLRSceneUploadReport loadAcousticScene(
      const std::string& materialDatabaseJson,
      const std::vector<std::string>& materialCategories,
      const std::vector<RLRMeshObject>& objects);

  /** Export and strictly parse RLR's world-space geometry/material readback. */
  RLRSceneReadbackReport writeSceneMeshOBJ(
      const std::string& outputPath) const;

  /**
   * Add a stable, named point or spherical source.
   *
   * The endpoint is stored logically. Native sources are rebuilt lazily in
   * bytewise source-ID order before the next simulation. The returned value is
   * its canonical index for the current logical set; use @ref sourceIndex after
   * all endpoints are declared rather than retaining an earlier value.
   */
  std::size_t addSource(const std::string& sourceId,
                        const std::array<float, 3>& position,
                        float radius = 0.0f);

  /** Add a named listener and return its current canonical logical index. */
  std::size_t addListener(
      const std::string& listenerId,
      const std::array<float, 3>& position,
      const std::array<float, 4>& orientationWxyz,
      RLRChannelLayoutType layoutType = RLRChannelLayoutType::Mono,
      std::size_t channelCount = 1,
      float radius = 0.1f,
      const std::string& hrtfFilePath = "");

  /** Update an existing source without changing its canonical native index. */
  void setSourcePosition(const std::string& sourceId,
                         const std::array<float, 3>& position);
  void setSourceRadius(const std::string& sourceId, float radius);

  /** Update an existing listener pose without changing its native index. */
  void setListenerPose(
      const std::string& listenerId,
      const std::array<float, 3>& position,
      const std::array<float, 4>& orientationWxyz);
  void setListenerRadius(const std::string& listenerId, float radius);

  /** Return the current canonical bytewise-ID native index for an endpoint. */
  std::size_t sourceIndex(const std::string& sourceId) const;
  std::size_t listenerIndex(const std::string& listenerId) const;

  /** Return sorted pose/layout receipts for the current logical endpoints. */
  std::vector<RLRSourceReceipt> sourceReceipts() const;
  std::vector<RLRListenerReceipt> listenerReceipts() const;

  /** Simulate all pairs and immediately copy every RLR-owned IR buffer. */
  std::vector<RLROwnedIR> simulateOwned();

  /**
   * @brief Test whether a normalized world-space ray intersects the scene.
   *
   * RLR only exposes a boolean for this fast query; @c hasHitDetails is false
   * in the returned result.
   */
  RLRRayResult traceRayAnyHit(const std::array<float, 3>& origin,
                              const std::array<float, 3>& direction,
                              float minDistance,
                              float maxDistance) const;

  /**
   * Trace the first hit and return metric distance and RLR's un-normalized
   * surface normal. A hit normal is finite and non-zero, but its magnitude
   * depends on triangle geometry and is not a direction invariant.
   */
  RLRRayResult traceRayFirstHit(const std::array<float, 3>& origin,
                                const std::array<float, 3>& direction,
                                float minDistance,
                                float maxDistance) const;

  /** Return RLR's post-simulation enclosed-space diagnostic. */
  float indirectRayEfficiency() const;

  /**
   * Write RLR's own metrics for a named pair and return per-channel files.
   * RLR treats @p outputPathPrefix as a prefix and appends channel suffixes.
   */
  std::vector<std::string> writeIRMetrics(
      const std::string& listenerId,
      const std::string& sourceId,
      const std::string& outputPathPrefix) const;

  /** Destroy all acoustic state and recreate a fresh context. */
  void reset();

  bool sceneLoaded() const;
  std::size_t objectCount() const;
  std::size_t sourceCount() const;
  std::size_t listenerCount() const;
  RLRContextConfiguration configuration() const;
  std::vector<RLRMaterialUploadReceipt> materialUploadReceipts() const;

 private:
  RLRA_ContextConfiguration makeNativeConfiguration() const;
  RLRA_Context createNativeContext() const;
  static void destroyNativeContext(RLRA_Context context) noexcept;
  std::vector<std::size_t> canonicalSourceOrder() const;
  std::vector<std::size_t> canonicalListenerOrder() const;
  std::size_t findSourceRecord(const std::string& sourceId) const;
  std::size_t findListenerRecord(const std::string& listenerId) const;
  void refreshCanonicalEndpointIndices();
  void realizeEndpoints();

  mutable std::mutex mutex_;
  const RLRContextConfiguration configuration_;
  RLRA_Context context_ = nullptr;
  bool sceneLoaded_ = false;
  bool hasSimulated_ = false;
  bool endpointsDirty_ = false;
  std::vector<RLRSourceReceipt> sources_;
  std::vector<RLRListenerReceipt> listeners_;
  std::vector<RLRMaterialUploadReceipt> materialUploadReceipts_;
  std::string expectedWorldGeometryCanonical_;
  std::string expectedWorldGeometrySha1_;
  std::string expectedMaterialCoefficientCanonical_;
  std::string expectedMaterialCoefficientSha1_;
  std::size_t expectedVertexCount_ = 0;
  std::size_t expectedTriangleCount_ = 0;
  std::size_t expectedMaterialBlockCount_ = 0;
};

}  // namespace audio
}  // namespace esp

#endif  // ESP_AUDIO_RLRACOUSTICCONTEXT_H_
