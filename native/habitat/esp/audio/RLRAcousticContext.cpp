// Copyright (c) Meta Platforms, Inc. and its affiliates.
// Modifications Copyright (c) AVEngine contributors.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "esp/audio/RLRAcousticContext.h"
#include "esp/audio/RLRAcousticContextTestAccess.h"

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>

#include <Corrade/Utility/Sha1.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <iterator>
#include <locale>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace esp {
namespace audio {
namespace {

struct MaterialDefinition {
  std::string name;
  std::vector<std::string> labels;
  std::vector<double> absorption;
  std::vector<double> scattering;
  std::vector<double> transmission;
};

struct MaterialDatabaseSnapshot {
  std::vector<MaterialDefinition> materials;
  std::string bytes;
};

struct CanonicalGeometry {
  std::string bytes;
  std::string sha1;
  std::size_t vertexCount = 0;
  std::size_t triangleCount = 0;
};

struct CanonicalMaterialCoefficients {
  std::string bytes;
  std::string sha1;
  std::size_t materialBlockCount = 0;
};

struct ParsedSceneOBJ {
  CanonicalGeometry geometry;
  std::size_t materialBlockCount = 0;
  std::size_t materialAssignmentCount = 0;
  std::string materialCoefficientBytes;
  std::string materialCoefficientSha1;
};

class NativeContextGuard {
 public:
  explicit NativeContextGuard(RLRA_Context context) : context_{context} {}
  ~NativeContextGuard() {
    if (context_) {
      RLRA_DestroyContext(context_);
    }
  }

  NativeContextGuard(const NativeContextGuard&) = delete;
  NativeContextGuard& operator=(const NativeContextGuard&) = delete;

  RLRA_Context get() const { return context_; }
  RLRA_Context release() {
    RLRA_Context result = context_;
    context_ = nullptr;
    return result;
  }

 private:
  RLRA_Context context_ = nullptr;
};

class PrivateTemporaryDirectory {
 public:
  explicit PrivateTemporaryDirectory(const std::string& prefix) {
    static std::atomic<std::uint64_t> serial{0};
    std::error_code error;
    const std::filesystem::path temporaryRoot =
        std::filesystem::temp_directory_path(error);
    if (error) {
      throw std::runtime_error(
          "cannot resolve temporary directory for RLR evidence");
    }
    const auto timestamp = std::chrono::steady_clock::now()
                               .time_since_epoch()
                               .count();
    const auto threadHash =
        std::hash<std::thread::id>{}(std::this_thread::get_id());
    for (std::size_t attempt = 0; attempt < 100; ++attempt) {
      directory_ = temporaryRoot /
                   (prefix + std::to_string(timestamp) + "-" +
                    std::to_string(threadHash) + "-" +
                    std::to_string(serial.fetch_add(1)));
      error.clear();
      if (std::filesystem::create_directory(directory_, error)) {
        break;
      }
      directory_.clear();
    }
    if (directory_.empty()) {
      throw std::runtime_error("cannot create an exclusive temporary "
                               "directory for RLR evidence");
    }
    std::filesystem::permissions(
        directory_, std::filesystem::perms::owner_all,
        std::filesystem::perm_options::replace, error);
    if (error) {
      cleanup();
      throw std::runtime_error(
          "cannot secure the temporary RLR evidence directory");
    }
  }

  ~PrivateTemporaryDirectory() { cleanup(); }

  PrivateTemporaryDirectory(const PrivateTemporaryDirectory&) = delete;
  PrivateTemporaryDirectory& operator=(const PrivateTemporaryDirectory&) =
      delete;

  std::filesystem::path path(const std::string& filename) const {
    return directory_ / filename;
  }

 private:
  void cleanup() noexcept {
    if (!directory_.empty()) {
      std::error_code ignored;
      std::filesystem::remove_all(directory_, ignored);
      directory_.clear();
    }
  }

  std::filesystem::path directory_;
};

class TemporaryMaterialDatabase {
 public:
  explicit TemporaryMaterialDatabase(const std::string& bytes)
      : directory_{"avengine-rlr-material-"},
        path_{directory_.path("materials.json")} {
    std::ofstream output{path_, std::ios::binary | std::ios::trunc};
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    output.close();
    if (!output) {
      throw std::runtime_error("cannot write material database snapshot");
    }
    std::error_code error;
    error.clear();
    std::filesystem::permissions(
        path_, std::filesystem::perms::owner_read |
                   std::filesystem::perms::owner_write,
        std::filesystem::perm_options::replace, error);
    if (error) {
      throw std::runtime_error("cannot secure material database snapshot");
    }
    std::ifstream verification{path_, std::ios::binary};
    std::ostringstream verifiedBytes;
    verifiedBytes << verification.rdbuf();
    if (!verification || verifiedBytes.str() != bytes) {
      throw std::runtime_error("material database snapshot verification failed");
    }
  }

  TemporaryMaterialDatabase(const TemporaryMaterialDatabase&) = delete;
  TemporaryMaterialDatabase& operator=(const TemporaryMaterialDatabase&) =
      delete;

  std::string path() const { return path_.string(); }

 private:
  PrivateTemporaryDirectory directory_;
  std::filesystem::path path_;
};

void checkRlr(const RLRA_Error error, const std::string& operation) {
  if (error != RLRA_Success) {
    throw std::runtime_error(operation + " failed with RLRA error " +
                             std::to_string(static_cast<int>(error)));
  }
}

bool isFinite(const float value) {
  return std::isfinite(static_cast<double>(value));
}

template <std::size_t size>
void requireFinite(const std::array<float, size>& values,
                   const std::string& label) {
  if (!std::all_of(values.begin(), values.end(), isFinite)) {
    throw std::invalid_argument(label + " must contain only finite values");
  }
}

std::string lowerAscii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](char character) {
    return static_cast<char>(
        std::tolower(static_cast<unsigned char>(character)));
  });
  return value;
}

std::vector<double> validateFrequencyCurve(const rapidjson::Value& material,
                                           const char* field,
                                           const bool unitInterval) {
  if (!material.HasMember(field) || !material[field].IsArray()) {
    throw std::invalid_argument(std::string("material field '") + field +
                                "' must be an array");
  }
  const auto& curve = material[field];
  if (curve.Size() < 4 || curve.Size() % 2 != 0) {
    throw std::invalid_argument(std::string("material field '") + field +
                                "' must contain frequency/value pairs");
  }

  double previousFrequency = -std::numeric_limits<double>::infinity();
  std::vector<double> values;
  values.reserve(curve.Size() / 2);
  for (rapidjson::SizeType index = 0; index < curve.Size(); index += 2) {
    if (!curve[index].IsNumber() || !curve[index + 1].IsNumber()) {
      throw std::invalid_argument(std::string("material field '") + field +
                                  "' must contain only numbers");
    }
    const double frequency = curve[index].GetDouble();
    const double value = curve[index + 1].GetDouble();
    if (!std::isfinite(frequency) || frequency <= previousFrequency ||
        frequency <= 0.0 || !std::isfinite(value)) {
      throw std::invalid_argument(std::string("material field '") + field +
                                  "' contains an invalid curve");
    }
    if (unitInterval && (value < 0.0 || value > 1.0)) {
      throw std::invalid_argument(std::string("material field '") + field +
                                  "' values must be in [0, 1]");
    }
    if (!unitInterval && value < 0.0) {
      throw std::invalid_argument(std::string("material field '") + field +
                                  "' values must be non-negative");
    }
    previousFrequency = frequency;
    values.emplace_back(value);
  }
  return values;
}

MaterialDatabaseSnapshot parseAndValidateMaterialDatabase(
    const std::string& path) {
  if (path.empty()) {
    throw std::invalid_argument("material database path must not be empty");
  }
  std::ifstream stream{path, std::ios::binary};
  if (!stream) {
    throw std::invalid_argument("cannot open material database: " + path);
  }

  std::ostringstream raw;
  raw << stream.rdbuf();
  if (!stream.eof() && stream.fail()) {
    throw std::invalid_argument("cannot read material database: " + path);
  }
  MaterialDatabaseSnapshot snapshot;
  snapshot.bytes = raw.str();
  if (snapshot.bytes.empty()) {
    throw std::invalid_argument("material database must not be empty");
  }
  rapidjson::Document document;
  document.Parse(snapshot.bytes.data(), snapshot.bytes.size());
  if (document.HasParseError()) {
    throw std::invalid_argument(
        "cannot parse material database at byte " +
        std::to_string(document.GetErrorOffset()) + ": " +
        rapidjson::GetParseError_En(document.GetParseError()));
  }
  if (!document.IsObject() || !document.HasMember("materials") ||
      !document["materials"].IsArray() ||
      document["materials"].Empty()) {
    throw std::invalid_argument(
        "material database must contain a non-empty 'materials' array");
  }

  snapshot.materials.reserve(document["materials"].Size());
  std::set<std::string> globallyUniqueLabels;
  for (const auto& material : document["materials"].GetArray()) {
    if (!material.IsObject()) {
      throw std::invalid_argument("every material must be an object");
    }
    MaterialDefinition definition;
    if (!material.HasMember("name") || !material["name"].IsString() ||
        std::strlen(material["name"].GetString()) == 0) {
      throw std::invalid_argument(
          "every material must contain a non-empty string name");
    }
    definition.name = material["name"].GetString();
    if (!material.HasMember("labels") || !material["labels"].IsArray()) {
      throw std::invalid_argument("every material must contain a labels array");
    }
    if (material["labels"].Empty()) {
      throw std::invalid_argument(
          "every material must contain at least one label");
    }
    for (const auto& label : material["labels"].GetArray()) {
      if (!label.IsString() || label.GetStringLength() == 0) {
        throw std::invalid_argument(
            "material labels must be non-empty lowercase strings");
      }
      const std::string value{label.GetString(), label.GetStringLength()};
      const std::string lowered = lowerAscii(value);
      if (!globallyUniqueLabels.insert(lowered).second) {
        throw std::invalid_argument(
            "material labels must be globally unique case-insensitively: " +
            value);
      }
      if (value != lowered) {
        throw std::invalid_argument(
            "material labels must be non-empty lowercase strings: " + value);
      }
      definition.labels.emplace_back(value);
    }

    definition.absorption =
        validateFrequencyCurve(material, "absorption", true);
    definition.scattering =
        validateFrequencyCurve(material, "scattering", true);
    definition.transmission =
        validateFrequencyCurve(material, "transmission", true);
    validateFrequencyCurve(material, "damping", false);
    if (!material.HasMember("speed") || !material["speed"].IsNumber() ||
        !std::isfinite(material["speed"].GetDouble()) ||
        material["speed"].GetDouble() <= 0.0) {
      throw std::invalid_argument(
          "material field 'speed' must be a finite positive number");
    }
    snapshot.materials.emplace_back(std::move(definition));
  }
  return snapshot;
}

std::size_t validateMaterialCategory(
    const std::string& category,
    const std::vector<MaterialDefinition>& materials) {
  if (category.empty()) {
    throw std::invalid_argument("material categories must not be empty");
  }

  const std::string loweredCategory = lowerAscii(category);
  std::size_t bestMatchCount = 0;
  std::vector<std::size_t> winners;
  for (std::size_t materialIndex = 0; materialIndex < materials.size();
       ++materialIndex) {
    std::size_t matchCount = 0;
    for (const std::string& label : materials[materialIndex].labels) {
      const std::string loweredLabel = lowerAscii(label);
      if (loweredCategory.find(loweredLabel) != std::string::npos) {
        ++matchCount;
      }
    }
    if (matchCount > bestMatchCount) {
      bestMatchCount = matchCount;
      winners.assign(1, materialIndex);
    } else if (matchCount != 0 && matchCount == bestMatchCount) {
      winners.emplace_back(materialIndex);
    }
  }

  if (bestMatchCount == 0 || winners.empty()) {
    throw std::invalid_argument(
        "material category '" + category +
        "' has no RLR label match; refusing silent default fallback");
  }
  if (winners.size() != 1) {
    throw std::invalid_argument("material category '" + category +
                                "' ambiguously matches multiple materials");
  }

  const MaterialDefinition& winner = materials[winners.front()];
  const bool hasExactLabel =
      std::any_of(winner.labels.begin(), winner.labels.end(),
                  [&loweredCategory](const std::string& label) {
                    return lowerAscii(label) == loweredCategory;
                  });
  if (!hasExactLabel) {
    throw std::invalid_argument(
        "material category '" + category +
        "' only matches an RLR label by substring; an exact label is required");
  }
  return winners.front();
}

void validateConfiguration(const RLRContextConfiguration& configuration) {
  if (configuration.frequencyBands != 4 &&
      configuration.frequencyBands != 8) {
    throw std::invalid_argument("frequencyBands must be exactly 4 or 8");
  }
  if (configuration.directSHOrder > 5 ||
      configuration.indirectSHOrder > 5) {
    throw std::invalid_argument(
        "directSHOrder and indirectSHOrder must be in [0, 5]");
  }
  if (configuration.directRayCount == 0 ||
      configuration.indirectRayCount == 0 ||
      configuration.indirectRayDepth == 0 ||
      configuration.sourceRayCount == 0 ||
      configuration.sourceRayDepth == 0) {
    throw std::invalid_argument(
        "all direct/indirect/source ray counts and depths must be positive");
  }
  if (configuration.threadCount == 0) {
    throw std::invalid_argument("threadCount must be positive");
  }
  if (configuration.maxDiffractionOrder > 10) {
    throw std::invalid_argument("maxDiffractionOrder must not exceed 10");
  }
  if (!isFinite(configuration.sampleRate) ||
      configuration.sampleRate <= 0.0f ||
      !isFinite(configuration.maxIRLength) ||
      configuration.maxIRLength <= 0.0f || !isFinite(configuration.unitScale) ||
      configuration.unitScale <= 0.0f ||
      !isFinite(configuration.globalVolume) ||
      configuration.globalVolume < 0.0f) {
    throw std::invalid_argument(
        "sampleRate, maxIRLength, unitScale and globalVolume are invalid");
  }
  requireFinite(configuration.hrtfRight, "hrtfRight");
  requireFinite(configuration.hrtfUp, "hrtfUp");
  requireFinite(configuration.hrtfBack, "hrtfBack");
}

void validateQuaternion(const std::array<float, 4>& quaternion,
                        const std::string& label) {
  requireFinite(quaternion, label);
  double squaredNorm = 0.0;
  for (const float value : quaternion) {
    const double promoted = static_cast<double>(value);
    squaredNorm += promoted * promoted;
  }
  if (std::abs(squaredNorm - 1.0) > 1.0e-4) {
    throw std::invalid_argument(label + " must be normalized [w, x, y, z]");
  }
}

void validateObject(const RLRMeshObject& object,
                    const std::size_t materialCount,
                    std::set<std::string>& objectIds,
                    std::vector<std::size_t>& materialUseCounts,
                    RLRSceneUploadReport& report) {
  if (object.objectId.empty() || !objectIds.insert(object.objectId).second) {
    throw std::invalid_argument(
        "acoustic object IDs must be non-empty and unique: " +
        object.objectId);
  }
  if (object.vertices.size() < 9 || object.vertices.size() % 3 != 0) {
    throw std::invalid_argument("object '" + object.objectId +
                                "' vertices must have shape [N, 3]");
  }
  if (!std::all_of(object.vertices.begin(), object.vertices.end(), isFinite)) {
    throw std::invalid_argument("object '" + object.objectId +
                                "' vertices must be finite");
  }
  if (object.triangles.empty() || object.triangles.size() % 3 != 0) {
    throw std::invalid_argument("object '" + object.objectId +
                                "' triangles must have shape [M, 3]");
  }
  const std::size_t triangleCount = object.triangles.size() / 3;
  if (object.triangleMaterialIds.size() != triangleCount) {
    throw std::invalid_argument(
        "object '" + object.objectId +
        "' must provide exactly one material ID for every triangle");
  }
  requireFinite(object.position, "object position");
  validateQuaternion(object.orientationWxyz, "object orientation_wxyz");

  const std::size_t vertexCount = object.vertices.size() / 3;
  for (std::size_t triangleIndex = 0; triangleIndex < triangleCount;
       ++triangleIndex) {
    const std::uint32_t a = object.triangles[triangleIndex * 3];
    const std::uint32_t b = object.triangles[triangleIndex * 3 + 1];
    const std::uint32_t c = object.triangles[triangleIndex * 3 + 2];
    if (a >= vertexCount || b >= vertexCount || c >= vertexCount) {
      throw std::invalid_argument("object '" + object.objectId +
                                  "' contains an out-of-range vertex index");
    }
    if (a == b || b == c || a == c) {
      throw std::invalid_argument("object '" + object.objectId +
                                  "' contains a degenerate triangle");
    }

    const float abX = object.vertices[b * 3] - object.vertices[a * 3];
    const float abY = object.vertices[b * 3 + 1] - object.vertices[a * 3 + 1];
    const float abZ = object.vertices[b * 3 + 2] - object.vertices[a * 3 + 2];
    const float acX = object.vertices[c * 3] - object.vertices[a * 3];
    const float acY = object.vertices[c * 3 + 1] - object.vertices[a * 3 + 1];
    const float acZ = object.vertices[c * 3 + 2] - object.vertices[a * 3 + 2];
    const double crossX = static_cast<double>(abY) *
                              static_cast<double>(acZ) -
                          static_cast<double>(abZ) *
                              static_cast<double>(acY);
    const double crossY = static_cast<double>(abZ) *
                              static_cast<double>(acX) -
                          static_cast<double>(abX) *
                              static_cast<double>(acZ);
    const double crossZ = static_cast<double>(abX) *
                              static_cast<double>(acY) -
                          static_cast<double>(abY) *
                              static_cast<double>(acX);
    if (crossX * crossX + crossY * crossY + crossZ * crossZ <= 1.0e-20) {
      throw std::invalid_argument("object '" + object.objectId +
                                  "' contains a zero-area triangle");
    }

    const std::uint32_t materialId =
        object.triangleMaterialIds[triangleIndex];
    if (materialId >= materialCount) {
      throw std::invalid_argument("object '" + object.objectId +
                                  "' contains an out-of-range material ID");
    }
    ++materialUseCounts[materialId];
  }

  report.objectIds.emplace_back(object.objectId);
  report.vertexCount += vertexCount;
  report.triangleCount += triangleCount;
}

bool bytewiseIdLess(const std::string& left, const std::string& right) {
  return std::lexicographical_compare(
      left.begin(), left.end(), right.begin(), right.end(),
      [](const char leftByte, const char rightByte) {
        return static_cast<unsigned char>(leftByte) <
               static_cast<unsigned char>(rightByte);
      });
}

std::vector<std::string> channelLabels(
    const RLRChannelLayoutType layoutType,
    const std::size_t channelCount) {
  switch (layoutType) {
    case RLRChannelLayoutType::Mono:
      return {"mono"};
    case RLRChannelLayoutType::Binaural:
      return {"left", "right"};
    case RLRChannelLayoutType::Ambisonics: {
      if (channelCount == 4) {
        // RLR's real ACN/N3D first-order basis in +X-right, +Y-up,
        // +Z-back world coordinates. Samples are raw and world-aligned.
        return {"W", "Y", "Z", "X"};
      }
      std::vector<std::string> result;
      result.reserve(channelCount);
      for (std::size_t channelIndex = 0; channelIndex < channelCount;
           ++channelIndex) {
        result.emplace_back("ACN" + std::to_string(channelIndex));
      }
      return result;
    }
  }
  throw std::invalid_argument("unknown listener channel layout type");
}

RLRA_Ray makeRay(const std::array<float, 3>& origin,
                 const std::array<float, 3>& direction,
                 const float minDistance,
                 const float maxDistance) {
  requireFinite(origin, "ray origin");
  requireFinite(direction, "ray direction");
  if (!isFinite(minDistance) || !isFinite(maxDistance) || minDistance < 0.0f ||
      maxDistance <= minDistance) {
    throw std::invalid_argument(
        "ray distances must be finite and satisfy 0 <= min < max");
  }
  const double directionX = static_cast<double>(direction[0]);
  const double directionY = static_cast<double>(direction[1]);
  const double directionZ = static_cast<double>(direction[2]);
  const double norm = std::sqrt(directionX * directionX +
                                directionY * directionY +
                                directionZ * directionZ);
  if (norm <= 1.0e-12) {
    throw std::invalid_argument("ray direction must be non-zero");
  }

  RLRA_Ray ray{};
  for (std::size_t index = 0; index < 3; ++index) {
    ray.origin[index] = origin[index];
    ray.direction[index] =
        static_cast<float>(static_cast<double>(direction[index]) / norm);
  }
  ray.tMin = minDistance;
  ray.tMax = maxDistance;
  return ray;
}

std::string sha1(const std::string& bytes) {
  return Corrade::Utility::Sha1::digest(bytes).hexString();
}

std::string canonicalCoordinate(double value) {
  if (!std::isfinite(value)) {
    throw std::runtime_error("canonical geometry contains a non-finite value");
  }
  if (std::abs(value) < 0.5e-6) {
    value = 0.0;
  }
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << std::fixed << std::setprecision(6) << value;
  return output.str();
}

std::string positionToken(const std::array<double, 3>& position) {
  return canonicalCoordinate(position[0]) + " " +
         canonicalCoordinate(position[1]) + " " +
         canonicalCoordinate(position[2]);
}

std::string canonicalFaceToken(const std::array<std::string, 3>& vertices) {
  const std::array<std::string, 3> rotations{{
      vertices[0] + "|" + vertices[1] + "|" + vertices[2],
      vertices[1] + "|" + vertices[2] + "|" + vertices[0],
      vertices[2] + "|" + vertices[0] + "|" + vertices[1],
  }};
  return *std::min_element(rotations.begin(), rotations.end());
}

CanonicalGeometry finishCanonicalGeometry(
    std::vector<std::string> vertices,
    std::vector<std::string> triangles) {
  std::sort(vertices.begin(), vertices.end());
  std::sort(triangles.begin(), triangles.end());
  CanonicalGeometry result;
  result.vertexCount = vertices.size();
  result.triangleCount = triangles.size();
  std::ostringstream canonical;
  canonical.imbue(std::locale::classic());
  canonical << "AVENGINE_RLR_WORLD_GEOMETRY_V1\n";
  for (const std::string& vertex : vertices) {
    canonical << "v " << vertex << '\n';
  }
  for (const std::string& triangle : triangles) {
    canonical << "f " << triangle << '\n';
  }
  result.bytes = canonical.str();
  result.sha1 = sha1(result.bytes);
  return result;
}

std::array<double, 3> transformVertex(const RLRMeshObject& object,
                                      const std::size_t vertexIndex) {
  const std::array<double, 3> vector{{
      static_cast<double>(object.vertices[vertexIndex * 3]),
      static_cast<double>(object.vertices[vertexIndex * 3 + 1]),
      static_cast<double>(object.vertices[vertexIndex * 3 + 2]),
  }};
  const double w = static_cast<double>(object.orientationWxyz[0]);
  const std::array<double, 3> q{{
      static_cast<double>(object.orientationWxyz[1]),
      static_cast<double>(object.orientationWxyz[2]),
      static_cast<double>(object.orientationWxyz[3]),
  }};
  const std::array<double, 3> twiceCross{{
      2.0 * (q[1] * vector[2] - q[2] * vector[1]),
      2.0 * (q[2] * vector[0] - q[0] * vector[2]),
      2.0 * (q[0] * vector[1] - q[1] * vector[0]),
  }};
  const std::array<double, 3> secondCross{{
      q[1] * twiceCross[2] - q[2] * twiceCross[1],
      q[2] * twiceCross[0] - q[0] * twiceCross[2],
      q[0] * twiceCross[1] - q[1] * twiceCross[0],
  }};
  return {{
      vector[0] + w * twiceCross[0] + secondCross[0] +
          static_cast<double>(object.position[0]),
      vector[1] + w * twiceCross[1] + secondCross[1] +
          static_cast<double>(object.position[1]),
      vector[2] + w * twiceCross[2] + secondCross[2] +
          static_cast<double>(object.position[2]),
  }};
}

CanonicalGeometry canonicalExpectedGeometry(
    const std::vector<RLRMeshObject>& objects) {
  std::vector<std::string> vertices;
  std::vector<std::string> triangles;
  for (const RLRMeshObject& object : objects) {
    std::vector<std::string> objectVertices;
    objectVertices.reserve(object.vertices.size() / 3);
    for (std::size_t vertexIndex = 0;
         vertexIndex < object.vertices.size() / 3; ++vertexIndex) {
      objectVertices.emplace_back(
          positionToken(transformVertex(object, vertexIndex)));
    }
    vertices.insert(vertices.end(), objectVertices.begin(),
                    objectVertices.end());
    for (std::size_t index = 0; index < object.triangles.size(); index += 3) {
      triangles.emplace_back(canonicalFaceToken({{
          objectVertices[object.triangles[index]],
          objectVertices[object.triangles[index + 1]],
          objectVertices[object.triangles[index + 2]],
      }}));
    }
  }
  return finishCanonicalGeometry(std::move(vertices), std::move(triangles));
}

void appendCoefficientCurve(std::ostringstream& canonical,
                            const char* field,
                            const std::vector<double>& values) {
  for (std::size_t index = 0; index < values.size(); ++index) {
    canonical << field << " " << index << " "
              << canonicalCoordinate(values[index]) << '\n';
  }
}

void appendMaterialCoefficientBlock(
    std::ostringstream& canonical,
    const std::size_t objectIndex,
    const std::size_t localMaterialIndex,
    const std::vector<double>& absorption,
    const std::vector<double>& scattering,
    const std::vector<double>& transmission) {
  canonical << "object " << objectIndex << " material " << localMaterialIndex
            << '\n';
  appendCoefficientCurve(canonical, "absorption", absorption);
  appendCoefficientCurve(canonical, "scattering", scattering);
  appendCoefficientCurve(canonical, "transmission", transmission);
}

CanonicalMaterialCoefficients canonicalExpectedMaterialCoefficients(
    const std::vector<RLRMeshObject>& objects,
    const std::vector<std::size_t>& resolvedMaterialIndices,
    const std::vector<MaterialDefinition>& materials) {
  std::ostringstream canonical;
  canonical.imbue(std::locale::classic());
  canonical << "AVENGINE_RLR_MATERIAL_COEFFICIENTS_V1\n";
  CanonicalMaterialCoefficients result;
  for (std::size_t objectIndex = 0; objectIndex < objects.size();
       ++objectIndex) {
    const RLRMeshObject& object = objects[objectIndex];
    std::vector<bool> materialUsed(resolvedMaterialIndices.size(), false);
    for (const std::uint32_t materialId : object.triangleMaterialIds) {
      materialUsed[materialId] = true;
    }
    std::size_t localMaterialIndex = 0;
    for (std::size_t categoryIndex = 0;
         categoryIndex < materialUsed.size(); ++categoryIndex) {
      if (!materialUsed[categoryIndex]) {
        continue;
      }
      const MaterialDefinition& material =
          materials[resolvedMaterialIndices[categoryIndex]];
      appendMaterialCoefficientBlock(
          canonical, objectIndex, localMaterialIndex, material.absorption,
          material.scattering, material.transmission);
      ++localMaterialIndex;
      ++result.materialBlockCount;
    }
  }
  result.bytes = canonical.str();
  result.sha1 = sha1(result.bytes);
  return result;
}

std::size_t parseSize(const std::string& text, const std::string& label) {
  std::size_t consumed = 0;
  std::size_t value = 0;
  try {
    value = static_cast<std::size_t>(std::stoull(text, &consumed));
  } catch (const std::exception&) {
    throw std::runtime_error("invalid " + label + " in RLR OBJ");
  }
  if (consumed != text.size()) {
    throw std::runtime_error("invalid " + label + " in RLR OBJ");
  }
  return value;
}

std::size_t parseFaceIndex(const std::string& token) {
  const std::string index = token.substr(0, token.find('/'));
  const std::size_t value = parseSize(index, "face index");
  if (value == 0) {
    throw std::runtime_error("RLR OBJ face indices must be positive");
  }
  return value - 1;
}

void parseCoefficientLine(const std::string& line,
                          const std::string& prefix,
                          const std::string& field,
                          std::vector<double>& values) {
  if (line.rfind(prefix, 0) != 0) {
    throw std::runtime_error("invalid " + field +
                             " coefficient line in RLR scene OBJ");
  }
  constexpr char delimiter[] = ", Value: ";
  const std::size_t delimiterPosition = line.find(delimiter, prefix.size());
  if (delimiterPosition == std::string::npos) {
    throw std::runtime_error("invalid " + field +
                             " coefficient line in RLR scene OBJ");
  }
  const std::size_t index =
      parseSize(line.substr(prefix.size(), delimiterPosition - prefix.size()),
                field + " coefficient index");
  if (index != values.size()) {
    throw std::runtime_error(field +
                             " coefficient indices must be contiguous");
  }
  std::istringstream valueStream{
      line.substr(delimiterPosition + std::strlen(delimiter))};
  valueStream.imbue(std::locale::classic());
  double value = 0.0;
  std::string extra;
  if (!(valueStream >> value) || (valueStream >> extra) ||
      !std::isfinite(value) || value < 0.0 || value > 1.0) {
    throw std::runtime_error(field +
                             " coefficient must be finite and in [0, 1]");
  }
  values.emplace_back(value);
}

ParsedSceneOBJ parseSceneOBJBytes(const std::string& bytes) {
  std::istringstream input{bytes};

  struct MaterialCoefficientBlock {
    std::vector<double> absorption;
    std::vector<double> scattering;
    std::vector<double> transmission;
  };
  struct ObjectBlock {
    std::size_t vertexOffset = 0;
    std::vector<MaterialCoefficientBlock> materials;
    std::optional<std::size_t> declaredVertexCount;
    std::optional<std::size_t> declaredTriangleCount;
    std::optional<std::size_t> declaredMaterialAssignmentCount;
    std::size_t vertexCount = 0;
    std::vector<std::array<std::size_t, 3>> faces;
  } block;

  std::vector<std::string> vertices;
  std::vector<std::array<std::size_t, 3>> faceIndices;
  std::size_t materialBlockCount = 0;
  std::size_t materialAssignmentCount = 0;
  std::size_t objectBlockCount = 0;
  std::ostringstream materialCoefficientCanonical;
  materialCoefficientCanonical.imbue(std::locale::classic());
  materialCoefficientCanonical
      << "AVENGINE_RLR_MATERIAL_COEFFICIENTS_V1\n";
  bool blockActive = false;

  const auto blockComplete = [&]() {
    return blockActive && block.declaredVertexCount &&
           block.declaredTriangleCount &&
           block.declaredMaterialAssignmentCount &&
           block.vertexCount == *block.declaredVertexCount &&
           block.faces.size() == *block.declaredTriangleCount;
  };
  const auto finishBlock = [&]() {
    if (!blockActive) {
      return;
    }
    if (block.materials.empty() || !block.declaredVertexCount ||
        !block.declaredTriangleCount ||
        !block.declaredMaterialAssignmentCount) {
      throw std::runtime_error(
          "RLR scene OBJ object block has incomplete headers");
    }
    if (*block.declaredVertexCount == 0 ||
        *block.declaredTriangleCount == 0 ||
        block.vertexCount != *block.declaredVertexCount ||
        block.faces.size() != *block.declaredTriangleCount ||
        *block.declaredMaterialAssignmentCount != block.faces.size()) {
      throw std::runtime_error(
          "RLR scene OBJ object-block counts do not match its payload");
    }
    for (std::size_t materialIndex = 0;
         materialIndex < block.materials.size(); ++materialIndex) {
      const MaterialCoefficientBlock& material =
          block.materials[materialIndex];
      if (material.absorption.empty() || material.scattering.empty() ||
          material.transmission.empty()) {
        throw std::runtime_error(
            "RLR scene OBJ material coefficient block is incomplete");
      }
      appendMaterialCoefficientBlock(
          materialCoefficientCanonical, objectBlockCount, materialIndex,
          material.absorption, material.scattering, material.transmission);
    }
    const std::size_t vertexEnd =
        block.vertexOffset + *block.declaredVertexCount;
    for (const auto& face : block.faces) {
      if (face[0] < block.vertexOffset || face[0] >= vertexEnd ||
          face[1] < block.vertexOffset || face[1] >= vertexEnd ||
          face[2] < block.vertexOffset || face[2] >= vertexEnd) {
        throw std::runtime_error(
            "RLR scene OBJ face escapes its object-block vertex range");
      }
    }
    faceIndices.insert(faceIndices.end(), block.faces.begin(),
                       block.faces.end());
    materialBlockCount += block.materials.size();
    materialAssignmentCount += *block.declaredMaterialAssignmentCount;
    ++objectBlockCount;
    block = ObjectBlock{};
    blockActive = false;
  };

  std::string line;
  while (std::getline(input, line)) {
    if (line.rfind("# Material Index : ", 0) == 0) {
      if (blockComplete()) {
        finishBlock();
      } else if (blockActive &&
                 (block.declaredVertexCount || block.declaredTriangleCount ||
                  block.declaredMaterialAssignmentCount ||
                  block.vertexCount != 0 || !block.faces.empty())) {
        throw std::runtime_error(
            "RLR scene OBJ starts a material block before the preceding "
            "object payload is complete");
      }
      if (!blockActive) {
        blockActive = true;
        block.vertexOffset = vertices.size();
      }
      const std::size_t materialIndex =
          parseSize(line.substr(std::strlen("# Material Index : ")),
                    "material block index");
      if (materialIndex != block.materials.size()) {
        throw std::runtime_error(
            "RLR scene OBJ material indices must restart at zero and be "
            "contiguous in every object block");
      }
      block.materials.emplace_back();
    } else if (line.rfind("# Absorption", 0) == 0) {
      if (!blockActive || block.materials.empty() ||
          block.declaredVertexCount || block.declaredTriangleCount ||
          block.declaredMaterialAssignmentCount) {
        throw std::runtime_error(
            "absorption coefficient is outside an RLR material block");
      }
      parseCoefficientLine(line, "# Absorption - Index:", "absorption",
                           block.materials.back().absorption);
    } else if (line.rfind("# Scattering", 0) == 0) {
      if (!blockActive || block.materials.empty() ||
          block.declaredVertexCount || block.declaredTriangleCount ||
          block.declaredMaterialAssignmentCount) {
        throw std::runtime_error(
            "scattering coefficient is outside an RLR material block");
      }
      parseCoefficientLine(line, "# Scattering - Index:", "scattering",
                           block.materials.back().scattering);
    } else if (line.rfind("# Transmission", 0) == 0) {
      if (!blockActive || block.materials.empty() ||
          block.declaredVertexCount || block.declaredTriangleCount ||
          block.declaredMaterialAssignmentCount) {
        throw std::runtime_error(
            "transmission coefficient is outside an RLR material block");
      }
      parseCoefficientLine(line, "# Transmission - Index:", "transmission",
                           block.materials.back().transmission);
    } else if (line.rfind("# Vertex Count: ", 0) == 0) {
      if (!blockActive || block.materials.empty() ||
          block.declaredVertexCount) {
        throw std::runtime_error(
            "invalid or duplicate vertex-count header in RLR scene OBJ");
      }
      block.declaredVertexCount =
          parseSize(line.substr(std::strlen("# Vertex Count: ")),
                    "declared vertex count");
    } else if (line.rfind("# Triangle Count: ", 0) == 0) {
      if (!blockActive || block.materials.empty() ||
          block.declaredTriangleCount) {
        throw std::runtime_error(
            "invalid or duplicate triangle-count header in RLR scene OBJ");
      }
      block.declaredTriangleCount =
          parseSize(line.substr(std::strlen("# Triangle Count: ")),
                    "declared triangle count");
    } else if (line.rfind("# Material Count: ", 0) == 0) {
      if (!blockActive || block.materials.empty() ||
          block.declaredMaterialAssignmentCount) {
        throw std::runtime_error(
            "invalid or duplicate material-count header in RLR scene OBJ");
      }
      block.declaredMaterialAssignmentCount =
          parseSize(line.substr(std::strlen("# Material Count: ")),
                    "declared material assignment count");
    } else if (line.rfind("v ", 0) == 0) {
      if (!blockActive || !block.declaredVertexCount ||
          !block.declaredTriangleCount ||
          !block.declaredMaterialAssignmentCount ||
          block.vertexCount >= *block.declaredVertexCount ||
          !block.faces.empty()) {
        throw std::runtime_error(
            "RLR scene OBJ vertex is outside its declared object block");
      }
      std::istringstream values{line.substr(2)};
      values.imbue(std::locale::classic());
      std::array<double, 3> position{};
      if (!(values >> position[0] >> position[1] >> position[2])) {
        throw std::runtime_error("invalid vertex in RLR scene OBJ");
      }
      vertices.emplace_back(positionToken(position));
      ++block.vertexCount;
    } else if (line.rfind("f ", 0) == 0) {
      if (!blockActive || !block.declaredVertexCount ||
          !block.declaredTriangleCount ||
          !block.declaredMaterialAssignmentCount ||
          block.vertexCount != *block.declaredVertexCount ||
          block.faces.size() >= *block.declaredTriangleCount) {
        throw std::runtime_error(
            "RLR scene OBJ face is outside its declared object block");
      }
      std::istringstream values{line.substr(2)};
      std::array<std::string, 3> tokens;
      std::string extra;
      if (!(values >> tokens[0] >> tokens[1] >> tokens[2]) ||
          (values >> extra)) {
        throw std::runtime_error(
            "RLR scene OBJ must contain only triangular faces");
      }
      block.faces.push_back({{parseFaceIndex(tokens[0]),
                              parseFaceIndex(tokens[1]),
                              parseFaceIndex(tokens[2])}});
    } else if (line == "# Listeners" || line == "# Sources") {
      if (blockActive) {
        finishBlock();
      }
    }
  }
  if (!input.eof()) {
    throw std::runtime_error("failed while reading RLR scene OBJ");
  }
  if (blockActive) {
    finishBlock();
  }
  if (vertices.empty() || faceIndices.empty() || materialBlockCount == 0) {
    throw std::runtime_error("RLR scene OBJ contains no acoustic objects");
  }

  std::vector<std::string> triangles;
  triangles.reserve(faceIndices.size());
  for (const auto& face : faceIndices) {
    if (face[0] >= vertices.size() || face[1] >= vertices.size() ||
        face[2] >= vertices.size()) {
      throw std::runtime_error(
          "RLR scene OBJ contains an out-of-range face index");
    }
    triangles.emplace_back(canonicalFaceToken(
        {{vertices[face[0]], vertices[face[1]], vertices[face[2]]}}));
  }
  ParsedSceneOBJ result;
  result.geometry =
      finishCanonicalGeometry(std::move(vertices), std::move(triangles));
  result.materialBlockCount = materialBlockCount;
  result.materialAssignmentCount = materialAssignmentCount;
  result.materialCoefficientBytes = materialCoefficientCanonical.str();
  result.materialCoefficientSha1 = sha1(result.materialCoefficientBytes);
  return result;
}

void appendLittleEndian(std::string& bytes,
                        const std::uint64_t value,
                        const std::size_t byteCount) {
  for (std::size_t index = 0; index < byteCount; ++index) {
    bytes.push_back(
        static_cast<char>((value >> static_cast<unsigned>(index * 8)) & 0xff));
  }
}

void appendLengthPrefixed(std::string& bytes, const std::string& value) {
  appendLittleEndian(bytes, value.size(), 8);
  bytes.append(value);
}

RLRMaterialUploadReceipt makeUploadReceipt(
    const std::string& objectId,
    const std::string& materialCategory,
    const std::vector<std::uint32_t>& indices) {
  constexpr char header[] = "AVENGINE_RLR_ADD_MESH_INDICES_V1\0";
  std::string payload{header, sizeof(header) - 1};
  appendLengthPrefixed(payload, objectId);
  appendLengthPrefixed(payload, materialCategory);
  appendLittleEndian(payload, 3, 8);
  appendLittleEndian(payload, indices.size(), 8);
  for (const std::uint32_t index : indices) {
    appendLittleEndian(payload, index, 4);
  }
  RLRMaterialUploadReceipt receipt;
  receipt.objectId = objectId;
  receipt.materialCategory = materialCategory;
  receipt.triangleCount = indices.size() / 3;
  receipt.indexCount = indices.size();
  receipt.canonicalPayloadByteCount = payload.size();
  receipt.canonicalPayloadSha1 = sha1(payload);
  return receipt;
}

}  // namespace

namespace detail {

std::string parseRLRSceneOBJMaterialCoefficientSha1ForTesting(
    const std::string& bytes) {
  return parseSceneOBJBytes(bytes).materialCoefficientSha1;
}

}  // namespace detail

RLRAcousticContext::RLRAcousticContext(
    const RLRContextConfiguration& configuration)
    : configuration_{configuration}, context_{createNativeContext()} {}

RLRAcousticContext::~RLRAcousticContext() {
  const std::lock_guard<std::mutex> lock{mutex_};
  destroyNativeContext(context_);
  context_ = nullptr;
}

RLRA_ContextConfiguration RLRAcousticContext::makeNativeConfiguration() const {
  validateConfiguration(configuration_);
  RLRA_ContextConfiguration native{};
  native.thisSize = sizeof(RLRA_ContextConfiguration);
  checkRlr(RLRA_ContextConfigurationDefault(&native),
           "RLRA_ContextConfigurationDefault");
  native.frequencyBands = configuration_.frequencyBands;
  native.directSHOrder = configuration_.directSHOrder;
  native.indirectSHOrder = configuration_.indirectSHOrder;
  native.directRayCount = configuration_.directRayCount;
  native.indirectRayCount = configuration_.indirectRayCount;
  native.indirectRayDepth = configuration_.indirectRayDepth;
  native.sourceRayCount = configuration_.sourceRayCount;
  native.sourceRayDepth = configuration_.sourceRayDepth;
  native.maxDiffractionOrder = configuration_.maxDiffractionOrder;
  native.threadCount = configuration_.threadCount;
  native.sampleRate = configuration_.sampleRate;
  native.maxIRLength = configuration_.maxIRLength;
  native.unitScale = configuration_.unitScale;
  native.globalVolume = configuration_.globalVolume;
  std::copy(configuration_.hrtfRight.begin(), configuration_.hrtfRight.end(),
            native.hrtfRight);
  std::copy(configuration_.hrtfUp.begin(), configuration_.hrtfUp.end(),
            native.hrtfUp);
  std::copy(configuration_.hrtfBack.begin(), configuration_.hrtfBack.end(),
            native.hrtfBack);
  native.direct = configuration_.direct;
  native.indirect = configuration_.indirect;
  native.diffraction = configuration_.diffraction;
  native.transmission = configuration_.transmission;
  native.meshSimplification = configuration_.meshSimplification;
  native.temporalCoherence = configuration_.temporalCoherence;
  return native;
}

RLRA_Context RLRAcousticContext::createNativeContext() const {
  RLRA_ContextConfiguration native = makeNativeConfiguration();
  RLRA_Context context = nullptr;
  checkRlr(RLRA_CreateContext(&context, &native), "RLRA_CreateContext");
  if (!context) {
    throw std::runtime_error("RLRA_CreateContext returned a null context");
  }
  return context;
}

void RLRAcousticContext::destroyNativeContext(RLRA_Context context) noexcept {
  if (context) {
    RLRA_DestroyContext(context);
  }
}

RLRSceneUploadReport RLRAcousticContext::loadAcousticScene(
    const std::string& materialDatabaseJson,
    const std::vector<std::string>& materialCategories,
    const std::vector<RLRMeshObject>& objects) {
  const std::lock_guard<std::mutex> lock{mutex_};
  const MaterialDatabaseSnapshot materialDatabase =
      parseAndValidateMaterialDatabase(materialDatabaseJson);
  const std::vector<MaterialDefinition>& materials = materialDatabase.materials;
  if (materialCategories.empty()) {
    throw std::invalid_argument(
        "at least one material category must be declared");
  }
  std::set<std::string> loweredCategories;
  std::set<std::size_t> uniqueResolvedMaterialIndices;
  std::vector<std::size_t> resolvedMaterialIndices;
  resolvedMaterialIndices.reserve(materialCategories.size());
  for (const std::string& category : materialCategories) {
    const std::string lowered = lowerAscii(category);
    if (!loweredCategories.insert(lowered).second) {
      throw std::invalid_argument(
          "material categories must be unique case-insensitively: " + category);
    }
    const std::size_t resolvedMaterialIndex =
        validateMaterialCategory(category, materials);
    if (!uniqueResolvedMaterialIndices.insert(resolvedMaterialIndex).second) {
      throw std::invalid_argument(
          "material categories must resolve one-to-one to unique material "
          "database entries; duplicate resolution at category '" +
          category + "'");
    }
    resolvedMaterialIndices.emplace_back(resolvedMaterialIndex);
  }
  if (objects.empty()) {
    throw std::invalid_argument("at least one acoustic object is required");
  }

  RLRSceneUploadReport report;
  report.objectCount = objects.size();
  report.materialCategoryCount = materialCategories.size();
  std::set<std::string> objectIds;
  std::vector<std::size_t> materialUseCounts(materialCategories.size(), 0);
  for (const RLRMeshObject& object : objects) {
    validateObject(object, materialCategories.size(), objectIds,
                   materialUseCounts, report);
  }
  for (std::size_t materialIndex = 0;
       materialIndex < materialCategories.size(); ++materialIndex) {
    if (materialUseCounts[materialIndex] == 0) {
      throw std::invalid_argument("declared material category '" +
                                  materialCategories[materialIndex] +
                                  "' is unused");
    }
    report.triangleCountByMaterial.emplace(materialCategories[materialIndex],
                                           materialUseCounts[materialIndex]);
    const std::size_t resolvedIndex = resolvedMaterialIndices[materialIndex];
    report.resolvedMaterialIndexByCategory.emplace(
        materialCategories[materialIndex], resolvedIndex);
    report.resolvedMaterialNameByCategory.emplace(
        materialCategories[materialIndex], materials[resolvedIndex].name);
  }
  report.materialDatabaseSha1 = sha1(materialDatabase.bytes);
  const CanonicalGeometry expectedGeometry = canonicalExpectedGeometry(objects);
  if (expectedGeometry.vertexCount != report.vertexCount ||
      expectedGeometry.triangleCount != report.triangleCount) {
    throw std::runtime_error(
        "internal expected-geometry counts do not match upload receipt");
  }
  report.expectedWorldGeometrySha1 = expectedGeometry.sha1;
  report.expectedCanonicalByteCount = expectedGeometry.bytes.size();
  const CanonicalMaterialCoefficients expectedMaterialCoefficients =
      canonicalExpectedMaterialCoefficients(objects, resolvedMaterialIndices,
                                            materials);
  report.expectedMaterialCoefficientSha1 = expectedMaterialCoefficients.sha1;
  report.expectedMaterialCoefficientByteCount =
      expectedMaterialCoefficients.bytes.size();

  NativeContextGuard candidate{createNativeContext()};
  const TemporaryMaterialDatabase nativeMaterialDatabase{
      materialDatabase.bytes};
  const std::string nativeMaterialPath = nativeMaterialDatabase.path();
  checkRlr(RLRA_SetMaterialDatabaseJSON(candidate.get(),
                                        nativeMaterialPath.c_str()),
           "RLRA_SetMaterialDatabaseJSON");

  for (std::size_t objectIndex = 0; objectIndex < objects.size();
       ++objectIndex) {
    const RLRMeshObject& object = objects[objectIndex];
    checkRlr(RLRA_AddObject(candidate.get()),
             "RLRA_AddObject(" + object.objectId + ")");
    if (RLRA_GetObjectCount(candidate.get()) != objectIndex + 1) {
      throw std::runtime_error("RLR object index drift while uploading " +
                               object.objectId);
    }
    checkRlr(RLRA_SetObjectPosition(candidate.get(), objectIndex,
                                    object.position.data()),
             "RLRA_SetObjectPosition(" + object.objectId + ")");
    checkRlr(RLRA_SetObjectOrientationQuaternion(
                 candidate.get(), objectIndex, object.orientationWxyz.data()),
             "RLRA_SetObjectOrientationQuaternion(" + object.objectId + ")");
    checkRlr(RLRA_AddMeshVertices(candidate.get(), object.vertices.data(),
                                  object.vertices.size() / 3),
             "RLRA_AddMeshVertices(" + object.objectId + ")");

    std::vector<std::vector<std::uint32_t>> groupedIndices(
        materialCategories.size());
    for (std::size_t triangleIndex = 0;
         triangleIndex < object.triangleMaterialIds.size(); ++triangleIndex) {
      auto& group = groupedIndices[object.triangleMaterialIds[triangleIndex]];
      group.insert(group.end(),
                   object.triangles.begin() + triangleIndex * 3,
                   object.triangles.begin() + triangleIndex * 3 + 3);
    }
    for (std::size_t materialIndex = 0;
         materialIndex < groupedIndices.size(); ++materialIndex) {
      const auto& indices = groupedIndices[materialIndex];
      if (indices.empty()) {
        continue;
      }
      checkRlr(RLRA_AddMeshIndices(candidate.get(), indices.data(),
                                   indices.size(), 3,
                                   materialCategories[materialIndex].c_str()),
               "RLRA_AddMeshIndices(" + object.objectId + ", " +
                   materialCategories[materialIndex] + ")");
      ++report.materialUploadCallCount[materialCategories[materialIndex]];
      report.materialUploadReceipts.emplace_back(makeUploadReceipt(
          object.objectId, materialCategories[materialIndex], indices));
    }
    checkRlr(RLRA_FinalizeObjectMesh(candidate.get(), objectIndex),
             "RLRA_FinalizeObjectMesh(" + object.objectId + ")");
  }
  report.expectedMaterialBlockCount = report.materialUploadReceipts.size();
  if (report.expectedMaterialBlockCount !=
      expectedMaterialCoefficients.materialBlockCount) {
    throw std::runtime_error(
        "internal material coefficient evidence does not match upload "
        "receipts");
  }

  destroyNativeContext(context_);
  context_ = candidate.release();
  sceneLoaded_ = true;
  hasSimulated_ = false;
  endpointsDirty_ = false;
  sources_.clear();
  listeners_.clear();
  materialUploadReceipts_ = report.materialUploadReceipts;
  expectedWorldGeometryCanonical_ = expectedGeometry.bytes;
  expectedWorldGeometrySha1_ = expectedGeometry.sha1;
  expectedMaterialCoefficientCanonical_ = expectedMaterialCoefficients.bytes;
  expectedMaterialCoefficientSha1_ = expectedMaterialCoefficients.sha1;
  expectedVertexCount_ = expectedGeometry.vertexCount;
  expectedTriangleCount_ = expectedGeometry.triangleCount;
  expectedMaterialBlockCount_ = report.expectedMaterialBlockCount;
  return report;
}

RLRSceneReadbackReport RLRAcousticContext::writeSceneMeshOBJ(
    const std::string& outputPath) const {
  const std::lock_guard<std::mutex> lock{mutex_};
  if (!sceneLoaded_) {
    throw std::logic_error("an acoustic scene must be loaded first");
  }
  if (outputPath.empty()) {
    throw std::invalid_argument("debug OBJ output path must not be empty");
  }

  PrivateTemporaryDirectory readbackDirectory{"avengine-rlr-readback-"};
  const std::filesystem::path privatePath =
      readbackDirectory.path("scene.obj");
  const std::string privatePathString = privatePath.string();
  checkRlr(RLRA_WriteSceneMeshOBJ(context_, privatePathString.c_str()),
           "RLRA_WriteSceneMeshOBJ");

  std::error_code permissionError;
  std::filesystem::permissions(
      privatePath, std::filesystem::perms::owner_read |
                       std::filesystem::perms::owner_write,
      std::filesystem::perm_options::replace, permissionError);
  if (permissionError) {
    throw std::runtime_error("cannot secure the private RLR scene OBJ");
  }
  std::ifstream privateInput{privatePath, std::ios::binary};
  if (!privateInput) {
    throw std::runtime_error("cannot open the private RLR scene OBJ");
  }
  std::ostringstream rawReadback;
  rawReadback << privateInput.rdbuf();
  if (privateInput.bad()) {
    throw std::runtime_error("failed while reading the private RLR scene OBJ");
  }
  const std::string readbackBytes = rawReadback.str();
  if (readbackBytes.empty()) {
    throw std::runtime_error(
        "RLRA_WriteSceneMeshOBJ did not create a non-empty file");
  }
  const ParsedSceneOBJ parsed = parseSceneOBJBytes(readbackBytes);

  std::ofstream callerOutput{outputPath,
                             std::ios::binary | std::ios::trunc};
  if (!callerOutput) {
    throw std::runtime_error("cannot open requested RLR scene OBJ path: " +
                             outputPath);
  }
  callerOutput.write(readbackBytes.data(),
                     static_cast<std::streamsize>(readbackBytes.size()));
  callerOutput.close();
  if (!callerOutput) {
    throw std::runtime_error("cannot copy verified RLR scene OBJ to: " +
                             outputPath);
  }

  RLRSceneReadbackReport report;
  report.outputPath = outputPath;
  report.vertexCount = parsed.geometry.vertexCount;
  report.triangleCount = parsed.geometry.triangleCount;
  report.materialBlockCount = parsed.materialBlockCount;
  report.materialAssignmentCount = parsed.materialAssignmentCount;
  report.expectedVertexCount = expectedVertexCount_;
  report.expectedTriangleCount = expectedTriangleCount_;
  report.expectedMaterialBlockCount = expectedMaterialBlockCount_;
  report.canonicalByteCount = parsed.geometry.bytes.size();
  report.expectedCanonicalByteCount = expectedWorldGeometryCanonical_.size();
  report.worldGeometrySha1 = parsed.geometry.sha1;
  report.expectedWorldGeometrySha1 = expectedWorldGeometrySha1_;
  report.materialCoefficientByteCount =
      parsed.materialCoefficientBytes.size();
  report.expectedMaterialCoefficientByteCount =
      expectedMaterialCoefficientCanonical_.size();
  report.materialCoefficientSha1 = parsed.materialCoefficientSha1;
  report.expectedMaterialCoefficientSha1 =
      expectedMaterialCoefficientSha1_;
  report.worldGeometryMatches =
      parsed.geometry.bytes == expectedWorldGeometryCanonical_;
  report.materialCoefficientsMatch =
      parsed.materialCoefficientBytes ==
      expectedMaterialCoefficientCanonical_;
  report.materialEvidenceMatches =
      report.materialBlockCount == report.expectedMaterialBlockCount &&
      report.materialAssignmentCount == report.expectedTriangleCount &&
      report.materialCoefficientsMatch;
  report.verificationPassed =
      report.worldGeometryMatches && report.materialEvidenceMatches;
  report.perFaceMaterialIdsAvailable = false;
  report.materialEvidenceMode =
      "exact_upload_receipt_plus_resolved_coefficient_blocks";
  return report;
}

std::vector<std::size_t> RLRAcousticContext::canonicalSourceOrder() const {
  std::vector<std::size_t> result;
  result.reserve(sources_.size());
  for (std::size_t index = 0; index < sources_.size(); ++index) {
    result.emplace_back(index);
  }
  std::sort(result.begin(), result.end(),
            [this](const std::size_t left, const std::size_t right) {
              return bytewiseIdLess(sources_[left].sourceId,
                                    sources_[right].sourceId);
            });
  return result;
}

std::vector<std::size_t> RLRAcousticContext::canonicalListenerOrder() const {
  std::vector<std::size_t> result;
  result.reserve(listeners_.size());
  for (std::size_t index = 0; index < listeners_.size(); ++index) {
    result.emplace_back(index);
  }
  std::sort(result.begin(), result.end(),
            [this](const std::size_t left, const std::size_t right) {
              return bytewiseIdLess(listeners_[left].listenerId,
                                    listeners_[right].listenerId);
            });
  return result;
}

std::size_t RLRAcousticContext::findSourceRecord(
    const std::string& sourceId) const {
  const auto iterator =
      std::find_if(sources_.begin(), sources_.end(),
                   [&sourceId](const RLRSourceReceipt& receipt) {
                     return receipt.sourceId == sourceId;
                   });
  if (iterator == sources_.end()) {
    throw std::invalid_argument("unknown source ID: " + sourceId);
  }
  return static_cast<std::size_t>(std::distance(sources_.begin(), iterator));
}

std::size_t RLRAcousticContext::findListenerRecord(
    const std::string& listenerId) const {
  const auto iterator =
      std::find_if(listeners_.begin(), listeners_.end(),
                   [&listenerId](const RLRListenerReceipt& receipt) {
                     return receipt.listenerId == listenerId;
                   });
  if (iterator == listeners_.end()) {
    throw std::invalid_argument("unknown listener ID: " + listenerId);
  }
  return static_cast<std::size_t>(
      std::distance(listeners_.begin(), iterator));
}

void RLRAcousticContext::refreshCanonicalEndpointIndices() {
  const auto sourceOrder = canonicalSourceOrder();
  for (std::size_t nativeIndex = 0; nativeIndex < sourceOrder.size();
       ++nativeIndex) {
    auto& receipt = sources_[sourceOrder[nativeIndex]];
    receipt.nativeIndex = nativeIndex;
    receipt.nativeRealized = !endpointsDirty_;
  }
  const auto listenerOrder = canonicalListenerOrder();
  for (std::size_t nativeIndex = 0; nativeIndex < listenerOrder.size();
       ++nativeIndex) {
    auto& receipt = listeners_[listenerOrder[nativeIndex]];
    receipt.nativeIndex = nativeIndex;
    receipt.nativeRealized = !endpointsDirty_;
  }
}

void RLRAcousticContext::realizeEndpoints() {
  if (!endpointsDirty_) {
    return;
  }

  try {
    checkRlr(RLRA_ClearSources(context_), "RLRA_ClearSources");
    checkRlr(RLRA_ClearListeners(context_), "RLRA_ClearListeners");

    const auto sourceOrder = canonicalSourceOrder();
    for (std::size_t nativeIndex = 0; nativeIndex < sourceOrder.size();
         ++nativeIndex) {
      const auto& source = sources_[sourceOrder[nativeIndex]];
      checkRlr(RLRA_AddSource(context_),
               "RLRA_AddSource(" + source.sourceId + ")");
      if (RLRA_GetSourceCount(context_) != nativeIndex + 1) {
        throw std::runtime_error(
            "RLR assigned an unexpected canonical source index");
      }
      checkRlr(RLRA_SetSourcePosition(context_, nativeIndex,
                                      source.position.data()),
               "RLRA_SetSourcePosition(" + source.sourceId + ")");
      checkRlr(RLRA_SetSourceRadius(context_, nativeIndex, source.radius),
               "RLRA_SetSourceRadius(" + source.sourceId + ")");
    }

    const auto listenerOrder = canonicalListenerOrder();
    for (std::size_t nativeIndex = 0; nativeIndex < listenerOrder.size();
         ++nativeIndex) {
      const auto& listener = listeners_[listenerOrder[nativeIndex]];
      RLRA_ChannelLayout layout{};
      layout.channelCount = listener.channelCount;
      layout.type = static_cast<RLRA_ChannelLayoutType>(listener.layoutType);
      checkRlr(RLRA_AddListener(context_, &layout),
               "RLRA_AddListener(" + listener.listenerId + ")");
      if (RLRA_GetListenerCount(context_) != nativeIndex + 1) {
        throw std::runtime_error(
            "RLR assigned an unexpected canonical listener index");
      }
      checkRlr(RLRA_SetListenerPosition(context_, nativeIndex,
                                        listener.position.data()),
               "RLRA_SetListenerPosition(" + listener.listenerId + ")");
      checkRlr(RLRA_SetListenerOrientationQuaternion(
                   context_, nativeIndex, listener.orientationWxyz.data()),
               "RLRA_SetListenerOrientationQuaternion(" +
                   listener.listenerId + ")");
      checkRlr(RLRA_SetListenerRadius(context_, nativeIndex, listener.radius),
               "RLRA_SetListenerRadius(" + listener.listenerId + ")");
      if (!listener.hrtfFilePath.empty()) {
        checkRlr(RLRA_SetListenerHRTF(context_, nativeIndex,
                                      listener.hrtfFilePath.c_str()),
                 "RLRA_SetListenerHRTF(" + listener.listenerId + ")");
      }
    }
  } catch (...) {
    // Leave the logical receipts intact. A later simulation can retry the
    // complete canonical realization rather than exposing a partial mapping.
    RLRA_ClearSources(context_);
    RLRA_ClearListeners(context_);
    endpointsDirty_ = true;
    hasSimulated_ = false;
    refreshCanonicalEndpointIndices();
    throw;
  }

  endpointsDirty_ = false;
  hasSimulated_ = false;
  refreshCanonicalEndpointIndices();
}

std::size_t RLRAcousticContext::addSource(
    const std::string& sourceId,
    const std::array<float, 3>& position,
    const float radius) {
  const std::lock_guard<std::mutex> lock{mutex_};
  if (!sceneLoaded_) {
    throw std::logic_error("an acoustic scene must be loaded first");
  }
  if (sourceId.empty() ||
      std::any_of(sources_.begin(), sources_.end(),
                  [&sourceId](const RLRSourceReceipt& receipt) {
                    return receipt.sourceId == sourceId;
                  })) {
    throw std::invalid_argument("source IDs must be non-empty and unique: " +
                                sourceId);
  }
  requireFinite(position, "source position");
  if (!isFinite(radius) || radius < 0.0f) {
    throw std::invalid_argument("source radius must be finite and non-negative");
  }

  RLRSourceReceipt receipt;
  receipt.sourceId = sourceId;
  receipt.position = position;
  receipt.radius = radius;
  sources_.emplace_back(std::move(receipt));
  endpointsDirty_ = true;
  hasSimulated_ = false;
  refreshCanonicalEndpointIndices();
  return sources_.back().nativeIndex;
}

std::size_t RLRAcousticContext::addListener(
    const std::string& listenerId,
    const std::array<float, 3>& position,
    const std::array<float, 4>& orientationWxyz,
    const RLRChannelLayoutType layoutType,
    const std::size_t channelCount,
    const float radius,
    const std::string& hrtfFilePath) {
  const std::lock_guard<std::mutex> lock{mutex_};
  if (!sceneLoaded_) {
    throw std::logic_error("an acoustic scene must be loaded first");
  }
  if (listenerId.empty() ||
      std::any_of(listeners_.begin(), listeners_.end(),
                  [&listenerId](const RLRListenerReceipt& receipt) {
                    return receipt.listenerId == listenerId;
                  })) {
    throw std::invalid_argument(
        "listener IDs must be non-empty and unique: " + listenerId);
  }
  requireFinite(position, "listener position");
  validateQuaternion(orientationWxyz, "listener orientation_wxyz");
  if (!isFinite(radius) || radius < 0.0f) {
    throw std::invalid_argument(
        "listener radius must be finite and non-negative");
  }
  switch (layoutType) {
    case RLRChannelLayoutType::Mono:
      if (channelCount != 1) {
        throw std::invalid_argument("mono listeners must have one channel");
      }
      break;
    case RLRChannelLayoutType::Binaural:
      if (channelCount != 2) {
        throw std::invalid_argument(
            "binaural listeners must have exactly two channels");
      }
      break;
    case RLRChannelLayoutType::Ambisonics: {
      const std::size_t root = static_cast<std::size_t>(
          std::llround(std::sqrt(static_cast<double>(channelCount))));
      if (root == 0 || root * root != channelCount || root - 1 > 5) {
        throw std::invalid_argument(
            "ambisonic channel count must encode an order in [0, 5]");
      }
      if (!hrtfFilePath.empty()) {
        throw std::invalid_argument(
            "an HRTF file requires a binaural listener layout");
      }
      break;
    }
    default:
      throw std::invalid_argument("unknown listener channel layout type");
  }
  if (!hrtfFilePath.empty() &&
      layoutType != RLRChannelLayoutType::Binaural) {
    throw std::invalid_argument(
        "an HRTF file requires a binaural listener layout");
  }

  RLRListenerReceipt receipt;
  receipt.listenerId = listenerId;
  receipt.position = position;
  receipt.orientationWxyz = orientationWxyz;
  receipt.layoutType = layoutType;
  receipt.channelCount = channelCount;
  receipt.radius = radius;
  receipt.hrtfFilePath = hrtfFilePath;
  listeners_.emplace_back(std::move(receipt));
  endpointsDirty_ = true;
  hasSimulated_ = false;
  refreshCanonicalEndpointIndices();
  return listeners_.back().nativeIndex;
}

void RLRAcousticContext::setSourcePosition(
    const std::string& sourceId,
    const std::array<float, 3>& position) {
  const std::lock_guard<std::mutex> lock{mutex_};
  if (!sceneLoaded_) {
    throw std::logic_error("an acoustic scene must be loaded first");
  }
  requireFinite(position, "source position");
  auto& source = sources_[findSourceRecord(sourceId)];
  if (!endpointsDirty_) {
    try {
      checkRlr(RLRA_SetSourcePosition(context_, source.nativeIndex,
                                      position.data()),
               "RLRA_SetSourcePosition(" + sourceId + ")");
    } catch (...) {
      endpointsDirty_ = true;
      hasSimulated_ = false;
      refreshCanonicalEndpointIndices();
      throw;
    }
  }
  source.position = position;
  hasSimulated_ = false;
}

void RLRAcousticContext::setSourceRadius(const std::string& sourceId,
                                         const float radius) {
  const std::lock_guard<std::mutex> lock{mutex_};
  if (!sceneLoaded_) {
    throw std::logic_error("an acoustic scene must be loaded first");
  }
  if (!isFinite(radius) || radius < 0.0f) {
    throw std::invalid_argument("source radius must be finite and non-negative");
  }
  auto& source = sources_[findSourceRecord(sourceId)];
  if (!endpointsDirty_) {
    try {
      checkRlr(RLRA_SetSourceRadius(context_, source.nativeIndex, radius),
               "RLRA_SetSourceRadius(" + sourceId + ")");
    } catch (...) {
      endpointsDirty_ = true;
      hasSimulated_ = false;
      refreshCanonicalEndpointIndices();
      throw;
    }
  }
  source.radius = radius;
  hasSimulated_ = false;
}

void RLRAcousticContext::setListenerPose(
    const std::string& listenerId,
    const std::array<float, 3>& position,
    const std::array<float, 4>& orientationWxyz) {
  const std::lock_guard<std::mutex> lock{mutex_};
  if (!sceneLoaded_) {
    throw std::logic_error("an acoustic scene must be loaded first");
  }
  requireFinite(position, "listener position");
  validateQuaternion(orientationWxyz, "listener orientation_wxyz");
  auto& listener = listeners_[findListenerRecord(listenerId)];
  if (!endpointsDirty_) {
    try {
      checkRlr(RLRA_SetListenerPosition(context_, listener.nativeIndex,
                                        position.data()),
               "RLRA_SetListenerPosition(" + listenerId + ")");
      checkRlr(RLRA_SetListenerOrientationQuaternion(
                   context_, listener.nativeIndex, orientationWxyz.data()),
               "RLRA_SetListenerOrientationQuaternion(" + listenerId + ")");
    } catch (...) {
      endpointsDirty_ = true;
      hasSimulated_ = false;
      refreshCanonicalEndpointIndices();
      throw;
    }
  }
  listener.position = position;
  listener.orientationWxyz = orientationWxyz;
  hasSimulated_ = false;
}

void RLRAcousticContext::setListenerRadius(const std::string& listenerId,
                                           const float radius) {
  const std::lock_guard<std::mutex> lock{mutex_};
  if (!sceneLoaded_) {
    throw std::logic_error("an acoustic scene must be loaded first");
  }
  if (!isFinite(radius) || radius < 0.0f) {
    throw std::invalid_argument(
        "listener radius must be finite and non-negative");
  }
  auto& listener = listeners_[findListenerRecord(listenerId)];
  if (!endpointsDirty_) {
    try {
      checkRlr(RLRA_SetListenerRadius(context_, listener.nativeIndex, radius),
               "RLRA_SetListenerRadius(" + listenerId + ")");
    } catch (...) {
      endpointsDirty_ = true;
      hasSimulated_ = false;
      refreshCanonicalEndpointIndices();
      throw;
    }
  }
  listener.radius = radius;
  hasSimulated_ = false;
}

std::vector<RLROwnedIR> RLRAcousticContext::simulateOwned() {
  const std::lock_guard<std::mutex> lock{mutex_};
  if (!sceneLoaded_) {
    throw std::logic_error("an acoustic scene must be loaded first");
  }
  if (sources_.empty() || listeners_.empty()) {
    throw std::logic_error(
        "at least one named source and listener are required");
  }
  realizeEndpoints();
  hasSimulated_ = false;
  checkRlr(RLRA_Simulate(context_), "RLRA_Simulate");

  const std::size_t expectedCount = sources_.size() * listeners_.size();
  if (RLRA_GetIRCount(context_) != expectedCount) {
    throw std::runtime_error("RLR returned an unexpected IR pair count");
  }
  std::vector<RLROwnedIR> result;
  result.reserve(expectedCount);
  const auto listenerOrder = canonicalListenerOrder();
  const auto sourceOrder = canonicalSourceOrder();
  for (std::size_t listenerIndex = 0; listenerIndex < listenerOrder.size();
       ++listenerIndex) {
    const auto& listener = listeners_[listenerOrder[listenerIndex]];
    for (std::size_t sourceIndex = 0; sourceIndex < sourceOrder.size();
         ++sourceIndex) {
      const auto& source = sources_[sourceOrder[sourceIndex]];
      RLROwnedIR ir;
      ir.listenerId = listener.listenerId;
      ir.sourceId = source.sourceId;
      ir.listenerNativeIndex = listenerIndex;
      ir.sourceNativeIndex = sourceIndex;
      ir.sampleRate = configuration_.sampleRate;
      ir.channelCount =
          RLRA_GetIRChannelCount(context_, listenerIndex, sourceIndex);
      ir.sampleCount =
          RLRA_GetIRSampleCount(context_, listenerIndex, sourceIndex);
      if (ir.channelCount != listener.channelCount) {
        throw std::runtime_error(
            "RLR returned a channel count that does not match listener '" +
            listener.listenerId + "'");
      }
      if (ir.channelCount == 0 || ir.sampleCount == 0) {
        throw std::runtime_error("RLR returned an empty IR for listener '" +
                                 ir.listenerId + "' and source '" +
                                 ir.sourceId + "'");
      }
      ir.layoutType = listener.layoutType;
      ir.channelOrdering =
          listener.layoutType == RLRChannelLayoutType::Ambisonics ? "ACN"
                                                                  : "native";
      ir.normalization =
          listener.layoutType == RLRChannelLayoutType::Ambisonics ? "N3D"
                                                                  : "not_applicable";
      ir.coordinateFrame =
          listener.layoutType == RLRChannelLayoutType::Ambisonics
              ? "avengine_world"
              : (listener.layoutType == RLRChannelLayoutType::Binaural
                     ? "listener_local"
                     : "not_directional");
      if (listener.layoutType == RLRChannelLayoutType::Ambisonics) {
        ir.ambisonicOrder = static_cast<std::size_t>(
            std::llround(std::sqrt(static_cast<double>(ir.channelCount)))) -
                            1;
        ir.layoutId = ir.ambisonicOrder == 1
                          ? "rlr_foa_acn_n3d_world_v1"
                          : "rlr_ambisonics_acn_n3d_world_order" +
                                std::to_string(ir.ambisonicOrder) + "_v1";
      } else if (listener.layoutType == RLRChannelLayoutType::Binaural) {
        ir.layoutId = "rlr_binaural_lr_v1";
      } else {
        ir.layoutId = "rlr_mono_v1";
      }
      ir.channelLabels = channelLabels(listener.layoutType, ir.channelCount);
      ir.samples.reserve(ir.channelCount * ir.sampleCount);
      for (std::size_t channelIndex = 0; channelIndex < ir.channelCount;
           ++channelIndex) {
        const float* channel = RLRA_GetIRChannel(
            context_, listenerIndex, sourceIndex, channelIndex);
        if (!channel) {
          throw std::runtime_error("RLR returned a null IR channel pointer");
        }
        if (!std::all_of(channel, channel + ir.sampleCount, isFinite)) {
          throw std::runtime_error("RLR returned non-finite IR samples");
        }
        ir.samples.insert(ir.samples.end(), channel,
                          channel + ir.sampleCount);
      }
      result.emplace_back(std::move(ir));
    }
  }
  hasSimulated_ = true;
  return result;
}

RLRRayResult RLRAcousticContext::traceRayAnyHit(
    const std::array<float, 3>& origin,
    const std::array<float, 3>& direction,
    const float minDistance,
    const float maxDistance) const {
  const std::lock_guard<std::mutex> lock{mutex_};
  if (!hasSimulated_) {
    throw std::logic_error("ray queries require a completed simulation");
  }
  RLRA_Ray ray = makeRay(origin, direction, minDistance, maxDistance);
  checkRlr(RLRA_TraceRayAnyHit(context_, &ray), "RLRA_TraceRayAnyHit");
  RLRRayResult result;
  result.hit = ray.hit == RLRA_RayHit_True;
  result.hasHitDetails = false;
  result.distance = 0.0f;
  return result;
}

RLRRayResult RLRAcousticContext::traceRayFirstHit(
    const std::array<float, 3>& origin,
    const std::array<float, 3>& direction,
    const float minDistance,
    const float maxDistance) const {
  const std::lock_guard<std::mutex> lock{mutex_};
  if (!hasSimulated_) {
    throw std::logic_error("ray queries require a completed simulation");
  }
  RLRA_Ray ray = makeRay(origin, direction, minDistance, maxDistance);
  checkRlr(RLRA_TraceRayFirstHit(context_, &ray),
           "RLRA_TraceRayFirstHit");
  RLRRayResult result;
  result.hit = ray.hit == RLRA_RayHit_True;
  result.hasHitDetails = result.hit;
  if (result.hit) {
    result.distance = ray.tMax;
    std::copy(std::begin(ray.normal), std::end(ray.normal),
              result.normal.begin());
    if (!isFinite(result.distance) ||
        !std::all_of(result.normal.begin(), result.normal.end(), isFinite) ||
        !std::any_of(result.normal.begin(), result.normal.end(),
                     [](const float value) { return value != 0.0f; })) {
      throw std::runtime_error(
          "RLR returned non-finite or zero first-hit details");
    }
  }
  return result;
}

float RLRAcousticContext::indirectRayEfficiency() const {
  const std::lock_guard<std::mutex> lock{mutex_};
  if (!hasSimulated_) {
    throw std::logic_error(
        "indirect ray efficiency requires a completed simulation");
  }
  const float result = RLRA_GetIndirectRayEfficiency(context_);
  if (!isFinite(result) || result < 0.0f || result > 1.0f) {
    throw std::runtime_error("RLR returned invalid indirect ray efficiency");
  }
  return result;
}

std::vector<std::string> RLRAcousticContext::writeIRMetrics(
    const std::string& listenerId,
    const std::string& sourceId,
    const std::string& outputPathPrefix) const {
  const std::lock_guard<std::mutex> lock{mutex_};
  if (!hasSimulated_) {
    throw std::logic_error("IR metrics require a completed simulation");
  }
  if (outputPathPrefix.empty()) {
    throw std::invalid_argument("IR metrics output prefix must not be empty");
  }
  const std::size_t listenerIndex =
      listeners_[findListenerRecord(listenerId)].nativeIndex;
  const std::size_t sourceIndex =
      sources_[findSourceRecord(sourceId)].nativeIndex;
  checkRlr(RLRA_WriteIRMetrics(context_, listenerIndex, sourceIndex,
                               outputPathPrefix.c_str()),
           "RLRA_WriteIRMetrics");

  const std::size_t channelCount =
      RLRA_GetIRChannelCount(context_, listenerIndex, sourceIndex);
  std::vector<std::string> result;
  result.reserve(channelCount);
  for (std::size_t channelIndex = 0; channelIndex < channelCount;
       ++channelIndex) {
    const std::string metricsPath =
        outputPathPrefix + " ch" + std::to_string(channelIndex) +
        " metrics.txt";
    std::ifstream output{metricsPath, std::ios::binary | std::ios::ate};
    if (!output || output.tellg() <= 0) {
      throw std::runtime_error(
          "RLRA_WriteIRMetrics did not create a non-empty per-channel file");
    }
    result.emplace_back(metricsPath);
  }
  return result;
}

void RLRAcousticContext::reset() {
  const std::lock_guard<std::mutex> lock{mutex_};
  NativeContextGuard candidate{createNativeContext()};
  destroyNativeContext(context_);
  context_ = candidate.release();
  sceneLoaded_ = false;
  hasSimulated_ = false;
  endpointsDirty_ = false;
  sources_.clear();
  listeners_.clear();
  materialUploadReceipts_.clear();
  expectedWorldGeometryCanonical_.clear();
  expectedWorldGeometrySha1_.clear();
  expectedMaterialCoefficientCanonical_.clear();
  expectedMaterialCoefficientSha1_.clear();
  expectedVertexCount_ = 0;
  expectedTriangleCount_ = 0;
  expectedMaterialBlockCount_ = 0;
}

bool RLRAcousticContext::sceneLoaded() const {
  const std::lock_guard<std::mutex> lock{mutex_};
  return sceneLoaded_;
}

std::size_t RLRAcousticContext::objectCount() const {
  const std::lock_guard<std::mutex> lock{mutex_};
  return context_ ? RLRA_GetObjectCount(context_) : 0;
}

std::size_t RLRAcousticContext::sourceCount() const {
  const std::lock_guard<std::mutex> lock{mutex_};
  return sources_.size();
}

std::size_t RLRAcousticContext::listenerCount() const {
  const std::lock_guard<std::mutex> lock{mutex_};
  return listeners_.size();
}

std::size_t RLRAcousticContext::sourceIndex(
    const std::string& sourceId) const {
  const std::lock_guard<std::mutex> lock{mutex_};
  return sources_[findSourceRecord(sourceId)].nativeIndex;
}

std::size_t RLRAcousticContext::listenerIndex(
    const std::string& listenerId) const {
  const std::lock_guard<std::mutex> lock{mutex_};
  return listeners_[findListenerRecord(listenerId)].nativeIndex;
}

std::vector<RLRSourceReceipt> RLRAcousticContext::sourceReceipts() const {
  const std::lock_guard<std::mutex> lock{mutex_};
  std::vector<RLRSourceReceipt> result;
  result.reserve(sources_.size());
  for (const std::size_t recordIndex : canonicalSourceOrder()) {
    result.emplace_back(sources_[recordIndex]);
  }
  return result;
}

std::vector<RLRListenerReceipt> RLRAcousticContext::listenerReceipts() const {
  const std::lock_guard<std::mutex> lock{mutex_};
  std::vector<RLRListenerReceipt> result;
  result.reserve(listeners_.size());
  for (const std::size_t recordIndex : canonicalListenerOrder()) {
    result.emplace_back(listeners_[recordIndex]);
  }
  return result;
}

RLRContextConfiguration RLRAcousticContext::configuration() const {
  const std::lock_guard<std::mutex> lock{mutex_};
  return configuration_;
}

std::vector<RLRMaterialUploadReceipt>
RLRAcousticContext::materialUploadReceipts() const {
  const std::lock_guard<std::mutex> lock{mutex_};
  return materialUploadReceipts_;
}

}  // namespace audio
}  // namespace esp
