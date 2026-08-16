// Copyright (c) Meta Platforms, Inc. and its affiliates.
// Modifications Copyright (c) AVEngine contributors.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "esp/bindings/Bindings.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstring>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "esp/sensor/configure.h"

#ifdef ESP_BUILD_WITH_RLR_ADAPTER
#include "esp/audio/RLRAcousticContext.h"
#endif

namespace py = pybind11;
using py::literals::operator""_a;

namespace esp {
namespace audio {

#ifdef ESP_BUILD_WITH_RLR_ADAPTER
namespace {

template <class Function>
auto callWithPythonExceptionTranslation(Function&& function)
    -> decltype(function()) {
  try {
    return function();
  } catch (const py::builtin_exception& error) {
    error.set_error();
    throw py::error_already_set{};
  } catch (const std::invalid_argument& error) {
    PyErr_SetString(PyExc_ValueError, error.what());
    throw py::error_already_set{};
  } catch (const std::exception& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
    throw py::error_already_set{};
  }
}

template <class T>
std::vector<T> copyExactArray(const py::handle value,
                              const std::string& label,
                              const int dimensions,
                              const py::ssize_t trailingSize) {
  if (!py::isinstance<py::array>(value)) {
    throw py::type_error(label + " must be a NumPy array");
  }
  const py::array array = py::reinterpret_borrow<py::array>(value);
  if (!array.dtype().is(py::dtype::of<T>())) {
    throw py::type_error(label + " has the wrong dtype");
  }
  if ((array.flags() & py::array::c_style) == 0) {
    throw py::value_error(label + " must be C-contiguous");
  }
  if (array.ndim() != dimensions ||
      (dimensions > 1 && array.shape(dimensions - 1) != trailingSize)) {
    throw py::value_error(label + " has the wrong shape");
  }

  const py::buffer_info info = array.request();
  const auto* begin = static_cast<const T*>(info.ptr);
  return std::vector<T>(begin, begin + array.size());
}

std::vector<RLRMeshObject> parseMeshObjects(const py::iterable& values) {
  const std::set<std::string> expectedKeys{
      "object_id", "vertices", "triangles", "triangle_material_ids",
      "position", "orientation_wxyz"};
  std::vector<RLRMeshObject> result;
  for (const py::handle value : values) {
    if (!py::isinstance<py::dict>(value)) {
      throw py::type_error("every acoustic object must be a dict");
    }
    const py::dict object = py::reinterpret_borrow<py::dict>(value);
    std::set<std::string> actualKeys;
    for (const auto& item : object) {
      if (!py::isinstance<py::str>(item.first)) {
        throw py::type_error("acoustic object keys must be strings");
      }
      actualKeys.emplace(py::cast<std::string>(item.first));
    }
    if (actualKeys != expectedKeys) {
      throw py::value_error(
          "acoustic object keys must be exactly: object_id, vertices, "
          "triangles, triangle_material_ids, position, orientation_wxyz");
    }

    RLRMeshObject parsed;
    parsed.objectId = py::cast<std::string>(object["object_id"]);
    parsed.vertices =
        copyExactArray<float>(object["vertices"], "vertices", 2, 3);
    parsed.triangles = copyExactArray<std::uint32_t>(
        object["triangles"], "triangles", 2, 3);
    parsed.triangleMaterialIds = copyExactArray<std::uint32_t>(
        object["triangle_material_ids"], "triangle_material_ids", 1, 0);
    parsed.position =
        py::cast<std::array<float, 3>>(object["position"]);
    parsed.orientationWxyz =
        py::cast<std::array<float, 4>>(object["orientation_wxyz"]);
    result.emplace_back(std::move(parsed));
  }
  return result;
}

py::array_t<float> ownedIRSamples(const RLROwnedIR& ir) {
  if (ir.samples.size() != ir.channelCount * ir.sampleCount) {
    throw std::runtime_error("owned IR shape does not match its sample storage");
  }
  py::array_t<float> result{
      {static_cast<py::ssize_t>(ir.channelCount),
       static_cast<py::ssize_t>(ir.sampleCount)}};
  std::memcpy(result.mutable_data(), ir.samples.data(),
              ir.samples.size() * sizeof(float));
  return result;
}

}  // namespace
#endif

void initAudioPropagationBindings(pybind11::module& module) {
#ifdef ESP_BUILD_WITH_RLR_ADAPTER
  py::class_<RLRContextConfiguration>(module, "RLRContextConfiguration")
      .def(py::init<>())
      .def_readwrite("frequency_bands",
                     &RLRContextConfiguration::frequencyBands)
      .def_readwrite("direct_sh_order",
                     &RLRContextConfiguration::directSHOrder)
      .def_readwrite("indirect_sh_order",
                     &RLRContextConfiguration::indirectSHOrder)
      .def_readwrite("direct_ray_count",
                     &RLRContextConfiguration::directRayCount)
      .def_readwrite("indirect_ray_count",
                     &RLRContextConfiguration::indirectRayCount)
      .def_readwrite("indirect_ray_depth",
                     &RLRContextConfiguration::indirectRayDepth)
      .def_readwrite("source_ray_count",
                     &RLRContextConfiguration::sourceRayCount)
      .def_readwrite("source_ray_depth",
                     &RLRContextConfiguration::sourceRayDepth)
      .def_readwrite("max_diffraction_order",
                     &RLRContextConfiguration::maxDiffractionOrder)
      .def_readwrite("thread_count", &RLRContextConfiguration::threadCount)
      .def_readwrite("sample_rate", &RLRContextConfiguration::sampleRate)
      .def_readwrite("max_ir_length", &RLRContextConfiguration::maxIRLength)
      .def_readwrite("unit_scale", &RLRContextConfiguration::unitScale)
      .def_readwrite("global_volume", &RLRContextConfiguration::globalVolume)
      .def_readwrite("hrtf_right", &RLRContextConfiguration::hrtfRight)
      .def_readwrite("hrtf_up", &RLRContextConfiguration::hrtfUp)
      .def_readwrite("hrtf_back", &RLRContextConfiguration::hrtfBack)
      .def_readwrite("direct", &RLRContextConfiguration::direct)
      .def_readwrite("indirect", &RLRContextConfiguration::indirect)
      .def_readwrite("diffraction", &RLRContextConfiguration::diffraction)
      .def_readwrite("transmission", &RLRContextConfiguration::transmission)
      .def_readwrite("mesh_simplification",
                     &RLRContextConfiguration::meshSimplification)
      .def_readwrite("temporal_coherence",
                     &RLRContextConfiguration::temporalCoherence);

  py::enum_<RLRChannelLayoutType>(module, "RLRChannelLayoutType")
      .value("Mono", RLRChannelLayoutType::Mono)
      .value("Binaural", RLRChannelLayoutType::Binaural)
      .value("Ambisonics", RLRChannelLayoutType::Ambisonics);

  py::class_<RLRSourceReceipt>(module, "RLRSourceReceipt")
      .def_readonly("source_id", &RLRSourceReceipt::sourceId)
      .def_readonly("position", &RLRSourceReceipt::position)
      .def_readonly("radius", &RLRSourceReceipt::radius)
      .def_readonly("native_index", &RLRSourceReceipt::nativeIndex)
      .def_readonly("native_realized", &RLRSourceReceipt::nativeRealized);

  py::class_<RLRListenerReceipt>(module, "RLRListenerReceipt")
      .def_readonly("listener_id", &RLRListenerReceipt::listenerId)
      .def_readonly("position", &RLRListenerReceipt::position)
      .def_readonly("orientation_wxyz",
                    &RLRListenerReceipt::orientationWxyz)
      .def_readonly("layout_type", &RLRListenerReceipt::layoutType)
      .def_readonly("channel_count", &RLRListenerReceipt::channelCount)
      .def_readonly("radius", &RLRListenerReceipt::radius)
      .def_readonly("hrtf_file_path", &RLRListenerReceipt::hrtfFilePath)
      .def_readonly("native_index", &RLRListenerReceipt::nativeIndex)
      .def_readonly("native_realized", &RLRListenerReceipt::nativeRealized);

  py::class_<RLRMaterialUploadReceipt>(module, "RLRMaterialUploadReceipt")
      .def_readonly("object_id", &RLRMaterialUploadReceipt::objectId)
      .def_readonly("material_category",
                    &RLRMaterialUploadReceipt::materialCategory)
      .def_readonly("triangle_count",
                    &RLRMaterialUploadReceipt::triangleCount)
      .def_readonly("index_count", &RLRMaterialUploadReceipt::indexCount)
      .def_readonly("canonical_payload_byte_count",
                    &RLRMaterialUploadReceipt::canonicalPayloadByteCount)
      .def_readonly("canonical_payload_sha1",
                    &RLRMaterialUploadReceipt::canonicalPayloadSha1);

  py::class_<RLRSceneUploadReport>(module, "RLRSceneUploadReport")
      .def_readonly("object_count", &RLRSceneUploadReport::objectCount)
      .def_readonly("vertex_count", &RLRSceneUploadReport::vertexCount)
      .def_readonly("triangle_count", &RLRSceneUploadReport::triangleCount)
      .def_readonly("material_category_count",
                    &RLRSceneUploadReport::materialCategoryCount)
      .def_readonly("object_ids", &RLRSceneUploadReport::objectIds)
      .def_readonly("triangle_count_by_material",
                    &RLRSceneUploadReport::triangleCountByMaterial)
      .def_readonly("material_upload_call_count",
                    &RLRSceneUploadReport::materialUploadCallCount)
      .def_property_readonly(
          "resolved_material_name_by_category",
          [](const RLRSceneUploadReport& report) {
            py::dict result;
            for (const auto& item : report.resolvedMaterialNameByCategory) {
              result[py::str(item.first)] = py::str(item.second);
            }
            return result;
          })
      .def_readonly("resolved_material_index_by_category",
                    &RLRSceneUploadReport::resolvedMaterialIndexByCategory)
      .def_readonly("material_upload_receipts",
                    &RLRSceneUploadReport::materialUploadReceipts)
      .def_readonly("expected_material_block_count",
                    &RLRSceneUploadReport::expectedMaterialBlockCount)
      .def_readonly("material_database_sha1",
                    &RLRSceneUploadReport::materialDatabaseSha1)
      .def_readonly("expected_world_geometry_sha1",
                    &RLRSceneUploadReport::expectedWorldGeometrySha1)
      .def_readonly("expected_canonical_byte_count",
                    &RLRSceneUploadReport::expectedCanonicalByteCount)
      .def_readonly("expected_material_coefficient_sha1",
                    &RLRSceneUploadReport::expectedMaterialCoefficientSha1)
      .def_readonly(
          "expected_material_coefficient_byte_count",
          &RLRSceneUploadReport::expectedMaterialCoefficientByteCount);

  py::class_<RLRSceneReadbackReport>(module, "RLRSceneReadbackReport")
      .def_readonly("output_path", &RLRSceneReadbackReport::outputPath)
      .def_readonly("vertex_count", &RLRSceneReadbackReport::vertexCount)
      .def_readonly("triangle_count", &RLRSceneReadbackReport::triangleCount)
      .def_readonly("material_block_count",
                    &RLRSceneReadbackReport::materialBlockCount)
      .def_readonly("material_assignment_count",
                    &RLRSceneReadbackReport::materialAssignmentCount)
      .def_readonly("expected_vertex_count",
                    &RLRSceneReadbackReport::expectedVertexCount)
      .def_readonly("expected_triangle_count",
                    &RLRSceneReadbackReport::expectedTriangleCount)
      .def_readonly("expected_material_block_count",
                    &RLRSceneReadbackReport::expectedMaterialBlockCount)
      .def_readonly("canonical_byte_count",
                    &RLRSceneReadbackReport::canonicalByteCount)
      .def_readonly("expected_canonical_byte_count",
                    &RLRSceneReadbackReport::expectedCanonicalByteCount)
      .def_readonly("world_geometry_sha1",
                    &RLRSceneReadbackReport::worldGeometrySha1)
      .def_readonly("expected_world_geometry_sha1",
                    &RLRSceneReadbackReport::expectedWorldGeometrySha1)
      .def_readonly("material_coefficient_byte_count",
                    &RLRSceneReadbackReport::materialCoefficientByteCount)
      .def_readonly(
          "expected_material_coefficient_byte_count",
          &RLRSceneReadbackReport::expectedMaterialCoefficientByteCount)
      .def_readonly("material_coefficient_sha1",
                    &RLRSceneReadbackReport::materialCoefficientSha1)
      .def_readonly("expected_material_coefficient_sha1",
                    &RLRSceneReadbackReport::expectedMaterialCoefficientSha1)
      .def_readonly("world_geometry_matches",
                    &RLRSceneReadbackReport::worldGeometryMatches)
      .def_readonly("material_coefficients_match",
                    &RLRSceneReadbackReport::materialCoefficientsMatch)
      .def_readonly("material_evidence_matches",
                    &RLRSceneReadbackReport::materialEvidenceMatches)
      .def_readonly("verification_passed",
                    &RLRSceneReadbackReport::verificationPassed)
      .def_readonly("per_face_material_ids_available",
                    &RLRSceneReadbackReport::perFaceMaterialIdsAvailable)
      .def_readonly("material_evidence_mode",
                    &RLRSceneReadbackReport::materialEvidenceMode);

  py::class_<RLROwnedIR>(module, "RLROwnedIR")
      .def_readonly("listener_id", &RLROwnedIR::listenerId)
      .def_readonly("source_id", &RLROwnedIR::sourceId)
      .def_readonly("listener_native_index",
                    &RLROwnedIR::listenerNativeIndex)
      .def_readonly("source_native_index", &RLROwnedIR::sourceNativeIndex)
      .def_readonly("sample_rate", &RLROwnedIR::sampleRate)
      .def_readonly("channel_count", &RLROwnedIR::channelCount)
      .def_readonly("sample_count", &RLROwnedIR::sampleCount)
      .def_readonly("layout_type", &RLROwnedIR::layoutType)
      .def_readonly("ambisonic_order", &RLROwnedIR::ambisonicOrder)
      .def_readonly("channel_ordering", &RLROwnedIR::channelOrdering)
      .def_readonly("normalization", &RLROwnedIR::normalization)
      .def_readonly("coordinate_frame", &RLROwnedIR::coordinateFrame)
      .def_readonly("layout_id", &RLROwnedIR::layoutId)
      .def_readonly("channel_labels", &RLROwnedIR::channelLabels)
      .def_property_readonly("samples", [](const RLROwnedIR& ir) {
        return callWithPythonExceptionTranslation(
            [&]() { return ownedIRSamples(ir); });
      });

  py::class_<RLRRayResult>(module, "RLRRayResult")
      .def_readonly("hit", &RLRRayResult::hit)
      .def_readonly("has_hit_details", &RLRRayResult::hasHitDetails)
      .def_readonly("distance", &RLRRayResult::distance)
      .def_readonly("normal", &RLRRayResult::normal);

  py::class_<RLRAcousticContext>(module, "RLRAcousticContext")
      .def(py::init([](const RLRContextConfiguration& configuration) {
             return callWithPythonExceptionTranslation([&]() {
               return std::make_unique<RLRAcousticContext>(configuration);
             });
           }),
           "configuration"_a = RLRContextConfiguration{})
      .def(
          "load_acoustic_scene",
          [](RLRAcousticContext& self,
             const std::string& materialDatabaseJson,
             const std::vector<std::string>& materialCategories,
             const py::iterable& objects) {
            return callWithPythonExceptionTranslation([&]() {
              std::vector<RLRMeshObject> parsed = parseMeshObjects(objects);
              py::gil_scoped_release release;
              return self.loadAcousticScene(materialDatabaseJson,
                                            materialCategories, parsed);
            });
          },
          "material_database_json"_a, "material_categories"_a, "objects"_a)
      .def(
          "write_scene_mesh_obj",
          [](const RLRAcousticContext& self, const std::string& outputPath) {
            return callWithPythonExceptionTranslation([&]() {
              py::gil_scoped_release release;
              return self.writeSceneMeshOBJ(outputPath);
            });
          },
          "output_path"_a)
      .def(
          "add_source",
          [](RLRAcousticContext& self, const std::string& sourceId,
             const std::array<float, 3>& position, const float radius) {
            return callWithPythonExceptionTranslation(
                [&]() { return self.addSource(sourceId, position, radius); });
          },
          "source_id"_a, "position"_a, "radius"_a = 0.0f)
      .def(
          "add_listener",
          [](RLRAcousticContext& self, const std::string& listenerId,
             const std::array<float, 3>& position,
             const std::array<float, 4>& orientationWxyz,
             const RLRChannelLayoutType layoutType,
             const std::size_t channelCount, const float radius,
             const std::string& hrtfFilePath) {
            return callWithPythonExceptionTranslation([&]() {
              return self.addListener(listenerId, position, orientationWxyz,
                                      layoutType, channelCount, radius,
                                      hrtfFilePath);
            });
          },
          "listener_id"_a, "position"_a, "orientation_wxyz"_a,
          "layout_type"_a = RLRChannelLayoutType::Mono,
          "channel_count"_a = 1, "radius"_a = 0.1f,
          "hrtf_file_path"_a = "")
      .def(
          "set_source_position",
          [](RLRAcousticContext& self, const std::string& sourceId,
             const std::array<float, 3>& position) {
            callWithPythonExceptionTranslation(
                [&]() { self.setSourcePosition(sourceId, position); });
          },
          "source_id"_a, "position"_a)
      .def(
          "set_source_radius",
          [](RLRAcousticContext& self, const std::string& sourceId,
             const float radius) {
            callWithPythonExceptionTranslation(
                [&]() { self.setSourceRadius(sourceId, radius); });
          },
          "source_id"_a, "radius"_a)
      .def(
          "set_listener_pose",
          [](RLRAcousticContext& self, const std::string& listenerId,
             const std::array<float, 3>& position,
             const std::array<float, 4>& orientationWxyz) {
            callWithPythonExceptionTranslation([&]() {
              self.setListenerPose(listenerId, position, orientationWxyz);
            });
          },
          "listener_id"_a, "position"_a, "orientation_wxyz"_a)
      .def(
          "set_listener_radius",
          [](RLRAcousticContext& self, const std::string& listenerId,
             const float radius) {
            callWithPythonExceptionTranslation(
                [&]() { self.setListenerRadius(listenerId, radius); });
          },
          "listener_id"_a, "radius"_a)
      .def(
          "source_index",
          [](const RLRAcousticContext& self, const std::string& sourceId) {
            return callWithPythonExceptionTranslation(
                [&]() { return self.sourceIndex(sourceId); });
          },
          "source_id"_a)
      .def(
          "listener_index",
          [](const RLRAcousticContext& self, const std::string& listenerId) {
            return callWithPythonExceptionTranslation(
                [&]() { return self.listenerIndex(listenerId); });
          },
          "listener_id"_a)
      .def("source_registration_receipts",
           [](const RLRAcousticContext& self) {
             return callWithPythonExceptionTranslation(
                 [&]() { return self.sourceReceipts(); });
           })
      .def("listener_registration_receipts",
           [](const RLRAcousticContext& self) {
             return callWithPythonExceptionTranslation(
                 [&]() { return self.listenerReceipts(); });
           })
      .def("simulate_owned", [](RLRAcousticContext& self) {
        return callWithPythonExceptionTranslation([&]() {
          py::gil_scoped_release release;
          return self.simulateOwned();
        });
      })
      .def(
          "trace_ray_any_hit",
          [](const RLRAcousticContext& self,
             const std::array<float, 3>& origin,
             const std::array<float, 3>& direction, const float minDistance,
             const float maxDistance) {
            return callWithPythonExceptionTranslation([&]() {
              py::gil_scoped_release release;
              return self.traceRayAnyHit(origin, direction, minDistance,
                                         maxDistance);
            });
          },
          "origin"_a, "direction"_a, "min_distance"_a, "max_distance"_a)
      .def(
          "trace_ray_first_hit",
          [](const RLRAcousticContext& self,
             const std::array<float, 3>& origin,
             const std::array<float, 3>& direction, const float minDistance,
             const float maxDistance) {
            return callWithPythonExceptionTranslation([&]() {
              py::gil_scoped_release release;
              return self.traceRayFirstHit(origin, direction, minDistance,
                                           maxDistance);
            });
          },
          "origin"_a, "direction"_a, "min_distance"_a, "max_distance"_a)
      .def("indirect_ray_efficiency", [](const RLRAcousticContext& self) {
        return callWithPythonExceptionTranslation([&]() {
          py::gil_scoped_release release;
          return self.indirectRayEfficiency();
        });
      })
      .def(
          "write_ir_metrics",
          [](const RLRAcousticContext& self, const std::string& listenerId,
             const std::string& sourceId,
             const std::string& outputPathPrefix) {
            return callWithPythonExceptionTranslation([&]() {
              py::gil_scoped_release release;
              return self.writeIRMetrics(listenerId, sourceId,
                                         outputPathPrefix);
            });
          },
          "listener_id"_a, "source_id"_a, "output_path_prefix"_a)
      .def("reset", [](RLRAcousticContext& self) {
        callWithPythonExceptionTranslation([&]() { self.reset(); });
      })
      .def_property_readonly("scene_loaded", &RLRAcousticContext::sceneLoaded)
      .def_property_readonly("object_count", &RLRAcousticContext::objectCount)
      .def_property_readonly("source_count", &RLRAcousticContext::sourceCount)
      .def_property_readonly("listener_count",
                             &RLRAcousticContext::listenerCount)
      .def_property_readonly("configuration",
                             &RLRAcousticContext::configuration)
      .def_property_readonly("material_upload_receipts",
                             &RLRAcousticContext::materialUploadReceipts);
#else
  module.attr("RLRContextConfiguration") = py::none();
  module.attr("RLRChannelLayoutType") = py::none();
  module.attr("RLRSourceReceipt") = py::none();
  module.attr("RLRListenerReceipt") = py::none();
  module.attr("RLRSceneUploadReport") = py::none();
  module.attr("RLRMaterialUploadReceipt") = py::none();
  module.attr("RLRSceneReadbackReport") = py::none();
  module.attr("RLROwnedIR") = py::none();
  module.attr("RLRRayResult") = py::none();
  module.attr("RLRAcousticContext") = py::none();
#endif
}

}  // namespace audio
}  // namespace esp
