# SPEAR python_ext S3b optional native build

This directory contains the selected MIT SPEAR C++ extension source and
AVEngine-owned CMake wiring for an optional top-level avengine_spear_ext
module. It does not create a spear package or a spear_ext compatibility alias,
and it does not alter a runner or UE runtime selection.

## Origin and treatment

| Field | Recorded value |
| --- | --- |
| Upstream | spear-sim/spear@251bd5e0d3d1e7297ec072bb9b0df9ef63f864b7 |
| Relationship | adapted selected C++ source; AVEngine reimplements the build wiring and makes only the recorded module-name and standard-C++ portability adjustments |
| License | MIT; full text is retained at LICENSES/SPEAR-MIT.txt |

The selected C++ source retains the upstream RPC and nanobind interfaces.
cpp/spear_ext.cpp changes NB_MODULE(spear_ext, ...) to
NB_MODULE(avengine_spear_ext, ...) so the Python initialization symbol and the
AVEngine client import agree.

cpp/msgpack_utils.h preserves the upstream zero-copy MessagePack BIN to
std::span<uint8_t> conversion body. Its explicit specialization is placed at
namespace scope and the primary template definition has a normal closing brace,
which makes the same code valid under standard GCC C++20 rather than relying on
toolchain-specific extension treatment in the transition checkout.

## Selected source map

The retained upstream C++ files are:

- cpp/assert.h
- cpp/client.h
- cpp/func_signature_registry.h
- cpp/msgpack_adaptors.h
- cpp/msgpack_utils.h
- cpp/spear_ext.cpp
- cpp/std.h
- cpp/types.h

CMakeLists.txt is AVEngine-owned. It resolves a user-supplied installed rpclib
SDK, Python 3.11, nanobind, and Threads; it does not use an upstream SPEAR
checkout, dependency source tree, package metadata, or binary.

## Build an isolated staging module

Create distinct, empty build and installation directories first. The install
prefix must already exist and resolve outside this source directory.

    export AVENGINE_SPEAR_RPCLIB_ROOT=/path/to/installed/rpclib-prefix
    export AVENGINE_SPEAR_PYTHON=/path/to/python3.11
    build_dir=/path/to/fresh/build
    stage_dir=/path/to/fresh/stage

    cmake -S native/spear/python_ext -B "$build_dir" -G "Unix Makefiles" \
      -DCMAKE_INSTALL_PREFIX="$stage_dir" \
      -DAVENGINE_SPEAR_RPCLIB_ROOT="$AVENGINE_SPEAR_RPCLIB_ROOT" \
      -DPython_EXECUTABLE="$AVENGINE_SPEAR_PYTHON" \
      -DAVENGINE_SPEAR_NANOBIND_DIR="$("$AVENGINE_SPEAR_PYTHON" -m nanobind --cmake_dir)"
    make -C "$build_dir"
    cmake --install "$build_dir"

The external rpclib root must contain its installed CMake export below
lib/cmake/rpclib. The external nanobind directory must contain
nanobind-config.cmake; derive it from the same explicit Python interpreter.
CMake searches only those supplied roots for rpclib and nanobind, so an ambient
package or a SPEAR checkout cannot satisfy either dependency.

The resulting module is named with the active Python extension suffix under
the fresh stage directory, for example
avengine_spear_ext.cpython-311-x86_64-linux-gnu.so. Python 3.11 uses its normal
minor-version-specific extension ABI; this target does not claim stable-ABI or
cross-minor binary compatibility.

To make that top-level module discoverable, either intentionally install it to
the same interpreter's platform-library directory:

    stage_dir=$("$AVENGINE_SPEAR_PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["platlib"])')

or retain a separate development/isolated stage and set both the stage and
AVEngine source explicitly for the import:

    PYTHONPATH="$stage_dir:/path/to/AVEngine/src" "$AVENGINE_SPEAR_PYTHON" -c 'import avengine_spear_ext'

## Validation boundary

The focused native validation is deliberately host-only:

- a strict C++20 temporary test packs MessagePack BIN bytes, unpacks them, and
  verifies the zero-copy span byte count and contents;
- the extension is built and installed into a fresh prefix;
- the installed module must export PyInit_avengine_spear_ext and have no
  RPATH, RUNPATH, SPEAR-checkout string, or dynamic librpc dependency; and
- a Python -S process with only the fresh stage, AVEngine source, and explicit
  physical runtime paths imports the module and AVEngine client, then exercises
  Client construction/termination plus DataBundle and PackedArray state without
  calling initialize, ping, RPC, UE, or a runner.

The span is non-owning. The selected RPC-return path keeps its MessagePack
object handle alive while the Internal PackedArray capsule uses that span; the
host-only test does not prove UE protocol compatibility, real RPC behavior, or
that lifetime path.

## Deliberate boundary

This selection excludes upstream CMakeLists.txt and pyproject.toml,
checkout-relative rpclib and BUILD paths, third_party source, Git metadata,
build trees, wheels, object files, shared libraries, Python cache files, the
remaining SPEAR Python client/services, and every UE plugin, project
configuration, map, content, asset, and generated output.

The optional extension links the external rpclib SDK at build time. AVEngine
does not copy that SDK, its headers, archive, a SPEAR checkout, or a prebuilt
extension into this repository. If a downstream redistributes a built module,
it must handle the applicable rpclib and transitive dependency notices for that
binary separately.

Building and importing the module alone does not start RPC, UE, or a runner,
and is not a SPEAR/UE runtime cutover.
