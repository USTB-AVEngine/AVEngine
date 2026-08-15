# SPEAR python_ext S2 source staging

This directory holds the selected source-only SPEAR python extension closure
needed by a later AVEngine-owned build. It is not build wiring, a Python
package installation, or a runtime cutover. Current execution continues to
use the external SPEAR runtime and its installed extension binary.

## Origin and treatment

| Field | Recorded value |
| --- | --- |
| Upstream | spear-sim/spear@251bd5e0d3d1e7297ec072bb9b0df9ef63f864b7 |
| Relationship | adapted selected source-only staging; the retained C++ files keep their upstream copyright and MIT headers |
| License | MIT; full text is retained at LICENSES/SPEAR-MIT.txt |

The selected C++ source retains the upstream RPC and nanobind interfaces
without AVEngine behavior changes. The Python wrapper is retained as the
upstream package initializer.

## Selected source map

The S2 selection contains exactly these upstream files:

- cpp/assert.h
- cpp/client.h
- cpp/func_signature_registry.h
- cpp/msgpack_adaptors.h
- cpp/msgpack_utils.h
- cpp/spear_ext.cpp
- cpp/std.h
- cpp/types.h
- python/spear_ext/__init__.py

Each file has the matching path below native/spear/python_ext in this
repository. The C++ sources still include the upstream nanobind and rpclib
interfaces. Their dependency source, headers, libraries, and build
configuration are deliberately not staged here.

## Deliberate boundary

S2 intentionally excludes:

- upstream CMakeLists.txt and pyproject.toml, including checkout-relative
  rpclib and BUILD paths;
- third_party, Git metadata, build trees, wheels, object files, shared
  libraries, and Python cache files;
- the remaining SPEAR Python client and services; and
- UE plugins, project configuration, maps, content, assets, and generated
  output.

A later explicit batch may add AVEngine-owned build wiring and validate a
fresh extension build. Until that work is complete, no AVEngine runtime import
or external SPEAR/UE execution path is changed by this staging.
