# Installed Habitat runtime prefix — rebuild recipe and provenance

This document records how the external installed runtime prefixes under
`/data/avengine_external/runtime-prefixes/` were built, so a fresh machine can
rebuild them from this repository plus official upstream archives alone. No
legacy fork or sibling checkout is required: the Habitat sources are
`native/habitat/` in this repository, and every third-party dependency below is
an official upstream archive with a pinned revision.

Status: factual build record (single-repo closure C4, 2026-08-21). This is a
record, not a new gate or contract.

## Prefix inventory

| Prefix | Source | Role |
| --- | --- | --- |
| avengine-habitat-pbr-ibl-c78db29-20260821T0111Z | `native/habitat` @ `c78db295c680107a014c4896170ca451fb1756da` (this repository) | production visual + physics + RLR-adapter runtime for the installed M1/M5/M5.1/M6x/M7 chain |
| avengine-habitat-rlr-adapter-bfeacb8-r1-20260820T1950Z | `native/habitat` @ `bfeacb8644a9b111e95fce69b20701a3bcc4af44` | acoustic authority runtime (M4 current); its own PROVENANCE.json sits in the prefix |
| avengine-habitat-rlr-adapter-ec209a6-20260820T1615Z | `native/habitat` @ `ec209a6e61a154910fe25df1af74ad8921e8debc` | superseded by bfeacb8-r1; retained for record |
| magnum-python-cp312-45811bb-20260820T1845Z | official magnum-bindings + pybind11 archives | Python site for magnum/corrade bindings (`AVENGINE_HABITAT_MAGNUM_PYTHON_SITE`) |

All named source commits are ancestors of this branch; any clone of this
repository can check them out.

## Third-party dependency matrix (statically linked into the prefixes)

Official upstream GitHub archives. Durable copies of every archive listed here
are retained at `/data/avengine_external/builds/dependency-archives-20260821/`.

| Dependency | Revision / version | Upstream | Build shape |
| --- | --- | --- | --- |
| corrade | `451284cddcdc91300a59a194a11728e8124cb664` (archive sha256 `c3bc5842811b27a62a3430d2899c07c067fc5a3d7fc34693e21d7e4ee63395ca`) | mosra/corrade | static, PIC, RelWithDebInfo |
| magnum | `70b0d76fcbb5d6d0fe43b3119446b0045fef64e5` (archive sha256 `5bd060f95f263a21df3dc62ca01c256d025dde0e84d66d245d64d93fc353c3c4`) | mosra/magnum | static, PIC, RelWithDebInfo |
| magnum-plugins | `393b7cb0c098a261a79dbba3230520e008b414a0` (archive sha256 `3fa155ea936fc292b6fdc058d647aa2df06a89e1e293140d84bc116f25874dd7`) | mosra/magnum-plugins | static, PIC, RelWithDebInfo |
| magnum-integration (Bullet) | `6fca807891f05203fe5003275116b9854e918d87` | mosra/magnum-integration | static, PIC, Release |
| bullet3 | 3.25 (archive `bullet3-2c204c49.tar.gz`) | bulletphysics/bullet3 | static, PIC, Release, `USE_DOUBLE_PRECISION=OFF`, `BUILD_PYBULLET=OFF` |
| recastnavigation | `6dc1667f580357e8a2154c28b7867bea7e8ad3a7` | recastnavigation/recastnavigation | static, PIC, RelWithDebInfo |
| rapidjson | `73063f5002612c6bf64fe24f851cd5cc0d83eef9` | Tencent/rapidjson | headers, RelWithDebInfo consumer check |
| tinyxml2 | 6.2.0 | leethomason/tinyxml2 | static, PIC, RelWithDebInfo |
| pybind11 | `a2e59f0e7065404b44dfe92a28aca47ba1378dc4` | pybind/pybind11 | headers for the bindings prefix |
| magnum-bindings | `45811bb52e749677d5bc43d62b384ec546ed93bc` | mosra/magnum-bindings | bindings headers + the magnum-python site build |

## Build steps

### 1. Dependency prefixes (five roles)

Each is a plain upstream CMake build installed into its own prefix. The
retained trees from the 2026-08 builds live on the build host under
`/data/jzy/tmp/avengine-h*` (volatile; the archives above are the durable
source record):

1. **corrade → magnum → magnum-plugins**, in that order, into one shared
   prefix (role "h11"): `-DCMAKE_BUILD_TYPE=RelWithDebInfo
   -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON` plus
   `-DCORRADE_BUILD_STATIC=ON` / `-DMAGNUM_BUILD_STATIC=ON`.
2. **bullet3 + magnum-integration(BulletIntegration)** into one prefix
   (role "h25b"): bullet3 with `-DCMAKE_BUILD_TYPE=Release
   -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
   -DUSE_DOUBLE_PRECISION=OFF -DBUILD_PYBULLET=OFF`; then magnum-integration
   against the h11 + bullet prefixes.
3. **recastnavigation** into its own prefix (role "h9"): static, PIC,
   RelWithDebInfo.
4. **rapidjson + tinyxml2** (plus a recastnavigation copy) into one prefix
   (role "h6"): static, PIC, RelWithDebInfo.
5. **pybind11 + magnum-bindings headers** into one prefix (role "h19").

### 2. Habitat runtime prefix (final)

Recorded configure, from the retained build cache
`/data/avengine_external/builds/avengine-habitat-pbr-ibl-c78db29-20260821T0111Z/CMakeCache.txt`
(compiler: system `/usr/bin/c++`, GCC):

    cmake -S native/habitat -B "$BUILD" \
      -DCMAKE_BUILD_TYPE=Release \
      -DAVENGINE_HABITAT_BUILD_BULLET=ON \
      -DAVENGINE_HABITAT_BUILD_PYTHON_BINDINGS=ON \
      -DAVENGINE_HABITAT_BUILD_RLR_ADAPTER=ON \
      -DAVENGINE_HABITAT_BUILD_RLR_ADAPTER_TESTS=OFF \
      -DAVENGINE_HABITAT_BUILD_LEGACY_AUDIO_SENSOR=OFF \
      -DAVENGINE_HABITAT_GFX_BATCH_WITH_CUDA=OFF \
      -DAVENGINE_HABITAT_INSTALL_RUNTIME=ON \
      -DAVENGINE_HABITAT_RUNTIME_PREFIX="$PREFIX_DIR" \
      -DAVENGINE_RLR_SDK_ROOT="$RLR_SDK_ROOT" \
      -DAVENGINE_MAGNUM_BINDINGS_PYTHON_INCLUDE_DIR="$H19/include" \
      -DCMAKE_PREFIX_PATH="$H19;$H25B;$H11;$H9;$H6"
    cmake --build "$BUILD" -j"$(nproc)"

`AVENGINE_HABITAT_INSTALL_RUNTIME=ON` populates `$PREFIX_DIR` with the
`habitat_sim/` runtime tree and `config/`; the runtime is then activated by
`prepare_installed_habitat_runtime` through `AVENGINE_HABITAT_RUNTIME_PREFIX`,
never by importing from a Git checkout.

### 3. magnum-python site

Built from the pinned pybind11 and magnum-bindings archives against the h11
prefix, with the `avengine-habitat-runtime` Python 3.12 interpreter. The full
record (URLs, revisions, archive SHA256s) is retained inside the prefix at
`magnum-python-cp312-45811bb-20260820T1845Z/provenance/`.

### 4. External SDKs and OS-package inputs

- **RLR SDK**: pinned distribution `4fd446b4abb5c71fb7a232a083bbddd65f25fc6f`
  (see THIRD_PARTY_NOTICES.md); a legal user-installed external SDK at
  `AVENGINE_RLR_SDK_ROOT`. The binding resolves `libRLRAudioPropagation.so`
  from it at import; the binding ELF itself carries no RPATH/RUNPATH.
- **HRTF**: tool entry points currently default `--hrtf` to the OS libmysofa
  data file `/usr/share/libmysofa/MIT_KEMAR_normal_pinna.sofa` (an OS package
  input, not a Git dependency). A versioned 16 kHz derivative lives under
  `/data/avengine_external/rlr-sdk/hrtf/`. Switching the tool defaults to the
  external copy would change rendered binaural bytes, so it is deliberately
  left as a pending owner decision.

### 5. Validation after a rebuild

1. `prepare_installed_habitat_runtime` smoke: habitat_sim and magnum import
   from the prefix/site paths only, `AudioSensorSpec` present.
2. `readelf -d` on `habitat_sim/_ext/*.so`: no RPATH/RUNPATH; NEEDED is system
   libraries plus `libRLRAudioPropagation.so` only.
3. Functional: the fixed-apartment canary and an M7 room batch run on the new
   prefix.
