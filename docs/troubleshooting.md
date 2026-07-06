# Troubleshooting

## setup.sh reports "target exists but is neither symlink nor git repo"

Something (not setup.sh) put a plain directory at `external/<dep>`. Delete
it and rerun:

```bash
rm -rf external/<dep>
bash scripts/setup.sh
```

## setup.sh reports "local_hint exists but origin doesn't match"

Your local pre-existing clone at `local_hint` has a different git remote
than manifest expects. Either (a) update that clone's origin, or (b) use
`--force-clone <dep>` to clone into `external/` instead.

## conda env create fails on spear_ext

`spear_ext` is a compiled C++ extension. It is NOT on PyPI. Nor can you
`pip install -e python_ext` directly — it requires **Unreal Engine's
bundled clang + libc++**, not system gcc.

After creating `spear-env`, run:

```bash
# Install pure-python spear (RPC client wrapper)
/data/jzy/miniconda3/envs/spear-env/bin/pip install -e external/SPEAR/python

# Install the C++ extension using UE's toolchain. SPEAR ships a script that
# figures out the right paths:
cd external/SPEAR
/data/jzy/miniconda3/envs/spear-env/bin/python tools/install_python_extension.py \
    --unreal-engine-dir /data/UE_5.5 \
    --conda-env spear-env
```

**⚠ Gotcha**: `tools/install_python_extension.py` uses `conda activate <env>`
under the hood. If your shell profile prefixes PATH with another conda env
(check `env | grep PATH | head -1`), the wrapper may install to that env
instead. **Workaround**: call `pip install -e python_ext` directly using
spear-env's pip binary, passing the UE clang path explicitly:

```bash
UE=/data/UE_5.5
CLANG=$UE/Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/v23_clang-18.1.0-rockylinux8/x86_64-unknown-linux-gnu/bin/clang++
LIBCXX=$UE/Engine/Source/ThirdParty/Unix/LibCxx
CXX_FLAGS="-std=c++20 -O3 -D_LIBCPP_ENABLE_EXPERIMENTAL -nostdinc++ -I$LIBCXX/include/c++/v1 -Wno-reserved-macro-identifier -stdlib=libc++ -L$LIBCXX/lib/Unix/x86_64-unknown-linux-gnu -lc++"
/data/jzy/miniconda3/envs/spear-env/bin/pip install -e external/SPEAR/python_ext \
    -C cmake.define.CMAKE_CXX_COMPILER="$CLANG" \
    -C cmake.define.CMAKE_CXX_FLAGS="$CXX_FLAGS"
```

## conda env create fails on gpuRIR

gpuRIR is NOT on PyPI. Build from source (requires CUDA toolkit + gcc):

```bash
git clone https://github.com/DavidDiazGuerra/gpuRIR /tmp/gpuRIR
conda activate sao-env
pip install /tmp/gpuRIR
```

## Pipeline reports "apartment_furniture_map.json not found"

Confirm `external/SPEAR/data/apartment_furniture_map.json` exists.
This file ships with SPEAR at commit >= bc8ce323.

## Pipeline reports "Quaternius rig not found at /data/jzy/code/Spatial/..."

Until Spec 2 lands, SPEAR expects this absolute path. Symlink (needs sudo):

```bash
sudo mkdir -p /data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library
sudo ln -s $(pwd)/assets/mesh_library/quaternius_animalpack /data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_animalpack
sudo ln -s $(pwd)/assets/mesh_library/quaternius_farm       /data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_farm
```
