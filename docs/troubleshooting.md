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

`spear_ext` is a compiled C++ extension. It is NOT on PyPI. After creating
`spear-env`:

```bash
conda activate spear-env
cd external/SPEAR
pip install -e python       # spear-sim (Python RPC client)
pip install -e python_ext   # spear-ext (needs SPEAR's cpp/ pre-built)
```

See SPEAR docs for building `cpp/` (Unreal Engine 5 build tools required).

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
