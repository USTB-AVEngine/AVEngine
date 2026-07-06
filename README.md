# AVEngine — Audio-Visual Engine

Research infrastructure for **Attribute-Conditioned Spatial Audio-Visual
Reasoning (ASAR)**. Combines Unreal Engine 5 (via SPEAR RPC) for
photorealistic multi-view rendering with gpuRIR for 4-channel
first-order-ambisonic room-impulse-response simulation, yielding audio-video
scenes where the **spatial ground truth is exact** (mic position, source
positions, camera intrinsics all deterministic).

Currently supports 5 animated animal tags (dogs, cats, chipmunk) + 7 static
ungulate tags in two rooms (apartment_0000 real Kujiale scan; procedural
shoebox). Outputs 640×480 15 fps 5s MP4 with muxed stereo audio.

⚠ **Private research project (not open source yet).** Contact author before
redistribution.

## Directory layout

```
AVEngine/
├── README.md                # this file
├── manifest.yaml            # single source of truth for deps + data
├── scripts/setup.sh         # `bash scripts/setup.sh` populates external/
├── envs/*.yml               # 3 conda env recipes (create manually)
├── assets/mesh_library/     # Quaternius rigged animal GLBs (CC0)
├── docs/                    # pipeline docs, specs, plans, image assets
└── external/                # git-ignored; populated by setup.sh
    ├── SPEAR/               # pipeline main; fork of spear-sim/spear
    └── Hunyuan3D-2.1/       # 3D asset generator (Tencent, upstream)
```

## Setup (Linux, GPU)

**Prereqs**: bash 4+, python3 + pyyaml, git, conda (miniconda/anaconda),
NVIDIA GPU with driver 550+, UE 5.5 build tools if you'll re-cook the
SpearSim project.

### Step 1 — Clone + populate deps

```bash
git clone <AVEngine repo url> /data/jzy/code/AVEngine
cd /data/jzy/code/AVEngine
bash scripts/setup.sh
```

`setup.sh` is idempotent. On the author's machine (with pre-existing
`/data/jzy/code/SPEAR` etc.) it creates symlinks; on your machine it clones
into `external/`.

### Step 2 — Create 3 conda envs

```bash
for env in spear-env sao-env hunyuan3d-env; do
    conda env create -f envs/$env.yml
done
```

**Post-install steps not in the yml files**:

- `spear-env` needs `spear_ext` (SPEAR's compiled C++ RPC extension) and
  `spear-sim` (Python client). Neither is on PyPI. After env creation:
  ```bash
  conda activate spear-env
  cd external/SPEAR
  pip install -e python       # spear-sim
  pip install -e python_ext   # spear-ext (requires cpp/ built; see SPEAR docs)
  ```
- `sao-env` needs `gpuRIR` (not on PyPI):
  ```bash
  git clone https://github.com/DavidDiazGuerra/gpuRIR /tmp/gpuRIR
  conda activate sao-env
  pip install /tmp/gpuRIR   # requires CUDA toolkit + gcc
  ```
- Env creation can take 30-60 min due to CUDA torch downloads.

### Step 3 — Provide external data

`setup.sh` does NOT download data. Place these at the paths listed in
`manifest.yaml` `external_data`:

| Path | Size | Source |
|------|------|--------|
| `/data/datasets/omniaudio/train-data-az-360-large` | ~40 GB | AudioSet wavs (contact author) |
| `/data/datasets/omniaudio/stable-audio-open` | ~5 GB | https://huggingface.co/stabilityai/stable-audio-open-1.0 |
| `/data/jzy/code/Hunyuan3D-2.1/pretrained_models` | ~20 GB | https://huggingface.co/Tencent-Hunyuan/Hunyuan3D-2.1 |

### Step 4 — Symlink mesh_library to SPEAR's expected path ⚠

Until Spec 2 (SPEAR path parameterization) lands, SPEAR hardcodes
`/data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_*`.
On collaborator machines, run (needs sudo):

```bash
sudo mkdir -p /data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library
sudo ln -s $(pwd)/assets/mesh_library/quaternius_animalpack /data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_animalpack
sudo ln -s $(pwd)/assets/mesh_library/quaternius_farm       /data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_farm
```

## First demo — two dogs in a room

```bash
conda activate spear-env
export DISPLAY=:99
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
python external/SPEAR/tools/gpurir_scenes/scene_two_dogs.py --skip-audio
```

Expected outputs (after ~5-10 min UE render):

- `external/SPEAR/tmp/gpurir_scenes_v1/two_dogs/shoebox/view0.mp4`
- `external/SPEAR/tmp/gpurir_scenes_v1/two_dogs/apartment/view0.mp4`

Each MP4 is 640×480, 5s, 75 frames, no audio. Omit `--skip-audio` to also
generate GPURIR 4-channel audio and mux `view*_with_audio.mp4` files.

## Directory cheat sheet — where does X live?

| Feature | Path |
|---------|------|
| Pipeline main entrypoints | `external/SPEAR/tools/gpurir_scenes/` |
| Species → rig map | `external/SPEAR/tools/species_rig_map.py` |
| Furniture collision map | `external/SPEAR/data/apartment_furniture_map.json` |
| Rigged 3D animal meshes | `assets/mesh_library/` |
| Chinese pipeline doc | `docs/pipeline_zh.md` |
| English pipeline doc | `docs/pipeline_en.md` |
| Design specs | `docs/superpowers/specs/` |
| Implementation plans | `docs/superpowers/plans/` |

## Troubleshooting

See [`docs/troubleshooting.md`](docs/troubleshooting.md). Common gotchas:

- **`conda activate` must be `spear-env`** — do NOT use `thu` or other env;
  RPC silently fails on wrong Python
- **`DISPLAY=:99` required** — UE needs an X server (headless X counts)
- **`furniture_map.json missing`** — SPEAR must be at commit ≥ `bc8ce323`

## Contact

Ziyang Ji — [`Eastforward`](https://github.com/Eastforward) on GitHub.
Research collaboration welcome; please ping before redistributing.

## License

See [`LICENSE`](LICENSE). Currently proprietary; open-source release pending.
Third-party components (Quaternius rigs, SPEAR upstream, Hunyuan3D-2.1)
retain their own licenses; see `manifest.yaml` `upstream` fields.
