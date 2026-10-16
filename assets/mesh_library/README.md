# mesh_library

Rigged 3D animal models used by the AVEngine pipeline.

## Contents

- `quaternius_animalpack/` — 5 GLB: Cat, Dog, Eagle, Piranha, Wolf (2.6 MB)
- `quaternius_farm/` — 7 GLB: Cow, Horse, Llama, Pig, Pug, Sheep, Zebra (2.0 MB)

Total: 12 GLB, ~4.6 MB.

## Provenance

All meshes are from **Quaternius** free 3D asset packs:
- Animal Pack: https://quaternius.com/packs/animatedanimals.html
- Farm Animal Pack: https://quaternius.com/packs/animatedfarmanimals.html

## License

Both packs are released by Quaternius under **CC0 1.0 Universal (Public Domain)**:
https://creativecommons.org/publicdomain/zero/1.0/

Redistribution as part of AVEngine is unrestricted.

## Usage by AVEngine pipeline

Referenced from `external/SPEAR/tools/species_rig_map.py::QUATERNIUS_DIR`.
Currently only `Cat.glb` and `Dog.glb` are actively used (map to 5 animated
tags: dog_golden, dog_husky, cat_persian, cat_tabby, chipmunk). Other GLBs
kept for future rig-family expansion (e.g., Horse for future animated
ungulates once robust_skin_transfer supports semantic bone names).

## ⚠ Hardcoded path caveat

Until Spec 2 (SPEAR path parameterization) lands, SPEAR reads these from
`/data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_*`.
On collaborator machines, symlink after `scripts/setup.sh` (see setup.sh
"Next steps" output).
