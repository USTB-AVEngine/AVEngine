# Citation and Attribution Policy

Use `CITATION.cff` for the AVEngine software snapshot and `CITATIONS.bib` for
the required upstream research. AVEngine has no released paper, DOI, or release
date at M0; none should be invented.

## Always cite for the primary runtime

- Habitat 1.0, Habitat 2.0, and Habitat 3.0 for the reused Habitat platform.
- SoundSpaces 2.0 for the RLR acoustic propagation foundation.

Keep the Habitat-Sim MIT notice and history. RLR is separately licensed under
CC BY-NC 4.0, so every use must remain non-commercial, provide attribution and
the license, identify modifications, and avoid implying upstream endorsement.

## Cite conditionally

When a released experiment actually uses an optional component, model, asset,
or dataset—such as gpuRIR, SPEAR, ReplicaCAD, AudioSet, Pixal3D, TRELLIS.2,
SkinTokens, Hunyuan, or a contributed noise model—add that project's canonical
citation and license notice. A bibliographic citation does not replace asset or
model permission.

## Author and contribution wording

Ziyang Ji is recorded as the AVEngine project author/maintainer. Do not use
that metadata to replace Meta or other upstream authorship in the Habitat/RLR
source history. Runtime-fork files must distinguish unchanged upstream code,
AVEngine modifications, and AVEngine integration/planning material.

Before a public release, regenerate a resolved dependency SBOM, preserve
license texts and model cards, verify every released asset by hash, and ensure
the README, dataset card, paper, CFF, BibTeX, notices, and code headers all make
the same reuse/extension/original-contribution distinction.
