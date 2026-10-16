# Room-selected backend boundary

Backend integrations are selected explicitly by room family. Habitat-Sim is
production visual for MP3D; UE/SPEAR is production visual for native Apartment
and InteriorAgent/Kujiale. An MP3D UE import remains comparison-only, and
Skokloster remains excluded unless explicitly reauthorized.

The fast bootstrap does not install Unreal Engine, download datasets or load
gpuRIR and generative-model tooling. External installation is a dependency
boundary, not a statement that every non-default backend is experimental.

During the source migration, Habitat and SPEAR still resolve through maintained
external workspaces. The final layout integrates the selected distributable
source into the canonical AVEngine repository while keeping UE, data, assets,
weights, generated media and build products outside Git.

No visual backend may redefine AVEngine's central entity, room, Timeline,
source, navigation, audio, flag, label, evidence or admission authority.
