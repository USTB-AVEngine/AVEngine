# Historical and optional backends

UE/SPEAR, gpuRIR and the earlier Hunyuan/Stable-Audio asset workflows are
retained only for migration comparisons and explicitly scoped research. They
are not required to install AVEngine, run fast tests, validate schemas or use
the Habitat-native interfaces.

The archived operating notes are in `AGENTS_LEGACY_SPEAR.md`. They contain old
machine-local paths and must not be copied into current configuration. Anyone
reproducing a legacy comparison supplies their own checkout and data roots,
records exact versions and hashes, and invokes a separately named legacy tool.
No current setup command downloads or configures those dependencies.

Current room-specific UE/SPEAR comparison work and external residential-scene
adapters are documented separately in
[OPTIONAL_RESIDENTIAL_SCENE_BACKENDS.md](../architecture/OPTIONAL_RESIDENTIAL_SCENE_BACKENDS.md).
They keep all external datasets outside Git and do not restore the historical
backend as AVEngine's task/audio authority.
