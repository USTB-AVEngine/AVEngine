# Historical and explicitly selected backends

The archived operating notes in `AGENTS_LEGACY_SPEAR.md` describe earlier
SPEAR/UE, gpuRIR and Hunyuan/Stable-Audio workflows. They retain old
machine-local paths for historical interpretation and must not be copied into
current configuration.

This archive does not classify every current UE/SPEAR route as historical:

- MP3D production scene execution, pixels, sensors and pose use Habitat-Sim;
  its UE import is a `comparison_visual` diagnostic only.
- Native `apartment_0000` uses UE/SPEAR for production visual execution.
- InteriorAgent/Kujiale uses the UE/SPEAR USD/MDL adapter for production visual
  execution over an explicitly selected external scene.
- Skokloster remains excluded unless the project owner explicitly reauthorizes
  it for a named task.

During the single-source migration, native Habitat and SPEAR workspaces remain
transition inputs. They are not the final repository architecture. After
cutover, required distributable integration source lives in the canonical
AVEngine repository, while Unreal Engine installations, datasets, room assets,
weights, generated media and build products remain external.

The fast bootstrap does not install Unreal Engine, download external datasets or
configure gpuRIR/generative-model research tools. That installation boundary
does not downgrade Apartment or Kujiale to comparison visual.

Current room-specific behavior is documented in
[OPTIONAL_RESIDENTIAL_SCENE_BACKENDS.md](../architecture/OPTIONAL_RESIDENTIAL_SCENE_BACKENDS.md).
An optional backend cannot redefine AVEngine's entity, room, Timeline, source,
flag, audio, label, evidence or admission authority.
