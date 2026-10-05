# Inserted source material acceptance

The current MP3D/HM3D multi-actor capture now prepares a versioned GLB for each
inserted source and explicitly renders it through PBR. The original source
package, its admission state, physical scale, geometry, UVs, skin, actions and
scanned room are retained. No shared runtime rebuild is required.

The British shorthair case reproduced white specular spots through its original
`shader_type: phong` AO template even though its PNG base-color and roughness
textures were present. Its source metallic factor was already zero. Other
sources also had missing metallic factors (effective glTF default 1), including
fur and human clothing. Selecting PBR must therefore accompany explicit material
parameters; selecting PBR alone would expose those invalid metallic defaults.

## Material preparation and capture

`avengine.assets.source_materials.prepare_source_glb` creates
`source_materials_v1/<source-content-and-recipe>/visual.materials_v1.glb` plus a
receipt. It preserves the original BIN prefix and base-color pixels/factors,
embeds external PNG/JPEG images, and appends corrected metallic/roughness images.
Texture failures are errors, never a silent gray fallback. Native RGB inputs
with Basis/KTX2 declarations or a `.basis.glb` filename are rejected before
Simulator construction; compressed scene variants must first be converted
outside this capture path.

Current profiles use metallic 0 for fur, skin, cloth, wood and plastic. Fur has a
roughness floor of 0.85, clothing 0.78, head/helmet 0.65, fabric/wood/matte props
0.82, and other dielectric/mixed props 0.65. Exposed silver floor drains and
construction tools use metallic 0.65 and roughness at least 0.60. Floors apply to
the effective roughness factor multiplied by the texture's green channel. The
current one-material mixed appliance models keep their painted finishes as
dielectrics; they do not provide a trustworthy per-part metal/glass mask.

The PBR shader disables image-based environment reflection, enables direct
lights, and maps material textures to linear light and output to sRGB. Native
light intensities, exposure and base colors are retained. AO clone templates
override only the render asset and shader. Rigid templates use the derived GLB.
The target-only semantic pass uses the same derived asset. Versioned AO JSONs
explicitly declare the derived render asset and the unchanged original URDF and
scale. The native loader's actual full render path is checked; public template
handles alone are insufficient evidence of which file was consumed.

Capture writes `source_materials.json` and `source_material_checks.json` into its
output. `AVENGINE_SOURCE_MATERIAL_CACHE` optionally selects a caller-owned cache;
the default is `<capture-output>/source_material_cache`. Writes are atomic so
parallel workers can use an explicitly configured shared cache. External image
content participates in the cache recipe; cached GLBs are checksum-verified.
Original asset
paths and registry files are never rewritten. The material code does not select
a GPU or launch a job; the production scheduler retains that responsibility.

## Automatic gates

The generator samples each source's semantic pixels at the first, middle and
last frame. It compares normalized RGB chromaticity to texels sampled on the
original mesh's actual UVs. A near-white pixel is a novel highlight only if its
mean RGB exceeds original-texture brightness by a relative and absolute margin.
White fur and ceramic are not treated as highlights merely for being white.

Thresholds are chromaticity error P50 <= 0.12 and P95 <= 0.28; highlight ratios
<= 2% for fur, 4% for human skin/clothing, 8% for ordinary props, and 12% for
profiles containing legitimate metal. A mixed-material source uses the largest
of its material highlight budgets because the object semantic mask does not
identify individual materials. Color or highlight violations abort capture
with an error and a failing receipt. Fewer than 100 visible source pixels are
recorded as `not_run`, never counted as a material pass; production visibility
checks remain responsible for intentionally occluded/offscreen samples.

The diagnostic matrix also renders an independent flat original-texture view
with the native `no_lights` setup and measures same-pixel RGB RMSE and
chromaticity. For nonmetal surfaces it rejects median normalized RGB error
above 0.35. Metal base color encodes specular reflectance, so unlit-to-lit RGB
magnitude is reported but not used as a diffuse-brightness gate; hue and
novel-highlight limits still apply. The flat
reference uses an unlit color-only GLB in a separate NONE-stage Simulator. Its
pixels are the original sRGB texture values, without a second gamma transform.
The normal native light setup would multiply flat colors by ambient intensity
0.4 and cannot be used as a texture reference. A known RGB(64,128,192) canary
must read back those exact values before freezing a native matrix. Before and
after also use independent Simulators, preserving their respective old/new
PBR configurations. Erode the intersection of source
masks one pixel for comparisons; exclude a two-pixel source boundary when
checking that the scanned background did not change.

Normalizing tiny RGB values exaggerates changes from neutral specular light on
near-black fur. Same-view chromaticity errors are weighted continuously by
`clip(mean(reference_sRGB) / 0.1, 0, 1) ** 2`; colors at or above 0.1 keep full hue
sensitivity. This confidence weight is an explicit operational policy, not a
measured optical property. The absolute RGB gate remains active for nonmetals. The runtime
palette gate has no per-pixel reference and retains its conservative palette
comparison. Rendered pixels with mean sRGB <= 0.02 have undefined hue and are
excluded from hue error, while staying in highlight and absolute-RGB statistics.
If the original palette P95 mean exceeds 0.1 and rendered P95 mean is <= 0.02,
the palette gate fails as `source_texture_color_unobservable`; a whole colored
source cannot silently become black. An all-black source still rejects a
uniform gray fallback above the same RGB-error threshold. The dark-sable
source's original albedo is mostly black; preserving its texture cannot prove
that its coat color is authentic or restore an absent brown fur pattern.

The palette test is a lighting-tolerant color check, not proof of spatial pattern
fidelity. Byte equality of the preserved base-color images, independent flat
reference comparisons and human review of the native matrix provide that
additional evidence. The roughness floors and numerical image thresholds are
an operational acceptance policy, not measurements of the physical materials.

## Native review workflow

Run tools from the same checkout as the generator with `PYTHONPATH=src:.`.

1. `tools/assets/audit_source_materials.py inventory` accepts repeated
   `--registry` flags (later rows win), `--external-index`, `--plan-root`,
   `--asset-output` and `--output`. It records every file, decoded image format,
   material parameter and original AO config, and fails on missing used IDs.
2. `render-one` consumes that inventory and an explicit asset ID, real room
   manifest/M1 request, runtime/data roots, output, GPU device ID, camera pose
   and object position. Run serially, with scheduler-verified GPU headroom.
   The longest GLB geometry edge is normalized to one metre for diagnostic
   display only. All assets in a house use a fixed camera and floor point.
3. Run the complete gate, which performs no GPU work:

   ```bash
   python tools/assets/check_source_material_matrix.py \
     --inventory /caller/output/inventory.json \
     --render-root /caller/output/renders/matrix \
     --reference-calibration /caller/output/evidence/reference_color_readback_proof.json \
     --house mp3d --house hm3d \
     --output /caller/output/material_acceptance.json \
     --html /caller/output/asset_materials_comparison.html
   ```

The complete gate requires every inventory source in each house; missing,
offscreen, failing, inconsistent-camera or changed-background results cause a
nonzero exit. It checks original/derived content hashes, base-color pixels,
geometry, skin, animation and transparency invariants. The HTML embeds PNGs and
supports local filtering and enlargement. Native diagnostic renders are not
counted as production Episodes or asset admission.

`--result-name` selects a separately preserved analysis result when a metric
policy is corrected without changing native images. Retain the original
`result.json`, original metrics and native PNGs; record hashes and the reason
for reanalysis. Do not change thresholds to hide a material regression.

CPU regressions cover compressed imports, missing textures, original asset and
color preservation, shared-cache publication, gray fallback, excess highlights,
white fur, offscreen status, incomplete matrices and generator capture wiring.
The legacy `capture_mp3d_multi_actor` entrypoint and v7 streaming
`NativeContinuousVideo` use the same source preparation and first/middle/last
frame gate. Configure source PBR on the actual Simulator mediator after the
scan loads, before loading sources. The scan can contain PBR materials itself;
its existing drawables must retain their original PBR handle and image settings.
The native matrix applies the same ordering and rejects changed background
pixels outside the source boundary. The v7 candidate caller supplies an attempt-owned material cache,
includes `MATERIAL_REVISION` in the visual stage recipe, and requires both
`source_materials.json` and `source_material_checks.json` before completion.
Existing visual stages from an older recipe are rejected on resume; use a new
output for the new recipe, preserving previous candidates. Direct diagnostic
callers can supply `source_material_cache_directory` or a task-local
`AVENGINE_SOURCE_MATERIAL_CACHE`; the fallback is a retained unique temporary
directory under `TMPDIR`. Asset admission, motion, audio and formal-release
blocks are unchanged. No registry or generator CLI flag is required.

The direct-GLB MP3D streaming canary has a partial modal semantic actor mask:
its scan semantic geometry covers body pixels that the RGB sensor sees. Native
blank-stage controls at the identical camera and M2 pose provide independent
full-body masks for offline appearance/background review. A separate actor-free
scan control checks that configuring source PBR has no effect on scan RGB.
These diagnostics do not replace production pixels, change semantic outputs, or
claim that an occluded runtime sample passed. The runtime gate measures the
source pixels actually admitted by the modal semantic image; the complete
fixed-view asset matrix and full-body canaries provide additional coverage.
