# Source Asset Registry Design

## Goal

Create a single source-of-truth registry for reusable audible visual assets.
Dataset specs should choose a reviewed asset by `asset_id`; clip metadata should
record how that asset was used in one event. The registry must preserve enough
generation, appearance, rig, audio, and review metadata to support later
queries such as "a darker dog", "a visible speaking human", or "two different
cats".

## Terms

- `asset`: a reusable source asset variant, such as
  `dog_golden_0001`, `dog_beagle_0002`, or `human_female_0001`.
- `event`: one use of an asset inside one clip, including trajectory,
  visibility, and sound timing.
- `legacy_tag`: the current SPEAR render/audio identifier, such as
  `dog_golden` or `cat_british_shorthair_v2`. This stays for compatibility.

Do not use `instance` for per-clip usage. If we need "big beagle" versus
"small beagle" or "dark golden" versus "light golden", those are separate
asset variants with separate `asset_id` values.

## File Layout

The registry lives in SPEAR data because the dataset generator consumes it:

```text
external/SPEAR/data/source_assets_v1/registry.json
external/SPEAR/data/source_assets_v1/<category>/<family>/<asset_id>/asset.json
```

Binary artifacts remain where their pipeline produced them, usually under
`external/SPEAR/tmp/hy3d_batch/approved/{legacy_tag}` for currently approved
Hunyuan animal assets. Registry files should store repo-relative SPEAR paths
where possible.

## Top-Level Registry

`registry.json` is a small index. It should be cheap to load and suitable for
filtering without opening every asset file.

```json
{
  "schema_version": "source_assets_v1",
  "assets": [
    {
      "asset_id": "dog_golden_0001",
      "asset_class": "animal",
      "category": "dog",
      "family": "golden_retriever",
      "path": "dog/golden_retriever/dog_golden_0001/asset.json",
      "overall_status": "approved"
    }
  ]
}
```

Required fields:

- `asset_id`: stable unique key used by dataset specs.
- `asset_class`: broad class, such as `animal` or `human`.
- `category`: user-facing type, such as `dog`, `cat`, or `human`.
- `family`: narrower reusable group, such as `beagle`, `golden_retriever`,
  `british_shorthair`, `male_adult`, or `female_adult`.
- `path`: path to the detailed `asset.json`, relative to the registry root.
- `overall_status`: `candidate`, `needs_review`, `approved`, or `rejected`.

## Asset Manifest

Each approved reusable asset has one `asset.json`.

```json
{
  "schema_version": "source_asset_v1",
  "asset_id": "dog_golden_0001",
  "legacy_tag": "dog_golden",
  "asset_class": "animal",
  "category": "dog",
  "family": "golden_retriever",
  "variant": {
    "variant_index": 1,
    "size": "medium",
    "coat_type": "long",
    "intended_color_label": "golden"
  },
  "generation": {
    "source_pipeline": "hunyuan3d",
    "model": "hunyuan3d-2.1",
    "seed": null,
    "positive_prompt": null,
    "negative_prompt": null,
    "text_description": "golden retriever dog",
    "created_at": "2026-07-09"
  },
  "appearance": {
    "dominant_colors": [
      {
        "role": "coat_primary",
        "hex": "#8A5A2B",
        "rgb": [138, 90, 43],
        "lab": [42.1, 14.7, 34.2],
        "coverage": 0.62,
        "source": "measured_from_texture"
      }
    ],
    "color_tags": ["golden", "brown"],
    "lightness": 0.42,
    "saturation": 0.58
  },
  "visual_assets": {
    "mesh_original": "tmp/hy3d_batch/approved/dog_golden/mesh.glb",
    "mesh_oriented": "tmp/hy3d_batch/approved/dog_golden/mesh_oriented.glb",
    "mesh_runtime": "tmp/hy3d_batch/approved/dog_golden/mesh_runtime.glb",
    "diffuse": "tmp/hy3d_batch/approved/dog_golden/hy3d_diffuse.jpg",
    "roughness": "tmp/hy3d_batch/approved/dog_golden/hy3d_roughness.jpg",
    "metallic": "tmp/hy3d_batch/approved/dog_golden/hy3d_metallic.jpg",
    "review_image": "tmp/hy3d_batch/approved/dog_golden/direction_preview_review.png",
    "direction_json": "tmp/hy3d_batch/approved/dog_golden/direction.json"
  },
  "rig": {
    "skeleton_family": "quaternius_dog",
    "animations": ["Idle", "Walking"],
    "loop_required": true
  },
  "audio": {
    "default_lookup": "dog_bark",
    "allowed_lookups": ["dog_bark", "dog_growl", "dog_sharp_bark"]
  },
  "review": {
    "overall_status": "approved",
    "appearance_status": "approved",
    "direction_status": "approved",
    "texture_status": "approved",
    "rig_status": "approved",
    "audio_mapping_status": "approved",
    "approved_by": "jzy",
    "approved_at": "2026-07-09",
    "notes": null
  }
}
```

### Intended Versus Measured Appearance

Generation-time intent and measured output are separate:

- `variant.intended_color_label` and `generation.text_description` record what
  Flux/Hunyuan was asked to make, such as "dark golden retriever".
- `appearance.dominant_colors` records what the produced texture/render
  actually contains.

Do not store every color. Store the 3-5 dominant or semantically important
colors needed for comparisons. Use `lightness` for dark/light comparisons and
LAB distance for nearest-color matching.

## Event Metadata

Clip events record one use of a reusable asset and should not duplicate texture
or generation metadata.

```json
{
  "event_id": "clip_0003_source_0001",
  "asset_id": "dog_golden_0001",
  "audio_lookup": "dog_bark",
  "trajectory": {
    "type": "left_rear_to_right_front",
    "world_xyz_per_frame": []
  },
  "visibility_requirement": "invisible_then_visible",
  "sound_requirement": {
    "active_frame_range": [10, 70]
  }
}
```

Existing clip `sources` may continue to carry legacy render/audio fields while
the pipeline migrates. New code should include `asset_id` whenever a source was
selected from the registry.

## Registration Lifecycle

1. Candidate generation writes raw artifacts and generation metadata.
2. Direction, texture, rig, animation, audio mapping, and review gates run.
3. Only assets whose required review fields are `approved` are registered for
   production source pools.
4. Dataset specs refer to `asset_id`, not only `legacy_tag`.
5. The scene generator resolves `asset_id` through the registry and emits
   compatibility fields: `tag`, `audio_lookup`, and `asset_id`.
6. Per-clip metadata records events: trajectory, visibility requirement, and
   effective sound frames.

If an asset is not approved, the registry loader must reject it by default.

## Initial Migration

The initial registry should cover the currently reviewed M1 source pool:

- `dog_golden_0001` for legacy tag `dog_golden`
- `dog_beagle_0002` for legacy tag `dog_beagle_v2`
- `cat_british_shorthair_0002` for legacy tag `cat_british_shorthair_v2`

Older untextured or incomplete tags can stay out of the registry until their
texture/runtime/review gates are complete.

## Non-Goals

- Do not move binary assets in this change.
- Do not rewrite render/audio passes away from `tag`; use registry resolution
  to keep them compatible.
- Do not require automatic perfect Hunyuan orientation or identity validation.
  Human review remains the final approval gate for now.
