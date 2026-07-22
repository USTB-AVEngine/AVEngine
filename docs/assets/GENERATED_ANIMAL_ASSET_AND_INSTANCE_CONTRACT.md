# Generated animal asset and instance contract

Status: normative for current asset research; production promotion still follows
[`ADR-0006`](../adr/ADR-0006-template-authoritative-animal-assets.md).

## The rule that must not be forgotten

**A new species, breed or materially different morphotype is a new source asset,
not an instance of an existing animal.** Reuse the build procedure and a
compatible motion family; never reuse another breed's body shape as geometry
authority.

For example, a Border Collie must not be made by recolouring, resizing or
slightly deforming the accepted Labrador, Beagle or Quaternius dog. It needs
its own breed-correct canonical image, Pixel3D mesh, repaired topology,
target-native skeleton/weights and deformation review. Quaternius may supply a
compatible walk motion only after the new mesh and rig exist.

## Source asset versus instance

A **source asset** owns the animal's identity and anatomy:

- species, breed and morphotype;
- silhouette, proportions, mesh, topology, UVs and material slots;
- semantic skeleton, joint placement, skin weights and collision shape;
- canonical scale, mouth/emitter anchor and motion adapter;
- geometry, deformation, contact and visual-review evidence.

An **instance** references an accepted source asset and may vary only within
that asset's validated domain:

- `size`: `small / medium / large`;
- `body_build`: `slim / standard / stocky`;
- explicit `life_stage`;
- one of that breed's three reviewed coat profiles.

A coat change is an instance-level appearance operation only when the existing
mesh already represents the same breed and morphotype. A breed change is never
a coat change.

## Identity decision

Before generating anything, answer these questions in order:

1. Does an accepted source asset already have the requested species, breed and
   morphotype?
2. Is the requested variation inside that asset's reviewed size, build,
   life-stage and coat ranges?
3. If both answers are yes, create an instance of that asset.
4. Otherwise, declare a new `research_candidate` source asset and run the full
   workflow below. Do not silently fall back to a generic dog or cat.

## Authority at each stage

| Stage | Authority | Must not become authority |
| --- | --- | --- |
| Breed identity | rights-reviewed real photographs | a generic clay animal |
| Canonical 2D image | approved FLUX/Qwen output derived from those references | a structurally tidy but wrong-breed image |
| New geometry | Pixel3D output from the approved canonical image | an existing breed's mesh or silhouette |
| Topology | explicit repair and mesh QA | rigging tricks that conceal broken geometry |
| Rig | target-native semantic joints and weights | copied joints/weights from another body shape |
| Motion | a compatible motion family retargeted semantically | the motion donor's mesh, proportions or skin |
| Coat | breed-and-coat-specific real references | RGB tinting or cross-breed colour names |

## Required new-asset workflow

1. **Declare identity.** Record species, breed, morphotype, intended size/build/
   life-stage domain and three valid breed-scoped coat names.
2. **Collect real references.** Use several rights-reviewed views that show the
   breed's head, torso, legs, feet and single tail clearly.
3. **Generate the canonical 2D animal.** Use undistilled FLUX or Qwen with the
   references. Check breed silhouette, one head, four complete limbs, one tail,
   paws and neutral pose.
4. **Hard stop for project-owner review.** Do not send the image to Pixel3D
   until the owner accepts both breed identity and anatomy. Four visible limbs
   do not compensate for the wrong breed.
5. **Create a genuinely new mesh.** Run Pixel3D from the accepted image, then
   inspect front, rear and both side views. Reject merged legs, duplicated
   appendages, wrong silhouette or missing feet before rigging.
6. **Repair topology.** Resolve non-manifold regions and unsafe appendage
   topology without reshaping the animal into an existing template.
7. **Create the target-native rig.** TokenRig or an equivalent tool must infer
   this mesh's own skeleton and weights. Audit semantic joint placement before
   retargeting.
8. **Normalize the target rest frame.** Use the reviewed anatomical front to
   make heading cardinal, then fit one support plane through the lower endpoints
   of the four semantically identified foot chains. Rigidly rotate the complete
   mesh-and-rig hierarchy so that plane faces world up and translate its lowest
   reviewed foot to ground height. This stage may correct only the asset frame;
   it must not move individual feet, flatten the animal's back, alter topology,
   replace joints or change skin weights. Reject non-planar foot evidence or a
   support tilt above the reviewed bound instead of forcing a result.
9. **Retarget motion only.** Transfer compatible Idle/Walking motion by semantic
   joints. Correct coordinate/yaw conventions explicitly; never import the
   donor's shape or skin.
10. **Review complete animation cycles.** Check side/front/rear videos for travel
   direction, hind-leg orientation, joint folding, foot contact, sliding and
   tail deformation.
11. **Register only after acceptance.** The candidate becomes a reusable source
    asset only after its required geometry, deformation and runtime gates pass.
    Instance-level coat/size/build/life-stage generation comes afterwards.

Current execution policy for this research workflow is full GPU residency. Do
not enable `low_vram`, sequential CPU offload or model CPU offload. If a model
does not fit, use a supported multi-GPU layout or report the exact blocker.

## Forbidden shortcuts

- Do not call a recoloured or resized existing breed a new breed.
- Do not strengthen a generic clay/template guide until it erases breed shape
  merely to obtain four visible limbs.
- Do not send a wrong-breed 2D image through Pixel3D and hope topology, rigging
  or retargeting will restore the breed later.
- Do not copy another animal's mesh, silhouette, joints or skin weights and call
  the result an instance-level variant.
- Do not use simple RGB multiplication as evidence of a real coat such as Blue
  Abyssinian; use breed-specific reference-guided appearance editing.
- Do not hide geometry or deformation failures behind one plausible camera
  view. Keep rejected candidates rejected.

## What may be reused from a successful animal

Reuse the **pipeline mechanics**:

```text
real breed references
  -> undistilled FLUX/Qwen canonical image
  -> owner image review
  -> Pixel3D new mesh
  -> topology QA/repair
  -> TokenRig target-native skeleton and weights
  -> reviewed heading and four-foot support-plane normalization
  -> compatible motion retarget
  -> geometry/deformation/runtime QA and review video
```

Do not reuse the successful animal's mesh, silhouette, joint locations, skin
weights or coat. A Labrador proves that the process can work; it is not the
geometric starting point for a Border Collie.

## Retained Border Collie failure lesson

The rejected Border Collie cross-check demonstrated the exact failure this
contract prevents:

1. the first FLUX image had useful Border Collie identity;
2. a second generation over-weighted the Quaternius structural guide to expose
   all limbs;
3. that image lost the Border Collie silhouette and became a generic dog;
4. Pixel3D faithfully reconstructed the wrong shape;
5. geometry and walking-deformation audits then failed.

The lesson is not to repair or recolour that candidate. Appendage completeness
and breed identity must both pass at the canonical-image gate, before expensive
3D work begins.

## Relationship to ADR-0006

ADR-0006 keeps audited templates as the formal production default because
arbitrary target-native meshes have historically produced unstable topology,
weights and deformation. This document makes the target-native research route
precise: it creates an independent `research_candidate` source asset and must
not masquerade as an instance of an accepted template.

A successful research candidate does not silently reverse ADR-0006. Promotion
of target-native generation to the production default requires the complete
evidence in ADR-0006's validation plan and an explicit reviewed ADR update.

The first accepted research cross-check of the complete target-native route is
recorded in
[`BORDER_COLLIE_TARGET_NATIVE_CROSSCHECK_20260723.md`](BORDER_COLLIE_TARGET_NATIVE_CROSSCHECK_20260723.md).
The reusable SPEAR-side post-TokenRig execution entry is
`tools/run_target_native_generated_quadruped_review.py`; it enforces heading,
rig audit, support-plane leveling, motion retarget, rotation-invariant
Walk/Idle deformation audit and six-view media readback in that order.
