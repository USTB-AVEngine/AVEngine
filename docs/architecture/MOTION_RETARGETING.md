# Offline Motion Retargeting

Status: bounded offline research infrastructure with one executed Beagle
replacement promoted to `canary_qualified`. Its automatic motion/package QA,
world-contact/root-cadence gate, hash-bound visual review and clean formal
Habitat capture passed. This closes the M2 research canary but does not
authorize dataset registration or another asset instance.

AVEngine retargets motion while compiling an animal asset. Habitat never
chooses a source action, guesses a skeleton correspondence or runs an online
retarget solver. The runtime receives only an already audited target skeleton
and explicit root/joint states.

The current implementation establishes three reusable pieces:

- rest-aware world-space quaternion math;
- versioned body-plan and motion-family profiles; and
- body-plan-neutral motion measurements driven by profile-owned thresholds.

These pieces are necessary but not sufficient for admission of a new asset.
Contact, scene root speed, deformation, Habitat playback and hash-bound human
review remain independent gates; the final M2 Beagle passed all of them for
its exact hashes.

## Compilation boundary

```text
audited source motion + source skeleton
  + target-authoritative mesh/skin/rest skeleton
  + exact motion-retarget profile
  -> offline rest-aware retarget
  -> output GLB and hash-bound retarget evidence
  -> independent GLB round-trip and M2 action bake
  -> generic profile-bound motion QA
  -> body-plan contact and root-speed QA
  -> per-instance deformation/runtime review
  -> research-candidate package
  -> hash-bound human review
  -> possible canary_qualified admission
  -> clean formal Habitat canary capture
```

Successful retargeting never skips a later arrow. In particular, a generated
GLB or a passing motion report remains a `research_candidate` and carries
`qualification_claim: false`.

The target is authoritative for mesh geometry, UVs, materials, skin weights
and rest skeleton. Retargeting may create new animation data, but it must not
silently substitute source geometry or source skinning. An exported GLB must
be independently reopened and re-audited; an in-memory Blender comparison is
not enough to prove the exported bytes and accessors are acceptable.

## Rest-aware world-left rotation transfer

All equations below use Hamilton quaternion multiplication, `xyzw` storage and
world-space rotations. Let:

- `q_s_rest` be one source joint's world rotation in the source rest pose;
- `q_s(t)` be its sampled source world rotation;
- `q_t_rest` be the mapped target joint's world rotation in the target rest
  pose;
- `b` be the audited motion-basis rotation from the source motion frame to the
  target motion frame; and
- `a` be the declared motion amplitude in `[0, 1]`.

The source world-space delta is

```text
delta_s(t) = q_s(t) * inverse(q_s_rest)
```

The optional basis change is a quaternion conjugation:

```text
delta_aligned(t) = b * delta_s(t) * inverse(b)
```

Amplitude scales the motion, not the target rest pose:

```text
delta_scaled(t) = slerp(identity, delta_aligned(t), a)
```

The target world rotation is then

```text
q_t(t) = delta_scaled(t) * q_t_rest
```

This is called **world-left** transfer because the motion delta
pre-multiplies the target rest rotation. The historical rest-local/right-delta
construction used a different non-commutative order. It could turn a normal
sagittal limb swing into lateral motion when source and target bone rolls
differed. At a source rest pose, `delta_s` is exactly identity, so the result
is exactly `q_t_rest`, even when the two rigs use different rest axes.

Every input and output quaternion is finite, normalized and placed in a
deterministic hemisphere. `q` and `-q` therefore cannot create different
hashes. A zero quaternion, non-finite component or amplitude outside `[0, 1]`
fails closed.

The target hierarchy is evaluated parent first. Desired world rotations are
converted into the corresponding target-local keyed rotations through the
actual target parent transforms. Unmapped target joints stay at their authored
rest-local transforms. Root handling is explicit: the current bounded profile
keeps root rotation and translation at target rest; a future profile may use
the declared `retarget_world_delta` root-rotation policy, but undeclared root
motion is never inferred.

Source seconds are preserved. Output and QA sample rates are explicit divisors
of the 48 kHz authority clock. Retarget sampling does not create a free-running
Habitat animation clock.

## Adapter and motion-family boundary

The quaternion equations contain no concept of a paw, wing or fin. A profile
owns all semantic choices:

- source and target skeleton identities;
- one-to-one semantic joint mappings;
- semantic chains and target end effectors;
- action-name mappings;
- source-to-target motion basis and root policy;
- body-plan coordinate axes;
- motion-family-specific QA thresholds; and
- the allowed instance attribute domain.

`body_plan_id`, `motion_family_id` and `adapter_id` have different scopes.
Sharing a broad body plan does not make two gaits interchangeable. For
example, a canine, feline and ungulate may all use a generic quadruped math
adapter while requiring different motion-family profiles, joint maps,
end-effectors and thresholds.

Current fail-closed support is:

| Body plan or mode | Required boundary | Current availability |
| --- | --- | --- |
| Dog ground locomotion | Exact canine profile, audited source skeleton and target template | A bounded Dog-to-Beagle profile and one qualified Beagle canary exist; neither automatically qualifies another instance |
| Cat ground locomotion | A separate feline motion-family profile and feline canary | Not available; a dog profile must not be reused |
| Other quadrupeds | Species/family-specific profile, such as an ungulate gait | Not available unless separately registered and audited |
| Bird ground locomotion | `avian_biped_locomotion_v1`, two-foot contact and walking/hopping QA | Reserved and rejected by the loader |
| Bird flight | `avian_flight_v1`, wing-cycle, flight-axis and no-ground-contact QA | Reserved and rejected by the loader |
| Fish swimming | `fish_swim_v1`, axial-wave, fin-phase and no-ground-contact QA | Reserved and rejected by the loader |
| Unknown adapter or body-plan/profile mismatch | No fallback | Rejected |

Reserved adapter names are not partial support. Until their solver policy,
semantic profile, tests and real canary exist, loading them fails closed. The
system must never make a canine walk "work" on a cat, bird or fish by matching
similar-looking bone names.

## Generic motion QA

The generic QA layer receives already resolved semantic-chain samples in a
profile-declared `(forward, lateral, vertical)` actor frame. It has no hard
coded dog joint names. For every required chain it can measure:

- terminal forward, lateral and vertical excursion in meters;
- the same excursions normalized by target rest-chain length;
- lateral-to-forward excursion ratio;
- per-joint geodesic angular excursion;
- maximum per-joint angular speed, including a declared cyclic boundary;
- caller-declared group means, such as fore versus hind chains;
- ratios between group/axis excursions; and
- left/right or other declared chain symmetry.

Missing chains or joints, inconsistent sample counts, non-finite arrays,
non-unit quaternions, invalid rest lengths, malformed thresholds and undefined
required ratios produce a deterministic failing report. A report always
records that formal dataset registration is not authorized.

Thresholds belong to the exact motion profile. A Dog-to-Beagle threshold is
not evidence for a feline, bird or fish profile. Raw meter metrics are kept for
physical interpretation, while rest-length-normalized metrics support bounded
comparisons across target sizes.

## Independent quality gates

Motion plausibility, support physics and scene travel must not be collapsed
into one boolean:

| Gate | What it establishes | What it does not establish |
| --- | --- | --- |
| Generic motion QA | Mapped chains move with bounded excursion, speed, ratios and symmetry | Correct foot plants, floor clearance or actor travel speed |
| Contact QA | Body-plan-specific support/swing phases, anchor sliding, penetration and hovering | Appropriate scene trajectory or visual skin deformation |
| Root-speed QA | The explicit `world_from_actor` trajectory has a speed/cadence compatible with the selected action and contacts | Limb geometry, weights or contact labels |
| Deformation QA | The realized mesh remains finite, connected and visually plausible over the action | Contact truth or qualification |
| Habitat fixed-state QA | The declared root/joint state is applied and read back without clock advancement | Anatomical acceptability |
| Human visual review | The exact hash-bound media is accepted for the bounded canary | Dataset-wide approval or reuse by another motion family |
| Provenance/use review | Source and derivative use is allowed for the declared purpose | Technical motion quality |

In particular, a stationary-root action can pass generic limb-motion QA while
sliding badly once a scene trajectory is applied. Conversely, a plausible
root speed cannot rescue an incorrect gait. Contact and root-speed evidence
must bind the exact retargeted action, target skeleton, profile and trajectory.

## Attribute domain and per-instance validation

A motion profile declares only a bounded compatibility domain. The current
attribute axes are:

| Attribute | Canonical domain | Interpretation |
| --- | --- | --- |
| `size` | `small`, `medium`, `large` | Semantic category; realized metric size must be measured later |
| `body_build` | `slim`, `standard`, `stocky` | Bounded morphology variation within the target template family |
| breed coat | exactly three unique values from one breed-specific `coat_profile_id` | Names and transforms are breed-specific, never a cross-breed color vocabulary |
| `life_stage` | `young`, `adult`, `senior` | Semantic life-stage category, not a fabricated exact age |

For the current Beagle example, the three coat values are
`light_tricolor`, `standard_tricolor` and `dark_tricolor`. A Golden Retriever,
cat or other breed needs its own three-value coat profile; it cannot silently
reuse the Beagle labels.

An allowed attribute value is a generation/compilation contract, not evidence
that an output realizes it. Every concrete instance must be revalidated after
its attributes are applied, even when it shares a skeleton and motion profile
with an accepted instance. At minimum, per-instance evidence must recheck:

1. request/profile hashes and exact source/target template identity;
2. realized topology, rest skeleton, skin weights and target-authority hashes;
3. physical size and grounding after actor scale;
4. body-build geometry, breed-specific coat/PBR realization and life-stage
   cues against the immutable instance request;
5. retargeted action round-trip, generic motion QA and deformation QA using
   the realized rest lengths and mesh;
6. contact phases and root-speed compatibility for the exact trajectory;
7. Habitat fixed-state playback and co-located `view0` evidence; and
8. a new hash-bound human review and provenance/use decision.

Coat-only changes may leave joint arrays unchanged, but they still require
identity, material and visual verification. Size, body build and life-stage
changes can alter rest lengths, ground clearance, deformation and visible
motion, so a profile-level canary cannot automatically approve their instances.

The broader immutable instance/request evidence model is documented in
[controlled_source_asset_attribute_workflow.md](../controlled_source_asset_attribute_workflow.md).

## Current M2 status and non-claims

The world-left route produced the Rocketbox Beagle replacement action and
identical repeat-build GLB bytes. Generic motion QA passed: fore-left/right
forward excursion is `0.214509/0.219877 m`, hind-left/right forward excursion
is `0.283474/0.292728 m`, hind lateral excursion is
`0.007053/0.014780 m`, the rest-length-normalized hind/fore forward ratio is
`0.930712`, and maximum left/right symmetry difference is `0.032122`.

The actor-space contact warnings were retained as diagnostics, then resolved
by a separate world-space audit that binds the exact action to a constant
`0.297 m/s` root trajectory. It measured a maximum four-paw contact step of
`0.013894547981602673 m` under the `0.015 m` gate. The unchanged visual/action
hashes passed user review, the package was promoted to `canary_qualified`, and
the clean 75-state formal Habitat capture passed on exactly `view0`.

This result is limited to the exact M2 Beagle research canary. It does not
approve a new appearance realization, another canine, a feline/ungulate/avian
profile, or formal dataset registration. Each must repeat the applicable
per-instance gates above.

See [M2_STATUS.md](../roadmap/M2_STATUS.md) for exact final identities and
[M2_EXECUTION.md](../roadmap/M2_EXECUTION.md) for the admission/formal runbook.

## Implementation references

- Generic math: `src/avengine/motion/math.py`
- Strict profiles and adapter availability: `src/avengine/motion/profiles.py`
- Generic motion QA: `src/avengine/motion/qa.py`
- Profile schema: `schemas/motion_retarget_profile_v1.schema.json`
- Current bounded profile:
  `examples/m2/motion_profiles/quadruped_dog_to_rocketbox_beagle_v1.json`
- Offline Blender compiler: `tools/motion/retarget_blender.py`
- Profile-bound M2 motion audit: `tools/motion/audit_m2_retarget.py`
