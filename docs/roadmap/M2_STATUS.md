# M2 Status: Articulated Dog Runtime

Formal M2 research-canary status: `pass` on 2026-07-17.

The exact v7/r5 package is `canary_qualified`. Automatic QA, a hash-bound user
visual decision, source-license review, a world-space four-paw contact/root
cadence audit and a clean formal Habitat capture all passed. This is a bounded
research-canary qualification. `formal_dataset_registration_authorized` stays
`false`; M2 does not admit the dog to a released dataset or approve later
appearance/species variants.

## Exit-gate summary

| Gate | Result | Exact scope |
| --- | --- | --- |
| Static, deformation, animation and profile motion QA | `pass` | fixed Rocketbox Beagle visual and Idle/Walk action |
| Legacy hind-leg under-articulation metric | `pass` | no longer triggered by the replacement action |
| Four-paw world contact and root cadence | `pass` | `0.013894547981602673 m <= 0.015 m` |
| Human visual review | `pass` | unchanged visual/action hashes; user accepted rear-leg naturalness |
| Rights/use review | `pass` | Microsoft Rocketbox MIT snapshot; `research_canary` use |
| Package admission | `pass` | `canary_qualified`, not dataset-registered |
| Clean formal Habitat capture | `pass` | 75 states, exactly `view0`, RGB/depth/semantic |

## Final immutable identities

All `tmp/` paths are ignored local evidence and are not distributed by normal
Git history. SHA-256 values identify the exact bytes used for this decision.

| Artifact | SHA-256 |
| --- | --- |
| `tmp/m2/rocketbox_beagle_m2_world_contact_v2/contact_phases.json` | `0d3649be5efb3eae50d955aef536805aa30c78374dfddffdc08171afe6e2bf6f` |
| `tmp/m2/rocketbox_beagle_m2_world_contact_v2/world_contact_audit.json` | `355e52e289dccc202b0d928f4d5969ba6f32c4789b9de7977c3993e912b7a297` |
| `tmp/m2/rocketbox_beagle_m2_candidate_v7_world_contact_r5/asset_manifest.json` | `488b6a00337b0fcb180f3491f207ffddf6cab54c71de88575aad159bd2ad428a` |
| `tmp/m2/rocketbox_beagle_m2_review_request_v7_world_contact_r5.json` | `7de38736116c810be2ce15ac51b29bcfaa5ef64ec1894568fef811c1e09a3386` |
| `tmp/m2/rocketbox_beagle_m2_habitat_review_v7_world_contact_r5/evidence.json` file bytes | `2cfd1e99690f4df393d9b287431aa6299f7f51d07cf727ca508286b4263dd107` |
| `tmp/m2/rocketbox_beagle_m2_canary_v7_world_contact_r5/asset_manifest.json` | `ad5df61f4b955980f6ab3d6d00e44f31942cb31ce3d2f2a3c1a1f353c307e240` |
| `admission/human_visual_review.json` | `ed5cc23694e014c17049df76e7f9bc1d7b192b4351e672f8abf00db321fa47ef` |
| `admission/canary_provenance.json` | `c2360af05f1922f14a5ecd59df6ae9073faba85fd74dc501db6f3458df11b0c6` |
| `tmp/m2/rocketbox_beagle_m2_formal_request_v7_world_contact_r5.json` | `8f77e6034b5ad4389f7b5828e7cd81049e8e50184f46f271bc6273b9fd63b5cc` |
| `tmp/m2/rocketbox_beagle_m2_formal_capture_v7_world_contact_r5/evidence.json` file bytes | `7644c985f93228a515f7f338dbd2791952c7bbb2295af5714fe9cbb73a8baaf3` |
| formal evidence canonical content | `a8bd355f70713feb029d5af9575ba48583277ce3d2d34d122117556268b2d9fb` |

The fixed visual GLB is
`788a667537f7660bac5e128c38c2182453d1d4a9a4f8380343e7a9fa1947538c`.
The Idle and Walk pose artifact is
`b77457be2808fc0495ca7a8bc97978681598afca1caff18aca0761de5891c645`.
These hashes are identical to the user-reviewed r3 candidate; v7/r5 adds the
accepted cadence-bound contact phases and root trajectory without changing the
mesh, skin or joint animation.

## Motion and world-contact result

The replacement action passes the generic profile gates and no longer exhibits
the legacy rear-leg under-articulation signature:

- mean front-paw forward range: `0.21719317695994803 m`;
- mean hind-paw forward range: `0.28810084185901486 m`;
- mean hind-paw lateral range: `0.010916511876523767 m`;
- normalized hind/fore forward ratio: `0.930712`;
- maximum left/right symmetry difference: `0.032122`.

The earlier actor-space sliding warnings were not hidden or waived. A separate
solver detects stance from low paw height plus rearward actor-space velocity,
then fits one constant root step by minimizing the worst world-space contact
residual. For the exact 45 Walk states it selected:

- root step: `0.0198 m/frame` at 15 Hz;
- root speed: `0.29700000000000004 m/s`;
- path length: `0.8712000000000001 m`;
- 21 consecutive stance-frame pairs across four paws;
- maximum residual: `0.013894547981602673 m` under the `0.015 m` gate.

The solver is body-plan neutral: it consumes declared semantic contact anchors
and a forward direction. A species/body-plan profile must still supply valid
anchors and pass its own QA; this canine result cannot silently qualify cats,
birds, horses or other motion families.

## Human review and provenance boundary

The recorded user statement is:

> 视频里面的后腿我觉得已经自然了，所以你可以继续完成M2没完成的地方并提交收尾

The user reviewed the r3 RGB diagnostic at SHA-256
`f789260e70a99b008685377b9d18d239d4bdbf6aa71fd20ccda4f09ee8bf03a9`.
Admission binds that diagnostic, the unchanged v7/r5 visual/action hashes, the
final three review modalities, the passing world-contact audit and the exact
candidate/request/evidence hashes. The final review media are:

| Modality | SHA-256 |
| --- | --- |
| RGB | `97494d7b4e1b10ad51c517dd046a8839a48f391cbab4d3bb7e9cc14b700c1c2d` |
| depth | `8c41bffab002127260a44a086a5a80f5dbb5a512c6003d0c5e6d1cca5825e03e` |
| semantic | `f11f6d9ed911b02cd357d5633de966cee89deea07b689a76f80379a5b3c33fa2` |

The source is Microsoft Rocketbox `Dog_Beagle_01` at revision
`0943055db6ec570bcef9f2c8b41c9e5467c808f9`. Admission snapshots and hashes the
MIT `LICENSE.md` and source `README.md`, sets `allowed_use: research_canary` and
`redistribution: allowed`, and explicitly leaves formal dataset registration
unauthorized.

## Formal Habitat evidence

The final capture loaded the `canary_qualified` package and emitted
`evidence_kind: completed_formal_habitat_capture`, `status: pass`,
`formal_view_ids: ["view0"]`, `review_view_ids: []`, and `review_only: false`.
It proved:

- 75 explicit states: 15 Idle, 45 Walk and 15 Idle, ticks `0..236800`;
- one co-located RGB/depth/semantic rig, not three camera viewpoints;
- one fresh Simulator, zero physics steps and world time `0.0 -> 0.0`;
- all pose/applied-state hashes and array readbacks passed;
- maximum root readback error `2.3841724416940023e-07`;
- maximum joint-quaternion readback error `3.241414114896202e-08`;
- semantic ID 200 visible in all frames, with 887 to 962 pixels.

| Formal array | Shape | SHA-256 |
| --- | --- | --- |
| RGB | `75 x 240 x 320 x 4` | `8cc7ddf56a9694385f4d3170afe680cbdb970afea4232e422365701cfb746ee0` |
| depth | `75 x 240 x 320` | `bca2060404ab05ac3813911c22e84f396e3e2f89ee60809c0a85645b63b6d962` |
| semantic | `75 x 240 x 320` | `baa9dc0c3176352f5253f967ae69dd10f4becc9de189b0590c905f700d4be5c0` |

Capture ran with clean AVEngine commit
`b3d3a63055d5ec5017824148968644c7f11fe631` and clean Habitat runtime commit
`bcca512aa58e8b2819454716b710ef3da72f7f47`. The imported native binding SHA-256
`06079ce3a06053088e921a1852a98fba10a5409009f8502c44ed03f75dbd1211`
matched `runtime.lock.yaml`, and the binary originated from the locked runtime
root.

## What remains outside M2

M2 qualifies one fixed Beagle research canary, not an arbitrary animal
generator. Appearance attributes, nine Beagle realizations, cat/horse/Golden
Retriever assets and real-room coverage are the next goal and require new
per-instance evidence. Acoustic propagation, multi-source RLR, authoritative
audio/visual timeline admission and dataset registration remain M3-M6 gates.

The executable admission and formal-capture procedure is in
[M2_EXECUTION.md](M2_EXECUTION.md).
