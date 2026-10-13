# Native quadruped motion repair

The animal_motion package compiles a target's own Walking skeleton into a slow, planted quadruped gait, replays it in Habitat, reads back the mouth and body landmarks, and qualifies the moving RIR against independent native executions. It preserves the target mesh, topology, skin weights and registered identity. The optional static-mesh rigging tool is a research prototype; its output is never admitted merely because export succeeded.

The runtime request API is animal_motion.service.execute_request, exposed by tools/assets/run_animal_motion_request.py. It attempts animation first. Missing animation or a rejected motion may use the registered standing mesh translated along a native route at no more than 0.08 m/s. A failed slide becomes cannot_move. Environmental errors propagate. The resulting motion metadata is retained by source QA evidence and question serialization; answer/truth calculations are unchanged.

## Evidence layers

1. compile_animal_motion produces a flat motion.json with heading, stance drift/speed, contact and 180/90 degree negative controls. flat_pass_native_pending is not native admission.
2. validate_animal_motion_habitat reads actual native transforms and mouth attachments, tests the original posed skin and scene triangles, and verifies the complete silhouette. native_animal_direction_controls actually rotates drawables while retaining the route.
3. prepare_animal_audio_endpoints and qualify_animal_audio_parallel execute fresh native processes for endpoints and three independent references per midpoint. Frequency-dependent endpoint coherence changes the audio renderer, not the original acceptance thresholds. The real convolution and effective impulse response are tested together. A +6 dB waveform must fail the unchanged qualifier.
4. UE export/import/capture tools retain native skeletal, transform, paw-depth and emitter readback. Import success and short smoke captures are explicitly separate from full geometry/audio qualification.

Four-paw gap remains 12 mm; full-skin floor penetration retains the existing 1 mm plus numerical tolerance; non-floor clearance remains greater than 2 cm; native navigation drift remains 2 mm; RIR direction and segment sweep remain 5 degrees. The new heading detector uses 15 degrees and stance checks use 3 mm accumulated drift and 30 mm/s velocity. No registry, source asset, truth or original scorer is modified.

The geometry adapters stage.py and UE adapters ue_floor.py, ue_runtime.py, ue_exposure.py and ue_gpu_policy.py incorporate the task's previously inspected integration/Kujiale implementations as owned modules. They execute installed runtime prefixes and caller-declared inputs; they do not import or execute another agent's worktree. Static rigging records donor motion provenance and preserves target geometry.

Inputs are supplied by private inventory, candidate/semantic anchor, house, runtime and route JSON files. Generic rigs require Idle and Walking, a head/mouth attachment, and four hip-knee-ankle-toe chains in uniform target normalization. The full reproducible 48g handoff, measurements and per-animal receipts are external task artifacts named VALIDATE_RIGGED_ANIMAL.md, PROGRESS.md and JOBS.json. They must be read for current native status; this source document makes no whole-roster or whole-room admission claim.


## Audited target chains and engine consumption

For target skeletons with extra finger, wrist or hock joints, a private inventory record may specify motion_profile.chains. Each of FL/FR/HL/HR lists the target's own hip, knee, ankle and optional toe node names. Intermediate descendants may be skipped, but reversed ancestry and a hip containing another paw or the spine are rejected. Three-node limbs use their terminal foot directly. No registered source file is rewritten.

execute_request verifies the requested route, speed, duration and target source when reusing native evidence. Its motion_scene output carries animal_motion_audio and native_frame_source alongside motion_metadata. The engine's existing v7_registered_motion_audio.moving_stem entry consumes this explicit qualified animal binding: it matches the existing acoustic-package/runtime identity, checks the actual mouth trajectory and listener, reevaluates all three raw native midpoint references and the +6 dB negative control, and runs the frequency-coherent full-tail convolution. Requests without this binding retain their existing renderer path.

UE capture writes the measured frame receipt and MP4 before cleanup. It measures unobstructed pixels against an isolated native animal depth capture, and checks route navigation. The separate UE direction-control tool physically rotates the same native skeletal actor by 180 and 90 degrees; these controls are excluded from admission. Engine cleanup status remains separate from persisted measurements.
