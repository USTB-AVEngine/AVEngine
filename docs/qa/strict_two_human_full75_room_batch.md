# Strict full75 room-batch staging

Outcome: a same-room, one-process, ten-Episode design is technically feasible,
but it is intentionally **not yet authorized**.  The existing Apartment runner
already loads one SPEAR process and camera once, then creates and destroys fresh
actor hierarchies for multiple scenarios.  The remaining unknown is stricter:
the object-ID segmentation service must survive a terminate/reinitialize cycle
without retaining a prior Episode's skeletal proxies.  A two-Episode runtime
canary is therefore the hard GO/NO-GO gate.

No GPU was used and no A remote file was changed while producing this staging
area.

## Proposed execution topology

One production shard contains exactly ten Episodes from one native map.  It has
one GPU producer and one low-priority CPU consumer:

1. Validate all ten suite plans, 75-frame timelines, authoritative WAVs,
   complete exact-RIR caches and binaural delivery receipts before opening UE.
2. Launch the room once on physical GPU1 and create one multimodal camera.
3. For each Episode, create fresh actor anchor/visual hierarchies, assign
   Episode-specific stable names, run the normal 75-frame pass and two 75-frame
   target-only passes, and retain all 225 runtime readbacks plus the live asset
   readback.
4. Write the GPU-facing result to four preallocated raw memmaps.  Flush and
   `fsync` every data and metadata file; publish `RAW_READY.json` last via a
   same-directory atomic rename and parent-directory `fsync`.
5. Destroy visual actors before anchors, terminate segmentation, wait two
   frames, and negatively prove that old actor handles and stable names are
   absent.  Reinitialize an empty segmentation service, prove that no old proxy
   descriptors remain, then terminate it again.  Any residue stops the shard.
6. While the GPU producer captures the next Episode, a single CPU worker at
   nice +10 and two threads per numeric library consumes the preceding raw
   receipt.  Queue depth is exactly two; reaching it applies backpressure.
7. The CPU worker derives depth-authority masks/truth, writes compressed NPZ,
   encodes/muxes the videos, validates hashes and counts, then atomically
   publishes the per-Episode `FINAL_READY.json`.  Raw is never counted as a
   passing Episode.
8. The room process closes after the last raw capture.  The batch can publish
   `BATCH_READY.json` only after all ten independent final receipts validate.

The immutable raw arrays per Episode are:

| File | Shape | Dtype | Exact bytes |
| --- | --- | --- | ---: |
| `normal_depth_m.f16le` | 75×720×1280 | little-endian float16 | 138,240,000 |
| `target_only_source1_depth_m.f16le` | 75×720×1280 | little-endian float16 | 138,240,000 |
| `target_only_source2_depth_m.f16le` | 75×720×1280 | little-endian float16 | 138,240,000 |
| `normal_object_ids.u32le` | 75×720×1280 | little-endian uint32 | 276,480,000 |

The fixed total is 691,200,000 bytes per Episode.  The observed dynamic canary
adds about 77.9 MB of RGB PNGs, so one ten-Episode raw shard needs about 7.7 GB
before metadata.  Reserve 10 GB of working space per active shard.  Raw spools
remain until the complete batch validates; deleting them is a separate,
explicitly authorized destructive action.  Keeping raw for all 100 Episodes
would add roughly 77 GB.

## Checkpoint and resume semantics

Each Episode writes to a new `attempt_NNN` directory.  Nothing is overwritten.
A crash before `RAW_READY.json` leaves a preserved partial attempt that is
ignored on resume; capture restarts in the next attempt directory.  A complete
raw receipt can be finalized again without opening SPEAR.  A passing final
receipt can be skipped only after its request digest, complete input-binding
digest and strict 75/75/75/75 readback contract validate again.

Native failures stop before another Episode begins.  CPU failures stop capture
as soon as observed; because finalization overlaps native capture, at most the
single Episode already inside the capture call can finish raw capture after the
CPU failure occurs.  Neither it nor any other raw artifact becomes qualified.

## Efficiency estimate

The latest dynamic canary took 82.54 seconds end to end.  The first RGB arrived
17.45 seconds after launch.  A naive shared process that only removes repeated
startup is therefore about 1.24× faster for ten Episodes.

Separating raw capture from CPU finalization removes NPZ compression, mask
derivation, video encode/mux and artifact hashing from the GPU producer's serial
path.  Based on artifact timestamps—not a shared-process measurement—the
expected range is about 442–497 seconds for ten Episodes, or 1.66–1.87×.  A
pessimistic model is about 552 seconds, or 1.49×.  These numbers must be replaced
by the two-Episode canary timing before scheduling 100 Episodes.

If segmentation reset fails, use the safe fallback: one fresh packaged process
per Episode on GPU1, emitting the same raw receipt, plus one asynchronous CPU
finalizer.  It retains startup cost but should still be around 1.38× by
overlapping CPU work; it never reuses segmentation state.

## Staged files

- `tools/qa/run_strict_two_human_full75_room_batch.py`: request validator,
  no-clobber checkpoint/resume state machine and fail-closed batch ledger.
- `tools/qa/capture_spear_native_pixel_room_batch.py`: concrete one-process
  Apartment session adapter. It reuses the existing native pixel/ground-snap
  primitives without modifying them, but execution requires both `--execute`
  and an explicitly authorized request.
- `tools/qa/strict_two_human_raw_spool.py`: exact memmap writer and atomic raw
  receipt publication.
- `tools/qa/strict_two_human_cpu_finalize_queue.py`: one-process, nice +10,
  two-thread-capped CPU worker adapter used behind queue-depth-two backpressure.
- `tools/qa/finalize_strict_two_human_raw_episode.py`: CPU-only mask/truth,
  compressed-array, video/mux, strict pixel/runtime gate and final-receipt stage.
- `tools/qa/spear_room_batch_lifecycle.py`: shared-camera, fresh-actor and
  segmentation reset/negative-existence lifecycle proposed for the runtime
  canary.
- `examples/qa/native_strict_two_human_room_batch_reset_canary_v1.json`: two
  existing, fully precomputed static Episodes bound to a new no-clobber output;
  `execution_authorized` remains false.
- `artifacts/full75_room_batch_benchmark_estimate_v1.json`: observed timing
  anchors, transparent projections and storage budget.
- `tests/unit/test_strict_two_human_full75_room_batch.py`: CPU-only contract,
  raw publication, stop-on-failure and resume tests.

## Integration order

1. Extract the GPU-facing body of the current single-Episode capture after
   instance/camera creation into an in-process `capture_episode_raw` adapter
   that uses the staged spool and lifecycle helpers; its existing CPU-facing
   tail is represented by the staged raw finalizer.
2. Run unit tests in A's normal test environment.
3. Dry-resolve the two-Episode canary and perform an independent request/path
   audit.
4. Only with explicit GPU authorization, run those two Episodes in one process.
   Inspect frame 0/37/74 for both Episodes and require zero old handles, stable
   names and proxy descriptors after Episode 1.
5. If it passes, bind its `BATCH_READY.json` into ten-Episode production
   requests.  If it fails, select the process-per-Episode producer fallback.

The canary and any later shard retain `formal_episode_count=0` and
`qualification_claim=false`.  Batch reset success does not clear the separate
ground-contact release blocker; the raw capture context records that boundary
for every Episode.

Dynamic rows have a second independent release blocker. A production request
cannot resolve them from a 75-frame trajectory alone: each row must bind a
motion-realism receipt proving native-rate active-interval mapping (no global
time stretch), measured active speed for every moving actor, monotonic clip
phase, synchronized foot plants, and no phase freeze. The existing stretched
dynamic pipeline canaries do not meet this contract and cannot enter first20 or
release accounting.
