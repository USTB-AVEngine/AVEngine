# MP3D full-catalog native-probe subset (2026-08-24)

## Owner boundary

This slice consumes a complete target-agnostic legal camera-placement catalog
after enumeration. It never changes placement legality. For every camera it
joins the retained 75-frame, two-source Habitat emitter-link readback and emits
columnar per-frame listener-local azimuth, elevation, distance, camera depth,
normalized pinhole centre coordinates and centre-frustum flags. Native
occlusion and semantic pixels remain explicitly pending.

The post-hoc join ignores semantic room names and targets. It has no visibility,
60 cm separation, side-margin, crossing, exchange, off-screen or diversity
gate. Zero-coverage placements remain in the full descriptor JSONL. Source
identity is bound through the existing native capture receipt because the NPY
array shape alone cannot distinguish a silent source-axis swap.

## Probe-only selection

The requested twelve native probes are selected by a deterministic greedy
continuous score. Analytic quality prefers the lower of the two source
centre-frustum coverage fractions, their mean, and a small continuous distance
preference. Novelty is a weighted continuous combination of horizontal camera
distance, circular yaw difference, coverage-profile Jaccard distance and camera
height difference. No term is a pass/fail threshold, and unselected cameras are
not rejected or removed.

Each selected placement receives an independent validated M1 request. This
does not reuse or certify RIR identity; any later listener position must produce
its own audio/RIR evidence.

## Full native floor authority correction

The older catalog at
`/data/jzy/tmp/mp3d_camera_placement_catalog_c9c2274_r1` used the floor exporter
default limit of 128 samples and therefore was not the complete discrete legal
domain. The current integration reran the native floor export without that
limit: 632 candidates passed and 228 failed. The expanded unfiltered catalog is:

`/data/jzy/tmp/mp3d_camera_placement_catalog_all_8c4e7df_r1`

It contains 22,752 placements: 632 legal floor points x low/base/high camera
height x twelve 30-degree yaw values. The clean-code real descriptor/subset run
must consume this expanded catalog. The older 4,608-placement artifact remains
historical evidence only.

## Verification boundary

Focused CPU tests and the first fresh expanded-catalog artifact are pending the
implementation commit. Expected outputs are one complete descriptor JSONL,
one compact source-track record, one subset summary and twelve independent M1
requests. No Habitat, GPU, native pixel capture, occlusion, audio/RIR,
mini-pilot, viewpoint certification or formal Episode runs in this slice.

No Studio placement, hash, frozen contract, baseline or permanent gate is
introduced. The artifact stays `research_only=true`,
`episode_counted=false`, and formal release remains `not_run`.
