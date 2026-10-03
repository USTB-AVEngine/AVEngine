# CPU training house selection

The training pool allows known SO/M1/v6 exposure. Reserve every clean original
HM3D validation house and current development/test house. HM3D candidates must
pass the original whole-house gate, frozen room rules and smy `use` annotation.
Selection reads the complete room registry, rather than the original clean-only
acoustic candidate list.

The MP3D adapter joins `.house` O/C/R records and semantic PLY `object_id`.
Raw Z-up coordinates become Habitat `(x,z,-y)`. Ground-face unions determine
measured scope; region boxes are identity metadata. Existing native navmeshes
are read only. Each floor records native snap and a downward raw-scan probe
before navigation-grid and placement measurements. The shared room decisions,
98% floor dominance, scope masks and frozen parameters remain unchanged.

The CPU overhead image uses real raw-scan geometry and texture pixels.
MP3D `GOOGLE_texture_basis` assets require `AVENGINE_BASIS_CPU_DECODER` pointing
to the separately built `basis_cpu_decode.cpp` executable. Link that AVEngine
adapter against the external [Basis Universal transcoder](https://github.com/BinomialLLC/basis_universal)
(Apache-2.0); it uses no renderer or GPU. Keep the external source, license,
compiler command, revision and executable under the explicit output root.
The decoder streams the original compressed bytes through stdin and retains
encoded raster rows and the Basis Y-flipped diagnostic; it writes no scene
assets or shared caches. Missing
texture evidence remains unknown and cannot pass the black-pixel rule. These
CPU images do not establish equivalence to production Habitat RGB. MP3D has no
smy annotations and all selected rooms remain pending smy review.

`tools/acoustics/train_house_pool.py` prepares manifests from explicitly supplied
registries and compares two first-batch calibration houses.
`python -m tools.rooms.room_selection.mp3d_train_pool` registers and measures
MP3D using an explicit inventory, output root and frozen threshold file.
`tools/acoustics/summarize_train_house_pool.py` prepares acoustic sampling and
combines results in the first-batch CSV/JSON schema. It also emits exposure
evidence, the final holdout audit and the predetermined acoustic room scope.

Keep native acoustic code at the requested first-batch Git revision. Sample at
most three rooms before seeing acoustic outputs; exposure never affects
acceptance. Preserve the six previous MP3D conclusions. Native input failures
remain review items; do not remove degenerate faces to force a load.

Use an installed Habitat runtime prefix, its matching Magnum Python site,
MP3D input root and RLR SDK via the existing `AVENGINE_*` runtime options.
Set `CUDA_VISIBLE_DEVICES` empty, bind room workers to a small CPU allocation
and limit external BLAS/OMP threads. RLR retains its original eight-thread
setting and must have a separately budgeted CPU allocation. Store all outputs,
temporary files, logs and native compilation products under the run output
root, outside the source checkout and shared dataset directories.

Reports distinguish rule-pass candidate room counts from acoustically sampled
room counts. House-level acoustic diagnosis cannot certify every unmeasured
room, production visuals or future source/listener positions.
