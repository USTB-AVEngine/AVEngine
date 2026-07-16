# M0 Baseline Status

This is the immutable-at-closure M0 snapshot; its M1 `not_run` row records the
state at that earlier gate. For the current post-M1 result, see
[M1_STATUS.md](M1_STATUS.md).

Recorded: 2026-07-16. Status values are only `pass`, `fail`, `blocked`, or
`not_run`. A successful build does not imply that a runtime or scientific
canary passed.

## Source identity

| Item | Recorded value |
| --- | --- |
| Legacy AVEngine migration base | `92775d4d2050a3a9b277357eb83c9243468f4cd3` |
| Legacy SPEAR migration base | `7fbf3632fdb63cc2eceea564811c9597cabfb199` |
| Habitat source baseline/upstream merge base | `57ee4941dc4765240f0f91f70b2c97a919bf9038` |
| Habitat runtime M0 baseline commit | `425fe084eb680844b2b01d86904b9a72c4896d7a` (three documentation-only commits ahead of upstream) |
| Upstream Habitat release tag commit | `v0.3.4` at `24dfda7a746037f7b6b63730a02e3d05a688fff0` |
| Habitat Python distribution metadata | `0.3.3` (unchanged upstream metadata) |
| RLR submodule | `4fd446b4abb5c71fb7a232a083bbddd65f25fc6f` |
| Runtime fork branch | `feature/habitat-native-runtime-foundation` |
| Main branch | `feature/habitat-native-foundation` |

The runtime lock inside the fork records its source baseline rather than a
self-referential commit. This main-repository lock pins the resulting runtime
governance commit explicitly.

## Executed environment

- Ubuntu 22.04, Linux `6.8.0-83-generic`, x86-64.
- g++ 11.4.0; Conda 26.1.1.
- `avengine-habitat-runtime`: Python 3.12.13, CMake 3.27.9, Ninja
  1.13.2, scikit-build-core 1.0.3, pybind11 3.0.4, NumPy 2.3.5,
  numpy-quaternion 2024.0.13, pytest 9.1.1, Git LFS 3.7.1.
- Four NVIDIA GeForce RTX 4090 D GPUs, driver 560.28.03. The build itself used
  `HABITAT_WITH_CUDA=OFF`, `HABITAT_BUILD_GUI_VIEWERS=OFF`,
  `HABITAT_WITH_BULLET=ON` (default), and `HABITAT_WITH_AUDIO=ON`.

## Verification table

| Check | Status | Exact result / reason |
| --- | --- | --- |
| Repository/remotes/merge-base audit | `pass` | origin/upstream roles and exact SHAs verified |
| Recursive runtime submodules | `pass` | initialized; RLR and all recorded pins match |
| Clean headless audio-enabled editable build | `pass` | `pip install -e . --no-build-isolation`, exit 0; Habitat 0.3.3 wheel built and installed |
| Import with upstream quaternion-order workaround | `pass` | `audio_enabled=True`, Bullet true, CUDA false |
| Direct `import habitat_sim` in a fresh process | `fail` | abort exit 134, `free(): invalid pointer` |
| RLR shared-library link resolution | `pass` | binding resolves `$ORIGIN/libRLRAudioPropagation.so`; no missing library |
| Scene-independent original Habitat unit subset | `pass` | 163 passed, 2 CUDA-required skipped, 1 warning |
| Selected scene-dependent `test_random_seed.py` | `pass` | passed after the official test-scene archive was installed |
| Full original test collection | `pass` | 776 tests collected; one torch-dependent test skipped |
| Habitat test-scene acquisition | `pass` | official Meta v1.0 ZIP installed locally; archive SHA-256 `072c053c00d8e49132e48837eea38f8b6cd19e3079c40fd705ffef12347d7302` |
| Official Hugging Face test-scene Git/LFS route | `pass` | initial 504/timeouts were transient; a later `ls-remote` and full shallow clone at `910c783fb954da8497ea5f811b843a76590ddddc` checked out all four LFS objects with `HF_ENDPOINT` unset and Git LFS 3.7.1 |
| MP3D example, example objects, and Locobot inputs | `pass` | official packages declared by the pinned Habitat downloader installed locally; archive hashes are in the runtime lock |
| First complete original pytest run | `fail` | 27 failed, 421 passed, 329 skipped, 26410 warnings in 796.57s |
| Six formerly asset-dependent example cases | `pass` | all passed targeted reruns after the three auxiliary packages were installed |
| Remaining Greedy follower/binding cases | `fail` | 21 failed in 76.63s; `PyCapsule.__next__ returned NULL without setting an exception` in `GreedyGeodesicFollower.find_path()` |
| AudioSensor construction/IR canary | `not_run` | controlled acoustic fixture and known stock hazards require M3/M4 tests |
| M1 visual/room canary | `not_run` | no M1 scene package or capture implementation yet |
| M2 articulated-animal canary | `not_run` | no canonical `canary_qualified` package/runtime adapter yet |
| M3 acoustic-material activation canary | `not_run` | no explicit AcousticScenePackage ingestion yet |
| M4 named multi-source/listener canary | `not_run` | modern RLR C API adapter not implemented |
| M5 timeline semantic canary | `not_run` | schema exists; builder/semantic validator not implemented |
| M5 counterfactual pair | `not_run` | depends on M1-M5 runtime capabilities |
| M6 end-to-end dataset admission | `not_run` | registry/QA/CLI/admission not implemented |

## Exact build and test commands

```bash
env HABITAT_BUILD_GUI_VIEWERS=OFF \
  HABITAT_WITH_AUDIO=ON \
  HABITAT_WITH_CUDA=OFF \
  CMAKE_BUILD_PARALLEL_LEVEL=4 \
  conda run -n avengine-habitat-runtime \
  python -m pip install -e . --no-build-isolation

conda run -n avengine-habitat-runtime python -m pytest -q \
  tests/test_common_utils.py tests/test_configs.py tests/test_controls.py \
  tests/test_noise_models.py tests/test_quaternion_utils.py \
  tests/test_registry_unit.py tests/test_utils.py \
  tests/test_validators_unit.py tests/test_viz_utils.py
```

The expanded suite added `tests/test_random_seed.py`; full collection used
`python -m pytest --collect-only -q`, and the complete run used
`python -m pytest -q`. The persistent `HF_ENDPOINT=https://hf-mirror.com`
setting was removed and Git LFS 3.7.1 installed. The official Hugging Face
route initially remained unavailable, so the exact official Meta test-scene
archive was installed from the URL retained in Habitat's history. A later
official-route retry succeeded for both `git ls-remote` and a full shallow clone
with all four LFS objects checked out. The common GLB/navmesh hashes match the
Meta archive, showing that the earlier 504 was a transient egress/route problem
rather than a remaining mirror or LFS configuration problem. The pinned
downloader, invoked after importing `quaternion`, also installed
`habitat_example_objects`, `mp3d_example_scene`, and `locobot_merged`. These
assets are ignored local test inputs and are not part of either Git repository.

The first full run is retained verbatim rather than retrospectively rewritten.
After the auxiliary downloads, the six data-dependent failures passed targeted
reruns and `python -m pytest --last-failed -q --tb=no` confirmed that the 21
binding failures remain. No evidence yet isolates Python 3.12 or pybind11 as
the root cause.

## Baseline risks carried into implementation

- The pinned stock AudioSensor uses a deprecated one-source/one-listener RLR
  wrapper, indexes a Vector3 at `[3]`, and does not reset all initialization,
  source, and material flags. M3/M4 require executable regression canaries
  before AudioSensor output is trusted.
- The direct-import abort is a real runtime defect despite the upstream test
  workaround and must not disappear from the M1 environment report.
- The Greedy follower PyCapsule iteration failure is also a real runtime defect.
  It is independent of the resolved dataset downloads and needs a focused
  compatibility diagnosis before M1 navigation can be considered healthy.
- The clean build proves compilation/linking only. It does not validate room
  geometry, articulation, acoustic materials, multi-pair IRs, synchronization,
  counterfactual integrity, or dataset admission.
- RLR remains CC BY-NC 4.0; the current audio route is non-commercial.

M0 therefore establishes a reproducible and honest engineering baseline. The
next feature gate is M1; no M1-M7 capability is claimed complete here.
