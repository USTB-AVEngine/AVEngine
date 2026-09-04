# Generated-animal Blender helper closure

Status: AVEngine-local adapted source, 2026-09-04.

The generated-animal chain uses six small Blender/Python helper files selected
from the retained Eastforward/SPEAR animal tooling. The source carrier was a
retained `SPEAR-lead-b` working tree whose Git pointer was unavailable during
this migration; no source commit or current checkout path is asserted here.
The migration preserves the helper algorithms and their existing support-plane,
semantic-rig, no-clobber, and readback checks. It changes only the import root
and execution location so the chain does not execute a SPEAR checkout for
these helpers.

| AVEngine path | Retained source path | Treatment |
| --- | --- | --- |
| `tools/assets/blender_normalize_generated_animal_heading.py` | `tools/blender_normalize_generated_animal_heading.py` | Selected Blender entry point; local path only |
| `tools/assets/blender_level_generated_animal_support_plane.py` | `tools/blender_level_generated_animal_support_plane.py` | Selected Blender entry point; imports the local three-module closure |
| `tools/assets/blender_retarget_quaternius_to_generated_quadruped.py` | `tools/blender_retarget_quaternius_to_generated_quadruped.py` | Selected Blender entry point; imports the local semantic helper |
| `tools/assets/generated_animal_support_plane.py` | `tools/generated_animal_support_plane.py` | Selected NumPy support-plane implementation |
| `tools/assets/generated_animal_support_plane_contract.py` | `tools/generated_animal_support_plane_contract.py` | Selected stdlib validation constants and serialized-evidence checks |
| `tools/assets/generated_quadruped_semantics.py` | `tools/generated_quadruped_semantics.py` | Selected stdlib semantic hierarchy decomposition |

The three entry points require the installed Blender Python API and NumPy. The
retarget command may read an explicitly supplied Quaternius donor GLB; that is
an external motion asset and is not source code migrated by this change.

`tools/assets/run_generated_animal_chain.sh` now invokes the three entry points
from its own AVEngine checkout. Its old `--spear-root` option remains accepted
for command-line compatibility, emits a deprecation warning, and is ignored;
no helper or import path is taken from that argument. The independent
`--rigger-root` SkinTokens inference step remains external and is deliberately
outside this six-file migration. Pixal3D is also a separate upstream stage and
is not called by this shell script.

The selected source is covered by the SPEAR/Intel MIT terms. Retain
`LICENSES/SPEAR-MIT.txt` with this adapted source and preserve the attribution
when redistributing it. SPEAR terms do not license the Quaternius donor,
Blender, Unreal content, model checkpoints, or generated assets; each keeps its
own license and provenance treatment.

Validation for this slice is limited to shell syntax, Python bytecode syntax,
and focused source-path tests. No Blender, GPU model, UE, or SPEAR process is
started by the tests.