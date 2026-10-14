# Owner-selected cat motion policy — 2026-10-14

The explicit `--validation-policy owner_relaxed_20261014` option applies the owner's 09:3x SGT update to the existing cat binding candidates. The default remains `strict`.

| Check | Owner policy |
| --- | --- |
| Body heading versus actual movement | At most 20 degrees |
| Paw contact distance | At most 12 mm |
| New penetration across semantic regions | At most 400 mm² per frame, summed across region pairs |
| New cross-region penetration depth | At most 10 mm |
| Intersections within the same semantic region | Recorded; excluded from rejection |
| Support-point drift during a step | At most 15 mm |
| Triangle edge growth | At most 2.0 times the standing mesh |
| Per-asset moving-source audio | Deferred to the production question generator |

The relaxed surface check requires the actual float32 vertex stream used by the native renderer. Native floor, stage clearance, visibility, mouth attachment, complete stance/swing cycles, finite geometry and seam checks remain in place. Native 180-degree and 90-degree drawable controls must both be rejected.

A `motion_qualified` result describes motion only. It does not assert completed audiovisual admission. Previously completed audio measurements remain attached to their original trajectories. Existing complete native receipts may be re-evaluated under the new policy; these receipts identify the original run and explicitly state that no fresh native execution occurred.

Reproduction uses the surface checker, bounded native validator and native direction-control tool with this explicit policy. The unit regression checks the changed heading, step drift, cross-region area/depth and intraregion treatment, including rejection examples. Results, images, runtime scripts and private absolute paths stay in the task artifact directory outside Git.
