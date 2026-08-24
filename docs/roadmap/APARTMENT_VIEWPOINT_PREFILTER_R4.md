# Apartment viewpoint prefilter r4

## Checkpoint 2026-08-24: reject low-utility raster candidates

The first real builder attempts separated environment failures from candidate
quality:

- r1 stopped before scene creation because the adapter-linked Habitat prefix
  could not load `libRLRAudioPropagation.so`; the builder now accepts and passes
  the explicit external RLR SDK root to the installed runtime.
- r2 stopped before scene resolution because
  `AVENGINE_LEGACY_APARTMENT_PACKAGE_ROOT` was not set. No candidate claim was
  made from either failed run.
- r3 completed native Habitat scene/navmesh/rigid-obstacle loading, but its 2-D
  selection admitted candidates that were unusable for a four-point pilot.

The r3 selected metrics were:

| viewpoint | raster fraction | passed corridors |
| --- | ---: | ---: |
| legacy `view0` | 0.387970 | 5 |
| `window_facing_low` | 0.003008 | 0 |
| `living_reverse_high` | 0.757895 | 4 |
| `kitchen_reverse_high` | 0.003008 | 0 |
| `kitchen_reverse_low` | 0.075188 | 1 |
| `window_facing_high` | 0.009023 | 0 |

I inspected the r3 Topdown at
`/data/jzy/tmp/apartment_viewpoint_builder_04c2b03_r3/`. The three near-zero
coverage views occupied thin or isolated navigable slivers and pointed away
from most of the usable room; the low kitchen view also had only one usable
corridor. Only legacy view0 and living-high visibly had meaningful route
support. This is the concrete failure that justifies the production-builder
thresholds; they are not a generic release gate.

The reusable prefilter API now has
`minimum_passed_corridor_count` with default zero. Both the legacy camera and
generated candidates must satisfy the requested raster-fraction and corridor
thresholds. Filtering occurs before required-role coverage and pairwise
selection, so a missing semantic or height role fails closed rather than
silently substituting a weak view. The builder requests at least `0.10` raster
coverage and four passed corridors, because r3 showed that `0.003--0.075` with
zero or one corridor cannot populate the required four pilot points.

After confirming GPU0 was idle (16 MiB, 0% utilization), r4 ran once with the
same retained room, M3 package, anchor library, corridor bank, installed
Habitat prefix and RLR SDK. It published fresh output at
`/data/jzy/tmp/apartment_viewpoint_builder_04c2b03_r4/` in 21.7 seconds. Of 538
raw candidates, 58 passed candidate-quality filtering. The selected result is:

| viewpoint | roles | raster fraction | passed corridors |
| --- | --- | ---: | ---: |
| `apartment_view0` | base, legacy | 0.387970 | 5 |
| `apartment_living_reverse_low` | living, low | 0.757895 | 4 |
| `apartment_kitchen_reverse_high` | kitchen, high | 0.139850 | 5 |
| `apartment_window_facing_high` | window, high | 0.667669 | 5 |
| `apartment_living_reverse_high` | living, high | 0.300752 | 4 |
| `apartment_kitchen_reverse_low` | kitchen, low | 0.748872 | 6 |

I inspected the r4 Topdown. All six camera markers lie on substantial retained
navmesh regions, and the r3 right-edge isolated window views are gone. The
window-high and living/kitchen views now point through the corridor-supported
interior rather than surviving solely on one semantic focus. Labels overlap in
the dense upper-left area, but coordinates, direction rays and corridor lines
are internally consistent.
A second independent root-agent review inspected the same r4 PNG and agreed
that the isolated/micro-coverage candidates were gone, all six views occupied
substantial navigable regions, and their direction rays pointed inward. Both
reviews remain Topdown-only and make no first-person pixel claim.

r4 is still only a 2-D draft. The PNG and JSON continue to say native Habitat
pixels, vertical FOV, rigid occlusion, first-person composition and lighting are
pending. No Studio or UE was used, no pixel certification or four-point pilot
was performed, and all downstream episode counts remain zero.
