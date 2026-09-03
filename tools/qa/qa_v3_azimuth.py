"""The one place a camera-frame azimuth becomes a published azimuth.

The engine, the solver, the clearance table and the acoustic ILD/ITD check all
work in a right-positive camera frame.  Published questions use the DCASE FOA
convention instead -- +x front, +y left, azimuth positive to the left -- which
is what the field expects and what this project's own Spatial-Omni prompts
already say ("azimuth is in [-180, 180] degrees with positive values to the
left").  Before this module the two projects were mirror images of each other,
so any cross-project comparison silently negated one side.

The conversion happens here, at the publication edge, and nowhere else.  A
global flip was rejected on purpose: ``avengine.timeline.metrics`` derives
``expected_ild_sign`` from the sign of a *listener-local* azimuth, which is
numerically equal to the camera azimuth today because the listener sits at the
camera pose, but is a different quantity.  Negating it would invert both the
expectation and the measurement, so the acoustic check would pass forever
while looking green -- worse than having no check.

Every published azimuth therefore travels with CONVENTION, and any consumer
that finds a number without one should refuse to use it rather than guess.
On 2026-09-03 owner answered two calibration items on the assumption that
right was 0 and straight ahead was 90; the same three answers scored a median
error of 48.97 deg under one reading and 30.0 deg under the other, which is
what an unlabelled angle measures.
"""

from __future__ import annotations

CONVENTION = "dcase_foa_left_positive"
CONVENTION_NOTE = (
    "DCASE FOA relative to the camera: +x front, +y left, +z up; azimuth in "
    "[-180, 180] degrees, positive to the left"
)
ENGINE_FRAME_NOTE = "right-positive camera frame; internal only, never published"


def to_published_deg(engine_frame_deg: float) -> float:
    """Camera-frame right-positive degrees -> published left-positive degrees."""

    published = -float(engine_frame_deg)
    # Keep -180 out of the published range so the two ends never both appear.
    return 180.0 if published == -180.0 else published


def to_published_band(engine_band) -> tuple[float, float]:
    """A right-positive [lo, hi) wedge -> the same wedge, published.

    Negating swaps the ends, so the half-open side moves with them: a negated
    ``[a, b)`` is really ``(-b, -a]``.  This returns an ordered ``[lo, hi)``
    pair anyway, which re-closes the interval on the other end and would move a
    candidate sitting exactly on a band edge into its neighbour.  That is safe
    only because the solver keeps designed azimuths ``DESIGN_EDGE_MARGIN_DEG``
    (0.25 deg) clear of every band edge, so no candidate ever sits on one.  If
    that margin is ever removed, this function has to carry the closed end
    explicitly instead.
    """

    lo, hi = (to_published_deg(v) for v in engine_band)
    return (hi, lo) if hi < lo else (lo, hi)


def side_word(published_deg: float) -> str:
    """Which side of your facing direction, in the published convention."""

    return "left" if float(published_deg) > 0.0 else "right"


def landmark_sentence(half_fov_deg: float) -> str:
    """The convention, spelled out, for the stem of every azimuth question.

    owner 2026-09-03 asked for the landmark angles to be named rather than
    implied: "在 prompt 里，就把所有的主要度数说清楚分别在哪个角度就行".
    """

    edge = abs(float(half_fov_deg))
    return (
        "Use the DCASE FOA coordinate system relative to the camera: +x front, "
        "+y left, +z up. Azimuth is in [-180, 180] degrees with positive "
        "values to the left: 0 degrees is straight ahead, +90 is directly to "
        "your left, -90 is directly to your right, and +/-180 is directly "
        f"behind you. The left and right edges of the video frame are about "
        f"+{edge:g} and -{edge:g} degrees."
    )


def published_block(engine_frame_deg: float) -> dict:
    """The published number together with the convention it is expressed in."""

    return {
        "azimuth_deg": round(to_published_deg(engine_frame_deg), 3),
        "convention": CONVENTION,
        "convention_note": CONVENTION_NOTE,
    }
