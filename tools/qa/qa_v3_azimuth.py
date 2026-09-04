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
ENGINE_CONVENTION = "engine_right_positive"
_CONVENTION_ALIASES = {
    "dcase_foa_left_positive": CONVENTION,
    "left_positive": CONVENTION,
    "left_positive_deg": CONVENTION,
    "engine_right_positive": ENGINE_CONVENTION,
    "right_positive": ENGINE_CONVENTION,
    "right_positive_deg": ENGINE_CONVENTION,
}


def canonical_convention(value=None) -> str:
    """Resolve a profile convention without applying a global sign flip."""
    if value is None:
        return CONVENTION
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return _CONVENTION_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"unknown azimuth convention {value!r}") from exc


def convention_note(convention=None) -> str:
    resolved = canonical_convention(convention)
    return CONVENTION_NOTE if resolved == CONVENTION else ENGINE_FRAME_NOTE


def to_convention_deg(engine_frame_deg: float, convention=None) -> float:
    resolved = canonical_convention(convention)
    if resolved == CONVENTION:
        return to_published_deg(engine_frame_deg)
    return float(engine_frame_deg)


def to_convention_arc(engine_arc, convention=None):
    from qa_v3_arc import Arc

    if not isinstance(engine_arc, Arc):
        raise TypeError("engine_arc must be a qa_v3_arc.Arc")
    resolved = canonical_convention(convention)
    return to_published_arc(engine_arc) if resolved == CONVENTION else engine_arc


def convention_block(engine_frame_deg: float, convention=None) -> dict:
    resolved = canonical_convention(convention)
    return {
        "azimuth_deg": round(to_convention_deg(engine_frame_deg, resolved), 3),
        "convention": resolved,
        "convention_note": convention_note(resolved),
    }


def to_published_deg(engine_frame_deg: float) -> float:
    """Camera-frame right-positive degrees -> published left-positive degrees."""

    published = -float(engine_frame_deg)
    # Keep -180 out of the published range so the two ends never both appear.
    return 180.0 if published == -180.0 else published


def wrapping_band(band) -> bool:
    """Whether a ``[lo, hi)`` pair is a wedge that crosses +-180.

    A wedge is stored as an ordered pair, which cannot express one that wraps:
    ``[170, -170)`` and ``[-170, 170)`` are the same pair once ordered, but one
    is a 20-degree wedge behind the listener and the other is the 340 degrees in
    front of it.  Every consumer here therefore refuses a wrapping wedge rather
    than silently publishing its complement.  Answers stayed inside the camera
    cone until 2026-09-03, so nothing could wrap; the owner's ruling that day
    opened the answer range to the full circle, which makes this reachable.
    Giving the wrapping case a representation is part of building that family --
    a start plus a signed sweep, say -- and is deliberately not guessed here.
    """

    lo, hi = (float(v) for v in band)
    return hi < lo


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

    if wrapping_band(engine_band):
        raise ValueError(
            f"engine band {tuple(engine_band)} crosses +-180; an ordered "
            "[lo, hi) pair cannot express a wrapping wedge, and publishing it "
            "would return the complement. Give the wedge a wrap-aware "
            "representation before using it (see wrapping_band)")
    # Interval endpoints must retain -180 and +180 as distinct boundaries.
    # Point normalization maps -180 to +180, which would turn [135, 180]
    # into its 315-degree complement if applied before sorting the bounds.
    lo, hi = (float(v) for v in engine_band)
    return (-hi, -lo)


def to_published_arc(engine_arc):
    """Convert a signed engine-frame :class:`qa_v3_arc.Arc` at the edge.

    Negating both the start and sweep preserves a seam crossing and a sweep
    wider than 180 degrees.  Converting the endpoints independently would
    collapse those cases into the complementary wedge.
    """
    from qa_v3_arc import Arc

    if not isinstance(engine_arc, Arc):
        raise TypeError("engine_arc must be a qa_v3_arc.Arc")
    return Arc(start_deg=-engine_arc.start_deg, sweep_deg=-engine_arc.sweep_deg)


def side_word(published_deg: float) -> str:
    """Which side of your facing direction, in the published convention."""

    return "left" if float(published_deg) > 0.0 else "right"


def landmark_sentence(half_fov_deg: float, convention=None) -> str:
    """The convention, spelled out, for the stem of every azimuth question.

    owner 2026-09-03 asked for the landmark angles to be named rather than
    implied: "在 prompt 里，就把所有的主要度数说清楚分别在哪个角度就行".
    """

    edge = abs(float(half_fov_deg))
    resolved = canonical_convention(convention)
    if resolved == ENGINE_CONVENTION:
        return (
            "Use the AVEngine camera coordinate system: 0 degrees is straight "
            "ahead, positive values turn to your right, +90 is directly right, "
            "-90 is directly left, and +/-180 is directly behind you. The left "
            f"and right edges of the video frame are about -{edge:g} and "
            f"+{edge:g} degrees."
        )
    return (
        "Use the DCASE FOA coordinate system relative to the camera: +x front, "
        "+y left, +z up. Azimuth is in [-180, 180] degrees with positive "
        "values to the left: 0 degrees is straight ahead, +90 is directly to "
        "your left, -90 is directly to your right, and +/-180 is directly "
        f"behind you. The left and right edges of the video frame are about "
        f"+{edge:g} and -{edge:g} degrees."
    )


def published_block(engine_frame_deg: float) -> dict:
    """The published DCASE number together with its convention."""
    return convention_block(engine_frame_deg, CONVENTION)
