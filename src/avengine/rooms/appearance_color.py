"""Illumination-relative colour families for registered appearance evidence.

A registered appearance value such as ``yellow`` or ``standard_white_tan``
describes an asset's surface, not the pixels one room happened to produce.
Between the surface and the pixels sit the room's lights and the camera's
exposure, and those differ from room to room: a warm interior turns a white
shirt into a dim beige, while a baked scan leaves the same shirt bright and
neutral.  Absolute HSV cut-offs therefore read the lighting rather than the
garment, and they read it differently in every room.

Everything in this module is measured against the scene's own neutral
reference, taken from the very frame being inspected and from pixels that do
not belong to the inspected instance.  That reference supplies two things: a
white point, so a colour cast is divided out before any hue is named, and a
lightness scale, so "white" means "about as light as the neutral surfaces in
this room" instead of "above 0.65 of the 8-bit range".  Both are properties the
frame carries with it, which is why the same constants apply to a UE apartment,
a Kujiale flat and an HM3D or MP3D scan without a single per-room number.

Hues are named in CIE Lab, where equal angular distances are roughly equal
perceptual differences, instead of in HSV, where the red sector is narrow and
swallows every warm neutral.  What comes out is still coarse evidence: a family
histogram plus a verdict on the one registered value the caller asked about.
A value whose family predicate is unknown is reported as a gap, never guessed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

# ---------------------------------------------------------------------------
# Measurement constants. Each is a property of colour naming under unknown
# lighting, not of any particular room, and all of them are applied unchanged
# to every capture.
# ---------------------------------------------------------------------------

#: CIE Lab chroma below which a pixel carries no usable hue. Texture, shading
#: noise and the renderer's own tonemap move a genuinely neutral surface a few
#: Lab units off the a*b* origin, so a small chroma is not evidence of a hue.
NEUTRAL_CHROMA = 12.0

#: Relative lightness (a pixel's L* over the scene neutral L*) below which a
#: pixel is too dark to name. Dark surfaces collapse towards the a*b* origin, so
#: a dark brown and a dark navy are not separable in a rendered frame.
DARK_RELATIVE_LIGHTNESS = 0.30

#: Floor on the scene neutral lightness. A frame whose neutral surfaces render
#: this dark carries no surface bright enough to act as a reference, and letting
#: the scale follow it down would turn a black room into a white one. Pinning the
#: scale instead makes such a frame read as what it is: dark everywhere.
MINIMUM_REFERENCE_LIGHTNESS = 35.0

#: Saturation above which the frame's least chromatic pixels cannot be a neutral
#: surface under any plausible room light. A neutral wall under 2700 K tungsten,
#: about the warmest lamp an interior uses, still lands near 0.45; past this the
#: estimate is a coloured surface, and dividing by it would inject a cast instead
#: of removing one, so the correction is dropped and the pixels are read as they
#: are.
MAXIMUM_REFERENCE_SATURATION = 0.55

#: Lightness assumed for a neutral surface when the frame carries no reference
#: outside the inspected instance. Nothing can be measured about the room in that
#: case, so the classifier falls back to the assumption the old absolute rules
#: encoded: the scene is lit well enough that a light neutral surface would
#: render near the top of the range. Real captures never take this branch, since
#: a target never fills the whole frame; fixtures and probes do.
NO_REFERENCE_LIGHTNESS = 75.0

#: Share of the inspected pixels whose hue must be measurable before a hue can be
#: named at all. Below this the target reads as neutral and whatever hue is left
#: is usually not the item: on a person a small warm region is nearly always bare
#: skin, which is why a grey-rendered blue shirt could otherwise be certified by
#: the colour of its own face.
MINIMUM_NAMEABLE_HUE_FRACTION = 0.25

#: The same floor for an item that registers a second colour. Declaring a check
#: or a stripe says the item is patterned, and a patterned item carries its hues
#: over a partly neutral ground -- the measured blue-and-tan plaid reads as a hue
#: in a fifth of its pixels. The relaxation is only available to an item whose
#: registration says so; an item that declares nothing is judged at the full
#: quarter. Either way the hue-carrying pixels must also clear the same absolute
#: support floor the registered colour itself has to clear.
DECLARED_PATTERN_NAMEABLE_HUE_FRACTION = 0.10

#: Default share a value's families must reach. A hue is judged among the pixels
#: whose hue is measurable, so it needs a clear majority of them; a lightness
#: band is judged among all of the target's pixels, where shadow and trim are
#: part of what is being weighed, so it needs a plurality with a margin.
DEFAULT_HUE_SHARE = 0.45
DEFAULT_BAND_SHARE = 0.35

#: Achromatic bands in relative lightness. A surface reflecting about as much as
#: the room's neutral surfaces reads as white; half of that reads as a mid grey.
ACHROMATIC_BANDS: tuple[tuple[str, float, float], ...] = (
    ("black", 0.0, DARK_RELATIVE_LIGHTNESS),
    ("gray", DARK_RELATIVE_LIGHTNESS, 0.55),
    ("light_gray", 0.55, 0.82),
    ("white", 0.82, math.inf),
)

#: Hue families on the CIE Lab hue circle. These angles are not the HSV ones:
#: pure red sits near 40 degrees, gold near 85, green near 136, blue near 306.
HUE_FAMILIES: tuple[tuple[str, float, float], ...] = (
    ("red", 345.0, 40.0),
    ("orange", 40.0, 72.0),
    ("yellow", 72.0, 115.0),
    ("green", 115.0, 200.0),
    ("cyan", 200.0, 250.0),
    ("blue", 250.0, 320.0),
    ("magenta", 320.0, 345.0),
)

ACHROMATIC_NAMES: tuple[str, ...] = tuple(name for name, _, _ in ACHROMATIC_BANDS)
CHROMATIC_NAMES: tuple[str, ...] = tuple(name for name, _, _ in HUE_FAMILIES)
FAMILY_NAMES: tuple[str, ...] = ACHROMATIC_NAMES + CHROMATIC_NAMES

COLOUR_MODEL = "illumination_relative_colour_families_v1"


def _lab(rgb_pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return CIE Lab lightness, chroma and hue angle for a list of RGB pixels."""
    import cv2

    pixels = np.asarray(rgb_pixels, dtype=np.uint8).reshape(-1, 1, 3)
    lab = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)
    lightness = lab[:, 0] * 100.0 / 255.0
    a_star = lab[:, 1] - 128.0
    b_star = lab[:, 2] - 128.0
    chroma = np.hypot(a_star, b_star)
    hue = (np.degrees(np.arctan2(b_star, a_star)) + 360.0) % 360.0
    return lightness, chroma, hue


def estimate_scene_neutral(
    rgb: np.ndarray,
    *,
    exclude_mask: np.ndarray | None = None,
    chroma_percentile: float = 30.0,
    minimum_pixels: int = 256,
) -> dict[str, Any]:
    """Measure this frame's own white point and neutral lightness.

    The least chromatic pixels of a frame are the ones for which the grey-world
    assumption actually holds: walls, floors, ceilings, worktops, doors. Their
    channel-wise median estimates the light that reached the camera, and its
    lightness is the scale every later comparison is expressed in.

    The inspected instance is excluded, so a person in a large saturated shirt
    cannot define the white point they are then measured against, and a target
    that fills the frame cannot quietly become its own reference.
    """
    image = np.asarray(rgb, dtype=np.float32) / 255.0
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("scene neutral estimation needs an RGB image")
    high = image.max(axis=2)
    low = image.min(axis=2)
    saturation = np.where(high > 1e-6, (high - low) / np.maximum(high, 1e-6), 0.0)
    # Clipped highlights carry no colour and crushed blacks carry only noise.
    usable = (high > 0.05) & (high < 0.99)
    fallbacks: list[str] = []
    if exclude_mask is not None:
        keep = ~np.asarray(exclude_mask, dtype=bool)
        if keep.shape != usable.shape:
            raise ValueError("exclusion mask and frame dimensions differ")
        if int((usable & keep).sum()) >= minimum_pixels:
            usable = usable & keep
        else:
            fallbacks.append("inspected_instance_covers_the_frame")
    if int(usable.sum()) < minimum_pixels:
        usable = np.ones(high.shape, dtype=bool)
        fallbacks.append("exposure_window_empty")
    threshold = float(np.percentile(saturation[usable], chroma_percentile))
    selected = usable & (saturation <= max(threshold, 1e-3))
    if int(selected.sum()) < minimum_pixels:
        selected = usable
        fallbacks.append("low_chroma_sample_too_small")
    reference = np.median(image[selected], axis=0).astype(np.float64)
    mean_level = float(np.mean(reference))
    peak = float(np.max(reference))
    reference_saturation = (peak - float(np.min(reference))) / max(peak, 1e-6)
    independent = "inspected_instance_covers_the_frame" not in fallbacks
    if not independent:
        # The reference would be the target measuring itself, which can only
        # ever report that the target is neutral and mid-scene. Correct nothing
        # and read the pixels against an assumed light neutral instead.
        gains = np.ones(3, dtype=np.float64)
    elif reference_saturation > MAXIMUM_REFERENCE_SATURATION:
        # Nothing in this frame is neutral enough to be a white point.
        gains = np.ones(3, dtype=np.float64)
        fallbacks.append("no_neutral_surface_in_frame")
    else:
        gains = np.clip(mean_level / np.maximum(reference, 1e-3), 0.6, 1.8)
    level = int(round(min(max(mean_level, 0.0), 1.0) * 255.0))
    grey = np.full((1, 1, 3), level, dtype=np.uint8)
    measured_lightness = float(_lab(grey)[0][0])
    if not independent:
        reference_lightness = NO_REFERENCE_LIGHTNESS
        fallbacks.append("no_independent_reference_assumed_light_neutral")
    else:
        reference_lightness = max(measured_lightness, MINIMUM_REFERENCE_LIGHTNESS)
        if reference_lightness > measured_lightness:
            fallbacks.append("scene_neutral_below_the_reference_floor")
    return {
        "reference_rgb": [round(float(value), 5) for value in reference],
        "reference_lightness": round(reference_lightness, 3),
        "measured_reference_lightness": round(measured_lightness, 3),
        "reference_saturation": round(reference_saturation, 4),
        "white_balance_gains": [round(float(value), 4) for value in gains],
        "sampled_pixel_fraction": round(float(selected.sum()) / float(selected.size), 4),
        "chroma_percentile": float(chroma_percentile),
        "fallbacks": fallbacks,
        "method": "median_of_least_chromatic_pixels_outside_the_inspected_instance",
    }


def white_balanced(rgb: np.ndarray, neutral: Mapping[str, Any]) -> np.ndarray:
    """Divide out the estimated illuminant: a von Kries correction in RGB."""
    gains = np.asarray(neutral["white_balance_gains"], dtype=np.float32)
    image = np.asarray(rgb, dtype=np.float32)
    if image.shape[-1] != 3:
        raise ValueError("white balance needs a trailing RGB axis")
    return np.clip(image * gains, 0.0, 255.0).astype(np.uint8)


def colour_family_counts(
    rgb_pixels: np.ndarray, reference_lightness: float,
) -> tuple[dict[str, int], dict[str, float]]:
    """Sort pixels into achromatic bands and Lab hue families.

    Lightness is always relative to the scene's neutral reference, so the same
    surface lands in the same band whether its room is bright or dim. A pixel
    darker than the dark limit is called black whatever its hue, because at that
    level the hue is not measurable.
    """
    pixels = np.asarray(rgb_pixels, dtype=np.uint8).reshape(-1, 3)
    counts = {name: 0 for name in FAMILY_NAMES}
    medians: dict[str, float] = {}
    if pixels.size == 0:
        return counts, medians
    lightness, chroma, hue = _lab(pixels)
    relative = lightness / max(float(reference_lightness), 1e-3)
    labels = np.zeros(len(relative), dtype=np.int8)
    index = {name: position for position, name in enumerate(FAMILY_NAMES)}
    too_dark = relative < DARK_RELATIVE_LIGHTNESS
    labels[too_dark] = index["black"]
    achromatic = (~too_dark) & (chroma < NEUTRAL_CHROMA)
    for name, low, high in ACHROMATIC_BANDS:
        if name == "black":
            continue
        labels[achromatic & (relative >= low) & (relative < high)] = index[name]
    chromatic = (~too_dark) & (chroma >= NEUTRAL_CHROMA)
    for name, low, high in HUE_FAMILIES:
        span = (hue >= low) & (hue < high) if low < high else (hue >= low) | (hue < high)
        labels[chromatic & span] = index[name]
    for name in FAMILY_NAMES:
        selected = labels == index[name]
        counts[name] = int(selected.sum())
        if counts[name] >= 16:
            medians[name] = round(float(np.median(relative[selected])), 3)
    return counts, medians


# ---------------------------------------------------------------------------
# Registered value -> family predicate.
#
# The table speaks only of colour families. A new asset that registers "beige"
# is judged by the beige predicate whatever mesh, room or renderer it arrives
# with; nothing here is keyed to an asset id, a room package or a map name.
# ---------------------------------------------------------------------------

#: Naming an achromatic surface is one band coarser than naming a hue: a white
#: surface out of the light reads as a light grey, and a light grey under a lamp
#: reads as white. A neighbouring band therefore supports an achromatic value,
#: while a hue error never does. Black is given no lighter neighbour, so a mid
#: grey is never certified as black.
ACHROMATIC_NEIGHBOURS: dict[str, tuple[str, ...]] = {
    "white": ("white", "light_gray"),
    "light_gray": ("light_gray", "white", "gray"),
    "gray": ("gray", "light_gray"),
    "black": ("black",),
}

#: Warm hues shade into one another as the light warms or cools: a mustard shirt
#: drifts between orange and yellow, and a red coat between red and orange. Each
#: warm value therefore tolerates its warm neighbour instead of competing with
#: it; the share requirement, not the neighbour, is what keeps a brown surface
#: from being certified as yellow.
WARM = ("red", "orange", "yellow")
LIGHT_NEUTRAL = ("white", "light_gray")


@dataclass(frozen=True)
class ColourPredicate:
    """What a registered appearance value claims about the family histogram."""

    #: Families whose pixels support the value.
    support: tuple[str, ...]
    #: Families that neither support nor contradict it: pattern partners, trim,
    #: shading, and the warm neighbours a hue drifts into under coloured light.
    tolerated: tuple[str, ...] = ()
    #: A sanity window on the median relative lightness of the supporting
    #: pixels. It rejects an absurd reading (a near-black region claimed as a
    #: tint) without pretending to certify a tint against a shade.
    lightness_window: tuple[float, float] = (0.0, math.inf)
    #: Components a two- or three-tone registration must also show, each a
    #: family group with its own minimum share of the inspected pixels.
    components: tuple[tuple[tuple[str, ...], float], ...] = ()
    #: Minimum share the supporting families must reach, over the denominator
    #: that fits the value: measurable-hue pixels for a hue, all inspected pixels
    #: for a lightness band or a mixed two-tone claim. ``None`` takes the default.
    minimum_share: float | None = None
    #: What the value means in family terms, carried into the evidence.
    note: str = ""


_HUMAN_PREDICATES: dict[str, ColourPredicate] = {
    # Every human carries bare skin on the upper body -- arms, neck, face -- and
    # skin is a warm hue no garment registers. It is tolerated for every human
    # garment colour rather than allowed to compete with one.
    "blue": ColourPredicate(support=("blue",), tolerated=("cyan", "orange")),
    "green": ColourPredicate(support=("green",), tolerated=("orange",)),
    "yellow": ColourPredicate(support=("yellow",), tolerated=("orange",)),
    "burgundy": ColourPredicate(
        support=("red",),
        tolerated=("orange", "magenta"),
        lightness_window=(0.0, 0.95),
        note="a shade of the red family; the tint/shade split is reported, not certified",
    ),
    "pink": ColourPredicate(
        support=("red", "magenta"),
        tolerated=("orange",),
        lightness_window=(0.32, math.inf),
        note="a tint of the red family; the tint/shade split is reported, not certified",
    ),
    "white": ColourPredicate(support=ACHROMATIC_NEIGHBOURS["white"], minimum_share=0.35),
}

_DEVICE_PREDICATES: dict[str, ColourPredicate] = {
    "white": ColourPredicate(support=ACHROMATIC_NEIGHBOURS["white"], minimum_share=0.35),
    "white_satin": ColourPredicate(support=ACHROMATIC_NEIGHBOURS["white"], minimum_share=0.35),
    "light_gray": ColourPredicate(support=ACHROMATIC_NEIGHBOURS["light_gray"], minimum_share=0.35),
    "light_gray_fabric": ColourPredicate(support=ACHROMATIC_NEIGHBOURS["light_gray"], minimum_share=0.35),
    "silver": ColourPredicate(support=ACHROMATIC_NEIGHBOURS["light_gray"], minimum_share=0.35),
    "warm_gray": ColourPredicate(
        support=ACHROMATIC_NEIGHBOURS["light_gray"],
        tolerated=("orange",),
        minimum_share=0.35,
        note="a grey with a warm tint: an achromatic band, with the warm drift tolerated",
    ),
    "gray": ColourPredicate(support=ACHROMATIC_NEIGHBOURS["gray"], minimum_share=0.35),
    "black": ColourPredicate(support=("black",), minimum_share=0.45),
    "matte_black": ColourPredicate(support=("black",), minimum_share=0.45),
    "black_ash": ColourPredicate(support=("black",), minimum_share=0.45),
    "charcoal": ColourPredicate(support=("black",), minimum_share=0.45),
    "dark": ColourPredicate(support=("black",), minimum_share=0.45),
    "beige": ColourPredicate(
        support=("orange", "yellow") + LIGHT_NEUTRAL,
        tolerated=("gray",),
        lightness_window=(0.45, math.inf),
        minimum_share=0.35,
        note="a light warm neutral: a weak sand hue over a light achromatic band",
    ),
    "sandstone": ColourPredicate(
        support=("orange", "yellow") + LIGHT_NEUTRAL,
        tolerated=("gray",),
        lightness_window=(0.45, math.inf),
        minimum_share=0.35,
        note="a light warm neutral: a weak sand hue over a light achromatic band",
    ),
    "walnut_veneer": ColourPredicate(
        support=("orange", "red"),
        tolerated=("yellow", "black"),
        lightness_window=(0.0, 0.85),
        note="a dark warm wood: a warm hue below the room's neutral lightness",
    ),
    "walnut": ColourPredicate(
        support=("orange", "red"),
        tolerated=("yellow", "black"),
        lightness_window=(0.0, 0.85),
        note="a dark warm wood: a warm hue below the room's neutral lightness",
    ),
    "brown": ColourPredicate(
        support=("orange", "red"),
        tolerated=("yellow", "black"),
        lightness_window=(0.0, 0.85),
    ),
    "red": ColourPredicate(support=("red",), tolerated=("orange", "magenta")),
    "ruddy": ColourPredicate(support=("red", "orange"), tolerated=("yellow",)),
    "blue": ColourPredicate(support=("blue",), tolerated=("cyan",)),
    "green": ColourPredicate(support=("green",)),
    "yellow": ColourPredicate(support=("yellow",), tolerated=("orange",)),
}

_ANIMAL_PREDICATES: dict[str, ColourPredicate] = {
    "standard_yellow": ColourPredicate(
        support=("yellow", "orange"),
        tolerated=LIGHT_NEUTRAL + ("black", "red"),
        lightness_window=(0.55, math.inf),
        note="a light golden coat: a warm hue about as light as the room's neutrals",
    ),
    "standard_blue": ColourPredicate(
        support=ACHROMATIC_NEIGHBOURS["gray"],
        tolerated=("black", "cyan", "blue"),
        minimum_share=0.35,
        note="the registered feline blue is a grey-blue coat, not a blue hue",
    ),
    "standard_red": ColourPredicate(
        support=("red", "orange"),
        tolerated=("yellow", "black") + LIGHT_NEUTRAL,
    ),
    "standard_ruddy": ColourPredicate(
        support=("red", "orange"),
        tolerated=("yellow", "black") + LIGHT_NEUTRAL,
    ),
    "standard_sable": ColourPredicate(
        support=("red", "orange", "yellow"),
        tolerated=("black",) + LIGHT_NEUTRAL,
        lightness_window=(0.0, 0.72),
        note="a dark warm coat: a warm hue well below the room's neutral lightness",
    ),
    "dark_sable": ColourPredicate(
        support=("red", "orange", "yellow", "black"),
        tolerated=LIGHT_NEUTRAL,
        lightness_window=(0.0, 0.70),
        minimum_share=0.35,
        note="a very dark warm coat: warm where it is readable and black where it is not",
    ),
    "standard_black_white": ColourPredicate(
        support=("black",) + LIGHT_NEUTRAL,
        tolerated=("gray",),
        components=((("black",), 0.20), (LIGHT_NEUTRAL, 0.10)),
        minimum_share=0.45,
        note="a two-tone coat: a dark and a light component must both be present",
    ),
    "standard_red_white": ColourPredicate(
        support=("red", "orange") + LIGHT_NEUTRAL,
        tolerated=("yellow", "black", "gray"),
        components=((("red", "orange"), 0.10), (LIGHT_NEUTRAL, 0.10)),
        minimum_share=0.40,
        note="a two-tone coat: a warm and a light component must both be present",
    ),
    "standard_white_tan": ColourPredicate(
        # Light first: the coat is named for its white ground, and the leading
        # family is what appearance_distinction_family reports.
        support=LIGHT_NEUTRAL + ("orange", "yellow"),
        tolerated=("red", "black", "gray"),
        components=((LIGHT_NEUTRAL, 0.10), (("orange", "yellow"), 0.10)),
        minimum_share=0.40,
        note="a two-tone coat: a light and a tan component must both be present",
    ),
    "standard_tricolor": ColourPredicate(
        support=("black", "red", "orange") + LIGHT_NEUTRAL,
        tolerated=("yellow", "gray"),
        components=((LIGHT_NEUTRAL, 0.08), (("black",), 0.12), (("red", "orange"), 0.08)),
        minimum_share=0.45,
        note="a three-tone coat: light, dark and warm components must all be present",
    ),
    "standard_seal_point": ColourPredicate(
        support=("black", "orange", "yellow") + LIGHT_NEUTRAL,
        tolerated=("red", "gray"),
        components=((LIGHT_NEUTRAL + ("orange", "yellow"), 0.20), (("black",), 0.10)),
        minimum_share=0.45,
        note="a pointed coat: a light warm body with dark extremities",
    ),
}
_ANIMAL_PREDICATES["light_tricolor"] = _ANIMAL_PREDICATES["standard_tricolor"]
_ANIMAL_PREDICATES["dark_tricolor"] = _ANIMAL_PREDICATES["standard_tricolor"]

PREDICATES_BY_KIND: dict[str, dict[str, ColourPredicate]] = {
    "human": _HUMAN_PREDICATES,
    "device": _DEVICE_PREDICATES,
    "animal": _ANIMAL_PREDICATES,
}


def predicate_for(value: str, entity_kind: str) -> ColourPredicate | None:
    """Resolve the family predicate a registered value stands for, or None."""
    wanted = str(value).strip().casefold()
    kind = str(entity_kind).strip().casefold()
    table = PREDICATES_BY_KIND.get(kind)
    if table is not None and wanted in table:
        return table[wanted]
    # The same word names the same colour on an unexpected entity kind.
    for other in ("device", "animal", "human"):
        if wanted in PREDICATES_BY_KIND[other]:
            return PREDICATES_BY_KIND[other][wanted]
    return None


def appearance_distinction_family(value: str, entity_kind: str = "") -> str | None:
    """The coarse family two registered values must differ in to be told apart.

    Two values that resolve to the same family cannot be separated in rendered
    pixels: pink and burgundy are a tint and a shade of one hue, so a frame shows
    the same family for both and no amount of thresholding recovers which is
    which. A question that asks a participant to tell two actors apart by colour
    therefore needs their registered values to land in different families here.

    The family is the first family a value's predicate is built on, which is the
    one the value is named for; it is derived from the predicate table rather
    than listed separately, so a new value joins the right group automatically.
    """
    predicate = predicate_for(value, entity_kind)
    if predicate is None or not predicate.support:
        return None
    return predicate.support[0]


def secondary_tolerated_families(value: str, entity_kind: str = "") -> tuple[str, ...]:
    """The families a registered second colour of a two-tone item covers.

    A checked or striped item carries a second colour that is part of the item,
    not evidence against the first. Declaring it in the registry is what lets the
    classifier stop counting it as a competitor; an item that declares nothing is
    judged exactly as before.
    """
    predicate = predicate_for(value, entity_kind)
    return tuple(predicate.support) if predicate is not None else ()


def supported_appearance_values() -> frozenset[str]:
    """Every registered value this classifier can actually observe."""
    return frozenset(value for table in PREDICATES_BY_KIND.values() for value in table)


def evaluate_registered_value(
    counts: Mapping[str, int],
    medians: Mapping[str, float],
    value: str,
    entity_kind: str,
    *,
    minimum_support_pixels: int,
    dominance_ratio: float,
    minimum_share: float | None = None,
    secondary_value: str | None = None,
    declared_primary_value: str | None = None,
) -> dict[str, Any]:
    """Decide whether the family histogram shows the registered value.

    The rule is the same for every value: the families the value stands for must
    carry a real share of the inspected pixels and must outweigh the families it
    could be confused with. Competitors of the same kind -- another hue against
    a hue, another lightness band against a band -- must be beaten by the
    dominance margin, because a near tie means the colour cannot be named.
    Competitors of the other kind only have to be beaten outright, because
    stripes, trim and shading routinely put a second kind on a garment.
    """
    predicate = predicate_for(value, entity_kind)
    total = int(sum(int(counts.get(name, 0)) for name in FAMILY_NAMES))
    if predicate is None:
        return {
            "accepted": False,
            "unsupported": True,
            "analysed_pixels": total,
            "support_pixels": 0,
            "support_share": 0.0,
        }
    support_families = tuple(predicate.support)
    support = sum(int(counts.get(name, 0)) for name in support_families)
    neutral_support = any(name in ACHROMATIC_NAMES for name in support_families)
    nameable_hue = sum(int(counts.get(name, 0)) for name in CHROMATIC_NAMES)
    if neutral_support:
        # A lightness band is readable everywhere in the target, so every pixel
        # counts: shadow and trim are part of what is being weighed.
        denominator, denominator_name = total, "inspected_pixels"
        default_share = DEFAULT_BAND_SHARE
    else:
        # A hue is only readable where a hue exists. A pixel in shadow is not
        # evidence against a colour; it is the absence of evidence, so it leaves
        # the denominator instead of voting against the registered value.
        denominator, denominator_name = nameable_hue, "pixels_with_a_measurable_hue"
        default_share = DEFAULT_HUE_SHARE
    share_floor = float(
        (predicate.minimum_share if predicate.minimum_share is not None else default_share)
        if minimum_share is None else minimum_share
    )
    share = support / max(1, denominator)
    excluded = set(support_families) | set(predicate.tolerated)
    # A registration speaks about the colours it declares. Asking whether the
    # same item could be some third colour is a question its declaration says
    # nothing about, so the declared pattern neither excuses a competitor nor
    # lowers the bar for naming a hue in that case.
    checking_the_declaration = (
        declared_primary_value is None
        or str(declared_primary_value).strip().casefold() == str(value).strip().casefold()
    )
    declared_secondary = (
        secondary_tolerated_families(secondary_value, entity_kind)
        if checking_the_declaration and isinstance(secondary_value, str) and secondary_value.strip()
        else ()
    )
    # A declared second colour leaves the competition, but it stays in the
    # denominator, so the registered first colour still has to carry its share of
    # the item. Declaring a second colour cannot rescue a first colour that is
    # actually a minority of what the frame shows.
    excluded |= set(declared_secondary)
    if not neutral_support:
        # Shadow and neutral trim never contradict a registered hue.
        excluded |= set(ACHROMATIC_NAMES)
    same_kind: dict[str, int] = {}
    other_kind: dict[str, int] = {}
    for name in FAMILY_NAMES:
        if name in excluded:
            continue
        bucket = same_kind if (name in ACHROMATIC_NAMES) == neutral_support else other_kind
        bucket[name] = int(counts.get(name, 0))
    strongest_same = max(same_kind.items(), key=lambda item: item[1], default=("", 0))
    strongest_other = max(other_kind.items(), key=lambda item: item[1], default=("", 0))
    # The lightness reading of the supporting pixels, weighted by how many each
    # family contributed, so a sliver of one family cannot swing the window.
    weighted = [
        (int(counts.get(name, 0)), float(medians[name]))
        for name in support_families
        if name in medians and int(counts.get(name, 0)) > 0
    ]
    support_lightness = (
        sum(weight * level for weight, level in weighted) / sum(weight for weight, _ in weighted)
        if weighted
        else None
    )
    low, high = predicate.lightness_window
    components = [
        {
            "families": list(families),
            "minimum_share": round(float(floor), 3),
            "share": round(sum(int(counts.get(name, 0)) for name in families) / max(1, total), 4),
        }
        for families, floor in predicate.components
    ]
    reasons: list[str] = []
    if support < int(minimum_support_pixels):
        reasons.append("too_few_supporting_pixels")
    nameable_floor = (
        DECLARED_PATTERN_NAMEABLE_HUE_FRACTION if declared_secondary
        else MINIMUM_NAMEABLE_HUE_FRACTION
    )
    if not neutral_support and (
        nameable_hue < nameable_floor * max(1, total)
        or nameable_hue < int(minimum_support_pixels)
    ):
        reasons.append("target_is_too_dark_or_too_neutral_for_a_hue")
    if share < share_floor:
        reasons.append("registered_families_are_a_minority_of_the_target")
    if strongest_same[1] > 0 and support < float(dominance_ratio) * strongest_same[1]:
        reasons.append(f"another_{'band' if neutral_support else 'hue'}_competes:{strongest_same[0]}")
    if strongest_other[1] > support:
        reasons.append(f"a_stronger_component_disagrees:{strongest_other[0]}")
    if support_lightness is not None and not (low <= support_lightness <= high):
        reasons.append("supporting_pixels_are_outside_the_value_lightness_window")
    for component in components:
        if component["share"] < component["minimum_share"]:
            reasons.append("missing_component:" + "+".join(component["families"]))
    return {
        "accepted": not reasons,
        "unsupported": False,
        "analysed_pixels": total,
        "support_families": list(support_families),
        "tolerated_families": list(predicate.tolerated),
        "declared_secondary_value": secondary_value if declared_secondary else None,
        "declared_secondary_families": list(declared_secondary),
        "support_pixels": support,
        "support_share": round(share, 4),
        "support_share_denominator": denominator_name,
        "denominator_pixels": int(denominator),
        "nameable_hue_pixels": int(nameable_hue),
        "minimum_nameable_hue_fraction": round(float(nameable_floor), 3),
        "minimum_support_share": round(share_floor, 3),
        "support_relative_lightness": (
            round(support_lightness, 3) if support_lightness is not None else None
        ),
        "lightness_window": [low, None if math.isinf(high) else high],
        "strongest_competing_same_kind": list(strongest_same),
        "strongest_competing_other_kind": list(strongest_other),
        "components": components,
        "rejections": reasons,
        "predicate_note": predicate.note,
    }
