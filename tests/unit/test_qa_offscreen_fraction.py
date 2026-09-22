"""The share of episodes whose subject the camera never shows is a batch knob.

Without it a scale-up marks two slots per room, so a fifty-episode room answers
96% of its questions from the picture alone.
"""
import pytest

from avengine.qa.batch_manifest import build_scaleup_slots


ROOMS = [
    {"room_id": "room_a", "family": "hm3d"},
    {"room_id": "room_b", "family": "mp3d"},
]


def _off_screen_share(slots):
    return sum(1 for slot in slots if slot.get("off_screen")) / len(slots)


def test_default_keeps_two_slots_per_room():
    slots = build_scaleup_slots(ROOMS, seed=20260907, episodes_per_room=20)
    per_room = {}
    for slot in slots:
        per_room.setdefault(slot["room_id"], []).append(slot)
    assert {room: sum(1 for s in group if s.get("off_screen"))
            for room, group in per_room.items()} == {"room_a": 2, "room_b": 2}


@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.6, 1.0])
def test_fraction_is_met_in_every_room(fraction):
    slots = build_scaleup_slots(
        ROOMS, seed=20260907, episodes_per_room=20, off_screen_fraction=fraction)
    per_room = {}
    for slot in slots:
        per_room.setdefault(slot["room_id"], []).append(slot)
    for group in per_room.values():
        share = _off_screen_share(group)
        # Two slots are marked before the fraction applies, so it is a floor.
        assert share >= min(fraction, 2 / len(group)) - 1e-9
        assert share >= fraction - 1e-9 or share == 2 / len(group)


def test_fraction_reaches_the_release_gate_floor():
    slots = build_scaleup_slots(
        ROOMS, seed=20260907, episodes_per_room=20, off_screen_fraction=0.65)
    assert _off_screen_share(slots) >= 0.6


def test_same_seed_names_the_same_episodes():
    first = build_scaleup_slots(
        ROOMS, seed=20260907, episodes_per_room=20, off_screen_fraction=0.6)
    second = build_scaleup_slots(
        ROOMS, seed=20260907, episodes_per_room=20, off_screen_fraction=0.6)
    assert ([s["episode_id"] for s in first if s.get("off_screen")]
            == [s["episode_id"] for s in second if s.get("off_screen")])


def test_class_pairs_pad_the_same_way_with_and_without_the_fraction():
    """The off-screen draw must not consume the generator the padding uses."""
    plain = build_scaleup_slots(ROOMS, seed=20260907, episodes_per_room=20)
    marked = build_scaleup_slots(
        ROOMS, seed=20260907, episodes_per_room=20, off_screen_fraction=0.6)
    assert ([s["source_classes"] for s in plain]
            == [s["source_classes"] for s in marked])


def test_out_of_range_fraction_is_refused():
    with pytest.raises(ValueError, match="off_screen_fraction"):
        build_scaleup_slots(
            ROOMS, seed=20260907, episodes_per_room=20, off_screen_fraction=1.5)


def test_profile_carries_the_knob_the_sampler_reads():
    slots = build_scaleup_slots(
        ROOMS, seed=20260907, episodes_per_room=20, off_screen_fraction=0.6)
    marked = [s for s in slots if s.get("off_screen")]
    assert marked
    for slot in marked:
        key = ("anchor_visibility" if slot["off_screen"] == "anchor"
               else "competitor_visibility")
        assert slot["profile"][key] == "off_screen"
