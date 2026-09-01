"""Unit tests for avengine.dataset.apartment_dynamic_audio.derive_slot_bindings."""

from __future__ import annotations

import pytest

from avengine.timeline.current_mp3d_dynamic_audio import CurrentMP3DDynamicAudioError
from avengine.dataset.apartment_dynamic_audio import derive_slot_bindings


def _selection():
    return {
        "actors": [
            {
                "source_slot_id": "source1",
                "asset_id": "human_a",
                "legacy_timeline_actor_id": "human_1",
            },
            {
                "source_slot_id": "source2",
                "asset_id": "human_b",
                "legacy_timeline_actor_id": "human_2",
            },
        ]
    }


def _asset(asset_id, height):
    return {
        "asset_id": asset_id,
        "default_emitter_anchor_id": "mouth",
        "emitter_anchors": [
            {"anchor_id": "mouth", "offset_m": [0.0, height, 0.0]}
        ],
    }


def _endpoint(eid, instance, asset_id):
    return {
        "source_endpoint_id": eid,
        "binding": {
            "entity_instance_id": instance,
            "entity_asset_id": asset_id,
            "emitter_anchor_id": "mouth",
        },
    }


def _registries():
    source_assets = {"assets": [_asset("human_a", 1.64), _asset("human_b", 1.64)]}
    endpoints = {
        "source_endpoints": [
            _endpoint("ep_h1_a", "human_1", "human_a"),
            _endpoint("ep_h2_b", "human_2", "human_b"),
            _endpoint("ep_h1_b", "human_1", "human_b"),
        ]
    }
    return source_assets, endpoints


def test_derives_endpoints_and_heights():
    source_assets, endpoints = _registries()
    slots, heights = derive_slot_bindings(_selection(), source_assets, endpoints)
    assert slots == {"source1": "ep_h1_a", "source2": "ep_h2_b"}
    assert heights == {"source1": 1.64, "source2": 1.64}


def test_optional_canonical_height_preserves_endpoints_and_normalizes_geometry():
    source_assets, endpoints = _registries()
    source_assets["assets"][1]["emitter_anchors"][0]["offset_m"][1] = 1.58
    slots, heights = derive_slot_bindings(
        _selection(), source_assets, endpoints,
        canonical_emitter_height_m=1.61)
    assert slots == {"source1": "ep_h1_a", "source2": "ep_h2_b"}
    assert heights == {"source1": 1.61, "source2": 1.61}


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan")])
def test_invalid_canonical_height_fails_closed(value):
    source_assets, endpoints = _registries()
    with pytest.raises(CurrentMP3DDynamicAudioError,
                       match="canonical emitter height"):
        derive_slot_bindings(
            _selection(), source_assets, endpoints,
            canonical_emitter_height_m=value)


def test_derives_four_contiguous_source_slots():
    selection = _selection()
    selection["actors"].extend(
        [
            {
                "source_slot_id": "source3",
                "asset_id": "human_c",
                "legacy_timeline_actor_id": "human_3",
            },
            {
                "source_slot_id": "source4",
                "asset_id": "human_d",
                "legacy_timeline_actor_id": "human_4",
            },
        ]
    )
    source_assets, endpoints = _registries()
    source_assets["assets"].extend([_asset("human_c", 1.62), _asset("human_d", 1.61)])
    endpoints["source_endpoints"].extend(
        [
            _endpoint("ep_h3_c", "human_3", "human_c"),
            _endpoint("ep_h4_d", "human_4", "human_d"),
        ]
    )
    slots, heights = derive_slot_bindings(selection, source_assets, endpoints)
    assert slots == {
        "source1": "ep_h1_a",
        "source2": "ep_h2_b",
        "source3": "ep_h3_c",
        "source4": "ep_h4_d",
    }
    assert heights["source3"] == 1.62
    assert heights["source4"] == 1.61


def test_source_slot_gap_fails_closed():
    selection = _selection()
    selection["actors"][1]["source_slot_id"] = "source3"
    source_assets, endpoints = _registries()
    with pytest.raises(CurrentMP3DDynamicAudioError, match="contiguous source1"):
        derive_slot_bindings(selection, source_assets, endpoints)


def test_missing_endpoint_fails_closed():
    source_assets, endpoints = _registries()
    endpoints["source_endpoints"] = endpoints["source_endpoints"][:1]
    with pytest.raises(CurrentMP3DDynamicAudioError, match="exactly one endpoint"):
        derive_slot_bindings(_selection(), source_assets, endpoints)


def test_duplicate_endpoint_fails_closed():
    source_assets, endpoints = _registries()
    endpoints["source_endpoints"].append(_endpoint("ep_dup", "human_1", "human_a"))
    with pytest.raises(CurrentMP3DDynamicAudioError, match="exactly one endpoint"):
        derive_slot_bindings(_selection(), source_assets, endpoints)


def test_unregistered_asset_fails_closed():
    source_assets, endpoints = _registries()
    source_assets["assets"] = source_assets["assets"][1:]
    with pytest.raises(CurrentMP3DDynamicAudioError, match="unregistered asset"):
        derive_slot_bindings(_selection(), source_assets, endpoints)
