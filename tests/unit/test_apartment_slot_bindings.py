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
