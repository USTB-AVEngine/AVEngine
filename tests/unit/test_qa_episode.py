"""Unit tests for the unified QA Episode data structure."""

from __future__ import annotations

import pytest

from avengine.m5.timeline import FRAME_COUNT, build_timeline
from avengine.qa.episode import (
    Episode,
    EpisodeError,
    EpisodeEvent,
    QAPair,
    VisibilityRecord,
    QA_EPISODE_SCHEMA,
    VISIBILITY_OUT_OF_VIEW,
    VISIBILITY_CLEAR,
    VISIBILITY_OCCLUDED,
    VISIBILITY_FULLY_OCCLUDED,
    EVENT_ENTER_FRUSTUM,
    EVENT_EXIT_FRUSTUM,
    EVENT_BECOME_VISIBLE,
    EVENT_OCCLUSION_START,
    EVENT_FULLY_OCCLUDED,
    EVENT_REAPPEAR,
    classify_visibility,
    detect_visibility_events,
    make_visibility_record,
    validate_qa_episode,
    validate_qa_episode_schema,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Minimal valid timeline fixture
# ═══════════════════════════════════════════════════════════════════════════════

def _minimal_episode_request():
    """Return a minimal M5 episode request that validates."""
    from avengine.contracts.json_io import canonical_json_sha256

    payload = {
        "schema": "avengine_m5_episode_request_v1",
        "request_id": "test_request_01",
        "counterfactual_pair_id": "test_pair_01",
        "qualification_claim": False,
        "seed": 0,
        "timeline_profile": {
            "time_base_hz": 48000,
            "duration_ticks": 240000,
            "video": {
                "fps_num": 15,
                "fps_den": 1,
                "frame_count": 75,
                "ticks_per_frame": 3200,
                "view_ids": ["view0"],
            },
            "audio": {
                "sample_rate_hz": 16000,
                "sample_count": 80000,
                "ticks_per_sample": 3,
                "authority": {
                    "format_id": "rlr_foa_acn_n3d_world_v1",
                    "ambisonic_order": 1,
                    "channel_count": 4,
                    "raw_channel_order": ["W", "Y", "Z", "X"],
                    "acn_indices": [0, 1, 2, 3],
                    "normalization": "N3D",
                    "coordinate_frame": "avengine_world",
                    "handedness": "right",
                    "axes": {"right": "+X", "up": "+Y", "back": "+Z", "forward": "-Z"},
                    "raw_array_layout": "channel_major_[channels,samples]",
                    "dtype": "float32_le",
                },
            },
        },
        "visual_vocal_articulation": {
            "mode": "disabled_for_shortcut_control",
            "mouth_motion_present": False,
        },
        "listener": {
            "listener_id": "listener0",
            "camera_rig_id": "camera_rig_0",
            "view_id": "view0",
        },
        "actors": [
            {
                "actor_id": "source1",
                "asset_id": "test_asset_01",
                "template_id": "test_template_01",
                "body_plan_id": "quadruped_canine",
                "instance_offset_m": [0.0, 0.0, -2.0],
                "semantic_id": 1,
            },
            {
                "actor_id": "source2",
                "asset_id": "test_asset_02",
                "template_id": "test_template_02",
                "body_plan_id": "biped_human",
                "instance_offset_m": [0.0, 0.0, 2.0],
                "semantic_id": 2,
            },
        ],
        "sources": [
            {
                "source_id": "src1",
                "actor_id": "source1",
                "semantic_anchor_id": "muzzle",
                "emitter_link": "head",
                "emitter_path_sha256": "a" * 64,
            },
            {
                "source_id": "src2",
                "actor_id": "source2",
                "semantic_anchor_id": "muzzle",
                "emitter_link": "head",
                "emitter_path_sha256": "b" * 64,
            },
        ],
        "audio_program": {
            "program_id": "prog_01",
            "clip_source_interval": {"start_sample": 0, "end_sample": 5333},
            "fade_samples": 0,
            "linear_gain": 1.0,
            "simultaneous_windows": [
                {"window_id": "w01", "start_sample": 0, "end_sample": 5333},
                {"window_id": "w02", "start_sample": 10000, "end_sample": 15333},
                {"window_id": "w03", "start_sample": 20000, "end_sample": 25333},
                {"window_id": "w04", "start_sample": 30000, "end_sample": 35333},
                {"window_id": "w05", "start_sample": 40000, "end_sample": 45333},
                {"window_id": "w06", "start_sample": 50000, "end_sample": 55333},
            ],
        },
        "events": [
            {
                "event_id": "evt1",
                "actor_id": "source1",
                "source_id": "src1",
                "event_type": "vocalization",
                "audio_program_id": "prog_01",
                "emitter_link": "head",
                "emitter_path_sha256": "a" * 64,
                "dry_audio_asset_sha256": "c" * 64,
                "semantic_sync_required": True,
            },
            {
                "event_id": "evt2",
                "actor_id": "source2",
                "source_id": "src2",
                "event_type": "vocalization",
                "audio_program_id": "prog_01",
                "emitter_link": "head",
                "emitter_path_sha256": "b" * 64,
                "dry_audio_asset_sha256": "d" * 64,
                "semantic_sync_required": True,
            },
        ],
        "counterfactual": {
            "operation": "swap_dry_audio_source_routing",
            "variants": ["A", "B"],
            "frozen_fields": [
                "timeline.video",
                "timeline.actors",
                "timeline.frames",
                "timeline.audio_events_except_audio_asset_sha256",
                "visual_vocal_articulation",
                "listener",
                "actor_source_event_ids",
            ],
            "allowed_changed_fields": [
                "request.events[*].dry_audio_asset_sha256",
                "timeline.audio_events[*].audio_asset_sha256",
                "dynamic_audio_render_manifest.source_routes[*].dry_audio_asset_sha256",
            ],
            "derived_changed_fields": [
                "request.request_content_sha256",
                "dynamic_audio_render_manifest.timeline_content_sha256",
                "dynamic_audio_render_manifest.manifest_content_sha256",
            ],
        },
    }
    payload["request_content_sha256"] = canonical_json_sha256(payload)
    return payload


def _minimal_visual_frames():
    """Return 75 minimal visual frames with static actor states."""
    frames = []
    for fi in range(FRAME_COUNT):
        frames.append({
            "actor_states": [
                {
                    "actor_id": "source1",
                    "root_transform": {
                        "translation_m": [0.0, 0.0, float(fi) * 0.01],
                        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                        "scale": [1.0, 1.0, 1.0],
                    },
                    "action_id": "idle",
                    "action_time_ticks": fi * 3200,
                    "action_phase": 0.0,
                    "pose_hash": "e" * 64,
                    "contacts": {},
                    "mouth_state": {"open_ratio": 0.0, "vocalizing": False},
                },
                {
                    "actor_id": "source2",
                    "root_transform": {
                        "translation_m": [0.0, 0.0, float(fi) * -0.01],
                        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                        "scale": [1.0, 1.0, 1.0],
                    },
                    "action_id": "idle",
                    "action_time_ticks": fi * 3200,
                    "action_phase": 0.0,
                    "pose_hash": "f" * 64,
                    "contacts": {},
                    "mouth_state": {"open_ratio": 0.0, "vocalizing": False},
                },
            ],
            "view_pose_hashes": {"view0": "a" * 64},
        })
    return frames


def _valid_timeline():
    """Build a valid timeline for testing."""
    return build_timeline(_minimal_episode_request(), _minimal_visual_frames())


# ═══════════════════════════════════════════════════════════════════════════════
# Schema validation tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSchemaValidation:
    def test_minimal_valid_episode_passes(self):
        ep = (
            Episode("ep_test_01")
            .add_actor("source1", "beagle_01", "dog", 1, breed_id="beagle",
                       size="medium", body_build="standard", life_stage="adult",
                       coat_profile_id="dog_beagle_tricolor_v1", coat_value="standard_tricolor")
            .add_sound("bark_01", "dog_vocalization", "source1", sound_category="bark")
            .add_furniture_occluder("table_01", "table", 100)
        )
        ep.timeline = _valid_timeline()

        # Populate sidecars with valid non-empty paths
        ep.rgb_video = "ep_test_01/rgb.mp4"
        ep.semantic_video = "ep_test_01/semantic.mp4"
        ep.depth_frames = "ep_test_01/depth/"
        ep.target_only_masks = "ep_test_01/target_only/"
        ep.audio_mix_binaural = "ep_test_01/binaural.wav"
        ep.audio_mix_foa = "ep_test_01/foa.wav"
        ep.visibility_overlay = "ep_test_01/overlay/"

        for fi in range(FRAME_COUNT):
            ep.spatial_frames.append({
                "frame_index": fi,
                "actors": {"source1": {"position_m": [0, 0, 0], "forward_xyzw": [0, 0, 0, 1], "listener_relative": {"distance_m": 2.0, "azimuth_deg": 0, "elevation_deg": 0}, "in_frustum": True}},
                "listener": {"position_m": [0, 1.6, 0], "forward_xyzw": [0, 0, 0, 1]},
            })
            ep.motion_frames.append({"frame_index": fi, "actor_states": {"source1": "idle"}})
            ep.visibility_frames.append({
                "frame_index": fi,
                "actor_visibility": {
                    "source1": make_visibility_record(1000, 900, True).as_dict(),
                },
            })

        ep.add_qa(QAPair(
            "q1", "test_type", "What color?", "blue",
            answer_unique=True, fact_observable=True,
        ))

        doc = ep.build()
        assert doc["schema"] == QA_EPISODE_SCHEMA
        assert doc["episode_id"] == "ep_test_01"
        assert "episode_content_sha256" in doc
        assert len(doc["facts"]["spatial_facts"]["per_frame"]) == FRAME_COUNT

    def test_empty_episode_fails_schema(self):
        errors = validate_qa_episode_schema({})
        assert len(errors) > 0

    def test_missing_required_fields(self):
        errors = validate_qa_episode_schema({"schema": QA_EPISODE_SCHEMA})
        assert len(errors) > 0

    def test_content_hash_mismatch(self):
        doc = {
            "schema": QA_EPISODE_SCHEMA,
            "episode_content_sha256": "0" * 64,
        }
        errors = validate_qa_episode(doc)
        assert any("content_sha256" in e for e in errors)


# ═══════════════════════════════════════════════════════════════════════════════
# Episode builder tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestEpisodeBuilder:
    def test_build_requires_timeline(self):
        ep = Episode("ep_test").add_actor("a1", "beagle_01", "dog", 1)
        with pytest.raises(EpisodeError, match="timeline"):
            ep.build()

    def test_actor_management(self):
        ep = (
            Episode("ep_test")
            .add_actor("a1", "beagle_01", "dog", 1, breed_id="beagle",
                       size="medium", coat_profile_id="dog_beagle_v1", coat_value="tri")
            .add_actor("a2", "human_01", "human", 2, top_color="blue",
                       emitter_anchor_id="mouth", emitter_anchor_type="mouth",
                       emitter_offset_m=(0.0, 1.61, 0.0))
        )
        assert len(ep.actors) == 2
        assert ep.actors[0]["actor_id"] == "a1"
        assert ep.actors[0]["identity"]["breed_id"] == "beagle"
        assert ep.actors[0]["realized_visual_attributes"]["coat_profile"]["value"] == "tri"
        assert ep.actors[1]["realized_visual_attributes"]["clothing"]["top_color"] == "blue"

    def test_sound_management(self):
        ep = (
            Episode("ep_test")
            .add_actor("a1", "human_01", "human", 1)
            .add_sound("speech_01", "human_speech", "a1",
                       transcript="Hello", sound_category="speech",
                       duration_samples=16000)
        )
        assert len(ep.sounds) == 1
        assert ep.sounds[0]["transcript"] == "Hello"
        assert ep.sounds[0]["bound_to_actor"] == "a1"

    def test_sound_fact_added(self):
        ep = (
            Episode("ep_test")
            .add_actor("a1", "beagle_01", "dog", 1)
            .add_sound_fact("evt_01", "a1", "bark_01", 0, 48000, transcript="")
        )
        assert len(ep.sound_facts) == 1
        assert ep.sound_facts[0]["event_id"] == "evt_01"
        assert ep.sound_facts[0]["start_tick"] == 0
        assert ep.sound_facts[0]["end_tick"] == 48000
        assert ep.sound_facts[0]["start_frame"] == 0

    def test_visibility_record_added(self):
        ep = (
            Episode("ep_test")
            .add_actor("a1", "cat_01", "cat", 1)
        )
        rec = make_visibility_record(1000, 500, True, touches_frame_border=False)
        ep.add_visibility_record(0, "a1", rec)
        assert len(ep.visibility_frames) == 1
        av = ep.visibility_frames[0]["actor_visibility"]["a1"]
        assert av["amodal_pixels"] == 1000
        assert av["visible_pixels"] == 500

    def test_furniture_occluder(self):
        ep = (
            Episode("ep_test")
            .add_actor("a1", "dog_01", "dog", 1)
            .add_furniture_occluder("table_01", "table", 100)
            .add_furniture_occluder("sofa_01", "sofa", 101)
        )
        assert len(ep.furniture_occluders) == 2
        assert ep.furniture_occluders[1]["semantic_label"] == "sofa"

    def test_sidecar_paths(self):
        ep = Episode("ep_test").add_actor("a1", "dog_01", "dog", 1).add_sound("bark_01", "dog_vocalization", "a1")
        ep.rgb_video = "ep_001/rgb.mp4"
        ep.semantic_video = "ep_001/semantic.mp4"
        ep.depth_frames = "ep_001/depth/"
        ep.target_only_masks = "ep_001/target_only/"
        ep.audio_mix_binaural = "ep_001/binaural.wav"
        ep.audio_mix_foa = "ep_001/foa.wav"
        ep.visibility_overlay = "ep_001/overlay/"
        ep.timeline = _valid_timeline()
        for fi in range(FRAME_COUNT):
            ep.spatial_frames.append({"frame_index": fi, "actors": {"a1": {"position_m": [0, 0, 0], "forward_xyzw": [0, 0, 0, 1], "listener_relative": {"distance_m": 2.0, "azimuth_deg": 0, "elevation_deg": 0}, "in_frustum": True}}, "listener": {"position_m": [0, 1.6, 0], "forward_xyzw": [0, 0, 0, 1]}})
            ep.motion_frames.append({"frame_index": fi, "actor_states": {"a1": "idle"}})
            ep.visibility_frames.append({"frame_index": fi, "actor_visibility": {"a1": make_visibility_record(1000, 900, True).as_dict()}})
        doc = ep.build()
        assert doc["sidecars"]["rgb_video"] == "ep_001/rgb.mp4"

    def test_qa_pair_validation_blocks_invalid_answer(self):
        ep = (
            Episode("ep_test")
            .add_actor("a1", "dog_01", "dog", 1)
        )
        ep.timeline = _valid_timeline()
        for fi in range(FRAME_COUNT):
            ep.spatial_frames.append({"frame_index": fi, "actors": {"a1": {"position_m": [0, 0, 0], "forward_xyzw": [0, 0, 0, 1], "listener_relative": {"distance_m": 2.0, "azimuth_deg": 0, "elevation_deg": 0}, "in_frustum": True}}, "listener": {"position_m": [0, 1.6, 0], "forward_xyzw": [0, 0, 0, 1]}})
            ep.motion_frames.append({"frame_index": fi, "actor_states": {"a1": "idle"}})
            ep.visibility_frames.append({"frame_index": fi, "actor_visibility": {"a1": make_visibility_record(1000, 900, True).as_dict()}})
        ep.add_qa(QAPair(
            "q1", "test", "Q?", "A",
            answer_unique=False, fact_observable=True,
            rejection_reason="duplicate answer possible",
        ))
        with pytest.raises(EpisodeError, match="not unique"):
            ep.build()

    def test_round_trip_content_hash_stable(self):
        ep = Episode("ep_stable").add_actor("a1", "cat_01", "cat", 1).add_sound("meow_01", "cat_vocalization", "a1")
        ep.rgb_video = "ep_stable/rgb.mp4"
        ep.semantic_video = "ep_stable/semantic.mp4"
        ep.depth_frames = "ep_stable/depth/"
        ep.target_only_masks = "ep_stable/target_only/"
        ep.audio_mix_binaural = "ep_stable/binaural.wav"
        ep.audio_mix_foa = "ep_stable/foa.wav"
        ep.visibility_overlay = "ep_stable/overlay/"
        ep.timeline = _valid_timeline()
        for fi in range(FRAME_COUNT):
            ep.spatial_frames.append({"frame_index": fi, "actors": {"a1": {"position_m": [0, 0, 0], "forward_xyzw": [0, 0, 0, 1], "listener_relative": {"distance_m": 2.0, "azimuth_deg": 0, "elevation_deg": 0}, "in_frustum": True}}, "listener": {"position_m": [0, 1.6, 0], "forward_xyzw": [0, 0, 0, 1]}})
            ep.motion_frames.append({"frame_index": fi, "actor_states": {"a1": "idle"}})
            ep.visibility_frames.append({"frame_index": fi, "actor_visibility": {"a1": make_visibility_record(1000, 900, True).as_dict()}})
        doc1 = ep.build()
        doc2 = ep.build()
        assert doc1["episode_content_sha256"] == doc2["episode_content_sha256"]

    def test_seed_in_provenance(self):
        ep = Episode("ep_seed").add_actor("a1", "dog_01", "dog", 1).add_sound("bark_01", "dog_vocalization", "a1")
        ep.seed = 42
        ep.rgb_video = "ep_seed/rgb.mp4"
        ep.semantic_video = "ep_seed/semantic.mp4"
        ep.depth_frames = "ep_seed/depth/"
        ep.target_only_masks = "ep_seed/target_only/"
        ep.audio_mix_binaural = "ep_seed/binaural.wav"
        ep.audio_mix_foa = "ep_seed/foa.wav"
        ep.visibility_overlay = "ep_seed/overlay/"
        ep.timeline = _valid_timeline()
        for fi in range(FRAME_COUNT):
            ep.spatial_frames.append({"frame_index": fi, "actors": {"a1": {"position_m": [0, 0, 0], "forward_xyzw": [0, 0, 0, 1], "listener_relative": {"distance_m": 2.0, "azimuth_deg": 0, "elevation_deg": 0}, "in_frustum": True}}, "listener": {"position_m": [0, 1.6, 0], "forward_xyzw": [0, 0, 0, 1]}})
            ep.motion_frames.append({"frame_index": fi, "actor_states": {"a1": "idle"}})
            ep.visibility_frames.append({"frame_index": fi, "actor_visibility": {"a1": make_visibility_record(1000, 900, True).as_dict()}})
        doc = ep.build()
        assert doc["provenance"]["seed"] == 42


# ═══════════════════════════════════════════════════════════════════════════════
# Visibility classification tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestVisibilityClassification:
    def test_out_of_view_not_in_frustum(self):
        assert classify_visibility(5000, 5000, False) == VISIBILITY_OUT_OF_VIEW

    def test_out_of_view_zero_amodal(self):
        assert classify_visibility(0, 0, True) == VISIBILITY_OUT_OF_VIEW

    def test_visible_clear(self):
        assert classify_visibility(1000, 950, True) == VISIBILITY_CLEAR

    def test_visible_occluded(self):
        assert classify_visibility(1000, 500, True) == VISIBILITY_OCCLUDED

    def test_fully_occluded(self):
        assert classify_visibility(1000, 10, True) == VISIBILITY_FULLY_OCCLUDED

    def test_custom_thresholds(self):
        assert classify_visibility(1000, 800, True, clear_threshold=0.95) == VISIBILITY_OCCLUDED
        assert classify_visibility(1000, 800, True, clear_threshold=0.70) == VISIBILITY_CLEAR
        assert classify_visibility(1000, 30, True, visible_threshold=0.10) == VISIBILITY_FULLY_OCCLUDED

    def test_make_visibility_record_auto_classification(self):
        rec = make_visibility_record(1000, 360, True, touches_frame_border=False)
        assert rec.visibility_state == VISIBILITY_OCCLUDED
        assert rec.visible_fraction == 0.36


# ═══════════════════════════════════════════════════════════════════════════════
# Event detection tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestEventDetection:
    def _make_frames(self, states: list[tuple[str, float]]):
        """Make a list of per-frame dicts from (state, fraction) tuples."""
        frames = []
        for fi, (state, frac) in enumerate(states):
            frames.append({
                "frame_index": fi,
                "actor_visibility": {
                    "a1": {
                        "amodal_pixels": 1000,
                        "visible_pixels": int(1000 * frac),
                        "visible_fraction": frac,
                        "visibility_state": state,
                        "touches_frame_border": False,
                    }
                },
            })
        return frames

    def test_enter_frustum_detected(self):
        frames = self._make_frames([
            (VISIBILITY_OUT_OF_VIEW, 0.0),
            (VISIBILITY_OUT_OF_VIEW, 0.0),
            (VISIBILITY_CLEAR, 1.0),
            (VISIBILITY_CLEAR, 1.0),
        ])
        events = detect_visibility_events(frames, "a1")
        assert len(events) == 2  # enter_frustum + become_visible
        assert events[0].event_type == EVENT_ENTER_FRUSTUM
        assert events[0].frame_index == 2

    def test_exit_frustum_detected(self):
        frames = self._make_frames([
            (VISIBILITY_CLEAR, 1.0),
            (VISIBILITY_CLEAR, 1.0),
            (VISIBILITY_OUT_OF_VIEW, 0.0),
        ])
        events = detect_visibility_events(frames, "a1")
        assert any(e.event_type == EVENT_EXIT_FRUSTUM for e in events)

    def test_occlusion_start_detected(self):
        frames = self._make_frames([
            (VISIBILITY_CLEAR, 0.95),
            (VISIBILITY_OCCLUDED, 0.40),
            (VISIBILITY_OCCLUDED, 0.30),
        ])
        events = detect_visibility_events(frames, "a1")
        assert any(e.event_type == EVENT_OCCLUSION_START for e in events)
        occ = next(e for e in events if e.event_type == EVENT_OCCLUSION_START)
        assert occ.frame_index == 1

    def test_reappear_detected(self):
        frames = self._make_frames([
            (VISIBILITY_CLEAR, 1.0),
            (VISIBILITY_OCCLUDED, 0.50),
            (VISIBILITY_FULLY_OCCLUDED, 0.02),
            (VISIBILITY_FULLY_OCCLUDED, 0.01),
            (VISIBILITY_OCCLUDED, 0.30),
            (VISIBILITY_CLEAR, 0.95),
        ])
        events = detect_visibility_events(frames, "a1")
        event_types = [e.event_type for e in events]
        assert EVENT_FULLY_OCCLUDED in event_types
        assert EVENT_REAPPEAR in event_types
        reappear = next(e for e in events if e.event_type == EVENT_REAPPEAR)
        assert reappear.frame_index == 4

    def test_no_events_on_static_clear(self):
        frames = self._make_frames([
            (VISIBILITY_CLEAR, 1.0),
            (VISIBILITY_CLEAR, 1.0),
            (VISIBILITY_CLEAR, 1.0),
        ])
        events = detect_visibility_events(frames, "a1")
        assert len(events) == 0

    def test_camera_motion_occlusion_reappear_sequence(self):
        """Canary 5 scenario: target → occluded → reappear as camera moves."""
        frames = self._make_frames([
            (VISIBILITY_CLEAR, 0.95),
            (VISIBILITY_CLEAR, 0.92),
            (VISIBILITY_OCCLUDED, 0.55),
            (VISIBILITY_OCCLUDED, 0.25),
            (VISIBILITY_FULLY_OCCLUDED, 0.02),
            (VISIBILITY_FULLY_OCCLUDED, 0.0),
            (VISIBILITY_OCCLUDED, 0.20),
            (VISIBILITY_OCCLUDED, 0.40),
            (VISIBILITY_CLEAR, 0.95),
        ])
        events = detect_visibility_events(frames, "a1")
        types = [e.event_type for e in events]
        # occlusion_start → fully_occluded → reappear
        assert EVENT_OCCLUSION_START in types
        assert EVENT_FULLY_OCCLUDED in types
        assert EVENT_REAPPEAR in types
        # occlusion_start comes before fully_occluded comes before reappear
        occ_idx = types.index(EVENT_OCCLUSION_START)
        full_idx = types.index(EVENT_FULLY_OCCLUDED)
        reap_idx = types.index(EVENT_REAPPEAR)
        assert occ_idx < full_idx < reap_idx


# ═══════════════════════════════════════════════════════════════════════════════
# QAPair tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestQAPair:
    def test_minimal_qa_pair(self):
        qa = QAPair("q1", "appearance_to_vocalization", "Did the blue-shirt person speak?",
                     "yes", answer_unique=True, fact_observable=True)
        d = qa.as_dict()
        assert d["question_id"] == "q1"
        assert d["validation"]["answer_unique"] is True
        assert d["validation"]["fact_observable"] is True

    def test_qa_pair_with_choices(self):
        qa = QAPair("q2", "direction", "Which side?", "left",
                     answer_unique=True, fact_observable=True,
                     choices=("left", "right"))
        d = qa.as_dict()
        assert d["choices"] == ["left", "right"]

    def test_qa_pair_with_rejection(self):
        qa = QAPair("q3", "appearance", "Q?", "A",
                     answer_unique=False, fact_observable=True,
                     rejection_reason="two people wear the same color")
        d = qa.as_dict()
        assert d["validation"]["rejection_reason"] == "two people wear the same color"


# ═══════════════════════════════════════════════════════════════════════════════
# EpisodeEvent tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestEpisodeEvent:
    def test_event_serialization(self):
        evt = EpisodeEvent(EVENT_OCCLUSION_START, 5, "source1")
        d = evt.as_dict()
        assert d["event_type"] == EVENT_OCCLUSION_START
        assert d["frame_index"] == 5
        assert d["actor_id"] == "source1"

    def test_event_with_occluder(self):
        evt = EpisodeEvent(
            EVENT_OCCLUSION_START, 5, "source1",
            occluder={"occluder_type": "furniture", "instance_id": "table_01", "semantic_label": "table"},
        )
        d = evt.as_dict()
        assert d["occluder"]["semantic_label"] == "table"
