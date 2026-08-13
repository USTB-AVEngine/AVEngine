"""Task 1 Acceptance Tests — Unified QA Episode Data Structure.

Run from the repository root::

    python -m pytest tests/acceptance/test_task1_acceptance.py -v

Each section mirrors the acceptance checklist in docs/planning/TASK1_DEVLOG.md.
"""

from __future__ import annotations

import pytest

from avengine.m5.timeline import FRAME_COUNT
from avengine.qa.episode import (
    DEFAULT_CLEAR_THRESHOLD,
    DEFAULT_VISIBLE_THRESHOLD,
    Episode,
    EpisodeError,
    EpisodeEvent,
    QAPair,
    QA_EPISODE_SCHEMA,
    VisibilityRecord,
    VISIBILITY_CLEAR,
    VISIBILITY_FULLY_OCCLUDED,
    VISIBILITY_OCCLUDED,
    VISIBILITY_OUT_OF_VIEW,
    classify_visibility,
    detect_visibility_events,
    make_visibility_record,
    validate_qa_episode,
    validate_qa_episode_schema,
)

# Re-use the shared minimal-timeline fixture from the unit test module.
from tests.unit.test_qa_episode import _valid_timeline


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _minimal_episode() -> Episode:
    """Return an Episode builder pre-populated with the minimum valid data."""
    ep = (
        Episode("ep_accept")
        .add_actor("a1", "beagle_01", "dog", 1, breed_id="beagle",
                   size="medium", body_build="standard", life_stage="adult")
        .add_sound("bark_01", "dog_vocalization", "a1", sound_category="bark")
    )
    ep.timeline = _valid_timeline()
    ep.rgb_video = "accept/rgb.mp4"
    ep.semantic_video = "accept/semantic.mp4"
    ep.depth_frames = "accept/depth/"
    ep.target_only_masks = "accept/target_only/"
    ep.audio_mix_binaural = "accept/binaural.wav"
    ep.audio_mix_foa = "accept/foa.wav"
    ep.visibility_overlay = "accept/overlay/"
    ep.seed = 42

    for fi in range(FRAME_COUNT):
        ep.spatial_frames.append({
            "frame_index": fi,
            "actors": {
                "a1": {
                    "position_m": [0.0, 0.0, 0.0],
                    "forward_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "listener_relative": {
                        "distance_m": 2.0,
                        "azimuth_deg": 0.0,
                        "elevation_deg": 0.0,
                    },
                    "in_frustum": True,
                },
            },
            "listener": {
                "position_m": [0.0, 1.6, 0.0],
                "forward_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        })
        ep.motion_frames.append({
            "frame_index": fi,
            "actor_states": {"a1": "idle"},
        })
        ep.visibility_frames.append({
            "frame_index": fi,
            "actor_visibility": {
                "a1": make_visibility_record(1000, 900, True).as_dict(),
            },
        })

    return ep


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Package imports — every public symbol must resolve
# ═══════════════════════════════════════════════════════════════════════════════


class TestPackageImports:
    """Verify that ``from avengine.qa import ...`` exposes every public symbol."""

    def test_imports_from_package_top(self):
        from avengine.qa import (  # noqa: F811
            Episode, EpisodeError, EpisodeEvent, QAPair, VisibilityRecord,
            QA_EPISODE_SCHEMA,
            DEFAULT_CLEAR_THRESHOLD, DEFAULT_VISIBLE_THRESHOLD,
            VISIBILITY_OUT_OF_VIEW, VISIBILITY_CLEAR,
            VISIBILITY_OCCLUDED, VISIBILITY_FULLY_OCCLUDED, VISIBILITY_STATES,
            EVENT_ENTER_FRUSTUM, EVENT_EXIT_FRUSTUM, EVENT_BECOME_VISIBLE,
            EVENT_OCCLUSION_START, EVENT_FULLY_OCCLUDED, EVENT_REAPPEAR,
            EVENT_TYPES,
            MOTION_IDLE, MOTION_WALK, MOTION_OTHER, MOTION_STATES,
            OCCLUDER_ACTOR, OCCLUDER_FURNITURE, OCCLUDER_UNKNOWN, OCCLUDER_TYPES,
            classify_visibility, detect_visibility_events, make_visibility_record,
            validate_qa_episode, validate_qa_episode_schema,
        )
        # If we reach here, all symbols resolved.
        assert Episode is not None

    def test_imports_from_episode_module(self):
        """Direct module import must also resolve every symbol."""
        from avengine.qa.episode import (  # noqa: F811
            Episode, EpisodeError, EpisodeEvent, QAPair, VisibilityRecord,
            QA_EPISODE_SCHEMA,
            classify_visibility, detect_visibility_events, make_visibility_record,
            validate_qa_episode, validate_qa_episode_schema,
        )
        assert Episode is not None

    def test_public_symbols_in_all(self):
        """Every symbol listed in __all__ must actually exist on the module."""
        import avengine.qa.episode as mod
        for name in mod.__all__:
            assert hasattr(mod, name), f"{name} is in __all__ but missing from module"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Schema — validation and structural invariants
# ═══════════════════════════════════════════════════════════════════════════════


class TestSchemaAcceptance:
    """Schema-level checks that are broader than the unit tests."""

    def test_schema_constant_value(self):
        assert QA_EPISODE_SCHEMA == "avengine_qa_episode_v1"

    def test_empty_document_fails_schema(self):
        errors = validate_qa_episode_schema({})
        assert len(errors) > 0

    def test_minimal_valid_document_passes(self):
        ep = _minimal_episode()
        ep.add_qa(QAPair("q1", "test", "Q?", "A",
                         answer_unique=True, fact_observable=True))
        doc = ep.build()
        errors = validate_qa_episode(doc)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_required_top_level_fields(self):
        """Every top-level required field is present in a built doc."""
        ep = _minimal_episode()
        ep.add_qa(QAPair("q1", "test", "Q?", "A",
                         answer_unique=True, fact_observable=True))
        doc = ep.build()

        required = {
            "schema", "episode_id", "created", "assets_used", "scene",
            "timeline", "facts", "qa_pairs", "sidecars", "provenance",
            "episode_content_sha256",
        }
        missing = required - set(doc.keys())
        assert missing == set(), f"Missing top-level keys: {missing}"

    def test_content_hash_length_and_format(self):
        ep = _minimal_episode()
        ep.add_qa(QAPair("q1", "test", "Q?", "A",
                         answer_unique=True, fact_observable=True))
        doc = ep.build()
        sha = doc["episode_content_sha256"]
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Episode builder — end-to-end construction
# ═══════════════════════════════════════════════════════════════════════════════


class TestEpisodeBuilderAcceptance:
    """Full end-to-end Episode construction and validation."""

    def test_full_episode_builds_without_error(self):
        ep = _minimal_episode()
        ep.add_sound_fact("evt_01", "a1", "bark_01", 0, 48000)
        ep.add_furniture_occluder("table_01", "table", 100)
        ep.add_qa(QAPair("q1", "sound_presence", "Is the dog barking?", "yes",
                         answer_unique=True, fact_observable=True))
        doc = ep.build()
        assert doc["schema"] == QA_EPISODE_SCHEMA
        assert doc["episode_id"] == "ep_accept"
        assert len(doc["assets_used"]["actors"]) == 1
        assert len(doc["assets_used"]["sounds"]) == 1
        assert len(doc["facts"]["sound_facts"]) == 1
        assert len(doc["facts"]["spatial_facts"]["per_frame"]) == FRAME_COUNT
        assert len(doc["facts"]["motion_facts"]["per_frame"]) == FRAME_COUNT
        assert len(doc["facts"]["visibility_facts"]["per_frame"]) == FRAME_COUNT
        assert len(doc["qa_pairs"]) == 1
        assert len(doc["sidecars"]["rgb_video"]) > 0
        assert doc["provenance"]["seed"] == 42

    def test_build_requires_timeline(self):
        ep = Episode("ep_test").add_actor("a1", "dog_01", "dog", 1)
        with pytest.raises(EpisodeError, match="timeline"):
            ep.build()

    def test_build_requires_at_least_one_actor(self):
        ep = Episode("ep_no_actor")
        ep.timeline = _valid_timeline()
        with pytest.raises(EpisodeError, match="at least one actor"):
            ep.build()

    def test_actor_fields_round_trip(self):
        ep = _minimal_episode()
        ep.add_qa(QAPair("q1", "test", "Q?", "A",
                         answer_unique=True, fact_observable=True))
        doc = ep.build()
        actor = doc["assets_used"]["actors"][0]
        assert actor["actor_id"] == "a1"
        assert actor["entity_asset_id"] == "beagle_01"
        assert actor["identity"]["species_id"] == "dog"
        assert actor["identity"]["breed_id"] == "beagle"
        assert actor["realized_visual_attributes"]["size"] == "medium"
        assert actor["realized_visual_attributes"]["body_build"] == "standard"
        assert actor["realized_visual_attributes"]["life_stage"] == "adult"
        assert "emitter_anchor" in actor

    def test_furniture_occluder_optional(self):
        """build() must succeed both with and without furniture."""
        ep = _minimal_episode()
        ep.add_qa(QAPair("q1", "test", "Q?", "A",
                         answer_unique=True, fact_observable=True))
        doc_no_furniture = ep.build()
        assert "furniture_occluders" not in doc_no_furniture["scene"]

        ep.add_furniture_occluder("sofa_01", "sofa", 101)
        doc_with = ep.build()
        assert "furniture_occluders" in doc_with["scene"]
        assert len(doc_with["scene"]["furniture_occluders"]) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Content hash stability
# ═══════════════════════════════════════════════════════════════════════════════


class TestContentHashAcceptance:
    """Content hashes must be deterministic and tamper-evident."""

    def test_identical_inputs_produce_identical_hash(self):
        ep = _minimal_episode()
        ep.add_qa(QAPair("q1", "test", "Q?", "A",
                         answer_unique=True, fact_observable=True))
        h1 = ep.build()["episode_content_sha256"]
        h2 = ep.build()["episode_content_sha256"]
        assert h1 == h2

    def test_different_inputs_produce_different_hash(self):
        ep1 = _minimal_episode()
        ep1.add_qa(QAPair("q1", "test", "Q?", "A",
                          answer_unique=True, fact_observable=True))
        h1 = ep1.build()["episode_content_sha256"]

        ep2 = _minimal_episode()
        ep2.add_qa(QAPair("q1", "test", "Q?", "B",
                          answer_unique=True, fact_observable=True))
        h2 = ep2.build()["episode_content_sha256"]

        assert h1 != h2, "Different answers must produce different hashes"

    def test_hash_covers_full_document(self):
        """Tampering with a sidecar path must change the hash."""
        ep = _minimal_episode()
        ep.add_qa(QAPair("q1", "test", "Q?", "A",
                         answer_unique=True, fact_observable=True))
        doc1 = ep.build()

        # Mutate a sidecar path — this should invalidate the hash
        ep.rgb_video = "accept/tampered.mp4"
        doc2 = ep.build()

        assert doc1["episode_content_sha256"] != doc2["episode_content_sha256"]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Visibility classification — all four states + thresholds
# ═══════════════════════════════════════════════════════════════════════════════


class TestVisibilityAcceptance:
    """Acceptance-level visibility classification checks."""

    def test_all_four_states_are_distinct(self):
        states = {
            VISIBILITY_OUT_OF_VIEW,
            VISIBILITY_CLEAR,
            VISIBILITY_OCCLUDED,
            VISIBILITY_FULLY_OCCLUDED,
        }
        assert len(states) == 4

    def test_out_of_view_when_not_in_frustum(self):
        assert classify_visibility(5000, 5000, False) == VISIBILITY_OUT_OF_VIEW

    def test_out_of_view_when_zero_amodal(self):
        assert classify_visibility(0, 0, True) == VISIBILITY_OUT_OF_VIEW

    def test_visible_clear_at_default_threshold(self):
        assert classify_visibility(1000, 950, True) == VISIBILITY_CLEAR
        assert classify_visibility(1000, 900, True) == VISIBILITY_CLEAR  # boundary

    def test_visible_occluded_between_thresholds(self):
        assert classify_visibility(1000, 500, True) == VISIBILITY_OCCLUDED

    def test_fully_occluded_below_visible_threshold(self):
        assert classify_visibility(1000, 10, True) == VISIBILITY_FULLY_OCCLUDED
        # Strict inequality: fraction < visible_threshold (0.05).
        # 49/1000 = 0.049 → fully_occluded;  50/1000 = 0.05 → visible_occluded.
        assert classify_visibility(1000, 49, True) == VISIBILITY_FULLY_OCCLUDED
        assert classify_visibility(1000, 50, True) == VISIBILITY_OCCLUDED

    def test_make_visibility_record_auto_classifies(self):
        rec = make_visibility_record(1000, 360, True, touches_frame_border=True)
        assert rec.amodal_pixels == 1000
        assert rec.visible_pixels == 360
        assert rec.visible_fraction == 0.36
        assert rec.visibility_state == VISIBILITY_OCCLUDED
        assert rec.touches_frame_border is True


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Event detection — all six event types
# ═══════════════════════════════════════════════════════════════════════════════


class TestEventDetectionAcceptance:
    """Acceptance-level event detection checks."""

    @staticmethod
    def _frames(states):
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
                    },
                },
            })
        return frames

    def test_enter_and_exit_frustum(self):
        frames = self._frames([
            (VISIBILITY_OUT_OF_VIEW, 0.0),
            (VISIBILITY_CLEAR, 1.0),
            (VISIBILITY_OUT_OF_VIEW, 0.0),
        ])
        events = detect_visibility_events(frames, "a1")
        types = {e.event_type for e in events}
        assert "enter_frustum" in types
        assert "exit_frustum" in types

    def test_occlusion_start_and_fully_occluded(self):
        frames = self._frames([
            (VISIBILITY_CLEAR, 0.95),
            (VISIBILITY_OCCLUDED, 0.40),
            (VISIBILITY_OCCLUDED, 0.25),
            (VISIBILITY_FULLY_OCCLUDED, 0.02),
            (VISIBILITY_FULLY_OCCLUDED, 0.01),
        ])
        events = detect_visibility_events(frames, "a1")
        types = {e.event_type for e in events}
        assert "occlusion_start" in types
        assert "fully_occluded" in types

    def test_reappear_after_full_occlusion(self):
        frames = self._frames([
            (VISIBILITY_CLEAR, 1.0),
            (VISIBILITY_FULLY_OCCLUDED, 0.01),
            (VISIBILITY_FULLY_OCCLUDED, 0.01),
            (VISIBILITY_OCCLUDED, 0.40),
            (VISIBILITY_CLEAR, 0.95),
        ])
        events = detect_visibility_events(frames, "a1")
        types = {e.event_type for e in events}
        assert "fully_occluded" in types
        assert "reappear" in types

    def test_static_clear_produces_no_events(self):
        frames = self._frames([
            (VISIBILITY_CLEAR, 1.0),
            (VISIBILITY_CLEAR, 1.0),
            (VISIBILITY_CLEAR, 1.0),
        ])
        events = detect_visibility_events(frames, "a1")
        assert len(events) == 0

    def test_canary_5_sequence_ordering(self):
        """Target → occluded → reappear as camera moves (Canary 5)."""
        frames = self._frames([
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
        occ_idx = types.index("occlusion_start")
        full_idx = types.index("fully_occluded")
        reap_idx = types.index("reappear")
        assert occ_idx < full_idx < reap_idx


# ═══════════════════════════════════════════════════════════════════════════════
# 7. QAPair — construction and rejection
# ═══════════════════════════════════════════════════════════════════════════════


class TestQAPairAcceptance:
    """Acceptance-level QA pair checks."""

    def test_minimal_qa_pair_round_trips(self):
        qa = QAPair("q1", "sound_presence", "Is the dog barking?", "yes",
                    answer_unique=True, fact_observable=True)
        d = qa.as_dict()
        assert d["question_id"] == "q1"
        assert d["question_type"] == "sound_presence"
        assert d["validation"]["answer_unique"] is True
        assert d["validation"]["fact_observable"] is True

    def test_qa_pair_with_choices(self):
        qa = QAPair("q2", "direction", "Which side?", "left",
                    answer_unique=True, fact_observable=True,
                    choices=("left", "right", "front", "behind"))
        d = qa.as_dict()
        assert len(d["choices"]) == 4

    def test_non_unique_answer_is_rejected_at_build(self):
        ep = _minimal_episode()
        ep.add_qa(QAPair("q1", "test", "Q?", "A",
                         answer_unique=False, fact_observable=True,
                         rejection_reason="duplicate possible"))
        with pytest.raises(EpisodeError, match="not unique"):
            ep.build()

    def test_non_observable_fact_is_rejected_at_build(self):
        ep = _minimal_episode()
        ep.add_qa(QAPair("q1", "test", "Q?", "A",
                         answer_unique=True, fact_observable=False,
                         rejection_reason="occluded for whole episode"))
        with pytest.raises(EpisodeError, match="not observable"):
            ep.build()

    def test_duplicate_question_ids_are_rejected(self):
        ep = _minimal_episode()
        ep.add_qa(QAPair("q1", "test", "Q1?", "A",
                         answer_unique=True, fact_observable=True))
        ep.add_qa(QAPair("q1", "test", "Q2?", "B",
                         answer_unique=True, fact_observable=True))
        with pytest.raises(EpisodeError, match="not unique"):
            ep.build()


# ═══════════════════════════════════════════════════════════════════════════════
# 8. EpisodeEvent — serialization
# ═══════════════════════════════════════════════════════════════════════════════


class TestEpisodeEventAcceptance:
    """Acceptance-level event serialization checks."""

    def test_event_without_occluder(self):
        evt = EpisodeEvent("occlusion_start", 5, "source1")
        d = evt.as_dict()
        assert d["event_type"] == "occlusion_start"
        assert d["frame_index"] == 5
        assert d["actor_id"] == "source1"
        assert "occluder" not in d

    def test_event_with_furniture_occluder(self):
        evt = EpisodeEvent(
            "occlusion_start", 5, "source1",
            occluder={
                "occluder_type": "furniture",
                "instance_id": "table_01",
                "semantic_label": "table",
            },
        )
        d = evt.as_dict()
        assert d["occluder"]["occluder_type"] == "furniture"
        assert d["occluder"]["instance_id"] == "table_01"

    def test_event_with_actor_occluder(self):
        evt = EpisodeEvent(
            "occlusion_start", 7, "source1",
            occluder={
                "occluder_type": "actor",
                "actor_id": "source2",
            },
        )
        d = evt.as_dict()
        assert d["occluder"]["occluder_type"] == "actor"
        assert d["occluder"]["actor_id"] == "source2"


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Provenance — seed and optional fields
# ═══════════════════════════════════════════════════════════════════════════════


class TestProvenanceAcceptance:
    """Acceptance-level provenance checks."""

    def test_seed_is_recorded(self):
        ep = _minimal_episode()
        ep.seed = 12345
        ep.add_qa(QAPair("q1", "test", "Q?", "A",
                         answer_unique=True, fact_observable=True))
        doc = ep.build()
        assert doc["provenance"]["seed"] == 12345

    def test_optional_commits_are_included_when_set(self):
        ep = _minimal_episode()
        ep.avengine_commit = "abc123"
        ep.habitat_commit = "def456"
        ep.add_qa(QAPair("q1", "test", "Q?", "A",
                         answer_unique=True, fact_observable=True))
        doc = ep.build()
        assert doc["provenance"]["avengine_commit"] == "abc123"
        assert doc["provenance"]["habitat_commit"] == "def456"

    def test_optional_commits_are_absent_when_not_set(self):
        ep = _minimal_episode()
        ep.add_qa(QAPair("q1", "test", "Q?", "A",
                         answer_unique=True, fact_observable=True))
        doc = ep.build()
        assert "avengine_commit" not in doc["provenance"]
        assert "habitat_commit" not in doc["provenance"]
