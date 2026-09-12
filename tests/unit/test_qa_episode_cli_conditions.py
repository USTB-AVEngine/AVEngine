"""The plan-only CLI must plan the conditions its QA targets asked for.

These go through ``run_qa_episode.plan_request``, not the sampler library,
because the defect they pin lived in the CLI seam: the controller used to
pre-resolve a base condition profile with no question knobs and hand it to
``build_qa_episode_plan``, which made ``resolve_conditioned_request`` keep the
caller's profile and return before any compiled branch knob was applied. Every
branch knob was computed, recorded and dropped, and the snapshot the CLI wrote
described the dropped profile rather than the one the plan used.

Room packages, catalogs, renderer dispatch and plan snapshots are stubbed: they
are I/O this seam does not decide. The chain under test is real from
``plan_request`` down through ``build_qa_episode_plan`` ->
``solve_conditioned_episode`` -> ``build_conditioned_plan`` ->
``resolve_conditioned_request``, so a branch knob has to survive the whole way
for these to pass. Nothing here renders a pixel or claims any native evidence.
"""
from __future__ import annotations

import importlib.util
import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPOSITORY = Path(__file__).resolve().parents[2]


def _planner_declaration_is_accepted():
    """Whether the planner's live capability declaration still compiles.

    These tests use the planner's real declaration on purpose: a mocked one
    would let the CLI seam pass while the compiler and the planner disagreed
    about the knob vocabulary. When those two do disagree the failure belongs to
    whoever owns that pair, not to the controller, so the file says so and skips
    instead of reporting a refusal it does not own. The condition clears itself
    the moment the two agree again.
    """
    from avengine.qa import generation_conditions as conditions
    from avengine.rooms import conditioned_sampler as sampler
    try:
        conditions.resolve_generator_capabilities(sampler)
    except conditions.GenerationConditionError as error:
        return False, str(error)
    return True, ""


_DECLARATION_OK, _DECLARATION_ERROR = _planner_declaration_is_accepted()
pytestmark = pytest.mark.skipif(
    not _DECLARATION_OK,
    reason=("avengine.rooms.conditioned_sampler.describe_generator_capabilities and "
            "avengine.qa.generation_conditions._one_declaration currently disagree, so no "
            "conditioned plan can be solved by any caller: " + _DECLARATION_ERROR))


def load_controller():
    # AVENGINE_CLI_UNDER_TEST lets these run against an archived copy of the
    # controller, which is how the two defects they pin were demonstrated to
    # fail before the fix and pass after it. Unset, they test the real file.
    path = Path(os.environ.get("AVENGINE_CLI_UNDER_TEST")
                or REPOSITORY / "tools/studio/run_qa_episode.py")
    spec = importlib.util.spec_from_file_location("run_qa_episode_cli_conditions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The controller derives REPOSITORY from its own file location, which is
    # correct in place and wrong for an archived copy. Point it at the repository
    # under test either way, so an archived copy still records the same commit.
    module.REPOSITORY = REPOSITORY
    return module


# Two articulated humans and a prepared speech pool, in the shape the sampler
# is handed in production. Mirrors the fixtures in
# tests/unit/test_qa_conditioned_transition.py: source_activity_intervals_samples
# is what the P25 effective-sound cropping writes, and QA-08 compiles
# require_source_activity_measurement, so a row without it is refused before
# any visibility knob is reached.
TIMELINE = {"idle_action_id": "idle", "walking_action_id": "walk",
            "walk_phase_period_frames": 30, "body_plan_id": "biped_v1",
            "template_id": "human_v1", "local_anatomical_forward_axis": [1.0, 0.0, 0.0]}


def fixture_registry():
    return {"assets": [
        {"asset_id": f"human_{index}", "revision": "v1",
         "entity_class": "articulated_human", "identity": {"species_id": "human"},
         "display_label": f"person {index}",
         "realized_attributes": {"sex_or_gender_label": "male", "top_color": color},
         "timeline": dict(TIMELINE), "default_emitter_anchor_id": "mouth",
         "emitter_anchors": [{"anchor_id": "mouth", "offset_m": [0.0, 1.6, 0.0],
                              "offset_space": "final_scaled_asset_root"}]}
        for index, color in enumerate(["blue", "green"])]}


def fixture_sounds():
    return [{"sound_asset_id": f"speech_{index}", "sound_class": "speech", "gender": "M",
             "transcript": f"utterance {index}", "sample_count": 32000,
             "sample_rate_hz": 16000, "audible_start_sample": 800,
             "audible_end_sample_exclusive": 31200,
             "source_activity_intervals_samples": [[800, 31200]],
             "active_duration_s": 1.9, "path": f"/prepared/{index}.wav"}
            for index in range(2)]


def planning_resources():
    """One flat walkable room, no occluding mesh: real objects, no scene file."""
    from avengine.qa.answerability import MeshHandle
    from avengine.rooms.walkable_space import RasterWalkableSpace
    from avengine.routes.raster_pathfinder import RasterPathfinder

    pathfinder = RasterPathfinder(np.ones((40, 40), dtype=bool),
                                  bounds_m=[[0, -1, 0], [10, 1, 10]], floor_height_m=0.0)
    space = RasterWalkableSpace(pathfinder, {"floor_height_m": 0.0, "resolution_m": 0.25,
                                             "authority": "fixture_retained_grid"})
    mesh = MeshHandle(np.zeros((0, 3)), np.zeros((0, 3), dtype=int))
    layout = {"capture_resolution_hw": [720, 1280], "visual_lighting": {}}
    return space, mesh, layout


def request_body(branch, *, stated_profile=None, qa_id="QA-08"):
    profile = {"anchor_count": 1, "separation_bin_deg": [15, 180],
               "reserve_tail_s": 1.0, "retry_budget_within_profile": 60,
               "distance_range_m": [1.0, 8.0]}
    profile.update(stated_profile or {})
    return {
        "schema": "avengine_qa_episode_request_v1",
        "episode_id": f"cli_{qa_id.replace('-', '').lower()}_{branch}",
        "seed": 7,
        "sampling_policy": "conditioned_static_v2",
        "room_id": "fixture_room",
        "room_catalog": "<filled in by the fixture>",
        "sound_pool": "<filled in by the fixture>",
        "frame_count": 150, "frame_rate_hz": 15, "sample_rate_hz": 16000,
        "camera": {"motion": "static", "fov_deg": 85.0, "resolution_hw": [720, 1280],
                   "height_above_floor_m": 1.55},
        "entities": {"total_count": 2, "silent_count": 0, "min_articulated_count": 2,
                     "instances": [
                         {"instance_id": "hidden_target", "asset_id": "human_0",
                          "role": "anchor", "source_class": "articulated_human",
                          "speaking": True},
                         {"instance_id": "visible_speaker", "asset_id": "human_1",
                          "source_class": "articulated_human", "speaking": True}]},
        "source_asset_ids": ["human_0", "human_1"],
        "profile": profile,
        "qa_targets": [{
            "qa_id": qa_id, "branch": branch,
            "target_instance_ids": ["hidden_target"],
            "event": {"kind": "target_audible_window"},
            "items": 1, "forms": ["mcq", "open"], "target_source": "config",
        }],
    }


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """A controller whose room layer is stubbed and whose planner is the real one."""
    controller = load_controller()
    space, mesh, layout = planning_resources()

    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(
        {"schema": "avengine_room_catalog_v1",
         "rooms": [{"room_id": "fixture_room", "room_package": "fixture.json"}]}),
        encoding="utf-8")
    sound_path = tmp_path / "sounds.json"
    sound_path.write_text(json.dumps({"sounds": fixture_sounds()}), encoding="utf-8")

    package = {"room_id": "fixture_room", "schema": "avengine_room_package_v1"}
    route = SimpleNamespace(renderer="ue_spear", family="fixture",
                            walkable_kind="raster", planning_adapter="fixture",
                            production_family="fixture")

    monkeypatch.setattr(controller, "load_source_asset_runtime_registry",
                        lambda *a, **k: fixture_registry())
    monkeypatch.setattr(controller, "request_package_runtime", lambda request, catalog: {})
    monkeypatch.setattr(controller, "request_host_runtime_config", lambda *a, **k: {})
    monkeypatch.setattr(controller, "load_profile_registry", lambda *a, **k: {})
    monkeypatch.setattr(controller, "resolve_room_profile", lambda *a, **k: {})
    monkeypatch.setattr(controller, "room_render_parameters",
                        lambda profile, request: {"profile_id": "fixture_profile"})
    monkeypatch.setattr(controller, "request_with_effective_camera",
                        lambda request, render: request)
    monkeypatch.setattr(controller, "package_from_catalog_entry",
                        lambda *a, **k: deepcopy(package))
    monkeypatch.setattr(controller, "room_route", lambda *a, **k: route)
    monkeypatch.setattr(controller, "planning_room_mapping", lambda pkg: {"room_id": "fixture_room"})
    monkeypatch.setattr(controller, "capture_adapter_binding",
                        lambda *a, **k: {"selected_scene": "fixture", "runtime": {}})
    monkeypatch.setattr(controller, "renderer_capture_entrypoint", lambda renderer: "fixture")
    monkeypatch.setattr(controller, "room_capability_report",
                        lambda *a, **k: {"dimensions": {}})
    monkeypatch.setattr(controller, "write_room_package_plan_snapshot", lambda *a, **k: None)
    # The controller imports the pool loader inside plan_request, so the patch
    # has to land on the sampler module the import resolves to.
    import avengine.rooms.conditioned_sampler as sampler
    monkeypatch.setattr(sampler, "load_conditioned_sound_pool",
                        lambda payload, source_path=None: fixture_sounds())

    import avengine.capture.qa_plan_adapters as adapters
    monkeypatch.setattr(adapters, "load_planning_resources",
                        lambda room, request: (space, mesh, deepcopy(layout)))

    def plan(branch, *, stated_profile=None, qa_id="QA-08", name=None):
        body = request_body(branch, stated_profile=stated_profile, qa_id=qa_id)
        body["room_catalog"] = str(catalog_path)
        body["sound_pool"] = str(sound_path)
        output = tmp_path / (name or f"out_{qa_id}_{branch}_{len(list(tmp_path.iterdir()))}")
        return controller.plan_request(body, output), output

    return SimpleNamespace(controller=controller, plan=plan, tmp_path=tmp_path)


def test_branch_change_changes_the_knobs_the_cli_plan_was_solved_with(cli):
    hidden, _ = cli.plan("out_of_view", name="hidden")
    shown, _ = cli.plan("visible_clear", name="shown")

    assert hidden["condition_profile"]["anchor_visibility"] == "off_screen"
    assert shown["condition_profile"]["anchor_visibility"] == "in_fov"
    # The knob has to be attributed to the question, not to a sampler default,
    # or a default that happened to match would pass this test forever.
    assert (hidden["condition_profile"]["knob_sources"]["anchor_visibility"]
            == "compiled_question_condition")
    assert hidden["question_condition_match"]["knob_application"] == "applied_to_condition_profile"
    assert (hidden["question_condition_match"]["sampler_profile"]["anchor_visibility"]
            == "off_screen")
    # Two different requested branches must not produce the same episode.
    assert json.dumps(hidden, sort_keys=True) != json.dumps(shown, sort_keys=True)


def test_condition_profile_snapshot_is_the_profile_the_plan_used(cli):
    for branch in ("out_of_view", "visible_clear"):
        plan, output = cli.plan(branch, name=f"snapshot_{branch}")
        snapshot = json.loads((output / "condition_profile.json").read_text(encoding="utf-8"))
        assert snapshot == plan["condition_profile"], branch
        # The value a reader would quote must be the branch's own value.
        expected = "off_screen" if branch == "out_of_view" else "in_fov"
        assert snapshot["anchor_visibility"] == expected


def test_a_stated_profile_that_contradicts_the_branch_is_refused_not_applied(cli):
    with pytest.raises(cli.controller.QAPlanningError):
        cli.plan("out_of_view", stated_profile={"anchor_visibility": "in_fov"},
                 name="contradiction")

    output = cli.tmp_path / "contradiction"
    result = json.loads((output / "planning_result.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["gap_category"] == "evidence_missing_or_unsampled"
    reasons = " ".join(str(row.get("reason")) for row in result["room_attempts"])
    assert "ConditionedRequestConflict" in reasons
    assert "anchor_visibility" in reasons
    # The refusal must not leave a plan or a profile snapshot behind, because
    # either one would read as a condition that was met.
    assert not (output / "plan").exists()
    assert not (output / "condition_profile.json").exists()
    assert result["condition_profile"] is None
    # The stated value that caused the refusal has to be recoverable.
    assert result["requested_profile"]["anchor_visibility"] == "in_fov"


def test_an_illegal_condition_value_is_refused_not_planned(cli):
    """An unknown knob value has to end as a recorded gap, never as a plan.

    A stated value is used rather than a compiled one because which compiled
    values are legal is the compiler owner's to change; what must not change is
    that the controller refuses an illegal condition instead of planning some
    other visibility and reporting the condition as applied.
    """
    with pytest.raises(cli.controller.QAPlanningError):
        cli.plan("out_of_view", stated_profile={"anchor_visibility": "behind_the_sofa"},
                 name="illegal_value")

    output = cli.tmp_path / "illegal_value"
    result = json.loads((output / "planning_result.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    reasons = " ".join(str(row.get("reason")) for row in result["room_attempts"])
    assert "anchor_visibility" in reasons
    assert not (output / "plan").exists()
    assert not (output / "condition_profile.json").exists()
    assert result["requested_profile"]["anchor_visibility"] == "behind_the_sofa"


def test_a_planning_failure_records_the_gap_instead_of_raising_a_name_error(cli):
    """Every room refused is a recordable gap, not a crash in the reporting line.

    Removing the pre-resolved profile left ``condition_profile`` referenced but
    unbound on this path, so a refused request died with NameError and wrote no
    planning_result at all. A gap that raises instead of being written drops out
    of the coverage denominator entirely.
    """
    with pytest.raises(cli.controller.QAPlanningError) as error:
        cli.plan("out_of_view", stated_profile={"anchor_visibility": "in_fov"},
                 name="gap_recorded")
    assert "no existing room could realize the request" in str(error.value)

    result = json.loads(
        (cli.tmp_path / "gap_recorded" / "planning_result.json").read_text(encoding="utf-8"))
    assert set(result) >= {"status", "condition_profile", "requested_profile",
                           "room_attempts", "gap_category"}
    assert result["room_attempts"], "a refusal has to name the room it refused"
