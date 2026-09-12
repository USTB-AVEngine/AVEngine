from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from avengine.dataset import binding_group_native as native


REPOSITORY = Path(__file__).resolve().parents[1]
RELATION_ROOT = Path(
    "/data/datasets/avengine_workspaces/multi_home_activity_20260905/root/"
    "binding_dataset_20260909_v2/relation_mp3d_group_v1"
)
V0_VARIANT = RELATION_ROOT / "variants/v0_a0"
V1_VARIANT = RELATION_ROOT / "variants/v1_a0"
QUERY = {
    "reference_time_s": 0,
    "appearance_values": ["blue", "green"],
    "window_s": [4, 6],
}
ASSETS_V0 = [
    "rocketbox_human_male_adult_01_top_blue_research_v1",
    "rocketbox_human_male_adult_01_top_green_research_v1",
    "rocketbox_human_male_adult_01_top_yellow_research_v1",
]
ASSETS_V1 = [ASSETS_V0[0], ASSETS_V0[2], ASSETS_V0[1]]


def _unit(
    unit_id: str,
    stage: str,
    unit_kind: str,
    *,
    depends: tuple[str, ...] = (),
    members: tuple[str, ...] = (),
    member_index: int | None = None,
    visual_unit_id: str | None = None,
) -> dict:
    row = {
        "unit_id": unit_id,
        "stage": stage,
        "unit_kind": unit_kind,
        "depends_on_units": list(depends),
        "member_request_ids": list(members),
    }
    if member_index is not None:
        row["member_index"] = member_index
    if visual_unit_id is not None:
        row["visual_unit_id"] = visual_unit_id
    return row


def _relation_group_spec() -> dict:
    units = [
        _unit("v0", "plan", "visual_plan", members=("m0", "m1")),
        _unit("v1", "plan", "visual_plan", members=("m2", "m3")),
        _unit(
            "v0_capture", "capture", "visual_capture",
            depends=("v0",), members=("m0", "m1"),
        ),
        _unit(
            "v1_capture", "capture", "visual_capture",
            depends=("v1",), members=("m2", "m3"),
        ),
        _unit(
            "v0_a0", "audio", "audio",
            depends=("v0_capture",), members=("m0",),
            member_index=0, visual_unit_id="v0_capture",
        ),
        _unit(
            "v0_a1", "audio", "audio",
            depends=("v0_capture",), members=("m1",),
            member_index=1, visual_unit_id="v0_capture",
        ),
        _unit(
            "v1_a0", "audio", "audio",
            depends=("v1_capture",), members=("m2",),
            member_index=2, visual_unit_id="v1_capture",
        ),
        _unit(
            "v1_a1", "audio", "audio",
            depends=("v1_capture",), members=("m3",),
            member_index=3, visual_unit_id="v1_capture",
        ),
        _unit(
            "group", "assembly", "assembly",
            depends=("v0_capture", "v1_capture", "v0_a0", "v0_a1",
                     "v1_a0", "v1_a1"),
            members=("m0", "m1", "m2", "m3"),
        ),
    ]
    return {
        "group_id": "relation_stage_probe",
        "task_family": "visual_conditioned_relation",
        "room_id": "habitat_mp3d_example_17DRP5sb8fy",
        "member_request_ids": ["m0", "m1", "m2", "m3"],
        "shared_audio_member_ids": [["m0", "m2"], ["m1", "m3"]],
        "recipe": {
            "task_family": "visual_conditioned_relation",
            "motion_timing": "none",
            "plan_equivalence": "controlled_slots",
            "visual_intervention": "source_slot_permutation",
            "query_identity_policy": "slot",
            "audio_content_scope": "shared_audio_pairs",
        },
        "stage_units": units,
    }


def _relation_context() -> dict:
    v0_request = json.loads(
        (RELATION_ROOT / "requests/v0_scheduled_request.json").read_text(
            encoding="utf-8"
        )
    )
    v1_request = json.loads(
        (RELATION_ROOT / "requests/v1_scheduled_request.json").read_text(
            encoding="utf-8"
        )
    )
    requests = {
        "m0": deepcopy(v0_request),
        "m1": deepcopy(v0_request),
        "m2": deepcopy(v1_request),
        "m3": deepcopy(v1_request),
    }
    return native.relation_group_stage_context(
        group_spec=_relation_group_spec(),
        member_requests=requests,
        retained_visual_roots={"v0": V0_VARIANT, "v1": V1_VARIANT},
        world_id="world_mp3d_relation_0001",
        relation_query=QUERY,
    )


def _item(
    context: dict,
    unit_id: str,
    stage: str,
    *,
    attempt: int = 1,
    inputs: dict | None = None,
) -> dict:
    spec = next(
        row for row in context["group_spec"]["stage_units"]
        if row["unit_id"] == unit_id
    )
    return {
        "work_item_id": (
            f"{context['group_id']}/{unit_id}:{stage}:{attempt:02d}"
        ),
        "request_id": f"{context['group_id']}/{unit_id}",
        "scope_id": f"{context['group_id']}/{unit_id}",
        "group_id": context["group_id"],
        "task_family": context["task_family"],
        "unit_id": unit_id,
        "stage": stage,
        "member_request_ids": list(spec.get("member_request_ids") or ()),
        "fresh_output_relative": (
            f"{context['group_id']}/{unit_id}/{stage}/attempt_{attempt:02d}"
        ),
        "depends_on": list(spec.get("depends_on_units") or ()),
        "inputs": inputs or {},
        "resource": {"kind": "cpu", "execution": "cpu"},
        "payload": {},
    }


def test_relation_context_preserves_explicit_query_and_requires_three_sources():
    context = _relation_context()
    assert context["relation_query"] == QUERY
    assert context["contract"]["world_population"] == sorted(ASSETS_V0)
    assert context["contract"]["audio_columns"] == {
        "a0": ["v0_a0", "v1_a0"],
        "a1": ["v0_a1", "v1_a1"],
    }
    with pytest.raises(
        native.BindingNativeError,
        match="exactly three distinct source assets",
    ):
        native._relation_source_order(
            {
                "contract": {
                    "visual_units": {
                        "v0_capture": {
                            "source_asset_ids": ["one", "two"]
                        }
                    }
                }
            },
            "v0_capture",
        )


def test_relation_scoped_dispatch_uses_only_relation_runner_map(monkeypatch, tmp_path):
    seen = {}

    def fake_dispatch(item, context, **kwargs):
        seen.update(kwargs)
        return {"status": "pass"}

    monkeypatch.setattr(native, "run_group_stage_work_item", fake_dispatch)
    result = native.run_relation_group_stage_work_item(
        {"unit_id": "v0"},
        {"group_id": "relation_stage_probe"},
        output_root=tmp_path,
    )
    assert result == {"status": "pass"}
    assert seen["stage_runners"] is native.RELATION_STAGE_RUNNERS
    assert set(seen["stage_runners"]) == {
        "visual_plan", "visual_capture", "audio", "assembly"
    }


def test_real_retained_relation_stages_reuse_media_schedule_and_assemble(
    monkeypatch, tmp_path: Path
):
    context = _relation_context()
    work_root = tmp_path / "relation_work"
    results: list[dict] = []
    stage_outputs: dict[str, dict] = {}

    def run(unit_id: str, stage: str, *, inputs: dict | None = None):
        item = _item(context, unit_id, stage, inputs=inputs)
        result = native.run_relation_group_stage_work_item(
            item, context, output_root=work_root, results=results
        )
        assert result["status"] == "pass", result
        results.append(result)
        stage_outputs[unit_id] = result
        return result

    v0_plan = run("v0", "plan")
    v1_plan = run("v1", "plan")
    v0_capture = run("v0_capture", "capture", inputs={"v0": v0_plan})
    v1_capture = run("v1_capture", "capture", inputs={"v1": v1_plan})

    finalize_calls: dict[str, str | None] = {}
    ancillary_declaration = [{
        "path": str(RELATION_ROOT / "variants/v0_a0/delivery/audio/audio/foa/mixture.wav"),
        "layout_id": "foa",
        "order": 1,
        "normalization": "N3D",
        "coordinate_frame": "world",
        "clock": {
            "sample_rate_hz": 16000,
            "sample_count": 160000,
            "frame_rate_hz": 15,
            "frame_count": 150,
        },
    }]

    def fake_finalize(
        root: str | Path,
        request: dict,
        *,
        audio_report: str | Path | None = None,
        shared_visual_root: str | Path | None = None,
    ) -> dict:
        del request, shared_visual_root
        member_id = Path(root).parents[2].name
        source = RELATION_ROOT / "variants" / member_id
        finalize_calls[member_id] = (
            None if audio_report is None else str(Path(audio_report).resolve())
        )
        return {
            "facts": str(source / "delivery/facts.json"),
            "audio_report": str(source / "delivery/research_report.json"),
            "audio": str(
                (
                    source
                    if (source / "delivery/audio/audio/binaural/mixture.wav").is_file()
                    else RELATION_ROOT / "variants" / f"v0_a{member_id.rsplit('_a', 1)[-1]}"
                )
                / "delivery/audio/audio/binaural/mixture.wav"
            ),
            "questions": str(source / "delivery/questions.json"),
            "visual_video": str(source / "delivery/visual_rgb.mp4"),
            "preview": str(source / "delivery/preview.mp4"),
            "declared_audio_delivery": {
                "primary_layout": "binaural",
                "attached_view_layouts": [],
                "foa_normalization": "native_n3d",
            },
            "delivered_audio_layouts": {
                "status": "pass",
                "undelivered_attached_view_layouts": [],
            },
            "result": {
                "ancillary_audio_outputs": deepcopy(ancillary_declaration)
            },
            "elapsed_s": 0.0,
        }

    monkeypatch.setattr(native, "finalize_audio_assignment", fake_finalize)
    audio_results = []
    for unit_id, visual_id in (
        ("v0_a0", "v0_capture"),
        ("v0_a1", "v0_capture"),
        ("v1_a0", "v1_capture"),
        ("v1_a1", "v1_capture"),
    ):
        visual_result = stage_outputs[visual_id]
        audio_result = run(
            unit_id,
            "audio",
            inputs={visual_id: visual_result},
        )
        audio_results.append(audio_result)

    assert stage_outputs["v0_capture"]["outputs"]["capture"]
    assert audio_results[0]["outputs"]["relation_schedule"] == (
        audio_results[2]["outputs"]["relation_schedule"]
    )
    assert audio_results[1]["outputs"]["relation_schedule"] == (
        audio_results[3]["outputs"]["relation_schedule"]
    )
    assert audio_results[2]["outputs"]["shared_audio_column"]["reused"] is True
    assert audio_results[3]["outputs"]["shared_audio_column"]["reused"] is True
    assert finalize_calls["v0_a0"] is None
    assert finalize_calls["v1_a0"] is not None
    assert audio_results[0]["outputs"]["ancillary_audio_outputs"] == (
        ancillary_declaration
    )
    assert audio_results[2]["outputs"]["ancillary_audio_outputs"] == (
        ancillary_declaration
    )

    assembly_inputs = {
        unit_id: result
        for unit_id, result in stage_outputs.items()
        if unit_id in {"v0_capture", "v1_capture"}
    }
    assembly_inputs.update({
        unit_id: result
        for unit_id, result in zip(
            ("v0_a0", "v0_a1", "v1_a0", "v1_a1"), audio_results
        )
    })
    blocked_item = _item(
        context, "group", "assembly", attempt=1, inputs=assembly_inputs
    )
    blocked = native.run_relation_group_stage_work_item(
        blocked_item, context, output_root=work_root, results=results
    )
    assert blocked["status"] == "fail"
    assert "no attached-view mapping" in blocked["reason"]
    stripped_inputs = deepcopy(assembly_inputs)
    for audio_unit_id in ("v0_a0", "v0_a1", "v1_a0", "v1_a1"):
        stripped_inputs[audio_unit_id]["outputs"].pop(
            "ancillary_audio_outputs", None
        )
    stripped_results = deepcopy(results)
    for row in stripped_results:
        if str(row.get("scope_id") or "").endswith(
            ("v0_a0", "v0_a1", "v1_a0", "v1_a1")
        ):
            row.setdefault("outputs", {}).pop("ancillary_audio_outputs", None)
    assembly_item = _item(
        context, "group", "assembly", attempt=2, inputs=stripped_inputs
    )
    assembly = native.run_relation_group_stage_work_item(
        assembly_item, context, output_root=work_root, results=stripped_results
    )
    assert assembly["status"] == "pass", assembly
    validation = assembly["facts"]["validation"]
    assert validation["status"] == "pass"
    assert validation["shared_audio_checks"]
    assert len(validation["shared_audio_checks"]) == 2
    assert len(validation["shared_rgb_checks"]) == 2
    assert len(validation["answer_relation_checks"]) == 4
    assert all(
        row["answer_relation"] == "different"
        and row["media_check"] == "pass"
        for row in validation["answer_relation_checks"]
    )
    assert validation["media_validation"] == "media_checked"
    spec = json.loads(
        Path(assembly["facts"]["group_spec_path"]).read_text(encoding="utf-8")
    )
    assert spec["groups"][0]["query"] == QUERY
    assert spec["groups"][0]["task_family"] == "visual_conditioned_relation"
    assert assembly["outputs"]["native_visual_worlds_created"] == 0


def test_relation_query_is_not_silently_defaulted():
    spec = _relation_group_spec()
    v0 = json.loads(
        (RELATION_ROOT / "requests/v0_scheduled_request.json").read_text(
            encoding="utf-8"
        )
    )
    v1 = json.loads(
        (RELATION_ROOT / "requests/v1_scheduled_request.json").read_text(
            encoding="utf-8"
        )
    )
    requests = {
        "m0": deepcopy(v0),
        "m1": deepcopy(v0),
        "m2": deepcopy(v1),
        "m3": deepcopy(v1),
    }
    with pytest.raises(
        native.BindingNativeError,
        match="explicit relation query",
    ):
        native.relation_group_stage_context(
            group_spec=spec,
            member_requests=requests,
            retained_visual_roots={"v0": V0_VARIANT, "v1": V1_VARIANT},
        )
