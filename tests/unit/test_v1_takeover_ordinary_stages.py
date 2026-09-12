"""T07 ordinary Episode stage-chain wiring tests."""
from __future__ import annotations

import json
from pathlib import Path

from avengine.dataset import production_runner as runner


def _runner(tmp_path: Path) -> runner.ProductionRunner:
    manifest = {
        "schema": "avengine_qa_batch_manifest_v1",
        "batch_id": "t07",
        "episodes": [{
            "episode_id": "ordinary_01",
            "room_id": "room_t07",
            "request": {
                "schema": "avengine_native_qa_room_request_v1",
                "episode_id": "ordinary_01",
                "room_id": "room_t07",
                "seed": 1,
            },
        }],
        "production": {"core_groups": []},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return runner.ProductionRunner(
        manifest=manifest,
        manifest_path=manifest_path,
        run_root=tmp_path / "run",
        repository=tmp_path,
    )


def test_ordinary_plan_capture_audio_delivery_executors_are_registered():
    entries = {
        kind: runner.resolve_stage_executor(None, kind)
        for kind in ("plan", "capture", "audio", "delivery")
    }
    assert set(entries) == {"plan", "capture", "audio", "delivery"}
    assert all(entry.source.endswith("production_runner.ordinary")
               for entry in entries.values())


def test_ordinary_context_uses_stage_names_for_dependencies(tmp_path):
    instance = _runner(tmp_path)
    scope = runner.ScopeState(
        scope_kind="episode",
        scope_key="ordinary_01",
        room_id="room_t07",
        row=instance.rows_by_episode_id["ordinary_01"],
    )
    scope.results = [{
        "work_item_id": "ordinary_01:plan:01",
        "stage": "plan",
        "request_id": "ordinary_01",
        "scope_id": "ordinary_01",
        "status": "pass",
        "facts": {"episode_plan_path": "/tmp/plan"},
        "outputs": {},
        "depends_on": [],
    }]
    item = {
        "work_item_id": "ordinary_01:capture:01",
        "stage": "capture",
        "task_family": "visible_binding",
        "request_id": "ordinary_01",
        "attempt": 1,
        "depends_on": ["ordinary_01:plan:01"],
        "fresh_output_relative": "ordinary_01/capture/attempt_01",
        "resource": {
            "kind": "gpu_native_visual",
            "execution": "gpu",
            "runtime_context": "renderer_native",
        },
        "payload": {},
        "inputs": {},
    }
    context = instance._context_for(scope, item, None)
    assert context.member_request_ids == ("ordinary_01",)
    assert context.task_family is None
    assert set(context.upstream) == {"plan"}


def test_summary_marks_delivery_as_delivered_episode(tmp_path):
    instance = _runner(tmp_path)
    scope = instance.scopes[0]
    scope.results = [{
        "work_item_id": "ordinary_01:delivery:01",
        "stage": "delivery",
        "request_id": "ordinary_01",
        "scope_id": "ordinary_01",
        "status": "pass",
        "facts": {"facts_path": "/tmp/facts.json", "questions_path": "/tmp/questions.json"},
        "outputs": {
            "facts_path": "/tmp/facts.json",
            "questions_path": "/tmp/questions.json",
            "preview_path": "/tmp/preview.mp4",
            "audio_path": "/tmp/audio.wav",
            "episode_root": "/tmp/episode",
            "world_id": "ordinary_01",
            "room_family": "apartment",
            "room_id": "room_t07",
        },
        "depends_on": [],
    }]
    summary = instance._summary(1)
    assert summary["status"] == "complete"
    assert summary["delivered_groups"] == []
    assert summary["delivered_episodes"] == [{
        "episode_id": "ordinary_01",
        "facts_path": "/tmp/facts.json",
        "questions_path": "/tmp/questions.json",
        "video_path": "/tmp/preview.mp4",
        "audio_path": "/tmp/audio.wav",
        "room_family": "apartment",
        "room_id": "room_t07",
        "world_id": "ordinary_01",
        "episode_root": "/tmp/episode",
        "capture_stage_evidence": [],
        "audio_stage_evidence": [],
    }]


def test_export_ordinary_uses_core_free_catalog_path(monkeypatch, tmp_path):
    calls = {}

    def fake_catalog(specs, *, output, qa_sampling, items_per_type, seed):
        calls["specs"] = list(specs)
        calls["catalog_output"] = Path(output)
        Path(output).mkdir(parents=True)
        (Path(output) / "catalog_index.json").write_text("{}", encoding="utf-8")
        return {
            "av_sample_count": 1,
            "group_count": 0,
            "world_count": 1,
            "catalog_question_count": 25,
            "core_question_count": 0,
            "generated_by_qa": {},
        }

    def fake_export(core_bundle, catalog_index, output):
        calls["core_bundle"] = core_bundle
        calls["catalog_index"] = Path(catalog_index)
        calls["output"] = Path(output)
        return {"status": "exported", "delivery_kind": "episode_catalog"}

    monkeypatch.setattr(
        "avengine.qa.binding_catalog.derive_episode_catalog", fake_catalog
    )
    monkeypatch.setattr(
        "avengine.qa.binding_delivery.export_binding_delivery", fake_export
    )
    monkeypatch.setattr(
        "avengine.qa.binding_delivery.build_dataset_index",
        lambda output: {"record_kind_counts": {"episode": 1}},
    )
    result = runner.export_run_delivery(
        delivered_episodes=[{
            "episode_id": "ordinary_01",
            "facts_path": "/tmp/facts.json",
            "questions_path": "/tmp/questions.json",
            "video_path": "/tmp/preview.mp4",
            "audio_path": "/tmp/audio.wav",
            "room_family": "apartment",
            "room_id": "room_t07",
            "world_id": "ordinary_01",
        }],
        output=tmp_path / "delivery",
    )
    assert result["status"] == "exported"
    assert result["delivery_kind"] == "episode_catalog"
    assert calls["core_bundle"] is None
    assert calls["specs"][0]["media"] == {
        "video_path": "/tmp/preview.mp4",
        "audio_path": "/tmp/audio.wav",
    }


def test_mixed_core_and_ordinary_delivery_uses_m09_merge_and_existing_join(
        monkeypatch, tmp_path):
    calls = {}
    bundle = tmp_path / "core_bundle.json"
    bundle.write_text("{}")
    monkeypatch.setattr(runner, "_bundle_paths", lambda _groups: [bundle])
    monkeypatch.setattr(
        runner, "core_bundle_for",
        lambda _bundles, output: bundle,
    )

    def fake_core(paths, *, output, qa_sampling, items_per_type, seed):
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / "catalog_index.json").write_text("{}")
        calls["core_paths"] = list(paths)
        return {"av_sample_count": 1, "group_count": 1, "world_count": 1}

    def fake_episode(specs, *, output, qa_sampling, items_per_type, seed):
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / "catalog_index.json").write_text("{}")
        calls["episode_specs"] = list(specs)
        return {"av_sample_count": 1, "group_count": 0, "world_count": 1}

    def fake_merge(paths, *, output, seed):
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / "catalog_index.json").write_text("{}")
        calls["merge_paths"] = [str(path) for path in paths]
        return {"av_sample_count": 2, "group_count": 1, "world_count": 2}

    def fake_export(core_bundle, catalog_index, output):
        calls["joined"] = {
            "core_bundle": str(core_bundle),
            "catalog_index": str(catalog_index),
        }
        return {"status": "exported", "delivery_kind": "mixed"}

    monkeypatch.setattr(
        "avengine.qa.binding_catalog.derive_binding_catalog", fake_core
    )
    monkeypatch.setattr(
        "avengine.qa.binding_catalog.derive_episode_catalog", fake_episode
    )
    monkeypatch.setattr(
        "avengine.qa.binding_catalog.merge_binding_catalogs", fake_merge
    )
    monkeypatch.setattr(
        "avengine.qa.binding_delivery.export_binding_delivery", fake_export
    )
    monkeypatch.setattr(
        runner, "_attach_exported_audio_layouts", lambda *args: []
    )
    monkeypatch.setattr(
        "avengine.qa.binding_delivery.build_dataset_index",
        lambda output: {"record_kind_counts": {"core": 1, "episode": 1}},
    )
    result = runner.export_run_delivery(
        delivered_groups=[{"group_id": "g01"}],
        delivered_episodes=[{
            "episode_id": "ordinary_01",
            "video_path": "/tmp/video.mp4",
            "audio_path": "/tmp/audio.wav",
        }],
        output=tmp_path / "delivery",
    )
    assert result["status"] == "exported"
    assert result["delivery_kind"] == "mixed_merged_catalog"
    assert len(calls["merge_paths"]) == 2
    assert calls["joined"]["catalog_index"].endswith("merged/catalog_index.json")
def test_ordinary_delivery_manifest_uses_request_assets_without_core_assignments(tmp_path):
    row = {
        "episode_id": "ordinary_01",
        "room_id": "room_t07",
        "source_assignments": [],
        "request": {
            "source_asset_ids": ["asset_blue", "asset_green"],
        },
    }
    normalized = runner._ordinary_delivery_manifest_entry(row, tmp_path)
    assert normalized["source_assignments"] == [
        {"actor_id": "source1", "asset_id": "asset_blue"},
        {"actor_id": "source2", "asset_id": "asset_green"},
    ]
    assert row["source_assignments"] == []
def test_ordinary_delivery_manifest_fills_profile_from_written_plan(tmp_path):
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    (plan_dir / "episode_plan.json").write_text(json.dumps({
        "condition_profile": {
            "total_count": 2,
            "anchor_indices": [0],
            "separation_floor_deg": 15,
            "separation_bin_deg": [30.0, 60.0],
        },
    }), encoding="utf-8")
    row = {
        "episode_id": "ordinary_01",
        "requested_profile": {
            "anchor_count": 1,
            "separation_floor_deg": 15,
        },
        "source_assignments": [],
    }
    normalized = runner._ordinary_delivery_manifest_entry(row, tmp_path)
    assert normalized["requested_profile"]["total_count"] == 2
    assert normalized["requested_profile"]["anchor_indices"] == [0]
    assert normalized["requested_profile"]["separation_floor_deg"] == 15
def test_audio_layout_readback_checks_binaural_and_foa_files(tmp_path):
    import numpy as np
    import soundfile as sf

    report_path = tmp_path / "research_report.json"
    binaural = tmp_path / "binaural.wav"
    foa = tmp_path / "foa.wav"
    sf.write(binaural, np.zeros((16000, 2), dtype=np.float32), 16000)
    sf.write(foa, np.zeros((16000, 4), dtype=np.float32), 16000)
    report_path.write_text(json.dumps({
        "clock": {
            "frame_count": 150,
            "frame_rate_hz": 15,
            "sample_rate_hz": 16000,
            "sample_count": 16000,
            "time_base_hz": 48000,
        },
        "audio": {
            "post_assembly_convolution_gain": 0.5,
            "layout_delivery": {
                "binaural": {
                    "channel_count": 2,
                    "channel_labels": ["left", "right"],
                    "mixture": {"path": str(binaural)},
                },
                "ambisonics": {
                    "channel_count": 4,
                    "channel_labels": ["W", "Y", "Z", "X"],
                    "channel_order": "ACN",
                    "coordinate_frame": "avengine_world",
                    "foa_normalization": {
                        "delivered_normalization": "N3D",
                        "native_normalization": "N3D",
                        "conversion": "identity_native_encode",
                    },
                    "mixture": {"path": str(foa)},
                },
            },
        },
    }), encoding="utf-8")
    readback = runner._ordinary_audio_layout_readback(
        {
            "audio_report": str(report_path),
            "declared_audio_delivery": {
                "layouts": [
                    {"type": "binaural", "channel_count": 2},
                    {"type": "ambisonics", "channel_count": 4},
                ],
            },
        },
        {
            "time": {
                "frame_count": 150,
                "frame_rate_hz": 15,
                "sample_rate_hz": 16000,
                "sample_count": 16000,
                "time_base_hz": 48000,
            },
            "audio": {
                "wet_tail_intervals": [
                    {"event_id": "event_001", "start_s": 0.5, "end_s": 1.0}
                ],
                "source_activity_intervals_samples": {
                    "event_001": [{"start_sample": 8000, "end_sample_exclusive": 12000}]
                },
                "source_activity_coordinate_space": "episode_sample_clock",
            },
        },
    )
    assert readback["status"] == "pass"
    assert [row["channel_count"] for row in readback["layouts"]] == [2, 4]
def test_retained_audio_world_id_is_accounted_without_visual_charge(tmp_path):
    instance = _runner(tmp_path)
    scope = instance.scopes[0]
    scope.row = {**scope.row, "request": {"world_id": "world_t07"}}
    item = {
        "work_item_id": "ordinary_01:audio:01",
        "stage": "audio",
        "attempt": 1,
        "request_id": "ordinary_01",
        "payload": {"unit_kind": "audio"},
        "resource": {"runtime_context": "rlr_native"},
        "inputs": {},
    }
    descriptor = instance._native_attempt_descriptor(
        scope, item, "/retained/episode"
    )
    assert descriptor["logical_world_id"] == "world_t07"
    assert descriptor["native_visual_worlds"] == 0
    assert descriptor["capture_instances"] == 0
    assert descriptor["native_acoustic_launch_attempts"] == 1



def test_audio_report_flows_stage_summary_to_core_and_ordinary_export(
        monkeypatch, tmp_path):
    report_core = tmp_path / "core" / "research_report.json"
    report_episode = tmp_path / "episode" / "research_report.json"
    mixture_core = tmp_path / "core" / "foa.wav"
    mixture_episode = tmp_path / "episode" / "foa.wav"
    for report, mixture in ((report_core, mixture_core), (report_episode, mixture_episode)):
        mixture.parent.mkdir(parents=True, exist_ok=True)
        mixture.write_bytes(b"real-wav-placeholder")
        report.write_text(json.dumps({
            "audio": {
                "layout_delivery": {
                    "ambisonics": {
                        "mixture": {"path": str(mixture)},
                        "channel_count": 4,
                    }
                }
            }
        }), encoding="utf-8")

    declared = {"attached_view_layouts": ["ambisonics"]}
    core_facts = tmp_path / "core" / "facts.json"
    episode_facts = tmp_path / "episode" / "facts.json"
    core_facts.write_text(json.dumps({
        "source_paths": {"research_report": str(report_core)}
    }), encoding="utf-8")
    episode_facts.write_text(json.dumps({
        "source_paths": {"research_report": str(report_episode)}
    }), encoding="utf-8")

    core_scope = runner.ScopeState(
        scope_kind="core_group",
        scope_key="group_01",
        room_id="room_t07",
        task_family="visible_binding",
    )
    core_audio = {
        "work_item_id": "group_01/v0_a0:audio:01",
        "stage": "audio",
        "request_id": "group_01/v0_a0",
        "scope_id": "group_01/v0_a0",
        "status": "pass",
        "facts": {
            "facts_path": str(core_facts),
            "audio_report_path": str(report_core),
            "declared_audio_delivery": declared,
            "ancillary_audio_outputs": [{"layout": "ambisonics"}],
        },
        "outputs": {
            "member_request_id": "group_01_v0_a0",
            "audio": str(mixture_core),
        },
        "depends_on": [],
    }
    core_scope.results = [
        core_audio,
        {
            "work_item_id": "group_01/assembly:assembly:01",
            "stage": "assembly",
            "request_id": "group_01",
            "scope_id": "group_01/assembly",
            "status": "pass",
            "facts": {"assembled_path": str(tmp_path / "assembled")},
            "outputs": {},
            "depends_on": [],
        },
    ]
    core_instance_root = tmp_path / "core_instance"
    core_instance_root.mkdir(parents=True)
    core_instance = _runner(core_instance_root)
    core_instance.scopes = [core_scope]
    summary = core_instance._summary(1)
    core_evidence = summary["delivered_groups"][0]["audio_stage_evidence"]
    assert core_evidence[0]["audio_report_path"] == str(report_core)
    assert core_evidence[0]["ancillary_audio_outputs"] == [
        {"layout": "ambisonics"}
    ]

    episode_scope = runner.ScopeState(
        scope_kind="episode",
        scope_key="ordinary_01",
        room_id="room_t07",
        row={"episode_id": "ordinary_01"},
    )
    episode_audio = {
        **core_audio,
        "work_item_id": "ordinary_01:audio:01",
        "request_id": "ordinary_01",
        "scope_id": "ordinary_01",
        "facts": {
            "facts_path": str(episode_facts),
            "audio_report_path": str(report_episode),
            "declared_audio_delivery": declared,
        },
        "outputs": {"audio": str(mixture_episode)},
    }
    episode_scope.results = [episode_audio]
    episode_evidence = runner.ProductionRunner._audio_stage_evidence(episode_scope)
    assert episode_evidence[0]["episode_id"] == "ordinary_01"
    assert episode_evidence[0]["audio_report_path"] == str(report_episode)

    captured = []

    def fake_attach(destination, *, sample_id, layout, receipt, mixture):
        captured.append({
            "destination": str(destination),
            "sample_id": sample_id,
            "layout": layout,
            "receipt": str(receipt),
            "mixture": mixture,
        })
        return {"sample_id": sample_id, "layout": layout, "receipt": str(receipt)}

    monkeypatch.setattr(
        "avengine.qa.binding_delivery.attach_audio_layout", fake_attach
    )
    catalog = {
        "records": [
            {
                "sample_id": "sample_core",
                "member_id": "group_01_v0_a0",
                "facts_path": str(core_facts),
            },
            {
                "sample_id": "sample_episode",
                "episode_id": "ordinary_01",
                "facts_path": str(episode_facts),
            },
        ]
    }
    attachments = runner._attach_exported_audio_layouts(
        tmp_path / "delivery",
        catalog,
        summary["delivered_groups"],
        [{"audio_stage_evidence": episode_evidence}],
    )
    assert {row["sample_id"] for row in captured} == {
        "sample_core", "sample_episode"
    }
    assert {row["receipt"] for row in captured} == {
        str(report_core.resolve()), str(report_episode.resolve())
    }
    assert len(attachments) == 2


def test_core_audio_evidence_uses_group_and_facts_identity_for_duplicate_members(
        monkeypatch, tmp_path):
    evidence = []
    records = []
    for group_id in ("core_g01", "core_g02"):
        root = tmp_path / group_id
        root.mkdir()
        mixture = root / "foa.wav"
        mixture.write_bytes((group_id + "-foa").encode())
        report = root / "research_report.json"
        report.write_text(json.dumps({
            "audio": {
                "layout_delivery": {
                    "ambisonics": {
                        "mixture": {"path": str(mixture)},
                        "channel_count": 4,
                        "channel_labels": ["W", "Y", "Z", "X"],
                        "channel_order": "ACN",
                        "normalization": "N3D",
                        "coordinate_frame": "avengine_world",
                        "layout_id": "rlr_foa_acn_n3d_world_v1",
                    }
                }
            }
        }), encoding="utf-8")
        facts = root / "facts.json"
        facts.write_text(json.dumps({
            "source_paths": {"research_report": str(report)}
        }), encoding="utf-8")
        evidence.append({
            "group_id": group_id,
            "member_id": "v0_a0",
            "facts_path": str(facts),
            "audio_report_path": str(report),
            "declared_audio_delivery": {
                "attached_view_layouts": ["ambisonics"]
            },
            "ancillary_audio_outputs": [{"layout": "ambisonics"}],
        })
        records.append({
            "sample_id": f"sample_{group_id}",
            "group_id": group_id,
            "member_id": "v0_a0",
            "facts_path": str(facts),
        })
    captured = []

    def fake_attach(destination, *, sample_id, layout, receipt, mixture):
        captured.append((sample_id, layout, str(receipt), str(mixture)))
        return {"sample_id": sample_id, "layout": layout}

    monkeypatch.setattr(
        "avengine.qa.binding_delivery.attach_audio_layout", fake_attach
    )
    attached = runner._attach_exported_audio_layouts(
        tmp_path / "delivery",
        {"records": records},
        [{"audio_stage_evidence": evidence}],
        [],
    )
    assert {row[0] for row in captured} == {"sample_core_g01", "sample_core_g02"}
    by_sample = {row[0]: row[2] for row in captured}
    assert by_sample["sample_core_g01"].endswith("core_g01/research_report.json")
    assert by_sample["sample_core_g02"].endswith("core_g02/research_report.json")
    assert len(attached) == 2
