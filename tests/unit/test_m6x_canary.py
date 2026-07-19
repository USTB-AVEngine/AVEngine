from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import pytest

import avengine.m6x.canary as canary_module
from avengine.contracts.json_io import canonical_json_sha256, sha256_file
from avengine.m4.runtime import M4SimulationConfig
from avengine.m5.acoustics import DynamicRIRSequence
from avengine.m5_1.acoustics import (
    build_strided_review_keyframes,
    research_review_trajectory_record,
)
from avengine.m6.audio_program import materialize_audio_program_variant
from avengine.m6x.canary import (
    CaptureData,
    M6XCanaryError,
    _actual_source_paths,
    _fixed_acoustic_identity,
    _float32_stems_and_exact_mixture,
    _load_capture,
    _load_retained_master_sequence,
    _master_root_paths,
    _provisional_source_paths,
    _scenario_grid_and_sequence,
    _timeline,
    _validate_capture_reuse_contract,
    _validated_inputs,
    _write_review_index,
    _write_scenario_rir_evidence,
)


ROOT = Path(__file__).resolve().parents[2]


def _capture_reuse_fixture(
    tmp_path: Path,
) -> tuple[dict, list[dict], Path, Path]:
    room_manifest = tmp_path / "room_manifest.json"
    room_manifest.write_text(
        json.dumps({"room_id": "legacy_ue_apartment_0000_v1"}),
        encoding="utf-8",
    )
    request = tmp_path / "m1_capture_request.json"
    request.write_text(
        json.dumps(
            {
                "room_id": "legacy_ue_apartment_0000_v1",
                "primary_camera_rig": {
                    "world_from_rig": {
                        "translation_m": [-0.7, 1.471, 0.65],
                        "rotation_xyzw": [
                            0,
                            0.4617486132350339,
                            0,
                            0.8870108331782217,
                        ],
                    },
                    "shared_calibration": {"hfov_degrees": 105},
                },
            }
        ),
        encoding="utf-8",
    )
    evidence = {
        "schema": "avengine_m5_1_human_beagle_capture_v1",
        "status": "pass",
        "frame_count": 270,
        "frame_rate_hz": 15,
        "time_base_hz": 48_000,
        "anchor_order": [
            "human0.head",
            "human0.mouth_emitter",
            "dog0.mouth_emitter",
        ],
        "camera": {
            "position_m": [-0.7, 1.471, 0.65],
            "rotation_xyzw": [
                0,
                0.4617486132350339,
                0,
                0.8870108331782217,
            ],
            "horizontal_fov_deg": 105,
            "legacy_camera_contract_required": True,
        },
        "inputs": {
            "room_manifest": {"sha256": sha256_file(room_manifest)},
            "m1_request": {"sha256": sha256_file(request)},
        },
    }
    records = [
        {"frame_index": index, "pts_ticks": index * 3_200} for index in range(270)
    ]
    return evidence, records, room_manifest, request


def _inputs() -> dict:
    return _validated_inputs(
        config_root=ROOT / "examples/m6x/fixed_apartment",
        room_registry_path=ROOT / "examples/m6/rooms/room_registry.json",
        entity_registry_path=ROOT / "examples/m6/registries/entity_assets_v1.json",
        endpoint_registry_path=ROOT / "examples/m6/registries/source_endpoints_v1.json",
        sound_registry_path=ROOT / "examples/m6/registries/sound_assets_v1.json",
    )


def _acoustic_test_values(tmp_path: Path) -> tuple[SimpleNamespace, object, Path, dict]:
    values = _inputs()
    scene = SimpleNamespace(
        manifest={
            "package_mode": "research_candidate",
            "source_room": {"room_id": "legacy_ue_apartment_0000_v1"},
            "geometry": {"representation": "real_surface_mesh"},
            "materials": {
                "material_semantics": "research_placeholder",
                "qualification_claim": "unqualified_research_placeholder",
            },
        },
        package_id="legacy_ue_apartment_test_package",
        qa_reports={},
    )
    simulation = M4SimulationConfig.from_mapping(
        json.loads(
            (
                ROOT / "examples/m4/blender_custom/multi_source_canary_request.json"
            ).read_text(encoding="utf-8")
        )["simulation"]
    )
    hrtf = tmp_path / "test.sofa"
    hrtf.write_bytes(b"test-hrtf")
    identity = _fixed_acoustic_identity(
        scene,
        room_capsule=values["room_capsule"],
        room_registry=values["room_registry"],
    )
    return scene, simulation, hrtf, identity


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    (
        ("frame_rate_hz", 30, "frame_rate_hz must be 15"),
        ("time_base_hz", 16_000, "time_base_hz must be 48000"),
        (
            "anchor_order",
            [
                "human0.head",
                "dog0.mouth_emitter",
                "human0.mouth_emitter",
            ],
            "anchor_order must be",
        ),
    ),
)
def test_capture_reuse_rejects_timing_or_anchor_contract_changes(
    tmp_path: Path, field: str, invalid_value: object, message: str
) -> None:
    evidence, records, room_manifest, request = _capture_reuse_fixture(tmp_path)
    evidence[field] = invalid_value
    capture = tmp_path / "capture"
    arrays = capture / "arrays"
    arrays.mkdir(parents=True)
    for name in (
        "rgb.npy",
        "semantic.npy",
        "actor_world_matrices.npy",
        "anchor_positions_m.npy",
    ):
        (arrays / name).touch()
    (capture / "frame_readback.json").write_text(json.dumps(records), encoding="utf-8")
    (capture / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(M6XCanaryError, match=message):
        _load_capture(
            capture,
            room_manifest_path=room_manifest,
            m1_request_path=request,
        )


def test_capture_reuse_rejects_different_apartment_or_camera_input(
    tmp_path: Path,
) -> None:
    evidence, records, room_manifest, request = _capture_reuse_fixture(tmp_path)
    evidence["inputs"]["room_manifest"]["sha256"] = "0" * 64
    evidence["camera"]["horizontal_fov_deg"] = 90

    with pytest.raises(M6XCanaryError) as raised:
        _validate_capture_reuse_contract(
            evidence,
            records,
            room_manifest_path=room_manifest,
            m1_request_path=request,
        )
    assert "inputs.room_manifest differs" in str(raised.value)
    assert "camera differs" in str(raised.value)


def test_capture_reuse_rejects_frame_ticks_outside_declared_time_base(
    tmp_path: Path,
) -> None:
    evidence, records, room_manifest, request = _capture_reuse_fixture(tmp_path)
    records[17]["pts_ticks"] += 1

    with pytest.raises(M6XCanaryError, match="frame_index/pts_ticks"):
        _validate_capture_reuse_contract(
            evidence,
            records,
            room_manifest_path=room_manifest,
            m1_request_path=request,
        )


def test_actual_source_paths_use_named_capture_anchor_evidence() -> None:
    values = _inputs()
    anchor_positions = np.zeros((270, 3, 3), dtype=np.float64)
    anchor_positions[:, 0, :] = (7.0, 8.0, 9.0)
    anchor_positions[:, 1, :] = (1.0, 2.0, 3.0)
    anchor_positions[:, 2, :] = (4.0, 5.0, 6.0)
    capture = CaptureData(
        root=ROOT,
        rgb=np.empty(0),
        semantic=np.empty(0),
        actor_world_matrices=np.empty(0),
        anchor_positions_m=anchor_positions,
        records=(),
        evidence={
            "anchor_order": [
                "dog0.mouth_emitter",
                "human0.head",
                "human0.mouth_emitter",
            ]
        },
    )

    paths = _actual_source_paths(values["anchors"], capture)

    assert np.array_equal(paths["m6x_dog0_muzzle"], anchor_positions[:, 0, :])
    assert np.array_equal(paths["m6x_human0_mouth"], anchor_positions[:, 2, :])


def test_master_routes_have_exact_authored_holds_and_motion() -> None:
    values = _inputs()
    human, dog = _master_root_paths(values["trajectories"], values["anchors"])
    assert human.shape == dog.shape == (270, 3)
    assert np.allclose(human[:76], human[0], atol=1.0e-15, rtol=0.0)
    assert np.allclose(human[194], human[-1], atol=1.0e-15, rtol=0.0)
    assert np.allclose(human[194:], human[-1], atol=1.0e-15, rtol=0.0)
    assert np.allclose(dog[:196], dog[0], atol=1.0e-15, rtol=0.0)
    assert np.allclose(dog[269], dog[-1], atol=1.0e-15, rtol=0.0)
    assert np.linalg.norm(human[-1] - human[0]) > 0.9
    assert np.linalg.norm(dog[-1] - dog[0]) > 0.5


def test_every_scenario_materializes_a_schema_valid_timeline() -> None:
    values = _inputs()
    human, dog = _master_root_paths(values["trajectories"], values["anchors"])
    paths = dict(
        sorted(_provisional_source_paths(values["anchors"], human, dog).items())
    )
    actor = np.repeat(np.eye(4)[None, None, :, :], 270 * 2, axis=0).reshape(
        270, 2, 4, 4
    )
    actor[:, 0, :3, 3] = human
    actor[:, 1, :3, 3] = dog
    digest = "0" * 64
    capture = CaptureData(
        root=ROOT,
        rgb=np.zeros((270, 240, 320, 3), dtype=np.uint8),
        semantic=np.zeros((270, 240, 320), dtype=np.int32),
        actor_world_matrices=actor,
        anchor_positions_m=np.zeros((270, 3, 3), dtype=np.float64),
        records=tuple(
            {
                "human": {"pose_sha256": digest},
                "beagle": {"readback": {"state_sha256": digest}},
            }
            for _ in range(270)
        ),
        evidence={"status": "pass"},
    )
    listener = (-0.7, 1.471, 0.65)
    orientation = (0.8870108331782217, 0.0, 0.4617486132350339, 0.0)
    master_grid = build_strided_review_keyframes(
        paths,
        visual_frame_rate_hz=15,
        rir_stride_frames=3,
        listener_position_m=listener,
        listener_orientation_wxyz=orientation,
    )
    samples = np.zeros((90, 6, 2, 8), dtype=np.float32)
    lengths = np.full((90, 6), 8, dtype=np.uint32)
    master_sequence = DynamicRIRSequence(
        samples=samples,
        lengths=lengths,
        source_ids=master_grid.source_ids,
        keyframe_ticks=tuple(item.tick for item in master_grid.keyframes),
        keyframe_samples=tuple(item.sample_index for item in master_grid.keyframes),
        sample_rate_hz=16_000,
        layout_type="binaural",
        layout_id="rlr_binaural_lr_v1",
        channel_labels=("left", "right"),
        trajectory_sha256=canonical_json_sha256(
            research_review_trajectory_record(master_grid)
        ),
        metadata={},
    )

    observed = 0
    for scenario in values["suite"]["scenarios"]:
        window = scenario["capture_frame_window"]
        candidate_ids = tuple(
            item["source_endpoint_id"] for item in scenario["source_bindings"]
        )
        grid, sequence, trajectories = _scenario_grid_and_sequence(
            master_grid,
            master_sequence,
            source_paths=paths,
            candidate_source_ids=candidate_ids,
            start_frame=window["start_frame"],
            end_frame_exclusive=window["end_frame_exclusive"],
            listener_position_m=listener,
            listener_orientation=orientation,
        )
        assert grid.visual_frame_count == 75
        assert sequence.samples.shape == (25, 2, 2, 8)
        reference = scenario["audio_program_ref"]
        base = values["programs"][(reference["program_id"], reference["revision"])]
        programs = [
            materialize_audio_program_variant(
                base,
                variant,
                source_endpoint_registry=values["endpoints"],
                sound_asset_registry=values["sounds"],
            )
            for variant in scenario["audio_variants"]
        ]
        if scenario["scenario_id"] == "S2":
            silent = scenario["silent_negative_program_ref"]
            programs.append(
                values["programs"][(silent["program_id"], silent["revision"])]
            )
        for program in programs:
            timeline = _timeline(
                capture=capture,
                window_start=window["start_frame"],
                program=program,
                trajectories=trajectories,
                endpoints=values["endpoints"],
                sounds=values["sounds"],
                listener_record={
                    "position_m": list(listener),
                    "orientation_wxyz": list(orientation),
                },
            )
            assert timeline["video"]["frame_count"] == 75
            assert timeline["audio"]["channel_count"] == 2
            observed += 1
    assert observed == 8


def test_retained_master_sequence_requires_the_same_trajectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _inputs()
    human, dog = _master_root_paths(values["trajectories"], values["anchors"])
    paths = dict(
        sorted(_provisional_source_paths(values["anchors"], human, dog).items())
    )
    listener = (-0.7, 1.471, 0.65)
    orientation = (0.8870108331782217, 0.0, 0.4617486132350339, 0.0)
    grid = build_strided_review_keyframes(
        paths,
        visual_frame_rate_hz=15,
        rir_stride_frames=3,
        listener_position_m=listener,
        listener_orientation_wxyz=orientation,
    )
    samples = np.zeros((90, 6, 2, 8), dtype=np.float32)
    lengths = np.full((90, 6), 8, dtype=np.uint32)
    np.save(tmp_path / "samples.npy", samples, allow_pickle=False)
    np.save(tmp_path / "lengths.npy", lengths, allow_pickle=False)
    trajectory = research_review_trajectory_record(grid)
    scene, simulation, hrtf, identity = _acoustic_test_values(tmp_path)
    configuration = simulation.to_dict()
    configuration.pop("channel_layout")
    configuration.pop("speed_of_sound_m_s")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "source_ids": list(grid.source_ids),
                "trajectory": trajectory,
                "trajectory_sha256": canonical_json_sha256(trajectory),
                "sample_rate_hz": 16_000,
                "layout_type": "binaural",
                "layout_id": "rlr_binaural_lr_v1",
                "channel_labels": ["left", "right"],
                "upload_report": {},
                "scene_claim_boundary": {
                    "package_mode": "research_candidate",
                    "material_semantics": "research_placeholder",
                    "material_qualification_claim": (
                        "unqualified_research_placeholder"
                    ),
                    "qa_status_by_report": {},
                },
                "runtime": {"configuration_readback": configuration},
                "hrtf": {"sha256": sha256_file(hrtf)},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(canary_module, "_verify_upload_report", lambda *_: None)

    retained = _load_retained_master_sequence(
        tmp_path,
        grid=grid,
        scene=scene,
        simulation=simulation,
        hrtf_file_path=hrtf,
        acoustic_identity=identity,
    )

    assert retained.samples.shape == (90, 6, 2, 8)
    assert retained.source_ids == grid.source_ids
    assert retained.metadata["m6x_acoustic_identity"] == identity

    rir_metadata = _write_scenario_rir_evidence(
        tmp_path / "scenario",
        scenario_id="S4",
        grid=grid,
        sequence=retained,
    )
    record = json.loads(rir_metadata.read_text(encoding="utf-8"))
    assert record["status"] == "pass"
    assert record["samples_shape"] == [90, 6, 2, 8]
    assert (rir_metadata.parent / "samples.npy").is_file()
    assert (rir_metadata.parent / "lengths.npy").is_file()
    assert (rir_metadata.parent / "trajectory.json").is_file()
    assert record["sequence_metadata"]["m6x_acoustic_identity"] == identity

    def reject_upload(*_args) -> None:
        raise ValueError("different world geometry")

    monkeypatch.setattr(canary_module, "_verify_upload_report", reject_upload)
    with pytest.raises(M6XCanaryError, match="geometry/material receipt differs"):
        _load_retained_master_sequence(
            tmp_path,
            grid=grid,
            scene=scene,
            simulation=simulation,
            hrtf_file_path=hrtf,
            acoustic_identity=identity,
        )


def test_acoustic_scene_must_belong_to_fixed_room(tmp_path: Path) -> None:
    scene, _simulation, _hrtf, _identity = _acoustic_test_values(tmp_path)
    values = _inputs()
    scene.manifest["source_room"]["room_id"] = "replicacad_apt_0"
    with pytest.raises(M6XCanaryError, match="source_room differs"):
        _fixed_acoustic_identity(
            scene,
            room_capsule=values["room_capsule"],
            room_registry=values["room_registry"],
        )


def test_persisted_mixture_is_the_exact_float32_stem_sum() -> None:
    first = np.zeros((2, 80_000), dtype=np.float64)
    second = np.zeros((2, 80_000), dtype=np.float64)
    first[:, 10:20] = 0.123456789
    second[:, 15:25] = -0.023456781

    stems, mixture = _float32_stems_and_exact_mixture(
        {
            "source_a": SimpleNamespace(episode=first),
            "source_b": SimpleNamespace(episode=second),
        },
        ("source_a", "source_b"),
    )

    expected = np.zeros((2, 80_000), dtype=np.float32)
    np.add(expected, stems["source_a"], out=expected)
    np.add(expected, stems["source_b"], out=expected)
    assert np.array_equal(mixture, expected)


def test_review_index_exposes_listener_events_checks_and_rir(tmp_path: Path) -> None:
    files = {
        name: tmp_path / name
        for name in ("clean.mp4", "diagnostic.mp4", "mixture.wav", "stem.wav")
    }
    for path in files.values():
        path.touch()
    rir = tmp_path / "rir/metadata.json"
    rir.parent.mkdir()
    rir.write_text("{}", encoding="utf-8")
    index = _write_review_index(
        tmp_path,
        [
            {
                "scenario_id": "S5",
                "purpose": "los_nlos_contrast",
                "variant_name": "A",
                "source_ids": ("los", "nlos"),
                "event_windows": ("los: 0.25–2.00s", "nlos: 2.75–4.50s"),
                "spatial_states": (
                    "los: front, camera in_fov, acoustic los",
                    "nlos: rear, camera out_of_fov, acoustic nlos",
                ),
                "status": "pass",
                "checks": "visual / acoustic / timeline / flags: pass",
                "clean_video": files["clean.mp4"],
                "diagnostic_video": files["diagnostic.mp4"],
                "mixture": files["mixture.wav"],
                "stems": (files["stem.wav"],),
                "rir_metadata": rir,
            }
        ],
        listener_position_m=(-0.7, 1.471, 0.65),
        listener_yaw_deg=55.0,
    )
    html = index.read_text(encoding="utf-8")
    assert "[-0.7, 1.471, 0.65]" in html
    assert "yaw <code>55°" in html
    assert "out_of_fov" in html
    assert "RIR evidence" in html
    assert "{escape(" not in html
