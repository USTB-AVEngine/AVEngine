from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from avengine.contracts.json_io import write_json
from avengine.m6.audio_program import AudioProgramError, bind_audio_program_hash
from avengine.m6x.semantic_rir_cache import SemanticRIRCacheSession
from avengine.m7.asset_bound_audio import (
    AssetBoundAudioError,
    float32_stems_and_exact_mix,
    render_asset_bound_binaural,
)
from tools.m7.render_asset_bound_binaural_batch import (
    AudioProgramSpec,
    _prepare_semantic_audio_program_variants,
    _write_semantic_and_verify,
    parse_args,
    render_batch,
)


def _nested_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key for item in value.values() for key in _nested_keys(item)
        }
    if isinstance(value, (list, tuple)):
        return {key for item in value for key in _nested_keys(item)}
    return set()


def _semantic_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    audio_path = tmp_path / "speech.wav"
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(np.full(45_912, 4_000, dtype="<i2").tobytes())
    endpoint_path = tmp_path / "semantic_endpoints.json"
    content_path = tmp_path / "semantic_contents.json"
    binding_path = tmp_path / "semantic_binding.json"
    program_path = tmp_path / "audio_program.json"
    write_json(
        endpoint_path,
        {
            "schema": "avengine_semantic_source_endpoint_registry_v1",
            "registry_id": "episode_semantic_endpoints",
            "revision": "planning_v1",
            "source_endpoint_ids": {
                "source1_emitter": "source1",
                "source2_emitter": "source2",
            },
        },
    )
    write_json(
        content_path,
        {
            "schema": "avengine_semantic_sound_content_registry_v1",
            "registry_id": "episode_semantic_contents",
            "revision": "planning_v1",
            "contents": [
                {
                    "content_id": "speech_content",
                    "sound_asset_id": "speech_asset",
                    "voice_id": "speaker",
                    "source_audio_uri": "semantic://speech_content",
                    "sample_rate_hz": 16_000,
                    "channel_count": 1,
                    "sample_count": 45_912,
                }
            ],
        },
    )
    write_json(
        binding_path,
        {
            "schema": "avengine_semantic_audio_binding_v1",
            "episode_id": "episode",
            "variant_id": "A",
            "content_bindings": {
                "speech_content": {
                    "content_id": "speech_content",
                    "path": str(audio_path),
                    "sample_rate_hz": 16_000,
                    "channel_count": 1,
                    "sample_count": 45_912,
                }
            },
        },
    )
    write_json(
        program_path,
        bind_audio_program_hash(
            {
                "schema": "avengine_semantic_audio_program_v1",
                "program_id": "episode_audio",
                "revision": "planning_v1",
                "mode": "one_active_of_n",
                "timeline": {
                    "time_base_hz": 48_000,
                    "ticks_per_frame": 3_200,
                    "video_fps": 15,
                    "frame_count": 75,
                    "sample_rate_hz": 16_000,
                    "ticks_per_sample": 3,
                    "sample_count": 80_000,
                },
                "candidate_source_endpoint_ids": [
                    "source1_emitter",
                    "source2_emitter",
                ],
                "events": [
                    {
                        "event_id": "target_speech",
                        "source_endpoint_id": "source1_emitter",
                        "content_id": "speech_content",
                        "start_tick": 7_467 * 3,
                        "end_tick_exclusive": 53_379 * 3,
                        "start_sample": 7_467,
                        "end_sample_exclusive": 53_379,
                        "source_start_sample": 0,
                        "source_end_sample_exclusive": 45_912,
                        "source_sample_rate_hz": 16_000,
                        "source_channel_count": 1,
                        "source_sample_count": 45_912,
                        "linear_gain": 1.0,
                        "fade_samples": 0,
                        "render_source_stem": True,
                    }
                ],
                "source_specific_stems": True,
                "admission_state": "research",
            }
        ),
    )
    return program_path, endpoint_path, content_path, binding_path


def _write_semantic_rir_fixture(tmp_path: Path) -> tuple[Path, Path]:
    plan = tmp_path / "plan"
    cache = tmp_path / "rir_cache"
    shards = cache / "shards"
    plan.mkdir()
    shards.mkdir(parents=True)
    write_json(
        plan / "asset_emitter_binding_report.json",
        {
            "status": "pass",
            "scenarios": [
                {
                    "output_episode_id": "episode",
                    "binding_report": {
                        "bindings": [
                            {"source_slot_id": "source1", "asset_id": "female"},
                            {"source_slot_id": "source2", "asset_id": "male"},
                        ]
                    },
                }
            ],
        },
    )
    jobs = [
        {
            "job_id": f"job_{slot}",
            "source_position_m": [float(index + 1), 1.5, 2.0],
            "listener_position_m": [0.0, 1.5, 0.0],
            "listener_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "uses": [
                {
                    "episode_id": "episode",
                    "source_slot_id": slot,
                    "frame_index": frame,
                }
                for frame in range(75)
            ],
        }
        for index, slot in enumerate(("source1", "source2"))
    ]
    plan_path = plan / "rir_job_plan.json"
    write_json(
        plan_path,
        {
            "schema": "avengine_room_rir_job_plan_v2",
            "status": "planned_not_run",
            "listener_pose_mode": "fixed",
            "listener_position_m": [0.0, 1.5, 0.0],
            "listener_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "stride_frames": 1,
            "cache_key_fields": [
                "source_position_m",
                "listener_position_m",
                "listener_orientation_wxyz",
            ],
            "requested_pair_state_count": 150,
            "unique_rir_job_count": 2,
            "jobs": jobs,
        },
    )
    selection = {
        "schema": "avengine_rir_cache_acoustic_selection_binding_v1",
        "selection_mode": "registry",
        "registry_selection_applied": True,
        "room_ref": {
            "registry_id": "room_registry",
            "room_id": "apartment",
            "revision": "room_v1",
        },
        "profile_ref": {"profile_id": "profile", "revision": "profile_v1"},
        "binding_id": "apartment_profile",
    }
    write_json(
        cache / "request.json",
        {
            "schema": "avengine_semantic_rir_cache_request_v1",
            "status": "ready_structural_and_sample_validation",
            "qualification_claim": False,
            "plan": {
                "path": str(plan_path.resolve()),
                "full_job_count": 2,
                "selected_job_offset": 0,
                "selected_job_count": 2,
                "acoustic_state_binding": "source_listener_pose_per_job_v1",
            },
            "acoustic_scene": {
                "manifest_path": str(tmp_path / "missing.json"),
                "package_id": "fixture_package",
            },
            "simulation": {
                "request_path": str(tmp_path / "missing_sim.json"),
                "effective": {
                    "frequency_bands": 4,
                    "direct_sh_order": 3,
                    "indirect_sh_order": 1,
                    "direct_ray_count": 500,
                    "indirect_ray_count": 5000,
                    "indirect_ray_depth": 200,
                    "source_ray_count": 500,
                    "source_ray_depth": 20,
                    "max_diffraction_order": 10,
                    "thread_count": 1,
                    "sample_rate_hz": 16000.0,
                    "max_ir_seconds": 4.0,
                    "unit_scale": 1.0,
                    "global_volume": 1.0,
                    "speed_of_sound_m_s": 343.0,
                    "direct": True,
                    "indirect": True,
                    "diffraction": True,
                    "transmission": True,
                    "mesh_simplification": False,
                    "temporal_coherence": False,
                    "channel_layout": {"type": "binaural", "channel_count": 2},
                },
            },
            "acoustic_selection_binding": selection,
            "output": {
                "layout_type": "binaural",
                "channel_count": 2,
                "layout_id": "rlr_binaural_lr_v1",
                "hrtf_path": str(tmp_path / "missing.hrtf"),
                "compressed_npz_shards": True,
            },
            "runtime_policy": {
                "native_batch_size": 8,
                "coordinate_translation_m": [0.0, 0.0, 0.0],
                "source_radius_m": 0.0,
                "listener_radius_m": 0.0,
                "persistent_context": True,
                "listener_pose_update_policy": "set_listener_pose_on_change_v1",
                "scene_upload_count": 1,
                "compute_device": "CPU",
                "gpu_acceleration": False,
                "execution_mode": "native_default",
            },
        },
    )
    write_json(
        cache / "receipt.json",
        {
            "schema": "avengine_semantic_rir_cache_receipt_v1",
            "status": "pass",
            "qualification_claim": False,
            "claim_boundary": (
                "native CPU RIR samples with structural pose/use, native "
                "source/listener receipts, and decoded-sample validation"
            ),
            "full_plan_complete": True,
            "full_plan_job_count": 2,
            "selected_job_count": 2,
            "sample_rate_hz": 16_000,
            "layout_type": "binaural",
            "layout_id": "rlr_binaural_lr_v1",
            "channel_count": 2,
            "dry_audio_independent": True,
            "compute_device": "CPU",
            "acoustic_state_binding": "source_listener_pose_per_job_v1",
            "listener_pose_update_policy": "set_listener_pose_on_change_v1",
            "acoustic_selection_mode": "registry",
            "retained_shard_count": 1,
            "native_execution": True,
            "native_scene_upload_structurally_validated": True,
            "native_source_listener_receipts_validated": True,
            "native_realized_job_count": 2,
            "native_simulate_owned_call_count": 1,
            "producer_backend": "RLR Audio Propagation",
            "cache_artifact": "room impulse response (RIR)",
            "configured_thread_count": 1,
            "outputs": {
                "request": "request.json",
                "index": "index.json",
                "timing": "timing.json",
                "shards": "shards/",
            },
        },
    )
    entries = [
        {
            "job_id": job["job_id"],
            "job_index": index,
            "shard": "shards/shard_000000.npz",
            "row": index,
            "sample_count": 2,
            "source_position_m": job["source_position_m"],
            "listener_position_m": job["listener_position_m"],
            "listener_orientation_wxyz": job["listener_orientation_wxyz"],
        }
        for index, job in enumerate(jobs)
    ]
    write_json(
        cache / "index.json",
        {
            "schema": "avengine_semantic_rir_cache_index_v1",
            "status": "pass",
            "qualification_claim": False,
            "full_plan_complete": True,
            "selected_job_count": 2,
            "acoustic_state_binding": "source_listener_pose_per_job_v1",
            "acoustic_selection_mode": "registry",
            "entries": entries,
        },
    )
    write_json(
        cache / "timing.json",
        {
            "schema": "avengine_semantic_rir_cache_timing_v1",
            "status": "pass",
            "setup": {
                "schema": "avengine_semantic_native_rir_setup_v1",
                "runtime": {
                    "schema": "avengine_semantic_habitat_rlr_runtime_v1",
                    "binding_api": "habitat_sim.RLRAcousticContext_v1",
                    "quaternion_module_path": "/fixture/quaternion.py",
                    "habitat_module_path": (
                        "/fixture/src_python/habitat_sim/__init__.py"
                    ),
                    "binding_module_path": (
                        "/fixture/build/install/platlib/habitat_sim/_ext/"
                        "habitat_sim_bindings.cpython-312-x86_64-linux-gnu.so"
                    ),
                    "rlr_library_path": (
                        "/fixture/build/install/platlib/habitat_sim/_ext/"
                        "libRLRAudioPropagation.so"
                    ),
                },
                "configuration_readback": {
                    key: value
                    for key, value in {
                        "frequency_bands": 4,
                        "direct_sh_order": 3,
                        "indirect_sh_order": 1,
                        "direct_ray_count": 500,
                        "indirect_ray_count": 5000,
                        "indirect_ray_depth": 200,
                        "source_ray_count": 500,
                        "source_ray_depth": 20,
                        "max_diffraction_order": 10,
                        "thread_count": 1,
                        "sample_rate_hz": 16000.0,
                        "max_ir_seconds": 4.0,
                        "unit_scale": 1.0,
                        "global_volume": 1.0,
                        "speed_of_sound_m_s": 343.0,
                        "direct": True,
                        "indirect": True,
                        "diffraction": True,
                        "transmission": True,
                        "mesh_simplification": False,
                        "temporal_coherence": False,
                        "channel_layout": {"type": "binaural", "channel_count": 2},
                    }.items()
                    if key not in {"speed_of_sound_m_s", "channel_layout"}
                },
                "compute_device": "CPU",
                "qualification_claim": False,
                "upload": {
                    "status": "pass_structural_native_upload",
                    "object_count": 1,
                    "vertex_count": 3,
                    "triangle_count": 1,
                    "material_category_count": 1,
                    "object_ids": ["object"],
                    "triangle_count_by_material": {"wall": 1},
                    "material_upload_call_count": {"wall": 1},
                    "resolved_material_name_by_category": {"wall": "wall"},
                    "resolved_material_index_by_category": {"wall": 0},
                },
                "wall_seconds": 0.01,
                "process_cpu_seconds": 0.01,
            },
            "batches": [
                {
                    "batch_index": 0,
                    "job_count": 2,
                    "simulate_wall_seconds": 0.1,
                    "simulate_process_cpu_seconds": 0.1,
                    "serialization_wall_seconds": 0.1,
                }
            ],
            "selected_job_count": 2,
            "simulate_wall_seconds": 0.1,
            "serialization_wall_seconds": 0.1,
            "run_wall_seconds": 0.3,
            "jobs_per_simulate_second": 20.0,
        },
    )
    samples = np.zeros((2, 2, 2), dtype="<f4")
    samples[0, :, 0] = (1.0, 0.5)
    samples[1, :, 0] = (0.25, 1.0)
    np.savez(
        shards / "shard_000000.npz",
        job_indices=np.asarray([0, 1], dtype="<u4"),
        job_ids=np.asarray([job["job_id"] for job in jobs]),
        source_positions_m=np.asarray(
            [job["source_position_m"] for job in jobs], dtype="<f8"
        ),
        listener_positions_m=np.asarray(
            [job["listener_position_m"] for job in jobs], dtype="<f8"
        ),
        listener_orientations_wxyz=np.asarray(
            [job["listener_orientation_wxyz"] for job in jobs], dtype="<f8"
        ),
        lengths=np.asarray([2, 2], dtype="<u4"),
        samples=samples,
        sample_rate_hz=np.asarray(16_000, dtype="<u4"),
        layout_id=np.asarray("rlr_binaural_lr_v1"),
        channel_labels=np.asarray(["left", "right"]),
        simulate_wall_seconds=np.asarray(0.1, dtype="<f8"),
        simulate_process_cpu_seconds=np.asarray(0.1, dtype="<f8"),
        indirect_ray_efficiency=np.asarray(0.5, dtype="<f8"),
    )
    return plan, cache


def test_semantic_program_runs_exact_dry_to_real_cpu_binaural_chain(
    tmp_path: Path,
) -> None:
    program, endpoints, contents, binding = _semantic_fixture(tmp_path)
    prepared, library = _prepare_semantic_audio_program_variants(
        specs=(AudioProgramSpec(program),),
        expected_episode_id="episode",
        semantic_source_endpoint_registry_path=endpoints,
        semantic_sound_content_registry_path=contents,
        semantic_audio_binding_path=binding,
    )

    item = prepared[0]
    source1 = item.dry_by_source_slot["source1"]
    source2 = item.dry_by_source_slot["source2"]
    assert not np.any(source1[:7_467])
    assert np.any(source1[7_467:53_379])
    assert not np.any(source1[53_379:])
    assert not np.any(source2)

    rirs = np.zeros((1, 2, 2, 1), dtype=np.float64)
    rirs[0, 0, :, 0] = (1.0, 0.5)
    rirs[0, 1, :, 0] = (0.25, 1.0)
    stems, _mixture = render_asset_bound_binaural(
        item.dry_by_source_slot,
        rir_samples=rirs,
        rir_lengths=np.ones((1, 2), dtype=np.uint32),
        source_ids=("source1", "source2"),
        keyframe_samples=(0,),
    )
    stored_stems, stored_mixture = float32_stems_and_exact_mix(
        stems, source_ids=("source1", "source2")
    )
    assert np.all(np.isfinite(stored_mixture))
    assert np.array_equal(
        stored_mixture, stored_stems["source1"] + stored_stems["source2"]
    )
    output = tmp_path / "semantic_binaural.wav"
    record = _write_semantic_and_verify(
        output, stored_mixture, role="semantic_unit_binaural"
    )
    assert output.is_file()
    assert not output.with_suffix(".wav.json").exists()
    forbidden = {
        "sha256",
        "input_sha256",
        "file_sha256",
        "audio_sha256",
        "sidecar_sha256",
        "byte_size",
        "audio_byte_size",
        "resident_sample_payload_bytes",
    }
    assert not (_nested_keys(item.instance_record) & forbidden)
    assert not (_nested_keys(library) & forbidden)
    assert not (_nested_keys(record) & forbidden)


@pytest.mark.parametrize("extra_field", ["file_sha256", "byte_size"])
def test_semantic_program_rejects_metadata_on_unused_content_binding(
    tmp_path: Path, extra_field: str
) -> None:
    program, endpoints, contents_path, binding_path = _semantic_fixture(tmp_path)
    contents = json.loads(contents_path.read_text())
    binding = json.loads(binding_path.read_text())
    unused = {
        "content_id": "unused_content",
        "sound_asset_id": "unused_asset",
        "voice_id": "unused_voice",
        "source_audio_uri": "semantic://unused_content",
        "sample_rate_hz": 16_000,
        "channel_count": 1,
        "sample_count": 45_912,
    }
    contents["contents"].append(unused)
    binding["content_bindings"]["unused_content"] = {
        "content_id": "unused_content",
        "path": binding["content_bindings"]["speech_content"]["path"],
        "sample_rate_hz": 16_000,
        "channel_count": 1,
        "sample_count": 45_912,
        extra_field: "not_allowed",
    }
    write_json(contents_path, contents)
    write_json(binding_path, binding)

    with pytest.raises(AudioProgramError, match="binding structure"):
        _prepare_semantic_audio_program_variants(
            specs=(AudioProgramSpec(program),),
            expected_episode_id="episode",
            semantic_source_endpoint_registry_path=endpoints,
            semantic_sound_content_registry_path=contents_path,
            semantic_audio_binding_path=binding_path,
        )


def test_semantic_render_batch_is_digest_free_structural_cpu_e2e(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    program, endpoints, contents, binding = _semantic_fixture(tmp_path)
    plan, cache = _write_semantic_rir_fixture(tmp_path)

    def forbidden_call(*_args, **_kwargs):
        raise AssertionError("semantic render must not enter a legacy digest path")

    module = __import__(
        "tools.m7.render_asset_bound_binaural_batch", fromlist=["unused"]
    )
    monkeypatch.setattr(module, "RIRCacheSession", forbidden_call)
    monkeypatch.setattr(module, "sha256_file", forbidden_call)
    monkeypatch.setattr(module, "canonical_json_sha256", forbidden_call)
    output = tmp_path / "rendered"
    assert (
        render_batch(
            plan_root=plan,
            rir_cache_root=cache,
            asset_audio=None,
            asset_channel_policies=None,
            asset_gains=None,
            variants_per_episode=1,
            fade_samples=0,
            maximum_mixture_peak=1.0,
            retain_stems=True,
            output=output,
            audio_program_specs=(AudioProgramSpec(program),),
            semantic_source_endpoint_registry_path=endpoints,
            semantic_sound_content_registry_path=contents,
            semantic_audio_binding_path=binding,
        )
        == output.resolve()
    )
    mixture = output / "audio/binaural/episode__v00.wav"
    stems = {
        slot: output / f"audio/binaural/stems/{slot}/episode__v00.wav"
        for slot in ("source1", "source2")
    }
    from avengine.spatial_audio.audio import read_float32_wav

    decoded_mix = read_float32_wav(mixture, verify_sidecar=False).samples
    decoded_stems = {
        slot: read_float32_wav(path, verify_sidecar=False).samples
        for slot, path in stems.items()
    }
    assert decoded_mix.shape == (2, 80_000)
    assert np.array_equal(
        decoded_mix, decoded_stems["source1"] + decoded_stems["source2"]
    )
    assert not np.any(decoded_stems["source1"][:, :7_467])
    assert np.any(decoded_stems["source1"][:, 7_467:53_379])
    # The dry event is exactly [7467, 53379); the two-tap RIR may retain one
    # physical convolution-tail sample at the exclusive endpoint.
    assert not np.any(decoded_stems["source1"][:, 53_380:])
    assert not np.any(decoded_stems["source2"])
    assert not list(output.rglob("*.wav.json"))
    forbidden = {
        "sha256",
        "file_sha256",
        "input_sha256",
        "audio_sha256",
        "sidecar_sha256",
        "byte_size",
        "audio_byte_size",
        "resident_sample_payload_bytes",
    }
    for path in output.rglob("*.json"):
        assert not (_nested_keys(json.loads(path.read_text())) & forbidden), path


def test_semantic_render_batch_atomic_publish_does_not_replace_racing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    program, endpoints, contents, binding = _semantic_fixture(tmp_path)
    plan, cache = _write_semantic_rir_fixture(tmp_path)
    output = tmp_path / "racing_render"
    module = __import__(
        "tools.m7.render_asset_bound_binaural_batch", fromlist=["unused"]
    )
    original_publish = module.atomic_publish_directory

    def create_destination_then_publish(policy, staging, destination):
        destination.mkdir()
        (destination / "concurrent_marker").write_text("preserve")
        return original_publish(policy, staging, destination)

    monkeypatch.setattr(
        module, "atomic_publish_directory", create_destination_then_publish
    )
    with pytest.raises(FileExistsError, match="refusing to replace"):
        render_batch(
            plan_root=plan,
            rir_cache_root=cache,
            asset_audio=None,
            asset_channel_policies=None,
            asset_gains=None,
            variants_per_episode=1,
            fade_samples=0,
            maximum_mixture_peak=1.0,
            retain_stems=True,
            output=output,
            audio_program_specs=(AudioProgramSpec(program),),
            semantic_source_endpoint_registry_path=endpoints,
            semantic_sound_content_registry_path=contents,
            semantic_audio_binding_path=binding,
        )

    assert sorted(path.name for path in output.iterdir()) == ["concurrent_marker"]
    assert (output / "concurrent_marker").read_text() == "preserve"
    assert not list(tmp_path.glob(".racing_render.staging.*"))


@pytest.mark.parametrize("kind", ["plan_root", "cache_root", "plan_file"])
def test_semantic_render_batch_rejects_symlinked_selected_roots(
    tmp_path: Path, kind: str
) -> None:
    program, endpoints, contents, binding = _semantic_fixture(tmp_path)
    plan, cache = _write_semantic_rir_fixture(tmp_path)
    if kind == "plan_file":
        path = plan / "rir_job_plan.json"
        target = tmp_path / "outside_rir_job_plan.json"
        path.rename(target)
        path.symlink_to(target)
    else:
        path = plan if kind == "plan_root" else cache
        target = tmp_path / f"real_{path.name}"
        path.rename(target)
        path.symlink_to(target, target_is_directory=True)
    with pytest.raises(AssetBoundAudioError, match="non-symlink"):
        render_batch(
            plan_root=plan,
            rir_cache_root=cache,
            asset_audio=None,
            asset_channel_policies=None,
            asset_gains=None,
            variants_per_episode=1,
            fade_samples=0,
            maximum_mixture_peak=1.0,
            retain_stems=True,
            output=tmp_path / "must_not_render",
            audio_program_specs=(AudioProgramSpec(program),),
            semantic_source_endpoint_registry_path=endpoints,
            semantic_sound_content_registry_path=contents,
            semantic_audio_binding_path=binding,
        )


@pytest.mark.parametrize("kind", ["dangling_leaf", "symlink_parent"])
def test_semantic_render_batch_rejects_symlinked_output_path(
    tmp_path: Path, kind: str
) -> None:
    program, endpoints, contents, binding = _semantic_fixture(tmp_path)
    plan, cache = _write_semantic_rir_fixture(tmp_path)
    outside = tmp_path / "outside"
    if kind == "dangling_leaf":
        target = outside / "rendered"
        output = tmp_path / "output_link"
        output.symlink_to(target, target_is_directory=True)
    else:
        outside.mkdir()
        parent = tmp_path / "linked_parent"
        parent.symlink_to(outside, target_is_directory=True)
        output = parent / "rendered"
        target = outside / "rendered"

    with pytest.raises(AssetBoundAudioError, match="symlinks"):
        render_batch(
            plan_root=plan,
            rir_cache_root=cache,
            asset_audio=None,
            asset_channel_policies=None,
            asset_gains=None,
            variants_per_episode=1,
            fade_samples=0,
            maximum_mixture_peak=1.0,
            retain_stems=True,
            output=output,
            audio_program_specs=(AudioProgramSpec(program),),
            semantic_source_endpoint_registry_path=endpoints,
            semantic_sound_content_registry_path=contents,
            semantic_audio_binding_path=binding,
        )
    assert not target.exists()


@pytest.mark.parametrize("episode_id_kind", ["absolute", "parent_traversal"])
def test_semantic_render_rejects_episode_id_output_escape(
    tmp_path: Path, episode_id_kind: str
) -> None:
    program, endpoints, contents, binding = _semantic_fixture(tmp_path)
    plan, cache = _write_semantic_rir_fixture(tmp_path)
    report_path = plan / "asset_emitter_binding_report.json"
    report = json.loads(report_path.read_text())
    escaped_root = tmp_path / "escaped_audio"
    episode_id = (
        str(escaped_root) if episode_id_kind == "absolute" else "../../escaped_audio"
    )
    report["scenarios"][0]["output_episode_id"] = episode_id
    write_json(report_path, report)

    output = tmp_path / "must_not_render"
    with pytest.raises(AssetBoundAudioError, match="path-safe stable"):
        render_batch(
            plan_root=plan,
            rir_cache_root=cache,
            asset_audio=None,
            asset_channel_policies=None,
            asset_gains=None,
            variants_per_episode=1,
            fade_samples=0,
            maximum_mixture_peak=1.0,
            retain_stems=True,
            output=output,
            audio_program_specs=(AudioProgramSpec(program),),
            semantic_source_endpoint_registry_path=endpoints,
            semantic_sound_content_registry_path=contents,
            semantic_audio_binding_path=binding,
        )
    assert not output.exists()
    assert not escaped_root.exists()
    assert not list(tmp_path.rglob("escaped_audio*.wav"))


def _open_semantic_rir_fixture(plan: Path, cache: Path):
    module = __import__(
        "tools.m7.render_asset_bound_binaural_batch", fromlist=["unused"]
    )
    return module._SemanticRIRCacheSession(
        cache_root=cache,
        plan_path=plan / "rir_job_plan.json",
        expected_episode_id="episode",
        frame_count=75,
        frame_rate_hz=15,
    )


def test_owned_semantic_reader_matches_m7_compatibility_adapter(
    tmp_path: Path,
) -> None:
    plan, cache = _write_semantic_rir_fixture(tmp_path)
    arguments = {
        "cache_root": cache,
        "plan_path": plan / "rir_job_plan.json",
        "expected_episode_id": "episode",
        "frame_count": 75,
        "frame_rate_hz": 15,
    }
    owned_session = SemanticRIRCacheSession(**arguments)
    adapted_session = _open_semantic_rir_fixture(plan, cache)
    assert isinstance(adapted_session, SemanticRIRCacheSession)
    assert (
        adapted_session.acoustic_selection_binding
        == owned_session.acoustic_selection_binding
    )

    owned = owned_session.load_episode("episode")
    adapted = adapted_session.load_episode("episode")
    np.testing.assert_array_equal(adapted.samples, owned.samples)
    np.testing.assert_array_equal(adapted.lengths, owned.lengths)
    assert adapted.source_slot_ids == owned.source_slot_ids
    assert adapted.visual_frame_indices == owned.visual_frame_indices
    assert adapted.keyframe_samples == owned.keyframe_samples
    assert adapted.sample_rate_hz == owned.sample_rate_hz
    assert adapted.layout_type == owned.layout_type
    assert adapted.layout_id == owned.layout_id
    assert adapted.channel_labels == owned.channel_labels
    assert adapted.evidence == owned.evidence


@pytest.mark.parametrize("mutation", ["single_frame", "extra_episode"])
def test_semantic_rir_rejects_incomplete_or_foreign_episode_grid(
    tmp_path: Path, mutation: str
) -> None:
    plan, cache = _write_semantic_rir_fixture(tmp_path)
    plan_path = plan / "rir_job_plan.json"
    document = json.loads(plan_path.read_text())
    if mutation == "single_frame":
        for job in document["jobs"]:
            job["uses"] = job["uses"][:1]
        document["requested_pair_state_count"] = 2
    else:
        document["jobs"][0]["uses"][-1]["episode_id"] = "foreign_episode"
    write_json(plan_path, document)
    with pytest.raises(AssetBoundAudioError):
        _open_semantic_rir_fixture(plan, cache)


def test_semantic_rir_accepts_legacy_colocated_runtime_receipt(tmp_path: Path) -> None:
    plan, cache = _write_semantic_rir_fixture(tmp_path)
    timing = json.loads((cache / "timing.json").read_text())
    runtime = timing["setup"]["runtime"]
    assert (
        Path(runtime["habitat_module_path"]).parent
        != Path(runtime["binding_module_path"]).parent
    )
    assert Path(runtime["rlr_library_path"]).parent == Path(
        runtime["binding_module_path"]
    ).parent
    assert _open_semantic_rir_fixture(plan, cache).load_episode(
        "episode"
    ).samples.shape == (
        75,
        2,
        2,
        2,
    )


def _upgrade_semantic_fixture_to_current_installed(
    tmp_path: Path, cache: Path
) -> dict[str, object]:
    runtime_prefix = tmp_path / "installed_runtime"
    habitat_module = runtime_prefix / "habitat_sim/__init__.py"
    habitat_binding = (
        runtime_prefix
        / "habitat_sim/_ext/"
        "habitat_sim_bindings.cpython-312-x86_64-linux-gnu.so"
    )
    magnum_python_site = tmp_path / "magnum_python_site"
    sdk_root = tmp_path / "rlr_sdk"
    sdk_header = sdk_root / "headers/RLRAudioPropagation.h"
    sdk_library = sdk_root / "libs/linux/x64/libRLRAudioPropagation.so"
    quaternion_module = tmp_path / "quaternion.py"
    for path in (habitat_module, habitat_binding, sdk_header, sdk_library):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    magnum_python_site.mkdir()
    quaternion_module.write_bytes(b"fixture")
    identity: dict[str, object] = {
        "identity_schema": "avengine_current_installed_rlr_runtime_v1",
        "mode": "current-installed",
        "habitat_runtime_prefix": str(runtime_prefix),
        "habitat_sim_module": str(habitat_module),
        "habitat_sim_binding": str(habitat_binding),
        "magnum_python_site": str(magnum_python_site),
        "rlr_sdk_root": str(sdk_root),
        "rlr_sdk_header": str(sdk_header),
        "rlr_sdk_library": str(sdk_library),
        "rlr_adapter_enabled": True,
        "binding_api": "habitat_sim.RLRAcousticContext_v1",
    }
    timing_path = cache / "timing.json"
    timing = json.loads(timing_path.read_text())
    timing["setup"]["runtime"] = {
        "schema": "avengine_semantic_habitat_rlr_runtime_v1",
        "binding_api": "habitat_sim.RLRAcousticContext_v1",
        "quaternion_module_path": str(quaternion_module),
        "habitat_module_path": str(habitat_module),
        "binding_module_path": str(habitat_binding),
        "rlr_library_path": str(sdk_library),
        "runtime_mode": "current-installed",
        "runtime_identity": identity,
    }
    write_json(timing_path, timing)
    return identity


def test_semantic_rir_accepts_current_installed_external_sdk_layout(
    tmp_path: Path,
) -> None:
    plan, cache = _write_semantic_rir_fixture(tmp_path)
    identity = _upgrade_semantic_fixture_to_current_installed(tmp_path, cache)

    episode = _open_semantic_rir_fixture(plan, cache).load_episode("episode")

    assert episode.samples.shape == (75, 2, 2, 2)
    assert Path(str(identity["rlr_sdk_library"])).parent != Path(
        str(identity["habitat_sim_binding"])
    ).parent


@pytest.mark.parametrize(
    "mutation",
    [
        "identity_library",
        "sdk_git_checkout",
        "binding_outside_prefix",
        "coherent_split_habitat",
        "coherent_sdk_alias",
    ],
)
def test_semantic_rir_rejects_invalid_current_installed_runtime_identity(
    tmp_path: Path, mutation: str
) -> None:
    plan, cache = _write_semantic_rir_fixture(tmp_path)
    identity = _upgrade_semantic_fixture_to_current_installed(tmp_path, cache)
    timing_path = cache / "timing.json"
    timing = json.loads(timing_path.read_text())
    runtime = timing["setup"]["runtime"]
    if mutation == "identity_library":
        replacement = Path(str(identity["rlr_sdk_root"])) / "other/libRLRAudioPropagation.so"
        replacement.parent.mkdir()
        replacement.write_bytes(b"other")
        runtime["runtime_identity"]["rlr_sdk_library"] = str(replacement)
    elif mutation == "sdk_git_checkout":
        (Path(str(identity["rlr_sdk_root"])) / ".git").mkdir()
    elif mutation == "binding_outside_prefix":
        replacement = tmp_path / "outside/habitat_sim_bindings.cpython-312-x86_64-linux-gnu.so"
        replacement.parent.mkdir()
        replacement.write_bytes(b"outside")
        runtime["binding_module_path"] = str(replacement)
        runtime["runtime_identity"]["habitat_sim_binding"] = str(replacement)
    elif mutation == "coherent_split_habitat":
        replacement = (
            Path(str(identity["habitat_runtime_prefix"]))
            / "alternate/habitat_sim/_ext/"
            "habitat_sim_bindings.cpython-312-x86_64-linux-gnu.so"
        )
        replacement.parent.mkdir(parents=True)
        replacement.write_bytes(b"split")
        runtime["binding_module_path"] = str(replacement)
        runtime["runtime_identity"]["habitat_sim_binding"] = str(replacement)
    else:
        sdk_root = Path(str(identity["rlr_sdk_root"]))
        replacement_header = sdk_root / "alternate/headers/RLRAudioPropagation.h"
        replacement_library = (
            sdk_root
            / "alternate/libs/linux/x64/libRLRAudioPropagation.so"
        )
        replacement_header.parent.mkdir(parents=True)
        replacement_library.parent.mkdir(parents=True)
        replacement_header.write_bytes(b"alias-header")
        replacement_library.write_bytes(b"alias-library")
        runtime["rlr_library_path"] = str(replacement_library)
        runtime["runtime_identity"]["rlr_sdk_header"] = str(replacement_header)
        runtime["runtime_identity"]["rlr_sdk_library"] = str(replacement_library)
    write_json(timing_path, timing)

    with pytest.raises(AssetBoundAudioError):
        _open_semantic_rir_fixture(plan, cache)


def test_semantic_rir_accepts_fixed_jobs_with_top_level_pose_fallback(
    tmp_path: Path,
) -> None:
    plan, cache = _write_semantic_rir_fixture(tmp_path)
    plan_path = plan / "rir_job_plan.json"
    document = json.loads(plan_path.read_text())
    for job in document["jobs"]:
        job.pop("listener_position_m")
        job.pop("listener_orientation_wxyz")
    write_json(plan_path, document)
    episode = _open_semantic_rir_fixture(plan, cache).load_episode("episode")
    assert episode.samples.shape == (75, 2, 2, 2)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_state", "duplicate acoustic pose"),
        ("fixed_mismatch", "fixed plan pose"),
        ("fixed_missing_top_level", "fixed listener position"),
        ("per_episode_top_level", "top-level pose"),
    ],
)
def test_semantic_rir_rejects_pose_mode_or_acoustic_state_drift(
    tmp_path: Path, mutation: str, message: str
) -> None:
    plan, cache = _write_semantic_rir_fixture(tmp_path)
    plan_path = plan / "rir_job_plan.json"
    document = json.loads(plan_path.read_text())
    if mutation == "duplicate_state":
        document["jobs"][1]["source_position_m"] = document["jobs"][0][
            "source_position_m"
        ]
    elif mutation == "fixed_mismatch":
        document["jobs"][1]["listener_position_m"][0] = 0.5
    elif mutation == "fixed_missing_top_level":
        document.pop("listener_position_m")
    else:
        document["listener_pose_mode"] = "per_episode_frame"
    write_json(plan_path, document)
    with pytest.raises(AssetBoundAudioError):
        _open_semantic_rir_fixture(plan, cache)


@pytest.mark.parametrize("kind", ["fixed_json", "shard"])
def test_semantic_rir_rejects_symlinked_inputs(tmp_path: Path, kind: str) -> None:
    plan, cache = _write_semantic_rir_fixture(tmp_path)
    path = (
        cache / "request.json"
        if kind == "fixed_json"
        else cache / "shards/shard_000000.npz"
    )
    target = tmp_path / f"outside_{path.name}"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(
        AssetBoundAudioError,
        match="escapes its selected root|fixed regular JSON|symlink",
    ):
        _open_semantic_rir_fixture(plan, cache)


@pytest.mark.parametrize(
    "mutation",
    [
        "native_false",
        "hybrid_receipt",
        "upload_count",
        "batch_cpu_time",
        "run_wall",
        "canonical_input_path",
        "config_readback",
        "batch_partition",
        "runtime_package_tail",
    ],
)
def test_semantic_rir_rejects_fabricated_native_or_timing_claims(
    tmp_path: Path, mutation: str
) -> None:
    plan, cache = _write_semantic_rir_fixture(tmp_path)
    if mutation in {"native_false", "hybrid_receipt"}:
        path = cache / "receipt.json"
        document = json.loads(path.read_text())
        if mutation == "native_false":
            document["native_execution"] = False
        else:
            document["file_sha256"] = "legacy-evidence-must-not-mix"
    elif mutation in {"canonical_input_path", "batch_partition"}:
        path = cache / "request.json"
        document = json.loads(path.read_text())
        if mutation == "canonical_input_path":
            document["acoustic_scene"]["manifest_path"] = (
                "/fixture/../fixture/scene.json"
            )
        else:
            document["runtime_policy"]["native_batch_size"] = 1
    else:
        path = cache / "timing.json"
        document = json.loads(path.read_text())
        if mutation == "upload_count":
            document["setup"]["upload"]["triangle_count"] = 2
        elif mutation == "config_readback":
            document["setup"]["configuration_readback"]["direct_ray_count"] += 1
        elif mutation == "batch_cpu_time":
            document["batches"][0]["simulate_process_cpu_seconds"] = 0.2
        elif mutation == "runtime_package_tail":
            document["setup"]["runtime"]["binding_module_path"] = (
                "/fixture/build/install/platlib/not_habitat/_ext/"
                "habitat_sim_bindings.cpython-312-x86_64-linux-gnu.so"
            )
            document["setup"]["runtime"]["rlr_library_path"] = (
                "/fixture/build/install/platlib/not_habitat/_ext/"
                "libRLRAudioPropagation.so"
            )
        else:
            document["run_wall_seconds"] = 0.0
    write_json(path, document)
    with pytest.raises(AssetBoundAudioError):
        _open_semantic_rir_fixture(plan, cache)


@pytest.mark.parametrize("mutation", ["fortran_order", "duplicate_job_id", "extra_row"])
def test_semantic_rir_rejects_nonproducer_shard_structure(
    tmp_path: Path, mutation: str
) -> None:
    plan, cache = _write_semantic_rir_fixture(tmp_path)
    shard = cache / "shards/shard_000000.npz"
    with np.load(shard, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    if mutation == "fortran_order":
        arrays["source_positions_m"] = np.asfortranarray(arrays["source_positions_m"])
    elif mutation == "duplicate_job_id":
        arrays["job_ids"][1] = arrays["job_ids"][0]
    else:
        for name in (
            "job_indices",
            "job_ids",
            "source_positions_m",
            "listener_positions_m",
            "listener_orientations_wxyz",
            "lengths",
            "samples",
        ):
            arrays[name] = np.concatenate((arrays[name], arrays[name][-1:]), axis=0)
        arrays["job_indices"][-1] = 2
        arrays["job_ids"][-1] = "extra_job"
    np.savez(shard, **arrays)
    with pytest.raises(AssetBoundAudioError):
        _open_semantic_rir_fixture(plan, cache)


def test_semantic_rir_rejects_producer_dtype_drift(tmp_path: Path) -> None:
    plan, cache = _write_semantic_rir_fixture(tmp_path)
    shard = cache / "shards/shard_000000.npz"
    with np.load(shard, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    arrays["sample_rate_hz"] = np.asarray(16_000, dtype="<i8")
    np.savez(shard, **arrays)
    with pytest.raises(AssetBoundAudioError, match="metadata is invalid"):
        _open_semantic_rir_fixture(plan, cache)


def test_semantic_cli_is_explicit_and_legacy_inputs_are_mutually_exclusive(
    tmp_path: Path,
) -> None:
    program, endpoints, contents, binding = _semantic_fixture(tmp_path)
    args = parse_args(
        [
            "--plan-root",
            str(tmp_path / "plan"),
            "--rir-cache",
            str(tmp_path / "cache"),
            "--audio-program",
            str(program),
            "--semantic-source-endpoint-registry",
            str(endpoints),
            "--semantic-sound-content-registry",
            str(contents),
            "--semantic-audio-binding",
            str(binding),
            "--output",
            str(tmp_path / "output"),
        ]
    )
    assert args.semantic_source_endpoint_registry == endpoints
    assert args.semantic_sound_content_registry == contents
    assert args.semantic_audio_binding == binding

    plan = tmp_path / "plan"
    plan.mkdir()
    write_json(
        plan / "asset_emitter_binding_report.json",
        {
            "status": "pass",
            "scenarios": [
                {
                    "output_episode_id": "episode",
                    "binding_report": {
                        "bindings": [
                            {"source_slot_id": "source1", "asset_id": "female"},
                            {"source_slot_id": "source2", "asset_id": "male"},
                        ]
                    },
                }
            ],
        },
    )
    write_json(
        plan / "rir_job_plan.json",
        {
            "schema": "avengine_room_rir_job_plan_v2",
            "status": "planned_not_run",
            "listener_pose_mode": "fixed",
            "jobs": [],
        },
    )
    with pytest.raises(AssetBoundAudioError, match="mutually exclusive"):
        render_batch(
            plan_root=plan,
            rir_cache_root=tmp_path / "cache",
            asset_audio={},
            asset_channel_policies={},
            asset_gains={},
            variants_per_episode=1,
            fade_samples=0,
            maximum_mixture_peak=0.95,
            retain_stems=True,
            output=tmp_path / "output",
            audio_program_specs=(AudioProgramSpec(program),),
            source_endpoint_registry_path=endpoints,
            sound_asset_registry_path=contents,
            endpoint_to_source_slot={},
            sound_audio={},
            semantic_source_endpoint_registry_path=endpoints,
            semantic_sound_content_registry_path=contents,
            semantic_audio_binding_path=binding,
        )
