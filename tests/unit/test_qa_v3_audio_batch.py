"""Contract tests for the QA-v3 sequential audio runner."""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_qa_v3_audio_batch",
    REPOSITORY / "tools/qa/run_qa_v3_audio_batch.py",
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def test_per_point_m1_request_overrides_legacy_batch_fallback(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    point = inputs / "card1F_001"
    point.mkdir(parents=True)
    per_point = point / "m1_capture_request.json"
    per_point.write_text("{}")
    fallback = tmp_path / "fallback.json"
    fallback.write_text("{}")
    assert TOOL.point_m1_request(inputs, "card1F_001", str(fallback)) == per_point


def test_legacy_batch_fallback_remains_available(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    fallback = tmp_path / "fallback.json"
    fallback.write_text("{}")
    assert TOOL.point_m1_request(inputs, "old_point", str(fallback)) == fallback


def test_missing_point_and_fallback_m1_request_fails(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    with pytest.raises(SystemExit, match="M1 request is missing"):
        TOOL.point_m1_request(inputs, "card1F_001", str(tmp_path / "absent.json"))


def test_program_path_matches_generator_main_and_gatea_names(tmp_path: Path) -> None:
    programs = tmp_path / "programs"
    programs.mkdir()
    main = programs / "qa_v3_dog_card1F_001_rand_v1.json"
    gatea = programs / "qa_v3_dog_card1F_001_rand_gateA_v1.json"
    main.write_text("{}")
    gatea.write_text("{}")
    assert TOOL.program_path(programs, "card1F_001", "main") == main
    assert TOOL.program_path(programs, "card1F_001", "gateA") == gatea


def test_hrtf_is_required_only_for_binaural_layout() -> None:
    assert TOOL.hrtf_args({}, ("ambisonics",)) == []
    assert TOOL.hrtf_args({"hrtf": "/tmp/hrtf.sofa"}, ("binaural",)) == [
        "--hrtf",
        "/tmp/hrtf.sofa",
    ]
    with pytest.raises(SystemExit, match="required when layouts include binaural"):
        TOOL.hrtf_args({}, ("binaural",))


def test_canonical_emitter_policy_is_explicit_and_validated() -> None:
    assert TOOL.canonical_emitter_args({}) == []
    assert TOOL.canonical_emitter_args({
        "canonical_emitter_height_m": 0.61575,
    }) == ["--canonical-emitter-height-m", "0.61575"]
    with pytest.raises(SystemExit, match="finite and positive"):
        TOOL.canonical_emitter_args({"canonical_emitter_height_m": 0})



def _write_float32_wav(
    path: Path,
    *,
    channels: int = 2,
    sample_rate_hz: int = 16_000,
    frame_count: int = 4,
    fact_frame_count: int | None = None,
) -> None:
    """Write the small IEEE-float fixture expected from the audio renderer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = struct.pack(
        "<HHIIHH",
        3,
        channels,
        sample_rate_hz,
        sample_rate_hz * channels * 4,
        channels * 4,
        32,
    )
    fact = struct.pack(
        "<I", frame_count if fact_frame_count is None else fact_frame_count
    )
    data = b"\0" * (frame_count * channels * 4)
    chunks = (
        b"fmt " + struct.pack("<I", len(fmt)) + fmt
        + b"fact" + struct.pack("<I", len(fact)) + fact
        + b"data" + struct.pack("<I", len(data)) + data
    )
    path.write_bytes(b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks)


def _write_audio_receipt(
    output: Path,
    *,
    status: str = "pass",
    sample_rate_hz: int = 16_000,
    sample_count: int = 4,
    execution_variant: str | None = None,
) -> None:
    (output / "audio" / "binaural").mkdir(parents=True, exist_ok=True)
    receipt = {
        "status": status,
        "audio": {
            "sample_rate_hz": sample_rate_hz,
            "sample_count": sample_count,
        },
    }
    if execution_variant is not None:
        receipt["execution_variant"] = execution_variant
    (output / "research_receipt.json").write_text(json.dumps(receipt))


def test_point_state_requires_success_receipt_and_matching_wav_metadata(
    tmp_path: Path,
) -> None:
    output = tmp_path / "point"
    _write_audio_receipt(output)
    _write_float32_wav(output / "audio" / "binaural" / "mixture.wav")
    _write_float32_wav(
        output / "audio" / "dry" / "source1.wav", channels=1
    )
    assert TOOL.point_state(output) == "complete"

    failed = tmp_path / "failed"
    _write_audio_receipt(failed, status="fail")
    _write_float32_wav(failed / "audio" / "binaural" / "mixture.wav")
    assert TOOL.point_state(failed) == "partial"

    mismatched = tmp_path / "mismatched"
    _write_audio_receipt(mismatched)
    _write_float32_wav(
        mismatched / "audio" / "binaural" / "mixture.wav", frame_count=3
    )
    assert TOOL.point_state(mismatched) == "partial"


def test_point_state_rejects_non_float_or_inconsistent_fact_metadata(
    tmp_path: Path,
) -> None:
    output = tmp_path / "codec"
    _write_audio_receipt(output)
    wav = output / "audio" / "binaural" / "mixture.wav"
    _write_float32_wav(wav, fact_frame_count=3)
    assert TOOL.point_state(output) == "partial"

    # The standard library writer emits PCM, while this renderer writes
    # IEEE-float32; a merely decodable PCM file must not count as complete.
    pcm = tmp_path / "pcm"
    _write_audio_receipt(pcm)
    pcm_wav = pcm / "audio" / "binaural" / "mixture.wav"
    fmt = struct.pack("<HHIIHH", 1, 2, 16_000, 16_000 * 8, 8, 32)
    data = b"\0" * (4 * 2 * 4)
    chunks = b"fmt " + struct.pack("<I", 16) + fmt
    chunks += b"data" + struct.pack("<I", len(data)) + data
    pcm_wav.parent.mkdir(parents=True, exist_ok=True)
    pcm_wav.write_bytes(
        b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks
    )
    assert TOOL.point_state(pcm) == "partial"




def test_point_state_requires_requested_layout_and_shared_clock(
    tmp_path: Path,
) -> None:
    output = tmp_path / "layouts"
    _write_audio_receipt(output)
    _write_float32_wav(output / "audio" / "binaural" / "mixture.wav")
    assert TOOL.point_state(output, layouts="binaural,ambisonics") == "partial"

    _write_float32_wav(
        output / "audio" / "foa" / "mixture.wav", channels=4
    )
    assert TOOL.point_state(output, layouts=("binaural", "ambisonics")) == "complete"

    mismatched = tmp_path / "cross_layout_clock"
    _write_audio_receipt(mismatched)
    _write_float32_wav(mismatched / "audio" / "binaural" / "mixture.wav")
    _write_float32_wav(
        mismatched / "audio" / "foa" / "mixture.wav",
        channels=4,
        frame_count=3,
    )
    assert TOOL.point_state(
        mismatched, layouts=("binaural", "ambisonics")
    ) == "partial"


def test_point_state_uses_multi_layout_receipt_declaration_by_default(
    tmp_path: Path,
) -> None:
    output = tmp_path / "declared-layouts"
    _write_audio_receipt(output)
    receipt = json.loads((output / "research_receipt.json").read_text())
    receipt["audio"]["layouts"] = ["binaural", "ambisonics"]
    receipt["audio"]["layout_type"] = "binaural"
    (output / "research_receipt.json").write_text(json.dumps(receipt))
    _write_float32_wav(output / "audio" / "binaural" / "mixture.wav")
    assert TOOL.point_state(output) == "partial"
    _write_float32_wav(output / "audio" / "foa" / "mixture.wav", channels=4)
    assert TOOL.point_state(output) == "complete"


def test_layout_normalization_rejects_unknown_and_duplicate_values() -> None:
    assert TOOL.normalize_layouts("binaural,ambisonics") == (
        "binaural",
        "ambisonics",
    )
    with pytest.raises(ValueError, match="unsupported layout"):
        TOOL.normalize_layouts("binaural,foa")
    with pytest.raises(ValueError, match="duplicate layout"):
        TOOL.normalize_layouts(["binaural", "binaural"])

def test_program_path_prefers_point_local_and_fact_declared_programs(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "inputs"
    programs = inputs / "programs"
    point = inputs / "card8_001"
    programs.mkdir(parents=True)
    point.mkdir()
    local_main = point / "audio_program.json"
    local_gatea = point / "audio_program_gateA.json"
    local_main.write_text("{}")
    local_gatea.write_text("{}")
    assert TOOL.program_path(
        programs, point.name, "main", inputs_root=inputs
    ) == local_main
    assert TOOL.program_path(
        programs, point.name, "gateA", inputs_root=inputs
    ) == local_gatea

    local_main.unlink()
    local_gatea.unlink()
    declared_main = point / "declared_main.json"
    declared_gatea = point / "declared_gatea.json"
    declared_main.write_text("{}")
    declared_gatea.write_text("{}")
    (point / "fact_record.json").write_text(json.dumps({
        "audio": {"main_program": declared_main.name}
    }))
    (point / "fact_record_gateA.json").write_text(json.dumps({
        "audio": {"program": declared_gatea.name}
    }))
    assert TOOL.program_path(
        programs, point.name, "main", inputs_root=inputs
    ) == declared_main
    assert TOOL.program_path(
        programs, point.name, "gateA", inputs_root=inputs
    ) == declared_gatea


def test_missing_fact_declared_program_fails_before_render(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    programs = inputs / "programs"
    point = inputs / "card8_001"
    programs.mkdir(parents=True)
    point.mkdir()
    (point / "fact_record.json").write_text(json.dumps({
        "audio": {"program": "missing.json"}
    }))
    with pytest.raises(SystemExit, match="declared audio program is missing"):
        TOOL.program_path(programs, point.name, "main", inputs_root=inputs)

    (point / "fact_record.json").write_text(json.dumps({
        "audio": {"program_id": "missing_id"}
    }))
    with pytest.raises(SystemExit, match="declared audio program id is missing"):
        TOOL.program_path(programs, point.name, "main", inputs_root=inputs)


def test_endpoint_registry_prefers_point_local_file(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    point = inputs / "card8_001"
    point.mkdir(parents=True)
    local = point / "source_endpoints.json"
    local.write_text("{}")
    global_registry = tmp_path / "global_endpoints.json"
    global_registry.write_text("{}")
    assert TOOL.endpoint_registry_path(
        inputs, point.name, global_registry
    ) == local


def test_sound_asset_map_and_legacy_bindings_are_optional(tmp_path: Path) -> None:
    mapping = tmp_path / "sound_asset_map.json"
    mapping.write_text("{}")
    args = TOOL.sound_asset_args(
        {
            "sound_asset_map": "sound_asset_map.json",
            "sound_asset_paths": {"speaker_yellow": "assets/yellow.wav"},
            "beagle_audio": "/legacy/beagle.wav",
        },
        config_path=tmp_path / "batch.json",
    )
    assert args == [
        "--sound-asset-map", str(mapping),
        "--sound-asset-path", "speaker_yellow=assets/yellow.wav",
        "--beagle-audio", "/legacy/beagle.wav",
    ]
    assert TOOL.sound_asset_args({}, config_path=tmp_path / "batch.json") == []


def test_config_repo_must_be_current_avengine_source(tmp_path: Path) -> None:
    config_path = tmp_path / "batch.json"
    assert TOOL.validate_config_repo(
        {"repo": str(TOOL.AVENGINE_REPOSITORY)}, config_path=config_path
    ) == TOOL.AVENGINE_REPOSITORY
    with pytest.raises(SystemExit, match="current AVEngine repository"):
        TOOL.validate_config_repo({"repo": str(tmp_path)}, config_path=config_path)



def test_main_uses_current_repo_point_bindings_and_sound_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = tmp_path / "inputs"
    point = inputs / "card8_001"
    point.mkdir(parents=True)
    (point / "timeline.json").write_text("{}")
    local_program = point / "audio_program.json"
    local_program.write_text("{}")
    local_gatea = point / "audio_program_gateA.json"
    local_gatea.write_text("{}")
    local_endpoints = point / "source_endpoints.json"
    local_endpoints.write_text("{}")
    (point / "actor_selection.json").write_text("{}")
    captures = tmp_path / "captures" / point.name
    captures.mkdir(parents=True)
    (captures / "research_receipt.json").write_text(
        json.dumps({"status": "research_only"})
    )
    m1 = tmp_path / "m1.json"
    m1.write_text("{}")
    global_endpoints = tmp_path / "global_endpoints.json"
    global_endpoints.write_text("{}")
    sound_map = tmp_path / "sound_asset_map.json"
    sound_map.write_text("{}")
    config_path = tmp_path / "batch.json"
    config = {
        "python": sys.executable,
        "repo": str(TOOL.AVENGINE_REPOSITORY),
        "m1_request": str(m1),
        "simulation_request": "simulation.json",
        "package_manifest": "packages.json",
        "source_endpoint_registry": str(global_endpoints),
        "sound_asset_registry": "sounds.json",
        "hrtf": "hrtf.bin",
        "runtime_prefix": "runtime",
        "rlr_sdk_root": "rlr",
        "magnum_python_site": "magnum",
        "source_asset_registry": "assets.json",
        "sound_asset_map": sound_map.name,
        "layouts": ["binaural", "ambisonics"],
    }
    config_path.write_text(json.dumps(config))
    output_root = tmp_path / "outputs"
    calls: list[dict] = []

    def fake_run(cmd, *, stdout, stderr, cwd):
        calls.append({"cmd": cmd, "cwd": cwd})
        output = Path(cmd[cmd.index("--output") + 1])
        execution_variant = cmd[cmd.index("--execution-variant") + 1]
        _write_audio_receipt(output, execution_variant=execution_variant)
        layout_text = cmd[cmd.index("--layouts") + 1]
        for layout in layout_text.split(","):
            _write_float32_wav(
                output / "audio" / ("foa" if layout == "ambisonics" else layout)
                / "mixture.wav",
                channels=4 if layout == "ambisonics" else 2,
            )
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(TOOL.subprocess, "run", fake_run)
    result = TOOL.main([
        "--inputs-root", str(inputs),
        "--captures-root", str(captures.parent),
        "--output-root", str(output_root),
        "--config", str(config_path),
        "--points", point.name,
        "--variants", "main,gateA",
    ])
    assert result == 0
    assert len(calls) == 2
    command = calls[0]["cmd"]
    assert calls[0]["cwd"] == str(TOOL.AVENGINE_REPOSITORY)
    assert command[1] == str(
        TOOL.AVENGINE_REPOSITORY /
        "tools/dataset/render_current_apartment_dynamic_audio.py"
    )
    assert command[command.index("--audio-program") + 1] == str(local_program)
    assert command[command.index("--source-endpoint-registry") + 1] == str(
        local_endpoints
    )
    assert command[command.index("--variant") + 1] == "A"
    assert command[command.index("--execution-variant") + 1] == "main"
    assert command[command.index("--layouts") + 1] == "binaural,ambisonics"
    assert command[command.index("--hrtf") + 1] == "hrtf.bin"
    assert command[command.index("--sound-asset-map") + 1] == str(sound_map)
    assert "--beagle-audio" not in command

    gate_command = calls[1]["cmd"]
    assert gate_command[gate_command.index("--audio-program") + 1] == str(local_gatea)
    assert gate_command[gate_command.index("--variant") + 1] == "A"
    assert gate_command[gate_command.index("--execution-variant") + 1] == "gateA"

    # Ambisonics-only output does not require HRTF and must not pass it to the
    # renderer, even when the prior config carried a stale HRTF value.
    config.pop("hrtf")
    config["layouts"] = ["ambisonics"]
    config_path.write_text(json.dumps(config))
    calls.clear()
    ambisonics_root = tmp_path / "ambisonics-only"
    result = TOOL.main([
        "--inputs-root", str(inputs),
        "--captures-root", str(captures.parent),
        "--output-root", str(ambisonics_root),
        "--config", str(config_path),
        "--points", point.name,
        "--variants", "main",
    ])
    assert result == 0
    assert len(calls) == 1
    ambisonics_command = calls[0]["cmd"]
    assert ambisonics_command[ambisonics_command.index("--layouts") + 1] == "ambisonics"
    assert "--hrtf" not in ambisonics_command


def test_point_local_m1_and_endpoints_need_no_batch_fallback(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    point = inputs / "dynamic_point"
    point.mkdir(parents=True)
    local_m1 = point / "m1_capture_request.json"
    local_endpoints = point / "source_endpoints.json"
    local_m1.write_text("{}")
    local_endpoints.write_text("{}")
    assert TOOL.point_m1_request(inputs, point.name) == local_m1.resolve()
    assert TOOL.endpoint_registry_path(inputs, point.name) == local_endpoints.resolve()


def test_missing_point_local_m1_and_endpoints_fail_without_fallback(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    with pytest.raises(SystemExit, match="no point-local|fallback <none>"):
        TOOL.point_m1_request(inputs, "missing")
    with pytest.raises(SystemExit, match="fallback <none>"):
        TOOL.endpoint_registry_path(inputs, "missing")


def test_relative_legacy_fallbacks_resolve_from_batch_config(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    config_root = tmp_path / "config"
    config_root.mkdir()
    m1 = config_root / "m1.json"
    endpoints = config_root / "endpoints.json"
    m1.write_text("{}")
    endpoints.write_text("{}")
    assert TOOL.point_m1_request(
        inputs, "legacy", "m1.json", config_base=config_root
    ) == m1.resolve()
    assert TOOL.endpoint_registry_path(
        inputs, "legacy", "endpoints.json", config_base=config_root
    ) == endpoints.resolve()
