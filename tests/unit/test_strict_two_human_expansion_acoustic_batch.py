from __future__ import annotations

import importlib.util
import json
import struct
from copy import deepcopy
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_PATH = (
    REPOSITORY
    / "tools/qa/build_strict_two_human_expansion_acoustic_batch.py"
)
TOOL_SPEC = importlib.util.spec_from_file_location(
    "build_strict_two_human_expansion_acoustic_batch", TOOL_PATH
)
assert TOOL_SPEC is not None and TOOL_SPEC.loader is not None
TOOL = importlib.util.module_from_spec(TOOL_SPEC)
TOOL_SPEC.loader.exec_module(TOOL)

PLAN = REPOSITORY / "examples/qa/native_strict_two_human_expansion_v1.json"
REGISTRY = REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json"
SOURCE_SUITE = (
    REPOSITORY
    / "tmp/lead_a_native_paper_balance_v1/stationary_finalized_gpu1_v3"
    / "suite_execution_plan.json"
)
CONTROLLED_REGISTRY = Path(
    "/data/jzy/code/SPEAR-lead-b/outputs/lead_b/audio_candidates_v1/"
    "controlled_sound_content_registry_v1.json"
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _prepare(tmp_path: Path) -> tuple[Path, dict]:
    preflight_path = TOOL.PREFLIGHT.build(PLAN, tmp_path / "strict8_preflight")
    output = tmp_path / "strict8_acoustic"
    manifest_path = TOOL.prepare(
        plan_path=PLAN,
        cpu_preflight_path=preflight_path,
        registry_path=REGISTRY,
        source_suite_path=SOURCE_SUITE,
        controlled_registry_path=CONTROLLED_REGISTRY,
        output=output,
    )
    return output, _load(manifest_path)


def test_full_dry_window_rejects_silent_fifteen_or_overflow() -> None:
    assert TOOL._speech_window(start_sample=7467, source_sample_count=25626) == (
        7467,
        33093,
        [7, 31],
    )
    assert TOOL._speech_window(start_sample=7467, source_sample_count=45912) == (
        7467,
        53379,
        [7, 50],
    )
    with pytest.raises(RuntimeError, match="exceeds five seconds"):
        TOOL._speech_window(start_sample=7467, source_sample_count=80000)


def test_wave_header_accepts_rendered_float32_contract(tmp_path: Path) -> None:
    channels = 2
    sample_rate_hz = 16000
    sample_count = 80000
    bits_per_sample = 32
    block_align = channels * bits_per_sample // 8
    data = b"\x00" * (sample_count * block_align)
    fmt = struct.pack(
        "<HHIIHH",
        3,
        channels,
        sample_rate_hz,
        sample_rate_hz * block_align,
        block_align,
        bits_per_sample,
    )
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(data)) + data
    wav_path = tmp_path / "float32.wav"
    wav_path.write_bytes(b"RIFF" + struct.pack("<I", len(body) + 4) + b"WAVE" + body)

    assert TOOL._wave_header(wav_path) == {
        "format_tag": 3,
        "channel_count": 2,
        "sample_rate_hz": 16000,
        "sample_count": 80000,
    }


def test_prepare_rejects_identity_speech_window_drift(tmp_path: Path) -> None:
    plan = _load(PLAN)
    invalid = deepcopy(plan)
    invalid["approved_identity_catalog"]["F"][
        "expected_speech_frame_window_inclusive"
    ] = [7, 31]
    invalid_plan = tmp_path / "invalid_plan.json"
    invalid_plan.write_text(json.dumps(invalid), encoding="utf-8")
    preflight = {
        "status": "pass_cpu_plan_pending_exact_rir_and_seven_sparse_native_gates",
        "row_count": 8,
    }
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")

    with pytest.raises(RuntimeError, match="identity F speech window mismatch"):
        TOOL.prepare(
            plan_path=invalid_plan,
            cpu_preflight_path=preflight_path,
            registry_path=REGISTRY,
            source_suite_path=SOURCE_SUITE,
            controlled_registry_path=CONTROLLED_REGISTRY,
            output=tmp_path / "invalid_output",
        )

_RETAINED_TMP_WORKSPACE = Path(__file__).resolve().parents[2] / "tmp"
# Guarding on tmp/ existing was wrong: running the engine in a
# checkout creates tmp/spear_instance_*, which made this look
# mounted and sent 49 tests into a run without their data.  The
# evidence mount signature is a lead_* workspace.
if not any(_RETAINED_TMP_WORKSPACE.glob("lead_*")):
    pytest.skip(
        "no lead_* evidence workspace under the repository tmp "
        "directory, so this checkout does not carry the retained "
        "strict-two-human evidence",
        allow_module_level=True,
    )
