#!/usr/bin/env python3
"""Sequential dynamic-audio runner for a qa-v3 design batch (stage two).

对设计批每个点位逐点调用 render_current_apartment_dynamic_audio.py。
固定共享依赖从 --config JSON 读并进入批次记录；每点渲染 main，
并按请求渲染 Gate A 等音频变体。

续跑语义与捕获调度器一致：
  - 完成点由 pass receipt 和实际 WAV 的格式/时钟共同确认后跳过；
  - 半成品拒绝自动清理或覆盖；
  - 任一点失败即停止。
输出根必须 fresh；续跑需显式 --resume。全链为 research_candidate。

config JSON 必备键:
  python, repo, simulation_request, package_manifest, sound_asset_registry,
  hrtf, runtime_prefix, rlr_sdk_root, magnum_python_site,
  source_asset_registry
每个候选优先读取自己的 m1_capture_request.json 和 source_endpoints.json；
仅旧候选需要可选的 m1_request/source_endpoint_registry 批级后备。
可选 sound_asset_map、sound_asset_paths；beagle_audio 仅为旧版兼容绑定。
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

REQUIRED_CONFIG_KEYS = (
    "python", "repo", "simulation_request", "package_manifest",
    "sound_asset_registry", "hrtf", "runtime_prefix", "rlr_sdk_root",
    "magnum_python_site", "source_asset_registry",
)
AVENGINE_REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qa_v3_request import batch_point_ids  # noqa: E402


def _read_float32_wav_metadata(path: Path) -> dict[str, int]:
    """Read RIFF/WAVE metadata without decoding the sample payload."""
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read WAV {path}: {error}") from error
    if len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a RIFF/WAVE file")
    fmt = None
    data_size = None
    fact_frames = None
    offset = 12
    while offset + 8 <= len(payload):
        chunk_id = payload[offset:offset + 4]
        size = struct.unpack_from("<I", payload, offset + 4)[0]
        start = offset + 8
        end = start + size
        if end > len(payload):
            raise ValueError(f"{path} contains a truncated WAVE chunk")
        chunk = payload[start:end]
        if chunk_id == b"fmt " and size >= 16:
            fmt = struct.unpack_from("<HHIIHH", chunk, 0)
        elif chunk_id == b"fact" and size >= 4:
            fact_frames = struct.unpack_from("<I", chunk, 0)[0]
        elif chunk_id == b"data":
            data_size = int(size)
        offset = end + (size & 1)
    if fmt is None or data_size is None:
        raise ValueError(f"{path} lacks WAVE fmt/data metadata")
    format_tag, channels, sample_rate, byte_rate, block_align, bits = fmt
    if (
        format_tag != 3
        or channels < 1
        or sample_rate < 1
        or bits != 32
        or block_align != channels * 4
        or byte_rate != sample_rate * block_align
        or data_size % block_align != 0
    ):
        raise ValueError(f"{path} is not canonical IEEE-float32 WAVE")
    frames = data_size // block_align
    if fact_frames is not None and fact_frames != frames:
        raise ValueError(f"{path} fact frame count differs from data chunk")
    return {
        "channel_count": int(channels),
        "sample_rate_hz": int(sample_rate),
        "frame_count": int(frames),
    }


def _receipt_audio_expectations(out_dir: Path) -> tuple[int, int] | None:
    receipt_path = out_dir / "research_receipt.json"
    mixture = out_dir / "audio" / "binaural" / "mixture.wav"
    if not receipt_path.is_file() or not mixture.is_file():
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(receipt, dict) or receipt.get("status") != "pass":
        return None
    audio = receipt.get("audio")
    if not isinstance(audio, dict):
        return None
    rate = audio.get("sample_rate_hz")
    count = audio.get("sample_count")
    if (
        isinstance(rate, bool)
        or not isinstance(rate, int)
        or rate < 1
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
    ):
        return None
    return rate, count


def point_state(out_dir: Path) -> str:
    if not out_dir.exists():
        return "missing"
    expected = _receipt_audio_expectations(out_dir)
    if expected is None:
        return "partial"
    expected_rate, expected_frames = expected
    wav_files = sorted((out_dir / "audio").rglob("*.wav"))
    if not wav_files:
        return "partial"
    for wav_path in wav_files:
        try:
            metadata = _read_float32_wav_metadata(wav_path)
        except ValueError:
            return "partial"
        expected_channels = 2 if "binaural" in wav_path.parts else 1
        if (
            metadata["channel_count"] != expected_channels
            or metadata["sample_rate_hz"] != expected_rate
            or metadata["frame_count"] != expected_frames
        ):
            return "partial"
    return "complete"




def _resolve_declared_path(value: str | Path, *, point_dir: Path, programs_dir: Path) -> Path:
    path = Path(value).expanduser()
    candidates = (
        [path]
        if path.is_absolute()
        else [point_dir / path, programs_dir / path]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise SystemExit(
        f"FAIL: declared audio program is missing for {point_dir.name}: "
        + " or ".join(str(candidate) for candidate in candidates)
    )


def program_path(
    programs_dir: Path,
    pid: str,
    variant: str,
    *,
    inputs_root: Path | None = None,
) -> Path:
    """Resolve a point-local or fact-declared program before legacy names."""
    point_dir = (
        Path(inputs_root) / pid if inputs_root is not None else programs_dir.parent / pid
    )
    local_name = "audio_program.json" if variant == "main" else "audio_program_gateA.json"
    local = point_dir / local_name
    if local.is_file():
        return local.resolve()

    fact_path = point_dir / (
        "fact_record.json" if variant == "main" else "fact_record_gateA.json"
    )
    if fact_path.is_file():
        try:
            fact = json.loads(fact_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise SystemExit(f"FAIL: cannot read {fact_path}: {error}") from error
        if isinstance(fact, dict):
            audio = fact.get("audio")
            owners = [audio, fact] if isinstance(audio, dict) else [fact]
            declared_keys = (
                ("program", "program_path", "main_program", "audio_program")
                if variant == "main" else
                ("program", "program_path", "gatea_program", "gateA_program",
                 "audio_program_gateA")
            )
            for owner in owners:
                if not isinstance(owner, dict):
                    continue
                for key in declared_keys:
                    declared = owner.get(key)
                    if isinstance(declared, (str, Path)):
                        return _resolve_declared_path(
                            declared, point_dir=point_dir, programs_dir=programs_dir
                        )
                    if isinstance(declared, dict):
                        declared_path = (
                            declared.get("path")
                            or declared.get("program_path")
                            or declared.get("file")
                        )
                        if isinstance(declared_path, (str, Path)):
                            return _resolve_declared_path(
                                declared_path,
                                point_dir=point_dir,
                                programs_dir=programs_dir,
                            )
                        declared_id = declared.get("program_id") or declared.get("id")
                        if isinstance(declared_id, str) and declared_id:
                            by_id = programs_dir / f"{declared_id}.json"
                            if by_id.is_file():
                                return by_id.resolve()
                            raise SystemExit(
                                f"FAIL: declared audio program id is missing for "
                                f"{point_dir.name}: {by_id}"
                            )
                program_id = owner.get("program_id")
                if isinstance(program_id, str) and program_id:
                    by_id = programs_dir / f"{program_id}.json"
                    if by_id.is_file():
                        return by_id.resolve()
                    raise SystemExit(
                        f"FAIL: declared audio program id is missing for "
                        f"{point_dir.name}: {by_id}"
                    )

    suffixes = {
        "main": "_rand_v1.json",
        "gateA": "_rand_gateA_v1.json",
    }
    suffix = suffixes.get(variant, f"_rand_{variant}_v1.json")
    matches = sorted(programs_dir.glob(f"qa_v3_*_{pid}{suffix}"))
    if len(matches) != 1:
        raise SystemExit(
            f"FAIL: expected exactly one {variant} program for {pid}, "
            f"found {len(matches)} in {programs_dir}"
        )
    return matches[0].resolve()


def _fallback_path(
    configured: str | Path | None,
    *,
    config_base: Path | None,
) -> Path | None:
    if configured is None:
        return None
    configured_path = Path(configured).expanduser()
    if not configured_path.is_absolute() and config_base is not None:
        configured_path = config_base / configured_path
    return configured_path.resolve()


def endpoint_registry_path(
    inputs_root: Path,
    pid: str,
    configured: str | Path | None = None,
    *,
    config_base: Path | None = None,
) -> Path:
    point_local = inputs_root / pid / "source_endpoints.json"
    if point_local.is_file():
        return point_local.resolve()
    configured_path = _fallback_path(configured, config_base=config_base)
    if configured_path is None or not configured_path.is_file():
        fallback = "<none>" if configured_path is None else str(configured_path)
        raise SystemExit(
            f"FAIL: source endpoint registry is missing for {pid}; "
            f"checked point-local source_endpoints.json and fallback {fallback}"
        )
    return configured_path


def sound_asset_args(config: dict, *, config_path: Path) -> list[str]:
    """Build renderer sound bindings from the optional map plus legacy fallback."""
    result: list[str] = []
    mapping = config.get("sound_asset_map")
    if mapping is not None:
        if not isinstance(mapping, (str, Path)):
            raise SystemExit("FAIL: sound_asset_map must be a JSON file path")
        mapping_path = Path(mapping).expanduser()
        if not mapping_path.is_absolute():
            mapping_path = config_path.parent / mapping_path
        if not mapping_path.is_file():
            raise SystemExit(f"FAIL: sound_asset_map is missing: {mapping_path}")
        result.extend(["--sound-asset-map", str(mapping_path.resolve())])
    assignments = config.get("sound_asset_paths", ())
    if isinstance(assignments, dict):
        if any(
            not isinstance(sound_id, str) or not sound_id
            or not isinstance(path, str) or not path
            for sound_id, path in assignments.items()
        ):
            raise SystemExit(
                "FAIL: sound_asset_paths must be SOUND_ASSET_ID=PATH strings"
            )
        assignments = [
            f"{sound_id}={path}"
            for sound_id, path in assignments.items()
        ]
    if assignments:
        if not isinstance(assignments, list) or any(
            not isinstance(value, str)
            or not value.partition("=")[0]
            or not value.partition("=")[1]
            or not value.partition("=")[2]
            for value in assignments
        ):
            raise SystemExit(
                "FAIL: sound_asset_paths must be SOUND_ASSET_ID=PATH strings"
            )
        for assignment in assignments:
            result.extend(["--sound-asset-path", assignment])
    legacy = config.get("beagle_audio")
    if legacy is not None:
        result.extend(["--beagle-audio", str(legacy)])
    return result


def validate_config_repo(config: dict, *, config_path: Path) -> Path:
    value = config.get("repo")
    if not isinstance(value, str) or not value:
        raise SystemExit("FAIL: config repo must be a nonempty path")
    configured = Path(value).expanduser()
    if not configured.is_absolute():
        configured = config_path.parent / configured
    configured = configured.resolve()
    if configured != AVENGINE_REPOSITORY:
        raise SystemExit(
            "FAIL: config repo must point to the current AVEngine repository "
            f"{AVENGINE_REPOSITORY}, got {configured}"
        )
    return AVENGINE_REPOSITORY


def point_m1_request(
    inputs_root: Path,
    pid: str,
    configured: str | Path | None = None,
    *,
    config_base: Path | None = None,
) -> Path:
    per_point = inputs_root / pid / "m1_capture_request.json"
    if per_point.is_file():
        return per_point.resolve()
    configured_path = _fallback_path(configured, config_base=config_base)
    if configured_path is None or not configured_path.is_file():
        fallback = "<none>" if configured_path is None else str(configured_path)
        raise SystemExit(
            f"FAIL: M1 request is missing for {pid}; "
            f"checked point-local m1_capture_request.json and fallback {fallback}"
        )
    return configured_path


def canonical_emitter_args(config: dict) -> list[str]:
    """Translate one explicit QA semantic-anchor policy into renderer args."""
    value = config.get("canonical_emitter_height_m")
    if value is None:
        return []
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise SystemExit(
            "FAIL: canonical_emitter_height_m must be finite and positive")
    return ["--canonical-emitter-height-m", str(value)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs-root", required=True, type=Path)
    parser.add_argument("--captures-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--variants", default="main",
                        help="逗号分隔:main[,gateA]")
    parser.add_argument("--points", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        print(f"config missing keys: {missing}", file=sys.stderr)
        return 2
    try:
        repo = validate_config_repo(cfg, config_path=config_path)
        sound_args = sound_asset_args(cfg, config_path=config_path)
    except SystemExit as error:
        print(str(error), file=sys.stderr)
        return 2
    if args.output_root.exists() and not args.resume:
        print(f"output root exists; pass --resume to continue: {args.output_root}",
              file=sys.stderr)
        return 2
    inputs_root = args.inputs_root.resolve()
    captures_root = args.captures_root.resolve()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    try:
        points = batch_point_ids(
            inputs_root, args.points.split(",") if args.points else None)
    except (OSError, ValueError, TypeError) as error:
        print(f"FAIL: cannot select design batch points: {error}", file=sys.stderr)
        return 2
    args.output_root.mkdir(parents=True, exist_ok=True)
    programs_dir = inputs_root / "programs"
    done = skipped = 0
    for pid in points:
        cap_dir = captures_root / pid
        capture_receipt = cap_dir / "research_receipt.json"
        if not capture_receipt.is_file():
            print(f"FAIL: capture for {pid} not complete at {cap_dir}",
                  file=sys.stderr)
            return 1
        try:
            capture_value = json.loads(
                capture_receipt.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            print(f"FAIL: capture receipt for {pid} is unreadable: {error}",
                  file=sys.stderr)
            return 1
        if (
            not isinstance(capture_value, dict)
            or capture_value.get("status") not in {"research_only", "pass"}
        ):
            print(
                f"FAIL: capture for {pid} has no successful receipt status",
                file=sys.stderr,
            )
            return 1
        # Gate B 孪生只渲 main(Gate A 换音频的对照对孪生无意义);
        # 孪生的 program 由 derive_twin_programs.py 预先派生(外观孪生
        # 的 endpoint 绑定随资产翻转,必须换绑重密封),每点用自己的。
        spec_path = inputs_root / pid / "spec.json"
        spec = json.loads(spec_path.read_text()) if spec_path.is_file() else {}
        point_variants = ["main"] if spec.get("twin_of") else variants
        m1_request = point_m1_request(
            inputs_root, pid, cfg.get("m1_request"),
            config_base=config_path.parent,
        )
        for variant in point_variants:
            out_dir = args.output_root / (pid if variant == "main"
                                          else f"{pid}_{variant}")
            state = point_state(out_dir)
            if state == "complete":
                skipped += 1
                continue
            if state == "partial":
                print(f"FAIL: {out_dir} is a partial render — refusing to clean "
                      f"or overwrite automatically; inspect and remove manually "
                      f"(b007-lesson guard)", file=sys.stderr)
                return 1
            prog = program_path(
                programs_dir, pid, variant, inputs_root=inputs_root
            )
            endpoint_path = endpoint_registry_path(
                inputs_root, pid, cfg.get("source_endpoint_registry"),
                config_base=config_path.parent,
            )
            cmd = [cfg["python"],
                   str(repo / "tools/dataset/render_current_apartment_dynamic_audio.py"),
                   "--visual-capture-dir", str(cap_dir),
                   "--m1-request", str(m1_request),
                   "--simulation-request", cfg["simulation_request"],
                   "--package-manifest", cfg["package_manifest"],
                   "--audio-program", str(prog),
                   "--source-endpoint-registry", str(endpoint_path),
                   "--sound-asset-registry", cfg["sound_asset_registry"],
                   "--hrtf", cfg["hrtf"],
                   "--runtime-prefix", cfg["runtime_prefix"],
                   "--rlr-sdk-root", cfg["rlr_sdk_root"],
                   "--magnum-python-site", cfg["magnum_python_site"],
                   "--actor-selection", str(inputs_root / pid /
                                            "actor_selection.json"),
                   "--source-asset-registry", cfg["source_asset_registry"],
                   "--variant", "A",
                   "--execution-variant", variant,
                   "--output", str(out_dir)]
            cmd.extend(sound_args)
            cmd.extend(canonical_emitter_args(cfg))
            log_path = args.output_root / f"{out_dir.name}.log"
            with open(log_path, "w") as log:
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                      cwd=str(repo))
            if proc.returncode != 0 or point_state(out_dir) != "complete":
                print(f"FAIL: audio render {out_dir.name} failed "
                      f"(exit {proc.returncode}); log: {log_path}",
                      file=sys.stderr)
                return 1
            done += 1
            print(f"ok {out_dir.name} log={log_path.name}")
    print(f"rendered={done} skipped_complete={skipped} "
          f"points={len(points)} variants={variants}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
