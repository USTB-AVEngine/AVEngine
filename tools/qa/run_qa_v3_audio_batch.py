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
from collections.abc import Mapping, Sequence
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

REQUIRED_CONFIG_KEYS = (
    "python", "repo", "simulation_request", "package_manifest",
    "sound_asset_registry", "runtime_prefix", "rlr_sdk_root",
    "magnum_python_site", "source_asset_registry",
)
AVENGINE_REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qa_v3_request import batch_point_ids  # noqa: E402

AUDIO_LAYOUTS = {
    "binaural": {
        "directory": "binaural",
        "channel_count": 2,
        "channel_labels": ("left", "right"),
    },
    "ambisonics": {
        "directory": "foa",
        "channel_count": 4,
        "channel_labels": ("W", "Y", "Z", "X"),
    },
}
DEFAULT_LAYOUTS = ("binaural",)
AUXILIARY_AUDIO_CHANNELS = {"dry": 1}


def normalize_layouts(value: object, *, owner: str = "layouts") -> tuple[str, ...]:
    """Normalize the explicit output-layout list used by the renderer."""
    if isinstance(value, str):
        raw = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raise ValueError(f"{owner} must be a comma-separated string or list")
    if not raw:
        raise ValueError(f"{owner} must contain at least one layout")
    result: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item or item != item.strip():
            raise ValueError(f"{owner} contains an invalid layout")
        if item not in AUDIO_LAYOUTS:
            raise ValueError(
                f"{owner} contains unsupported layout {item!r}; "
                f"expected one of {sorted(AUDIO_LAYOUTS)}"
            )
        if item in result:
            raise ValueError(f"{owner} contains duplicate layout {item!r}")
        result.append(item)
    return tuple(result)


def _load_audio_receipt(out_dir: Path) -> dict | None:
    receipt_path = out_dir / "research_receipt.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return receipt if isinstance(receipt, dict) else None


def _clock_values(value: object) -> list[tuple[int, int]]:
    """Collect direct and per-layout sample clocks from a receipt object."""
    if not isinstance(value, dict):
        return []
    values: list[tuple[int, int]] = []
    rate = value.get("sample_rate_hz")
    count = value.get("sample_count")
    if "sample_rate_hz" in value or "sample_count" in value:
        if not (
            isinstance(rate, int) and not isinstance(rate, bool) and rate > 0
            and isinstance(count, int) and not isinstance(count, bool) and count > 0
        ):
            raise ValueError("receipt audio clock must contain positive integer sample_rate_hz and sample_count")
        values.append((rate, count))
    for key in ("binaural", "ambisonics", "foa"):
        nested = value.get(key)
        if isinstance(nested, dict):
            values.extend(_clock_values(nested))
    layouts = value.get("layouts")
    if isinstance(layouts, dict):
        for nested in layouts.values():
            values.extend(_clock_values(nested))
    elif isinstance(layouts, list):
        for nested in layouts:
            values.extend(_clock_values(nested))
    return values


def _declared_layouts(receipt: dict | None) -> tuple[str, ...] | None:
    if not isinstance(receipt, dict):
        return None
    for key in ("expected_layouts", "layouts", "audio_layouts"):
        value = receipt.get(key)
        if value is not None:
            return normalize_layouts(value, owner=f"receipt.{key}")
    audio = receipt.get("audio")
    if isinstance(audio, dict):
        # New multi-layout receipts put the authoritative list under
        # audio.layouts while retaining layout_type for old readers.
        declared = audio.get("layouts")
        if declared is not None:
            return normalize_layouts(declared, owner="receipt.audio.layouts")
        layout_type = audio.get("layout_type")
        if layout_type == "binaural":
            return ("binaural",)
        if layout_type in {"ambisonics", "foa"}:
            return ("ambisonics",)
        present = tuple(
            name for name in ("binaural", "ambisonics")
            if isinstance(audio.get(name), dict)
        )
        if present:
            return present
    return None



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
    receipt = _load_audio_receipt(out_dir)
    if receipt is None or receipt.get("status") != "pass":
        return None
    values = _clock_values(receipt.get("audio"))
    program = receipt.get("audio_program")
    if isinstance(program, dict):
        values.extend(_clock_values(program.get("timeline")))
    if not values:
        values = _clock_values(receipt)
    if not values or len(set(values)) != 1:
        return None
    return values[0]


def point_state(out_dir: Path, *, layouts: object | None = None) -> str:
    if not out_dir.exists():
        return "missing"
    receipt = _load_audio_receipt(out_dir)
    try:
        declared_layouts = _declared_layouts(receipt)
        requested_layouts = layouts
        if requested_layouts is None:
            requested_layouts = declared_layouts or DEFAULT_LAYOUTS
        expected_layouts = normalize_layouts(requested_layouts)
        if (
            layouts is not None
            and declared_layouts is not None
            and declared_layouts != expected_layouts
        ):
            return "partial"
        expected = _receipt_audio_expectations(out_dir)
    except ValueError:
        return "partial"
    if expected is None:
        return "partial"
    expected_rate, expected_frames = expected
    audio_root = out_dir / "audio"
    if not audio_root.is_dir():
        return "partial"
    for layout in expected_layouts:
        layout_root = audio_root / AUDIO_LAYOUTS[layout]["directory"]
        if not (layout_root / "mixture.wav").is_file():
            return "partial"
        if not any(layout_root.rglob("*.wav")):
            return "partial"
    wav_files = sorted(audio_root.rglob("*.wav"))
    if not wav_files:
        return "partial"
    for wav_path in wav_files:
        try:
            metadata = _read_float32_wav_metadata(wav_path)
        except ValueError:
            return "partial"
        relative_parts = wav_path.relative_to(audio_root).parts
        layout_name = next(
            (
                name for name, spec in AUDIO_LAYOUTS.items()
                if spec["directory"] in relative_parts
            ),
            None,
        )
        if layout_name is not None:
            expected_channels = AUDIO_LAYOUTS[layout_name]["channel_count"]
        else:
            if not relative_parts or relative_parts[0] not in AUXILIARY_AUDIO_CHANNELS:
                return "partial"
            expected_channels = AUXILIARY_AUDIO_CHANNELS[relative_parts[0]]
        if (
            metadata["channel_count"] != expected_channels
            or metadata["sample_rate_hz"] != expected_rate
            or metadata["frame_count"] != expected_frames
        ):
            return "partial"
    return "complete"




def _safe_variant_name(value: object, *, owner: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(
            f"{owner} must be a single non-empty variant without path separators "
            "or control characters"
        )
    return value


def normalize_variants(
    value: object, *, owner: str = "variants"
) -> tuple[str, ...]:
    """Normalize requested execution variants while keeping names data-driven."""
    if isinstance(value, str):
        raw = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = list(value)
    else:
        raise ValueError(f"{owner} must be a comma-separated string or list")
    if not raw:
        raise ValueError(f"{owner} must contain at least one variant")
    result: list[str] = []
    for index, item in enumerate(raw):
        variant = _safe_variant_name(item, owner=f"{owner}[{index}]")
        if variant in result:
            raise ValueError(f"{owner} contains duplicate variant {variant!r}")
        result.append(variant)
    if "main" not in result:
        raise ValueError(f"{owner} must include main")
    return tuple(result)


def _variant_program_filename(variant: str) -> str:
    variant = _safe_variant_name(variant, owner="audio variant")
    return (
        "audio_program.json"
        if variant == "main"
        else f"audio_program_{variant}.json"
    )


def _variant_fact_filename(variant: str) -> str:
    variant = _safe_variant_name(variant, owner="audio variant")
    return (
        "fact_record.json"
        if variant == "main"
        else f"fact_record_{variant}.json"
    )


def _output_directory_name(point_id: str, variant: str) -> str:
    variant = _safe_variant_name(variant, owner="audio variant")
    return point_id if variant == "main" else f"{point_id}_{variant}"


def _validate_output_directory_names(
    point_variants: Mapping[str, Sequence[str]],
) -> None:
    """Reject point/variant output-name collisions before launching a render."""
    owners: dict[str, tuple[str, str]] = {}
    for point_id, variants in point_variants.items():
        if not isinstance(point_id, str) or not point_id or Path(point_id).name != point_id:
            raise ValueError(f"invalid point ID for audio output: {point_id!r}")
        for variant in variants:
            output_name = _output_directory_name(point_id, variant)
            previous = owners.get(output_name)
            if previous is not None:
                raise ValueError(
                    "audio output directory collision: "
                    f"{output_name!r} is produced by "
                    f"{previous[0]}/{previous[1]} and {point_id}/{variant}"
                )
            owners[output_name] = (point_id, variant)


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


def _fact_variant_paths(point_dir: Path, variant: str) -> tuple[tuple[Path, bool], ...]:
    variant = _safe_variant_name(variant, owner="audio variant")
    specific = point_dir / _variant_fact_filename(variant)
    if variant == "main":
        return ((specific, True),)
    # A shared main fact remains a compatibility fallback for variants whose
    # program is declared through audio.programs[variant].
    return ((specific, True), (point_dir / "fact_record.json", False))


def _fact_program_declarations(
    fact: Mapping[str, object],
    variant: str,
    *,
    allow_legacy_direct_keys: bool,
    allow_program_id: bool,
) -> list[object]:
    variant = _safe_variant_name(variant, owner="audio variant")
    owners: list[Mapping[str, object]] = []
    audio = fact.get("audio")
    if isinstance(audio, Mapping):
        owners.append(audio)
    owners.append(fact)
    result: list[object] = []
    mapping_keys = (
        "programs",
        "audio_programs",
        "program_by_variant",
        "program_paths",
        "audio_program_by_variant",
    )
    for owner in owners:
        for key in mapping_keys:
            declared = owner.get(key)
            if isinstance(declared, Mapping) and variant in declared:
                result.append(declared[variant])
        for key in ("program", "program_path", "audio_program"):
            declared = owner.get(key)
            if isinstance(declared, Mapping) and variant in declared:
                result.append(declared[variant])
        if allow_legacy_direct_keys:
            if variant == "main":
                keys = ("program", "program_path", "main_program", "audio_program")
            elif variant == "gateA":
                keys = (
                    "program",
                    "program_path",
                    "gatea_program",
                    "gateA_program",
                    "audio_program_gateA",
                )
            else:
                keys = (
                    "program",
                    "program_path",
                    "audio_program",
                    f"program_{variant}",
                    f"program_path_{variant}",
                    f"audio_program_{variant}",
                    f"{variant}_program",
                )
            for key in keys:
                declared = owner.get(key)
                if declared is not None:
                    result.append(declared)
        if allow_program_id:
            program_id = owner.get("program_id")
            if isinstance(program_id, str) and program_id:
                result.append({"program_id": program_id})
    return result


def _declared_program_path(
    declared: object,
    *,
    point_dir: Path,
    programs_dir: Path,
) -> Path | None:
    if isinstance(declared, (str, Path)):
        return _resolve_declared_path(
            declared, point_dir=point_dir, programs_dir=programs_dir
        )
    if not isinstance(declared, Mapping):
        return None
    declared_path = (
        declared.get("path")
        or declared.get("program_path")
        or declared.get("file")
    )
    if isinstance(declared_path, (str, Path)):
        return _resolve_declared_path(
            declared_path, point_dir=point_dir, programs_dir=programs_dir
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
    return None


def program_path(
    programs_dir: Path,
    pid: str,
    variant: str,
    *,
    inputs_root: Path | None = None,
) -> Path:
    """Resolve a point-local or fact-declared program before shared fallbacks."""
    variant = _safe_variant_name(variant, owner="audio variant")
    point_dir = (
        Path(inputs_root) / pid if inputs_root is not None else programs_dir.parent / pid
    )
    local = point_dir / _variant_program_filename(variant)
    if local.is_file():
        return local.resolve()

    for fact_path, variant_specific in _fact_variant_paths(point_dir, variant):
        if not fact_path.is_file():
            continue
        try:
            fact = json.loads(fact_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise SystemExit(f"FAIL: cannot read {fact_path}: {error}") from error
        if not isinstance(fact, Mapping):
            continue
        for declared in _fact_program_declarations(
            fact,
            variant,
            allow_legacy_direct_keys=variant_specific or variant == "main",
            allow_program_id=variant_specific or variant == "main",
        ):
            resolved = _declared_program_path(
                declared, point_dir=point_dir, programs_dir=programs_dir
            )
            if resolved is not None:
                return resolved

    if variant == "main":
        suffixes = ("_rand_v1.json",)
    elif variant == "gateA":
        suffixes = ("_rand_gateA_v1.json", "_gateA_rand_v1.json")
    else:
        suffixes = (
            f"_rand_{variant}_v1.json",
            f"_{variant}_rand_v1.json",
            f"_{variant}.json",
        )
    matches = sorted({
        match
        for suffix in suffixes
        for match in programs_dir.glob(f"qa_v3_*_{pid}{suffix}")
    })
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
    variant: str = "main",
) -> Path:
    variant = _safe_variant_name(variant, owner="audio variant")
    if variant != "main":
        specific = inputs_root / pid / f"m1_capture_request_{variant}.json"
        if specific.is_file():
            return specific.resolve()
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


def hrtf_args(config: dict, layouts: tuple[str, ...]) -> list[str]:
    """Pass HRTF only for a requested binaural render."""
    if "binaural" not in layouts:
        return []
    value = config.get("hrtf")
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(
            "FAIL: config hrtf is required when layouts include binaural"
        )
    return ["--hrtf", value]


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
                        help="逗号分隔执行变体（必须包含 main）")
    parser.add_argument(
        "--layouts",
        default=None,
        help="逗号分隔输出布局: binaural[,ambisonics]；默认 binaural，"
             "也可在 config.layouts 中声明",
    )
    parser.add_argument("--points", default=None)
    parser.add_argument(
        "--capture-by-variant",
        type=Path,
        help="JSON object {point_id: {variant: capture_dir}} for non-main segments",
    )
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
    try:
        variants = normalize_variants(args.variants)
    except ValueError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2
    try:
        layouts = normalize_layouts(
            args.layouts if args.layouts is not None
            else cfg.get("layouts", DEFAULT_LAYOUTS)
        )
    except ValueError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2
    try:
        points = batch_point_ids(
            inputs_root, args.points.split(",") if args.points else None)
    except (OSError, ValueError, TypeError) as error:
        print(f"FAIL: cannot select design batch points: {error}", file=sys.stderr)
        return 2
    point_variants_by_id: dict[str, tuple[str, ...]] = {}
    point_specs: dict[str, dict] = {}
    for pid in points:
        spec_path = inputs_root / pid / "spec.json"
        spec = json.loads(spec_path.read_text()) if spec_path.is_file() else {}
        if not isinstance(spec, dict):
            raise ValueError(f"spec for {pid} must be a JSON object")
        point_specs[pid] = spec
        point_variants_by_id[pid] = ("main",) if spec.get("twin_of") else variants
    try:
        _validate_output_directory_names(point_variants_by_id)
    except ValueError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2
    args.output_root.mkdir(parents=True, exist_ok=True)
    programs_dir = inputs_root / "programs"
    capture_by_variant: dict[str, dict[str, str]] = {}
    if args.capture_by_variant is not None:
        raw_map = json.loads(args.capture_by_variant.read_text(encoding="utf-8"))
        if not isinstance(raw_map, dict):
            print("FAIL: --capture-by-variant must be a JSON object", file=sys.stderr)
            return 2
        for point_id, variants_map in raw_map.items():
            if not isinstance(variants_map, dict):
                print(
                    f"FAIL: capture-by-variant[{point_id}] must be an object",
                    file=sys.stderr,
                )
                return 2
            capture_by_variant[str(point_id)] = {
                str(variant): str(path) for variant, path in variants_map.items()
            }
    done = skipped = 0
    for pid in points:
        default_cap_dir = captures_root / pid
        first_variant = point_variants_by_id[pid][0]
        cap_dir = Path(
            capture_by_variant.get(pid, {}).get(first_variant, default_cap_dir)
        )
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
        point_variants = point_variants_by_id[pid]
        for variant in point_variants:
            cap_dir = Path(
                capture_by_variant.get(pid, {}).get(variant, default_cap_dir)
            ).resolve()
            capture_receipt = cap_dir / "research_receipt.json"
            if not capture_receipt.is_file():
                print(
                    f"FAIL: capture for {pid}/{variant} not complete at {cap_dir}",
                    file=sys.stderr,
                )
                return 1
            m1_request = point_m1_request(
                inputs_root, pid, cfg.get("m1_request"),
                config_base=config_path.parent,
                variant=variant,
            )
            out_dir = args.output_root / _output_directory_name(pid, variant)
            state = point_state(out_dir, layouts=layouts)
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
                   "--runtime-prefix", cfg["runtime_prefix"],
                   "--rlr-sdk-root", cfg["rlr_sdk_root"],
                   "--magnum-python-site", cfg["magnum_python_site"],
                   "--actor-selection", str(inputs_root / pid /
                                            "actor_selection.json"),
                   "--source-asset-registry", cfg["source_asset_registry"],
                   "--variant", "A",
                   "--execution-variant", variant,
                   "--layouts", ",".join(layouts),
                   "--output", str(out_dir)]
            try:
                cmd.extend(hrtf_args(cfg, layouts))
            except SystemExit as error:
                print(str(error), file=sys.stderr)
                return 2
            cmd.extend(sound_args)
            cmd.extend(canonical_emitter_args(cfg))
            log_path = args.output_root / f"{out_dir.name}.log"
            with open(log_path, "w") as log:
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                      cwd=str(repo))
            if proc.returncode != 0 or point_state(out_dir, layouts=layouts) != "complete":
                print(f"FAIL: audio render {out_dir.name} failed "
                      f"(exit {proc.returncode}); log: {log_path}",
                      file=sys.stderr)
                return 1
            done += 1
            print(f"ok {out_dir.name} log={log_path.name}")
    print(f"rendered={done} skipped_complete={skipped} "
          f"points={len(points)} variants={variants} layouts={list(layouts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
