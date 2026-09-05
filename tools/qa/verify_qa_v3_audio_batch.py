#!/usr/bin/env python3
"""Batch-level verification of qa-v3 pilot audio renders (post-render gate).

对整个音频批做四层检查,任何一层失败即非零退出(失败即停):
1. 双声道结构:每个 mixture.wav 过 1.1 校验器同款判定(双声道 + 左右
   逐样本差异达标),这里直接内联同一判定式以免子进程 ×288;
2. receipt 一致性:当前点的 endpoint registry、audio_program 路径和
   qualification_claim 必须与实际输入一致（保留旧 QA-v2 兼容回退）;
3. onset 落位抽验:每点每事件,事件内 RMS 必须显著高于事件前静默段
   (比值下限显式参数);锚事件必须是最后事件且锚后尾静默 ≥ TAIL_MIN_S;
4. **音频变体非同一**:同点 main 与 gateA 的 mixture 波形最大逐样本差
   必须超过阈值。**这一层只证明音频确实被改了,不证明 Gate A 语义
   成立** —— 改增益、改噪声、改任意事件都能让波形不同。Gate A 要求的
   是"只改承载选择事实的音频变量、视觉不变、该题型的金标按预期翻转",
   那要逐题比对 main/gateA 的事实记录,不能由波形差异代替。

输出批级 manifest(no-clobber)。research_candidate。
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

AMP_EPSILON = 1e-6
MIN_DIFF_RATIO = 0.01
ONSET_RMS_RATIO_MIN = 5.0     # 事件内/事件前 RMS 比值下限(显式参数)
GATEA_MAX_DIFF_MIN = 1e-4     # main vs gateA 最大逐样本差下限
SR = 16000


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


def normalize_layouts(value: object, *, owner: str = "layouts") -> tuple[str, ...]:
    """Normalize canonical renderer layouts without accepting aliases."""
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


def check_ambisonics(wav: np.ndarray) -> str | None:
    """Validate a raw FOA array with an explicit channel-energy check."""
    if wav.ndim != 2 or wav.shape[1] != AUDIO_LAYOUTS["ambisonics"]["channel_count"]:
        return f"not 4-channel FOA: shape={wav.shape}"
    if not np.all(np.isfinite(wav)):
        return "FOA contains non-finite samples"
    channel_energy = np.mean(np.square(wav.astype(np.float64)), axis=0)
    if not np.all(np.isfinite(channel_energy)):
        return "FOA channel energy is non-finite"
    if not np.any(channel_energy > AMP_EPSILON ** 2):
        return "FOA channels are silent"
    return None


def check_layout_audio(wav: np.ndarray, layout: str) -> str | None:
    if layout == "binaural":
        return check_stereo(wav)
    if layout == "ambisonics":
        return check_ambisonics(wav)
    raise ValueError(f"unsupported audio layout {layout!r}")


def _clock_values(value: object) -> list[tuple[int, int]]:
    if not isinstance(value, dict):
        return []
    result: list[tuple[int, int]] = []
    rate = value.get("sample_rate_hz")
    count = value.get("sample_count")
    if "sample_rate_hz" in value or "sample_count" in value:
        if not (
            isinstance(rate, int) and not isinstance(rate, bool) and rate > 0
            and isinstance(count, int) and not isinstance(count, bool) and count > 0
        ):
            raise ValueError(
                "receipt audio clock must contain positive integer "
                "sample_rate_hz and sample_count"
            )
        result.append((rate, count))
    for key in ("binaural", "ambisonics", "foa"):
        nested = value.get(key)
        if isinstance(nested, dict):
            result.extend(_clock_values(nested))
    layouts = value.get("layouts")
    if isinstance(layouts, dict):
        for nested in layouts.values():
            result.extend(_clock_values(nested))
    elif isinstance(layouts, list):
        for nested in layouts:
            result.extend(_clock_values(nested))
    return result


def _declared_audio_clock(
    program: object, receipt: object
) -> tuple[tuple[int, int] | None, bool]:
    values: list[tuple[int, int]] = []
    if isinstance(program, dict):
        values.extend(_clock_values(program.get("timeline")))
    if isinstance(receipt, dict):
        try:
            values.extend(_clock_values(receipt.get("audio")))
            audio_program = receipt.get("audio_program")
            if isinstance(audio_program, dict):
                values.extend(_clock_values(audio_program.get("timeline")))
            if not values:
                values.extend(_clock_values(receipt))
        except ValueError:
            return None, True
    unique = set(values)
    if len(unique) > 1:
        return None, True
    return (values[0] if values else None), False


def _layout_mixture_path(render_dir: Path, layout: str) -> Path:
    return (
        render_dir / "audio" / AUDIO_LAYOUTS[layout]["directory"] / "mixture.wav"
    )


def _declared_layouts(receipt: object) -> tuple[str, ...] | None:
    if not isinstance(receipt, dict):
        return None
    for key in ("expected_layouts", "layouts", "audio_layouts"):
        value = receipt.get(key)
        if value is not None:
            return normalize_layouts(value, owner=f"receipt.{key}")
    audio = receipt.get("audio")
    if isinstance(audio, dict):
        declared = audio.get("layouts")
        if declared is not None:
            return normalize_layouts(
                declared, owner="receipt.audio.layouts"
            )
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


def check_receipt_layouts(
    receipt: object,
    expected_layouts: tuple[str, ...],
    expected_sr: int,
    expected_count: int,
) -> list[str]:
    """Check declared per-layout paths, labels and clocks against the contract."""
    if not isinstance(receipt, dict):
        return ["receipt is not an object"]
    audio = receipt.get("audio")
    if not isinstance(audio, dict):
        # Old QA-v2 receipts have no audio block. Keep binary-only legacy
        # verification readable; FOA still requires the explicit declaration.
        if "ambisonics" in expected_layouts:
            return ["receipt audio.by_layout is required for ambisonics"]
        return []
    by_layout = audio.get("by_layout")
    if by_layout is None:
        # Older binaural receipts predate the per-layout contract. They remain
        # readable; a newly requested FOA output must carry its labels because
        # WAV itself has no channel-order metadata.
        if "ambisonics" in expected_layouts:
            return ["receipt audio.by_layout is required for ambisonics"]
        return []
    if not isinstance(by_layout, dict):
        return ["receipt audio.by_layout must be an object"]
    failures: list[str] = []
    for layout in expected_layouts:
        entry = by_layout.get(layout)
        if not isinstance(entry, dict):
            failures.append(f"receipt audio.by_layout is missing {layout}")
            continue
        spec = AUDIO_LAYOUTS[layout]
        expected = {
            "layout_type": layout,
            "output_directory": spec["directory"],
            "channel_count": spec["channel_count"],
            "channel_labels": list(spec["channel_labels"]),
            "sample_rate_hz": expected_sr,
            "sample_count": expected_count,
        }
        for field, value in expected.items():
            if entry.get(field) != value:
                failures.append(
                    f"receipt {layout} {field}={entry.get(field)!r} "
                    f"does not match {value!r}"
                )
    return failures


def _inferred_layouts(audio_root: Path) -> tuple[str, ...]:
    declarations: list[tuple[str, ...]] = []
    if audio_root.is_dir():
        for render_dir in sorted(audio_root.iterdir()):
            if not render_dir.is_dir():
                continue
            receipt_path = render_dir / "research_receipt.json"
            if not receipt_path.is_file():
                continue
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            try:
                declared = _declared_layouts(receipt)
            except ValueError as error:
                raise ValueError(
                    f"{receipt_path}: invalid declared audio layouts: {error}"
                ) from error
            if declared is not None:
                declarations.append(declared)
    if not declarations:
        return DEFAULT_LAYOUTS
    first = declarations[0]
    if any(item != first for item in declarations[1:]):
        raise ValueError(
            "audio receipts declare conflicting expected layouts: "
            + ", ".join(str(list(item)) for item in declarations)
        )
    return first


def execution_variant_status(
    receipt: dict,
    expected_execution_variant: str | None,
) -> str:
    """Classify the external execution label without conflating AudioProgram A."""

    if expected_execution_variant is None:
        return "not_requested"
    value = receipt.get("execution_variant")
    if value is None:
        return "unverified"
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        return "invalid"
    if value != expected_execution_variant:
        return "mismatch"
    return "verified"


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


def _normalize_expected_variants(value: object, *, owner: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raw = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raise ValueError(f"{owner} must be a comma-separated string or list")
    if not raw:
        raise ValueError(f"{owner} must not be empty")
    result: list[str] = []
    for item in raw:
        item = _safe_variant_name(item, owner=owner)
        if item in result:
            raise ValueError(f"{owner} contains duplicate execution variant {item!r}")
        result.append(item)
    if "main" not in result:
        raise ValueError(f"{owner} must include main")
    return tuple(result)


def _declared_audio_variants(payload: object) -> tuple[str, ...] | None:
    if not isinstance(payload, dict):
        return None
    owners: list[dict] = [payload]
    for key in ("request", "request_snapshot", "declaration"):
        value = payload.get(key)
        if isinstance(value, dict):
            owners.append(value)
    for owner in owners:
        value = owner.get("audio_variants")
        if value is not None:
            return _normalize_expected_variants(
                value, owner="declared audio_variants"
            )
    question_request = payload.get("question_request")
    if isinstance(question_request, dict) and question_request.get(
        "counterfactual_questions"
    ):
        return ("main", "gateA")
    return None


def _expected_variants(
    design_root: Path,
    audio_root: Path,
    explicit: object,
) -> tuple[tuple[str, ...], str]:
    if explicit is not None:
        return (
            _normalize_expected_variants(explicit, owner="--variants"),
            "explicit",
        )
    candidates: list[Path] = []
    for root in (design_root, audio_root, audio_root.parent):
        for name in ("request.json", "batch_manifest.json", "scene_profile_matrix.json"):
            path = root / name
            if path.is_file() and path not in candidates:
                candidates.append(path)
    for root in (design_root, *design_root.parents[:12]):
        path = root / "request.json"
        if path.is_file() and path not in candidates:
            candidates.append(path)
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        declared = _declared_audio_variants(payload)
        if declared is not None:
            return declared, f"declared:{path}"
    # A Gate-A fact/program/question artifact is an actual design declaration.
    # Merely having a shared legacy program filename is insufficient.
    for marker in (
        design_root.rglob("fact_record_gateA.json"),
        design_root.rglob("audio_program_gateA.json"),
        design_root.rglob("questions_gateA.jsonl"),
    ):
        if next(iter(marker), None) is not None:
            return ("main", "gateA"), "design_gateA_artifact"
    if audio_root.is_dir() and any(
        path.is_dir() and path.name.endswith("_gateA")
        for path in audio_root.iterdir()
    ):
        return ("main", "gateA"), "observed_gateA_directory"
    return ("main",), "legacy_single_main_default"


def _design_point_ids(design_root: Path) -> tuple[str, ...]:
    """Discover real design point IDs without splitting names on underscores."""
    result: list[str] = []
    for path in sorted(design_root.iterdir()):
        if (
            not path.is_dir()
            or path.is_symlink()
            or path.name in {"programs", "logs"}
        ):
            continue
        has_marker = any(
            (
                (path / name).is_file()
                for name in ("timeline.json", "fact_record.json", "source_endpoints.json")
            )
        ) or any(path.glob("fact_record_*.json")) or any(
            path.glob("audio_program*.json")
        )
        if has_marker:
            result.append(path.name)
    return tuple(result)


def _render_directory_identity(
    name: str,
    expected_variants: Sequence[str],
    design_point_ids: Sequence[str],
) -> tuple[str, str]:
    """Resolve one output directory from known point IDs and variants.

    Matching complete design point names first is what keeps an ID such as
    ``candidate_alternate`` from being mistaken for point ``candidate``'s
    alternate output.  If both interpretations exist, fail explicitly.
    """
    candidates: list[tuple[str, str]] = []
    for point_id in design_point_ids:
        if name == point_id:
            candidates.append((point_id, "main"))
        for variant in expected_variants:
            if variant != "main" and name == f"{point_id}_{variant}":
                candidates.append((point_id, variant))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            f"audio render directory {name!r} is ambiguous: "
            + ", ".join(f"{point}/{variant}" for point, variant in candidates)
        )
    if design_point_ids:
        raise ValueError(
            f"audio render directory {name!r} does not match a design point "
            f"or requested variant {list(expected_variants)}"
        )
    # Retained legacy batches may have no point marker files.  Use only the
    # explicitly requested variant suffixes, longest first; never split a
    # point ID merely because it contains an underscore.
    for variant in sorted(
        (item for item in expected_variants if item != "main"),
        key=len,
        reverse=True,
    ):
        suffix = f"_{variant}"
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)], variant
    return name, "main"


def _split_render_directory(name: str) -> tuple[str, str]:
    if name.endswith("_gateA"):
        return name[:-6], "gateA"
    return name, "main"


def check_stereo(wav: np.ndarray) -> str | None:
    if wav.ndim != 2 or wav.shape[1] != 2:
        return f"not stereo: shape={wav.shape}"
    diff = np.abs(wav[:, 0] - wav[:, 1])
    ratio = float((diff > AMP_EPSILON).mean())
    if ratio < MIN_DIFF_RATIO:
        return f"channels nearly identical: diff_ratio={ratio:.5f}"
    return None


def _validate_source_endpoint_point_id(value: object, *, owner: str) -> str:
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
            f"{owner} must be a single non-empty point ID without path separators "
            "or control characters"
        )
    return value


def _resolve_existing_source_endpoint_path(
    value: object,
    *,
    owner: str,
    base: Path,
) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{owner} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"{owner} is missing or not a regular file: {resolved}")
    return resolved


def _parse_source_endpoint_overrides(
    values: Sequence[str] | None,
    *,
    base: Path,
) -> dict[str, Path]:
    if values is None:
        return {}
    if isinstance(values, (str, bytes)):
        raise ValueError("--source-endpoint must be repeated POINT_ID=PATH values")
    result: dict[str, Path] = {}
    for index, assignment in enumerate(values):
        if not isinstance(assignment, str):
            raise ValueError(
                f"source endpoint {index} must be a POINT_ID=PATH string"
            )
        point_id, separator, path_value = assignment.partition("=")
        if not separator or not point_id or not path_value:
            raise ValueError(
                f"source endpoint {index} must use POINT_ID=PATH"
            )
        point_id = _validate_source_endpoint_point_id(
            point_id, owner=f"source endpoint {index} point ID"
        )
        if point_id in result:
            raise ValueError(
                f"duplicate source endpoint point ID: {point_id!r}"
            )
        result[point_id] = _resolve_existing_source_endpoint_path(
            path_value,
            owner=f"source endpoint {index} path",
            base=base,
        )
    return result


def _reject_duplicate_endpoint_map_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"source endpoint map contains duplicate point ID: {key!r}"
            )
        result[key] = value
    return result


def _load_source_endpoint_map(path: Path | None) -> dict[str, Path]:
    if path is None:
        return {}
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(
            f"--source-endpoint-map is missing or not a regular file: {resolved}"
        )
    try:
        value = json.loads(
            resolved.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_endpoint_map_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            f"cannot read source endpoint map {resolved}: {error}"
        ) from error
    if not isinstance(value, Mapping):
        raise ValueError("source endpoint map must contain a JSON object")
    result: dict[str, Path] = {}
    for point_id, endpoint in value.items():
        point_id = _validate_source_endpoint_point_id(
            point_id, owner="source endpoint map point ID"
        )
        result[point_id] = _resolve_existing_source_endpoint_path(
            endpoint,
            owner=f"source endpoint map {point_id}",
            base=resolved.parent,
        )
    return result


def _expected_source_endpoint_path(
    design_root: Path,
    base: str,
    render_name: str,
    *,
    per_point: Mapping[str, Path],
    shared: Path | None,
) -> Path | None:
    local = (design_root / base / "source_endpoints.json").resolve()
    if local.is_file():
        return local
    return per_point.get(render_name) or per_point.get(base) or shared


def check_receipt(receipt: dict, expect_program: Path,
                  expected_endpoint: Path | None = None,
                  expected_execution_variant: str | None = None) -> str | None:
    if receipt.get("qualification_claim") is not False:
        return "qualification_claim is not false"
    try:
        reg = receipt["inputs"]["source_endpoint_registry"]["path"]
    except (KeyError, TypeError):
        return "receipt has no source endpoint registry"
    if not isinstance(reg, str) or not reg.strip():
        return "receipt source endpoint registry path is invalid"
    if expected_endpoint is None:
        # Retain the historical positive control for old QA v2 fixture batches;
        # current QA-v3 points pass their point-local registry explicitly.
        if "qa_v2/source_endpoints_qa_v2_v1.json" not in reg:
            return f"wrong endpoint registry: {reg}"
    else:
        expected_path = Path(expected_endpoint).expanduser().resolve()
        if not expected_path.is_file():
            return f"expected endpoint registry is missing: {expected_path}"
        actual_endpoint = Path(reg).expanduser().resolve()
        if actual_endpoint != expected_path:
            return f"wrong endpoint registry: {actual_endpoint} != {expected_path}"
    try:
        prog = receipt["audio_program"]["path"]
    except (KeyError, TypeError):
        return "receipt has no audio program"
    if Path(prog).name != expect_program.name:
        return f"program mismatch: {Path(prog).name} != {expect_program.name}"
    variant_state = execution_variant_status(
        receipt, expected_execution_variant
    )
    if variant_state == "mismatch":
        return (
            "wrong execution variant: "
            f"{receipt.get('execution_variant')!r} != {expected_execution_variant!r}"
        )
    if variant_state == "invalid":
        return "execution_variant is invalid"
    # A missing field is a legacy receipt. Keep it readable but do not claim
    # that the main/Gate-A execution identity was verified.
    return None


def _variant_program_filename(variant: str) -> str:
    variant = _safe_variant_name(variant, owner="audio variant")
    return (
        "audio_program.json"
        if variant == "main"
        else f"audio_program_{variant}.json"
    )


def _variant_fact_paths(
    design_root: Path, base: str, variant: str
) -> tuple[tuple[Path, bool], ...]:
    variant = _safe_variant_name(variant, owner="audio variant")
    point_dir = design_root / base
    specific = point_dir / (
        "fact_record.json"
        if variant == "main"
        else f"fact_record_{variant}.json"
    )
    if variant == "main":
        return ((specific, True),)
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
        path = Path(declared).expanduser()
        candidates = (
            [path]
            if path.is_absolute()
            else [point_dir / path, programs_dir / path]
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return None
    if not isinstance(declared, Mapping):
        return None
    declared_path = (
        declared.get("path")
        or declared.get("program_path")
        or declared.get("file")
    )
    if isinstance(declared_path, (str, Path)):
        return _declared_program_path(
            declared_path, point_dir=point_dir, programs_dir=programs_dir
        )
    declared_id = declared.get("program_id") or declared.get("id")
    if isinstance(declared_id, str) and declared_id:
        path = programs_dir / f"{declared_id}.json"
        return path.resolve() if path.is_file() else None
    return None


def _load_fact_for_variant(
    design_root: Path, base: str, variant: str
) -> tuple[dict | None, Path | None]:
    for path, _specific in _variant_fact_paths(design_root, base, variant):
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None, path
        if isinstance(value, Mapping):
            return dict(value), path
        return None, path
    return None, None


def find_program(programs_dir: Path, base: str, variant: str,
                 *, design_root: Path | None = None) -> Path | None:
    """Resolve one variant's point-local, fact-declared or shared program."""
    variant = _safe_variant_name(variant, owner="audio variant")
    root = Path(design_root) if design_root is not None else programs_dir.parent
    point_dir = root / base
    local = point_dir / _variant_program_filename(variant)
    if local.is_file():
        return local.resolve()

    for fact_path, variant_specific in _variant_fact_paths(root, base, variant):
        if not fact_path.is_file():
            continue
        try:
            value = json.loads(fact_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(value, Mapping):
            continue
        for declared in _fact_program_declarations(
            value,
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
        for match in programs_dir.glob(f"qa_v3_*_{base}{suffix}")
    })
    return matches[0].resolve() if len(matches) == 1 else None


def check_onsets(wav: np.ndarray, program: dict, authority: dict,
                 profile_id: str | None,
                 tail_min_s: float, sample_rate_hz: int = SR) -> list[str]:
    """Check event energy plus profile-specific anchor/tail structure.

    ``authority`` is a current fact record, or a retained run01 plan for
    backward-compatible verification of historical evidence.
    """
    errors = []
    mono = np.abs(wav).mean(axis=1)
    pre_window = max(1, int(round(sample_rate_hz * 0.1)))
    events = sorted(program["events"], key=lambda e: e["start_sample"])
    for e in events:
        s0, s1 = e["start_sample"], e["end_sample_exclusive"]
        pre = float(mono[max(0, s0 - pre_window):s0].mean()) if s0 >= pre_window else 0.0
        inside = float(mono[s0:s1].mean())
        if inside <= 0:
            errors.append(f"event@{s0}: zero energy inside event")
        elif pre > 0 and inside / pre < ONSET_RMS_RATIO_MIN:
            errors.append(f"event@{s0}: inside/pre RMS {inside / pre:.2f} "
                          f"< {ONSET_RMS_RATIO_MIN}")
    if "anchor_end_sample" in authority:
        anchor_end = int(authority["anchor_end_sample"])
        if anchor_end != max(e["end_sample_exclusive"] for e in events):
            errors.append("anchor is not the last event")
        tail_s = (len(mono) - anchor_end) / float(sample_rate_hz)
        if tail_s < tail_min_s - 1e-9:
            errors.append(f"tail {tail_s:.3f}s < TAIL_MIN {tail_min_s}")
        return errors
    identity_starts = {
        int(event["start_sample"])
        for event in authority.get("audio", {}).get("events", [])
        if event.get("purpose") == "identity_anchor"
    }
    identity = [
        event for event in events
        if int(event["start_sample"]) in identity_starts
    ]
    if profile_id in {"card1F", "card1B"}:
        if len(identity) != 1 or identity[0] is not events[-1]:
            errors.append("identity anchor is not the unique last event")
        elif profile_id == "card1F":
            tail_s = (len(mono) - identity[0]["end_sample_exclusive"]) / float(sample_rate_hz)
            if tail_s < tail_min_s - 1e-9:
                errors.append(f"tail {tail_s:.3f}s < TAIL_MIN {tail_min_s}")
    return errors


def _gatea_semantic_failures(
    main_fact: dict,
    gate_fact: dict,
    *,
    main_program: dict | None = None,
    gate_program: dict | None = None,
) -> list[str]:
    """Validate a route-only audio intervention from artifacts, not profile IDs."""

    missing: list[str] = []
    if isinstance(main_program, dict) and isinstance(gate_program, dict):
        main_events = main_program.get("events")
        gate_events = gate_program.get("events")
        if not isinstance(main_events, list) or not isinstance(gate_events, list):
            missing.append("events_present")
            main_events, gate_events = [], []
        if len(main_events) != len(gate_events):
            missing.append("event_count_preserved")
        elif main_events:
            timing_fields = (
                "start_tick", "end_tick_exclusive", "start_sample",
                "end_sample_exclusive", "source_start_sample",
                "source_end_sample_exclusive",
            )
            if any(
                any(left.get(key) != right.get(key) for key in timing_fields)
                for left, right in zip(main_events, gate_events)
            ):
                missing.append("event_times_preserved")
            if any(
                left.get("sound_asset_id") != right.get("sound_asset_id")
                for left, right in zip(main_events, gate_events)
            ):
                missing.append("sound_asset_sequence_preserved")
            if all(
                left.get("source_endpoint_id") == right.get("source_endpoint_id")
                for left, right in zip(main_events, gate_events)
            ):
                missing.append("audio_assignment_changed")
            ignored = {"source_endpoint_id", "event_id"}
            if any(
                {key: value for key, value in left.items() if key not in ignored}
                != {key: value for key, value in right.items() if key not in ignored}
                for left, right in zip(main_events, gate_events)
            ):
                missing.append("non_assignment_event_fields_preserved")
        main_candidates = main_program.get("candidate_source_endpoint_ids")
        gate_candidates = gate_program.get("candidate_source_endpoint_ids")
        if main_candidates != gate_candidates:
            missing.append("candidate_endpoints_preserved")
    else:
        # Retained facts produced before direct program comparison carry one of
        # these two explicit check vocabularies.
        old_required = (
            "event_count_same", "candidate_endpoints_same",
            "non_slot_event_fields_same", "slot_sequence_changed",
            "mcq_stem_same", "mcq_options_same", "open_stem_same",
            "mcq_gold_flipped", "open_gold_separated",
        )
        current_required = (
            "event_count_preserved", "event_times_preserved",
            "sound_asset_multiset_preserved", "audio_assignment_changed",
            "question_stem_preserved", "question_options_preserved",
            "question_gold_changed",
        )
        checks = {}
        for owner in (gate_fact, main_fact):
            metadata = owner.get("gatea")
            if isinstance(metadata, dict) and isinstance(metadata.get("checks"), dict):
                checks = metadata["checks"]
                if checks:
                    break
            direct = owner.get("gatea_checks")
            if isinstance(direct, dict) and direct:
                checks = direct
                break
        required = (
            current_required
            if any(key in checks for key in current_required)
            else old_required
        )
        missing.extend(key for key in required if checks.get(key) is not True)

    comparable = []
    for form, truth_key in (("mcq", "truth_option"), ("open", "truth_value")):
        main_form = main_fact.get(form)
        gate_form = gate_fact.get(form)
        if not isinstance(main_form, dict) or not isinstance(gate_form, dict):
            continue
        if main_form.get("stem") != gate_form.get("stem"):
            missing.append(f"{form}_stem_preserved")
        if form == "mcq" and main_form.get("options_space") != gate_form.get("options_space"):
            missing.append("mcq_options_preserved")
        if truth_key in main_form and truth_key in gate_form:
            comparable.append((form, main_form[truth_key], gate_form[truth_key]))
    if not comparable:
        missing.append("question_gold_comparable")
    elif any(main_value == gate_value for _, main_value, gate_value in comparable):
        missing.append("question_gold_changed")
    return list(dict.fromkeys(missing))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design-root", required=True, type=Path)
    parser.add_argument("--audio-root", required=True, type=Path)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument(
        "--variants",
        default=None,
        help="expected execution variants, comma-separated; defaults to design/request declaration",
    )
    parser.add_argument(
        "--layouts",
        default=None,
        help="expected audio layouts, comma-separated: binaural[,ambisonics]; "
             "defaults to the receipt declaration or binaural",
    )
    parser.add_argument(
        "--source-endpoint-registry",
        type=Path,
        help="shared fallback source endpoint registry for points without a local file",
    )
    parser.add_argument(
        "--source-endpoint-map",
        type=Path,
        help="JSON object mapping point IDs to endpoint registry paths",
    )
    parser.add_argument(
        "--source-endpoint",
        action="append",
        default=[],
        metavar="POINT_ID=PATH",
        help="per-point source endpoint registry override; may be repeated",
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.out.exists():
        print(f"refusing to overwrite: {args.out}", file=sys.stderr)
        return 2
    try:
        source_endpoint_shared = (
            _resolve_existing_source_endpoint_path(
                args.source_endpoint_registry,
                owner="--source-endpoint-registry",
                base=Path.cwd().resolve(),
            )
            if args.source_endpoint_registry is not None
            else None
        )
        source_endpoint_overrides = _load_source_endpoint_map(
            args.source_endpoint_map
        )
        repeated_source_endpoints = _parse_source_endpoint_overrides(
            args.source_endpoint,
            base=Path.cwd().resolve(),
        )
        duplicates = set(source_endpoint_overrides) & set(
            repeated_source_endpoints
        )
        if duplicates:
            raise ValueError(
                "source endpoint point IDs were declared more than once: "
                + ", ".join(sorted(duplicates))
            )
        source_endpoint_overrides.update(repeated_source_endpoints)
    except ValueError as error:
        print(f"invalid source endpoint configuration: {error}", file=sys.stderr)
        return 2
    try:
        expected_layouts = (
            normalize_layouts(args.layouts, owner="--layouts")
            if args.layouts is not None
            else _inferred_layouts(args.audio_root.resolve())
        )
    except ValueError as error:
        print(f"invalid layouts: {error}", file=sys.stderr)
        return 2
    params = json.loads(args.params.read_text())
    tail_min_s = float(params["TAIL_MIN_S"])
    programs_dir = args.design_root / "programs"
    expected_variants, expected_variants_source = _expected_variants(
        args.design_root.resolve(),
        args.audio_root.resolve(),
        args.variants,
    )
    expected_variant_set = set(expected_variants)
    audio_directories = sorted(
        d for d in args.audio_root.iterdir() if d.is_dir()
    )
    point_dirs = sorted(
        d for d in audio_directories
        if (d / "research_receipt.json").is_file()
    )
    failures: list[str] = []
    missing_receipt_directories = sorted(
        path.name
        for path in audio_directories
        if not (path / "research_receipt.json").is_file()
    )
    if missing_receipt_directories:
        failures.append(
            "audio render directories are missing research receipts: "
            + ", ".join(missing_receipt_directories)
        )
    design_point_ids = _design_point_ids(args.design_root.resolve())
    render_entries: list[tuple[Path, str, str, str]] = []
    for path in point_dirs:
        try:
            base, variant = _render_directory_identity(
                path.name, expected_variants, design_point_ids
            )
        except ValueError as error:
            failures.append(str(error))
            continue
        render_entries.append((path, path.name, base, variant))
    observed_directory_variants = {variant for _, _, _, variant in render_entries}
    if not audio_directories:
        failures.append("audio root contains no render directories")
    unexpected_variants = observed_directory_variants - expected_variant_set
    if unexpected_variants:
        failures.append(
            "unexpected render variants: "
            + ", ".join(sorted(unexpected_variants))
        )
    execution_variant_verified: list[str] = []
    execution_variant_unverified: list[str] = []
    execution_variant_failed: list[str] = []
    validated_render_ids: dict[str, set[str]] = {
        variant: set() for variant in expected_variants
    }
    # Keep the historical report key readable for consumers that only know
    # main/Gate-A, while actual requested variants are all reported below.
    validated_render_ids.setdefault("main", set())
    validated_render_ids.setdefault("gateA", set())
    checked = gatea_pairs = gatea_semantic_pairs = 0
    reference_layout = (
        "binaural" if "binaural" in expected_layouts else expected_layouts[0]
    )
    for d, name, base, variant in render_entries:
        if variant not in expected_variant_set:
            continue
        prog_path = find_program(
            programs_dir, base, variant, design_root=args.design_root
        )
        if prog_path is None:
            failures.append(f"{name}: program lookup was not unique")
            continue
        try:
            program = json.loads(prog_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            failures.append(f"{name}: audio program load failed: {error}")
            continue
        fact, _fact_path = _load_fact_for_variant(
            args.design_root.resolve(), base, variant
        )
        plan_path = (
            programs_dir / f"{prog_path.stem}.plan.json"
            if variant == "main" else None
        )
        plan = None
        if plan_path is not None and plan_path.is_file():
            try:
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                plan = None
        try:
            receipt = json.loads((d / "research_receipt.json").read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            failures.append(f"{name}: receipt load failed: {exc}")
            continue

        render_valid = True
        try:
            declared_layouts = _declared_layouts(receipt)
        except ValueError as error:
            failures.append(f"{name}: invalid declared audio layouts: {error}")
            declared_layouts = None
            render_valid = False
        if declared_layouts is not None and declared_layouts != expected_layouts:
            failures.append(
                f"{name}: receipt layouts {list(declared_layouts)} differ from "
                f"expected {list(expected_layouts)}"
            )
            render_valid = False

        execution_state = execution_variant_status(receipt, variant)
        if execution_state == "verified":
            execution_variant_verified.append(name)
        elif execution_state == "unverified":
            execution_variant_unverified.append(name)
            if variant != "main":
                failures.append(
                    f"{name}: execution_variant is unverified for requested {variant}"
                )
                render_valid = False
        elif execution_state in {"invalid", "mismatch"}:
            execution_variant_failed.append(name)

        declared_clock, clock_conflict = _declared_audio_clock(program, receipt)
        render_valid = render_valid and not clock_conflict
        if clock_conflict:
            failures.append(f"{name}: program/receipt audio clocks disagree")
        expected_sr, expected_count = declared_clock or (SR, 80000)
        for error in check_receipt_layouts(
            receipt, expected_layouts, expected_sr, expected_count
        ):
            failures.append(f"{name}: {error}")
            render_valid = False
        layout_wavs: dict[str, np.ndarray] = {}
        for layout in expected_layouts:
            layout_root = d / "audio" / AUDIO_LAYOUTS[layout]["directory"]
            mixture_path = layout_root / "mixture.wav"
            files = sorted(layout_root.rglob("*.wav")) if layout_root.is_dir() else []
            if not mixture_path.is_file():
                failures.append(
                    f"{name}: missing requested {layout} mixture: {mixture_path}"
                )
                render_valid = False
                continue
            if not files:
                failures.append(f"{name}: requested {layout} layout has no WAV files")
                render_valid = False
                continue
            for wav_path in files:
                try:
                    wav, sr = sf.read(wav_path, always_2d=True)
                except (OSError, RuntimeError) as exc:
                    failures.append(f"{name}: {layout} load failed: {exc}")
                    render_valid = False
                    continue
                if sr != expected_sr or wav.shape[0] != expected_count:
                    failures.append(
                        f"{name}: {layout} {wav_path.name} has {sr}Hz "
                        f"{wav.shape[0]} samples; expected "
                        f"{expected_sr}Hz {expected_count}"
                    )
                    render_valid = False
                    continue
                layout_error = check_layout_audio(wav, layout)
                if layout_error:
                    failures.append(f"{name}: {layout} {layout_error}")
                    render_valid = False
                if wav_path == mixture_path:
                    layout_wavs[layout] = wav

        expected_endpoint = _expected_source_endpoint_path(
            args.design_root.resolve(),
            base,
            name,
            per_point=source_endpoint_overrides,
            shared=source_endpoint_shared,
        )
        for err in filter(
            None,
            [
                check_receipt(
                    receipt,
                    prog_path,
                    expected_endpoint,
                    expected_execution_variant=variant,
                )
            ],
        ):
            failures.append(f"{name}: {err}")
            render_valid = False

        reference_wav = layout_wavs.get(reference_layout)
        if variant == "main":
            authority = fact if fact is not None else plan
            if authority is None:
                failures.append(f"{name}: no fact or retained plan authority")
                authority = {}
                render_valid = False
            if reference_wav is None:
                failures.append(
                    f"{name}: reference {reference_layout} mixture is unavailable"
                )
                render_valid = False
            else:
                for err in check_onsets(
                    reference_wav,
                    program,
                    authority,
                    fact.get("profile_id") if fact else None,
                    tail_min_s,
                    sample_rate_hz=expected_sr,
                ):
                    failures.append(f"{name}: {err}")
                    render_valid = False

        checked += 1
        if render_valid:
            validated_render_ids[variant].add(base)

        if variant == "gateA":
            main_render_dir = args.audio_root / base
            main_wav_path = _layout_mixture_path(main_render_dir, reference_layout)
            if main_wav_path.is_file() and reference_wav is not None:
                try:
                    main_wav, main_sr = sf.read(main_wav_path, always_2d=True)
                except (OSError, RuntimeError) as exc:
                    failures.append(f"{name}: main waveform could not be read: {exc}")
                else:
                    if main_sr != expected_sr or main_wav.shape != reference_wav.shape:
                        failures.append(
                            f"{name}: main/gateA {reference_layout} clocks/shapes differ: "
                            f"{main_sr}Hz {main_wav.shape} != "
                            f"{expected_sr}Hz {reference_wav.shape}"
                        )
                    else:
                        layout_error = check_layout_audio(main_wav, reference_layout)
                        if layout_error:
                            failures.append(
                                f"{name}: main {reference_layout} {layout_error}"
                            )
                        elif reference_layout == "binaural":
                            max_diff = float(
                                np.abs(reference_wav - main_wav).max()
                            )
                            if not np.isfinite(max_diff):
                                failures.append(
                                    f"{name}: main/gateA waveform difference is non-finite"
                                )
                            elif max_diff < GATEA_MAX_DIFF_MIN:
                                failures.append(
                                    f"{name}: gateA identical to main "
                                    f"(max diff {max_diff:.2e})"
                                )
                            else:
                                gatea_pairs += 1
                        else:
                            main_energy = np.mean(
                                np.square(main_wav.astype(np.float64)), axis=0
                            )
                            gate_energy = np.mean(
                                np.square(reference_wav.astype(np.float64)), axis=0
                            )
                            energy_diff = float(
                                np.abs(main_energy - gate_energy).max()
                            )
                            if not np.isfinite(energy_diff):
                                failures.append(
                                    f"{name}: main/gateA FOA channel energy is non-finite"
                                )
                            elif energy_diff < GATEA_MAX_DIFF_MIN ** 2:
                                failures.append(
                                    f"{name}: gateA FOA channel energies are identical "
                                    f"(max diff {energy_diff:.2e})"
                                )
                            else:
                                gatea_pairs += 1
            else:
                failures.append(
                    f"{name}: main {reference_layout} mixture is missing for comparison"
                )

            main_fact, _main_fact_path = _load_fact_for_variant(
                args.design_root.resolve(), base, "main"
            )
            gate_fact, _gate_fact_path = _load_fact_for_variant(
                args.design_root.resolve(), base, "gateA"
            )
            if main_fact is not None and gate_fact is not None:
                main_program_path = find_program(
                    programs_dir, base, "main", design_root=args.design_root
                )
                main_program = None
                if main_program_path is not None:
                    try:
                        main_program = json.loads(
                            main_program_path.read_text(encoding="utf-8")
                        )
                    except (OSError, UnicodeError, json.JSONDecodeError):
                        main_program = None
                missing = _gatea_semantic_failures(
                    main_fact,
                    gate_fact,
                    main_program=main_program,
                    gate_program=program,
                )
                if missing:
                    failures.append(f"{name}: Gate A semantic checks failed {missing}")
                else:
                    gatea_semantic_pairs += 1

    complete_main_ids = validated_render_ids.get("main", set())
    complete_gatea_ids = validated_render_ids.get("gateA", set())
    complete_pair_ids = complete_main_ids & complete_gatea_ids
    if not complete_main_ids:
        failures.append("missing complete main render")
    for requested_variant in expected_variants:
        if requested_variant == "main":
            continue
        complete_variant_ids = validated_render_ids.get(requested_variant, set())
        if not complete_variant_ids:
            failures.append(f"missing complete {requested_variant} render")
        if complete_variant_ids != complete_main_ids:
            if requested_variant == "gateA":
                failures.append(
                    "main and gateA render point IDs do not form complete pairs: "
                    f"main_only={sorted(complete_main_ids - complete_variant_ids)} "
                    f"gateA_only={sorted(complete_variant_ids - complete_main_ids)}"
                )
            else:
                failures.append(
                    f"main and {requested_variant} render point IDs do not form "
                    "complete pairs: "
                    f"main_only={sorted(complete_main_ids - complete_variant_ids)} "
                    f"{requested_variant}_only={sorted(complete_variant_ids - complete_main_ids)}"
                )
    if "gateA" in expected_variant_set:
        if not complete_pair_ids:
            failures.append("no complete main/gateA render pair")
        complete_pair_count = len(complete_pair_ids)
        if gatea_pairs != complete_pair_count:
            failures.append(
                "waveform nonidentity pair count "
                f"{gatea_pairs} != complete pair count {complete_pair_count}"
            )
        if gatea_semantic_pairs != complete_pair_count:
            failures.append(
                "Gate A semantic pair count "
                f"{gatea_semantic_pairs} != complete pair count {complete_pair_count}"
            )
        if execution_variant_unverified:
            failures.append(
                "execution_variant is unverified for requested main+gateA batch: "
                + ", ".join(sorted(execution_variant_unverified))
            )
    payload = {
        "schema": "qa_v3_audio_batch_verification_v1",
        "audio_root": str(args.audio_root),
        "expected_layouts": list(expected_layouts),
        "expected_variants": list(expected_variants),
        "expected_variants_source": expected_variants_source,
        "complete_render_point_ids": {
            **{
                variant: sorted(validated_render_ids.get(variant, set()))
                for variant in expected_variants
            },
            "main": sorted(complete_main_ids),
            "gateA": sorted(complete_gatea_ids),
        },
        "complete_pair_count": len(complete_pair_ids),
        "checked_renders": checked,
        "audio_variant_waveform_nonidentity_pairs": gatea_pairs,
        "gatea_semantic_flip_pairs": gatea_semantic_pairs,
        "gatea_semantic_flip": (
            "established from current paired fact records and structural checks"
            if gatea_semantic_pairs else
            "not established by this tool for retained rows without paired facts"
        ),
        "onset_rms_ratio_min": ONSET_RMS_RATIO_MIN,
        "gatea_max_diff_min": GATEA_MAX_DIFF_MIN,
        "execution_variant_verification": {
            "field": "execution_variant",
            "status": (
                "failed"
                if execution_variant_failed
                else "unverified"
                if execution_variant_unverified
                or not execution_variant_verified
                else "verified"
            ),
            "verified_renders": execution_variant_verified,
            "unverified_renders": execution_variant_unverified,
            "failed_renders": execution_variant_failed,
        },
        "failures": failures,
        "status": "research_candidate",
        "qualification_claim": False,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    print(f"checked={checked} gateA_pairs={gatea_pairs} "
          f"gateA_semantic_pairs={gatea_semantic_pairs} "
          f"failures={len(failures)} out={args.out}")
    if failures:
        for f in failures[:12]:
            print("FAIL:", f, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
