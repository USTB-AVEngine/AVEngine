#!/usr/bin/env python3
"""Generate the QA-01..QA-24 views from one native episode bundle.

The input is the raw bundle accepted by
avengine.qa.unified_catalog.normalize_episode_bundle. The command writes a
research-candidate question set and keeps deferred conditions per QA number;
it never overwrites an existing artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.qa.unified_catalog import (  # noqa: E402
    UnifiedQAError,
    generate_unified_questions,
    normalize_episode_bundle,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _qa_ids(value: str | None) -> list[str] | None:
    if value is None:
        return None
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise UnifiedQAError("--qa-ids must contain at least one QA number")
    return values


def _load_bundle(args: argparse.Namespace) -> dict[str, Any]:
    if args.input is not None:
        value = _read_json(args.input.resolve())
        if not isinstance(value, dict):
            raise UnifiedQAError("--input must contain a JSON object")
        return value
    raw: dict[str, Any] = {}
    if args.episode_id:
        raw["episode_id"] = args.episode_id
    for field, path in (
        ("plan", args.plan),
        ("frame_readbacks", args.frame_readbacks),
        ("pixel_visibility_truth", args.pixel_visibility_truth),
        ("audio_program", args.audio_program),
        ("audio_readback", args.audio_readback),
        ("research_report", args.research_report),
        ("appearance_review", args.appearance_review),
        ("voice_bindings", args.voice_bindings),
        ("sound_registry", args.sound_registry),
        ("source_endpoint_registry", args.source_endpoint_registry),
        ("occluder_evidence", args.occluder_evidence),
        ("occluder_registry", args.occluder_registry),
    ):
        if path is not None:
            raw[field] = _read_json(path.resolve())
            raw[f"{field}_path"] = str(path.resolve())
    if not raw:
        raise UnifiedQAError("provide --input or at least one native artifact option")
    return raw


def build(
    *,
    input_path: Path | None,
    raw: dict[str, Any] | None = None,
    output_path: Path,
    seed: str,
    qa_ids: list[str] | None,
    facts_output: Path | None = None,
    include_facts: bool = False,
) -> dict[str, Any]:
    if raw is None:
        if input_path is None:
            raise UnifiedQAError("input_path is required when raw bundle is absent")
        raw = _read_json(input_path)
    facts = normalize_episode_bundle(raw)
    result = generate_unified_questions(
        facts,
        qa_ids=qa_ids,
        seed=seed,
    )
    if facts_output is not None:
        _write_json(facts_output, facts)
        result["normalized_facts_path"] = str(facts_output.resolve())
    if not include_facts:
        result.pop("input_facts", None)
        result["input_facts_summary"] = facts["input_summary"]
    _write_json(output_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--episode-id")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--frame-readbacks", type=Path)
    parser.add_argument("--pixel-visibility-truth", type=Path)
    parser.add_argument("--audio-program", type=Path)
    parser.add_argument("--audio-readback", type=Path)
    parser.add_argument("--research-report", type=Path)
    parser.add_argument("--appearance-review", type=Path)
    parser.add_argument("--voice-bindings", type=Path)
    parser.add_argument("--sound-registry", type=Path)
    parser.add_argument("--source-endpoint-registry", type=Path)
    parser.add_argument("--occluder-evidence", type=Path)
    parser.add_argument("--occluder-registry", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", default="avengine-qa-20260906")
    parser.add_argument(
        "--qa-ids",
        help="comma-separated QA numbers; default is all QA-01 through QA-24",
    )
    parser.add_argument(
        "--facts-out",
        type=Path,
        help="optional sibling artifact for normalized facts",
    )
    parser.add_argument(
        "--include-facts",
        action="store_true",
        help="embed normalized facts in the question output",
    )
    args = parser.parse_args(argv)
    try:
        raw = _load_bundle(args)
        result = build(
            input_path=args.input.resolve() if args.input else None,
            raw=raw,
            output_path=args.out.resolve(),
            seed=args.seed,
            qa_ids=_qa_ids(args.qa_ids),
            facts_output=args.facts_out.resolve() if args.facts_out else None,
            include_facts=args.include_facts,
        )
    except (OSError, ValueError, UnifiedQAError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "episode_id": result["episode_id"],
                "counts": result["counts"],
                "output": str(args.out.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
