#!/usr/bin/env python3
"""Read, inspect and score a self-contained V1 QA dataset export.

Every subcommand needs the export root and nothing else, so this is also the
portability check: run it from an unrelated working directory.

    inspect_qa_dataset.py list      --root DIR [--split S] [--room R] [--qa QA-07]
    inspect_qa_dataset.py layouts   --root DIR
    inspect_qa_dataset.py media     --root DIR --sample ID [--audio-layout L]
    inspect_qa_dataset.py input     --root DIR --sample ID [--form mcq] [--language en]
    inspect_qa_dataset.py coverage  --root DIR [--split S]
    inspect_qa_dataset.py verify    --root DIR
    inspect_qa_dataset.py score     --root DIR --predictions FILE [--form mcq]
    inspect_qa_dataset.py gold-replay --root DIR [--form mcq]
    inspect_qa_dataset.py groups    --root DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[2]
if (REPOSITORY / "src" / "avengine").is_dir():
    sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.dataset.qa_dataset_reader import (  # noqa: E402
    FORMS,
    LANGUAGES,
    QaDatasetReadError,
    open_qa_dataset,
)


def _emit(value: object, out: Path | None) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    if out is None:
        print(text)
        return
    resolved = out.expanduser().resolve()
    if resolved.exists():
        raise QaDatasetReadError(f"refusing to overwrite output: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(text + "\n", encoding="utf-8")
    print(json.dumps({"out": str(resolved)}))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", required=True, type=Path, help="dataset export root")
    parser.add_argument("--out", type=Path, default=None, help="write JSON here instead of stdout")
    parser.add_argument("--form", choices=FORMS, default=None)
    parser.add_argument("--language", choices=LANGUAGES, default=None)
    parser.add_argument("--audio-layout", default=None)
    parser.add_argument("--video-view", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--room", default=None, help="room_family filter")
    parser.add_argument("--qa", default=None, help="QA id filter, e.g. QA-07")
    parser.add_argument("--record-kind", default=None, choices=("core_group_member", "episode"))
    parser.add_argument("--sample", default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "command",
        choices=(
            "list", "layouts", "media", "input", "coverage", "verify",
            "score", "gold-replay", "groups",
        ),
    )
    args = parser.parse_args(argv)

    config: dict[str, object] = {}
    for key, value in (
        ("form", args.form), ("language", args.language),
        ("audio_layout", args.audio_layout), ("video_view", args.video_view),
        ("split", args.split),
    ):
        if value is not None:
            config[key] = value
    try:
        reader = open_qa_dataset(args.root, config=config or None)
        if args.command == "list":
            ids = reader.list_samples(
                room_family=args.room, qa_id=args.qa, record_kind=args.record_kind,
                audio_layout=args.audio_layout, form=args.form,
            )
            if args.limit is not None:
                ids = ids[: args.limit]
            _emit(
                {
                    "root": str(reader.root),
                    "dataset_version": reader.dataset_version,
                    "counts": reader.counts,
                    "splits": {name: len(rows) for name, rows in reader.splits().items()},
                    "selected_sample_count": len(ids),
                    "samples": [
                        {
                            "sample_id": sample_id,
                            "record_kind": reader.sample(sample_id)["record_kind"],
                            "room_family": reader.sample(sample_id)["room_family"],
                            "world_key": reader.sample(sample_id)["world_key"],
                            "split": reader.sample(sample_id)["split"],
                            "audio_layouts": sorted(
                                (reader.sample(sample_id)["media"].get("audio") or {})
                            ),
                            "valid_main_questions": len(
                                reader.questions(sample_id, kinds=("main",))
                            ),
                            "valid_angle_followups": len(
                                reader.questions(sample_id, kinds=("angle_followup",))
                            ),
                        }
                        for sample_id in ids
                    ],
                },
                args.out,
            )
        elif args.command == "layouts":
            _emit(
                {
                    "declared_layouts": list(reader.audio_layouts()),
                    "declarations": {
                        layout: reader.layout_declaration(layout)
                        for layout in reader.audio_layouts()
                    },
                    "observation_protocols": reader.observation_protocols(),
                    "note": (
                        "Read channel order, normalization and coordinate frame from the "
                        "declaration. Nothing here is assumed."
                    ),
                },
                args.out,
            )
        elif args.command == "media":
            if not args.sample:
                parser.error("media needs --sample")
            _emit(reader.media(args.sample), args.out)
        elif args.command == "input":
            if not args.sample:
                parser.error("input needs --sample")
            _emit(reader.model_input(args.sample), args.out)
        elif args.command == "coverage":
            _emit(reader.coverage(form=args.form), args.out)
        elif args.command == "verify":
            _emit(
                {
                    "portability": reader.verify_self_contained(),
                    "world_accounting": reader.verify_world_accounting(),
                    "counts": reader.counts,
                },
                args.out,
            )
        elif args.command == "groups":
            private = reader.private()
            _emit(
                {
                    "groups": private.groups(),
                    "grouping_by_sample": {
                        sample_id: private.grouping(sample_id)
                        for sample_id in reader.list_samples()
                    },
                },
                args.out,
            )
        elif args.command == "score":
            if args.predictions is None:
                parser.error("score needs --predictions")
            payload = json.loads(args.predictions.expanduser().resolve().read_text("utf-8"))
            predictions = payload.get("predictions", payload)
            result = reader.private().score(predictions, form=args.form)
            summary = {key: value for key, value in result.items() if key != "records"}
            _emit(result if args.out else summary, args.out)
        else:
            _emit(reader.private().gold_replay_smoke(form=args.form), args.out)
    except (QaDatasetReadError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
