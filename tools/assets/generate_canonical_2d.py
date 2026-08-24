"""Generate one canonical 2D animal candidate from a clay pose guide.

The prompt is composed from the shared blocks in the prompt budget spec plus the
breed's own positive text, and is refused before any model is loaded when it
does not fit the model window.  One invocation, one image, no seed lottery and
no candidate ranking: a failed candidate is preserved as evidence, not retried.

Run with the image-generation environment, for example::

    $IMAGEGEN_PY tools/assets/generate_canonical_2d.py \
        --identity <fresh>/identity.json --clay-guide <guide>.png \
        --output-dir <fresh>/gate1 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUDGET_SPEC = REPOSITORY_ROOT / "examples/assets/prompt_budget_v1.json"
DEFAULT_MODEL_ROOT = Path(
    "/data/models/hub/models--black-forest-labs--FLUX.2-klein-base-4B"
)
EFFECTIVE_PROMPT_FORMAT = "{prompt} Avoid: {negative}."
CANVAS = 1024


class GenerationError(RuntimeError):
    """The request cannot be executed as specified."""


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_snapshot(model_root: Path) -> Path:
    snapshots = model_root / "snapshots"
    if not snapshots.is_dir():
        raise GenerationError(f"no snapshots directory under {model_root}")
    candidates = sorted(path for path in snapshots.iterdir() if path.is_dir())
    if len(candidates) != 1:
        raise GenerationError(
            f"expected exactly one snapshot under {snapshots}, found {len(candidates)}"
        )
    return candidates[0]


def compose_prompt(identity: Mapping[str, Any], budget: Mapping[str, Any]) -> dict[str, str]:
    positive = identity.get("positive_prompt")
    if not isinstance(positive, str) or not positive.strip():
        raise GenerationError("identity.positive_prompt is missing or empty")
    blocks = budget["shared_blocks"]
    prompt = f"{positive.strip()} {blocks['pose_guard_prompt'].strip()}"
    effective = EFFECTIVE_PROMPT_FORMAT.format(
        prompt=prompt, negative=blocks["negative_prompt"].strip()
    )
    return {"positive": positive.strip(), "prompt": prompt, "effective": effective}


def enforce_budget(
    parts: Mapping[str, str], budget: Mapping[str, Any], tokenizer: Any
) -> dict[str, int]:
    caps = budget["caps_tokens"]
    window = int(budget["model_window"]["max_sequence_length"])

    def count(text: str) -> int:
        return len(tokenizer(text)["input_ids"])

    counts = {
        "positive": count(parts["positive"]),
        "pose_guard": count(budget["shared_blocks"]["pose_guard_prompt"]),
        "negative": count(budget["shared_blocks"]["negative_prompt"]),
        "prompt": count(parts["prompt"]),
        "effective": count(parts["effective"]),
        "window": window,
    }
    failures = []
    if counts["positive"] > caps["positive_template_rendered"]:
        failures.append(
            f"positive {counts['positive']} > cap {caps['positive_template_rendered']}"
        )
    if counts["effective"] > window:
        failures.append(f"effective {counts['effective']} > window {window}")
    if failures:
        raise GenerationError("; ".join(failures) + " -- shorten the prompt, never the window")
    return counts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--clay-guide", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-spec", type=Path, default=DEFAULT_BUDGET_SPEC)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument(
        "--owner-gate-waived",
        action="store_true",
        help="record that the owner waived the canonical-image review for this run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.output_dir.exists():
            raise GenerationError(f"{args.output_dir} exists; choose a fresh path")
        identity = _load_json(args.identity)
        budget = _load_json(args.budget_spec)
        parts = compose_prompt(identity, budget)
        snapshot = _resolve_snapshot(args.model_root)
        if not args.clay_guide.is_file():
            raise GenerationError(f"clay guide not found: {args.clay_guide}")

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot / "tokenizer"), local_files_only=True
        )
        counts = enforce_budget(parts, budget, tokenizer)
        print(json.dumps(counts, ensure_ascii=False), flush=True)
    except GenerationError as error:
        print(f"generation refused: {error}", file=sys.stderr)
        return 2

    import torch
    from diffusers import Flux2KleinPipeline
    from PIL import Image

    guide_sha = _sha256_file(args.clay_guide)
    with Image.open(args.clay_guide) as opened:
        opened.load()
        if opened.size != (CANVAS, CANVAS):
            print(
                f"generation refused: clay guide must be {CANVAS}x{CANVAS}, got {opened.size}",
                file=sys.stderr,
            )
            return 2
        guide = opened.convert("RGB")

    print(f"loading {snapshot}", flush=True)
    pipeline = Flux2KleinPipeline.from_pretrained(
        str(snapshot), torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    generator = torch.Generator("cuda").manual_seed(args.seed)
    result = pipeline(
        image=guide,
        prompt=parts["effective"],
        width=CANVAS,
        height=CANVAS,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        max_sequence_length=counts["window"],
    )
    images = getattr(result, "images", None)
    if not images or len(images) != 1:
        print("generation refused: the pipeline must return exactly one image", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "candidate.png"
    images[0].convert("RGB").save(candidate_path)
    manifest = {
        "schema": "avengine_canonical_2d_candidate_v1",
        "identity": identity,
        "clay_guide": {"path": str(args.clay_guide), "sha256": guide_sha},
        "model": {
            "root": str(args.model_root),
            "snapshot": snapshot.name,
            "distilled": "base" not in args.model_root.name,
        },
        "parameters": {
            "seed": args.seed,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "width": CANVAS,
            "height": CANVAS,
            "max_sequence_length": counts["window"],
        },
        "prompt": parts,
        "token_counts": counts,
        "one_shot_execution": {
            "invocations_allowed": 1,
            "seed_retry_allowed": False,
            "candidate_ranking_allowed": False,
        },
        "gate": {
            "state": "owner_review_waived_for_pipeline_shakedown"
            if args.owner_gate_waived
            else "awaiting_owner_review",
            "registrable": False,
        },
        "candidate": {"path": str(candidate_path), "sha256": _sha256_file(candidate_path)},
    }
    (args.output_dir / "candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"wrote {candidate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
