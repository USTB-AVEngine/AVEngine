"""Fail closed when a candidate profile's effective prompt cannot fit the model window.

The FLUX.2 worker concatenates the negative prompt into the positive one
(``f"{prompt} Avoid: {negative}."``) and then passes a fixed
``max_sequence_length``.  Anything past that window is dropped silently, and the
dropped tail is the negative clause -- exactly the constraints that keep the
canonical image reconstruction-safe.  This checker composes the same string,
counts tokens with the model's own tokenizer, and refuses any profile that does
not fit.

Every sampled value combination is checked, not just one, so a profile is only
accepted when *no* combination can overflow.  Nothing about a species, breed or
coat appears here.

Example:
    $PY tools/assets/check_prompt_token_budget.py \
        --profile-dir <spear>/data/controlled_source_attributes_v1/candidate_profiles/animal \
        --report <fresh>/prompt_token_budget.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import string
import sys
from typing import Any, Mapping, Sequence

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from model_roots import resolve as resolve_model_root  # noqa: E402

MODEL_ENTRY = "flux2_klein_tokenizer"
DEFAULT_MAX_SEQUENCE_LENGTH = 512
# The worker's own joiner.  Kept here verbatim so the budget is measured on the
# string the pipeline actually receives.
EFFECTIVE_PROMPT_FORMAT = "{prompt} Avoid: {negative}."


class BudgetError(ValueError):
    """A profile cannot be measured, or does not fit the window."""


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _labels_for(profile: Mapping[str, Any]) -> Mapping[str, Mapping[str, str]]:
    contract = profile.get("generation_contract") or {}
    labels = contract.get("value_labels") or {}
    if not isinstance(labels, Mapping):
        raise BudgetError("generation_contract.value_labels must be a mapping")
    return labels


def _attribute_axes(profile: Mapping[str, Any]) -> dict[str, list[str]]:
    """Every placeholder value the positive template can be filled with."""
    axes: dict[str, list[str]] = {}
    for key, value in (profile.get("taxonomy") or {}).items():
        axes[key] = [value]
    for key, value in (profile.get("fixed_attributes") or {}).items():
        axes[key] = [value]
    for key, values in (profile.get("sampled_attribute_domains") or {}).items():
        if not values:
            raise BudgetError(f"sampled_attribute_domains.{key} is empty")
        axes[key] = list(values)
    return axes


def _render(template: str, values: Mapping[str, str], labels: Mapping[str, Mapping[str, str]]) -> str:
    fields = {name for _, name, _, _ in string.Formatter().parse(template) if name}
    missing = sorted(fields - set(values))
    if missing:
        raise BudgetError(f"template placeholders without a value: {missing}")
    rendered = {}
    for key in fields:
        raw = values[key]
        mapping = labels.get(key) or {}
        rendered[key] = mapping.get(raw, raw)
    return template.format(**rendered)


def measure_profile(
    profile: Mapping[str, Any], tokenizer: Any, max_sequence_length: int
) -> dict[str, Any]:
    contract = profile.get("generation_contract") or {}
    for field in ("positive_template", "pose_guard_prompt", "negative_prompt"):
        if not isinstance(contract.get(field), str) or not contract[field].strip():
            raise BudgetError(f"generation_contract.{field} is missing or empty")

    labels = _labels_for(profile)
    axes = _attribute_axes(profile)
    names = sorted(axes)

    def count(text: str) -> int:
        return len(tokenizer(text)["input_ids"])

    pose_guard_tokens = count(contract["pose_guard_prompt"])
    negative_tokens = count(contract["negative_prompt"])

    combinations: list[dict[str, Any]] = []
    for values in itertools.product(*(axes[name] for name in names)):
        combo = dict(zip(names, values, strict=True))
        positive = _render(contract["positive_template"], combo, labels)
        prompt = f"{positive} {contract['pose_guard_prompt']}"
        effective = EFFECTIVE_PROMPT_FORMAT.format(
            prompt=prompt, negative=contract["negative_prompt"]
        )
        effective_tokens = count(effective)
        combinations.append(
            {
                "sampled_values": {
                    key: combo[key]
                    for key in sorted(profile.get("sampled_attribute_domains") or {})
                },
                "positive_tokens": count(positive),
                "prompt_tokens": count(prompt),
                "effective_tokens": effective_tokens,
                "overflow_tokens": max(0, effective_tokens - max_sequence_length),
                "fits": effective_tokens <= max_sequence_length,
            }
        )

    worst = max(combinations, key=lambda item: item["effective_tokens"])
    return {
        "profile_schema_id": profile.get("profile_schema_id"),
        "profile_revision": profile.get("profile_revision"),
        "taxonomy": profile.get("taxonomy"),
        "max_sequence_length": max_sequence_length,
        "pose_guard_tokens": pose_guard_tokens,
        "negative_tokens": negative_tokens,
        "combinations_checked": len(combinations),
        "worst_case": worst,
        "fits": all(item["fits"] for item in combinations),
        "combinations": combinations,
    }


def _load_tokenizer(root: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover - environment dependent
        raise BudgetError(
            "transformers is unavailable; run this with the image-generation environment"
        ) from error
    if not root.exists():
        raise BudgetError(f"tokenizer root does not exist: {root}")
    return AutoTokenizer.from_pretrained(str(root), local_files_only=True)


def _profile_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = list(args.profiles or [])
    for directory in args.profile_dirs or []:
        paths.extend(sorted(directory.rglob("*.json")))
    if not paths:
        raise BudgetError("no profile was given; pass --profile or --profile-dir")
    return paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append", dest="profiles", type=Path)
    parser.add_argument("--profile-dir", action="append", dest="profile_dirs", type=Path)
    parser.add_argument("--tokenizer-root", type=Path, default=None)
    parser.add_argument(
        "--max-sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH
    )
    parser.add_argument("--report", type=Path, help="fresh path; never overwritten")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # Resolved after parsing: an explicit path must not require the
    # registry to exist.
    args.tokenizer_root = resolve_model_root(MODEL_ENTRY, override=args.tokenizer_root)
    try:
        paths = _profile_paths(args)
        tokenizer = _load_tokenizer(args.tokenizer_root)
    except BudgetError as error:
        print(f"budget check refused: {error}", file=sys.stderr)
        return 2

    results: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for path in paths:
        try:
            measured = measure_profile(_load_json(path), tokenizer, args.max_sequence_length)
        except BudgetError as error:
            skipped.append({"path": str(path), "reason": str(error)})
            continue
        measured["path"] = str(path)
        results.append(measured)

    if not results:
        print("budget check refused: no profile could be measured", file=sys.stderr)
        for skip in skipped:
            print(f"  skipped {skip['path']}: {skip['reason']}", file=sys.stderr)
        return 2

    over = [item for item in results if not item["fits"]]
    payload = {
        "schema": "avengine_prompt_token_budget_report_v1",
        "max_sequence_length": args.max_sequence_length,
        "tokenizer_root": str(args.tokenizer_root),
        "effective_prompt_format": EFFECTIVE_PROMPT_FORMAT,
        "profiles_measured": len(results),
        "profiles_over_budget": len(over),
        "profiles": results,
        "skipped": skipped,
    }
    if args.report is not None:
        if args.report.exists():
            print(f"budget check refused: {args.report} exists", file=sys.stderr)
            return 2
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {args.report}")

    for item in sorted(results, key=lambda entry: -entry["worst_case"]["effective_tokens"]):
        worst = item["worst_case"]
        state = "fits" if item["fits"] else f"OVER by {worst['overflow_tokens']}"
        print(
            f"{state:>14}  effective {worst['effective_tokens']:>4}"
            f"  (prompt {worst['prompt_tokens']:>4} + negative {item['negative_tokens']:>3})"
            f"  x{item['combinations_checked']}  {item['profile_schema_id']}"
            f"  [{Path(item['path']).name}]"
        )
    for skip in skipped:
        print(f"        skipped  {Path(skip['path']).name}: {skip['reason']}")
    return 1 if over else 0


if __name__ == "__main__":
    raise SystemExit(main())
