"""Derive the instance-level variant plan for accepted source assets.

Nothing about a species, breed or coat is written here.  Axis domains come from
``avengine.appearance.contracts``, realized attributes come from the runtime
profile registry, and derived asset ids follow one rule that the unit tests
round-trip against every registered generated asset.  A breed that is not
registered in the contract fails closed instead of being guessed.

Example:
    $PY tools/assets/plan_instance_variants.py \
        --asset-id generated_british_shorthair_blue_medium_stocky_adult_research_v1 \
        --output /data/avengine_external/review/<fresh>/plan.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from avengine.appearance.contracts import (  # noqa: E402
    CANONICAL_DOMAINS,
    COAT_PROFILE_DOMAINS,
    OPERATION_BY_AXIS,
    REALIZER_PARAMETER_BOUNDS,
)
from avengine.contracts.json_io import canonical_json_sha256  # noqa: E402

PLAN_SCHEMA = "avengine_instance_variant_plan_v1"
FANOUT_SPEC_SCHEMA = "avengine_instance_fanout_axes_v1"
DEFAULT_REGISTRY = REPOSITORY_ROOT / "examples/runtime/source_asset_runtime_profiles.json"
DEFAULT_FANOUT_SPEC = REPOSITORY_ROOT / "examples/assets/instance_fanout_axes_v1.json"

# The one operation that changes no geometry and no material: a size row can be
# derived by writing a runtime scale, without importing anything into UE.
RUNTIME_ONLY_OPERATION = "uniform_actor_scale_v1"


class PlanError(ValueError):
    """The requested asset or spec cannot produce a derivable plan."""


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_fanout_spec(path: Path) -> Mapping[str, Any]:
    spec = _load_json(path)
    if spec.get("schema") != FANOUT_SPEC_SCHEMA:
        raise PlanError(f"{path} is not a {FANOUT_SPEC_SCHEMA} document")
    enabled = tuple(spec.get("enabled_axes") or ())
    pinned = dict(spec.get("pinned_axes") or {})
    if not enabled:
        raise PlanError("the fan-out spec enables no axes")
    unknown = [axis for axis in enabled if axis not in OPERATION_BY_AXIS]
    unknown += [axis for axis in pinned if axis not in OPERATION_BY_AXIS]
    if unknown:
        raise PlanError(f"unknown appearance axes in the fan-out spec: {sorted(unknown)}")
    overlap = sorted(set(enabled) & set(pinned))
    if overlap:
        raise PlanError(f"axes cannot be enabled and pinned at once: {overlap}")
    missing = sorted(set(OPERATION_BY_AXIS) - set(enabled) - set(pinned))
    if missing:
        raise PlanError(f"the fan-out spec leaves axes undecided: {missing}")
    return {"spec_id": spec.get("spec_id"), "revision": spec.get("revision"),
            "enabled_axes": enabled, "pinned_axes": pinned}


def _source_attributes(entry: Mapping[str, Any]) -> dict[str, str]:
    realized = entry.get("realized_attributes") or {}
    coat = realized.get("coat_profile") or {}
    if not isinstance(coat, Mapping) or "value" not in coat or "profile_id" not in coat:
        raise PlanError(
            f"{entry.get('asset_id')} carries no coat_profile with profile_id and value; "
            "instance fan-out only applies to assets realized through the appearance contract"
        )
    attributes = {axis: realized.get(axis) for axis in ("size", "body_build", "life_stage")}
    missing = sorted(axis for axis, value in attributes.items() if not value)
    if missing:
        raise PlanError(f"{entry.get('asset_id')} is missing realized attributes: {missing}")
    attributes["coat_profile"] = coat["value"]
    attributes["_coat_profile_id"] = coat["profile_id"]
    return attributes


def _coat_domain(entry: Mapping[str, Any], profile_id: str) -> tuple[str, ...]:
    identity = entry.get("identity") or {}
    species = identity.get("species_id")
    breed = identity.get("breed_id")
    key = (species, breed, profile_id)
    domain = COAT_PROFILE_DOMAINS.get(key)
    if domain is None:
        raise PlanError(
            f"no reviewed coat domain is registered for {key}; add that breed's exact "
            "three-level domain to avengine.appearance.contracts before planning"
        )
    return tuple(domain)


def _axis_domains(
    entry: Mapping[str, Any], attributes: Mapping[str, str], enabled: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    domains: dict[str, tuple[str, ...]] = {}
    for axis in enabled:
        if axis == "coat_profile":
            domains[axis] = _coat_domain(entry, attributes["_coat_profile_id"])
        else:
            domain = CANONICAL_DOMAINS.get(axis)
            if not domain:
                raise PlanError(f"the appearance contract registers no domain for axis {axis}")
            domains[axis] = tuple(domain)
        if attributes[axis] not in domains[axis]:
            raise PlanError(
                f"{entry.get('asset_id')} carries {axis}={attributes[axis]!r} which is outside "
                f"the registered domain {domains[axis]}"
            )
    return domains


def _id_decomposition(asset_id: str, attributes: Mapping[str, str]) -> tuple[str, str, str]:
    """Split a registered asset id into (prefix, coat token, tail).

    The convention is observed on the asset itself rather than rebuilt from the
    breed id: some registered ids abbreviate the breed.  The coat token is
    whatever sits in front of the size/body_build/life_stage triple.
    """
    triple = "_".join((attributes["size"], attributes["body_build"], attributes["life_stage"]))
    marker = f"_{triple}"
    index = asset_id.rfind(marker)
    if index < 0:
        raise PlanError(
            f"{asset_id} does not encode its realized attributes ({triple}); "
            "the derivation rule cannot be applied to it"
        )
    head = asset_id[:index]
    tail = asset_id[index + len(marker):]
    coat_token = head.rsplit("_", 1)[-1] if "_" in head else head
    hue = attributes["coat_profile"]
    for separator_count in range(hue.count("_") + 1, 0, -1):
        candidate = "_".join(head.rsplit("_", separator_count)[-separator_count:])
        if hue.endswith(candidate):
            coat_token = candidate
            break
    prefix = head[: len(head) - len(coat_token)]
    return prefix, coat_token, tail


def _derive_asset_id(
    prefix: str,
    tail: str,
    source_coat_token: str,
    source_coat_value: str,
    combo: Mapping[str, str],
    pinned: Mapping[str, str],
) -> str:
    coat_value = combo["coat_profile"]
    token = source_coat_token if coat_value == source_coat_value else coat_value
    triple = (combo["size"], pinned["body_build"], pinned["life_stage"])
    return f"{prefix}{token}_" + "_".join(triple) + tail


def plan_for_entry(entry: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    attributes = _source_attributes(entry)
    enabled = list(spec["enabled_axes"])
    pinned_axes = {axis: attributes[axis] for axis in spec["pinned_axes"]}
    domains = _axis_domains(entry, attributes, enabled)
    prefix, coat_token, tail = _id_decomposition(entry["asset_id"], attributes)

    scale_axes = [axis for axis in enabled if OPERATION_BY_AXIS[axis] == RUNTIME_ONLY_OPERATION]
    asset_axes = [axis for axis in enabled if OPERATION_BY_AXIS[axis] != RUNTIME_ONLY_OPERATION]

    rows: list[dict[str, Any]] = []
    for values in itertools.product(*(domains[axis] for axis in enabled)):
        combo = dict(zip(enabled, values, strict=True))
        changed = sorted(axis for axis in enabled if combo[axis] != attributes[axis])
        rows.append(
            {
                "asset_id": _derive_asset_id(
                    prefix, tail, coat_token, attributes["coat_profile"], combo, pinned_axes
                ),
                "realized_attributes": {
                    **combo,
                    **pinned_axes,
                    "coat_profile_id": attributes["_coat_profile_id"],
                },
                "changed_axes": changed,
                "realization_operations": {axis: OPERATION_BY_AXIS[axis] for axis in changed},
            }
        )

    # Rows agreeing on every non-scale axis share one produced asset.  Inside a
    # group the row that keeps the source's scale values is the one the
    # appearance realization actually produces; its siblings are pure runtime
    # scales of it, so they cost nothing beyond a registry entry.
    groups: list[dict[str, Any]] = []
    by_key: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row["realized_attributes"][axis] for axis in asset_axes)
        by_key.setdefault(key, []).append(row)
    source_key = tuple(attributes[axis] for axis in asset_axes)
    for key, members in by_key.items():
        representative = next(
            row
            for row in members
            if all(row["realized_attributes"][axis] == attributes[axis] for axis in scale_axes)
        )
        reuses = key == source_key
        for row in members:
            is_representative = row is representative
            row["already_registered"] = reuses and is_representative
            row["requires_new_ue_asset"] = is_representative and not reuses
            row["runtime_only_derivation"] = not is_representative
            row["derived_from"] = None if is_representative else representative["asset_id"]
        groups.append(
            {
                "group": dict(zip(asset_axes, key, strict=True)),
                "representative_asset_id": representative["asset_id"],
                "asset_ids": [row["asset_id"] for row in members],
                "reuses_existing_ue_asset": reuses,
            }
        )

    return {
        "source_asset": {
            "asset_id": entry["asset_id"],
            "revision": entry.get("revision"),
            "identity": entry.get("identity"),
            "realized_attributes": {
                axis: attributes[axis] for axis in ("size", "body_build", "life_stage", "coat_profile")
            },
            "coat_profile_id": attributes["_coat_profile_id"],
            "admission_state": entry.get("admission_state"),
        },
        "enabled_axes": enabled,
        "pinned_axes": pinned_axes,
        "axis_domains": {axis: list(domains[axis]) for axis in enabled},
        "rows": rows,
        "ue_import_groups": groups,
        "summary": {
            "rows": len(rows),
            "already_registered": sum(1 for row in rows if row["already_registered"]),
            "runtime_only_derivations": sum(1 for row in rows if row["runtime_only_derivation"]),
            "ue_imports_required": sum(1 for row in rows if row["requires_new_ue_asset"]),
        },
    }


def build_plan(
    registry_path: Path, fanout_spec_path: Path, asset_ids: Sequence[str] | None
) -> dict[str, Any]:
    registry = _load_json(registry_path)
    spec = _load_fanout_spec(fanout_spec_path)
    entries = {entry["asset_id"]: entry for entry in registry.get("assets", [])}

    if asset_ids:
        missing = [asset_id for asset_id in asset_ids if asset_id not in entries]
        if missing:
            raise PlanError(f"asset ids absent from {registry_path}: {missing}")
        selected = [entries[asset_id] for asset_id in asset_ids]
    else:
        selected = list(registry.get("assets", []))

    plans: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for entry in selected:
        try:
            plans.append(plan_for_entry(entry, spec))
        except PlanError as error:
            if asset_ids:
                raise
            skipped.append({"asset_id": entry.get("asset_id", "<unnamed>"), "reason": str(error)})
    if not plans:
        raise PlanError(f"no asset in {registry_path} can produce a derivable plan")
    payload = {
        "schema": PLAN_SCHEMA,
        "fanout_spec": {"spec_id": spec["spec_id"], "revision": spec["revision"]},
        "source_asset_registry": str(registry_path.relative_to(REPOSITORY_ROOT))
        if registry_path.is_relative_to(REPOSITORY_ROOT)
        else str(registry_path),
        "realizer_parameter_bounds": {
            name: list(bounds) for name, bounds in REALIZER_PARAMETER_BOUNDS.items()
        },
        "plans": plans,
        "skipped": skipped,
        "totals": {
            "source_assets": len(plans),
            "rows": sum(plan["summary"]["rows"] for plan in plans),
            "ue_imports_required": sum(plan["summary"]["ue_imports_required"] for plan in plans),
            "runtime_only_derivations": sum(
                plan["summary"]["runtime_only_derivations"] for plan in plans
            ),
        },
    }
    payload["plan_content_sha256"] = canonical_json_sha256(payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-asset-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--fanout-spec", type=Path, default=DEFAULT_FANOUT_SPEC)
    parser.add_argument(
        "--asset-id",
        action="append",
        dest="asset_ids",
        help="plan this asset; repeatable. Omit to plan every eligible asset.",
    )
    parser.add_argument("--output", type=Path, help="fresh path for the plan; never overwritten")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = build_plan(args.source_asset_registry, args.fanout_spec, args.asset_ids)
    except PlanError as error:
        print(f"plan refused: {error}", file=sys.stderr)
        return 2
    if args.output is not None:
        if args.output.exists():
            print(f"plan refused: {args.output} already exists; choose a fresh path", file=sys.stderr)
            return 2
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {args.output}")
    for skip in payload["skipped"]:
        print(f"skipped {skip['asset_id']}: {skip['reason']}")
    for plan in payload["plans"]:
        summary = plan["summary"]
        print(
            f"{plan['source_asset']['asset_id']}: {summary['rows']} rows, "
            f"{summary['ue_imports_required']} UE imports, "
            f"{summary['runtime_only_derivations']} runtime-only, "
            f"{summary['already_registered']} already registered"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
