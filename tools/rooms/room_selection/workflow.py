"""Calibration/freeze/one-evaluation workflow for the merged measurement layer.

The persistent evaluation receipt prevents accidental repeated inspection or
post-holdout tuning. This is the second historical use of the same holdout,
not a new statistically independent test set.
"""

from __future__ import annotations
import argparse
import hashlib
import itertools
import shutil
import json
from pathlib import Path
from types import SimpleNamespace
import yaml
from .geometry import dominant_layer
from .protocol import decide
from .analysis import agreement, summarize
from .run import load_parameters, write_json, now, latest_tasks

MEASUREMENT_FILES = [
    "room_screening/geometry.py",
    "room_screening/compute_semantic_region_candidates.py",
    "room_screening/build_inventory.py",
    "room_screening/audit_furniture_obstacles.py",
    "room_selection/geometry.py",
    "room_selection/measurements.py",
    "room_selection/navigation.py",
    "room_selection/protocol.py",
    "room_selection/splitting.py",
    "room_selection/media.py",
    "room_selection/run.py",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prediction(row, p):
    if (
        row["region_id"] < 0
        or not row["semantic_region_present"]
        or (row.get("native_region") or {}).get("ambiguous_colours")
    ):
        return "review"
    chosen, _ = dominant_layer(
        row.get("floors", []), p["dominant_floor_area_fraction_min"]
    )
    if chosen is None:
        return "review"
    return decide(row["floors"][chosen]["metrics"], row.get("room_type", "unknown"), p)[
        "status"
    ]


def freeze(out):
    if (out / "freeze.json").exists():
        raise FileExistsError("merged thresholds already frozen")
    split = json.loads((out / "house_analysis_split.json").read_text())
    inputs = json.loads((out / "inputs.json").read_text())
    plan = json.loads((out / "calibration_plan.json").read_text())
    previous = json.loads((out / "previous_82382e2/agreement.json").read_text())
    rows = []
    measurements = {}
    for h in split["calibration"]:
        path = out / "houses" / f"{h}.json"
        data = json.loads(path.read_text())
        if data.get("status") != "measured":
            raise ValueError(f"calibration house not measured: {h}")
        rows.extend(data["rows"])
        snapshot = out / "calibration_measurements" / f"{h}.json"
        snapshot.parent.mkdir(exist_ok=True)
        shutil.copy2(path, snapshot)
        measurements[h] = digest(snapshot)
    if any(r["house"] not in split["calibration"] for r in rows):
        raise ValueError("holdout in calibration")
    spec, p = load_parameters(out / "thresholds.initial.yaml")
    names = list(plan["grid"])
    grid = []
    minimum_precision = previous["calibration"]["precision"] - 0.02
    for values in itertools.product(*(plan["grid"][n] for n in names)):
        q = dict(p, **dict(zip(names, values)))
        metrics = agreement(rows, set(split["calibration"]), lambda r: prediction(r, q))
        eligible = (
            metrics["recall"] >= plan["min_recall"]
            and metrics["precision"] >= minimum_precision
        )
        distance = sum(abs(q[n] - p[n]) / max(abs(p[n]), 0.01) for n in names)
        grid.append(
            dict(
                parameters={n: q[n] for n in names},
                metrics=metrics,
                eligible=eligible,
                distance_from_initial=distance,
            )
        )
    accepted = [g for g in grid if g["eligible"]]
    if not accepted:
        raise ValueError(
            "no calibration candidate meets predeclared recall/precision constraint; do not evaluate holdout"
        )
    best = max(
        accepted,
        key=lambda g: (
            g["metrics"]["precision"],
            g["metrics"]["recall"],
            g["metrics"]["agreement"],
            -g["distance_from_initial"],
        ),
    )
    for n, value in best["parameters"].items():
        spec["parameters"][n]["value"] = value
        spec["parameters"][n][
            "source"
        ] = "calibration_selection.json; predeclared calibration_plan.json; same 90 whole houses"
        spec["parameters"][n][
            "rationale"
        ] = "Shared measurement definition changed; select max precision subject to recall >=85% and prior calibration precision minus 2 percentage points. Holdout unavailable to selection."
    spec["freeze_status"] = (
        "calibration-only frozen merged protocol; historical holdout evaluation ordinal 2"
    )
    frozen = out / "thresholds.frozen.yaml"
    frozen.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))
    (out / "thresholds.used.yaml").write_text(frozen.read_text())
    code_root = Path(__file__).resolve().parents[1]
    result = dict(
        status="frozen",
        created_at_sgt=now(),
        calibration_house_count=len(split["calibration"]),
        holdout_house_count=len(split["holdout"]),
        historical_holdout_evaluation_ordinal=2,
        holdout_evaluations_this_protocol=0,
        thresholds_source=str(frozen),
        thresholds_sha256=digest(frozen),
        split_source=str(out / "house_analysis_split.json"),
        split_sha256=digest(out / "house_analysis_split.json"),
        measurement_code_sha256={
            name: digest(code_root / name) for name in MEASUREMENT_FILES
        },
        calibration_measurement_sha256=measurements,
        calibration_measurement_directory=str(out / "calibration_measurements"),
        selected_parameters=best["parameters"],
        selection_metrics=best["metrics"],
        previous_calibration_precision=previous["calibration"]["precision"],
    )
    selection = dict(
        calibration_only=True,
        plan_source=str(out / "calibration_plan.json"),
        calibration_house_count=len(split["calibration"]),
        min_recall=plan["min_recall"],
        minimum_precision=minimum_precision,
        selected=best,
        grid=grid,
        previous_calibration_metrics=previous["calibration"],
        tie_break_order=plan["selection_order"],
    )
    write_json(out / "calibration_selection.json", selection)
    write_json(out / "freeze.json", result)
    inputs["parameters"] = {k: v["value"] for k, v in spec["parameters"].items()}
    inputs["threshold_spec"] = spec
    inputs["calibration_selection"] = selection
    inputs["previous_output"] = json.loads((out / "scope_lineage.json").read_text())[
        "previous_output"
    ]
    write_json(out / "inputs.json", inputs)
    print("FROZEN", json.dumps(best, ensure_ascii=False), flush=True)
    return result


def verify_freeze(out):
    record = json.loads((out / "freeze.json").read_text())
    if digest(out / "thresholds.frozen.yaml") != record["thresholds_sha256"]:
        raise ValueError("frozen threshold file changed")
    if digest(out / "house_analysis_split.json") != record["split_sha256"]:
        raise ValueError("fixed house split changed")
    code_root = Path(__file__).resolve().parents[1]
    for name, value in record["measurement_code_sha256"].items():
        if digest(code_root / name) != value:
            raise ValueError(f"measurement code changed since freeze: {name}")
    return record


def evaluate(out):
    # Exclusive creation occurs before ANY label statistics. A failed attempt
    # remains visible; do not delete/retry the receipt to improve a result.
    freeze_record = verify_freeze(out)
    # Check run completeness before allocating an evaluation. This reads
    # execution status only, so a missing worker cannot consume label exposure.
    inputs = json.loads((out / "inputs.json").read_text())
    for house in inputs["houses"]:
        data = json.loads((out / "houses" / f"{house}.json").read_text())
        if data.get("status") != "measured":
            raise ValueError(f"full-run house not measured: {house}")
        if any(
            "STAGE2_NOT_REQUESTED" in row["stage2"].get("reason_codes", [])
            for row in data["rows"]
        ):
            raise ValueError(
                f"stage-1 calibration cache is not a full frozen run: {house}"
            )
    receipt_path = out / "holdout_evaluation_once.json"
    receipt = dict(
        status="started",
        evaluation_count=1,
        historical_evaluation_ordinal=2,
        started_at_sgt=now(),
        thresholds_source=freeze_record.get("thresholds_source"),
        statement="Second historical evaluation of same held-out houses; one evaluation of this merged frozen protocol. No post-holdout tuning.",
    )
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    inputs = json.loads((out / "inputs.json").read_text())
    rows = []
    from .runtime import TASKS_ROOT

    gates, _ = latest_tasks(TASKS_ROOT)
    for h in inputs["houses"]:
        path = out / "houses" / f"{h}.json"
        data = json.loads(path.read_text())
        if data.get("status") != "measured":
            raise ValueError(f"full-run house not measured: {h}")
        for row in data["rows"]:
            row["stage4"] = gates.get(h, row["stage4"])
        # Freeze applied during measurement; catch a stale initial-calibration cache.
        for row in data["rows"]:
            expected = prediction(row, inputs["parameters"])
            if expected != row["stage1"]["status"]:
                raise ValueError(f'stale decision {h}/{row["room_label"]}')
        write_json(path, data)
        rows.extend(data["rows"])
    inputs["latest_gate_counts"] = dict(
        __import__("collections").Counter(gates[h]["status"] for h in inputs["houses"])
    )
    write_json(out / "inputs.json", inputs)
    with (out / "rooms_registry.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    summarize(
        SimpleNamespace(
            output=out,
            inventory=Path(inputs["inventory_source"]),
            manual_splits=Path(inputs["manual_splits_source"]),
            clean_houses=Path(inputs["clean_houses_source"]),
        ),
        rows,
        inputs,
    )
    current = json.loads((out / "agreement.json").read_text())
    previous = json.loads((out / "previous_82382e2/agreement.json").read_text())
    comparison = dict(
        previous_commit="82382e2",
        previous_source=str(out / "previous_82382e2/agreement.json"),
        new_source=str(out / "agreement.json"),
        historical_holdout_evaluation_ordinal=2,
        old_holdout=previous["holdout"],
        new_holdout=current["holdout"],
        old_calibration=previous["calibration"],
        new_calibration=current["calibration"],
        changes_percentage_points={
            k: 100 * (current["holdout"][k] - previous["holdout"][k])
            for k in ["agreement", "precision", "recall"]
        },
    )
    write_json(out / "holdout_comparison.json", comparison)
    receipt.update(status="complete", finished_at_sgt=now(), metrics=current["holdout"])
    write_json(receipt_path, receipt)
    print(
        "HOLDOUT_ONCE", json.dumps(current["holdout"], ensure_ascii=False), flush=True
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["freeze", "verify-freeze", "evaluate", "publish"])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.stage == "freeze":
        freeze(a.output)
    elif a.stage == "verify-freeze":
        print(json.dumps(verify_freeze(a.output), ensure_ascii=False))
    elif a.stage == "evaluate":
        evaluate(a.output)
    else:
        from .report import publish

        publish(a.output)


if __name__ == "__main__":
    main()
