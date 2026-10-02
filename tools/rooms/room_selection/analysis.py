"""House-held-out agreement, split proxy IoU, funnel, human queues and report."""

from __future__ import annotations

import itertools
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import shapely
from shapely.geometry import shape, box
from scipy.optimize import linear_sum_assignment

from .protocol import decide

REASONS = {
    "TRANSIT_OR_STAIRS": "过道或楼梯",
    "SCAN_INCOMPLETE": "扫描残缺",
    "TOO_SMALL": "过小",
    "MULTIROOM_UNSPLITTABLE": "多房间不可切",
    "NON_RESIDENTIAL": "非居住空间",
    "CLUTTER_OCCLUSION": "杂乱遮挡",
    "FLOOR_OR_BOUNDARY": "楼层或边界不明确",
    "PLACEMENT_UNAVAILABLE": "相机与两个声源无法合理摆放",
    "SPLIT_APPROVE": "切分建议可接受",
    "SPLIT_CORRECT": "切分建议需修正",
    "MEDIA_MISSING": "审核素材缺失",
    "OTHER": "其他（必填说明）",
}


def write_json(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def agreement(rows, house_set=None, prediction=None):
    selected = [
        r
        for r in rows
        if r.get("human") and (house_set is None or r["house"] in house_set)
    ]
    table = {
        h: {a: 0 for a in ["pass", "fail", "review", "not_run"]}
        for h in ["use", "skip", "unsure"]
    }
    for r in selected:
        predicted = prediction(r) if prediction else r["stage1"]["status"]
        table[r["human"]["verdict"]][predicted] += 1
    tp = table["use"]["pass"]
    fp = table["skip"]["pass"]
    fn = sum(table["use"].values()) - tp
    tn = sum(table["skip"].values()) - fp
    n = tp + fp + fn + tn
    resolved = sum(table[h][a] for h in ["use", "skip"] for a in ["pass", "fail"])
    resolved_correct = tp + table["skip"]["fail"]
    return dict(
        human_count=len(selected),
        human_verdicts=dict(Counter(r["human"]["verdict"] for r in selected)),
        three_way_matrix=table,
        confusion_matrix=dict(TP=tp, FP=fp, FN=fn, TN=tn),
        positive_label="human use",
        negative_label="human skip",
        unsure_policy="excluded from binary metrics, retained in three-way matrix",
        auto_review_policy="non-pass counts as not selected; conditional resolved metrics separately reported",
        binary_denominator=n,
        agreement=(tp + tn) / n if n else None,
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        automatic_resolved_count=resolved,
        automatic_resolved_fraction=resolved / n if n else None,
        agreement_on_resolved=resolved_correct / resolved if resolved else None,
    )


def analyze_disagreements(rows, holdout):
    groups = defaultdict(list)
    for r in rows:
        if (
            r["house"] not in holdout
            or not r.get("human")
            or r["human"]["verdict"] not in ["use", "skip"]
        ):
            continue
        predicted = r["stage1"]["status"] == "pass"
        human = r["human"]["verdict"] == "use"
        if predicted != human:
            groups[r.get("room_type", "unknown")].append(r)
    out = []
    for name, records in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        population = [
            r
            for r in rows
            if r["house"] in holdout
            and r.get("room_type", "unknown") == name
            and r.get("human")
            and r["human"]["verdict"] in ["use", "skip"]
        ]
        examples = []
        for r in sorted(records, key=lambda r: (r["house"], r["region_id"]))[:5]:
            examples.append(
                dict(
                    house=r["house"],
                    room_label=r["room_label"],
                    human=r["human"]["verdict"],
                    automatic=r["stage1"],
                    human_note=r["human"].get("note"),
                    human_source=r["human"]["source"],
                    overhead=r.get("overhead"),
                    video_path=r.get("video_path"),
                    metrics=r.get("metrics"),
                )
            )
        out.append(
            dict(
                room_type=name,
                disagreement_count=len(records),
                count=len(population),
                rate=len(records) / len(population) if population else None,
                examples=examples,
                example_shortfall=max(0, 5 - len(examples)),
            )
        )
    return out


def manual_iou(rows, path, p):
    if path is None or not Path(path).exists():
        return dict(status="not_run", reason="manual JSON references unavailable")
    refs = json.loads(Path(path).read_text())["records"]
    by_key = {(r["house"], r["room_label"]): r for r in rows}
    groups = defaultdict(list)
    excluded = []
    for ref in refs:
        d = ref["data"]
        sid = d.get("id", Path(ref["source"]).stem)
        if ref["coordinate_status"] == "pending_original_bbox":
            excluded.append(
                dict(
                    id=sid,
                    source=ref["source"],
                    reason="9 original bboxes are not finished manual splits",
                )
            )
            continue
        house = d.get("house") or d.get("source_house")
        label = d.get("source_room_label")
        if not house or not label:
            excluded.append(
                dict(
                    id=sid,
                    source=ref["source"],
                    reason="source house/room not specified",
                )
            )
            continue
        groups[(house, label)].append(ref)
    comparisons = []
    available = []
    source_count = 0
    for key, manuals in groups.items():
        row = by_key.get(key)
        manual_polygons = []
        auto = []
        auto_heights = []
        if row:
            for f in row.get("floors", []):
                auto.extend(
                    [
                        shape(part["metrics"]["floor_polygon"])
                        for part in f["split_parts"]
                    ]
                )
                auto_heights.extend([f["metrics"]["floor_y_m"]] * len(f["split_parts"]))
        for ref in manuals:
            d = ref["data"]
            bbox = d.get("bbox_xz_m")
            manual_y = d.get("floor_y_m")
            geometry = None
            if row and bbox:
                matching = [
                    shape(f["metrics"]["floor_polygon"])
                    for f in row.get("floors", [])
                    if manual_y is None
                    or abs(f["metrics"]["floor_y_m"] - manual_y)
                    <= p["floor_height_separation_m"]
                ]
                if matching:
                    geometry = shapely.union_all(matching).intersection(
                        box(*bbox[0], *bbox[1])
                    )
            manual_polygons.append(geometry)
        valid = [
            i for i, g in enumerate(manual_polygons) if g is not None and not g.is_empty
        ]
        matrix = np.zeros((len(manuals), len(auto)))
        for i in valid:
            for j, g in enumerate(auto):
                manual_y = manuals[i]["data"].get("floor_y_m")
                if (
                    manual_y is not None
                    and abs(manual_y - auto_heights[j]) > p["floor_height_separation_m"]
                ):
                    continue
                a = manual_polygons[i]
                den = a.union(g).area
                matrix[i, j] = a.intersection(g).area / den if den else 0
        matched = {}
        if len(auto):
            ii, jj = linear_sum_assignment(matrix, maximize=True)
            matched = dict(zip(ii, jj))
            source_count += 1
        for i, ref in enumerate(manuals):
            d = ref["data"]
            record = dict(
                id=d.get("id", Path(ref["source"]).stem),
                house=key[0],
                source_room_label=key[1],
                manual_source=ref["source"],
                automatic_part_count=len(auto),
                manual_geometry_status=(
                    "semantic_floor_clipped_bbox_proxy" if i in valid else "unavailable"
                ),
                matched_iou=(
                    float(matrix[i, matched[i]])
                    if i in valid and i in matched
                    else (0.0 if i in valid else None)
                ),
                best_iou=(
                    float(matrix[i].max())
                    if i in valid and len(auto)
                    else (0.0 if i in valid else None)
                ),
                auto_proposal_available=bool(auto),
                automatic_stage2=(row["stage2"] if row else None),
                manual_extent_basis=d.get("scope_basis"),
                manual_bbox_area_m2=(
                    float((np.array(d["bbox_xz_m"][1]) - d["bbox_xz_m"][0]).prod())
                    if d.get("bbox_xz_m")
                    else None
                ),
                clipped_manual_area_m2=(
                    float(manual_polygons[i].area) if i in valid else None
                ),
            )
            comparisons.append(record)
            if record["matched_iou"] is not None and auto:
                available.append(record["matched_iou"])
    all_values = [r["matched_iou"] for r in comparisons if r["matched_iou"] is not None]
    ratios = [
        r["clipped_manual_area_m2"] / r["manual_bbox_area_m2"]
        for r in comparisons
        if r["clipped_manual_area_m2"] is not None and r["manual_bbox_area_m2"]
    ]
    unavailable = [r for r in comparisons if r["matched_iou"] is None]
    absent = [
        r
        for r in comparisons
        if r["matched_iou"] is not None and not r["auto_proposal_available"]
    ]
    return dict(
        status="qualified_bbox_proxy",
        source=str(path),
        total_manual_records=len(refs),
        excluded_count=len(excluded),
        excluded=excluded,
        compared_manual_count=len(all_values),
        manual_source_rooms_with_coordinates=len(groups),
        source_rooms_with_auto_proposal=source_count,
        mean_matched_iou_all_coordinate_valid=(
            float(np.mean(all_values)) if all_values else None
        ),
        mean_matched_iou_when_auto_proposal_exists=(
            float(np.mean(available)) if available else None
        ),
        proposal_available_manual_count=len(available),
        comparisons=comparisons,
        exact_mask_iou_status="not_available: no common validated same-floor floor masks/polygons; some pixel crops exist; crops may extend beyond the source region",
        matching="same-floor (<=0.3m) Hungarian one-to-one IoU; unmatched manual boxes count 0; pending/empty geometry excluded; zero-proposal cases included in all-case mean",
        diagnosis=dict(
            valid_without_proposal=len(absent),
            valid_with_proposal=len(available),
            empty_or_wrong_floor_proxy=len(unavailable),
            mean_bbox_fraction_covered_by_source_floor=(
                float(np.mean(ratios)) if ratios else None
            ),
            bbox_coverage_below_half_count=sum(r < 0.5 for r in ratios),
            no_proposal_stage2_reasons=dict(
                Counter(
                    c
                    for r in absent
                    for c in (
                        (r["automatic_stage2"] or {}).get("reason_codes")
                        or ["NOT_TRIGGERED"]
                    )
                )
            ),
            conclusion="Both reference scope and algorithm coverage limit IoU: source-region floor clips only part of hand crop, and no automatic proposal counts as zero. Not exact hand-mask accuracy.",
        ),
    )


def second_sample(rows, p):
    groups = defaultdict(list)
    for r in rows:
        if r.get("human"):
            groups[r["house"]].append(r)
    total = sum(map(len, groups.values()))
    target = max(len(groups), round(total * p["second_reviewer_fraction"]))
    rng = random.Random(p["random_seed"])
    allocation = {h: 1 for h in groups}
    remaining = target - len(groups)
    # At least one per house; allocate any remaining budget in proportion to its
    # unsampled rooms. With small strata this may differ from nominal 12.5%.
    for _ in range(remaining):
        names = [h for h in sorted(groups) if allocation[h] < len(groups[h])]
        weights = [len(groups[h]) - allocation[h] for h in names]
        h = rng.choices(names, weights=weights, k=1)[0]
        allocation[h] += 1
    sample = []
    for h in sorted(groups):
        population = sorted(groups[h], key=lambda r: r["region_id"])
        chosen = rng.sample(population, allocation[h])
        for r in sorted(chosen, key=lambda r: r["region_id"]):
            sample.append(
                dict(
                    house=h,
                    room_label=r["room_label"],
                    first_review_source=r["human"]["source"],
                    inclusion_probability=allocation[h] / len(population),
                    stratum_room_count=len(population),
                    stratum_sample_count=allocation[h],
                    second_verdict=None,
                    second_reason_codes=[],
                    second_reviewer=None,
                )
            )
    return dict(
        seed=p["random_seed"],
        nominal_fraction=p["second_reviewer_fraction"],
        reviewed_population=total,
        sample_count=len(sample),
        actual_fraction=len(sample) / total if total else None,
        house_strata=len(groups),
        design="whole-house strata; at least one random room per reviewed house; control total; store unequal inclusion probabilities",
        labels_status="not_judged; no two-human agreement or kappa computed",
        instructions="第二审核人先独立看图和视频，隐藏第一次 verdict 和自动建议；按固定理由清单导出判断。争议由两人复核，不由助手代判。",
        samples=sample,
    )


def candidates(rows, inputs, clean_path, inventory_path):
    if clean_path is None or not Path(clean_path).exists():
        return dict(
            status="not_run",
            reason="prior M1/SO clean-house list missing",
            houses=[],
            rooms=[],
        )
    clean = set(json.loads(Path(clean_path).read_text()))
    inventory = json.loads(
        Path(inventory_path or inputs["inventory_source"]).read_text()
    )
    so = set(inventory["so_overlap"]["houses"]["hm3d"])
    physical_exposure = inventory["our_house_exposure"]["physical_house_evidence"]
    rooms = []
    for r in rows:
        if r["family"] != "hm3d":
            continue
        if (
            not r.get("human")
            or r["human"]["verdict"] != "use"
            or r["stage1"]["status"] != "pass"
            or r["stage4"]["status"] != "pass"
        ):
            continue
        if (
            r["house"] not in clean
            or r["house"].split("_", 3)[3] in so
            or r["house"] in physical_exposure
        ):
            continue
        rooms.append(
            dict(
                house=r["house"],
                room_label=r["room_label"],
                floor_id=(r.get("floor_selection") or {}).get("selected_floor_id")
                or r["floors"][0]["floor_id"],
                human_source=r["human"]["source"],
                gate_source=r["stage4"]["source"],
                stage2=r["stage2"],
                floor_area_m2=r["metrics"]["floor_area_m2"],
                nav_main_area_m2=r["metrics"]["nav_main_area_m2"],
                black_fraction=r["metrics"]["black_fraction"],
                placement=r["metrics"]["placement"],
                floor_polygon=r["metrics"]["floor_polygon"],
                floor_selection=r.get("floor_selection"),
                scope_note="Use verdict belongs to original region; proposed splits need new human labels. Measurements and witness belong only to selected dominant floor.",
            )
        )
    houses = sorted({r["house"] for r in rooms})
    counts = Counter(h.split("_")[1] for h in houses)
    return dict(
        status="candidate_list_for_human_review",
        houses=houses,
        house_count=len(houses),
        room_count=len(rooms),
        rooms=rooms,
        split_house_counts=dict(counts),
        ready_without_split_proposal=[
            r for r in rooms if r["stage2"]["status"] == "not_triggered"
        ],
        clean_list_source=str(clean_path),
        so_source=str(inventory_path or inputs["inventory_source"])
        + ":so_overlap.houses.hm3d",
        m1_exclusion_source=str(clean_path),
        m1_method="use owner-provided previously computed clean-house list; additional explicit exposure cross-check from assets_inventory",
        limitation="SO anonymous scenes cannot all be assigned to houses; not a universal guarantee of no historical exposure; split children require a new human review.",
    )


def summarize(args, rows, inputs):
    out = args.output
    p = inputs["parameters"]
    previous = json.loads((out / "previous_82382e2/agreement.json").read_text())
    split = json.loads((out / "house_analysis_split.json").read_text())
    calibrated = set(split["calibration"])
    held = set(split["holdout"])
    agree = dict(
        source=str(out / "rooms_registry.jsonl"),
        split_source=str(out / "house_analysis_split.json"),
        thresholds_source=str(out / "thresholds.used.yaml"),
        all=agreement(rows),
        calibration=agreement(rows, calibrated),
        holdout=agreement(rows, held),
        documented_simple_baseline_all=previous["documented_simple_baseline_all"],
        documented_simple_baseline_holdout=previous[
            "documented_simple_baseline_holdout"
        ],
        simple_baseline_source=str(out / "previous_82382e2/agreement.json"),
        simple_baseline_rule="legacy floor_area_m2 >=6; >=1 main furniture category; no stairs/step/balustrade/railing among top_categories; human unsure excluded",
        historical_task_R_baseline=inputs["historical_baseline_task_R"],
        tuning=inputs["calibration_selection"],
        second_human_agreement=dict(
            status="not_run", reason="second reviewer has not judged the fixed sample"
        ),
    )
    write_json(out / "agreement.json", agree)
    differences = analyze_disagreements(rows, held)
    write_json(out / "disagreement_examples.json", differences)
    sample = json.loads(
        (out / "previous_82382e2/second_reviewer_sample.json").read_text()
    )
    population_keys = {(r["house"], r["room_label"]) for r in rows}
    if any(
        (r["house"], r["room_label"]) not in population_keys for r in sample["samples"]
    ):
        raise ValueError("previous second reviewer sample no longer registered")
    write_json(
        out / "second_reviewer_sample_changes.json",
        dict(
            retained=sample["sample_count"],
            replaced=0,
            reasons=[],
            source=str(out / "previous_82382e2/second_reviewer_sample.json"),
        ),
    )
    write_json(out / "second_reviewer_sample.json", sample)
    sample_keys = {(r["house"], r["room_label"]) for r in sample["samples"]}
    write_json(out / "review_reason_codes.json", REASONS)
    queue = []
    for r in rows:
        overlays = [
            f["stage2"].get("overlay_path")
            for f in r.get("floors", [])
            if f["stage2"].get("overlay_path")
        ]
        queue.append(
            dict(
                house=r["house"],
                room_label=r["room_label"],
                room_type=r.get("room_type", "unknown"),
                overhead_path=(
                    str(Path(r["overhead"]["image_cache"]).relative_to(out))
                    if r.get("overhead")
                    else None
                ),
                video_url=(
                    "/media/" + r["house"] + "/" + r["room_label"] + ".mp4"
                    if r.get("video_exists")
                    else None
                ),
                automatic=r["stage1"],
                split=r["stage2"],
                split_overlay_paths=overlays,
                floor_metrics=[f["metrics"] for f in r.get("floors", [])],
                floor_selection=r.get("floor_selection"),
                split_parts=[
                    dict(floor_id=f["floor_id"], **part)
                    for f in r.get("floors", [])
                    for part in f["split_parts"]
                ],
                gate=r["stage4"]["status"],
                existing_review_present=bool(r.get("human")),
                second_review=(r["house"], r["room_label"]) in sample_keys,
                review_status="not_reviewed_in_protocol",
                verdict=None,
                reason_codes=[],
                note="",
            )
        )
    with (out / "review_queue.jsonl").open("w") as stream:
        for q in queue:
            stream.write(json.dumps(q, ensure_ascii=False) + "\n")
    iou = manual_iou(rows, args.manual_splits, p)
    write_json(out / "split_iou.json", iou)
    v7 = (
        candidates(rows, inputs, args.clean_houses, args.inventory)
        if inputs["family"] == "hm3d"
        else dict(status="not_run", houses=[], rooms=[])
    )
    write_json(out / "v7_candidates.json", v7)
    with (out / "v7_candidates.tsv").open("w") as stream:
        stream.write(
            "house\troom_label\tfloor_id\tsplit_status\tfloor_area_m2\tnav_main_area_m2\n"
        )
        for r in v7.get("rooms", []):
            stream.write(
                "\t".join(str(r[k]) for k in ["house", "room_label", "floor_id"])
                + "\t"
                + r["stage2"]["status"]
                + "\t"
                + str(r["floor_area_m2"])
                + "\t"
                + str(r["nav_main_area_m2"])
                + "\n"
            )
    s1 = Counter(r["stage1"]["status"] for r in rows)
    s2 = Counter(r["stage2"]["status"] for r in rows)
    floors = [f for r in rows for f in r.get("floors", [])]
    children = [part for f in floors for part in f["split_parts"]]
    reasons = Counter(c for r in rows for c in r["stage1"]["reason_codes"])
    gate = Counter(r["stage4"]["status"] for r in rows)
    funnel = dict(
        source=str(out / "rooms_registry.jsonl"),
        counts_unit="original semantic region; floors/children separately counted, never added to original-room denominator",
        stage0=dict(
            incoming_houses=inputs["house_count"],
            outgoing_registered_regions=len(rows),
            nonnegative_regions=sum(r["region_id"] >= 0 for r in rows),
            unassigned_region_buckets=sum(r["region_id"] < 0 for r in rows),
            legacy_registered_rooms=sum(r.get("legacy_room") is not None for r in rows),
            human_reviewed_rooms=sum(r.get("human") is not None for r in rows),
            extra_regions_without_legacy_room=sum(
                r.get("legacy_room") is None for r in rows
            ),
        ),
        stage1=dict(
            incoming=len(rows),
            outgoing_pass=s1["pass"],
            retained_fail=s1["fail"],
            retained_review=s1["review"],
            not_run=s1["not_run"],
            reason_counts=dict(reasons),
            floor_units=len(floors),
            floor_decisions=dict(Counter(f["stage1"]["status"] for f in floors)),
        ),
        stage2=dict(
            incoming=len(rows),
            region_status_counts=dict(s2),
            triggered_regions=sum(
                r["stage2"]["status"] in ["review", "proposed"] for r in rows
            ),
            reason_counts=dict(
                Counter(c for r in rows for c in r["stage2"]["reason_codes"])
            ),
            overlay_status_counts=dict(
                Counter(
                    f["stage2"].get("overlay_status", "not_drawn")
                    for f in floors
                    if f["stage2"]["status"] == "proposed"
                )
            ),
            proposed_children=len(children),
            children_stage1_counts=dict(
                Counter(c["stage1"]["status"] for c in children)
            ),
            pass_original_without_split_suggestion=sum(
                r["stage1"]["status"] == "pass"
                and r["stage2"]["status"] == "not_triggered"
                for r in rows
            ),
        ),
        stage3=dict(
            queue_count=len(queue),
            second_review_sample_count=sample["sample_count"],
            human_decisions_this_run=0,
        ),
        stage4=dict(
            incoming=len(rows),
            house_gate_counts=inputs["latest_gate_counts"],
            region_gate_counts=dict(gate),
            reason_counts={"E2E_GATE_" + str(k).upper(): v for k, v in gate.items()},
            stage1_pass_and_gate_pass=sum(
                r["stage1"]["status"] == "pass" and r["stage4"]["status"] == "pass"
                for r in rows
            ),
        ),
        final_v7=dict(
            houses=v7.get("house_count", 0),
            rooms=v7.get("room_count", 0),
            ready_without_split_proposal=len(
                v7.get("ready_without_split_proposal", [])
            ),
        ),
    )
    write_json(out / "funnel.json", funnel)
    from .review import build_review

    build_review(out, rows, sample)
    print(
        "SUMMARY",
        json.dumps(
            dict(
                stage0=len(rows),
                stage1=dict(s1),
                stage2=dict(s2),
                holdout=agree["holdout"]["agreement"],
                v7_rooms=v7.get("room_count", 0),
            ),
            ensure_ascii=False,
        ),
        flush=True,
    )
