"""House-held-out agreement, split proxy IoU, funnel, human queues and report."""

from __future__ import annotations

import itertools
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


def baseline_prediction(r):
    old = r.get("legacy_room") or {}
    cats = set(old.get("top_categories") or [])
    main = cats & {
        "bed",
        "mattress",
        "sofa",
        "couch",
        "stove",
        "oven",
        "fridge",
        "sink",
        "desk",
        "table",
        "toilet",
        "bathtub",
    }
    stairs = cats & {"stairs", "stair", "step", "balustrade", "railing"}
    return (
        "pass"
        if (old.get("floor_area_m2") or 0) >= 6 and main and not stairs
        else "fail"
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


def tune_diagnostics(rows, calibration, p):
    # Initial task values remain the selected specification. Sensitivity is
    # evaluated exclusively on the calibration half and never selects on holdout.
    grid = []
    for area, short in itertools.product(
        p["calibration_area_grid_m2"], p["calibration_short_side_grid_m"]
    ):
        candidate = dict(p, floor_area_min_m2=area, short_side_min_m=short)

        def prediction(r):
            if (
                len(r.get("floors", [])) != 1
                or r["region_id"] < 0
                or (r.get("native_region") or {}).get("ambiguous_colours")
                or not r.get("semantic_region_present", True)
            ):
                return r["stage1"]["status"]
            return decide(
                r["floors"][0]["metrics"], r.get("room_type", "unknown"), candidate
            )["status"]

        metrics = agreement(rows, calibration, prediction)
        grid.append(
            dict(floor_area_min_m2=area, short_side_min_m=short, metrics=metrics)
        )
    return dict(
        selected="initial task R thresholds retained",
        selected_area=p["floor_area_min_m2"],
        selected_short_side=p["short_side_min_m"],
        reason="No automatic destructive filtering; geometry feasibility and missing evidence dominate. Report sensitivity without optimizing on the held-out labels.",
        calibration_only=True,
        sensitivity=grid,
    )


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
        if row:
            for f in row.get("floors", []):
                auto.extend(
                    [
                        shape(part["metrics"]["floor_polygon"])
                        for part in f["split_parts"]
                    ]
                )
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
        exact_mask_iou_status="not_available: manual masks/polygons absent; bbox crops may extend beyond original semantic region",
        matching="Hungarian one-to-one IoU; unmatched manual boxes count 0; pending/empty geometry excluded; zero-proposal cases included in all-case mean",
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
                floor_id=r["floors"][0]["floor_id"],
                human_source=r["human"]["source"],
                gate_source=r["stage4"]["source"],
                stage2=r["stage2"],
                floor_area_m2=r["metrics"]["floor_area_m2"],
                nav_main_area_m2=r["metrics"]["nav_main_area_m2"],
                black_fraction=r["metrics"]["black_fraction"],
                placement=r["metrics"]["placement"],
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
        documented_simple_baseline_all=agreement(rows, prediction=baseline_prediction),
        documented_simple_baseline_holdout=agreement(rows, held, baseline_prediction),
        simple_baseline_rule="legacy floor_area_m2 >=6; >=1 main furniture category; no stairs/step/balustrade/railing among top_categories; human unsure excluded",
        historical_task_R_baseline=inputs["historical_baseline_task_R"],
        tuning=tune_diagnostics(rows, calibrated, p),
        second_human_agreement=dict(
            status="not_run", reason="second reviewer has not judged the fixed sample"
        ),
    )
    write_json(out / "agreement.json", agree)
    differences = analyze_disagreements(rows, held)
    write_json(out / "disagreement_examples.json", differences)
    sample = second_sample(rows, p)
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
    (out / "review.html").write_text(
        Path(__file__).with_name("review.html").read_text()
    )
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
    report(args, inputs, funnel, agree, iou, v7, differences, sample)
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


def pct(x):
    return "未定义" if x is None else f"{x*100:.2f}%"


def report(args, inputs, funnel, agree, iou, v7, differences, sample):
    out = args.output
    held = agree["holdout"]
    s0 = funnel["stage0"]
    s1 = funnel["stage1"]
    s2 = funnel["stage2"]
    s4 = funnel["stage4"]
    lines = [
        f"结论：规范的自动几何筛选必须与人工审核结合；本轮在 {inputs['house_count']} 套 {inputs['family'].upper()} 上登记 {s0['outgoing_registered_regions']} 个语义 region/未分配桶，自动通过 {s1['outgoing_pass']} 个原 region，房子留出集与既有人审的一致率为 {pct(held['agreement'])}；SO 已知名单与既有 M1 排除清单交叉后保留 {v7.get('house_count',0)} 套、{v7.get('room_count',0)} 间选房候选。候选仍需人工终审，切分子块未代盖章。",
        "",
        f"统计生成时间：{datetime.now(ZoneInfo('Asia/Singapore')).isoformat()}；候选登记时间：{inputs['created_at_sgt']}（均为新加坡时间）。机器：48g / cw-SYS-4029GP-TRT3。",
        "",
        f"输入清单 `{inputs['inventory_source']}`；逐行证据 `{out/'rooms_registry.jsonl'}`；阈值 `{out/'thresholds.used.yaml'}`。",
        "",
        "所有原始 GLB、navmesh、rooms.json、smy 审核章和手工切分目录只读。未重新生成 navmesh，未写共享数据集，未起 GPU、训练或声学作业，未 push。",
        "",
        "流程与计算口径：",
        "",
        "0. 以本地已具备语义与现有 navmesh 的房子为限定总体；以 annotation region ID 全量登记，并保留 -1 未分配桶、旧记录缺失映射和没有地面几何的 region。旧房间登记不是总体边界。",
        "1. HM3D 语义 COLOR_0 按现有解析器转回 sRGB，与 semantic.txt 实例及 region 匹配；源坐标 x,y,z 转为 Habitat x,z,-y。同层 floor/rug/carpet/flooring 面水平投影求并集，家具按实例投影再并集。地面减家具为描述性指标，不替代地面面积。",
        "2. 每个地面层采用固定世界坐标 0.25 米网格中心；限制导航吸附漂移和高差，四邻接边用原生最短路检查，主岛面积为格子与地面交集并集估算。记录净空 ≥0.5 米的比例。现有 navmesh 的生成参数没有 sidecar 时标未知，不假装是本轮默认参数。",
        "3. 地面面重心高度跨度超过 0.3 米分层，所有层保留。多层 region 的原标签进入人工复核，不因某层通过就把整区自动通过。最小旋转外接矩形短边用于 2.2 米规则。",
        "4. 对已有 overview RGB 使用保存的 projection/view 矩阵绘制多边形 mask，扣除孔洞后测 RGB 三通道均 <8 的比例。与 ground 层高不匹配或素材不可读就记录未知。黑像素也可能是真实黑色纹理，不能解释为精确缺损率。",
        "5. 在主导航连通区域的净空达标点上确定性选最多 24 个相机、48 个声源候选；三点两两距离 1–5 米，两个声源水平夹角 ≤85°，三条视线均在原扫描 GLB 三角网格上用 CPU 射线检查。只报告具体可行坐标；未找到不等于数学上证明无解，声学未验证。",
        "6. 面积 >40 平方米、主体家具簇间距 >4 米、凸性 <0.65 触发切分建议。家具簇附近可达点作为种子；测地距离一元项与净空加权边容量做最小割，窄通道边切断成本更低。无足够家具种子的触发房间明确记 review。每个子块重跑阶段 1；非导航/未采样地面面积单列，子块是活动范围建议而非建筑房间真值。门洞没有显式门拓扑证明，因此只能称窄通道优先的代理切线。",
        "7. 房子级门禁取 task.json 的 created_at/task_id 最新记录；不因失败就删房间或改人审。候选清单合并人审 use、阶段 1 pass、房门禁 pass、已计算 clean-house 名单，并再次排除 SO 可识别 HM3D ID 与明确历史暴露。",
        "",
        "漏斗（出处 funnel.json；fail/review 均留在 registry）：",
        "",
        "|步骤|数量|",
        "|---|---|",
        f"|阶段 0 房子 / 原 region 登记|{inputs['house_count']} / {s0['outgoing_registered_regions']}|",
        f"|其中非负 region / 未分配桶|{s0['nonnegative_regions']} / {s0['unassigned_region_buckets']}|",
        f"|既有候选 / 既有人审 / 新增无旧登记 region|{s0['legacy_registered_rooms']} / {s0['human_reviewed_rooms']} / {s0['extra_regions_without_legacy_room']}|",
        f"|阶段 1 pass / fail / review / not_run|{s1['outgoing_pass']} / {s1['retained_fail']} / {s1['retained_review']} / {s1['not_run']}|",
        f"|分层单元|{s1['floor_units']}；{s1['floor_decisions']}|",
        f"|阶段 2 状态|{s2['region_status_counts']}|",
        f"|建议子块及重新筛选|{s2['proposed_children']}；{s2['children_stage1_counts']}|",
        f"|阶段 3 队列 / 第二人名单|{sample['reviewed_population']} 个既有人审；完整队列 {funnel['stage3']['queue_count']}；抽样 {sample['sample_count']}|",
        f"|阶段 4 最新房门禁|{s4['house_gate_counts']}|",
        f"|阶段 1 和房门禁同时 pass|{s4['stage1_pass_and_gate_pass']}|",
        f"|最终候选房子 / 原房间|{v7.get('house_count',0)} / {v7.get('room_count',0)}|",
        f"|其中未触发切分建议的原房间|{len(v7.get('ready_without_split_proposal',[]))}|",
        "",
        "一致性（出处 agreement.json、house_analysis_split.json）：",
        "",
        f"整套房子以种子 {inputs['parameters']['random_seed']} 按 train/val 分层后分成 calibration 与 holdout。calibration 单独进行 4/6/8 平方米与 2/2.2/2.4 米的敏感性检查；最终保留任务书初始值，未用 holdout 选阈值。holdout 有 {held['human_count']} 条既有人审，其中 {held['human_verdicts']}。use/skip 作为二元标签，unsure 单独保留；自动 review/not_run 按未选中计算部署一致率，同时报告仅确定自动结论的一致率和覆盖。",
        "",
        f"留出二元混淆矩阵 `{held['confusion_matrix']}`；分母 {held['binary_denominator']}；一致率 {pct(held['agreement'])}、精度 {pct(held['precision'])}、召回 {pct(held['recall'])}。仅自动确定样本覆盖 {pct(held['automatic_resolved_fraction'])}，其中一致率 {pct(held['agreement_on_resolved'])}。既有人审是对照标签，不是本轮双人一致性。",
        "",
        f"任务书提供的旧简单规则近似值是一致率约 72%、精度约 74%、召回约 87%（出处：任务 R 用户背景；精确实现与 unsure 口径未验证）。本轮另存明确实现的简单基线：留出一致率 {pct(agree['documented_simple_baseline_holdout']['agreement'])}、精度 {pct(agree['documented_simple_baseline_holdout']['precision'])}、召回 {pct(agree['documented_simple_baseline_holdout']['recall'])}；它不是对上述历史数字的精确复现。自动规则无法替代人工审核。",
        "",
        "分歧最大的类型及例子（每类至多五个；详细指标和素材在 disagreement_examples.json）：",
        "",
    ]
    for group in differences[:6]:
        ids = ", ".join(e["house"] + "/" + e["room_label"] for e in group["examples"])
        lines.append(
            f"- {group['room_type']}：分歧 {group['disagreement_count']}/{group['count']}（{pct(group['rate'])}）；{ids}。"
        )
    lines += [
        "",
        "自动切分与 smy 手工对照（出处 split_iou.json）：",
        "",
        f"状态 `{iou['status']}`。手工总记录 {iou.get('total_manual_records','未取到')}，排除 {iou.get('excluded_count','未定义')} 个未收窄或缺少来源的框；可计算 {iou.get('compared_manual_count','未定义')} 个。全部坐标有效样本的一对一匹配平均代理 IoU = {iou.get('mean_matched_iou_all_coordinate_valid','未定义')}；有自动建议的子集平均 = {iou.get('mean_matched_iou_when_auto_proposal_exists','未定义')}（{iou.get('proposal_available_manual_count','未定义')} 个手工块）。无建议或未匹配的有效框在全样本均值中计 0。",
        "",
        f"除上述待收窄记录外，还有 {len(iou.get('comparisons',[]))-iou.get('compared_manual_count',0)} 个框因同层语义地面缺失或相交为空而无法计算，详情在 split_iou.json 的 comparisons。",
        "",
        "这是「来源语义地面与手工 bbox 相交」对「自动活动范围」的代理 IoU。43 个手工交付目录没有 mask/多边形真值，9 个仍是原房间框；因此真正的手工 mask IoU 无法完成，不能把代理数字写成精确切分质量。人工裁剪超出来源 region 的范围也无法靠来源地面恢复。",
        "",
        "v7 清单（出处 v7_candidates.json 和 v7_candidates.tsv）：",
        "",
        f"房子列表：{', '.join(v7.get('houses',[])) or '无自动通过且满足全部条件的候选'}。每间房的坐标证据、自动指标、人审和门禁出处见 JSON/TSV。SO 排除使用 assets_inventory.json 的 so_overlap.houses.hm3d；M1 排除使用任务书给定的先前 clean-house 清单。该清单的历史计算并非本轮从全部 M1 训练流重新复算；匿名 SO 场景与未列出的历史暴露仍不能保证不重叠。",
        "",
        "给 smy 的后续人工审核说明：",
        "",
        "用 read-only review server 打开 review.html。review_queue.jsonl 为全 region 队列，显示现有俯视图、巡房视频、自动原因、切分叠加和房门禁；没有素材的项先补素材。新发现的 region 先做第一次判断；存疑、分层和自动子块分别复核，批准子块不能继承原 use。固定理由清单在 review_reason_codes.json。判断通过浏览器导出独立 JSON，不直接改原审核章。",
        f"第二审核人只审 second_reviewer_sample.json 的固定 {sample['sample_count']} 项（{pct(sample['actual_fraction'])}），覆盖 {sample['house_strata']} 个已审房子的 strata，名单种子固定。第二人模式默认隐藏第一次 verdict 和自动建议，先独立判断后合议分歧。没有代填 verdict；双人一致率、Cohen kappa 和仲裁结论都为 not_run，拿到导出的标签后再计算。样本有不同纳入概率，若估计总体一致率应使用名单记录的权重。",
        "",
        "运行与局限：",
        "",
        "代码、环境版本、CPU smoke 与整批日志在 evidence/、logs/、house_execution.json。新环境 /data/jzy/envs/room-selection-20261002 继承原运行库只读，新增 shapely 2.1.2、trimesh 4.8.3、rtree 1.4.1、networkx 3.5、plyfile 1.1.3。运行参数未知的旧导航网格、扫描纹理黑色、有限摆放搜索和小地面面高差分层都是需要抽查的代理局限。没有声学实测，不对全部真实住宅的泛化作结论。",
        "",
        "MP3D 的本地覆盖与三套试跑见 mp3d_inventory.json、mp3d_pilot/REPORT_zh.md（若尚未生成则未验证）。MP3D .house 的 region/level/object 关联可接入同一几何和导航流程，但本地未见同等已校准审核俯视图时，扫描质量必须留作未知，不能直接给出完整通过。数据格式参考 [Matterport 官方说明](https://github.com/niessner/Matterport/blob/master/data_organization.md)。",
        "",
        "机器与作业：本轮 CPU 控制脚本见 evidence/run_all.sh；没有正在运行作业的情况下无需停任务。若仍运行，先核对 evidence/job.pid 的 PID 和命令，再发送 TERM；具体运行状态以 execution_receipt.json 为准。",
        "",
    ]
    (out / "REPORT_zh.md").write_text("\n".join(lines))
