"""Freeze calibration decisions, enforce one held-out evaluation, and report.

Uses a supplied existing whole-house split; never re-registers or re-shuffles.
All derived writes stay under --output, except the explicitly supplied YAML.
"""

from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import yaml
from . import REPO_ROOT
from .analysis import agreement, pct
from .geometry import dominant_layer
from .run import assemble, load_parameters, latest_tasks, now, write_json
from tools.rooms.runtime_config import TASKS_ROOT

MEASUREMENT_FILES = [
    "tools/rooms/room_selection/geometry.py",
    "tools/rooms/room_selection/navigation.py",
    "tools/rooms/room_selection/protocol.py",
    "tools/rooms/room_selection/run.py",
    "tools/rooms/room_selection/splitting.py",
    "tools/rooms/room_selection/media.py",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(out, thresholds):
    add = out / "addendum1"
    baseline = add / "baseline_5d4eca6"
    if (add / "freeze.json").exists():
        raise RuntimeError("already frozen; do not overwrite a held-out specification")
    split = json.loads((out / "house_analysis_split.json").read_text())
    cal = set(split["calibration"])
    held = set(split["holdout"])
    measured = []
    for house in sorted(cal):
        record = json.loads(
            (add / "calibration_stage1/houses" / (house + ".json")).read_text()
        )
        assert record["status"] == "measured"
        measured.extend(record["rows"])
    assert {r["house"] for r in measured} == cal and not (cal & held)
    old = json.loads((baseline / "agreement.json").read_text())["calibration"]
    plan = json.loads((add / "calibration_plan.json").read_text())
    experiments = []
    for minimum in plan["dominance_candidates"]:

        def prediction(row):
            if (
                not row.get("floors")
                or row["region_id"] < 0
                or not row.get("semantic_region_present", True)
                or (row.get("native_region") or {}).get("ambiguous_colours")
            ):
                return row["stage1"]["status"]
            index, _ = dominant_layer(row["floors"], minimum)
            return (
                row["floors"][index]["stage1"]["status"]
                if index is not None
                else "review"
            )

        metrics = agreement(measured, cal, prediction)
        experiments.append(
            dict(dominant_floor_area_fraction_min=minimum, metrics=metrics)
        )
    acceptable = [
        e
        for e in experiments
        if e["metrics"]["recall"] > plan["goal_recall"]
        and e["metrics"]["precision"]
        >= old["precision"] - plan["maximum_precision_drop_pp"] / 100
    ]
    if not acceptable:
        raise RuntimeError(
            "Calibration recall/precision goal unmet; no holdout evaluation is authorized by this freeze command."
        )
    selected = max(
        acceptable,
        key=lambda e: (
            e["metrics"]["precision"],
            e["metrics"]["recall"],
            e["metrics"]["agreement"],
        ),
    )
    spec, _ = load_parameters(thresholds)
    spec["status"] = "addendum1_frozen_on_calibration_before_single_holdout_evaluation"
    minimum = selected["dominant_floor_area_fraction_min"]
    spec["parameters"]["dominant_floor_area_fraction_min"]["value"] = minimum
    spec["parameters"]["dominant_floor_area_fraction_min"][
        "rationale"
    ] = f'仅 calibration 在 {plan["dominance_candidates"]} 中选择；先要求召回>85%、精度比旧值下降≤2个百分点，再最大化精度。主层占各簇投影并集面积之和至少{minimum:.0%}；其余层全部保留。'
    thresholds.write_text(
        "# Frozen using calibration only; do not retune after holdout.\n"
        + yaml.safe_dump(spec, allow_unicode=True, sort_keys=False)
    )
    selection = dict(
        created_at_sgt=now(),
        calibration_only=True,
        old_calibration=old,
        experiments=experiments,
        selected=selected,
        selection_policy=plan["selection_rule"],
        source=str(add / "calibration_stage1/rooms_registry.jsonl"),
        split_source=str(out / "house_analysis_split.json"),
    )
    write_json(add / "calibration_selection.json", selection)
    shutil.copy2(thresholds, add / "thresholds.frozen.yaml")
    shutil.copy2(thresholds, out / "thresholds.used.yaml")
    document = dict(
        created_at_sgt=now(),
        threshold_source=str(add / "thresholds.frozen.yaml"),
        threshold_sha256=digest(add / "thresholds.frozen.yaml"),
        split_sha256=digest(out / "house_analysis_split.json"),
        measurement_file_sha256={
            name: digest(REPO_ROOT / name) for name in MEASUREMENT_FILES
        },
        calibration_house_count=len(cal),
        holdout_house_count=len(held),
        calibration_selection_source=str(add / "calibration_selection.json"),
        holdout_evaluations_allowed=1,
        policy="Freeze before measuring/evaluating holdout; calibration selected on whole houses only.",
    )
    write_json(add / "freeze.json", document)
    # Preserve old logs and jobs before replacing task-owned derived artifacts.
    for name in ["logs", "jobs"]:
        destination = baseline / name
        if not destination.exists():
            shutil.copytree(out / name, destination)
    spec, parameters = load_parameters(add / "thresholds.frozen.yaml")
    inputs = json.loads((baseline / "inputs.json").read_text())
    tasks, _ = latest_tasks(TASKS_ROOT)
    gate_counts = Counter()
    for house in inputs["houses"]:
        job = json.loads((out / "jobs" / f"{house}.json").read_text())
        for row in job["rows"]:
            if house in tasks:
                row["stage4"] = tasks[house]
        gate_counts.update([job["rows"][0]["stage4"]["status"]])
        write_json(out / "jobs" / f"{house}.json", job)
    inputs.update(
        created_at_sgt=now(),
        parameters=parameters,
        threshold_spec=spec,
        calibration_selection=selection,
        latest_gate_counts=dict(gate_counts),
        threshold_freeze_source=str(add / "freeze.json"),
    )
    write_json(out / "inputs.json", inputs)
    print("FROZEN", json.dumps(selected, ensure_ascii=False), flush=True)


def verify_freeze(out):
    add = out / "addendum1"
    f = json.loads((add / "freeze.json").read_text())
    assert digest(add / "thresholds.frozen.yaml") == f["threshold_sha256"]
    assert digest(out / "thresholds.used.yaml") == f["threshold_sha256"]
    assert digest(out / "house_analysis_split.json") == f["split_sha256"]
    for name, value in f["measurement_file_sha256"].items():
        assert digest(REPO_ROOT / name) == value, (
            "measurement implementation changed after freeze: " + name
        )
    return f


def evaluate(out):
    freeze_record = verify_freeze(out)
    add = out / "addendum1"
    inputs = json.loads((out / "inputs.json").read_text())
    for house in inputs["houses"]:
        record = json.loads((out / "houses" / f"{house}.json").read_text())
        assert record["status"] == "measured", "measurement incomplete: " + house
        assert record["finished_at_sgt"] >= freeze_record["created_at_sgt"], (
            "old house results remain: " + house
        )
    receipt = add / "holdout_evaluation_once.json"
    with receipt.open("x") as stream:
        json.dump(
            dict(
                status="started",
                started_at_sgt=now(),
                evaluation_count=1,
                freeze_source=str(add / "freeze.json"),
            ),
            stream,
            indent=2,
        )
    args = SimpleNamespace(
        output=out,
        inventory=None,
        manual_splits=out / "manual_reference.json",
        clean_houses=Path("/data/jzy/tmp/hm3d_clean_curated_houses_20261002.json"),
        no_analysis=False,
    )
    assemble(args)
    verify_freeze(out)
    result = json.loads((out / "agreement.json").read_text())
    old = json.loads((add / "baseline_5d4eca6/agreement.json").read_text())
    comparison = dict(
        created_at_sgt=now(),
        evaluation_count=1,
        old_source=str(add / "baseline_5d4eca6/agreement.json"),
        new_source=str(out / "agreement.json"),
        freeze_source=str(add / "freeze.json"),
        holdout=dict(
            old=old["holdout"],
            new=result["holdout"],
            delta_percentage_points={
                k: 100 * (result["holdout"][k] - old["holdout"][k])
                for k in ["agreement", "precision", "recall"]
            },
        ),
        calibration=dict(old=old["calibration"], new=result["calibration"]),
    )
    write_json(add / "comparison.json", comparison)
    write_json(
        receipt,
        dict(
            status="complete",
            finished_at_sgt=now(),
            evaluation_count=1,
            freeze_source=str(add / "freeze.json"),
            comparison_source=str(add / "comparison.json"),
            no_post_holdout_tuning=True,
        ),
    )
    print(
        "ONE_HOLDOUT_EVALUATION",
        json.dumps(comparison["holdout"], ensure_ascii=False),
        flush=True,
    )


def publish(out):
    verify_freeze(out)
    add = out / "addendum1"
    receipt = json.loads((add / "holdout_evaluation_once.json").read_text())
    assert receipt["status"] == "complete"
    comparison = json.loads((add / "comparison.json").read_text())
    funnel = json.loads((out / "funnel.json").read_text())
    v7 = json.loads((out / "v7_candidates.json").read_text())
    oldv7 = json.loads((add / "baseline_5d4eca6/v7_candidates.json").read_text())
    iou = json.loads((out / "split_iou.json").read_text())
    oldiou = json.loads((add / "baseline_5d4eca6/split_iou.json").read_text())
    sample = json.loads((out / "second_reviewer_sample.json").read_text())
    assert sample == json.loads(
        (add / "baseline_5d4eca6/second_reviewer_sample.json").read_text()
    ), "fixed second reviewer sample changed"
    old = comparison["holdout"]["old"]
    new = comparison["holdout"]["new"]
    spec, p = load_parameters(add / "thresholds.frozen.yaml")
    selection = json.loads((add / "calibration_selection.json").read_text())
    diagnostic = json.loads((add / "diagnostics/summary.json").read_text())
    rows = [
        json.loads(x) for x in (out / "rooms_registry.jsonl").read_text().splitlines()
    ]
    baseline = [
        json.loads(x)
        for x in (add / "baseline_5d4eca6/rooms_registry.jsonl")
        .read_text()
        .splitlines()
    ]
    oldby = {(r["house"], r["room_label"]): r for r in baseline}
    diagby = {(r["house"], r["room_label"]): r for r in diagnostic["examples"]}
    priority = []
    by_key = {(r["house"], r["room_label"]): r for r in rows}
    for r in rows:
        previous = oldby[r["house"], r["room_label"]]
        changed = previous["stage1"]["status"] != r["stage1"]["status"]
        reasons = []
        if (
            r.get("human")
            and r["human"]["verdict"] == "skip"
            and r["stage1"]["status"] == "pass"
        ):
            reasons.append("HUMAN_SKIP_AUTO_PASS")
        if (r.get("floor_selection") or {}).get("retained_secondary_floor_ids"):
            reasons.append("DOMINANT_FLOOR_SCOPE_CHECK")
        if changed:
            reasons.append("AUTOMATIC_STATUS_CHANGED")
        if reasons:
            priority.append(
                dict(
                    house=r["house"],
                    room_label=r["room_label"],
                    priority_reasons=reasons,
                    old_automatic=previous["stage1"],
                    new_automatic=r["stage1"],
                    verdict=None,
                    reason_codes=[],
                    note="",
                )
            )
    with (out / "review_priority_addendum1.jsonl").open("w") as stream:
        for r in priority:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")
    queue = [
        json.loads(x) for x in (out / "review_queue.jsonl").read_text().splitlines()
    ]
    for q in queue:
        key = q["house"], q["room_label"]
        r = by_key[key]
        previous = oldby[key]
        q["previous_automatic"] = previous["stage1"]
        q["automatic_status_changed"] = (
            q["automatic"]["status"] != previous["stage1"]["status"]
        )
        if key in diagby:
            q["calibration_diagnostic_path"] = (
                "addendum1/diagnostics/" + diagby[key]["diagnostic_png"]
            )
    with (out / "review_queue.jsonl").open("w") as stream:
        for q in queue:
            stream.write(json.dumps(q, ensure_ascii=False) + "\n")
    conclusion = f"结论：仅用 calibration 半集修正摆放候选净空与小面积高度簇误拒，冻结后在同一 holdout 半集评估一次；召回由 {pct(old['recall'])} 升至 {pct(new['recall'])}，精度由 {pct(old['precision'])} 变为 {pct(new['precision'])}，一致率由 {pct(old['agreement'])} 升至 {pct(new['agreement'])}；v7 候选为 {v7['house_count']} 套、{v7['room_count']} 间。自动筛选仍不能替代人审，手工 mask 切分质量尚未验证。"
    lines = [
        conclusion,
        "",
        f"追加 1 生成时间：{now()}（新加坡时间）；机器 48g / cw-SYS-4029GP-TRT3。",
        "",
        "诊断与改动依据：",
        "",
        "- 诊断严格限定既有固定 calibration 90 套房子，两条原因各随机抽 15 个既有 use 但被拒的例子；种子 20261003，名单及图在 addendum1/diagnostics/sample_manifest.json。每图含原俯视 RGB、相机/声源候选、射线命中点、投影面积加权及面数高度直方图；JSON 保留原 GLB triangle ID，未把命中点猜成某种家具。",
        "- calibration 的摆放误拒 175 间全部没有有效三点组合，111 间零距离合格射线，85 间声源候选少于 3 个。15 例原预算和增加预算均 0/15 找到；独立设备净空后 15/15 找到。原来的 0.5 米质量统计被同时当成所有设备半径，使家具房的候选挤成一团。",
        "- 保留 0.25 米网格和净空≥0.5米比例统计；摆放改为相机净空≥0.25米、声源≥0.15米，预算128/256。85°、三点两两1–5米、1.5/1.2米设备高度、三条原始扫描网格无遮挡射线与3厘米端点容差均保留。诊断没有支持删小家具遮挡，也没有证明85°约束太严，因此未按语义过滤遮挡。",
        "- 多层误拒 calibration 94 间，87 间旧最大簇面积占比≥90%；抽样15例中14例≥90%，异常高度簇常只是很小投影。旧按最低面起窗及未加权中位数会被小三角面牵动，任意第二高度簇都使整间送审。现在最大投影面积的≤0.3米窗口聚类，层高取面积加权中位数；各层面积仍用并集、不累加重叠面。地毯/台阶/错误地面语义的具体成因未逐面人工确认；不声称已识别家具顶面。",
        f'- 主层要求面积占各层并集面积之和≥{pct(p["dominant_floor_area_fraction_min"])}，只在主层范围判断；次层全部保留测量并进人审。真正双楼层或主层不足的 region 继续 review。修正 v7 floor_id 原先固定取 F0 的问题，改取实际主层，同时输出主层多边形。',
        "",
        "calibration 选择记录：",
        "",
        f'旧 precision {pct(selection["old_calibration"]["precision"])}，recall {pct(selection["old_calibration"]["recall"])}。预写 calibration_plan.json 的约束为 recall>85%、precision 比旧值最多下降2个百分点；在主层占比 90%、95%、98% 中先满足约束再取最高 precision，未读取新的 holdout 指标。',
        "|主层占比|calibration 一致率|precision|recall|",
        "|---|---:|---:|---:|",
    ]
    for e in selection["experiments"]:
        m = e["metrics"]
        lines.append(
            f'|{pct(e["dominant_floor_area_fraction_min"])}|{pct(m["agreement"])}|{pct(m["precision"])}|{pct(m["recall"])}|'
        )
    lines += [
        "",
        f"新阈值：tools/rooms/room_selection/thresholds.yaml；执行副本 addendum1/thresholds.frozen.yaml。全部参数、来源和理由均在 YAML；freeze.json 保存冻结时间、阈值、固定名单及测量实现摘要。其余面积6平方米、短边2.2米、导航4平方米/卫生间1.5平方米、黑像素15%与切分触发阈值不变。",
        "",
        "holdout 对照（出处 addendum1/comparison.json；旧来源 baseline_5d4eca6/agreement.json，新来源 agreement.json）：",
        "",
        "|版本|一致率|precision|recall|TP|FP|FN|TN|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in [("旧 5d4eca6", old), ("新冻结规则", new)]:
        c = m["confusion_matrix"]
        lines.append(
            f'|{name}|{pct(m["agreement"])}|{pct(m["precision"])}|{pct(m["recall"])}|{c["TP"]}|{c["FP"]}|{c["FN"]}|{c["TN"]}|'
        )
    delta = comparison["holdout"]["delta_percentage_points"]
    lines += [
        "",
        f'precision 改变 {delta["precision"]:+.2f} 个百分点，recall {delta["recall"]:+.2f}，一致率 {delta["agreement"]:+.2f}。holdout 固定91套，715个人审：use434、skip246、unsure35；binary分母680，unsure不参与二分类，自动review/fail均计未选。新 holdout 评估次数=1（addendum1/holdout_evaluation_once.json）；评估后没有再改阈值。',
        f'召回>85%的目标：{"已达到" if new["recall"]>.85 else "未达到；不按 holdout 反向调参"}。',
        "",
        "自动切分 IoU 诊断与结论：",
        "",
        f'旧全样本 bbox 代理 IoU 为 {oldiou["mean_matched_iou_all_coordinate_valid"]:.6f}，有建议8块的均值 {oldiou["mean_matched_iou_when_auto_proposal_exists"]:.6f}。25个可算手工块中17个没有自动建议计0，所以0.101178=8/25×0.316182；另9个未完成框排除、9个裁剪后空或楼层不匹配而不可算。不是单纯 bbox 替代 mask 所致。',
        "手工范围常超出原语义 region；旧25块中17块来源地面覆盖不到 bbox 的一半，平均覆盖39.11%。例如00378-R7-1的61.38平方米手工框与来源地面相交为空；00541-R5-1将范围扩到厨房而原region缺少主体家具种子。人工裁的是画面范围/活动区域，自动只对单一语义region地面上的家具簇切分，两者范围和目的并不统一。",
        "算法覆盖也不足：旧17块无建议中8块未触发、6块种子不足、3块无导航（触发码可重叠）。calibration有效6块全部无建议，无法据此证明 min-cut 会复现手工。未硬凑新种子、触发门槛或用手工框参与算法；切分算法保持原实现。",
        "修正对照匹配的一处口径问题：手工层高和自动子块层高必须相差≤0.3米，避免不同楼层的俯视轮廓误配。新分层会连带改变建议，因此重新报告代理结果，不能将变化当成经过 mask 真值验证的算法进步。",
        f'当前相同口径下全可算样本代理均值 {iou["mean_matched_iou_all_coordinate_valid"]}（{iou["compared_manual_count"]}块）；有建议子集 {iou["mean_matched_iou_when_auto_proposal_exists"]}（{iou["proposal_available_manual_count"]}块）。split_iou.json 逐块列出状态和不可算原因。需要 smy 补统一世界坐标多边形/同层mask、完成9个未收窄框，再评精确IoU；精确mask IoU仍未验证。',
        "",
        f'v7 清单从旧 {oldv7["house_count"]}套/{oldv7["room_count"]}间变为 {v7["house_count"]}套/{v7["room_count"]}间，其中未触发切分 {len(v7["ready_without_split_proposal"])}间。逐房子/房间清单在 v7_candidates.json、v7_candidates.tsv 及下文；仍使用任务书给定 clean-house 清单排除 M1、当前 SO 已知名单和明确暴露交叉排除。未重新复算全部M1训练流，匿名SO暴露仍未验证。',
        "",
    ]
    # Keep the complete protocol, funnel, discrepancy examples and house list.
    protocol_base = add / "evidence/protocol_report_base.md"
    if not protocol_base.exists():
        shutil.copy2(out / "REPORT_zh.md", protocol_base)
    protocol = protocol_base.read_text().split("\n", 2)[2]
    (out / "REPORT_zh.md").write_text("\n".join(lines) + "\n" + protocol)
    task = [
        f"结论：请按更新后的全量审核队列复核自动建议和主层范围；固定第二审核名单保留不变，助手未代判。",
        "",
        f"更新时间：{now()}（新加坡时间），所有路径在48g {out}。",
        "",
        f"1. review_queue.jsonl 共 {len(queue)}个原region，view/video/新自动结论/旧结论/分层指标/切分叠加均在队列；review_priority_addendum1.jsonl 有 {len(priority)}条优先复核项，理由为既有skip但自动pass、主层保留次层、或自动状态变化。先看主层多边形是否就是可用范围，不直接沿用整区bbox。",
        f"2. 诊断只用calibration，两条各15例；addendum1/diagnostics/sample_manifest.json给固定名单，PNG看相机（黄）、声源（青）、遮挡首命中（紫）和高度直方图；JSON给坐标和射线triangle ID。队列中这些房间有calibration_diagnostic_path。",
        "3. 父region use/skip/unsure与自动切分子块分别审核；原region use不能当子块use。用review_reason_codes.json固定理由，OTHER必填备注；导出独立JSON，不改原审核章、smy目录或原始数据。",
        f'4. 第二审核人只用second_reviewer_sample.json原来的{sample["sample_count"]}项（{pct(sample["actual_fraction"])}，覆盖{sample["house_strata"]}套已审房子，种子20261002），second模式隐藏第一人结论和自动建议，独立做判断后导出。尚无人做第二判，双人一致率/kappa/仲裁都未运行；不把本次自动-人工一致率当双人一致率。',
        "5. 对手工切分补统一、同层、世界坐标多边形或mask；先完成9个pending_original_bbox，核对00378-R7、00541-R5、00557-R5、00567-R12、00590-R5等跨region/扩大裁剪范围。明确手工目标是活动区域、房间还是画面裁剪，再提供来源region集合；现代理IoU不能当精确切分质量。",
        f'6. v7_candidates.json/tsv为{v7["house_count"]}套/{v7["room_count"]}间候选；需切分或有重要次层时先复核范围，再挑录制位。俯视图/巡房视频缺失项先补素材。审核服务沿用 tools/rooms/room_selection/serve_review.py（只读GET，浏览器本地保存后独立导出）；该任务未启动常驻服务。',
        "",
        "holdout冻结后只评一次；新增判断用于后续审核与新的实验协议，不反过来再调本次holdout。MP3D沿用前轮pilot结论，本追加未重跑：region可接入几何/nav检查，缺校准扫描质量素材和可靠统一边界，不能直接宣布完整流程通过。",
    ]
    (out / "SMY_REVIEW_TASKS_zh.md").write_text("\n\n".join(task) + "\n")
    write_json(
        add / "delivery.json",
        dict(
            created_at_sgt=now(),
            status="complete",
            machine="48g/cw-SYS-4029GP-TRT3",
            holdout_evaluation_count=1,
            holdout_new=new,
            v7_house_count=v7["house_count"],
            v7_room_count=v7["room_count"],
            second_sample_unchanged=True,
            threshold_source=str(add / "thresholds.frozen.yaml"),
            jobs_remaining=False,
        ),
    )
    print("PUBLISHED", conclusion, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["freeze", "verify-freeze", "evaluate", "publish"]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--thresholds", type=Path, default=Path(__file__).with_name("thresholds.yaml")
    )
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.output, args.thresholds)
    elif args.stage == "verify-freeze":
        verify_freeze(args.output)
        print("FREEZE_VERIFIED")
    elif args.stage == "evaluate":
        evaluate(args.output)
    else:
        publish(args.output)


if __name__ == "__main__":
    main()
