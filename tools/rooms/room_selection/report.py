"""Chinese result publication from saved frozen results; never reevaluates labels."""

from pathlib import Path
import json
from collections import defaultdict
from .run import write_json, now
from .runtime import MEDIA_ROOT
import sys


def pct(value):
    return "未定义" if value is None else f"{value*100:.2f}%"


def publish(out):
    def read(name):
        return json.loads((out / name).read_text())

    code_root = Path(__file__).resolve().parents[3]
    python = Path(sys.executable)
    inputs = read("inputs.json")
    funnel = read("funnel.json")
    agree = read("agreement.json")
    comparison = read("holdout_comparison.json")
    v7 = read("v7_candidates.json")
    sample = read("second_reviewer_sample.json")
    selection = read("calibration_selection.json")
    frozen = read("freeze.json")
    paired = read("paired_comparison/summary.json")
    iou = read("split_iou.json")
    previous_iou = read("previous_82382e2/split_iou.json")
    previous = comparison["old_holdout"]
    current = comparison["new_holdout"]
    p = inputs["parameters"]
    s1 = funnel["stage1"]
    s2 = funnel["stage2"]
    s4 = funnel["stage4"]
    lines = [
        f"结论：以 smy 的共享测量层接入选房判断后，181 栋 HM3D 全量流程完成；同一房子留出集第二次评估的一致率 {pct(current['agreement'])}、精度 {pct(current['precision'])}、召回 {pct(current['recall'])}，保留 {v7['house_count']} 套、{v7['room_count']} 间 v7 人工终审候选。自动流程仍不能替代人工审核，切分子块不得继承原 use。",
        "",
        f"生成时间 {now()}（新加坡时间）；机器 48g / cw-SYS-4029GP-TRT3。",
        "",
        f"代码底座 feature/smy-room-screening 4175585；判断层来源 82382e2；新分支 claude/room-selection-on-smy-20261003。提交身份见 DELIVERY.json。",
        "",
        f"产物根目录 `{out}`；主证据 rooms_registry.jsonl、inputs.json、house_execution.json、execution_receipt.json；基线来源 previous_82382e2/ 和 scope_lineage.json。",
        "",
        "原始语义/碰撞 GLB、semantic.txt、已有 navmesh、rooms.json、第一次人工审核章、smy 手工目录均只读。未重建 navmesh，未写共享数据目录，未使用 GPU，未 push。",
        "",
        "## 测量与判断的分工",
        "",
        "阶段 0 按语义 annotation 登记所有 region，保留旧库存/人工标签缺失映射、无地面和未分配 ID；房间总数不能由自动成功结果倒推。共享 CSV 解析保留重复 RGB 涉及的所有实例和区域，冲突面不分配给任意一个实例。",
        "",
        "地面测量、地面类别词表、±2 通道的唯一近邻颜色匹配、家具 shape-preserving 投影和 navmesh 三角形读取来自 room_screening。统一 ground/furniture 加载器，移除重复地面 GLB 解析。家具按已有 navmesh 的 agent_height 裁剪人体高度区间，并填实例内部孔洞，保留外部凹形；同 region 与未分配实例计入已知占地，跨 region 不静默扣除。占地是扫描代理，未验证建筑真实净面积。",
        "",
        "楼层判断保留选房层的面积加权有界高度窗口：每窗 ≤0.3 m、层高取加权中位数，保留全部非空层；主层占比达冻结值才使用该层作房间结论，实质多层记 review。smy 原候选层采用相邻高度间隙聚类和非加权中位数，仅用作测量诊断；navmesh 层簇数不是经过人工确认的建筑楼层数。",
        "",
        "navmesh 总面积取 smy 的同高度三角形并集与语义地面的连续交集；主连通区域由原生 PathFinder 检查过的 0.25 m 四邻接图确定，再用该图主区网格支持域裁剪连续交集计面积。原网格面积另存 nav_grid_main_area_m2。小扫描裂缝不能仅凭最大投影碎片判为导航断开；边界和低于网格尺度的狭道仍是代理限制。没有用家具试算修正面积替代正式准入面积。",
        "",
        f"摆放保留 CPU 原始碰撞 GLB 三条射线证据：相机/声源净空 ≥{p['placement_camera_clearance_m']}/{p['placement_source_clearance_m']} m，最多 {p['placement_camera_samples']}/{p['placement_source_samples']} 候选，三点两两距离 1–5 m、水平夹角 ≤85°；净空≥0.5m比例另报。有限搜索未找到不等于证明无解；声学未运行。扫描质量仅使用真实俯视图及其保存 view/projection，同层 mask 内黑像素比例；缺图或楼层不匹配记未知。",
        "",
        "切分保留家具簇种子、测地分区和净空加权最小割；面积 >40m²、主体簇间距 >4m 或凸性 <0.65 触发建议。没有足够种子明确 review，每个子块重新测量与过阶段 1。叠加图注明无 RGB 的几何图，未将其当扫描质量证据。",
        "",
        "## 10 栋对拍",
        "",
        f"名单在 paired_comparison/sample_manifest.json，全部来自 calibration：{paired['house_count']} 栋、{paired['region_count']} 个原 region。旧实现地面复算全部与 82382e2 保留测量一致。每个 >10% 的指标差异、绝对量、定义分解和楼层变更解释在 paired_comparison/summary.json 与 houses/*.json；数量 {paired['difference_counts']}。",
        "",
        "|房子|navmesh 诊断层簇|region|旧分层地面总和 m²|新分层地面总和 m²|",
        "|---|---:|---:|---:|---:|",
        *[
            f"|{r['house']}|{r['native_navmesh_floor_count']}|{r['regions']}|{r['old_total_floor_area_m2']:.3f}|{r['merged_total_floor_area_m2']:.3f}|"
            for r in paired["houses"]
        ],
        "",
        "选样涵盖单层/多高度簇、大/小户型和黑像素较高的样本；极端语义高度不能直接称真实楼层。大差异主要是地面词表/颜色匹配扩展、家具人体高度与凹形口径、网格吸附外延与连续 navmesh 面积口径。另修复重复颜色覆盖、家具审计覆盖同 region 多楼层的实现问题；未把定义不同称作几何真值改善。",
        "",
        "## calibration 选择与冻结",
        "",
        f"沿用 house_analysis_split.json 的原 calibration 90 栋 / holdout 91 栋，未重新随机分房。预先写 calibration_plan.json；仅 calibration 在面积 [4,6,8]、短边 [2,2.2,2.4]、主 navmesh [3,4,5]、主层占比 [0.90,0.95,0.98] 的 81 组中选择。要求召回≥85%、精度≥前次 calibration 精度减2个百分点，再最大化精度/召回/一致率；平分时优先接近初值。",
        "",
        f"冻结阈值（完整出处 thresholds.frozen.yaml / tools/rooms/room_selection/thresholds.yaml）：地面≥{p['floor_area_min_m2']}m²；短边≥{p['short_side_min_m']}m；主 navmesh≥{p['nav_main_area_min_m2']}m²（卫生间≥{p['bathroom_nav_main_area_min_m2']}m²）；主层占比≥{pct(p['dominant_floor_area_fraction_min'])}；黑像素≤{pct(p['black_fraction_max'])}；排除楼梯/走廊/车库/储物/户外且必须有摆放证据。冻结 {frozen['created_at_sgt']}，依据和全部网格在 calibration_selection.json。",
        "",
        f"calibration：{selection['selected']['metrics']['confusion_matrix']}；一致率 {pct(selection['selected']['metrics']['agreement'])}、精度 {pct(selection['selected']['metrics']['precision'])}、召回 {pct(selection['selected']['metrics']['recall'])}。freeze.json 记录改变结果的代码和阈值身份；holdout_evaluation_once.json 明确本协议仅1次评估。",
        "",
        "## 同一 holdout 的两次历史评估",
        "",
        "本次是这批 holdout 房子的第二次评估。第一次为 82382e2 冻结规则；这次冻结共享测量后的规则后只评一次，没有根据结果再调阈值。历史 holdout 已被观察，因此这不是新的独立泛化验证；未来论文应说明复用或另设新测试总体。两次都按 human use/skip 二元、unsure 单列，自动 review/not_run 计未选中。",
        "",
        "|冻结版本|一致率|精度|召回|TP|FP|FN|TN|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *[
            f"|{name}|{pct(m['agreement'])}|{pct(m['precision'])}|{pct(m['recall'])}|{m['confusion_matrix']['TP']}|{m['confusion_matrix']['FP']}|{m['confusion_matrix']['FN']}|{m['confusion_matrix']['TN']}|"
            for name, m in [
                ("82382e2（第一次）", previous),
                ("共享测量合并版（第二次）", current),
            ]
        ],
        "",
        f"新减旧（百分点）{comparison['changes_percentage_points']}；出处 holdout_comparison.json，旧值直接读取 previous_82382e2/agreement.json。留出标签 {current['human_verdicts']}、二元分母 {current['binary_denominator']}；未挑好的版本替换。",
        "",
        "任务书历史简单规则近似一致率72%、精度74%、召回87%来自任务 R 用户背景，精确实现和 unsure 口径未验证。该数字只作历史背景，不与本轮精确复现混写。双人一致率/kappa 未验证，第二人尚未判断。分歧类型及每类至多5例在 disagreement_examples.json。",
        "",
        "## 全量漏斗与候选",
        "",
        "|阶段|数量|",
        "|---|---|",
        f"|0 房子 / 原 region / 既有人审|{inputs['house_count']} / {funnel['stage0']['outgoing_registered_regions']} / {funnel['stage0']['human_reviewed_rooms']}|",
        f"|1 pass / fail / review|{s1['outgoing_pass']} / {s1['retained_fail']} / {s1['retained_review']}|",
        f"|1 全部分层单元|{s1['floor_units']}；{s1['floor_decisions']}|",
        f"|2 原区域状态|{s2['region_status_counts']}|",
        f"|2 子块与重新筛选|{s2['proposed_children']}；{s2['children_stage1_counts']}|",
        f"|3 原区域队列 / 第二人名单|{funnel['stage3']['queue_count']} / {sample['sample_count']}|",
        f"|4 房子门禁|{s4['house_gate_counts']}|",
        f"|1 + 房门禁 pass|{s4['stage1_pass_and_gate_pass']}|",
        f"|v7 房子 / 原房间 / 未触发切分|{v7['house_count']} / {v7['room_count']} / {len(v7['ready_without_split_proposal'])}|",
        "",
        "每步原因代码和计数在 funnel.json；所有 fail/review 留在 registry。房门禁取现有 task.json 最新 created_at/task_id，不回写任务。候选要求既有人审 use、自动 pass、房门禁 pass，并使用 owner 已计算 clean-house 清单与 SO 已知 HM3D ID /明确历史暴露再排除。完整房子+房间清单 v7_candidates.json / .tsv。M1 训练全集未在本次重新追溯，匿名 SO 仍不能保证全部识别；这些限制保留。",
        "",
        "|房子|原 region|",
        "|---|---|",
        *[
            f"|{h}|{', '.join(r['room_label'] for r in v7['rooms'] if r['house']==h)}|"
            for h in v7["houses"]
        ],
        "",
        "## 切分对照与人工任务",
        "",
        f"手工43个块中9个原始待收窄框排除，具坐标34个；本次可计算 {iou.get('compared_manual_count')} 个 bbox-地面代理。全样本平均 IoU {iou.get('mean_matched_iou_all_coordinate_valid')}，有自动建议的 {iou.get('proposal_available_manual_count')} 个平均 {iou.get('mean_matched_iou_when_auto_proposal_exists')}，没有建议在全样本计0。前次全样本 {previous_iou.get('mean_matched_iou_all_coordinate_valid')}、有建议 {previous_iou.get('mean_matched_iou_when_auto_proposal_exists')}，出处 previous_82382e2/split_iou.json；新定义下分母变化也明确记录。",
        "",
        f"诊断 {iou.get('diagnosis')}。手工裁剪可超出来源 region，个别像素裁剪不是统一同层地面 mask；准确 mask IoU 未验证。本次未据这个对照调算法或硬凑 IoU，切分仍为需人工修正的活动范围建议。",
        "",
        f"仅保留 smy tools/rooms/room_screening/review.html 与 review_server.py。review_manifest.json 统一 schema 含自动原因、切分图、房门禁、主层和 second_review；原 region 与每个新子块可独立选 use/skip/unsure 与固定理由。缺原图标明缺素材。second_reviewer_sample.json 原180项全部沿用（来源 previous_82382e2/second_reviewer_sample.json，替换0），{pct(sample['actual_fraction'])}，一房一项且包含概率不等；首次标签、自动提示在第二人服务端模式隐藏。反馈写指定的新输出，不改 smy 章。详见 SMY_REVIEW_TASKS_zh.md。",
        "",
        "两组单测、CPU 真场景冒烟和运行日志见 evidence/；单测不替代真实建筑边界与声学验证。MP3D 不是本追加的全量范围；本次共享测量 backend 仅支持 HM3D，旧 MP3D pilot 未重跑，不能把本轮校准数字用于 MP3D。",
    ]
    (out / "REPORT_zh.md").write_text("\n".join(lines) + "\n")
    task = [
        f"结论：请 smy 复核合并版共享测量的范围与单位，并完成 {v7['room_count']} 间 v7 候选及切分/缺素材项的人工终审；第二审核人独立完成原180项，未代填判断。",
        "",
        f"新加坡时间 {now()}；机器48g；产物 {out}。",
        "",
        "原区域队列 review_queue.jsonl，统一入口 review_manifest.json，理由清单 review_reason_codes.json；子块独立审核，不能继承原 region use。缺素材的项先补俯视图/视频，不靠自动结论填写。",
        "",
        "第一次/复核模式（反馈只写新输出目录）：",
        "```bash",
        f"cd {code_root}",
        f"{python} -m tools.rooms.room_screening.review_server --manifest {out}/review_manifest.json --asset-root {out} --media-root {MEDIA_ROOT} --feedback {out}/human_feedback_first.json --host 127.0.0.1 --port 8773",
        "```",
        "",
        "第二人独立模式；服务端只给固定名单，隐藏第一次结论与自动提示：",
        "```bash",
        f"{python} -m tools.rooms.room_screening.review_server --manifest {out}/review_manifest.json --asset-root {out} --media-root {MEDIA_ROOT} --feedback {out}/human_feedback_second.json --blind-second-reviewer --host 127.0.0.1 --port 8774",
        "```",
        "",
        "先独立选择 use / skip / unsure，填写固定理由和审核人代号；完成后保留两个独立反馈文件，再合议分歧。second_reviewer_sample_changes.json 显示沿用180/替换0；原名单纳入概率用于总体加权一致率。双人一致率与 Cohen kappa 尚未运行。服务绑定回环地址，由使用者自行安排 SSH 转发。",
        "",
        "本轮没有启动会留待人工使用的服务。后台批处理脚本/PID/日志见 evidence/ 与 execution_receipt.json；操作前核对 PID 和命令，不停止共享服务。",
    ]
    (out / "SMY_REVIEW_TASKS_zh.md").write_text("\n".join(task) + "\n")
    paired_lines = [
        "结论：10栋 calibration 对拍完成，地面、家具和 navmesh 的 >10% 差异逐项保留并说明；投影面积均为扫描代理，不是人工确认净面积。",
        "",
        f"时间 {now()}，出处 sample_manifest.json、summary.json、houses/*.json。",
        "",
        "旧实现地面复算核对全部通过；楼层/area-weighted eligibility 与 smy provisional clustering 分开列。",
        "",
        "|房子/房间/旧层|指标|旧值|新值|差异比例|原因|",
        "|---|---|---:|---:|---:|---|",
    ]
    for d in paired["differences"]:
        oldv = d.get("old")
        newv = d.get("new")
        delta = d.get("relative_difference")
        paired_lines.append(
            f"|{d['house']}/{d['room_label']}/{d.get('old_floor_id','all')}|{d['metric']}|{oldv if oldv is not None else ''}|{newv if newv is not None else ''}|{pct(delta) if delta is not None else '零基数或层变更'}|{d['explanation']}|"
        )
    (out / "paired_comparison/REPORT_zh.md").write_text("\n".join(paired_lines) + "\n")
    print("PUBLISHED", now(), flush=True)
