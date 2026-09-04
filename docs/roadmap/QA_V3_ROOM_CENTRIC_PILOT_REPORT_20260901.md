# QA v3 百题级房间中心工程 pilot（2026-09-01）

## 结论

百题级房间中心工程 pilot 已完成候选选择与代表性真实运行闭环：

- Apartment：18 个当前可运行 profile × 6 = 108 条；
- Kujiale：18 个当前可运行 profile × 6 = 108 条；
- 合计 216 条 research candidate；
- card12/card13/card14 在两个房间均保留 `resource_unavailable`，未伪造素材；
- 每条候选保留 MCQ/Open、main/Gate A AudioProgram、真实搜索分母和拒绝来源；
- 216/216 均有物化 Gate B：48 条扩展 profile 随候选生成，168 条双源
  profile 在选择后批量物化；
- Apartment 的 card11/card15a/card16 获得与本 pilot fact 明确绑定的像素通过候选；
- Apartment 的 card11/card15a/card16 完成 75 帧 UE 与 main/Gate A 双耳；
- Apartment 的 card17 完成两个不同 segment 的 75 帧 UE 与 segment1 双耳。

权威候选清单：
`/data/jzy/tmp/qa_v3_room_pilot_selected_2rooms_108each_20260901_v3/pilot_manifest.json`

代表性运行清单：
`/data/jzy/tmp/qa_v3_room_pilot_runtime_evidence_20260901_v1/pilot_runtime_manifest.json`

双源 Gate B 清单：
`/data/jzy/tmp/qa_v3_room_pilot_dual_gateb_168_20260901_v1/dual_gateb_manifest.json`

所有状态仍为 `research_candidate`；本报告不声明单模态认证、人类可答性、
正式题型准入或数据集发布。

## 房间中心候选生成

### 第一轮

基础矩阵：
`/data/jzy/tmp/qa_v3_room_pilot_2rooms_21profiles_cells6_20260901_v1`

第一轮结果：

- Apartment：108 条，18 个可运行 profile 全部 6/6；
- Kujiale：100 条；card1F 5/6、card1B 0/6、card5R 5/6；
- 其余 15 个可运行 profile 均为 6/6；
- 两个房间的 card12/13/14 均为精确素材缺口；
- 无 `profile_not_implemented`，无被吞掉的 `pipeline_error`。

### 定向补采

只对 Kujiale 缺口做 fresh/no-clobber 补采：

- card1F 补足 1 条；
- card5R 补足，并增加 closer/farther 分层池；
- card1B 用多个普通 seed 获取左右带，再用 30,000 次/格的 deep profile
  获取中心带，最终左/中/右各选 2 条；
- card17 增加独立批次，使第二段位置答案可按左/中/右各 2 条选择。

所有补采根目录均记录在最终 pilot manifest 的 `inputs.matrix_roots` 中。

## 配额与答案分布

每个房间：18 个可运行 profile，每 profile 6 条，总计 108 条；
card12/13/14 为 `resource_unavailable`。

主要答案分布：

- card1F/card1B：三方位带 2/2/2；
- 二值空间、距离与运动题：3/3；
- card4R/card7/card9：两外观 3/3；
- card5R：closer/farther 3/3；
- card8：四时间带 2/2/1/1；
- card11：三只可见外观各 1，`none` 为 3；
- card15a：叫过实例数 2/2/1/1；
- card15b：事件数 3/4 为 3/3；
- card17：第二段左/中/右 2/2/2。

聚合工具按 timeline 的相机与三帧 actor 位置去重，补采批次不会因 point id
重复而被误当成新候选。

## Gate B

### 扩展 profile

card11/card15a/card16/card17 在生成期写出 Gate B selection/timeline，并强制
“资产→逐帧位置”发生变化。两个房间共 48 条。

### 双源 profile

`materialize_qa_v3_dual_gateb.py` 对剩余 14 个 profile 的 168 条候选物化孪生：

- 外观题只交换资产，路线保持；
- 空间/距离/运动题只交换路线，资产保持；
- endpoint 按 Gate B selection 重新解析；
- AudioProgram 保持事件时刻与 sound asset multiset，重新绑定 endpoint；
- 每条验证 per-asset 视觉轨迹确实变化；
- card15b 标记 `expected_gold_relation=preserve`，其余标记 `flip`。

结果：216/216 条 selected candidate 均有 Gate B 产物。

## 代表性真实运行

### card11：谁发声 / 都不是

通过候选 `apartment_0000__card11__002`：

- pixel：`/data/jzy/tmp/qa_v3_room_pilot_card11_002_pixel_20260901_v1`
- join：`/data/jzy/tmp/qa_v3_room_pilot_card11_002_pixel_join_20260901_v1.json`
- UE：`/data/jzy/tmp/qa_v3_room_pilot_card11_002_visual_20260901_v1`
- main：`/data/jzy/tmp/qa_v3_room_pilot_card11_002_audio_main_20260901_v1`
- Gate A：`/data/jzy/tmp/qa_v3_room_pilot_card11_002_audio_gateA_20260901_v1`

前三只 `visible_occluded`，source4 `fully_occluded`。main 由 source4 发声，
答案为 `none`；Gate A 改为 source1 发声。onset 保持，mixture 不同。
card11_001 因 source4 仍可见而被拒。

### card15a：在场数 / 叫过数

通过候选 `apartment_0000__card15a__002`：

- pixel：`/data/jzy/tmp/qa_v3_room_pilot_card15a_002_pixel_20260901_v1`
- join：`/data/jzy/tmp/qa_v3_room_pilot_card15a_002_pixel_join_20260901_v1.json`
- UE：`/data/jzy/tmp/qa_v3_room_pilot_card15a_002_visual_20260901_v1`
- main：`/data/jzy/tmp/qa_v3_room_pilot_card15a_002_audio_main_20260901_v1`
- Gate A：`/data/jzy/tmp/qa_v3_room_pilot_card15a_002_audio_gateA_20260901_v1`

frame 30 四只都可见。main 有 2 个不同发声实例，Gate A 有 3 个；四个 onset
一致、sound asset multiset 保持、mixture 不同。card15a_001 因两只完全遮挡被拒。

### card16：首叫者片尾状态

pixel join 本轮增加绑定帧条件：main 与 Gate A 首叫实例在 frame 12 都必须可见。

通过候选 `apartment_0000__card16__005`：

- pixel：`/data/jzy/tmp/qa_v3_room_pilot_card16_005_pixel_20260901_v2`
- join：`/data/jzy/tmp/qa_v3_room_pilot_card16_005_pixel_join_20260901_v2.json`
- UE：`/data/jzy/tmp/qa_v3_room_pilot_card16_005_visual_20260901_v1`
- main：`/data/jzy/tmp/qa_v3_room_pilot_card16_005_audio_main_20260901_v1`
- Gate A：`/data/jzy/tmp/qa_v3_room_pilot_card16_005_audio_gateA_20260901_v1`

frame 12 两只均为 `visible_occluded`；frame 74 分别为 `out_of_view` 与
`visible_occluded`。main/Gate A onset 保持，首叫 endpoint 交换，mixture 不同。

拒绝台账：card16_001/002/004 的 main 首叫者绑定帧完全遮挡；card16_003
两只均完全遮挡；card16_005 pixel v1 的 SSH 在输出创建前断开，v2 fresh 重跑成功。

### card17：跨段

通过候选 `apartment_0000__card17__001`：

- segment1：`/data/jzy/tmp/qa_v3_room_pilot_card17_001_visual_segment1_20260901_v1`
- segment2：`/data/jzy/tmp/qa_v3_room_pilot_card17_001_visual_segment2_20260901_v1`
- audio：`/data/jzy/tmp/qa_v3_room_pilot_card17_001_audio_segment1_20260901_v1`

两个 segment 各 75 帧。运行时相机分别为 `(-187.0, 20.7, 52.52°)` 与
`(438.7, -7.5, 178.42°)`；逐帧相机与 actor readback 不同。

## 新增工具

- `assemble_qa_v3_room_pilot.py`：聚合基础矩阵与补采，几何去重，按答案层选择；
- `materialize_qa_v3_dual_gateb.py`：为双源候选生成可渲染 Gate B；
- `finalize_qa_v3_room_pilot.py`：绑定 pilot、pixel join 与 visual/audio receipt，
  验证 main/Gate A 结构和 card17 readback。

## 测试

- pilot/扩展路径定点：51 passed；
- 顶层 tests（排除 unit 与既有重型 verify-audio）：304 passed；
- 完整 unit：3193 passed / 29 failed / 21 errors / 107 skipped。

29F/21E 与审计前基线逐项一致，仍来自未挂载的 strict-two-human 工作区证物和
既有 tool-index 的 `tools/audit` 目录问题；本 pilot 触碰域无新增失败。

## 边界与下一阶段

本轮证明两个住宅各能形成 108 条、18 profile 全覆盖的 research candidate，
并证明像素终裁、main/Gate A、Gate B、N-actor 和跨段工程链可真实执行。

本轮未证明：216 条全部完成 UE/音频渲染、A-only/V-only 不可作弊、人类可答性、
容差校准、Kujiale 批量 UE、card12/13/14 素材就绪或正式数据准入。

下一阶段应使用本 pilot 清单开展分形式、分缺失模态认证，不得把
`research_candidate` 提升为正式数据。
