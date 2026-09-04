# QA v3 房间中心 scene × profile 调度器 smoke（2026-09-01）

> 状态：**调度器实现与低成本真实 smoke 完成**。本报告只覆盖
> geometry/timeline/AudioProgram/fact 的 research/dev 装配，不启动 UE、
> 像素终裁、双耳渲染或单模态认证，不构成题型或房间准入。

## 1. 目标与结论

数据生产改为以房间为行、题型 profile 为列。每个已登记房间独立尝试全部
请求 profile；一个组合失败不阻断同房间其他题。有限随机搜索未找到时只写
not_found_within_budget，绝不外推为“房间永远不支持”。

实现：

- 调度器：tools/qa/run_qa_v3_room_profile_scheduler.py
- 当前五 profile：examples/qa/qa_v3_current_profiles_v1.json
- 单测：tests/test_room_profile_scheduler.py
- smoke：
  /data/jzy/tmp/qa_v3_room_profile_scheduler_smoke_20260901_final_v1
- 权威矩阵：
  /data/jzy/tmp/qa_v3_room_profile_scheduler_smoke_20260901_final_v1/scene_profile_matrix.json

## 2. 状态语义

- generated：至少产出一个候选；如果没有 pixel 输入，证据仍只是
  geometry_candidate。
- not_found_within_budget：本次预算内零候选，不证明绝对不可行。
- scene_infeasible：只允许显式 exhaustive proof，当前 smoke 无此状态。
- pixel_rejected：可选 pixel 结果完整覆盖全部几何候选且通过数为零。
- pipeline_error：配置、依赖、代码或产物闭合错误；其余组合继续。
- profile_not_implemented：请求 profile 不在当前实现目录，不记成房间失败。

另行报告 quota_status：

- filled：达到请求配额；
- partial：生成了候选但未填满；
- empty：已实现 profile 零候选；
- not_run：题型未实现或流水线未运行。

## 3. 真实 smoke

输入两个真实住宅资产、两个路线域；每格请求 2 个候选。额外请求 card16，
验证“题型未实现”不会混入房间不可行。

| scene | profile | attempt_status | quota | candidates/requested | evaluated | budget exhausted |
| --- | --- | --- | --- | ---: | ---: | ---: |
| apartment_0000 | card1F | generated | filled | 2/2 | 287 | 0 |
| apartment_0000 | card1B | generated | filled | 2/2 | 281 | 0 |
| apartment_0000 | card7 | generated | filled | 2/2 | 7 | 0 |
| apartment_0000 | card8 | generated | filled | 2/2 | 16 | 0 |
| apartment_0000 | card9 | generated | filled | 2/2 | 8 | 0 |
| apartment_0000 | card16 | profile_not_implemented | not_run | 0/2 | — | — |
| Kujiale livingroom_491 | card1F | generated | partial | 1/2 | 4,982 | 1 |
| Kujiale livingroom_491 | card1B | not_found_within_budget | empty | 0/2 | 6,000 | 2 |
| Kujiale livingroom_491 | card7 | generated | filled | 2/2 | 238 | 0 |
| Kujiale livingroom_491 | card8 | generated | filled | 2/2 | 381 | 0 |
| Kujiale livingroom_491 | card9 | generated | partial | 1/2 | 3,264 | 1 |
| Kujiale livingroom_491 | card16 | profile_not_implemented | not_run | 0/2 | — | — |

闭合：

- scene_count=2；
- expected/observed matrix cells=12/12；
- attempted_every_requested_profile_per_scene=true；
- attempt status：generated 9、not_found 1、not_implemented 2；
- quota status：filled 7、partial 2、empty 1、not_run 2；
- 两份 room_attempt_manifest 均有 6/6 题型尝试记录；
- 每格保留真实搜索分母、预算耗尽数与拒绝原因分布。

结果验证了核心语义：Kujiale 的 card1B 未找到没有阻断 card7/card8/card9；
card1F/card9 的部分成功没有被“generated”字段掩盖；card16 未实现没有被
误写为场景不支持。

## 4. 选择与证据边界

调度器只读取：

- scene config、导航路线库与相机基础请求；
- profile 与研究参数；
- 几何求解器产生的 manifest；
- 可选的显式 pixel 汇总。

它不读取模型分数、A-only/V-only 探针、最终答案表现或下游认证结果，不会
根据这些结果切换题型。每个 scene × profile 都独立尝试并保留自己的输出。

当前 generated 只证明装配链能生成候选。正式数据生产仍须接后续阶段：

1. UE/native visual capture；
2. 像素真值终裁；
3. 动态双耳音频与 Gate A；
4. MCQ/Open 双形式；
5. A-only/V-only/AV 与人类可答性认证。

## 5. 后续建议

先用少量已注册住宅扩展矩阵，不要求遍历全库。每个房间尝试全部已实现
profile；最终从矩阵中同时按“每种题覆盖多个房间”和“每个房间贡献多种题”
选择。⑯、⑤R、⑥R实现后只需增加 profile 列重跑；多源族仍等待
N-actor/N-source 四源 canary。
