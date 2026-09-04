# QA v3 全 profile 引擎实现与双房间验证（2026-09-01）

## 结论

QA v3 当前目录已包含 21 个可调度 profile。双房间调度器不再把任何一个请求记为
`profile_not_implemented`；每个房间都会独立尝试每个 profile，一个格失败不会切换题型，
也不会影响同房间其他题型。

本轮的最终低成本矩阵是：

- 2 个场景资产：`apartment_0000` 与
  `interioragent_kujiale_0020_livingroom_491`；
- 21 个 profile × 2 个房间 = 42 个独立格；
- 34 格生成 geometry/timeline/AudioProgram/fact 候选；
- 6 格明确为 `resource_unavailable`；
- 2 格为有限预算内 `not_found_within_budget`；
- 0 格 `profile_not_implemented`，0 格 `pipeline_error`。

权威矩阵：
`/data/jzy/tmp/qa_v3_all21_two_room_matrix_20260901_v4_reviewfix/scene_profile_matrix.json`。

这些结果是引擎能力与候选搜索证据，不是题型准入、单模态认证或正式数据集计数。

### 总审阻塞修复

Claude 总审发现旧版 N-actor RNG 只使用种子字符串前 4 字节。现已改为对完整种子
做 SHA-256 并取 64 位熵；同前缀、不同后缀的阳性测试会生成不同 plan。修复后：

- Apartment 四个扩展 profile 使用 4 个不同机位，Kujiale 同样为 4 个；
- 两个房间的 card17 segment1/segment2 均使用不同机位和路线；
- card16 外层重试不再重复同一 plan；失败搜索会累计真实评估分母；
- 非 `SearchExhausted` 异常直接上抛，由调度器记为 `pipeline_error`；错误 snapshot
  阳性探针已得到 `completed_with_pipeline_errors`；
- Gate B 生成期按“资产→逐帧位置”比较，槽位重标注但视觉不变会硬失败；
- card11 使用可见的外观标签而不是 `source1` 等内部槽位名；
- pixel join 新产物记录 fact 与 pixel truth 的绝对路径及 SHA-256，避免跨批碰巧等价。

失败语义阳性证据：

- 非搜索异常：`/data/jzy/tmp/qa_v3_reviewfix_pipeline_error_probe_20260901_v1`
  明确得到 `pipeline_error`；
- 七次不可行搜索：`/data/jzy/tmp/qa_v3_reviewfix_search_denominator_probe_20260901_v1`
  明确记录 `evaluated_combinations=7`，不再出现“预算耗尽但分母为 0”。

## 引擎能力变化

### 通用 N-actor / N-source

以下原先固定为 `source1/source2` 的执行链已改为连续的
`source1..sourceN`：

- actor selection 与逐帧 timeline；
- UE actor spawn、根姿态和动画读回；
- RGB 捕获；
- target-only depth、语义 ID 与逐实例像素真值；
- endpoint registry 与 emitter height 解析；
- AudioProgram 候选端点与逐事件 sound asset；
- 动态 RIR 与双耳 stem/mixture 输出。

四角色真实 canary 已完成：

- 75 帧四角色 UE RGB：
  `/data/jzy/tmp/qa_v3_n4_apartment_visual_20260901_v2`；
- 四实例像素真值：
  `/data/jzy/tmp/qa_v3_n4_apartment_pixel_20260901_v3`；
- 四端点、25 个 RIR 关键帧、4 个独立双耳 stem：
  `/data/jzy/tmp/qa_v3_n4_apartment_audio_20260901_v1`。

### 扩展 profile 执行器

`design_qa_v3_extended_profile.py` 负责 ⑪、⑫、⑬、⑭、⑮a、⑯、⑰。
它与双源执行器共享房间配置、路线库、相机/听者姿态和 runtime registry，不含房间专用坐标。

每个可实例化候选会写出：

- MCQ 与 Open 两种题面及同源事实；
- actor selection、timeline、M1 request、endpoint registry；
- main 与 Gate A AudioProgram；
- Gate B actor selection、timeline 和干预说明；
- 像素依赖声明或精确引擎真值；
- 显式研究边界。

当前素材不足不会伪装成场景搜索失败：执行器先做 registry preflight，并返回
`resource_unavailable` 和精确缺项。

## 21 个 profile 当前状态

| profile | 题义 | Apartment | Kujiale | 当前证据 |
|---|---|---:|---:|---|
| card1F | 末声后片尾方位 | generated | generated | 既有 run02 AV + 本轮两房间搜索 |
| card1B | 末声者过去时刻方位 | generated | budget 内未找到 | 既有 run02 AV + 本轮两房间搜索 |
| card2 | 当前发声者数值/方位带 | generated | generated | 两房间 geometry/timeline/program |
| card3 | 首声来自左/右 | generated | generated | 两房间 geometry/timeline/program |
| card4R | 指定时刻谁更近 | generated | generated | 两房间 geometry/timeline/program |
| card5 | 发声期间靠近/远离 | generated | generated | 两房间 geometry/timeline/program |
| card5R | 声停后靠近/远离 | generated | budget 内未找到 | 两房间 geometry/timeline/program |
| card6 | 第二声期间动/静 | generated | generated | 两房间 geometry/timeline/program |
| card6R | 第二声后静默段动/静 | generated | generated | 两房间 geometry/timeline/program |
| card7 | 查询时刻哪只在叫 | generated | generated | 既有 run02 AV + 本轮两房间搜索 |
| card8 | 指定外观实例首叫时刻 | generated | generated | 既有 run02 AV + 本轮两房间搜索 |
| card9 | 先叫者外观 | generated | generated | 既有 run02 AV + 本轮两房间搜索 |
| card10 | 首声者发声时动/静 | generated | generated | 两房间 geometry/timeline/program |
| card11 | 三只可见 + 都不是 | generated | generated | 两房间候选；Apartment 首个像素候选被正确拒绝 |
| card12 | 指定外观实例发何种声 | 素材不足 | 素材不足 | 缺第 4 种登记语义声类 |
| card13 | 指定衣色者说了什么 | 素材不足 | 素材不足 | 缺第 4 受控衣色和 4 条带转写语音 |
| card14 | 说指定句者穿什么 | 素材不足 | 素材不足 | 同 card13 |
| card15a | 在场数 + 叫过几只 | generated | generated | Apartment 完整视觉/像素/main+Gate A 双耳实跑 |
| card15b | 总共几声 | generated | generated | 两房间 geometry/timeline/program；纯音频对照 |
| card16 | 首叫者片尾遮挡四态 | generated | generated | Apartment dev 参数完整视觉/像素/main+Gate A 双耳实跑 |
| card17 | 跨段身份与第二段位置 | generated | generated | Apartment 两段各 75 帧真实 UE + 第一段双耳实跑；future extension |

“budget 内未找到”只表示本次随机搜索配额未填满，不是房间永远不支持该题。

## profile 级真实运行证据

### ⑮a 在场数 / 叫过数

同一个 Apartment 候选完成：

- 四角色 75 帧 UE 回放：
  `/data/jzy/tmp/qa_v3_all21_card15a_apartment_visual_20260901_v1`；
- frame 30 四实例均为可见状态，像素 join 通过：
  `/data/jzy/tmp/qa_v3_extended_pixel_join_card15a_20260901_v2_bound.json`；
- main 双耳：
  `/data/jzy/tmp/qa_v3_all21_card15a_apartment_audio_20260901_v1`；
- Gate A 双耳：
  `/data/jzy/tmp/qa_v3_all21_card15a_apartment_audio_gateA_20260901_v1`。

main 与 Gate A 都保持 4 个事件和相同 onset；不同发声实例数从 1 变为 4。
两份 mixture 的 SHA-256 不同，说明干预实际进入了渲染结果。这里的 hash 只是本次
波形比较结果，不是新增冻结 contract。

### ⑯ 两跳遮挡

在 Apartment dev 参数下，一个完整候选已闭合：

- 75 帧 UE：
  `/data/jzy/tmp/qa_v3_extended_card16_apartment_visual_20260901_v2`；
- 原生像素：
  `/data/jzy/tmp/qa_v3_extended_card16_apartment_pixel_20260901_v2`；
- 像素 join：
  `/data/jzy/tmp/qa_v3_extended_pixel_join_card16_20260901_v3_bound.json`；
- main / Gate A 双耳：
  `/data/jzy/tmp/qa_v3_extended_card16_apartment_audio_main_20260901_v2` 与
  `/data/jzy/tmp/qa_v3_extended_card16_apartment_audio_gateA_20260901_v2`。

main 首叫端点为 source1，Gate A 首叫端点为 source2，事件时刻不变；frame 74 的
像素真值分别是 `visible_occluded` 与 `out_of_view`，MCQ/Open 金标因此分离。
最终双房间矩阵使用更严格的 min-distance 参数重新搜索，Apartment 与 Kujiale 均生成
geometry candidate；上述 dev 像素/音频证据仍只绑定其自己的明确输入，不替代矩阵候选。

### ⑰ 跨段

同一 selection 的两段各完成 75 帧真实 UE 回放：

- segment 1：
  `/data/jzy/tmp/qa_v3_reviewfix_card17_matrixv4_visual_segment1_20260901_v1`；
- segment 2：
  `/data/jzy/tmp/qa_v3_reviewfix_card17_matrixv4_visual_segment2_20260901_v1`；
- segment 1 双耳：
  `/data/jzy/tmp/qa_v3_reviewfix_card17_matrixv4_audio_segment1_20260901_v1`。

两段运行时相机分别为 `(277.9, 406.6, -32.79°)` 与
`(53.8, -2.4, -124.52°)`，frame 40 角色位置也不同，真实证明了换路线/机位后
重新求位置真值；它仍是 future extension，未证明人类可辨识，也未进入主投稿题型准入。

### ⑪ 像素拒绝证据

首个 Apartment geometry candidate 的 source1..source4 在 frame 30 都可见，因此
不符合“三只可见、第四只画外/全遮”的题面。像素 join 将其记为
`pixel_rejected`：

`/data/jzy/tmp/qa_v3_extended_pixel_join_card11_20260901_v2_bound.json`。

这不是 card11 实现失败，而是说明真实像素终裁正在阻止几何近似冒充合格样本。
三份 bound join 均内嵌自己的 fact/pixel 路径与摘要；card11/card15a 仍明确绑定历史
matrix v1 候选，不作为 reviewfix matrix v4 的像素证据。后续生产需要继续采样，
直到第四只确实不可见且 matched-DoA/物理探针同时通过。

## 测试

- 新增路径定点测试：48 passed（含种子、Gate B、像素输入绑定与 ⑫⑬⑭ 语义路径）；
- 顶层 tests（排除既有重型 verify-audio 与 unit）：301 passed；
- 完整 `tests/unit`：3193 passed / 29 failed / 21 errors / 107 skipped。

完整 unit 的 29F/21E 与审计前基线类别一致：缺失
`strict_two_human_*` 工作区证物，以及既有 `tools/audit` 未进入 tool-index
目录表；本轮触碰的测试域零失败。与上一审计的 3190 passed 相比新增 3 个通过项，
来自 N-source unit 覆盖。

## 尚未完成

- ⑫：需要第 4 种可登记、可渲染、片段内实际出现的语义声类；
- ⑬⑭：需要第 4 受控衣色，以及至少 4 条含 transcript 的 speech asset；
- ⑪：需要像素引导的继续采样和 matched-DoA/物理侧信道实测；
- 34 个 generated 格仍只是 research geometry candidate；未完成双轨单模态认证、
  人类可答性、容差校准和正式数据集准入；
- Kujiale 本轮只验证通用路线/相机/事实装配，未新增该场景的 UE 打包渲染证据。

远端未推送；本报告及代码只在服务器 worktree 中。
