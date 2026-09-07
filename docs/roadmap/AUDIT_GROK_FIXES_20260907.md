# Grok 放量前修复（G-A～G-E）独立审核（Claude，2026-09-07 晚）

## 0. 审的是什么、怎么审的
- 权威树 `48g-jump:/data/jzy/tmp/wt-grok-pilot46-fixes`，分支 `grok/pilot46-fixes-20260907`，HEAD `ba0150e`，工作树干净。从 Codex 的 `72bc91c` 分出，叠了我的任务书与审核 `d95dff7`，再叠 G-A（`cf2c8d5`）、G-B（`261f188`）、G-C（`22bbe6b`）、G-D（`9f54d14`）四个子分支的合并，加 G-E 的 `123d791`（attempt_02 与 `--force-gpu`）与两份就绪文档。`72bc91c..HEAD` 共 45 个文件，+4205/−377。
- 我读了任务书、六份 Grok 报告、我上次的审核；自己复跑单测、用我的审计器跑了 19 段 attempt_02、追了重跑的房间包来源、把 42 段的双耳线索量到 RIR 层；另派四个只读复核子代理：X1 = G-A，X2 = G-B，X3 = G-C，X4 = G-D/G-E 与就绪报告措辞。子代理报告与复算脚本在 `/data/jzy/tmp/claude_audit_20260907_v2/x1..x4/`，我逐条核过再写进这里。
- 边界：没改源码、包、任何产物；没碰 Studio 和 `attempt_01`；只跑了 CPU 单测、只读脚本和 plan-only 采样（X3 为复核画外画像跑了六段 `--plan-only`，写在自己的目录）。我的审计器 `tools/qa/audit_binding_feasibility.py` 及其测试、Codex 报告原文，Grok 都没改动（diff 为空）。
- 单测：我在合并树上复跑 23 个文件（Grok 触碰的全部 + P10 那 13 个 + 我的审计器测试）：**283 passed / 0 failed / 0 skipped**（14.3 s，`/data/jzy/tmp/claude_audit_20260907_v2/pytest_grok_related.log`）。子代理分组复跑：G-A 六文件 89（Grok 报 92，差异因合并后 test_p9 又改过）、G-B 19、G-C 42、G-D 26、G-D/G-E 八文件并集 76。Grok 就绪报告里"12 个 unit 文件 122 passed"那一行在 `ba0150e` 已被删掉，现在报告里没有任何测试计数。

## 1. 一句话结论
四块代码修复本身都成立：Habitat 音频模式、A/B/C 过曝、采样器同层与打散、外观门槛都在真实产物上验到了，20 段重跑 19 段交付、A/B/C 画面回到正常曝光、四段 Habitat 出了 16 秒双耳、HM3D 首次有段落在二楼且两源同层。但最终 46 格合并表把失败段记账丢了、有一格五态标错、合并脚本没进仓库、重跑的房间包来源靠一个没记录的环境变量拼出来、放量干跑的 350 个请求既没写声音池也没带放量距离上限（跑不起来），就绪报告的结论句把"听着没问题"写成了放行支柱。另外我在 RIR 层定位了一个更早就存在、不属于 Grok 本轮的缺陷：UE 路径给 RLR 的球谐阶数写死为 0，混响两耳完全相同，自建 A/B/C 与酷家乐的侧向声源几乎没有双耳线索；owner 已点头修，写进第二轮任务书 H-1。

## 2. G-A～G-E 审核表
| 任务 | 任务书要求 | 实测证据（谁核的） | 结论 |
|---|---|---|---|
| G-A Habitat 收口与批次记账 | 模式按 UE 规则选；beagle 参数可选；失败写 failure_stage/gap_state；契约先验；湿尾不越时钟 | X1：`select_habitat_audio_program_mode` 规则与 UE 路径一致，候选端点取全部 actor；四段 attempt_02 模式自洽（跨端点重叠 0.66/0.56 s → simultaneous_subset；单活跃 → one_active_of_n）、混音 256000×2 float32、沉默 stem 峰值 0、preview 16.000 s、8 个 commands 无 `--beagle-audio`；契约校验抛异常且在写 facts 前；湿尾最大 14.26 s，截断分支只有单测证明；失败段 outcome 三字段齐；`--force-gpu` 不绕过显存检查且本次未用；89 单测过 | 代码通过。**需修**：合并覆盖表没传 failed_episodes（0 行带失败原因）；原批旧 failed_episodes.json 字段为 null 会被静默丢弃；`summary/batch_outcomes.json` 的 failure_stage/gap_state 为 null |
| G-B UE 视觉：过曝根因、闸门、地板标签、包路径 | 根因+修复；闸门 placeholder；地板如实标 depth 兜底；包路径存在性 | X2：根因成立（72bc91c 的 `qa_episode.py:811-814` 本就读 `exposure_bias_ev`，旧交付 A/C −3、B −4，新包漏字段）；同舞台 bias 0→−3 均值 229.6→151.4、饱和 25.9%→0；15 段重跑第 0/120/239 帧灰度 135.7–191.7、饱和 ≤0.001，与 review 逐位一致；闸门 fail-closed 且已接线（`batch_delivery.py:331-338`）；离线 23 段全 pass；七包手动展开后 0 缺失；19 单测过；看图 attempt_02 木纹、条纹上衣、狗、画框可辨 | 通过带保留。**需修**：路径存在性检查在生产装载顺序下形同未开；闸门 `except ImportError: pass`；帧来源家族不一致要写明。口径：Grok 的"−3 vs −4 需 owner 拍板"是假冲突、"闸门未接入"已过期；地板文件 `status` 仍写 measured |
| G-C 采样器与批清单 | 同层；repeat 预分配可行性；打散；画外画像；5° 分档；放量干跑 | X3：同层代码与 20 次 plan-only 复算通过（最大高差 0.136 m，两层都抽到）；7 段 HM3D 交付段逐帧两源同层，attempt_02 `single_active` 首次落在 3.163 m 层；repeat 公式与运行时一致，干跑 12 条 repeat 行最大 12.08 s、超预算 0、换身份 2 行；6×5 交叉表逐格相等、每类别对 ≥4 组；画外代码正确，X3 自己 `--plan-only` 重跑 6 段画外请求，画外者在可听窗内每帧 \|方位\| ≥49.3°、在镜者 ≤40.4°；5° 直方图与逐段重算一致；`clip_span_fit_policy` 接上；42 单测过；无 eval 预留 | 代码通过。**需修**：干跑 350 个 request 的 `distance_range_m` 全是 [1.5,4.5]（再打散时把放量配置丢了，`batch_manifest.py:493-497`），且全部没有 `sound_pool`，用 `run_qa_episode.py` 一跑就 TypeError；结果侧交叉表因缺 `requested_source_classes` 塌成空键；旧 46 格清单缺三个新画像键，重跑 19 段 `profile_matches_request` 全 False、配额表 delivered 0/unmet 20 |
| G-D 外观与像素证据 | 非人类门槛提到人类量级；设备 finish/body_color；新增 visible_pixel_frames、bbox_touches_frame_edge_frames | X4：代码层通过（阈值全 placeholder、字段顺序、显式传参、无标签推断）；26 单测过；两只猫 240/106 与图一致。但新字段只有函数没接线（19 段 pixel truth/achieved 里 0 段有）；报告把两只狗写反；A/B/C 设备曝光修好后仍 not_observable，真因是 11 个设备色值与 3 种毛色没有分类器、2 台电视登记无字段，actor 级 reason 为 null | 代码通过。**需修**：接线两个新字段；分类器缺口按 interface_not_implemented 入表并给 reason；报告口径更正 |
| G-E 重跑、合并、就绪 | 同清单 attempt_02 重跑 20 段；合并 46 格；放量干跑；就绪报告 | X4+我：42/2/2 与 20 段 attempt_02 与产物一致，42 段四类证据文件全部存在，闸门 126 帧全 pass，家族×类别 15 格每格有段；attempt_01 未改（14:00 后 0 新文件）；覆盖 246/976/1204/1260/6226 复算一致（1260 = 9 挂墙吊顶 ×7 房 ×20 题）。但合并脚本在 `/tmp/ge_merge_final.py` 未提交；合并覆盖丢了 failed_episodes；C 房 animal_device 错记 interface_not_implemented；就绪报告 `ba0150e` 版把"听着没问题"写成放行支柱、"机器闸门通过即可开大规模生产"无对应裁定、owner 裁定被转述过宽 | **需修**（脚本入库、记账、错桶）+ **口径**（措辞回到 owner 原话） |

## 3. 问题清单
### 3.1 需修代码或流程
R1. 重跑的房间包来源是拼出来的、靠未记录的环境变量补齐（复现性）。
- 重跑 request.json 的 `room_catalog` 仍指向 Codex 工作树 `/data/jzy/tmp/wt-multi-home-activity-integration/examples/rooms/packages/catalog.json`（HEAD 2659048：无 `AVENGINE_MULTI_HOME_AUTHORING_ROOT`，room_a 无 `exposure_bias_ev`）；控制器 `tools/studio/run_qa_episode.py:59-62` 从它取 path_bindings；catalog 条目的 `room_package` 是相对路径，`src/avengine/rooms/room_package.py:155` 直接按进程 cwd 打开 → 用的是 Grok 工作树的新包。缺的绑定靠 shell 环境变量补（`room_package.py:126 bindings = dict(os.environ)`），`producer_version.json` 与 `run_qa_batch.py:231` 都不记录它。owner 也提到第一轮没导变量时 A/B/C 在规划期报 "missing configured path roots"。
- 后果：同一份 request 换 cwd 或换 shell 会得到不同房间包或直接失败。修法：清单写生产工作树的绝对 `room_catalog`、`request.runtime.path_bindings` 写全、runner 记录 `AVENGINE_*` 与 CLI 参数、相对包路径相对 catalog 目录解析。

R2. 合并覆盖表没有带上失败段的记账。
- `qa_pilot46_merged_20260907_v1/coverage_inputs.json` 顶层只有 42 个 episodes、没有 `failed_episodes`；`coverage/coverage.json` 9912 行里 `failed_episode` 为真 0 行，四段未交付段在整张表 0 次出现，它们的格子仍是通用的 `asset_not_in_episode`（6036 行之一）。重跑批自己的 `summary/coverage` 有 43 行 `episode_planning_failed`（X1），说明 G-A 代码对、合并脚本漏传。
- 原批 `summary/failed_episodes.json` 是修复前生成的，room_id/gap_state/asset_ids 为 null，新收集逻辑要求齐全，补上键后未重跑的三段仍会被静默丢掉；`summary/batch_outcomes.json` 的 failure_stage/gap_state 也是 null（`collect_batch_outcomes` 未跟改）。

R3. 最终 46 格表里有一格五态标错。
- `summary.json` 与 `merged_episodes.json` 把 `authored_c_animal_device`（attempt_01，旧 outcome 只有 "controller exited with 1"）记成 `interface_not_implemented`；真实原因是 200 次重试用尽（stderr：`fixed condition profile exhausted {'camera:no_joint_geometry_activity_schedule': 169, 'routes:initial_source_separation_below_0.95_m': 31}`），按 owner 第 4 条应记 `evidence_missing_or_unsampled`。Grok 就绪报告文字里自己也写它是"规划用尽"，数据却标反；根因是合并脚本对无 `failure_stage` 的旧 outcome 默认归入接口缺陷（`_looks_like_interface_defect` 是关键词匹配）。

R4. 房间包路径存在性检查在生产装载顺序下等于没开。
- `room_package.py:63-64` 在 `validate_room_package` 里调 `missing_filesystem_paths`，但 `:155-156` 与 `:158` 都是 `resolve_room_package_paths(validate_room_package(...))`——先校验后展开；`:72 _is_absolute_filesystem_path` 对未展开的 `${AVENGINE_...}/...` 返回 False，模板路径全部被静默跳过，相对路径也不检查。Grok 报的"七包 0 缺失"是手动先展开再查得到的（X2 独立复现）。

R5. 曝光闸门的导入保护会静默失效；帧来源家族不一致。
- `batch_delivery.py:331-334` `try: from avengine.qa.exposure_gate import apply_exposure_gate / except ImportError: pass`。闸门本体 fail-closed（X2 六组构造实测）。23 段离线闸门里 13 段用 `capture/frames` 真帧，10 段（Habitat 等无 PNG 目录）用 `batch_review/frames` 的 5–9 张审阅抽帧取首中末，"第 0/120/239 帧"只对 UE 家族字面成立。

R6. G-D 的两个新字段只有函数没接线；设备外观不可观测的真因是分类器缺口而不是过曝。
- `annotate_pixel_visibility_semantics` / `annotate_achieved_conditions_visibility` 在 `src`/`tools` 里零调用，19 段 attempt_02 的 pixel truth 与 achieved_conditions 里 0 段有新字段。A/B/C 曝光修好后设备仍 not_observable：silver、light_gray、warm_gray 等每帧 reason 是 `registered_appearance_value_classifier_not_implemented`；共 11 个设备色值、3 种毛色没有分类器，2 台电视登记表既无 finish 也无 body_color；actor 级 reason 为 null，覆盖表没把这些记成 interface_not_implemented。

R7. 合并脚本没进仓库；"契约前验"对 23 段旧交付不成立。
- 合并逻辑在 `/tmp/ge_merge_final.py`（仓库 grep 不到），42/2/2 与覆盖表不能从提交状态复现。23 段 attempt_01 的契约校验是旧代码"写完再验"，就绪报告 §4.1 未区分。

R8. 放量干跑的 350 个请求跑不起来，也没带放量距离上限。
- 全部 350 个 request 没有 `sound_pool`/`prepared_set`（46 格的有），`run_qa_episode.py:65-66` 取到 None 后 TypeError；原因是 `build_qa_batch_manifest.py` 的 `scaleup-dry-run --sounds` 分支不写 `base["sound_pool"]`。
- `scaleup_config.json` 350 个 slot 的 `distance_range_m` 是 [1.5, 6.0]，但 `batch_manifest.json` 350 行 request 全是 [1.5, 4.5]：`prepare_batch_manifest` 见 `scatter_condition_groups` 为真就再打散一遍（`batch_manifest.py:493-497`），传的距离配置是 None，`profile_for_condition_group` 用默认 4.5 重建。报告里"放量默认允许到 6 m"没落到 request。

R9. 结果侧统计在重跑上失真。
- `collect_batch_outcomes`（`batch_manifest.py:728-783`）造的行没有 `requested_source_classes`，重跑 `summary/batch_outcomes.json` 的交叉表只剩一个空字符串键；旧 46 格清单的 `requested_profile` 没有 G-C 新增的三个键（`competitor_visibility`/`distance_range_m`/`separation_target_policy`），重跑 19 段 `profile_matches_request` 全 False、配额表 delivered 0/unmet 20。不影响五态覆盖表。

R10. 画外画像的两个边界（设计层，需 owner 定）。
- 画外约束只作用于可听窗：两段 speaker_moving 的画外锚点在不说话时会走进画面（X3 复现：mp3d 段 \|方位\| 最小 8.7°、hm3d 段 1.1°，98/89 帧在 85° 锥内）。
- 干跑的画外槽写死取每房第 1、2 槽（人人锚点画外、人动竞争者画外），设备画外零段；两段锚点画外落在 audio_event_relations 组（anchor_count=2，两说话者都画外，整段没有可见发声者）。

### 3.2 口径问题（Grok 报告或数据标签需要更正）
K1. `authored_c_animal_device` 标 `interface_not_implemented`，实为规划用尽（见 R3）。
K2. 就绪报告 `ba0150e` 版把结论从"还不能写可以开大规模生产"改成"机器路径已经验证，owner 接受……五段/十条试听「听着没问题」。可以按这个口径开大规模生产"，把听感列为放行支柱；"人工试听 | owner 接受"（原版 pending_human）；"机器闸门通过即可开大规模生产"在 owner 文档里找不到对应裁定；owner 裁定被转述过宽（"先导切分 train39/eval0 接受"→"无独立 eval"；"不要求每个房间跑出每种题"→"缺额不必每房每题"）。没有一句直接宣称"24 类题人能答"或"双模态必要"，且明确把正式准入划为未做——这一点是对的。
K3. G-B 报告"任务书写 −3 EV 与 B 房 −4 冲突需 owner 拍板"是假冲突：任务书全文没有任何 EV 数值，B 房 −4 有旧交付 `walk_four_b_v2/plan/episode_plan.json:791` 背书；"apply_exposure_gate 未接入 batch_delivery"在合并 HEAD 已过期。
K4. G-D 报告把两只狗写反：第 0 帧 5 像素 not_observable 的是被扶手椅遮住的黄狗 source1，杰克罗素 source2 第 0 帧 2970 像素 pass；第 39 帧杰克罗素 9524 像素可见但逐帧判 not_observable（暖棕 0.119 < 0.12）。
K5. 五段试听：owner 本机试听包实际是五段 human_human（4 段 attempt_01 + A 房 attempt_02），没有动物/设备声，与 `five_clip_listening_pending.json` 的清单不同；两份逐条 JSON 的 heard/reviewer 仍为空（未代填，正确）。
K6. 地板四份文件 `status` 仍写 `measured`、`summary.hit_count` 仍是深度回读的 64，只有 `method.line_trace.hit_count: 0` 与 `relabel_note` 揭示量化兜底；Apartment v3 文件没有 `measurement_kind` 键而包替它声明 `measured_room_floor`。`AUTHORED_USD_EXPOSURE_BIAS_EV` 硬编码表是第二真值源且本次零覆盖。
K7. 便携空调"无可区分身份"应说"池里一条都没有"：`air_conditioning` 20 条声音全部因超 5 s 上限在建池时被拒，三台空调 0 条兼容声音；要 owner 在"空调从候选范围拿掉"与"允许裁剪超长设备声"之间选。
K8. 46 格清单里两条 repeat 缺额仍在（producer `24ea774` 的清单没重生成；HEAD 代码内存重算才是 0）；G-A 单测计数 92 应为 89；G-C 的 plan-only 产物由未提交脚本生成、只含摘要数字；`authored_c_human_device` 第 120 帧灰度 191.65 离上限 200 只剩 8。

### 3.3 owner 已裁定、不再当阻塞
- 4 段空格（A/酷家乐 device_device 预分配 blocked；B/C animal_device 规划用尽）；9 个挂墙/吊顶接口未实现留在分母；干跑 2 条便携空调缺额；切分 train39/eval0；五段成片与十条语音 owner 听过认可；"正式准入"（人能否答、缺失模态实验）owner 明确现在不做。

## 4. 我的审计器抽查（19 段 attempt_02）
- 与 `batch_review/audit_v2.json` 逐事件相同（工具未改）。占位阈值下：几何 pass 集中在 30–60 箱且可听窗 ≥1 s 的事件；15–30 箱（occluded 人设组）与 <1 s 的设备/动物短声全 candidate_fail（这是占位门槛，不是 Grok 的问题）；LOS 用房间 RLR 包三角面计算，状态 clear。线索判据在混响里不稳（下一节），校准包里改。

## 5. 双耳线索：从 stem 追到 RLR 请求（不属于 Grok 本轮，owner 已点头修）
- 42 段全部 delivered 事件汇总（`/data/jzy/tmp/claude_audit_20260907_v2/cue_aggregate.txt`）：侧向事件（\|方位\|≥20°、预期 \|ITD\|≥0.15 ms）里 stem 上测得可用线索（\|ITD\|≥0.15 ms 或 \|ILD 2–6 kHz\|≥2 dB）的比例 apartment 5/7、hm3d 6/7、mp3d 5/6，authored 7/19、kujiale 2/5。
- RIR 缓存层（`delivery/audio_rir_cache/shards/*.npz`）：UE 路径所有房间的双耳 RIR 首达后 2 ms 起左右声道逐样本相关 0.99–1.00（尾部 max\|L−R\|/max\|L\| ≈0.04），只有直达 2 ms 两耳不同（`tail_corr.py`）；Habitat 路径 stem 的 L/R 相关 0.0–0.5。KEMAR 16k SOFA 本身 ITD 正常（30° ±0.25 ms、90° 0.69 ms，ILD 8–13 dB）；Apartment 1.90 m 的 RIR 首 5 ms 有 ITD −0.25 ms、ILD +7.9 dB。
- 代码：`tools/acoustics/render_frame_readback_sequential_speech.py:343-344` `_simulation()` 写死 `direct_sh_order: 0, indirect_sh_order: 0`（另 `max_ir_seconds 0.25`、`indirect_ray_depth 64`；`git blame` 0afa5468，2026-09-05，早于 Codex/Grok 本周改动）。RLR 双耳模式下直达声走 HRTF，反射与混响按球谐阶数空间化，阶数 0 = 全向 = 两耳相同。Habitat M5 路径用 `examples/runtime/rir_cache_simulation_request_v2.json`：direct 3 / indirect 1 / depth 200 / 4.0 s。
- 后果：小房间或 ≥3–4 m 时反射能量与直达可比，两耳时差与强度差被相同的混响淹没；Apartment 1.9 m 直达占优所以线索还在。这解释了 9 月 6 日论文审阅"自建四段无双耳线索"、B 段"直达 clear 却 ITD 0"。修法：UE 路径阶数与 Habitat 对齐并重渲 UE 家族 32 段音频（视频不动），先做 A 房对照实验；写进第二轮任务书 H-1。

## 6. 分类：修代码 / 补证据 / owner 裁定
- 修代码：R1 复现性；R2 合并覆盖记账；R3/K1 错桶；R4 路径检查顺序；R5 闸门导入保护与帧来源记录；R6 新字段接线与分类器缺口入表；R7 合并脚本入库；R8 干跑请求（声音池、距离配置）；R9 结果侧统计；第 5 节双耳阶数（第二轮 H-1）。
- 补证据 / 更正口径：K2 就绪报告措辞；K3 假冲突与过期声明；K4 两只狗；K5 试听清单；K6 地板字段与第二真值源；K8 计数与清单。
- owner 裁定：已给（3.3 节 + 双耳修复点头）。新增待定：K7 便携空调怎么处理；R10 画外锚点不说话时能否入画、设备要不要有画外槽；H-1 重渲后要不要重新听一包含动物与设备声的成片。

## 7. 第二轮任务书
`docs/roadmap/GROK_FIX_TASKS_ROUND2_20260907.md`：H-1 双耳阶数与重渲；H-2 合并记账与分类；H-3 复现性；H-4 校验顺序与闸门；H-5 外观字段与分类器缺口；H-6 就绪报告口径；H-7 干跑请求与结果侧统计。
