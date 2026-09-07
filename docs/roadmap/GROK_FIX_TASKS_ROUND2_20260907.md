# 放量前修复第二轮任务书（交给 Grok，2026-09-07 深夜，Claude 起草，owner 已点头）

## 启动提示词（owner 一次性交给 Grok 的话）

请读 `docs/roadmap/GROK_FIX_TASKS_ROUND2_20260907.md`，这是第二轮任务书；依据是 Claude 对你第一轮修复的独立审核 `docs/roadmap/AUDIT_GROK_FIXES_20260907.md`（里面每条问题都给了文件行号、产物路径和复现命令）。第一轮的产物（`qa_pilot46_rerun_20260907_v1`、`qa_pilot46_merged_20260907_v1`）和原批 `attempt_01` 一律只读，新产物进新目录。按第 2 节分工开子代理并行做 H-2 到 H-7，H-1 由你亲自做（它改声学渲染，先做对照实验再改代码再重渲）。每项做完按文末格式写报告到 `docs/roadmap/grok_reports_20260907_round2/`。就绪报告这次只写事实和 owner 原话，不把"听着没问题"或"测试通过"写成放行依据。不许改阈值凑通过、不许删资产/题型/家族、不许代填人工试听、不许改 Claude 的审计器及其测试、不许改 Codex 与 Claude 的报告原文。

## 0. 环境与边界

- 从 `grok/pilot46-fixes-20260907` 的 `ba0150e` 新开分支 `grok/pilot46-fixes-round2-20260907`，用自己的 worktree（例如 `/data/jzy/tmp/wt-grok-pilot46-round2`）。Python、PYTHONPATH、`tmp/` 符号链接、GPU 礼貌与第一轮任务书第 0 节相同。
- 只读：`tmp/qa_pilot46_background_20260907_v1/`、`tmp/qa_pilot46_rerun_20260907_v1/`、`tmp/qa_pilot46_merged_20260907_v1/`。第二轮的音频重渲、重新合并全部写新目录。
- Claude 的复算脚本可直接复用（只读，不改）：`/data/jzy/tmp/claude_audit_20260907_v2/{tail_corr.py,itd_probe_multi.py,cue_aggregate.py,drr_probe.py,run_audit_rerun.py}`。
- 不 push、不合并 main、不动 Studio。

## 1. owner 裁定（第一轮六条继续有效，新增一条）

7. **双耳混响阶数修复：owner 2026-09-07 晚点头。** UE 路径渲染器给 RLR 的请求把球谐阶数写死为 0（`tools/acoustics/render_frame_readback_sequential_speech.py:343-344`），反射与混响在两耳完全相同（RIR 首达后 2 ms 起 L/R 相关 0.99–1.00），自建 A/B/C 与酷家乐的侧向声源几乎没有耳间时差与强度差。要改成有方向的混响并重渲 UE 家族的音频。视频与像素证据不动。

## 2. 任务

### H-1 双耳混响阶数（父代理亲自做；最高优先）
1. **对照实验先行**：取 `qa_pilot46_rerun_20260907_v1/episodes/qa_pilot46_20260907_authored_a_human_human/attempt_02/episode` 的同一 neutral readback 与计划，只把 `_simulation()` 的 `direct_sh_order/indirect_sh_order` 从 0/0 改为 3/1（其它参数先不动），把 RIR 重算到新目录（不写 attempt_02）。验收：用 `tail_corr.py` 的方法看两个 RIR 首达后 2 ms 起的 L/R 相关从 ≥0.99 降到 <0.9；用 `itd_probe_multi.py`/审计器看侧向源（−14°/+25°）在 stem 上出现与几何方向一致的 ITD/ILD。再加两组：`indirect_ray_depth` 64→200、`max_ir_seconds` 0.25→4.0（与 Habitat 路径 `examples/runtime/rir_cache_simulation_request_v2.json` 对齐），报 RIR 长度、湿尾长度、每个 RIR 的 RLR 耗时。把三组数字与 0/0 基线并列写进报告，选定生产参数（默认取与 Habitat 路径一致的 3/1/200/4.0，除非耗时不可接受，若不同要说明）。
2. **改代码**：阶数、深度、IR 长度从请求或房间包可配置，默认值写成与 Habitat 路径一致；RIR 缓存 request 与音频回执（`research_receipt.json` 的 `propagation`/`qa.propagation.simulation`）记录实际阶数；缓存 key 要包含这些参数，避免与旧 0 阶缓存混用。单测：请求写入与回执读回；0 阶旧缓存不被新请求命中。
3. **重渲 UE 家族 32 段 delivered 的音频**（apartment 7、authored 15、kujiale 6 里全部 delivered 段）：复用各自最终 attempt 的 `capture/`（视频不动），走 G-A 那种"复用 capture 重收口"的方式，新输出根 `tmp/qa_pilot46_audio_v2_20260907/episodes/<id>/attempt_03/`；重新生成 delivery（facts/questions/审计/闸门/batch_review）。Habitat 段不重渲（它们已是 3/1 阶）。
4. **验收**：32 段 RIR 尾部 L/R 相关 <0.9；按家族报"侧向事件（|方位|≥20°）stem 上 |ITD|≥0.15 ms 或 |ILD 2–6 kHz|≥2 dB 的比例"（用 `cue_aggregate.py` 的口径），与重渲前对照；混音峰值仍在 −6 到 −25 dBFS；题数变化如实报。

### H-2 合并、记账与分类（`tools/dataset/`、`src/avengine/qa/batch_delivery.py`、`batch_coverage.py`、`run_qa_batch.py`）
1. 合并脚本 `/tmp/ge_merge_final.py` 进仓库（例如 `tools/dataset/merge_qa_batch_attempts.py`），带单测，能从提交状态复现 46 格合并表；合并时把 `failed_episodes` 传进覆盖构建（第一轮合并表 9912 行里 0 行带失败记账）。
2. 回填三条旧记录（`authored_c_animal_device`、两条 blocked）的 `room_id/gap_state/asset_ids/failure_stage`（从 manifest 与 stderr 取，不猜），或用新记账器补跑一次记账；`authored_c_animal_device` 必须是 `evidence_missing_or_unsampled`（规划 200 次用尽），不是 `interface_not_implemented`。
3. `collect_batch_outcomes` 填 `failure_stage/gap_state`；`_looks_like_interface_defect` 的关键词匹配改成按失败阶段与异常类型的显式规则（规划用尽/预分配缺额 → evidence_missing；音频/收口/捕获的代码异常 → interface_not_implemented），单测三类各一条。
4. 覆盖表 `provenance` 指向生产工作树的 inventory/catalog，不是 Codex 工作树。

### H-3 复现性（`tools/dataset/build_qa_batch_manifest.py`、`run_qa_batch.py`、`tools/studio/run_qa_episode.py`、`src/avengine/rooms/room_package.py`）
1. 清单生成时 `request.room_catalog` 写生产工作树的绝对路径，`request.runtime.path_bindings` 写全（来自 catalog），不再依赖 shell 环境变量；`run_qa_batch.py` 的 producer.json 记录 `AVENGINE_*` 环境变量与完整 CLI 参数。
2. `package_from_catalog_entry` 的相对 `room_package` 路径相对 catalog 文件所在目录解析，不相对进程 cwd；plan 目录写入展开后的房间包快照与所用 path_bindings。
3. 单测：换 cwd 结果不变；缺绑定时报错信息列出变量名。

### H-4 校验顺序与闸门（`src/avengine/rooms/room_package.py`、`src/avengine/qa/exposure_gate.py`、`batch_delivery.py`）
1. 路径存在性检查在**展开之后**做（第一轮是 `resolve(validate(package))`，模板路径被静默跳过），相对路径相对仓库根或 catalog 目录解析后也检查；七包在生产装载顺序下真的跑出 0 缺失。
2. 闸门 `except ImportError: pass` 改为抛错或写 `exposure_gate.status=unavailable` 并让 review 失败；review 记录帧来源（`capture/frames` 真帧还是 `batch_review/frames` 审阅抽帧），Habitat 段没有 PNG 目录时从 `rgb.npy` 取第 0/中/末帧。
3. 曝光补偿只留一个真值源：删掉 `AUTHORED_USD_EXPOSURE_BIAS_EV` 硬编码表或给它加单测钉住与包一致；地板四份文件的 `status` 不再写 `measured`（写 `depth_readback_fallback`），`summary.hit_count` 不再沿用深度回读的 64。

### H-5 外观与像素字段（`src/avengine/rooms/qa_evidence.py`、`batch_delivery.py`、登记表）
1. 把 `annotate_pixel_visibility_semantics` / `annotate_achieved_conditions_visibility` 真接进 pixel truth 与 achieved_conditions 的生成（第一轮 19 段 attempt_02 里 0 段有 `visible_pixel_frames`/`bbox_touches_frame_edge_frames`）。
2. 外观分类器缺口如实入表：11 个设备色值（silver×4、white_satin×2、warm_gray、beige、light_gray、sandstone、light_gray_fabric）与 3 种毛色（dark_sable、standard_sable、standard_seal_point）没有分类器 → actor 级 reason 写 `registered_appearance_value_classifier_not_implemented`，覆盖表记 `interface_not_implemented`；2 台电视登记表既无 finish 也无 body_color → 补登记字段（实际颜色以资产为准，不猜）。
3. 在 Grok 自己的报告里更正 G-D 的两只狗写反（第 0 帧 5 像素的是被扶手椅遮住的黄狗 source1，杰克罗素 source2 第 0 帧 2970 像素 pass）。

### H-6 就绪报告口径（文档）
1. 结论只写事实：机器闸门通过、契约通过、owner 对哪几条缺额的原话；"听着没问题"与"测试通过"不作为放行依据出现在结论句里。
2. owner 裁定按原话转述（"不要求每个房间跑出每种题"针对先导四格；"先导切分 train39/eval0 接受"），不扩写成"无独立 eval"、"缺额不必每房每题"。
3. 单测计数附文件清单与 pytest 摘要。
4. 五段试听清单：说明 owner 本机试听包实际是五段 human_human（4 段 attempt_01 + A 房 attempt_02），与 `five_clip_listening_pending.json` 的清单不同；H-1 重渲后重新出一包，建议含动物与设备声各至少一段（owner 决定要不要听）。
5. 第一轮 G-B 报告里"任务书写 −3 EV 与 B 房 −4 冲突需 owner 拍板"是假冲突（任务书没有任何 EV 数值），在第二轮报告里更正；"闸门未接入 batch_delivery"已过期，同样更正。

### H-7 放量干跑请求与结果侧统计（`src/avengine/qa/batch_manifest.py`、`tools/dataset/build_qa_batch_manifest.py`）
1. `scaleup-dry-run --sounds` 分支要把声音池路径写进每个 request 的 `sound_pool`（第一轮 350 个 request 全缺，`run_qa_episode.py:65-66` 一跑就 TypeError）；干跑产物用 `--plan-only` 抽 6 段（每家族至少 1 段、含 1 段画外）实跑证明可执行。
2. `prepare_batch_manifest` 再打散时把 `scaleup.distance_range_m` 等放量配置丢了（`batch_manifest.py:493-497`），350 行 request 全是 [1.5, 4.5]；改成一次打散或把配置透传，验收：request 与 `scaleup_config.json` 的画像逐行一致。
3. `collect_batch_outcomes` 造的行补 `requested_source_classes`（结果侧交叉表塌成空键）；旧清单缺新画像键（`competitor_visibility`/`distance_range_m`/`separation_target_policy`）时按默认值比较，不再把 19 段重跑记成 `profile_matches_request=False`、配额 delivered 0。
4. 便携空调三台在声池里 0 条兼容声音（`air_conditioning` 20 条全部超 5 s 被拒）：报告改写为"没有声音"而非"身份用完"；处置等 owner 定（从候选范围拿掉，或允许裁剪超长设备声）。
5. 画外画像两条边界写清并等 owner 定：speaker_moving 的画外锚点在不说话时会走进画面（复核实测 mp3d 段 |方位| 最小 8.7°、hm3d 段 1.1°）；干跑画外槽写死取每房第 1、2 槽，设备画外零段。
6. 46 格清单里两条 repeat 缺额仍在（清单未重生成），如实写；放量清单重出后缺额应为 0（空调两条另计）。

## 3. 就绪门（第二轮）
1. 46 格合并表能从提交状态复现；四个未交付格子带阶段与五态，分类符合 owner 第 4 条。
2. UE 家族 32 段音频重渲完成并通过 H-1 验收；Habitat 段不变；全部 42 段闸门与契约通过。
3. 放量清单干跑用 H-3 后的生成器重出一份，`room_catalog`/`path_bindings` 完整、缺额 0（除已接受的便携空调两条）。
4. 相关单测 0 failed / 0 skipped，`docs/TOOL_INDEX.md` 一致。
5. 就绪报告按 H-6 口径重写，列出仍空着的项（挂装、人工校准、逐条试听、正式准入）。

## 4. 报告格式
同第一轮任务书第 5 节：改了哪些文件与提交号；测试计数与失败原文；验收产物路径与亲自核对结果；没做完的按三分法写；后续接口；需 owner 拍板的事单列。
