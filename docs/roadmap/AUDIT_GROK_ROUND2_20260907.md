# Grok 第二轮放量前修复（H-1～H-7）独立审核（Claude，2026-09-08 凌晨）

## 0. 审核对象、边界与方法

- 对象：`48g-jump:/data/jzy/tmp/wt-grok-pilot46-round2`，分支 `grok/pilot46-fixes-round2-20260907`，HEAD `6cd368e49c00d4a3f9c22b42285f4f47da8d4246`，`git status --porcelain` 0 行（审核开始与结束各查一次）。`git log ba0150e..HEAD` 共 19 个提交：789bc64 是我的第一轮审核与第二轮任务书，其上是 Grok 的 H-1～H-7 实现与六个子分支的合并；`git merge-base HEAD grok/pilot46-fixes-20260907` = 789bc64，与 owner 描述一致。
- 只读确认：`tools/qa/audit_binding_feasibility.py` 在本树、第一轮树、Codex 树三处 sha256 相同（`53c2dd24…`）；`tests/test_audit_binding_feasibility.py`、`docs/roadmap/codex_reports_20260906/`、`docs/roadmap/AUDIT_P1_P12_PILOT46_20260907.md`、`docs/roadmap/AUDIT_GROK_FIXES_20260907.md`、两份任务书、第一轮 `grok_reports_20260907/` 在 `789bc64..HEAD` 的 diff 为空。原批 `qa_pilot46_background_20260907_v1`、第一轮 `qa_pilot46_rerun_20260907_v1`、`qa_pilot46_merged_20260907_v1` 三个目录在 2026-09-07 21:30 之后 0 个文件被改（`find -newermt`）。Studio（PID 2266899）审核期间一直活着，我没碰它，也没起任何 GPU/UE 任务。
- 环境：`/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python`，`PYTHONPATH=src:tmp/native_python_addons_v1`，`import avengine` 解析到 `/data/jzy/tmp/wt-grok-pilot46-round2/src/avengine/__init__.py`；`tmp/` → `/data/datasets/avengine_workspaces/multi_home_activity_20260905/root/`。
- 方法：先自己核基本面（HEAD、状态、保护文件、单测复跑、审计器抽查），H-1 与专题 A/B 由我亲自复算；H-2、H-3+H-7、H-4+H-5 各派一个只读子代理按文件逐条核，报告在 `/data/jzy/tmp/claude_audit_round2_20260907/x1..x3/REPORT_x*.md`；子代理的每个关键数字我都自己再摸了一遍才写进本文。第一批子代理在 API 限额时中断过一次，第二批复用它们留下的脚本与中间产物重跑。
- Grok 的八份报告（`docs/roadmap/grok_reports_20260907_round2/`）只当索引；下文每条判定都给文件行号、产物路径或复现命令。所有新产物只写在 `/data/jzy/tmp/claude_audit_round2_20260907/`。

## 1. 结论（先说）

1. **七项代码修复都成立**，没有一处是"改了报告没改代码"。H-1 的阶数、深度、IR 长度可配置且默认与 Habitat 路径一致，进了回执与缓存身份，旧 0 阶缓存会被拒绝复用；H-2 合并脚本进仓库且能从提交状态逐字节复现 46 格；H-3 的绑定不再吃 shell 环境、包路径相对 catalog 解析、producer 记 argv 与环境；H-4 先展开再校验、闸门导入失败会让 review 失败、硬编码曝光表删净、地板文件是诚实改标签；H-5 两处接线在位、电视颜色的 texel 依据我亲自复核一字不差；H-7 声音池、距离透传、结果侧统计三处都修对了。相关单测我复跑 41 个文件 **581 passed / 0 failed / 0 skipped**。
2. **H-1 的物理目标达到了，但"stem 上可用线索比例明显变好"这句不成立。** 200 条 RIR 作业首达 2 ms 后左右相关最大 0.678（重渲前 ≥0.99），直达段的耳间时差保住了（与 KEMAR 同方位 ITD 之比中位 0.92），直达 ILD 符号与几何一致 24/28。可是按任务书写的 `cue_aggregate.py` 口径，28 个侧向事件里有线索的从 14 变成 20，不是 Grok 报的 24/28（那是它自己脚本的 30 ms 窗口径）；只看 ILD 是 12 变 10；审计器自己的 `delivered_cue_state` 从 47 pass 变成 44 pass。原因是 3/1 阶给了一条能量更大、两耳不相关的尾巴，长窗指标被稀释——这是我第一轮就欠的 C3（判据改直达窗加尾部相关），不是 Grok 的错，但在人工校准之前谁的"线索比例"都不能当放行数字。
3. **第二轮交付的三类产物都比合并后的代码旧，这是本轮最要紧的需修项。** （a）合并表 `qa_pilot46_merged_round2_20260907_v1` 在 H2 子树 `ec385e0` 上生成，比合并后 HEAD 少记 105 行外观分类器接口缺口（deferred 976 应为 871，interface 1260 应为 1365）、23 个闸门旁证缺帧来源、登记表少两台电视的颜色；（b）H-7 的 350 个放量 request 和 6 段 plan-only 全部钉在临时子树 `wt-grok-pilot46-H7` 上，而且是 789bc64 加三个未提交文件的脏树跑的，与第一轮 R1 同类复发；清单还没记 `source_registry` 与输入哈希，换成生产 registry 复现时 343/350 格的资产分配全变而聚合数字全同；（c）28 段 attempt_03 是 22:01–22:33 在只含 H-1 的树上收口的，H-4 的帧来源与 H-5 的像素字段一个都不在产物里。**代码不用改，产物要在合并后 HEAD 重出一遍**：合并表换目录重出、干跑在生产树重出并记全输入、attempt_03 的收口至少抽几段重跑。
4. **专题 A（削波）**：`authored_b_device_device` 的削波在 source1（音响放音乐）这一条 stem，不是叠加；峰值是 3/1 阶的反射堆出来的（直达段比 0 阶还低 2 dB，整条 RIR 比 0 阶多 6.1 dB 能量），56 条首帧 RIR 的晚期能量整齐抬高 7.5 dB。0 阶原片 −5.40 dBFS 本来就在我写的"−6 到 −25"之外。我的建议是保持渲染器 fail-closed，如果要保住这个格子并给放量兜底，加一个**全批同值、渲染前声明、回执照录**的卷积侧常数（0.5 给出 2.6 dB 余量），UE 28 段与 Habitat 14 段一起重渲以免家族间出现 6 dB 台阶；不要给单段加增益。owner 拍板。
5. **专题 B（−6～−25 dBFS 窗）**：这条窗只出现在我写的第二轮任务书里，代码里没有任何峰值门，第一轮已交付的 42 段本来就有 7 段高于 −6（UE 3、Habitat 4）。三段热成片没有任何样本超过 1.0，没有失真；它们该不该重渲只取决于专题 A 的决定。不要把这条窗变成闸门。
6. **就绪口径**：Grok 第二轮就绪报告没有再把听感或测试变绿写成放行依据，这一点做到了；但它的数字有四处要改（1260/976 过期、"56/56"只数了每段前两条作业、"catalog 为生产树绝对路径"不实、第一轮 20 格交叉表闸门重算后其实不过）。owner 本轮听了六段 attempt_03 说"其实听起来都没啥问题"，本文只当专题 A/B 的一条输入。**本文的结论句不以测试通过、不以听感为放行依据。**

## 2. H-1～H-7 逐项核对表

### H-1 双耳混响阶数（我亲自复算，详见第 4 节）

| 要求 | 实测 | 判定 |
|---|---|---|
| 对照实验先行 | `tmp/h1_sh_ablation_20260907/authored_a_human_human/ablation_metrics.json`：基线两条 RIR 2 ms 后 L/R 相关 0.9958 / 0.9966，3/1 各组 0.35–0.45；存储 IR 长度只随 depth 变（64→0.66 s，200→1.73–1.78 s），`max_ir_seconds` 不改变长度。Grok 表与文件一致 | 通过 |
| 阶数/深度/IR 长度可配置、默认与 Habitat 一致 | `render_frame_readback_sequential_speech.py`（703e4db）`DEFAULT_* = 3/1/200/4.0`；CLI 四个开关加 `--simulation-request` overlay；`build_audio_command` 从 request/runtime/plan 的 `simulation` 透传。**"或房间包可覆盖"不成立**：`src/avengine/rooms/` 没有任何地方把房间包字段写成 plan 的 `simulation`。小瑕疵：`render()` 第 2201 行只在 depth 等于默认值时才让 overlay 的 depth 生效 | 通过（两处口径/小修） |
| 回执与缓存记录实际阶数、旧缓存不混用 | 28 个缓存 `request.json` 的 `simulation.effective` 全是 3/1/200/4.0 双耳；27 份回执 `qa.propagation.simulation` 逐键相等；`_request_identity_payload()`（`rir_cache.py`）把 `simulation.effective` 纳入 `request_identity_sha256`，同一段 attempt_02/03 的 plan sha、scene sha 相同而 identity 不同（`3f7e0d42…` 对 `49162fed…`）；`_existing_rir_cache_sequence()` 在生产路径上拒绝 effective 不等的缓存 | 通过 |
| 重渲 UE 家族 delivered 段音频 | 28 段（apartment 7、authored 15、kujiale 6）；attempt_03 的 capture/plan/request 是指向最终 attempt 的符号链接；Habitat 无 attempt_03；三个只读目录 21:30 后 0 文件被改。任务书写"32 段"是我算错（7+15+6=28） | 通过 |
| RIR 尾部 L/R 相关 <0.9 | **200 条作业**全部 <0.9，最大 0.678；家族中位 apartment 0.446 / authored 0.529 / kujiale 0.436。Grok 的 `/tmp/h1_accept_28.py::rir_tail` 只读每段第一个 shard（2 条作业），所以报"56/56" | 通过（计数口径要改） |
| 侧向事件线索比例（任务书口径） | 28 个侧向事件：重渲前 14 → 重渲后 **20**（apartment 5→7，authored 7→9，kujiale 2→4）；只看 ILD≥2 dB：12→10；审计器 `delivered_cue_state` 全部 54 事件：pass 47 / fail 7 → pass 44 / fail 9 / unmeasured 1 | 通过但**不是 24/28**；Grok 用的是自己脚本 30 ms 窗口径 |
| 混音峰值 | 27 段：最低 −20.71、最高 −1.63、中位 −12.89 dBFS；比重渲前中位 +3.50 dB（均值 +3.30，最大 +7.37）；高于 −6 的 3 段；1 段削波拒写 | 事实一致；"窗"见第 6 节 |
| 题数 | 27 段 260 条，重渲前同 27 段 268 条（apartment 59→61，authored 153→145，kujiale 56→54） | 通过 |
| 闸门/契约 | `rerender_summary.json` 27 段 review delivered、exposure_gate pass；但收口用的是 22:00 时只含 H-1 的树，attempt_03 的 review 没有 H-4 的 `frame_source`、`achieved_conditions.json` 没有 H-5 字段（0/27） | 部分（产物早于代码） |
| 审计器抽查 | 未改过的 `audit_binding_feasibility.py` 对 27 段 attempt_03 重跑，`events` 与 Grok 的 `batch_review/audit_v2.json` 27/27 逐字节相同 | 通过 |

### H-2 合并、记账与分类（x1 复核，关键数字我抽核）

| 要求 | 实测 | 判定 |
|---|---|---|
| 合并脚本进仓库、只读输入、登记 TOOL_INDEX | `tools/dataset/merge_qa_batch_attempts.py` 在 6162efa（557 行），只写 `--output`（`:33-36`、`:430`；`batch_coverage.py:1649-1672` 对已存在目录抛 `FileExistsError`）；复现前后两个输入目录的 find 清单（11341 + 5668 行，含 mtime/字节数）逐行一致；`docs/TOOL_INDEX.md:416` 已登记 | 通过 |
| 能从提交状态复现 46 格 | 在 HEAD 6cd368e 重跑（`/data/jzy/tmp/claude_audit_round2_20260907/x1/run_merge_repro.sh`）：46 格、42/2/2、闸门 42 pass、9912 行、166 行带 `failed_episode`、4 条 `failed_episodes`、逐 episode 记账字段 0 差异 | 逻辑通过 |
| **交付产物与合并后 HEAD 一致？** | **不一致。** `tmp/qa_pilot46_merged_round2_20260907_v1/coverage_state_counts.json`：deferred 976 / interface 1260；HEAD 复现：deferred **871** / interface **1365**。逐行对齐（主键 asset_id+room_id+qa_id+scope+question_ids，两边键集合相同）只有 105 行变化，全部 `deferred_by_rule → interface_not_implemented`，reason 从"没有匹配的已审外观值"换成 `registered_appearance_value_classifier_not_implemented`（7 个资产、5 个房间、12 个题号；Grok 产物该 reason 0 行，复现 105 行，`grep -c` 210=105×2）。另两处同源痕迹：23 个 `exposure_gate_offline/*.json` 缺 H-4 加的 `frame_source` 块；H2 子树登记表比生产树少两台电视的 `body_color: black`（md5 `99ba0924…` 对 `796e8e2d…`）。原因是提交顺序：产物在子树 `wt-grok-pilot46-H2`（`ec385e0`）上生成，H-5（a219ae4）与 H-4（619d1fd）之后才合入 | **需修复（重出产物）** |
| 三条旧记录回填 | `authored_c_animal_device`：`failure_stage=planning`、`gap_state=evidence_missing_or_unsampled`、`reason_code=planning_exhausted`，reason 带 stderr 直方图 169+31=200；两条 blocked `reason_code=preallocation_gap`；`authored_b_animal_device` 有 room_id 与两个 asset_id；四条 room_id/asset_ids 与 manifest `source_assignments` 逐字一致 | 通过（R3、K1 已修） |
| 关键词匹配改显式规则 | `run_qa_batch.py:480-508` `gap_state_for_failure`：先看结构化 `reason_code`/`failure_histogram`，再按阶段分支，分支内按正则抠出的异常类型名查 16 个名字的白名单（`:392-411`）。主路径显式了，兜底仍是关键词（四处共 15 个短语）。三类单测在 `tests/unit/test_merge_qa_batch_attempts.py:274-292、294-341`；`_looks_like_interface_defect`（`:510-512`）成了无调用者的死代码 | 部分修复 |
| 默认落点 | 阶段未知且异常不在白名单 → `evidence_missing_or_unsampled`（`:506-507`）。方向对（第一轮默认相反，正是它把 C 房标错），但白名单外的异常名会被吞进"证据缺失"，引擎 bug 在表里看不见；没有单测钉住默认。`audio`（`:497-498`）与 `launch`（`:503-504`）无条件记 interface，不看异常类型，与任务书"…的代码异常"字面不符 | 需修代码或 owner 定口径 |
| `collect_batch_outcomes` 填两字段 | `batch_manifest.py:842-847、854` 填了，与 H-7 的 `requested_source_classes`（`:837-841、850`）共存。但它用的 `_outcome_failure_fields`（`:770-806`）是**第二套独立弱规则**，不调 `gap_state_for_failure`：八个典型 outcome 四个不一致，`capture_failed / delivery_failed / review_failed` 在旧 outcome 没自带 `gap_state` 时留 `None`。两批 `summary/batch_outcomes.json` 都是第二轮前生成的（只读未重出），两字段全 None、`requested_source_classes` 全 null：这条修复只有代码与单测证据 | 部分修复 + 补证据 |
| 166 行失败记账 | state 全 `evidence_missing_or_unsampled`、reason_code 全 `episode_planning_failed`，按段 40/40/43/43；阶段与五态在嵌套 `failed_episode` 对象里，行顶层没有 `failure_stage/gap_state` 键。四段全是规划期失败，符合 owner 第 4 条 | 通过（口径要写明） |
| provenance 指向生产树 | Codex 树引用 0 处，但三个键指向临时子树 `/data/jzy/tmp/wt-grok-pilot46-H2/examples/...`（`summary.json` 的 `scaleup_dry_run.path` 同，且引用的是第一轮干跑、缺口码还是旧名）。子树还在（`ec385e0`，`git worktree list` 12 条含 H2–H7），登记表内容与生产树不同。不是代码 bug：脚本默认 `REPOSITORY = parents[2]` 在生产树里本来就对，是 Grok 显式传了 `--repository .../wt-grok-pilot46-H2`（`H-2_merge_accounting.md:60`）。另外 `merged_episodes.json` 里 47 处 `command`（24 处 `episodes[*].command`、23 处 `review.audit.command`）仍指向 Codex 树，从原批记录原样继承，不是 H-2 引入 | 部分修复（R1 同类） |

### H-3 复现性（x2 复核，关键数字我抽核）

| 要求 | 实测 | 判定 |
|---|---|---|
| 不再吞 `os.environ` | `room_package.py:222-236`，`bindings = {}` 起步，只读 `runtime.path_bindings`（与 `runtime.mp3d_root`）；cwd=/tmp 下把 `AVENGINE_MISSING_PROBE` 塞进环境再展开仍抛 `ValueError: RoomPackage missing configured path roots: AVENGINE_MISSING_PROBE` | 通过 |
| 相对 `room_package` 相对 catalog 目录 | `room_package.py:258-283`；不给 `catalog_path` 抛 `ValueError: relative room_package path requires catalog_path`（`:275`）。**Grok 的 H-4 报告说"从 /tmp 直接调用仍会 FileNotFoundError"，在合并后 HEAD 上不成立，H-3 报告的说法才对** | 通过（H-4 报告一句要改） |
| `run_qa_episode.py` 请求绑定优先、传 catalog 路径、plan 两份快照 | `:28-39`（合并顺序第二轮前的内联代码就有，第二轮抽成函数加单测）、`:75-77`/`:101-102`/`:210-212`、`:160-163` 调 `write_room_package_plan_snapshot`（`room_package.py:286-306`） | 通过 |
| `build_qa_batch_manifest.py` 的"本工作树" | `:12` `REPOSITORY = Path(__file__).resolve().parents[2]`；`resolve_request_room_catalog`（`:55-77`）里 `--catalog` **最高优先级、原样采纳**（`:63-64`），`CODEX_CATALOG_MARKERS`（`:27`）只拦 `wt-multi-home-activity-integration` 一个串；`prepare` 没有 `--catalog`，`scaleup-dry-run` 有 | 通过，但 `--catalog` 是没有护栏的后门 |
| producer 记 `argv` 与 `AVENGINE_*` | `run_qa_batch.py:213-214、217-242、696-698、1106`；单测 `tests/unit/test_qa_batch_runner.py:399-408、410-430` | 通过 |
| 单测"换 cwd 不变"、"缺绑定列变量名" | `test_room_package_reproducibility.py` 7 条（含放诱饵 `room.json` 再 chdir 的那条、用真生产 catalog 的那条）、`test_build_qa_batch_manifest.py` 5 条；12 passed | 通过 |
| 残留口子 | `room_package.py:270`、`:219` 仍调 `os.path.expandvars`：设 `X2_LEAK_ROOT` 后 `resolve_catalog_room_package_path("${X2_LEAK_ROOT}/room_a.json", catalog_path=None)` 返回真实绝对路径、绕过 `catalog_path is None` 的 ValueError；生产 catalog 7 条 `room_package` 都不带 `${`，现在没有真实暴露。缺绑定报错只列**第一个**出问题字符串里的变量名（`:249-252`），单测恰好用"一个字符串两个变量"测不到 | 修代码（低） |

### H-4 校验顺序与闸门（x3 复核，关键数字我抽核）

| 要求 | 实测 | 判定 |
|---|---|---|
| 先展开再校验 | `room_package.py:332-335` `validate_room_package(resolve_room_package_paths(...), relative_roots=...)`；`missing_filesystem_paths`（`:151-193`）三档：绝对路径永远查、`tmp/ examples/ …` 仓库相对前缀拼 `REPOSITORY_ROOT`（`:18`，由 `__file__` 推出）永远查、其余相对路径在传 `relative_roots` 时查（生产路径 `:328` 总是传）。四个反例走生产函数：`${AVENGINE_X}/does_not_exist.json`、`tmp/does_not_exist.json`、`some_dir/does_not_exist.json` 全抛 ValueError（模板展开成真实路径再报缺失），对照组不报；cwd=仓库根与 /tmp 结果相同 | 通过（R4 已修） |
| 七包 0 缺失 | catalog `path_bindings` 对 7 条全部装载，两种 cwd 都 `all OK: True | total missing: 0` | 通过 |
| 闸门不再吞 ImportError | 第一轮的 `try/except ImportError: pass` 换成 `apply_review_exposure_gate`（`batch_delivery.py:249-272`，调用点 `:385`）：导入失败 `status=review_failed`、`exposure_gate.status=unavailable`、`frame_source.kind=unavailable`；正常路径把 `frame_source` 抄到 review 顶层（`:269-271`）。单测 `test_qa_batch_delivery.py:116-129` 通过 `_import_apply_exposure_gate` 间接层 monkeypatch，生产代码走同一函数，分支真被触发 | 通过（R5 已修） |
| 帧来源优先级、`frame_source.kind`、阈值 | `exposure_gate.py:152-201` `capture/frames` PNG → `capture/rgb.npy`（含 `episode/capture/rgb.npy`）第 0/中/末 → `batch_review/frames`；无帧 fail-closed（`:229-244`）；阈值 235 / 0.20 未动（`:19-20`）。对三棵产物树 92 个 episode root 只读调用 `_select_gate_frames`：UE 66 段 `capture/frames`，Habitat 18 段 `rgb.npy`，**0 段**退到 `batch_review/frames`（第一轮 10 段用审阅抽帧冒充），2 段 `None` 是根本没渲染的规划用尽格 | 通过 |
| 硬编码曝光表删除 | `grep -rn AUTHORED_USD_EXPOSURE_BIAS_EV src tools` 零命中，只剩 `tests/unit/test_authored_exposure_bias_source.py:27-28` 断言其不存在；补偿改从 `run_spear_residential_episode.py:78-105` 五级读取（CLI → `visual_plan.camera` → `episode.resources` → `room_package` → `planning_inputs`），取不到 `not_requested` 无兜底；上游 `qa_episode.py:811-814、883` 供得上值；包字段 A/C −3、B −4 顶层与 `planning_inputs` 两处都对 | 通过（K3 实体已修） |
| 地板四份文件 | 副本 `status=measurement_kind=depth_readback_fallback`、`summary.hit_count 0`、`summary.depth_readback_hit_count 64`、`method.line_trace.hit_count 0`；四包指向仓库内副本；与原件 `tmp/gb_floor_reference_20260907/` 字段级 diff：45 个叶子只有 6 个变（status、hit_count、hit_fraction、两个新计数、relabel_note），26 个测量数值逐一相同；`floor_trace_rows.json` 四房 sha256 与原件完全相同 | 通过（诚实改标签） |
| 仓库相对地板路径的 cwd 风险 | 校验器拼 `REPOSITORY_ROOT` 与 cwd 无关，但真正读文件的 `src/avengine/capture/qa_plan_adapters.py:63-75` `_floor_value` 把相对字符串直接交给 `_read`：cwd=/tmp 时七个房间**全部 FileNotFoundError**，同一 cwd 下 kujiale 包另有 14 条仓库相对路径打不开。改动前就是 `tmp/gb_…` 相对路径，不是本轮引入；搬进 `examples/` 反而提高了可复现性 | 修代码（中，非本轮回归） |

### H-5 外观与像素字段（x3 复核，关键数字我抽核）

| 要求 | 实测 | 判定 |
|---|---|---|
| 两处接线 | `qa_delivery.py:1111-1116`（读 `pixel_visibility_truth.json` 后 `annotate_pixel_visibility_semantics`，`git show 789bc64:` 该文件无此调用）；`batch_delivery.py:37-49` `attach_visibility_semantics`，`:319` 调用在 `:321` 写 `achieved_conditions.json` 之前；batch 侧丢弃注解后的真值（不回写 capture）是有意的 | 通过 |
| **真产物** | 28 个 attempt_03 里 27 个有 `batch_review/achieved_conditions.json`，含 `visible_pixel_frames`/`bbox_touches_frame_edge_frames`/`in_fov_definition` 的 **0/27**；facts.json 里的 4 处命中都在早已存在的 `appearance_review.*.pixel_visibility_semantics` 块。时间线：attempt_03 落盘 22:01:44–22:33:46，H-5 提交 a219ae4 是 22:21:28，合并 acfa744 是 22:33:25——产物早于代码。用 HEAD 的 `attach_visibility_semantics` 对同一输入现算，字段全部产出（apartment_animal_animal `source1/event_001 in_fov=11 visible_pixel_frames=11 edge=8`）。**只读边界零违规**：28 个符号链接背后的真值文件 mtime 全在 12:19–19:56，都没有 `visibility_semantics_authority`。Grok 的 H-5 报告自己写的是"新字段出现在下一次 finalize"，没有虚报；"attempt_03 在合并后 HEAD 收口"是我给子代理的前提，不成立 | 代码通过，产物证据缺失 |
| 分类器缺口入表 | `qa_evidence.py:656-667` actor 级 `classifier_gap_fields`（reason `registered_appearance_value_classifier_not_implemented`，`gap_category=interface_not_implemented`）；`batch_coverage.py:539-588、1434-1442` "查到才抬"为 `interface_not_implemented`。产物侧 actor 级 reason 三棵树 170 个 actor 记录 **0 个**（同样早于代码），check 级 reason 早已有。产物里实际触发过的无分类器色值 11 个（设备 7：white_satin、light_gray、matte_black、light_gray_fabric、warm_gray、sandstone、silver；毛色 4：standard_sable、standard_seal_point、dark_sable、standard_white_tan）；与 Grok 清单差 `beige`（产物未触发）与 `matte_black`（Grok 未列）两处，`standard_white_tan` 中途补了 `white_tan_*` 分类器 | 代码通过，产物证据缺失，清单差两处（证据不足） |
| 两台电视登记 | `examples/runtime/source_asset_runtime_profiles.json` 第二轮只改两行（两台电视 `realized_attributes.body_color: black`）。x3 手工解析 GLB 容器读 `Watertight_BaseColor` 贴图：central_pedestal 去纯黑 UV 空白后中位 **[51, 53, 54]**、two_splayed_feet **[25, 25, 27]**，`metallicFactor=0`，与 Grok 报告一字不差 | 通过（texel 已复核） |
| K4 两只狗 | `background/apartment_animal_animal/attempt_01` 像素真值：source1 黄拉布拉多第 0 帧 5 像素（target 3455，visible_occluded）、第 39 帧 7607；source2 杰克罗素第 0 帧 2970、第 39 帧 9524；Grok 引的 `warm_brown 0.119` 实测 0.11896（阈值 0.12），第 32–38 帧连续擦边 | 通过（K4 已修） |

### H-6 就绪报告口径（我核）

| 要求 | 实测 | 判定 |
|---|---|---|
| 结论只写事实，不以听感/测试变绿放行 | `READINESS_20260907_round2.md` §1 结论句列的是闸门、契约、owner 原话、仍空项，末句明写"不以听感、不以测试变绿作为放行条件" | 通过 |
| owner 裁定按原话 | §2.1–2.4 抄的是原句，"不要求每个房间跑出每种题"绑先导四格，"train39/eval0 接受"未扩写 | 通过 |
| 单测计数附清单 | §4 附 23 文件表与 pytest 摘要，合入后 16 文件 130 passed 也列了命令。我这边同 16 个文件收集数之和恰为 130，41 文件全集 581 | 通过 |
| 试听清单如实 | §5 写明 owner 本机包是五段 human_human、与两份 pending JSON 都不同、heard/reviewer 不代填。owner 后来又听了六段 attempt_03，报告写于其前，下一版应如实登记 | 通过（需更新） |
| 更正 G-B 假冲突与"未接入" | §6.1–6.2 与 H-4 报告 §6 都更正了 | 通过（K3 已修） |
| **数字与陈述是否与产物一致** | 四处不一致：`:214` "覆盖表 1260 = 9×7 房×20 题"与 `:252` 的 976/1260 是过期数（应 871/1365 = 1260 挂装 + 105 外观缺口）；`:23`/`:73` "catalog 为生产树绝对路径并带 path_bindings"不实（350 条指临时子树 H7）；"56/56 条 RIR 作业"只数了每段前两条（实为 200 条）；结果侧修复后第一轮 20 格的交叉表闸门其实不过（`min_distinct 1 < 3`，旧存档的"过"是空字符串类别对造成的假绿），报告没提 | 口径需改 |

### H-7 放量干跑与结果侧统计（x2 复核，关键数字我抽核）

| 要求 | 实测 | 判定 |
|---|---|---|
| 350 个 request 有 `sound_pool` | 350/350 同一路径 `.../p10_pilot46_manifest_20260907_v3/batch_sounds.json`，逐条存在；`prepared_set` 0 条残留（`build_qa_batch_manifest.py:214`） | 通过（R8 已修） |
| `distance_range_m` 三处一致 | requested_profile / request.profile / scaleup_config 槽位三处各 `{'[1.5, 6.0]': 350}`，互不一致 0 行；修在 `batch_manifest.py:75-80、254-256、261、529` | 通过 |
| repeat 缺额、缺口码 | `repeat_deficit_count = 0`；缺口 `{'no_compatible_sounds': 2}`（029 apartment、241 kujiale，都是便携空调，附 `compatible_sound_count: 0`；`batch_manifest.py:599-630` 与身份用尽码分开） | 通过（K7 已修） |
| 交叉表 | 六个类别对逐格与存表相等，`min_distinct 4 ≥ 3`，`meets_acceptance True` | 通过 |
| 画外 | 14 条 = 锚点 7 + 竞争者 7，只在 human-human / animal-human，七个房间都落在槽位 [0, 1]，设备画外 0，`anchor_count=2` 且锚点画外 2 段（051、101） | 事实一致；边界 owner 已裁定不挡 |
| **来源路径** | 350 条 `room_catalog`、`source_registry`、`controller_entrypoint` **全部指向 `/data/jzy/tmp/wt-grok-pilot46-H7/`**；`producer = {code_state: "working_tree", git_commit: 789bc64, working_tree_changes: [batch_manifest.py, test_qa_batch_manifest.py, build_qa_batch_manifest.py]}`；6 段 plan-only 同病。那棵树还在（现 HEAD 57c0ac0，干净），当初脏代码的确切状态恢复不出来；树一删 `run_qa_episode.py:75` 读 `room_catalog` 即失败。两棵树 `catalog.json` 逐字节相同，H7 的 registry 与 `git show 789bc64:examples/runtime/source_asset_runtime_profiles.json` 逐字节相同——输入可从 git 取回，是可修复的断链 | **需修复（R1 同类复发）** |
| **复现性与隐藏输入** | 生产树 catalog+registry 在 HEAD 干净树复现：`summary`（归一化路径后）、`scaleup_config` 逐槽、交叉表、缺口数、画外数、配额全同，但 **343/350 条 `source_assignments` 不同**（`sound_selection` 340、`source_asset_ids` 315），两条空调缺口换到 025/273；换成 H7 那份 registry：**350 条逐格重放**（只差 `request_path` 与 `producer`）。差异来源只有 HEAD registry 多的两行 `body_color: black`（a219ae4）。代码确定；但 `producer` 不记 `source_registry`、不记输入哈希（`build_qa_batch_manifest.py:147` 只加了 `room_catalog`），踩错 registry 时聚合数字全对、逐格资产全变 | 需修复（记输入与 sha256） |
| `--plan-only` 可执行 | Grok 6 段 returncode 全 0，但在 789bc64 跑，`plan/` 里**没有 `path_bindings.json`**（写它的改动在合并点 b0d6b78 才进来），证不了 H-3 的快照要求；x2 在 HEAD 对复现 request 补跑 2 段（001 画外、003 in_fov）：rc 0，`plan/path_bindings.json`（`catalog_path` + 15 绑定）、展开后的 `plan/room_package.json`、`condition_profile.distance_range_m = [1.5, 6.0]`；产物 `/data/jzy/tmp/claude_audit_round2_20260907/x2/planonly/` | 通过（引用证据要换） |
| `collect_batch_outcomes` 结果侧 | `batch_manifest.py:837-841、850` `requested_source_classes`；`_profile_for_match`（`:83-97`）给两侧补 `COMMON_PROFILE` 默认（`:46-63`），观测值不等于默认仍判失配。只读重算第一轮 20 格：True 19 / None 1、六种类别对、配额 delivered 19 / unmet 1、20/20 有 `requested_source_classes`；失配根因是请求侧少三个键、共有键零差异 | 通过（R9 已修） |
| **新事实** | 类别对修对后，第一轮 20 格交叉表闸门从存档 `meets_acceptance: true / min_distinct 5` 变成 **`False / min_distinct 1`（要求 3）**：六个类别对各只覆盖 1 个条件组。旧"过"是把 20 行塞进空字符串类别对的假绿；350 格放量是过的（4） | 补证据（就绪报告要改口） |
| 便携空调表述 | `batch_sounds.json`：sounds 495 / rejected 1175；`air_conditioning` 被拒 20 条、拒因 100% `registered_event_exceeds_explicit_clip_budget`、`source_spec.max_clip_s = 5`；进池 0；三台空调兼容声音各 0。被拒条目不存时长，x2 未独立量秒数 | 通过（K7 已修） |
| 46 格两条 repeat 缺额 | 仍在 p10 v3 清单：authored_a / kujiale device_device，13.0 s 对 13.408 / 13.420 s；清单未重生成，属应然 | 如实记录（K8） |

## 3. 问题清单（按严重度）

### 3.1 需修产物 / 需重出（代码不用改，产物比代码旧）

- **S1（高）合并表在合并后 HEAD 重出。** 换新目录名（脚本对已存在目录会拒绝，不会覆盖）：
  ```bash
  ssh 48g-jump 'cd /data/jzy/tmp/wt-grok-pilot46-round2 && PYTHONPATH=src:tmp/native_python_addons_v1 PYTHONDONTWRITEBYTECODE=1 /data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python tools/dataset/merge_qa_batch_attempts.py --original tmp/qa_pilot46_background_20260907_v1 --rerun tmp/qa_pilot46_rerun_20260907_v1 --output tmp/qa_pilot46_merged_round2_20260907_v2 --repository /data/jzy/tmp/wt-grok-pilot46-round2 --manifest tmp/p10_pilot46_manifest_20260907_v3/batch_manifest.json'
  ```
  验收：`coverage_state_counts.json` deferred 871 / interface 1365，`coverage.json` 里 `registered_appearance_value_classifier_not_implemented` 105 行，`exposure_gate_offline/*.json` 带 `frame_source`，provenance 指向生产树。我这边 `/data/jzy/tmp/claude_audit_round2_20260907/x1/merged_repro/` 已有一份可对拍。
- **S2（高）放量干跑在生产树重出，并把 `source_registry` 与三份输入（catalog、registry、sound pool）的 sha256 记进 `producer`。** 现在的 350 条钉在临时子树 H7 的脏树上。同时给 `--catalog` 加护栏（`build_qa_batch_manifest.py:63-64`：不得指向 `/data/jzy/tmp/wt-*` 临时树，或在 `producer` 标记"用了非生产 catalog"）。x2 的 `dryrun_repro/` 就是生产树复现，可直接对拍。
- **S3（中）attempt_03 的收口至少抽 2–3 段（含一段 Habitat）在 HEAD 上重跑 `finalize_batch_episode`/`finalize_qa_episode`，写新目录**，证明 H-4 的 `frame_source` 与 H-5 的 `visible_pixel_frames`、actor 级分类器 reason 在真产物上出现。音频与 RIR 不用重算（缓存同阶数会命中）。若 owner 决定专题 A 的全批常数，则 42 段音频重渲时顺带全部重新收口，一次解决。
- **S4（中）`summary/batch_outcomes.json` 用现有 outcome 记录在新目录重出一份**，证明 `failure_stage/gap_state/requested_source_classes` 真的会被填；`background` 那 4 行 `delivery_failed` 正好落在 S6 会留 `None` 的分支，是好样本。

### 3.2 需修代码

- **S5（中）`_outcome_failure_fields`（`batch_manifest.py:770-806`）改为复用 `gap_state_for_failure`**（或把分类器提到 `src/avengine/qa/` 共用），给 capture / delivery / review 三种失败各补一条单测；再加一条钉住"阶段未知且异常未知 → evidence_missing_or_unsampled"的默认，并给 `audio`/`launch` 无条件记 interface 写一句理由（或改成看异常类型）。白名单外的异常名现在会被吞进"证据缺失"，两个修法（反向白名单、或加可见的"未分类"出口）由 owner 定。`_looks_like_interface_defect`（`run_qa_batch.py:510-512`）是死代码，可删。
- **S6（中）校验器与消费者对相对路径口径不一致。** `missing_filesystem_paths` 拼 `REPOSITORY_ROOT`，`qa_plan_adapters._floor_value`（`src/avengine/capture/qa_plan_adapters.py:63-75`）按 cwd 读；cwd=/tmp 时校验 0 缺失、七包地板文件全部打不开。要么在 `resolve_room_package_paths` 里把仓库相对路径 rebase 成绝对，要么让消费者用同一个根。非本轮回归，但"校验绿、运行炸"值得堵。
- **S7（低）`room_package.py:219、270` 的 `os.path.expandvars` 仍是环境变量入口**（`${VAR}` 形式的 `room_package` 值会被 shell 展开并绕过 `catalog_path` 检查）；缺绑定报错只列第一个出问题字符串里的变量名（`:249-252`）。
- **S8（低）H-1 两处小修**：`render()` 第 2201 行的 depth 覆盖只在等于默认值时生效（显式 `--indirect-depth 200` 会被 overlay 文件盖掉）；"请求或房间包可覆盖"里"房间包"没有实现，要么接上要么改报告。
- **S9（owner 拍板后修）专题 A**：若选全批常数，把它做成请求/plan 里渲染前声明的字段，回执 `gain_application` 照录，然后 UE 28 + Habitat 14 段重渲音频并重新收口。见第 5 节。

### 3.3 需补证据 / 口径更正（改报告与文档）

- **K1** `READINESS_20260907_round2.md:214、252` 与 `H-2_merge_accounting.md` 的五态数字：1260 不全是挂装，应写 interface 1365 = 1260 挂装 + 105 外观分类器缺口，deferred 871。
- **K2** `READINESS:23、73` "catalog 为生产树绝对路径"应改为"指向临时子树 H7，需重出"（S2 之后再改回）。
- **K3** H-1 报告"56/56 条作业"改为全部作业数（200 条，全部 <0.9，最大 0.678）；"侧向线索 24/28"要注明是自己脚本的 30 ms 窗口径，并列任务书口径 20/28、ILD 单独 10/28、审计器 44/54。
- **K4** H-4 报告 §3.1 "从 /tmp 直接 `package_from_catalog_entry` 仍会 FileNotFoundError"在 HEAD 上不成立（是 ValueError），删或改。
- **K5** 就绪报告补一句：结果侧统计修对之后，第一轮 20 格的交叉表闸门实为不过（`min_distinct 1`），旧存档的"过"是空字符串类别对造成的假绿；放量 350 格是过的。
- **K6** 合并覆盖行的失败阶段与五态在嵌套 `failed_episode` 对象里、行顶层没有这两个键，文档要写明；`_profile_for_match` 给旧清单补默认值是一个假设，写进文档。
- **K7** 就绪报告 §5 试听：登记 owner 本轮听的六段 attempt_03 与原话，仍不代填 heard/reviewer。
- **K8** 无分类器色值清单：Grok 列 `beige`、产物未触发；产物触发 `matte_black`、Grok 未列；`standard_white_tan` 中途补了分类器。要对齐需把登记表注册色值集合与已实现分类器分支做一次差集（证据不足，不下"报错"结论）。
- **K9** 我自己的两处：第二轮任务书 H-1 写"32 段"应为 28；"混音峰值仍在 −6 到 −25 dBFS"是我按第一轮成片范围写的描述性带宽，不是闸门，第一轮 42 段里本就有 7 段在带外（见第 6 节）。
- **K10** Habitat `all_still` 段的 240 帧 rgb.npy 逐字节相同（12 段全是；两段 `speaker_moving` 有运动）：这是刚体资产不动、相机不动的设计结果，不是闸门缺陷，但曝光闸门"首中末"在这类段上等价于一帧；owner 应知道 Habitat 静止段就是一张静止画面配 16 秒音频。

### 3.4 owner 已裁定 / 待定

- 已裁定（照做，不再当阻塞）：先导四格空格；挂装留分母；空调两条干跑缺额接受；train39/eval0 接受；试听 JSON 不代填；正式准入本轮不做；双耳阶数修（已做）；便携空调处置、画外锚点入画、设备画外槽三条"无所谓，不挡后续"。
- 待定（本轮新出现）：专题 A 的处置（fail-closed 保持 vs 全批常数），见第 5 节；是否让 `merged_episodes.json` 里 47 处继承自原批的 Codex 树 `command` 路径在合并时改写（x1 问题 8）；分类器默认落点的修法（S5 两选一）；H-1 重渲后是否再听一包（owner 已自行听了六段）。

## 4. H-1 双耳混响阶数：我自己的复算

复算脚本都在 `/data/jzy/tmp/claude_audit_round2_20260907/`（`h1_verify.py`、`h1_more.py`、`clip_probe.py`、`rir_energy_compare.py`、`run_audit_a03.py`），输出在同目录 `mine/` 下。全部只读，没有改任何产物。

### 4.1 代码

- `tools/acoustics/render_frame_readback_sequential_speech.py`（提交 703e4db）：`_simulation()` 的四个参数改成可传入，默认 `DEFAULT_DIRECT_SH_ORDER = 3`、`DEFAULT_INDIRECT_SH_ORDER = 1`、`DEFAULT_INDIRECT_RAY_DEPTH = 200`、`DEFAULT_MAX_IR_SECONDS = 4.0`，与 Habitat 路径 `examples/runtime/rir_cache_simulation_request_v2.json` 一致。CLI 加了 `--direct-sh-order / --indirect-sh-order / --max-ir-seconds / --simulation-request`；`render()` 先用显式参数、再用 `--simulation-request` 只取四个键的 overlay。
- 旧缓存拒绝复用：`existing_rir_cache_simulation_matches()` 要求缓存 `request.json` 的 `simulation.effective` 与本次 `to_dict()` 完全相等，否则 `_existing_rir_cache_sequence()` 抛 ValueError（"refusing to reuse a different-order cache"）。这条检查在生产路径上：`_render_plan_audio()` → `_dynamic_rir_sequence()` → 两源且 `request.json` 存在时走 `_existing_rir_cache_sequence()`。
- 缓存身份：`src/avengine/acoustics/rir_cache.py` 的 `_request_identity_payload()` 把 `simulation.effective` 纳入 `request_identity_sha256`。实测同一段（`authored_a_human_human`）attempt_02 与 attempt_03 的 plan sha256、acoustic package sha256 相同，identity 却不同（`3f7e0d42…` 对 `49162fed…`），说明阶数确实进了身份。Grok 说 job plan 的 `cache_key_fields` 仍是位姿三元组、由 `rir_cache.py` 校验器钉死，属实（`render_frame_readback_sequential_speech.py:1018-1022`，`rir_cache.py:240/410`）。
- `src/avengine/rooms/qa_delivery.py::build_audio_command` 从 `request / runtime / plan` 的 `simulation` 块透传四个键，也透传 `simulation_request`。**房间包不在来源里**。
- 小瑕疵：`render()` 第 2201 行只在 `indirect_ray_depth == DEFAULT_INDIRECT_RAY_DEPTH` 时才让 overlay 的 depth 生效。不影响本轮产物（本轮没用 overlay 文件），但语义不对。
- 新单测 `tests/unit/test_frame_readback_binaural_sh_order.py` 5 条都在我复跑的 581 条里。

### 4.2 对照实验（A 房 human_human，`tmp/h1_sh_ablation_20260907/`）

我读了 `ablation_metrics.json` 的原始数字：基线 0/0/64/0.25 两条 RIR 首达 2 ms 后 L/R 相关 0.9958 / 0.9966；3/1 各组降到 0.35–0.45。存储 IR 长度只随 `indirect_ray_depth` 变（64→0.66 s，200→1.73–1.78 s），`max_ir_seconds` 0.25→4.0 不改变长度。Grok 表里的数字与文件一致。

### 4.3 重渲的 28 段（`tmp/qa_pilot46_audio_v2_20260907/episodes/<id>/attempt_03/`）

| 项 | 我的实测 | Grok 报告 | 判定 |
|---|---|---|---|
| 段数 | 28 段 UE 家族 delivered（apartment 7、authored 15、kujiale 6），attempt_03 的 `capture/`、`plan/`、`request.json` 等是指向最终 attempt 的符号链接；Habitat 段没有 attempt_03 | 28（任务书写 32） | 通过。任务书的"32"是我算错 |
| 原批与第一轮目录 | 三个只读目录 21:30 后 0 个文件被改 | 未改 | 通过 |
| 仿真参数 | 28 个缓存 `simulation.effective` 全是 3/1/200/4.0 双耳；27 份回执 `qa.propagation.simulation` 逐键相等；HRTF 均为 `mit_kemar_normal_pinna_16k.sofa`；旧 attempt 全是 0/0/64/0.25 | 同 | 通过 |
| RIR 尾部 L/R 相关 | **200 条作业**全部 <0.9：最大 0.678，家族中位 apartment 0.446 / authored 0.529 / kujiale 0.436 | "56/56" | 通过；计数口径要改 |
| 存储 IR 长度 | 1.45–2.17 s（旧 0.56–0.83 s） | 1.73/1.78 s（A 房） | 通过 |
| 混音峰值 | 27 段：最低 −20.71、最高 −1.63、中位 −12.89 dBFS；比重渲前中位 **+3.50 dB**、均值 +3.30、最大 +7.37（kujiale_human_device）、最小 −0.02；高于 −6 的 3 段 | 24/28 在窗内 | 事实一致；"窗"见第 6 节 |
| 削波 | `authored_b_device_device` 渲染器拒写，peak=1.1388 | 同 | 分析见第 5 节 |
| 侧向事件线索（任务书口径） | 28 个侧向事件：重渲前 14 → **20**（apartment 5→7，authored 7→9，kujiale 2→4）。只看 ILD≥2 dB：12 → **10**。审计器 `delivered_cue_state` 全部 54 个事件：pass 47 / fail 7 → pass 44 / fail 9 / unmeasured 1 | "24/28" | **口径不同**。Grok 用自己脚本 stem 起点后 30 ms 窗的 ITD/ILD；两种口径都不是人工校准过的判据 |
| RIR 直达段（首达 5 ms）耳间差 | 直达 ITD 与 KEMAR SOFA 同方位之比中位 0.92；直达 ILD 符号与几何一致 24/28（不一致的 4 条里 3 条 |ILD|<1.6 dB，另一条是动物段首帧作业与事件起点位姿不同）。方位约定：审计器 `listener_azimuth_deg` 右为正 | "ILD 符号与几何一致" | 通过 |
| RIR 能量拆分（56 条首帧作业，3/1 对 0/0） | 直达 0–2 ms 能量差中位 **+0.03 dB**；早期反射 2–50 ms **+4.5 dB**；晚期 >50 ms **+7.5 dB**（56 条全部落在 +6.6～+8.3 dB，与房间无关）；总能量 +4.2 dB；DRR 中位从 −4.5 dB 变成 −9.7 dB | 未报 | 补证据：晚期整齐抬高 7.5 dB 是解码归一化随阶数变化，不是房间几何；0 阶和 3/1 阶哪个绝对电平对，没有物理参照 |
| 题数 | 27 段 260 条，重渲前 268 条（apartment 59→61，authored 153→145，kujiale 56→54） | 同 | 通过 |
| 曝光闸门 / review | 27 段 review delivered、exposure_gate pass；帧没动 | 同 | 通过（但收口用的是只含 H-1 的树，见 H-5） |
| 审计器抽查 | 未改过的审计器对 27 段 attempt_03 重跑，`events` 与 Grok 的 `batch_review/audit_v2.json` 27/27 逐字节相同；几何态 26 fail / 28 pass 与重渲前相同 | — | 通过 |

### 4.4 我对 H-1 的判定

阶数修复本身成立：混响不再是两耳相同，直达段的耳间差保住了，参数进了回执和缓存身份，旧缓存不会被误用。但是**"stem 上有可用线索"的比例并没有像 Grok 报告那样明显变好**：按任务书口径 14→20，按 ILD 单独看 12→10，按审计器自己的判据 47→44。原因是 3/1 阶给了一个能量更大且两耳不相关的尾巴，长窗 ILD 被稀释，而审计器和我的 `cue_aggregate.py` 都是长窗/起点法。这是我第一轮就欠的 C3（判据改直达窗 + 尾部相关）；在人工校准包之前，谁的"可用线索比例"都不能拿来当放行数字。

## 5. 专题 A：`authored_b_device_device` 的削波

### 5.1 事实

1. 失败点：`src/avengine/timeline/current_mp3d_dynamic_audio.py:1774-1781` 的 `_write()`，peak>1 就抛 `CurrentMP3DDynamicAudioError("audio output would clip without normalization/limiting: peak=1.13879248")`。写出顺序是 dry → 每个 stem → mixture（`:1802-1810`）。attempt_03 目录里只有 `delivery/audio/audio/dry/`，`binaural/` 目录没建，说明**第一份双耳 stem 就没过**。
2. 我用该段写出的干轨（`dry/source1_mouth.wav`、`dry/source2_mouth.wav`，峰值都是 −3.00 dBFS）和两版 RIR 缓存离线卷积复算（`clip_probe.py`）：
   - 3/1 阶 RIR 卷 source1（音响放音乐，距 3.74 m，方位右 20°）：左 1.0732、右 **1.1388**，与渲染器报的峰值一致；卷 source2（滴水，4.44 m）：−16.3 dBFS。程序是 `sequential_sources`，三个事件不重叠，混音峰值等于 source1 stem 峰值。**削波在 stem，不是叠加。**
   - 只保留 RIR 前 2 ms（直达）：−14.2 / −15.6 dBFS；前 10 ms：−11.1 / −7.5；前 50 ms：−3.8 / −1.4；前 250 ms：+0.36 / +0.48；整条：+0.61 / +1.13。0 阶 RIR 同样拆：前 2 ms −12.4 / −13.2，前 50 ms −6.5 / −5.6，整条 −6.5 / −5.4。**峰值是反射堆出来的，直达反而比 0 阶低 2 dB。**
   - 这条 RIR 总能量比 0 阶高 6.1 dB，直达占比 −16.3 dB（0 阶 −8.1 dB）。全部 56 条首帧作业：晚期能量整齐 +7.5 dB。
3. 原片 −5.40 dBFS 本来就在任务书"−6 到 −25"之外（第 6 节）；它在第一轮已被记为 delivered。
4. 干轨电平来自声音库：`sound_event_library_v1_20260903` 的 event.wav 我随机抽 40 条，33 条峰值恰好 −3.00 dBFS（音乐与滴水都是），登记表 `linear_gain 1.0`、`normalization_applied false`。人声干轨不归一，本批最热 −0.77 dBFS（authored_b_human_human source1）。
5. 房间增益（湿 stem 峰值减干轨峰值）：0 阶 54 条 stem 最大 −1.68 dB；3/1 阶 52 条最大 +2.69 dB（authored_a_animal_device 的猫），削波这条 +4.13 dB。只要干轨 −3 dBFS 而房间增益超过 +3 dB 就会削波；−0.77 dBFS 的人声干轨配 +2 dB 房间增益也会。这是系统性的头顶空间问题，不是这一段特殊。
6. Habitat 14 段（早就是 3/1）混音峰值 −13.04～**−1.57** dBFS，4 段高于 −6，没削波是运气不是设计。
7. 增益是标量、两耳同乘，对 ITD/ILD 没有任何影响；fail-closed 也不影响线索。契约禁止的是按输出电平自动限幅或归一化；`linear_gain`（`gain_application.applied_at = dry_audio_assembly`，`application_count 1`）和 `post_assembly_convolution_gain`（现在恒为 1.0）都是回执里已有的、渲染前声明的字段。

### 5.2 两条路的后果（用本轮数字算）

| 方案 | 这一段 | 其余 27 段 | 与 Habitat 的关系 | 契约 |
|---|---|---|---|---|
| 保持 fail-closed，不加增益 | 无成片；B 房 device_device 格子从 delivered 退成空格（或退回 0 阶 attempt_02，那就是一批里混两种阶数） | 不动 | 各自原样 | 完全符合 |
| 只给这一段声明一个增益 | 能写出 | 不动 | 这一段比同批安静 | 形式上"预先声明"，实质是看了输出峰值再定的，等于单段归一化，且电平离群。**不建议** |
| 全批一个常数（例如 `post_assembly_convolution_gain = 0.5`，−6.02 dB），渲染前写进请求，回执照录 | −4.87 dBFS | 最高 −7.65、中位 −18.9、最低 −26.7 | 只重渲 UE 28 段：UE 整体比 Habitat 低 6 dB，家族可被电平区分；Habitat 14 段也重渲：−19.1～−7.6，全批一致 | 与输出无关、全批同值、写在回执，属于校准常数而不是归一化 |

### 5.3 我的建议（owner 拍板）

保持渲染器 fail-closed 不动；如果要保住这个格子并为放量兜底，加**一个全批常数**而不是单段增益，常数要覆盖已观测的最坏房间增益（+4.13 dB）加最热干轨（−0.77 dBFS），0.5（−6 dB）给出 2.6 dB 余量；常数写进请求/plan 并由回执 `gain_application` 照录，然后 UE 28 段和 Habitat 14 段一起重渲音频并重新收口（都是 CPU，UE 28 段本轮用了 33 分钟），以免家族间出现 6 dB 台阶。若 owner 选保持 fail-closed 且不加常数，放量时要预期同类拒写按比例出现（本轮 1/28），空格按五态记 `evidence_missing_or_unsampled`，不是接口未实现。另外单列一条待查：3/1 阶晚期能量整齐抬高 7.5 dB 是解码归一化差异，绝对电平哪个对需要有参照的校准，不是本轮能定的。

## 6. 专题 B：−6～−25 dBFS 这条窗，和三段高于 −6 的成片

### 6.1 这条窗是什么

- 代码里没有：`src/avengine/qa`、`src/avengine/rooms`、`src/avengine/timeline/current_mp3d_dynamic_audio.py`、`tools/qa`、`tools/dataset` 都没有 −6 或 −25 的峰值门；渲染器只有 peak>1 拒写这一条硬闸。
- 文档里只出现在我写的第二轮任务书 `docs/roadmap/GROK_FIX_TASKS_ROUND2_20260907.md:24`。第一轮审核报告没有这句。**它是我按第一轮成片大致范围写的描述性验收，不是闸门，也不是 placeholder。**
- 第一轮已交付的 42 段本来就不都在窗内：UE 28 段有 3 段高于 −6（authored_a_animal_device −6.00、kujiale_animal_device −5.97、authored_b_device_device −5.40），Habitat 14 段有 4 段（hm3d_animal_animal −1.57、hm3d_animal_device −3.34、hm3d_human_device −3.33、mp3d_human_animal −4.91）。第一轮闸门和 owner 都接受了它们。

### 6.2 三段的实测

| 段 | 热的 stem | 距离 | 干轨峰值 | 重渲后 stem 峰值 | 房间增益 | 重渲前混音峰值 → 重渲后 |
|---|---|---|---|---|---|---|
| authored_a_animal_device | source1 猫叫 | 4.34 m | −4.32 | −1.63 | +2.69 dB | −6.00 → −1.63（+4.37） |
| kujiale_animal_device | source1 猫叫 / source2 微波炉提示音 | 2.43 / 4.02 m | −3.00 / −3.00 | −2.21 / −2.91 | +0.79 / +0.09 dB | −5.97 → −2.21（+3.76） |
| authored_a_human_device | source2 搅拌机 | 4.04 m | −3.00 | −5.20 | −2.20 dB | −8.70 → −5.20（+3.50） |

三段都没有任何样本超过 1.0，文件是 float32 WAV，没有限幅、没有失真；它们的"问题"只是超出了我写的那条描述性带宽。owner 听了六段 attempt_03（含其中三段）说"其实听起来都没啥问题"，与技术结论一致，但那是听感，不是准入。

### 6.3 建议

- 不要把 −6～−25 变成闸门。没有任何证据说峰值在这个带宽外会影响可答性；Habitat 段一直就在带宽外。就绪报告应把它写成"峰值分布事实"，不写成验收项。要改的是我的任务书措辞，不是 Grok 的代码。
- 三段是否重渲只取决于专题 A 的决定：若加全批常数，它们随 28 段一起重渲并落到 −7.6 / −8.2 / −11.2；若保持 fail-closed 不加常数，它们没有重渲的理由。
- 若 owner 仍要把 −6 当上限守住，动的应当是全批同一个卷积侧常数（`post_assembly_convolution_gain`），不是按资产改干增益：干增益按事件/资产走登记表，会改变声源间的相对响度；卷积侧一个常数只改绝对电平。两者在线性链里对双耳线索都没有影响。

## 7. 单测复跑

在 `/data/jzy/tmp/wt-grok-pilot46-round2`（`import avengine` 解析到该树），`PYTHONPATH=src:tmp/native_python_addons_v1`，`-p no:cacheprovider`：Grok 触碰的 16 个文件 + 我第一轮 23 个文件的基线 + 审计器测试 + `tests/unit` 里引用被改模块的文件，去重后 **41 个文件，581 passed / 0 failed / 0 skipped**（33.9 s）。文件清单与逐文件收集数在 `/data/jzy/tmp/claude_audit_round2_20260907/pytest_files.txt`、`pytest_collect_by_file.txt`，日志 `pytest_round2.log`。第一次启动时 grep 把 `tests/unit/__pycache__` 里的 `.pyc` 也当成测试文件，pytest 报 "not found" 没跑成，改成 `--include='*.py'` 后重跑；记下以免别人踩同一坑。Grok 报的"16 个文件 130 passed"是其子集，同 16 个文件的收集数之和恰为 130。`tools/build_tool_index.py --check` exit 0。子代理另跑过 `test_merge_qa_batch_attempts.py + test_qa_batch_runner.py` 23 passed、`test_room_package_reproducibility.py + test_build_qa_batch_manifest.py` 12 passed。

## 8. owner 已裁定与待定项（如实记边界，不当阻塞）

- 第一轮六条与第二轮第 7 条继续有效；本轮 H-1 的实现与 owner 第 7 条一致。
- "不要求每个房间跑出每种题"只绑先导四格。**本轮新出现的空格**——`authored_b_device_device` 的 attempt_03 因削波无成片——不在四格里，处置见第 5 节。
- "先导切分 train39/eval0：接受"，不扩成"无独立 eval"。挂装 9 个资产留在分母。正式准入本轮不做，也不是就绪门。
- 人工试听 JSON 的 `heard / reviewer` 我没有代填，也没看到 Grok 代填（第一轮两份 pending JSON 与 P7 十条仍为 null）。
- owner 本轮新听了六段 attempt_03（`Documents/pilot46_listening_attempt03_20260907`：authored_a_animal_device −1.6、kujiale_animal_device −2.2、authored_a_human_device −5.2、apartment_device_device −11.7、authored_a_animal_animal −20.0、apartment_human_device −17.6 dBFS），原话"其实听起来都没啥问题"。本文只把它当专题 A/B 建议里的一条输入。Grok 的 H-6 §5 写"本轮未打包到 Mac"，是在 owner 听之前写的，不算错，下一版应如实登记（仍不代填）。
- 便携空调处置、画外锚点不说话时能否入画、设备要不要画外槽：owner 说无所谓，不挡后续。本文只记边界（H-7 条目），没有改候选范围或裁剪规则。

## 9. 分类汇总

| 类别 | 条目 |
|---|---|
| 需重出产物（代码不改） | S1 合并表在 HEAD 重出；S2 放量干跑在生产树重出并记全输入；S3 attempt_03 抽段在 HEAD 重新收口；S4 `batch_outcomes.json` 重出 |
| 需修代码 | S5 `_outcome_failure_fields` 复用主分类器 + 默认落点单测 + audio/launch 说明；S6 相对路径校验与消费口径统一；S7 `expandvars` 残留与缺绑定报错只列一个；S8 H-1 depth 覆盖优先级与"房间包"措辞；S2 附带的 `--catalog` 护栏 |
| 需补证据 / 改口径 | K1–K10（五态数字、catalog 来源、56/56 与 24/28、H-4 报告一句、20 格交叉表假绿、嵌套记账与默认值假设写进文档、试听登记、色值清单差集、我自己的"32 段"与"−6～−25"、Habitat 静止段说明）；另：3/1 阶晚期能量 +7.5 dB 的绝对电平校准待查；我的 C3 判据 |
| owner 已裁定 | 先导四格、挂装、空调两条、切分、试听不代填、正式准入不做、双耳阶数修、便携空调/画外/设备画外槽三条不挡 |
| owner 待定 | 专题 A 处置（fail-closed vs 全批常数，含是否 UE+Habitat 一起重渲）；`merged_episodes.json` 47 处继承的 Codex 树 `command` 是否改写；分类器默认落点两种修法选一 |

## 附：本轮新增文件

- 本报告：`docs/roadmap/AUDIT_GROK_ROUND2_20260907.md`（提交在 `grok/pilot46-fixes-round2-20260907`，未 push；Mac `~/Documents/Claude/` 同名副本）。
- 服务器 `/data/jzy/tmp/claude_audit_round2_20260907/`：`pytest_files.txt`、`pytest_collect_by_file.txt`、`pytest_round2.log`；我的脚本 `h1_verify.py`、`h1_more.py`、`clip_probe.py`、`rir_energy_compare.py`、`run_audit_a03.py` 与输出 `mine/*.txt`、`mine/h1_verify.json`、`mine/audit_attempt03/*.json`；子代理 `x1/`（`REPORT_x1.md`、`merged_repro/`、`compare_merge.*`、`delta105.*`、`classifier_probe.*`、`legacy_and_failedrows.*`、`find_codex_refs.*`）、`x2/`（`REPORT_x2.md`、`dryrun_repro/`、`dryrun_repro_h7reg/`、`t1`–`t8` 脚本与输出、`planonly/`）、`x3/`（`REPORT_x3.md`、`counterexample_*.json`、`h4_item*.py`、`h5_item*.py`）。
- Grok 的三个临时子树 `wt-grok-pilot46-H2/H7`（及 H3–H6）仍挂在 `git worktree list` 里；S1/S2 重出之后它们就可以收掉，收之前别删（350 条 request 还指着 H7）。
