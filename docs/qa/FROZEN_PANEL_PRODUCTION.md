# 冻结绑定面板：48g 试跑与 CCI 正式分片

流程复用已有 Habitat 原生人物/相机读回和 RLR CPU 声学链。每个独立父场景输出两个只换服装的视频、两个只换完整台词物理发声位置的 FOA，以及四个笛卡尔积成员。静态姿态不随音频改变；房间 mesh/material 和点嘴部发声位置固定。输入仅 QA-27 开放题。试跑 split=pilot；正式 split=test，12 套验收房各 60 组。

代码需要现有安装版 Habitat/Magnum/RLR、SciPy、NumPy、SoundFile、Shapely、ffmpeg；不安装或修改共享环境。新房必须先测地板，HM3D 原始 Z-up 转为 `[x,z,-y]`，导航沿用 common Y-up。仅允许验收文件已检查房间；新站位逐点查导航、地面多边形、身体可见射线和原生画面。质量不过整组保存到 rejected/，最多三个新候选，不按模型结果筛选。

本轮 48g 的实际路径、数字、失败记录见运行目录 `REPORT_zh.md` 和 `PROGRESS_zh.md`。正式数据全部在 CCI 生成。本交付没有在 CCI 启动作业；下列命令需由 CCI 生产线在已经分配资源后执行。

## CCI 输入准备

先在 CCI 建自己的授权检出/分支和输出根。准备 `cci_base.json`：保留试跑 config 的键结构，runtime 四条路径、模板 plan/dataset、speech、registry、hrtf 全部指向 CCI 实际已安装/已迁移文件；不要只改顶层路径而保留 speech/房屋 package 内部的 48g 绝对路径。脚本会检查 speech 内每条干声音频路径。HM3D scene、navmesh、semantic、annotation 和 acoustic package 由房屋验收线提供；不可改原始数据。配置 schema 可参照生成器源码，实际 pilot config 在运行目录 `inputs/config.json`。

`cci_houses.json` 为 12 项列表，每项具有 `house`、`rooms`（从新 recommended_room_scope 的 checked_rooms 原样选取）、`acceptance_result`、`room_manifest`、`acoustic_package_manifest`。后面三条路径都在 CCI，验收 status 必须 pass，必须带可用的 measured floor_polygon。正式 12 套房未经训练/选模暴露的审计由房屋验收线提供；本轮只验证已授权的两套。10 套尚未验收时不能运行正式配置。

准备试跑最终独立验证输出 `verify_final.json` 作为排除清单，不含答案。该文件的 20 个 physical_parent/video_pair 哈希用于拒绝复用试跑父场景。正式随机种子必须不同；首个父场景也随机取站位，不能改一个新 panel_id 后重复试跑。

在 CCI 授权检出中执行（环境变量由生产线填实际路径，未预设 CCI GPU）：

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PANEL_REPO/src:$PANEL_REPO"
"$PANEL_PYTHON" tools/qa/prepare_frozen_panel_config.py \
  --base-config "$PANEL_INPUTS/cci_base.json" \
  --houses "$PANEL_INPUTS/cci_houses.json" \
  --scope "$PANEL_INPUTS/recommended_room_scope.json" \
  --panel-id frozen_test_20261004 --split test --seed 2026100401 \
  --groups-per-house 60 --expected-houses 12 \
  --pilot-verification "$PANEL_INPUTS/verify_final.json" \
  --output "$PANEL_INPUTS/cci_panel.json"
```

house 列表顺序、配置和 seed 所有分片一致。独立父场景全局编号 i 分配为 i % shard_count；4 片各 180 组，每片每房 15 组。配额 `(j+2*house_index)%4` 使每房 60 组各配额 15 组，12 房每格 180 组。单片房内配额可偏斜，合并后检查全额。

## 分片与续跑

一个 worker 使用一张已经分配的 GPU、顺序渲染两个视频；RLR 8 线程，ffmpeg 限制线程。首次空余显存至少 12288 MiB；后续至少实测峰值 +4096 MiB，保留 12 GiB 保守要求。新的 CCI 房/卡峰值未验证，实测更大时提高配置 min_free_vram_mb。不抢占或杀其他人的进程。

每个 worker 使用不同 output，禁止两个 worker 共用一个目录；文件锁会拒绝。下例只展示第 0 片；第 1/2/3 片分别传对应编号、独立输出和其已分配 GPU。只有一张授权 GPU 时传 `0 1`，顺序生成 720 组。

```bash
bash examples/qa/frozen_panel/run_cci_shard.sh \
  "$PANEL_PYTHON" "$PANEL_INPUTS/cci_panel.json" \
  "$PANEL_OUTPUT/shard_00" 0 4 "$ASSIGNED_GPU_0" \
  > "$PANEL_LOGS/shard_00.log" 2>&1
```

后台或调度器运行时保存作业号/PID 和日志。`run_state.json` 记录 PID、命令、当前组和阶段，PROGRESS_zh.md 逐阶段追加；`evidence/*_execution.json` 保存耗时和退出码。续跑使用相同命令/config，已有成功阶段复用，独立验证再执行。资源不足、未知错误、证书错误立即停止；明确质量不足才整组剔除/重采样。封存后 worker 拒绝续写，不能靠覆盖承诺文件继续。

## 独立验证、封存及合并

生产过程中逐组运行 `tools/qa/verify_frozen_panel.py`。每片完成后先做一次全片验证，再由评分脚本封存。不得运行模型推理或评测来选组。

```bash
"$PANEL_PYTHON" tools/qa/verify_frozen_panel.py \
  --root "$PANEL_OUTPUT/shard_00" \
  --output "$PANEL_OUTPUT/shard_00/evidence/verify_final.json"
"$PANEL_PYTHON" tools/qa/score_frozen_panel.py seal \
  --root "$PANEL_OUTPUT/shard_00" --expected-groups 180
"$PANEL_PYTHON" tools/qa/score_frozen_panel.py audit --root "$PANEL_OUTPUT/shard_00"
```

四片都通过并封存后，评分脚本是唯一读取答案的合并器；新 aggregate 目录不得已存在。同配置、组号不重叠、成员不重叠、期望总量逐项检查。保留原分片的 native production 和外部输入，证书 provenance 指向这些原生证据；公共媒体优先硬链接，跨文件系统复制。

```bash
"$PANEL_PYTHON" tools/qa/score_frozen_panel.py assemble \
  --root "$PANEL_OUTPUT/frozen_720" --expected-groups 720 \
  --shard-root "$PANEL_OUTPUT/shard_00" --shard-root "$PANEL_OUTPUT/shard_01" \
  --shard-root "$PANEL_OUTPUT/shard_02" --shard-root "$PANEL_OUTPUT/shard_03"
"$PANEL_PYTHON" tools/qa/verify_frozen_panel.py \
  --root "$PANEL_OUTPUT/frozen_720" \
  --output "$PANEL_OUTPUT/frozen_720/evidence/verify_final.json"
```

验证器不导入生产器或证书器，重新解码逐帧 RGB、完整 FOA/binaural/mono PCM，核查时钟、原生人物深度/语义/关节/嘴部位置、允许字段、服装交换、独立完整音频卷积/路由及已验收房间范围。全片查 physical parent 与实际 video pair 唯一性、试跑排除清单；M1/M2/cross 缓存为 2/2/4。封存后的私有承诺和棋盘真值审计委托给评分脚本。

目录边界：

```text
inputs/                 声学/场景/素材/运行时配置与来源
production/             native arrays、计划、RIR、完整媒体（700）
rejected/               整组质量失败原件（700，不删除）
scoring_private/        certificates/、answers_parts/、answers.json、seal_audit.json（700）
public/groups/<opaque>/ video.mp4、foa.wav、binaural.wav、mono.wav、request.json
public/requests.jsonl    打乱的请求；没有 group/member/人物 ID 或答案
public/COMMITMENTS.sha256 私有文件与请求清单的 SHA256 承诺
evidence/ logs/          独立验证、耗时、执行日志
```

随机 nonce 进入私有答案文件，阻止蓝/绿二选一答案的简单枚举承诺。单一研究账户仍可读自己的 700 私有目录；隔离靠实际输入适配器只读 public 和严格白名单，不宣称抗同账户恶意进程。提交不包含原始数据、媒体或真值。

## 模型输入的后续接线

本轮仅运行 CPU 预处理和真实 AVModel-hub 输入 helper，无模型权重/tokenizer/encoder。`frozen_panel_inputs.build_model_payload` 输出公开题目、40 个实际时间戳 RGB 帧、完整 10 秒 binaural mean mono，level2 加 camera ACN/SN3D FOA。路径/随机样本 ID/缓存键在编码前剥离。

`audit_frozen_panel_model_bridge.py` 使用只读 AVModel-hub source，执行其实际 build_llm_input、_episode_key、抽帧、mono/FOA feature helper；按路径最后一个 groups/ 段建立成员 episode。可用 `--export-dir` 导出五类 ready specs。准备的 FOA 已是相机坐标 SN3D，消费 `_feature` 或 mono builder；旧 run_llm_fast CLI 的 `_prepare_runtime_foa_adapter` 需要 world-N3D/private facts，不适用，不能对本面板再次调用。没有宣称旧 CLI 全部可直接跑评测。

L1 heard/seen/cross 证据本轮未生成。未来 M1 用实际 audio PCM 哈希 + 模型/设置复用，M2 用实际 video pixels/选帧哈希 + 模型/设置复用，cross 用实际成员组合生成。`build_model_payload(...,'level1',evidence=...,producer=...,settings=...)` 拒绝没有媒体来源或父场景共享的证据；不能用私有事实回填。真实 tokenizer/resize/模型处理尚未验证，应在正式评测线记录其配置和最终送入张量，不改冻结真值/媒体。

`score_frozen_panel.py score` 后续可读 `sample_id,prediction_label` JSONL（blue/green/unknown），输出成员正确率和四成员全对率；本轮没有执行这一功能或模型评测。
