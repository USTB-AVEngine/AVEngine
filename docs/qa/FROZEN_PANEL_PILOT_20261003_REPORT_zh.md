20 组试跑已完成三层证书、独立验证和封存，流程可交 CCI 按分片生产 720 组；按 48g 实测外推，串行成功路径约 37.9 小时、当前重试保守预算约 51.2 小时，4 卡分片按较慢配额预留约 9.5–16.0 小时（CCI 硬件与另 10 套房未验证）。

报告时间：2026-10-03T16:43:28.619543+00:00（UTC）。机器 48g / cw-SYS-4029GP-TRT3。运行目录 `/data/jzy/tmp/claude_hardbank_20260923/test_panel_pilot_20261003`。代码 worktree `/data/jzy/code/AVEngine-test-panel-pilot-20261003`，分支 `codex/test-panel-pilot-20261003`，基点 `c21c387`；提交号见运行目录 `evidence/source_commit.txt`，不 push。

## 配额和独立性

| 测试房 | 同声线/异侧 | 同声线/同侧 | 不同声线/异侧 | 不同声线/同侧 | 总组数 |
|---|---:|---:|---:|---:|---:|
| hm3d_val_00808_y9hTuugGdiq | 3 | 3 | 2 | 2 | 10 |
| hm3d_val_00877_4ok3usBNeis | 2 | 2 | 3 | 3 | 10 |
| 总计 | 5 | 5 | 5 | 5 | 20 |

每房 10 无法单房四等分，两房采用互补 3/3/2/2 与 2/2/3/3；全试跑四格各 5。同/不同声线由干声 voice_preset 相等/不同判定，主观声线可分性未验证。正式每房 60 时每格 15，12 房每格 180。QA-27 开放题是主问题，四成员问题完全一致；QA-26 未加做。80 个成员全部 split=pilot，证书 pilot/exclude_final_panel=true，不进入最终面板。

使用 6 个已检查房间，10 个台词情境，每个情境在两套房分别新建父场景；同一父场景没有按不同问法或切窗重复计数。独立复验实得 20 套不同父场景几何与 20 对不同实际视频。出处：`evidence/verify_final.json`、`evidence/production_metrics.json`、私有 recipe/证书。房间组数：`{'hm3d_val_00808_y9hTuugGdiq/R4': 4, 'hm3d_val_00808_y9hTuugGdiq/R0': 3, 'hm3d_val_00808_y9hTuugGdiq/R5': 3, 'hm3d_val_00877_4ok3usBNeis/R7': 4, 'hm3d_val_00877_4ok3usBNeis/R8': 3, 'hm3d_val_00877_4ok3usBNeis/R12': 3}`。

只从 recommended_room_scope 许可房间取点；房间 source floor_polygon 与 native navmesh 再查、新相机/双人站位再查身体射线，失败整组拒绝。HM3D 原始 Z-up 转 common Y-up `[x,z,-y]`，逐房先取地板偏移并记录 native snap。00808 的 R4/R5 地板约 0.061391 m、R0 约 3.261391 m；00877 的 R7/R8/R12 约 2.064475 m，来源每组 recipe.geometry.floor_offsets_m。房屋验收文件来自 v7_batch1_20261003/measurement_final，原生声学包来自 measurement_v2。原件只读。

## 构造、证书与封存

已有 v6 成对链可复用到这两套 HM3D：仍是 Habitat 出原生 RGB/深度/语义/骨架/嘴部读回，RLR CPU 按真实房间 mesh/material 与实际嘴部位置出声场。新适配器只做房屋/坐标/站位/时钟选择与跨成员装配；未修改原生 Habitat/RLR 或既有校验器。使用已有 ASR-admitted CosyVoice 干声，不运行 TTS/ASR 或任何模型。干声路径、完整 PCM hash、台词、声线和既有 admission 在音频 receipt；两套音轨只改变完整台词 clip 的物理嘴部路由，按各位置 RIR 独立卷积。

每组两个视频只交换蓝/绿上衣。人物 mesh/body、根/关节姿态、位置、嘴部、相机、光照、时间固定；姿态全程静态，无音频驱动嘴型或动作。声学使用固定真实房间与点嘴部发声器，不建动态人体/服装声学遮挡代理；因此服装不改变声学几何/材质。这是当前建模边界，不能据此声称人体遮挡已物理验证。

每组一份证书：`scoring_private/certificates/pilot_000.json` 为实际样例，另有 pilot_001 至 pilot_019。构造层记录字段白名单、固定计划/原生几何、不变问题与棋盘关系；媒体层记录所有 150 帧解码 RGB 哈希、完整 FOA/binaural/mono PCM、时钟和长度；输入层记录 40 帧的实际索引/时间/像素、完整音轨、模型 payload 白名单与媒体内容驱动的 evidence keys。

独立验证器 `tools/qa/verify_frozen_panel.py` 不导入生产器/证书器，重新解码和检查原生数组；独立按固定干声、两个物理位置 RIR、共享增益及实际相机 basis 重建完整 float32 FOA/binaural，逐样本相等。全部 20 组三层通过，逐组 inconsistent_episode_relations=[]，同音频跨视频 FOA 哈希一致，同视频跨音频全部帧哈希一致。每成员 640×360、150 帧/15 fps、10 秒；完整音频 16000 Hz×160000 样本，FOA 4、binaural 2、mono 1 通道。mono 为 native binaural L/R 算术平均，FOA 不取均值。出处 `evidence/verify_final.json`。

联合可解性采用自动审核：原生 RGB 上衣粗色支持、FOA 直接方向、RIR C50、完整干声与目标段竞争声比，任何失败整组剔除。审核阈值沿用原生外观 512 支持像素/1.25 主导比；方向误差 ≤10°，C50≥0 dB，目标竞争比≥12 dB，异侧分离≥18°、同侧分离 12–34°。没有按模型表现筛组。

实测最小上衣色支持 550 像素；最小 C50 5.693 dB；最小目标/竞争声比 311.464 dB（两句不重叠，竞争声可为静音，不能解读为人类转写正确率）；最大 FOA 直接角误差 1.927235°。出处 `evidence/production_metrics.json` 与每组 joint_solvability。已查看 `review/contact_sheet_20.png` 的 20 组双视频首帧，蓝/绿上衣可见且互换；人工听感/逐字转写未验证。

真实人物 ID、正确答案和组映射只在评分私有目录/私有构造证据。scoring_private、production、rejected、review、inputs、evidence、logs 设 700；答案/证书 400，public 媒体和请求 444。评分脚本 `score_frozen_panel.py seal/audit` 是唯一答案文件读取器，检查四成员棋盘、真实物理发声源路由与原生审核服装颜色，带随机 nonce 的私有答案写 SHA256 承诺 `public/COMMITMENTS.sha256`，并随机打乱 `public/requests.jsonl`。完整封存审计 20 组/80 成员通过，见 `evidence/seal_result.json`、`seal_audit_final.json`。

全 20 组独立复验在封存前完成；封存后首组再次做完整三层验证，并把全部私有承诺检查委托给评分脚本，见 `evidence/verify_sealed_sample.json`。未把“封存后首组复验”写成“封存后 20 组又全部解码”。单一共享研究账户仍可读自己的私有目录；模型隔离来自 public-only 适配器与字段白名单，未验证抗同账户恶意进程。

## 实际模型输入接口

读取只读 AVModel-hub source `5eec6123f67c22cac8651b4c87ff52152a472b1a`；实际 build_llm_input/_episode_key/抽帧/mono/FOA feature helper 完成 CPU 核查，80 个成员形成 80 个不同成员 episode，inconsistent_episode_relations=[]。_episode_key 使用路径最后一个 groups/ 段；不是按父场景缓存一个 episode。出处 `evidence/model_bridge_final.json`。

记录 av、a_only、v_only、t_only、av_mono 五类 ready specs，位于 `evidence/model_ready_specs/`；普通 AV/L2/L1 原始 payload 由 `src/avengine/qa/frozen_panel_inputs.py` 装配。40 个真实 RGB 帧、完整 10 秒内容音频、camera ACN/SN3D FOA 及公开题目均核查；文件路径/随机样本 ID 只作传输定位，在编码前剥离，不进入文字或证据。实际 encoder cache key 的媒体文件哈希也核查；本轮没有真正计算 encoder/cache 证据。每组 M1 按实际音频内容 2 键，M2 按实际视频/帧 2 键，cross 按实际组合 4 键。

旧 run_llm_fast CLI 的 world-N3D→camera 转换需要 private facts，不适用于本轮已经转换好的 SN3D；必须消费 ready spec/_feature 或 frozen_panel_inputs，不能再旋转/缩放。桥接 helper 已验证，旧完整 bank/评测 CLI 未跑。tokenizer、图像 resize、encoder 最终张量及真正模型收到的 token 未验证；只记录其 source 中 VIDEO_MAX_PIXELS=100352、AUDIO_MAX_SAMPLES=320000，完整 10 秒未被 20 秒 cap 截断。

M1/M2/L1 预测证据均未生成。后续只接受具有实际媒体哈希与模型/设置来源的 heard/seen/cross；没有证据就拒绝，不从私有事实/答案回填。没有运行模型推理、训练、模型评测、TTS/ASR；score 的预测评分功能本轮未执行。

## 生产速度与 720 组外推

| 指标 | 每个成功独立四元组 | 出处 |
|---|---:|---|
| 完整成功路径 wall（含一次独立验证） | 3.157 分钟（2.803–3.493） | production/* 各 timing + verify_final.json |
| 测得 CPU user+system | 3.485 核分钟 | 同上；不是 wall×假定核数 |
| GPU 原生渲染占用 wall | 1.150 分钟 | visual/v0,v1/timing.json |
| 完整原生审计产物（硬链接去重） | 0.7573 GiB | production_metrics.json |
| 公共媒体（硬链接去重） | 9.149 MiB | production_metrics.json |

| 阶段 | 平均 wall 秒/组 | 平均 CPU 秒/组 |
|---|---:|---:|
| prepare | 25.041 | 26.847 |
| capture_v0 | 34.501 | 39.611 |
| capture_v1 | 34.486 | 39.296 |
| audio | 33.483 | 40.285 |
| certify | 30.989 | 34.145 |
| verify_once | 30.950 | 28.893 |

全部成功渲染在 48g GPU 2 串行完成，0.5 秒采样的自有 native 进程显存峰值 2519 MiB（2.460 GiB）。首次与后续都要求空余≥12288 MiB，此次峰值+4096 MiB 为 6615 MiB。早期 GPU 0 初始化失败/显存不足日志保留，未并行占用多张卡。RLR 每次 8 线程；没有停其他人的进程或改变共享环境。

7 个拒绝父场景全部保留在 rejected/：pilot_001 的 attempt0–2 为已修复导航/家具插值问题，attempt3–4 及 pilot_003 attempt0–1 为上衣像素支持不足；都是整组质量筛选。已记录重试阶段 wall 下界 8.239 分钟，CPU 下界 9.625 核分钟。早期失败阶段没有完整 timing，未编造数字。按 27 候选/20 成功全成本保守预算系数 1.35；其中导航问题已修复，不能把这个系数当作正式稳定拒绝率或统计置信区间。

720 组成功路径：37.890 小时；计入有记录重试的下界 42.833 小时；把当前全部拒绝候选都计为完整组的预算 51.151 小时。GPU 占用总量 13.797 小时、CPU 测得总量 41.815 核小时。4 个已分配 GPU 的独立 worker 理想外推 9.472–12.788 小时，各用一张 GPU/8 个 RLR 线程，合计 32 RLR 线程；CCI 应按实际 CPU/RAM/磁盘配额配置并发，未实测其争用和硬件速度。

当前 i%4 分片每片每房 15 组，但把异侧与同侧分到不同 worker。考虑配额速度不均：按试跑两种 side 的平均 wall 各乘 180，较慢片为 9.533 小时；7 个拒绝都在同侧，把同侧 17 候选/10 成功计完整成本，较慢片预算 16.001 小时。这个预算比简单总时长÷4 更保守，仍不是硬件实测或统计置信上界。若只有 3 张卡，--shard-count 3 各片 240 组、每房 20 且四格各 5，配额更均衡，按总量外推约 12.630–17.050 小时。来源 production_metrics.json 的逐组 wall/quota 与 rejected_results。

完整成功组原生证据预计 545.25 GiB，公共媒体 6.43 GiB；还需留拒绝候选/原型/日志空间。当前运行目录测得 19.467 GiB，包括保留的原型和失败产物；不能把公共媒体容量作为完整可审计交付的容量。CPU 捕获 wrapper 监控开销和早期失败缺计时使 CPU/重试量为下界；成功路径统计不含反复复验、搭环境、SSH 重连、人工审核时间。依据 `evidence/production_metrics.json`，可复算命令如下。

```bash
python tools/qa/summarize_frozen_panel.py --root "$PILOT_ROOT" \
  --verification "$PILOT_ROOT/evidence/verify_final.json" \
  --expected-groups 20 --output "$PILOT_ROOT/evidence/production_metrics_recheck.json"
```

## CCI 调用与交付

详细配置 schema、路径迁移、验收房屋清单、分片/续跑/封存/合并命令在代码 `docs/qa/FROZEN_PANEL_PRODUCTION.md`。CCI 尚未登录、写入或起作业。正式缺另 10 套通过验收且未暴露于训练/选模的房屋清单及 CCI 实际 runtime/assets/输出路径与已分配 GPU。不能在这两套房上凑 720。

配置准备：prepare_frozen_panel_config.py 在 CCI 用本地 base-config、12 房 houses JSON、新 scope 与本轮 verify_final 排除清单，指定 --split test --groups-per-house 60 --expected-houses 12 --seed 2026100401。正式种子与 pilot 不同，20 个几何/实际 video pair 全部排除。已在 48g 两房完成真实输入配置检查，见 logs/config_portability.log；12 房、CCI 路径可用性未验证。

```bash
bash examples/qa/frozen_panel/run_cci_shard.sh \
  "$PANEL_PYTHON" "$PANEL_INPUTS/cci_panel.json" \
  "$PANEL_OUTPUT/shard_00" 0 4 "$ASSIGNED_GPU_0"
# 其余片传 1/2/3、独立 output 和其已分配 GPU；单卡则传 0 1。
"$PANEL_PYTHON" tools/qa/score_frozen_panel.py seal \
  --root "$PANEL_OUTPUT/shard_00" --expected-groups 180
# 全片先独立验证；四片各封存后用 score_frozen_panel.py assemble 合成新根，expected-groups=720。
```

worker 每片持文件锁，编号 i%shard_count 不重叠，相同配置/种子且每片独立根；RLR CPU、native GPU 顺序执行，资源不足/未知错误停止，只有明确质量原因才整组重采样。PROGRESS_zh.md 每阶段追加，run_state.json 保存 PID/命令/阶段，execution receipts 留计时。续跑不覆盖已有证书/承诺，封存根拒绝续写。评分脚本合并时只读各片答案，检查父组/成员不重叠、配置一致、总量正确；保留原分片原生证据供总验。

主要新文件：produce_test_panel_pilot.py、run_frozen_panel_shard.py、prepare_frozen_panel_config.py、certify_frozen_panel.py、verify_frozen_panel.py、score_frozen_panel.py、audit_frozen_panel_model_bridge.py、summarize_frozen_panel.py，均在 tools/qa；输入模块 src/avengine/qa/frozen_panel_inputs.py；测试 tests/unit/test_frozen_panel_inputs.py、test_frozen_panel_sealing.py；CCI 启动脚本 examples/qa/frozen_panel/run_cci_shard.sh；工具索引自动更新。没有修改其他检出、原始数据、已有 checkpoint 或共享环境；没有 push。

48g 实际执行：run_frozen_panel_shard.py --shard-index 0 --shard-count 1 --gpu 2 --start-index 1 --end-index 20 --resume（从已有前三组续接）；逐组 prepare/capture/audio/certify/verify；全组 verify_final；实际输入 helper audit；scorer seal/audit；封存后首组 verify；summarize；pytest。所有完整命令和输出在 scripts/、logs/、evidence/ 与 timing/command receipt，不把一次 worker 的 58 分钟直接除以 20 充当产速。

软件检查：15 passed（logs/unit_panel_v3.log）；286 passed、36 skipped、5890 deselected（最终发布检查 logs/fast_unit_release.log，28.20 秒）。跳过项缺 checkout 中保留的外部 fixture；原生 Habitat/RLR 和解码检查由此次 20 组实际执行提供，未声称所有旧 native 测试跑过。CCI bash 语法和实际两房配置检查通过。

## 问题与未验证项

已修复：Habitat 配置必须 .scene_dataset_config.json 后缀、HM3D Z-up 和地板偏移、人物 semantic ID 与房间编号碰撞、同侧插值穿家具、PCM 独立重建浮点 camera basis 规范化差异、成员 episode/cross-evidence 不能按父场景共享。所有旧失败/原型原件保留，没有清理别人缓存或宽泛杀进程。

早期 recipe 缺 producer_sha256 的组：['pilot_000', 'pilot_002']。未补造历史源码哈希，封存 build_code_identity.json 明确 not_recorded；原生计划、媒体、命令与最新独立复验完整保留。调通过程生成器版本不同，正式生产只使用最终提交且记录源码 SHA/版本。

未验证：另 10 房及 CCI 实际速度/并发/空间；人工听感/逐字转写；实际 tokenizer/resize/encoder/模型 token；旧完整评测 CLI 接入。全 20 组自动联合 QA 与三层校验通过，不把这些未验证项写成已完成。正式 720 没有生成，pilot 20 明确排除。

生产 PID 3710811 已正常 complete（run_state.json）。交付核对未发现仍在运行的本任务作业，见 evidence/final_resume_audit.json；生产、全组验证和输入桥接进程均已退出。任何续跑作业的停止，必须先核对 run_state/PID 命令和父子关系，只停已确认属于本任务的进程。
