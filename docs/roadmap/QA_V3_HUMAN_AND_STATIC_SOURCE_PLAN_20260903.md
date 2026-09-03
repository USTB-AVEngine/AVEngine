# 人与静态声源的测试计划（2026-09-03，方案，不实施；同日按 Codex 纠正与 owner 裁定改版）

> 状态：提案，接在 `QA_V3_ASSET_POLICY_PROPOSAL_20260902.md` 之后。
> 第一版把"资产库存"只数了 QA 运行时注册表那一张表，结论错了。Codex 2026-09-03 指出服务器外部
> 正式研究资产树另有 44 个三维模型（12 个音响类）和 1200 条声音（600 条带转写的 VCTK 人声），我
> 逐项到服务器核实后按三层口径改写本文。owner 同日裁定：这些外部资产可以进入正式数据准入，不再只是
> research。两处改动都写在下面，原来"无音响资产""只有一条人声""猫没有声音""重新做一个音响网格"四句作废。

## 1. 三层库存（2026-09-03 服务器实测）

### 1.1 QA 运行时注册表（生成器现在能用的）

`examples/runtime/source_asset_runtime_profiles.json`：14 个视觉资产，狗 6、猫 2、人 6；人里受控上衣颜色 3 种
（酒红、绿、蓝）。声音只挂了 3 条旧素材：狗叫 1、人声 1、测试信号 1。它对下面两层**零引用**——
这才是缺口所在：不是没有资产，是没有桥。

### 1.2 外部正式研究资产树（三维模型）

`/data/avengine_external/assets/sound_source_assets_v1/index.json`（schema `avengine_sound_source_asset_index_v1`）：

| 项目 | 数字 |
| --- | ---: |
| 资产总数 | 44（刚体静物 40，带骨骼动物 4） |
| 音响类 `audio_playback` | 12：书架音箱 3、落地音箱 1、智能音箱 3、条形音箱 3、电视 2 |
| 动物 | 猫 3（缅甸猫 2、暹罗猫 1）、狗 1（杰克罗素） |
| 其他静物 | 空调 3、时钟 2、水暖件 8、厨电 4、通讯设备 3、门件 2、暖气 2、办公设备 2、安全设备 2 |

每条音响记录都带：最终网格 `finalized.glb` 与 sha、坐标系（前轴 +X）、发声锚点（如低音单元锥面，含
相对根节点的偏移与选取方法）、声学档案（允许的事件类：音乐、语音、任意 AudioSet 类、静音）、验收门
（水密、朝向、落地）与判定。**不需要再做音响网格。**

所有 44 条现在都是 `admission_state: research`、`formal_dataset_registration_authorized: false`，索引顶层也是 false。

### 1.3 声音库

`/data/avengine_external/assets/sound_library_v1`（原始）与 `sound_library_v1_prepared`（已统一到 16 kHz 单声道、
去直流、去首尾静音、峰值归一，`prepared_manifest.json`）：

| 项目 | 数字 |
| --- | ---: |
| 条目 | 1200 条、37 类；prepared 1151、alias 32、skipped 17 |
| 人声 `speech_playback` | 620 条：VCTK 600 条（24 个说话人 × 25 句，`clip.json` 带转写、说话人号、性别、口音、split=eval，600 条全部 prepared）+ fsd50k 20 条 |
| 狗叫 `dog_bark` | 21 条（19 prepared） |
| 猫叫 `cat_meow` | 20 条（18 prepared） |
| 音乐 `music_playback` | 20 条（15 prepared，5 alias） |

所以：人声不是一条而是 600 条且带标准答案文本；狗叫、猫叫都有；card13/14 要的"四条转写"素材是现成的。

## 2. owner 裁定与边界（2026-09-03）

1. **准入**：owner 裁定外部资产可以进入正式数据准入，不再限定 research。落地方式是资产索引里每条记录的
   `admission_state` 与 `formal_dataset_registration_authorized` 两个字段，它们是准入开关；改这两个字段是
   注册表层面的动作，不在本工作副本里，本轮我没有动外部索引。桥接进运行时注册表时要把外部记录的验收门
   证据与 sha 一并带过去，准入才有依据。
2. **题目产物**仍然是 `research_candidate`：资产准入和百题 pilot 认证是两条线，前者不自动推出后者。
3. Codex 的另一半意见保留：静态声源用于多样性与对照，不改变"动态人/动物 + 双模态必要性"这条主线。

## 3. 顺序

### 阶段 B0：桥接（先于一切，1–2 天）

把外部资产记录接进 QA 运行时注册表与 UE 端：

1. 注册表条目：从外部索引生成运行时档案（asset_id、族、网格包、发声锚点、声学档案、体尺、验收门、
   admission 字段原样带过来），不手抄。
2. UE 内容：`finalized.glb` 进现有 Blender 资产管线出 UE 包（人和狗走过的那条路），Blueprint 里发声端点
   按外部记录的锚点偏移放，不再"放在包围盒中心"。
3. 声音：运行时的声音选择改为按声学档案的事件类到 `sound_library_v1_prepared` 里取，替掉写死的 3 条。
4. 每桥接一件，跑一次现有的"放进房间渲一帧 + 深度可见性"探针，确认落地、朝向、尺寸。

### 阶段 H0：人的注册表补齐（0.5–1 天，不动生成器）

1. 每个人补 `body_m`（身高、体长、体宽；占位 1.7/0.5/0.5，可由包围盒校）。
2. `appearance_label` 与 `referring_expression`（上衣颜色词）。
3. 第四种受控上衣颜色：card13/14 要 4 种，现在 3 种；改色走 Blender 资产管线。转写素材已经够。

### 阶段 H1：两个人的题（生成器改动 1–2 天 + canary）

owner 已定起手：**两个改色人物、走路动作**，与两只狗平行；声音按 Codex 建议改为：

- **同一个 episode 里两个角色共用同一条波形**（身份只靠外观区分，声音不泄露答案）；
- **episode 之间从 VCTK 池轮换**（24 个说话人 × 25 句），不再全数据集固定一条人声；
- 轮换规则与所用 clip 的 sha 写进 fact；转写文本随 clip 带上，供 card13/14 做标准答案。

其余同第一版：题型配置加 `asset_policy`（同族不同实例，`allowed_families: ["dog", "human"]`，指代词按族取）；
`build_cell_plan` 的配对从政策取，`COAT_WORDS` 换成资产的 `appearance_label`；两房 canary 各 6 格；card1
候选跑像素真值；三项标定（机位堵死阈值改 1.7 m 目标带、预测可见性、分档边界）对人重做，不沿用狗的数字；
人类校准包等狗的 v4 先出。

### 阶段 S1：静态声源（接入现有 12 个音响，桥接 1 天 + 生成器 1 天 + canary）

1. 资产：不做新网格，桥接现有 12 个 `audio_playback`，注册为 `family: speaker`，`motion: must_be_still`，
   `voice.identity_coupled: false`，声学档案允许音乐、语音、任意 AudioSet 类、静音。
2. 摆放：零位移路线。Kujiale 有 200 条静止路线可用；Apartment 从可走栅格抽站点（路线合成那条线已经
   有这张栅格）。外部记录的 `resting_pose_attachment_surface` 是 floor，第一批放地上；放桌上要"桌面可放置"信息，后置。
3. 适用题型：`any_single_source` 的 card8（首叫时间带）、card15b（事件计数）；以后"声音从哪来"类混族题，
   题面只能说"声音"不能说"狗"。
4. **不适用**：card6R 这类"谁在动谁没动"的题不能拿音响当静止角色，外观把答案泄了；资产政策里写成同族配对才允许。
5. 音频：音响播音乐、人声或 AudioSet 类；fact 记"声音内容与身份脱钩"；探针专门查"只听声音能不能猜到是音响"。
6. canary：card8 用"一只狗 + 一个音响"与"两个音响"各 6 格，看首叫间隔链与 Gate A 翻转是否照常成立。

### 阶段 M1：猫与混族（资产政策定稿后即可做 canary，不必等 H1/S1）

Codex 建议猫可以提前：运行时注册表已有 2 只猫，外部树再有 3 只，猫叫 18 条 prepared。资产政策一过就跑
猫的两房 canary；混族题（狗 + 猫）按 `mixed_families_voice_decoupled` 或 `same_family_distinct_instances` 声明。

## 4. 验证原则

每一族都走同一条链：几何求解 → 时间线 → 像素真值 → join → 对照；三个标定过的东西（机位堵死阈值、预测
可见性、分档边界）换族就重新对照。桥接进来的每件资产先过一帧渲染探针再进题。

## 5. 成本汇总

| 阶段 | 工时 | GPU |
| --- | ---: | --- |
| B0 桥接 | 1–2 天 | 每件一帧探针，合计不到 1 小时 |
| H0 注册表 | 0.5–1 天 | 无（改色走资产管线另计） |
| H1 人的题 | 1–2 天 | 两房 canary 像素真值约 1 小时 |
| S1 音响 | 2 天（不含网格） | 同上 |
| M1 猫/混族 | 1 天 | 同上 |

## 6. 需要 owner 决定

1. 准入开关谁来拨：外部索引的 `admission_state` / `formal_dataset_registration_authorized` 由 owner 或注册表
   维护方改，还是授权我在桥接时按 owner 裁定改。
2. VCTK 轮换粒度：按 episode 换说话人，还是按批次换；是否限定 split=eval 之外的句子。
3. card13/14 要不要等第四种上衣颜色，还是先跑不需要 4 色的题型。
4. 音响第一批只放地上是否可以。
5. 音响不进"动与静"类题的规则是否认可。

## 7. 本方案不声明

不声明人的题、音响的题或猫的题可生成或可放量；不声明任何资产已完成准入（开关还没拨）；所有对照在做之前不预设结论。
