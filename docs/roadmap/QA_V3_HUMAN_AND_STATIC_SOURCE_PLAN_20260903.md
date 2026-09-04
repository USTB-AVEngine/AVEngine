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
| 人声 `speech_playback` | 620 条：VCTK 600 条（16 个 train 说话人 × 25 句 = 400；8 个 eval 说话人 × 25 句 = 200；`clip.json` 带转写、说话人号、性别、口音与 split，600 条全部 prepared）+ FSD50K 20 条 |
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

owner 已定起手：**两个改色人物、走路动作**，与两只狗平行。人声怎么分配，owner 2026-09-03 推翻了
"两角色共用同一条波形"那条，理由值得原样记下来：

> 本来现实生活中两个人的音色就是不一样的，那又如何？你不看画面照样不知道说话的那个人的衣服的颜色是什么吧？

这条反驳是对的，共用波形那条建议站不住，而且对 card13/card14 有害：

1. **音色本身不泄露答案。** card14 问"说这句话的人穿什么颜色"，答案是视觉属性。只听声音能知道的是
   "有人说了这句话"和"声音来自哪个方向"，要把方向对上身体、再读出衣服颜色，只能看画面。
   音色能泄露答案的唯一途径是**数据集里说话人和答案相关**（比如某个说话人总穿蓝的），
   那是配对方式的缺陷，不是音色的缺陷。
2. **共用波形会把这两道题弄坏。** card13 问"穿某颜色的人说了什么"，两个人说同一句话时这道题没有唯一答案；
   card14 问"说这句话的人是谁"，两个人说同一句话时"这句话"指代不明。这两道题**必须**两人说不同的句子。

所以人声按现实来：**两个角色各用各的说话人、各说不同的句子**。防捷径靠**配对随机化与配平**，不靠抹掉音色：

- 说话人与上衣颜色**独立随机配对**，并在批次内配平，任何说话人不得只与一种颜色同现；
- 说话人与答案（无论是转写还是颜色）不得相关，配平表进 manifest，和相机高度那层分层同一个机制；
- 保留现有的**只给音频的捷径探针**：如果只听声音就能把答案答到高于随机，就是配对出了问题，回去改配平，
  而不是改音色。

句子的取用范围 owner 2026-09-03 已定：**只用 VCTK 自带非测试切分的句子**，把它的 eval 留着，
这样将来跟别人在 VCTK 上的结果比才干净。所用 clip 的 sha、说话人编号、切分标记与转写文本都写进 fact，
转写供 card13/14 当标准答案。

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
4. **不适用**：card6R 这类「谁在动谁没动」的题不能拿音响当静止角色，外观把答案泄了。owner 2026-09-03 已把这条扩成通用规则（答案取决于声源自身位移的题共八个），见 `QA_V3_ASSET_POLICY_PROPOSAL_20260902.md` 第 8 节。
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
2. ~~VCTK 轮换粒度：按 episode 换说话人，还是按批次换；是否限定 split=eval 之外的句子。~~ **owner 2026-09-03 已定**：两人各用各的说话人（不共用波形，理由见 H1），句子只取 VCTK 非测试切分；防捷径改为说话人×颜色×答案的独立随机与配平，加只给音频的捷径探针。
3. card13/14 要不要等第四种上衣颜色，还是先跑不需要 4 色的题型。
4. ~~音响第一批只放地上是否可以。~~ **owner 2026-09-03 已定：可以，第一批只放地上。**
5. ~~音响不进「动与静」类题的规则是否认可。~~ **owner 2026-09-03 已定：认可，并按我的建议扩成通用规则**——答案取决于声源自身位移的题都不能拿静止声源当被问的那个声源（按现有题型数下来是八个，不止原来点名的三个），静止声源仍可当干扰项。规则与题型清单在 `QA_V3_ASSET_POLICY_PROPOSAL_20260902.md` 第 8 节。

## 7. 本方案不声明

不声明人的题、音响的题或猫的题可生成或可放量；不声明任何资产已完成准入（开关还没拨）；所有对照在做之前不预设结论。

---

# 资产存放路径与文件树（2026-09-03 实测）

owner 要一份"当前资产存在哪、树长什么样"。下面每个数字都是当天在服务器上量的，不是从记忆里抄的。
分层的口径沿用 [三层库存](#) 那一节：**生成器直接读的一张表** / **外部资产树** / **声音库**，
再加上 **UE 侧**（cook 进渲染用的 stage）与 **房间事实**（我这条线产出的东西）。

## 一、生成器直接读的那张表（在仓库里，最小）

```
examples/runtime/source_asset_runtime_profiles.json      60 KB
  14 个资产：articulated_animal 8（狗 6 + 猫 2）、articulated_human 6
  6 个别名
```

这是唯一被 QA 生成器与渲染路径读的注册表。**它不引用外部资产树**，所以外部树的准入改动不会自动影响出题。

## 二、外部声源资产树（44 个三维模型，903 MB）

```
/data/avengine_external/assets/sound_source_assets_v1/          903 MB
  index.json                     ← 44 条记录 + 验收门 + 族级开关 + 准入授权记录
  <category>/<type>/<variant>/   ← 布局标准就是这三级（owner 记得的那个）
      asset.json                 ~10 KB   这一条的全部事实
      finalized.glb              ~5 MB    最终网格
      watertight.glb             ~5 MB    水密版本
      emitter_marker.glb         ~5 MB    带发声锚点标记的版本
      evidence/                           生成与验收证据
```

类别与数量：

| 类别 | 个数 | | 类别 | 个数 |
| --- | ---: | --- | --- | ---: |
| audio_playback（音响） | 12 | | household_clock | 2 |
| plumbing_fixture | 8 | | door_hardware | 2 |
| kitchen_appliance | 4 | | heating_fixture | 2 |
| climate_control | 3 | | office_device | 2 |
| cat | 3 | | safety_device | 2 |
| communication_device | 3 | | dog | 1 |

同目录另有两份历史快照，别当现役：`sound_source_assets_v1.pre_size_demotion_20260826`、
`sound_source_assets_v1_before_resting_pose_20260827_v1`。

## 三、声音库（1200 条，源 492 MB + 备好 204 MB）

```
/data/avengine_external/assets/sound_library_v1/            492 MB   源
/data/avengine_external/assets/sound_library_v1_prepared/   204 MB   16 kHz 单声道、峰值归一 −3 dBFS
  <class>/<clip_id>/clip.wav                 备好的波形
  <class>/<clip_id>/clip.json                事实（只在源库那边）
  prepared_manifest.json                     1151 备好 / 32 别名 / 17 跳过
  36 个类目录
```

人声在 `speech_playback/` 下，613 条里 **600 条是 VCTK**（另 13 条来自 FSD50K）。
每条 VCTK 的 `clip.json` 带转写、说话人、性别、口音、切分：

```json
{"transcript": "Please call Stella.", "speaker_id": "p225", "gender": "F",
 "accent": "English", "split": "eval", "dry": true, "controlled_content": true}
```

**按 owner"只用非测试切分"的裁定，可用池是**：

| 切分 | 说话人 | 句子 |
| --- | ---: | ---: |
| train（可用） | 16 | **400** |
| eval（留着不用） | 8 | 200 |

每个说话人恰好 25 句。所以 card13/card14 的人声池是 16 个说话人 × 25 句 = 400 句，
两人各用各的说话人时，一段视频消耗 2 个说话人 2 句。
准备清单、事件清单、pool 与声资产注册表会继续携带侧车明确提供的 `speaker_id`、`utterance_id`、`transcript`、`split` 字段；旧侧车缺少这些可选字段时保持缺失，不从路径或 source 文本推断。

## 四、UE 侧：内容注册表与打包好的 stage

```
/data/avengine_external/ue-assets/actor_content_registry_v9_20260823T033709Z/   389 MB
  cpp/unreal_projects/SpearSim/Content/MyAssets/Audioset/
      Blueprints/gate_<资产标签>/BP_gate_<资产标签>.uasset
      Meshes/gate_<资产标签>/{Idle,Walking}.uasset
```

**闭包报告是从这棵树里挑文件 cook 进 stage 的**，不是从某个工程检出里挑，这一点第四色进 UE 时很关键。

```
/data/avengine_external/ue-package-stages/
  qa_v3_apartment_n4_pixel_20260901_v1        14 GB   ← 公寓渲染在用
  qa_v3_kujiale_baked_lit_275809d_20260901     ~7 GB   ← Kujiale 渲染在用
  其余六个历史 stage                            7–12 GB 每个
```

## 五、人物源与四种上衣色

```
/data/datasets/rocketbox/Microsoft-Rocketbox/          27 GB   官方源（只读）
/data/datasets/rocketbox/approved_baselines/           72 MB   密封基线（retarget.blend/.glb）
    rocketbox_neutral_walk_v1/{rocketbox_male_adult_01,rocketbox_female_adult_01}/

<SPEAR 根>/tmp/rocketbox_native_runtime_v1/<tag>/           runtime.glb + variant_manifest.json
<SPEAR 根>/tmp/rocketbox_native_runtime_ue_v3/<tag>_ue_v3/  UE 归一化版 + normalization_manifest.json
<SPEAR 根>/tmp/rocketbox_native_ue_import_v3/<tag>_ue_v3/   ue_import_manifest.json（编辑器导入的记录）
```

三种旧色在 workspace SPEAR 根下（`45e3dec20372` 酒红、`cdd6afc5b879` 绿、`f0c379dd868d` 蓝），
**第四色黄 `ec958e7654fc` 在 `/data/jzy/code/SPEAR-lead-b-m6-atomic-audit/tmp/` 下**，
前两段（原生 runtime、UE 归一化）已产出，第三段（编辑器导入）还没跑。

## 六、房间事实（我这条线产出的，都在 /data/jzy/tmp）

| 产物 | 大小 | 说明 |
| --- | ---: | --- |
| `qa_v3_camera_clearance_table_apartment_20260903_v2` | 3.3 GB | 按实测地面重渲的机位净空表 |
| `qa_v3_camera_clearance_table_kujiale_20260902_v1` | 341 MB | Kujiale 净空表 |
| `qa_v3_walkable_grid_apartment_20260903_v2` | 16 MB | 可走栅格（导航网格 10 cm） |
| `qa_v3_walkable_grid_kujiale_20260903_v2` | 12 KB | 可走栅格（可行区 5 cm） |
| `qa_v3_floor_reference_apartment_20260903_v2` | 3.5 MB | 地板 27.11 cm，两法互证 |
| `qa_v3_floor_reference_kujiale_20260903_v3` | 288 KB | 地板 0.02 cm，低相机深度法 |

房间与声学的外部输入：

```
/data/avengine_external/studio/tasks/20260826T183507Z-kujiale_acoustic_package/   280 MB  RLR 声学包
/data/avengine_external/studio/tasks/20260826T185508Z-kujiale_route_bank/          11 MB  路线库
/data/avengine_external/review/apartment_route_bank_20260825T0700Z/               6.4 MB  路线库
/data/avengine_external/rlr-sdk/hrtf/mit_kemar_normal_pinna_16k_v*/                       HRTF
```

## 七、命名规则（owner 2026-09-03 已定）

布局标准是 `<category>/<type>/<variant>`，三级、很短，没有问题。**长的是 asset_id**，
它把类别拍平后又拼上了准入状态与版本。owner 裁定：状态不再进 ID，
新批次的 ID 是 `generated_<type>_<variant>_v<N>`（提交见 `publish_static_source_assets.py`）。
已发布的 44 个 ID 保持原样，因为 `..._research_v<N>` 这套命名法与狗/猫/人物那几族共用，
那几族被上千个已产出文件引用；回溯改名要单独排一次并清算引用。
