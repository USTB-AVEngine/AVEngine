# AVEngine 受控声源资产属性 JSON 与生成工作流

> 状态：当前属性设计的单一事实来源（SSOT）
> 更新日期：2026-07-13
> 适用范围：Rocketbox 人类路线、FLUX.2 + Pixal3D 动物路线，以及后续
> Apartment 音视频数据集注册

## 1. 目的与已经冻结的决定

本文规定属性 JSON 如何产生、如何控制视觉资产生成、如何关联音频，以及
生成后的实际尺寸和 QA 如何回写。后续代码和批量任务必须遵守以下决定：

- 每个 instance 是一个独立个体，只保存绝对属性；禁止在身份 JSON 中使用
  `from`、`to`、`one_step_lighter`、`lighter_than_original` 等相对修改描述。
- 每个**可采样的离散外观属性**最多三个取值；必要时只能有两个或一个。
  固定属性、物理测量值、哈希、许可证和 QA 状态不受“三个取值”的限制。
- 动物按物种/品种分别定义属性域。金毛、比格、哈巴狗、暹罗猫和虎斑猫
  不共享一套不准确的颜色名称。
- 每次先采样完整属性，再自动生成一个完整 prompt，只执行一次 FLUX.2
  图生图；禁止通过“先变大、再变深、再变壮”的连续编辑积累漂移。
- 人类第一版不随机修改眼睛，不随机增加、移除或修改帽子、眼镜、首饰和
  其他附件。原 Rocketbox 角色已有的附件作为固定几何保留。
- Rocketbox 是固定模板库，不是自由换装系统。长袖、短袖、长裤、短裤、
  制服和附件差异来自不同的 `base_avatar_id`，不是由 FLUX 在贴图上凭空生成。
- 人类可采样项仅来自该 Rocketbox 模板已通过 mask/material 审计的区域。
  第一版候选项是上衣颜色、下装颜色、鞋色和头发颜色；不能分离的区域不开放。
- 纯色由“属性 JSON -> 语义 mask -> 确定性材质变换”实现。FLUX.2 只用于
  mask 内的织物、牛仔、皮革等受控纹理细节，不负责改变几何，也不负责最终
  精确颜色。
- 动物使用 FLUX.2 生成受控参考图，Pixal3D 是默认 image-to-3D 后端。
  Hunyuan3D 及其衍生物继续保持 `technical_spike_only` 或 `rejected`，不能
  进入正式训练/评测资产。
- `size=large` 是语义属性，不是实际厘米。目标尺寸来自版本化品种配置，
  最终尺寸必须由网格/骨架/UE 代码测量后写入 `physical_measurements`。

### 1.1 一页式工作流

目前的权威工作流不是“写一句 prompt 然后相信生成结果”，而是下面这条可认证
的数据链：

```text
审计基础模板/品种
  → attribute_profile（允许采样什么）
  → 平衡采样器（一次采出完整绝对属性）
  → instance_request（本实例准备是什么）
  → route compiler
      ├─ animal: 完整 prompt → 一次 FLUX.2 → 2D gate → Pixal3D
      └─ human: base_avatar_id → MaterialEditPlan → mask 内确定性改色
  → 静态/绑定/Walking/Idle/UE/音频 QA
  → physical_measurements（生成后的实际测量）
  → source_asset_v2（实际得到且有证据的资产）
  → lineage-safe split + realized QA + scene source pool
  → Apartment 场景选择具体音频、轨迹和时序
```

各字段真正控制的内容如下：

| JSON 内容 | 来源 | 控制对象 | 不能被它证明或修改的内容 |
|---|---|---|---|
| `fixed_attributes` | 品种/模板审计 | prompt 中固定描述，或 Rocketbox 固定几何语义 | 不能被随机采样器修改 |
| `sampled_attribute_domains` | 经审核的 profile | 定义合法离散空间，每项 1--3 值 | 不是某个实例的结果 |
| `sampled_attributes` | 平衡采样器 | 一个实例的完整绝对标签 | 不能证明模型真的实现了标签 |
| `generation_plan.prompt` | profile compiler | 动物的一次 FLUX.2 图生图 | 不直接控制真实厘米、骨架或音频 |
| `material_edit_plan` | profile compiler | Rocketbox 的模板、语义 mask 和确定性颜色参数 | 不生成新袖长、裤长、鞋型或附件几何 |
| `target_physical_profile` | 版本化品种/模板配置 | 给出后处理要达到的目标尺寸 | 不是实测值 |
| `physical_measurements` | 网格、骨架和 UE 回读 | 记录真实 bounds、肩高、actor scale 和落地结果 | 只能在生成后写入 |
| `acoustic_profile` | 物种/角色声音配置 | 限定可选声音类别和语料池 | 不选择某一条实际波形 |
| 场景 audio manifest | 场景编译器 | 选择具体音频、时间、增益、重复事件和空间位置 | 不修改资产外观身份 |

因此，属性 JSON 是生成意图和可审计标签的来源；`source_asset_v2` 和媒体 QA 才是
实现证据。两者不能合并成一个会被各阶段反复改写的文件。

## 2. “声源资产”包含什么

一个 AVEngine 声源资产不是只有 GLB。它至少由以下内容共同构成：

1. 绝对语义属性：物种、品种或人类基础角色、颜色、体型等。
2. 视觉资产：参考图、PBR 网格、骨架、Walking/Idle 动画和 UE Blueprint。
3. 物理信息：方向、地面、运行缩放、实际尺寸、碰撞和声源高度。
4. 声音能力：声音类别、允许的语料池和许可证要求。
5. 来源信息：输入、模型 revision、prompt、seed、代码、许可证及所有哈希。
6. QA 与状态：静态、动画、UE、音频、媒体审核和最终资产分类。

属性 JSON 负责生成和约束第 1 项，并为其余阶段提供输入。它不能替代生成后
测量、许可证审核和媒体 QA。

## 3. 五层 JSON，不混淆生命周期

实际落地时不是把一个 JSON 从“计划中”不断改写成“已完成”，而是形成一条
只追加、不覆盖的证据链。当前实现使用下面五层对象：

| 层级 | 对象/文件 | 由谁产生 | 下游只读取什么 | 是否含真实生成结果 |
|---|---|---|---|---|
| 1 | `attribute_profile` | 模板/品种审计人员与 profile validator | 合法属性域、固定项、路线、模型 revision、mask、目标物理配置 | 否 |
| 2 | `instance_request` | 确定性平衡采样器 | 一个实例的完整绝对属性、seed、prompt 或 MaterialEditPlan | 否 |
| 3 | `execution_job` | request compiler | 按 `route` 分发给 FLUX.2/Pixal3D 或 Rocketbox 执行器 | 否 |
| 4 | `source_asset_v2` | 各生成/测量/QA 阶段通过后新建 | 实际工件、测量、骨架、声音能力、许可证和 QA | 是 |
| 5 | dataset/scene manifests | 数据集编译器与场景生成器 | split、QA 对、可选声源、具体音频与时序 | 是 |

权威关系是：profile 决定“允许生成什么”，request 决定“这次准备生成什么”，
`source_asset_v2` 证明“实际上得到了什么”。prompt 不是语义真值，模型输出也
不能反过来静默修改 request；实现不符合 request 时应拒绝该 attempt。

### 3.1 属性配置 `attribute_profile`

这是对某个动物品种或 Rocketbox 基础角色的“允许空间”定义，只创建一次，
经过审计后固定。它包含：

- 固定属性；
- 可采样属性及各自最多三个合法值；
- 属性组合约束；
- prompt 中使用的自然语言映射；
- 目标物理尺寸配置；
- 骨架/动画和声音类别配置。

动物示意：

```json
{
  "schema": "avengine_attribute_profile_v1",
  "profile_schema_id": "dog_golden_retriever_v1",
  "asset_class": "animal",
  "taxonomy": {
    "species": "dog",
    "breed": "golden_retriever"
  },
  "fixed_attributes": {
    "life_stage": "adult",
    "coat_length": "long",
    "coat_pattern": "solid",
    "ear_shape": "drop",
    "tail_shape": "feathered"
  },
  "sampled_attribute_domains": {
    "size": ["small", "medium", "large"],
    "coat_color": ["light_golden", "golden", "dark_golden"],
    "body_build": ["slim", "standard", "stocky"]
  },
  "target_physical_profile_id": "dog_golden_retriever_physical_v1",
  "rig_profile_id": "quadruped_dog_v1",
  "acoustic_profile_id": "dog_vocalization_v1"
}
```

当前已实际执行的 Rocketbox 男性 profile 节选：

```json
{
  "schema": "avengine_attribute_profile_v1",
  "profile_schema_id": "rocketbox_adults_male_adult_01_material_v1",
  "asset_class": "human",
  "base_avatar_id": "rocketbox_adults_male_adult_01",
  "fixed_attributes": {
    "gender": "male",
    "life_stage": "adult",
    "top_geometry": "plaid_short_sleeve_shirt",
    "bottom_geometry": "shorts",
    "footwear_geometry": "shoes",
    "headwear": "absent",
    "eyewear": "absent",
    "accessories": "base_locked"
  },
  "sampled_attribute_domains": {
    "top_color": ["blue", "green", "burgundy"]
  },
  "locked_attributes": [
    "identity",
    "body_geometry",
    "skin",
    "eyes",
    "garment_geometry",
    "headwear",
    "eyewear",
    "accessories"
  ]
}
```

这个 profile 的三种颜色已经实际生成。其他 Rocketbox 角色仍必须来自各自的
角色和 mask 审计，不能根据文件名或职业名称猜测，也不能复用男性 01 的 shirt
mask。某个模板没有独立头发或裤子 mask 时，相应字段就不能出现在
`sampled_attribute_domains`。

### 3.2 生成请求 `instance_request`

批量采样器从一个已认证的 `attribute_profile` 中生成不可变请求。请求表示
“准备生成什么”，此时没有伪造的实际厘米或 QA 结果。

```json
{
  "schema": "avengine_instance_request_v1",
  "instance_id": "dog_golden_8a4b91c2d147",
  "profile_schema_id": "dog_golden_retriever_v1",
  "profile_sha256": "...",
  "sampler": {
    "algorithm": "balanced_quota_sampler_v1",
    "batch_seed": 20260713,
    "sample_ordinal": 184
  },
  "fixed_attributes": {
    "species": "dog",
    "breed": "golden_retriever",
    "life_stage": "adult",
    "coat_length": "long",
    "coat_pattern": "solid",
    "ear_shape": "drop",
    "tail_shape": "feathered"
  },
  "sampled_attributes": {
    "size": "large",
    "coat_color": "light_golden",
    "body_build": "standard"
  },
  "target_physical_profile": {
    "profile_id": "dog_golden_retriever_physical_v1",
    "size_category": "large",
    "target_shoulder_height_cm": 60.0,
    "tolerance_cm": 4.0
  },
  "generation_plan": {
    "route": "flux2_pixal3d_animal_v1",
    "prompt_template_id": "quadruped_i2i_v1",
    "prompt": "A large adult golden retriever with a standard anatomically normal body, light golden long fur, a solid coat pattern, drop ears and a feathered tail. Preserve the canonical quadruped standing pose, side viewpoint, body orientation, visible separated legs and paws, tail separation and plain background.",
    "negative_prompt": "cropped body, sitting, lying down, merged legs, hidden paws, fused tail, extra limbs, background geometry, text",
    "generation_seed": 184
  }
}
```

`instance_id` 由规范化请求内容的 SHA-256 前缀产生。规范化输入至少包含
`profile_schema_id`、profile 哈希、batch seed、ordinal 和完整绝对属性，
因此同一请求可以复现，同时不会把实例描述成“由另一个个体修改而来”。

### 3.3 生成后资产 `source_asset_v2`

每个阶段只能向自己的新产物目录原子发布记录。最终资产 manifest 关联原始
请求并增加实际工件、测量、声音能力和 QA：

```json
{
  "schema": "source_asset_v2",
  "asset_id": "dog_golden_8a4b91c2d147",
  "request_sha256": "...",
  "asset_class": "animal",
  "taxonomy": {
    "species": "dog",
    "breed": "golden_retriever"
  },
  "semantic_attributes": {
    "size": "large",
    "coat_color": "light_golden",
    "body_build": "standard",
    "life_stage": "adult",
    "coat_length": "long",
    "coat_pattern": "solid"
  },
  "physical_measurements": {
    "status": "measured",
    "method": "canonical_rig_and_ue_measurement_v1",
    "runtime": {
      "actor_scale": 0.643,
      "shoulder_height_cm": 60.1,
      "head_height_cm": 78.4,
      "body_length_cm": 97.6
    }
  },
  "rig": {
    "profile_id": "quadruped_dog_v1",
    "actions": ["Walking", "Idle"],
    "front_axis": "positive_x"
  },
  "acoustic_profile": {
    "profile_id": "dog_vocalization_v1",
    "default_event_class": "dog_bark",
    "allowed_event_classes": ["dog_bark", "dog_growl", "silent"]
  },
  "provenance": {
    "flux_revision": "...",
    "pixal3d_revision": "0b31f9160aa400719af409098bff7936a932f726",
    "input_sha256": "...",
    "prompt_sha256": "...",
    "output_glb_sha256": "..."
  },
  "qa": {
    "reference_2d": "passed",
    "static_mesh": "passed",
    "binding": "passed",
    "walking": "passed",
    "idle": "passed",
    "ue_import_readback": "passed",
    "apartment_media": "passed"
  },
  "state_classification": "research_candidate"
}
```

生成请求不被就地改写为“成功”。失败尝试保留自己的 attempt ledger；通过
所有要求的产物再生成新的资产 manifest。这样可以保存失败证据并避免一个
JSON 在多进程中被反复覆盖。

上面的 JSON 是便于阅读的字段节选。严格字段契约的代码权威是
`external/SPEAR/tools/controlled_source_asset_schema.py`。工件记录统一使用
`{root_id, path, sha256, size_bytes}`，不能只保存一条依赖当前机器目录结构的
绝对路径。

## 4. 属性 JSON 如何生成

### 4.1 先审计模板或品种配置

动物配置由品种定义，Rocketbox 配置由具体 `base_avatar_id` 定义。创建配置时：

1. 写入固定属性；
2. 只开放可以由当前路线稳定实现的属性；
3. 每个开放属性检查取值数为 1--3；
4. 写入不合法组合约束；
5. 锁定配置文件和所有依赖工件的哈希；
6. 配置通过验证后才进入采样器。

禁止使用一张“所有动物通用颜色表”。例如：

| Profile | 合法颜色属性示例 |
|---|---|
| `dog_golden_retriever_v1` | `light_golden / golden / dark_golden` |
| `dog_beagle_v1` | `light_tricolor / standard_tricolor / dark_tricolor` |
| `dog_pug_v1` | `fawn / apricot / black` |
| `cat_siamese_v1` | 暹罗猫专用重点色配置，不能采样虎斑条纹 |
| `cat_tabby_v1` | 该模板实际支持的三种底色/条纹组合 |

这些值必须经过品种合理性和生成可行性检查后才能成为正式 profile；表中名称
不是对尚未审核 profile 的自动批准。

### 4.2 使用带配额的平衡采样

批量任务不采用简单的独立随机，因为它可能生成大量 `medium`，却几乎没有
`small` 或 `large`。采样器按 batch 对每个开放属性建立数量接近的配额，
然后用固定 seed 打乱组合。

采样器必须记录：

- profile ID 和 profile SHA-256；
- batch seed、sample ordinal 和算法版本；
- 完整 fixed/sample 属性；
- 约束验证结果；
- 规范化请求哈希。

如果某一组合违反品种或模板约束，采样器确定性地选择下一个合法组合，而不是
临时删除字段或让 prompt 自由发挥。

### 4.3 自动编译 prompt 或材质计划

采样结果不能交给人工重新写 prompt。编译器必须检查：

- 每个 sampled 属性在生成计划中恰好出现一次；
- 所有影响外观/姿态的 fixed 属性均被补入；
- 不包含 profile 未声明的属性；
- pose、相机、方向、肢体分离和非目标区域锁定词完整；
- prompt、negative prompt 和模板 revision 都被哈希记录。

这里有一个重要边界：profile 是经过审计后由人维护的“合法属性空间”，实例
JSON 则全部由程序生成。批量运行时不让人或语言模型临时补字段，实际算法是：

```text
读取并认证 profile
  → 枚举/打乱所有合法绝对属性组合
  → 按每个属性的配额选择组合
  → 由 value_labels/template 自动编译完整 prompt 或 MaterialEditPlan
  → 计算 request SHA-256 和 instance_id
  → 写入只读 instance_requests.json / execution_jobs.json
```

因此，“浅色金毛”不是由一条 `make_it_lighter` 操作得到。它是在本实例创建时
直接采样为 `coat_color=light_golden`；其余 `size`、`body_build` 和固定品种属性
也会在同一次 request 中完整出现。人类同理，`top_color=green` 是该 Rocketbox
实例的绝对标签，不保存“从蓝色改成绿色”的身份历史。

`physical_attributes` 也不是由 FLUX 猜出来：语义 `size` 来自 profile 的离散域，
目标厘米来自版本化 `target_physical_profiles`，真实厘米只由生成后的网格、骨架
和 UE 回读产生。三者分别保留，禁止互相冒充。

### 4.4 当前可执行的 profile -> request 编译入口

已落地的 profile 根目录是：

```text
external/SPEAR/data/controlled_source_attributes_v1/profiles/
├── animal/
│   ├── cat_siamese_v1.json
│   ├── cat_tabby_v1.json
│   ├── dog_beagle_v1.json
│   ├── dog_golden_retriever_v1.json
│   └── dog_pug_v1.json
└── human/
    └── rocketbox_adults_male_adult_01_material_v1.json
```

这六个 profile 当前均为 `research_candidate`。五个动物 profile 的参考图来源
状态仍是 `legacy_unknown`，目标体型参考也是 `provisional`；它们可以用于管线
验证，但不满足正式注册门槛。Rocketbox 男性 profile 的 FBX、作者身高清单和
上衣 mask 已做哈希认证，但目前只开放 `top_color`。

生成一批不可变输入的命令是：

```bash
cd /data/jzy/code/AVEngine
/data/jzy/miniconda3/envs/spear-env/bin/python \
  external/SPEAR/tools/build_controlled_source_asset_inputs.py \
  --profile external/SPEAR/data/controlled_source_attributes_v1/profiles \
  --count-per-profile 9 \
  --seed 20260713 \
  --plan-id controlled_source_profiles_v3 \
  --split-salt avengine-controlled-source-v1 \
  --max-qa-pairs-per-split 1000 \
  --output-dir \
    external/SPEAR/tmp/controlled_source_asset_input_v1/all_profiles_20260713_v3
```

该命令在发布任何 JSON 前会读取真实输入文件，核对大小和 SHA-256。默认根为
`spear_repo=external/SPEAR` 和
`rocketbox_0943055=/data/datasets/rocketbox/Microsoft-Rocketbox`；其他根必须用
`--artifact-root ROOT_ID=PATH` 显式传入。输出目录必须不存在，工具拒绝覆盖。

输出文件的职责如下：

| 文件 | 内容 | 可否直接称为数据集资产 |
|---|---|---|
| `profile_snapshot.json` | 六个 profile、profile SHA-256、所有依赖工件的现场认证结果 | 否 |
| `instance_requests.json` | 绝对属性、采样 provenance、目标物理配置、生成计划及 request SHA-256 | 否 |
| `execution_jobs.json` | 按两个 route 分组的执行队列 | 否 |
| `qa_pair_plan.json` | 从绝对属性推导的候选问题；答案状态仍为 pending | 否 |
| `generation_plan.json` | lineage split、请求索引、计划 QA 对和自动泄漏检查 | 否 |

2026-07-13 的只读 canary 结果为 6 个 profile、54 个 request：45 个动物
FLUX.2/Pixal3D job、9 个 Rocketbox material job、57 个计划 QA pair。profile
snapshot SHA-256 是
`4297e52e5a6e5ee399602afbda67b5abf3d347c25f24392a8ef5758e2d95f66b`。
五个动物 profile 的三个开放属性均为精确 `3/3/3` 覆盖。

这里的 `count-per-profile=9` 是跨路线编译 canary，不是正式资产数量承诺。
当前 Rocketbox profile 只有三种唯一 `top_color`，所以 9 个 request 中只有 3
种唯一视觉材质。正式执行/注册必须按
`base_avatar_id + sampled_attributes + material-plan revision` 去重，不能把相同
材质的不同 ordinal/seed 冒充新个体。动物 profile 有三个属性，当前 9 个请求
是九个不同组合。

## 5. 动物分支：属性 JSON 控制 FLUX.2 和 Pixal3D

```text
品种 attribute_profile
        ↓
平衡采样完整绝对属性
        ↓
instance_request + 自动完整 prompt
        ↓
一次 FLUX.2 Klein 图生图
        ↓
2D 属性/姿态/肢体分离 QA
        ↓
Pixal3D PBR GLB
        ↓
静态 QA → LOD → 对应物种骨架/权重
        ↓
Walking + Idle → GLB 回读 → UE 导入回读
        ↓
实际尺寸/地面/方向测量
        ↓
物种声音映射 → Apartment 视频/音频 QA
        ↓
source_asset_v2 注册
```

FLUX.2 接收的是完整视觉描述，但不接收真实厘米作为可靠控制信号。
`small/medium/large` 控制视觉体型；Pixal3D 后处理和 UE scale 控制真实世界
尺寸。2D 结果只要有一个 sampled 属性未实现，或固定姿态/四肢分离被破坏，
该 attempt 就进入 `rejected`，不进入昂贵的 3D 和动画阶段。

每个动物品种映射到经过验证的骨架/动作族。不能为了批量通过，把马、狗、猫
都套同一套动作；尚无可靠动作族的品种保持候选或拒绝，不伪装成已支持。

### 5.1 `execution_job` 怎样变成一次 FLUX.2 调用

动物执行器只消费
`execution_jobs.json.routes.flux2_pixal3d_animal_v1[]`。每个 job 已包含：

- `instance_id` 和被哈希认证的 `request_sha256`；
- reference image 的 `root_id/path/sha256/size_bytes`；
- 唯一的完整 `prompt`、`negative_prompt` 和 `generation_seed`；
- 固定的 FLUX.2、Pixal3D、DINO revision；
- 目标物理配置、骨架动作族和声音能力。

执行器必须按以下顺序工作：

1. 回查 `instance_requests.json`，验证 job 的 request SHA-256 和 profile SHA-256；
2. 解析 named artifact root，再次认证 reference image，禁止按文件名猜输入；
3. 每个 GPU 常驻加载一次 job 指定 revision 的 FLUX.2 Klein；
4. 对一个 request 只执行一次 image-to-image，输入完整 prompt、negative prompt、
   seed 和基准参考图；
5. 写入新的 attempt 目录，保存输入/输出图哈希、模型 revision、参数、耗时和
   失败状态；
6. 先做 2D 属性、轮廓、姿态、四肢分离审核；只有通过的图才进入 Pixal3D；
7. Pixal3D 继续使用 pinned persistent worker，输出 PBR GLB 后依次进行静态、
   LOD、绑定、Walking/Idle、GLB 回读和 UE 审核；
8. 在网格/骨架/UE 中测量实际尺寸，完成音频能力与许可证记录后，另行创建
   `source_asset_v2`。

例如，金毛 request 的 `coat_color=dark_golden`、`size=large` 和
`body_build=stocky` 都会由 profile 中的 `value_labels` 展开进同一个 prompt。
因此每个属性都明确出现一次，同时 pose guard 继续锁定侧视、四足站立、腿脚
分离和纯背景。执行器不得自己追加“更可爱”“更蓬松”等 profile 外属性。

`size` 在 FLUX.2 中只控制视觉相对体型。真实厘米由
`target_physical_profile.target_value_cm` 与生成后测量计算统一 runtime scale，
不能把 prompt 中的 `large` 直接当成真实肩高。

### 5.2 当前动物执行边界

规范化执行适配器已经落地。45 个已认证动物 request 已全部实际执行：先用
5 个计划 QA pair、共 10 个实例完成端到端静态 canary，再对未执行的 35 个
request 继续运行同一条不可覆盖的证据链。最终有 42 个静态合格候选、2 个在
2D 阶段拒绝、1 个在 Pixal 静态阶段拒绝；失败实例保留原 request 和失败原因，
没有用新 seed 或其他外观偷偷替换。

最初 10 个静态 canary 的权威证据如下：

| 阶段 | 权威 manifest / 媒体 | 当前结果 |
|---|---|---|
| FLUX.2 | `external/SPEAR/tmp/controlled_source_asset_execution_v1/animal_flux2_qa_canary_v1_20260713/flux2_batch_manifest.json` | 4 GPU 常驻 worker；10/10 候选生成；batch `0f59efea63083eab096f4751d2d21f026a09395f37f6d7e734fa0b0cfa4d175c` |
| 2D 属性审核 | `external/SPEAR/tmp/controlled_source_asset_execution_v1/animal_flux2_qa_canary_v1_20260713_reviews_v1/review_batch_manifest.json` | 10 approved、0 rejected；review `4c16d6742750c7507f12f78761c7f3fea8030a000a095e58cbd34ad7f31cf9ed` |
| ISNet/Pixal 输入 | `external/SPEAR/tmp/controlled_source_asset_execution_v1/animal_pixal_inputs_v1_20260713/pixal_inputs_manifest.json` | 10/10 认证为 1024 RGBA；manifest `30962493b664d360d7f2e44bd5ce50e6b1c40f03d539c08bead028bfa872434f` |
| Pixal3D PBR GLB | `external/SPEAR/tmp/controlled_source_asset_execution_v1/animal_pixal_qa_canary_v1_20260713/pixal_batch_manifest.json` | 3 个可用 GPU 常驻 worker；10/10 GLB2/PBR 回读通过；batch `f2889b3acd95ee06925fed733b0bbca7ba0fd925b41b75f5a9b43a7fd64d9ab0` |
| 静态多视图 | `external/SPEAR/tmp/controlled_source_asset_execution_v1/animal_pixal_static_reviews_v3_20260713/static_review_batch_manifest.json` | 10/10 Front/Back/Side/Top/接触图生成；batch `6b10adce7214b40dc62f932b2ceb67c70faff56eeade0b18f20d3fb859611a7f` |
| 静态决定 | `external/SPEAR/tmp/controlled_source_asset_execution_v1/animal_pixal_static_decisions_v1_20260713/static_decision_batch_manifest.json` | 10/10 `approved_for_lod_and_binding`；size 全部延后到 metric 3D；decision `aa9bc95bbced6f9e055fefb3e692366405901856550c43b39e32f011f386f1e3` |
| 候选资产注册 | `external/SPEAR/tmp/controlled_source_asset_execution_v1/animal_source_assets_v2_20260713_v1/registry_manifest.json` | 10 个 `research_candidate`；registry `20e72c8b0c4dcd0ba39d4f05aa553abce093d47c4e60f7a961d62df4eb9b1b07` |

其余 35 个 request 的实际执行结果如下：

| 阶段 | 权威 manifest / 媒体 | 实际结果 |
|---|---|---|
| FLUX.2 | `external/SPEAR/tmp/controlled_source_asset_execution_v1/animal_flux2_remaining35_v1_20260713/flux2_batch_manifest.json` | 4 GPU 常驻 worker；35/35 候选生成；batch `be5ebf9c9307aebe20ccc571a5a8e69ae9ddedc816539de392b98747724eafbd` |
| 2D 属性审核 | `external/SPEAR/tmp/controlled_source_asset_execution_v1/animal_flux2_remaining35_reviews_v1_20260713/review_batch_manifest.json` | 33 approved、2 rejected；两个虎斑候选分别出现分叉尾尖和完整双尾；review `201371d67b0557c9cfa6d3aa6e72237337e80aef0ec118101d437f9929bb5dbe` |
| ISNet/Pixal 输入 | `external/SPEAR/tmp/controlled_source_asset_execution_v1/animal_pixal_inputs_remaining33_v1_20260713/pixal_inputs_manifest.json` | 33/33 认证为 1024 RGBA；manifest `2e4fb4bf701f903d49c99c1f2f264dc98e8f3cd331c8a15230bf7290af6d5579` |
| Pixal3D PBR GLB | `external/SPEAR/tmp/controlled_source_asset_execution_v1/animal_pixal_remaining33_v1_20260713/pixal_batch_manifest.json` | 4 GPU 常驻 worker；33/33 GLB2/PBR 回读通过；batch `e99ea28258a384644c48702ac5ddf7ed851f718a28ac26aa7a958808ac646009` |
| 静态多视图 | `external/SPEAR/tmp/controlled_source_asset_execution_v1/animal_pixal_static_reviews_remaining33_v1_20260713/static_review_batch_manifest.json` | 33/33 Front/Back/Side/Top/Quarter/contact sheet；batch `48a08de55f87e135c4b66bb715be671b1a8299fab494d864ae74e01b504be758` |
| 静态决定 | `external/SPEAR/tmp/controlled_source_asset_execution_v1/animal_pixal_static_decisions_remaining33_v1_20260713/static_decision_batch_manifest.json` | 32 approved、1 rejected；拒绝的比格犬同时有短尾主体和一整段悬浮白尾；decision `402203c2273a256d8779735c2580cc5069e5d3725e8ad1959fc4a1d28ed99a18` |
| 候选资产注册 | `external/SPEAR/tmp/controlled_source_asset_execution_v1/animal_source_assets_remaining33_v2_20260713/registry_manifest.json` | 只注册 32 个 approved `research_candidate`；registry `46617330f3846a938e7ee6adc9f9833467b01efdc338a33a31c41d0e85e378bc` |

按 profile 分组的 33 个静态总览位于
`external/SPEAR/tmp/controlled_source_asset_execution_v1/animal_pixal_static_reviews_remaining33_overview_v1_20260713/`。
所有审核决定绑定到对应 `review_sha256`，并由
`external/SPEAR/data/controlled_source_attributes_v1/reviews/animal_pixal_static_remaining33_20260713_v1.json`
保存。注册器要求决策完整覆盖全部 Pixal attempt，但只注册 approved 子集；
单个静态失败不会再拖垮整批注册。

总览图是
`external/SPEAR/tmp/controlled_source_asset_execution_v1/animal_pixal_static_reviews_v3_20260713_overview/all_static_contact_sheets.png`。
静态门禁确认 10 个都是完整、可识别、四肢分开的猫/狗候选；已保留的注意项是
暹罗猫少量游离胡须片、一个比格犬可见平面感以及金毛分层毛片。原始 GLB
约 92.3--99.0 万三角面，尚未 LOD/绑定。Pixal 原材质的
`metallicFactor=1.0` 没有被覆写；审核渲染只应用与现有 UE 动物导入一致的
non-metallic/roughness 预览，并将该预览参数写入审核记录。

这些记录证明的是“属性 JSON → FLUX.2 → 2D gate → ISNet → Pixal3D → 静态
source_asset_v2”已对完整 45-request canary 走通，不证明动画和场景已经通过。
42 个候选的
`physical_measurements`、LOD、物种骨架绑定、Walking、Idle、UE、Apartment、
音频和正式 rights clearance 仍是后续门禁，所以它们不会进入场景白名单。

### 5.3 当前静态 canary 耗时

| 阶段 | 并行方式 | 实测 |
|---|---|---|
| FLUX.2 模型加载 | 4 GPU，各加载一次 | 每 worker 约 107.5--107.9 秒 |
| FLUX.2 单候选推理 | 4 GPU 常驻 worker | 20.9--25.9 秒，平均 22.95 秒 |
| Pixal3D 模型加载 | GPU 1--3，各加载一次 | 约 118.6--120.8 秒 |
| Pixal3D 单候选推理+GLB 导出 | 3 GPU 常驻 worker | 97.5--372.6 秒，平均 186.17 秒 |
| Pixal3D 10 个候选批次 wall time | 3 GPU | 914.93 秒（约 15.25 分钟，含加载） |
| Pixal3D 剩余 33 个候选模型加载 | 4 GPU，各加载一次 | 126.26--131.87 秒 |
| Pixal3D 剩余 33 个单候选推理+GLB 导出 | 4 GPU 常驻 worker | 95.60--362.62 秒，平均 183.82 秒 |
| Pixal3D 剩余 33 个批次 wall time | 4 GPU | 1818.58 秒（约 30 分 19 秒，含加载和最终回读） |

这说明当前静态管线的主要计算瓶颈是 Pixal3D，而不是 JSON 编译、注册或 QA
数据集构建。Pixal 的扩散采样阶段 GPU 利用率高，而参数化、UV 和 GLB
finalize 阶段主要占 CPU，所以瞬时 GPU 利用率低不代表任务停止。本批使用的
静态 round-robin 分片还出现尾部不均衡：GPU 2 先结束时其他 GPU 仍有任务。
批量生产必须继续使用 persistent worker，不能每个实例重新加载模型，并应改为
共享可抢占任务队列，让先完成的 worker 自动领取下一个 job。更完整的
人类/动物/UE 耗时权威表继续见
`docs/pipeline_timing_and_scaling_audit.md`。

## 6. 人类稳定分支：属性 JSON 控制 Rocketbox

```text
115 个 Rocketbox base_avatar 审计目录
        ↓
按 base_avatar_id 选择固定身份/身体/服装几何
        ↓
读取该模板允许的材质属性域
        ↓
平衡采样颜色 → instance_request
        ↓
编译 MaterialEditPlan
        ↓
纯色：语义 mask + 确定性材质变换
纹理细节：mask 内 FLUX.2 → 合成/烘焙回原 UV
        ↓
证明几何/骨架/非目标像素未变
        ↓
直接复用原生 Walking + Standing_Idle
        ↓
UE 导入回读、Apartment 媒体与语音 QA
        ↓
source_asset_v2 注册
```

这里“JSON 控制 Rocketbox”的准确含义是：

- `base_avatar_id` 选择一个已经存在的 Rocketbox 角色；
- `fixed_attributes` 描述该角色原有几何，不允许实例采样器修改；
- `sampled_attributes` 被编译为具体语义 mask 和颜色参数；
- 输出仍是同一个 mesh、骨架、UV 和服装款式的新材质实例。

它不表示 Rocketbox 会生成一件不存在的衣服。短袖/长袖、短裤/长裤和职业
制服差异通过选择不同的基础角色获得。眼睛、帽子、眼镜、饰品和其他附件在
第一版全部锁定，不进入随机采样。

纯色不需要为了“使用过 FLUX”而强行经过生成模型。确定性变换更容易保持格子、
褶皱、阴影/PBR 细节以及非目标区域。只有需要新增受控织物细节时，才在已批准
mask 内调用 FLUX.2，并在最终合成时保证 mask 外字节不变。

Rocketbox 已封存 baseline 只读。所有实例必须从已认证输入复制到新的 staging/
output 目录，禁止在 baseline 中就地改色或导出。

### 6.1 `execution_job` 怎样变成 Rocketbox 材质实例

Rocketbox 执行器只消费
`execution_jobs.json.routes.rocketbox_material_v1[]`。以当前男性 profile 的
`top_color=burgundy` 为例，编译后的 MaterialEditPlan 已明确给出：

```json
{
  "base_avatar_id": "rocketbox_adults_male_adult_01",
  "geometry_changes_allowed": false,
  "edits": [
    {
      "attribute": "top_color",
      "value": "burgundy",
      "semantic_mask": "shirt_main_color",
      "operation": "replace_base_color_preserve_pbr_detail_v1",
      "target_srgb_u8": [122, 48, 69],
      "target_srgb_hex": "#7A3045"
    }
  ]
}
```

实际执行顺序必须是：

1. 认证 profile、request、原 FBX、mask registry 和作者身高 inventory；
2. 计算唯一材质 variant key 并去重；
3. 将原 FBX/纹理复制到全新的 staging 目录，baseline 永远只读；
4. 从 registry 精确定位 `shirt_main_color`，执行确定性
   `replace_base_color_preserve_pbr_detail_v1`；
5. 验证 mask 外像素、mesh、UV、骨架、Walking 和 Standing Idle 均未变化；
6. 导出/回读并复用已验证的 Rocketbox 原生 runtime 与 UE 导入路径；
7. 保持 `actor_scale=1.0` 和作者身高，完成人类语音/Apartment 媒体 QA；
8. 只有实际工件和全部状态有证据时，才创建 `source_asset_v2`。

当前 `flux_texture_detail.enabled=false`，所以上衣纯色变体不会调用 FLUX.2。
未来若某个模板确实需要织物细节，必须先把该模板的语义 mask 注册到 profile，
FLUX.2 也只能看见/修改 mask 内区域，最终仍需烘焙回原 UV。不能用这一例外
修改袖长、裤长、鞋型、帽子或眼镜几何。

### 6.2 人类 profile 从哪里来

人类 profile 不是让语言模型根据职业名称猜出来的。每个 `base_avatar_id` 都要
从 115 角色 inventory 中读取作者身份、身高、骨架和原服装，再由 mask 审计
决定哪些颜色字段可以开放。没有独立 mask 的区域就是 fixed/locked 属性。

目前只有 `rocketbox_adults_male_adult_01_material_v1` 达到这种 profile 输入
认证程度，并且只证明上衣颜色。女性、儿童和职业角色虽然都已经能以原材质
在 UE 中 Walking/Idle，但它们的颜色属性必须在各自 mask 审计完成后逐个新增
profile；不能把男性 shirt mask 直接套到其他角色。

### 6.3 当前 Rocketbox 受控批次的实际结果

`rocketbox_adults_male_adult_01_material_v1` 已经完整走过规范化输入、材质执行、
原生 Walking/Standing Idle runtime、UE 用 in-place runtime、`source_asset_v2` 和
候选数据集编译，不再只是设计说明：

| 阶段 | 只读 manifest | 结果 |
|---|---|---|
| authenticated preflight | `external/SPEAR/tmp/controlled_source_asset_execution_v1/all_profiles_20260713_v3_preflight_v1/execution_preflight.json` | 9 个 request 去重为 3 个唯一材质 job |
| 材质执行 | `external/SPEAR/tmp/controlled_source_asset_execution_v1/rocketbox_material_v1_20260713_v1/material_batch_manifest.json` | blue/green/burgundy 全通过，6 个重复 request 被抑制 |
| 原生 runtime | `external/SPEAR/tmp/controlled_source_asset_execution_v1/rocketbox_runtime_handoff_v1_20260713_v1/runtime_handoff_manifest.json` | 3/3 含 Walking + Standing_Idle，mesh/skin/action 合同相同 |
| UE runtime 归一化 | `external/SPEAR/tmp/controlled_source_asset_execution_v1/rocketbox_ue_runtime_handoff_v1_20260713_v1/ue_runtime_handoff_manifest.json` | 3/3 metric、in-place、80 joints，材质保持 |
| 资产注册 | `external/SPEAR/tmp/controlled_source_asset_execution_v1/rocketbox_source_assets_v2_20260713_v1/registry_manifest.json` | 3 个候选 `source_asset_v2`，不把重复 seed 注册为新个体 |
| 候选数据集 | `external/SPEAR/tmp/controlled_source_asset_execution_v1/rocketbox_candidate_dataset_v3_20260713/dataset_manifest.json` | 3 个资产、3 个 QA pair、每个都只以 `top_color` 为已证实 QA 属性；同一 lineage 同 split；scene readiness 为 0/3 |

关键 manifest 哈希分别是：material
`d9a468875b542ed905753e9a36e67ff0524bf1dc4aef77892962e334b860a36f`、native
runtime `b45aa2fa3aa8d953f1a1bd0731f3de15ed781e6b939899197f8a5c16a7cd8e0b`、
UE runtime `63572f5a1da5cce1ff08df8af158a527460efc431a11b9404576e843010fd61a`
和 registry
`f1a2f3884ff5715b25a755e3bebd709fd43948c3e3314332936be376e2b6b51e`。

这里的三个资产仍是 `research_candidate`：材质、静态、绑定、Walking 和 Idle
已有证据，但 UE Editor 实际导入回读、Apartment 媒体和具体语音 QA 仍是
`pending`。所以它们可以用于验证属性到 QA 的数据合同，却还不能被 Apartment
生成器当成“全部通过”的正式声源。

## 7. Route-2 人类新几何分支

如果以后确实需要当前 115 个 Rocketbox 模板不存在的新服装几何，该请求不是
普通 instance 扩增，而是一个新的模板资格认证任务：

```text
完整属性 JSON → FLUX.2 参考图 → Pixal3D → TokenRig
→ 静态绑定 QA → Walking/Idle → UE/Apartment QA
→ 通过后成为新的 qualified template revision
```

不能先批量随机新几何，再默认所有结果共享一次审核。当前第一版人类随机空间
不包括眼睛和附件，也不依赖这条高风险分支。

已有 `route2_controlled_human_instance_space_v3` 是此前更宽的研究合同，其中含
帽子、眼镜和超过三个颜色值。它保留为历史/技术证据，不是本文冻结后的第一版
批量属性采样器，不能据此声称那些模板已经可用。

## 8. `physical_measurements` 如何产生

物理信息分成三类，不能混写：

| 生命周期 | 字段 | 获得方式 | 用途 |
|---|---|---|---|
| 采样前 | profile 中的 `target_physical_profiles` | 许可证快照可追溯的品种尺寸资料，或明确标为 `provisional` 的管线校准值 | 定义 small/medium/large 对应的目标和容差 |
| 采样时 | request 中的 `target_physical_profile` | 由本次绝对 `size` 值确定性选择 | 告诉 3D/UE 后处理要达到什么，不冒充测量结果 |
| 生成后 | `source_asset_v2.physical_measurements` | 网格、语义骨骼和 UE bounds 代码实测 | 数据集和场景使用的真实尺寸、缩放与声源高度 |

例如金毛 `size=large` 可以令目标肩高为某个经配置的厘米值，但 FLUX 图中看起来
较大并不等于已经达到该厘米值。只有 Pixal 网格完成绑定、计算统一 actor scale，
并在 UE 以厘米回读通过后，才能写入实际 shoulder height。

生成后按以下顺序测量：

1. 认证模型坐标和前/上轴；
2. 用有效脚底点确定地面 `Z=0`；
3. 用排除少量离群碎片的稳健 bounds 测量网格高度和长度；
4. 绑定后用语义肩、髋、脚骨测量肩高/髋高；
5. 由 `actor_scale = target / measured` 计算统一运行缩放；
6. UE 导入后以厘米重新读取 bounds、语义骨骼和地面接触；
7. 只有 UE 读回值在容差内，才写入最终 `physical_measurements.runtime`。

找不到可靠语义骨骼时，相关字段写 `null` 并记录原因，不能填估计值冒充测量。

Rocketbox 例外：115 个角色保留作者身高和 `actor_scale=1.0`。当前 UE 审计高度
约为 142.93--188.38 cm，不能把儿童缩放到成人身高，也不能把所有人归一化到
同一高度。声源高度由实测角色高度派生，而不是由颜色属性决定。

## 9. 属性 JSON 如何关联声音

外观生成 prompt 不生成声音。`attribute_profile` 只声明声音能力，最终场景
manifest 再锁定具体波形：

### 动物

- `species/breed/life_stage` 映射到物种合理且许可证明确的声音池；
- 资产 manifest 保存 `acoustic_profile_id`、默认和允许的 event class；
- 场景生成时选择具体音频，保存文件哈希、许可证、增益和时间区间；
- 短叫声先做能量分段，再以有静音间隔的事件重复到视频时长；禁止无缝循环
  造成连续机械噪声；
- Idle/Walking 不强制持续发声，动作和叫声时间分别记录。

### 人类

- 基础角色的 gender/life stage 等语义用于选择许可证明确的真实语音；
- 具体语音文件、speaker、gender、transcript、语料许可证和 SHA-256 写入场景
  manifest；
- 资产 JSON 只保存允许的语音 profile，不把某一句话永久焊死在角色 GLB 中。

场景中的声音发射位置由每帧 actor 轨迹与实测声源高度产生。最终数据必须保存
dry source、空间化输出、混音、活动区间以及可复现的音频参数。

## 10. QA 对如何自动产生

QA 对比较两个独立实例的绝对属性，不在实例 JSON 内记录变化历史：

```json
{
  "schema": "avengine_instance_pair_v1",
  "instance_a": "dog_golden_8a4b91c2d147",
  "instance_b": "dog_golden_f10cd818aa03",
  "same_attributes": [
    "species",
    "breed",
    "life_stage",
    "body_build",
    "coat_length",
    "coat_pattern"
  ],
  "different_attributes": {
    "size": ["small", "large"],
    "coat_color": ["dark_golden", "light_golden"]
  }
}
```

自然语言问题由 pair builder 自动生成，例如“哪一只是更大的金毛？”；答案
来自两个绝对 profile 的比较，而不是 `one_step_lighter` 等编辑日志。

QA 有两个严格区分的阶段：

- `qa_pair_plan.json`：只说明依据 request 属性“计划问什么”，答案标记为
  `planned_from_attributes_pending_visual_asset`；
- `qa_dataset.json`：只从已经生成并通过相应视觉 QA 的 `source_asset_v2` 构建，
  才能称为 realized evidence。

这样可以避免 FLUX.2 实际没有把金毛变深，却直接用 prompt 标签生成一个错误
答案。QA 构建器也只在同一 profile、同一 split 内配对；默认优先构建只差一个
属性的 pair，便于把问题归因到单一属性。

数据集编译器还会为每个资产计算 `qa_evidence_attributes`，并只保留两个资产都
有证据的差异属性。外观属性要求 `reference_2d=passed|not_applicable` 且
`static_mesh=passed`；动物的 `size` 还要求
`physical_measurements.status=measured`，并且 runtime 中确实存在 profile 指定的
肩高/髋高等测量字段。因此“图上提示词写了 large”绝不会提前产生“哪只更大”
的 realized 答案。

### 10.1 强制音视频联合的 reasoning QA

instance-pair QA 回答资产之间的属性差异；Apartment 的核心场景 QA 则必须是
真正的跨模态推理：只听音频不能回答，只看视频也不能回答。典型程序先用音频
确定说话人、声源或时间锚点，再在视频轨迹中查询外观、动作、位置或遮挡；也可
先从视频得到候选对象，再用空间音频确定实际发声者。建议记录格式为：

```json
{
  "schema": "avengine_av_reasoning_qa_set_v1",
  "scene_id": "apartment_mixed_example_0007",
  "example_only": true,
  "qa_pairs": [
    {
      "qa_id": "avqa_speaker_visual_0001",
      "task": "audio_to_visual_attribute",
      "question": "说出“请把门关上”的人穿什么颜色的上衣？",
      "answer": {
        "text": "蓝色。",
        "canonical": {
          "speaker_asset_id": "rocketbox_female_nurse_01_blue",
          "upper_garment_color": "blue"
        }
      },
      "modality_requirement": {
        "required_modalities": ["audio", "video"],
        "audio_only_answerable": false,
        "video_only_answerable": false,
        "cross_modal_join_required": true
      },
      "evidence": {
        "audio": {
          "event_id": "speech_0004",
          "transcript": "请把门关上",
          "interval_seconds": [7.5, 10.1],
          "source_asset_id": "rocketbox_female_nurse_01_blue"
        },
        "video": {
          "track_asset_id": "rocketbox_female_nurse_01_blue",
          "visible_interval_seconds": [7.5, 10.1],
          "observed_upper_garment_color": "blue"
        },
        "join": {
          "key": "source_asset_id",
          "operation": "audio_speaker_to_video_track"
        }
      },
      "reasoning_program": [
        "find_speech_event_by_transcript",
        "resolve_audio_source_asset",
        "join_source_to_video_track",
        "read_visible_upper_garment_color"
      ]
    },
    {
      "qa_id": "avqa_event_motion_0001",
      "task": "audio_anchored_visual_temporal_reasoning",
      "question": "狗连续叫了两声以后，哪只动物最先绕到圆桌左侧？",
      "answer": {
        "text": "较大的浅色金毛。",
        "canonical": {
          "asset_id": "dog_golden_retriever_large_light_0003"
        }
      },
      "modality_requirement": {
        "required_modalities": ["audio", "video"],
        "audio_only_answerable": false,
        "video_only_answerable": false,
        "cross_modal_join_required": true
      },
      "evidence": {
        "audio": {
          "event_ids": ["dog_bark_0002a", "dog_bark_0002b"],
          "event_pattern": "two_consecutive_barks",
          "anchor_end_seconds": 4.8
        },
        "video": {
          "candidate_asset_ids": [
            "dog_beagle_medium_dark_0001",
            "dog_golden_retriever_large_light_0003",
            "cat_tabby_small_0002"
          ],
          "first_round_table_left_entry": {
            "asset_id": "dog_golden_retriever_large_light_0003",
            "seconds": 6.2
          }
        },
        "join": {
          "operation": "select_first_visual_event_after_audio_anchor"
        }
      },
      "reasoning_program": [
        "detect_two_consecutive_barks",
        "take_second_bark_end_as_time_anchor",
        "find_all_post_anchor_table_left_entries",
        "select_earliest_entry",
        "describe_selected_asset"
      ]
    }
  ]
}
```

同一合同还能生成“护士说话期间深色比格犬在走还是站”“第一声从左侧传来的
狗叫由深色还是浅色狗发出”“猫叫发生时猫是否被圆桌遮挡”等问题。自动出题器
必须执行以下门禁：

- 删除音频证据后仍能唯一回答，或删除视频证据后仍能唯一回答时，降级为普通
  单模态 QA，不得进入 AV reasoning 集合；
- 音频事件与视频帧必须共享经过验证的时钟和 `source_asset_id`，不能靠文本猜配；
- 说话人/叫声事件不唯一、目标不可见、同步误差越界或遮挡证据不完整时不出题；
- 外观属性必须由视觉 QA 支持，尺寸比较必须由实测物理属性支持；
- 答案同时保存自然语言 `text`、规范化 `canonical`、双模态 `evidence` 和可执行
  的 `reasoning_program`，使中英文或不同问法共享同一事实。

## 11. 数据划分与注册状态

- Rocketbox 所有材质变体继承 `base_avatar_id` 的 train/validation/test split，
  防止同一人物不同颜色跨集合泄漏。
- 动物共享同一生成参考/模板 lineage 的实例也必须组内划分，避免近重复个体
  跨集合。
- 一个命令成功不等于正式资产。状态只能是：
  `formal_dataset_asset`、`research_candidate`、`technical_spike_only` 或
  `rejected`。
- Pixal3D 当前依赖和 SkinTokens 训练来源风险必须保留在 provenance/rights
  blocker 中；视觉通过不能自动清除许可证风险。

生成完成后的权威数据集编译入口是两步式。第一步先把 profile snapshot、原始
`instance_requests.json` 和 realized `source_asset_v2` 冻结为一个不可变输入
manifest：

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
/data/jzy/miniconda3/envs/spear-env/bin/python \
  tools/build_controlled_source_dataset_input_manifest.py \
  --profile /path/to/profile_snapshot.json \
  --request-batch /path/to/instance_requests.json \
  --asset /path/to/realized/source_asset_v2/manifests \
  --dataset-id controlled_source_dataset_v1 \
  --split-salt avengine-controlled-source-v1 \
  --output /new/nonexistent/input/dataset_input_manifest.json

/data/jzy/miniconda3/envs/spear-env/bin/python \
  tools/build_controlled_source_dataset.py \
  --input-manifest /new/nonexistent/input/dataset_input_manifest.json \
  --output-dir /new/nonexistent/output/directory
```

第一步默认只接受 `formal_dataset_asset`。研究阶段如确需检查候选资产，必须在
输入冻结命令显式传 `--allow-state research_candidate`；输出仍会标注“pending
formal acceptance”，不能伪装为正式数据。冻结器会确定性重建 request batch，
并要求每个 realized asset 与且只与一个 request 在 profile、属性、目标物理配置、
rig、声音配置和模型 revision 上完全一致。第二步重新认证输入文件、profile
依赖、所有资产工件和许可证快照，并输出：

- `dataset_manifest.json`：按 lineage 哈希分组的资产与 QA 对；
- `qa_dataset.json`：realized QA 证据；
- `scene_source_pool.json`：全部已注册资产及其场景就绪状态；
- `artifact_audit.json`：输入 profile、资产和许可证的 SHA-256 认证记录。
- `dataset_input_manifest.json`：实际消费的 profile/request/asset 文件及 1:1
  request 绑定；
- `build_receipt.json`：把输入 manifest、数据集、QA、source pool 和 artifact
  audit 的哈希绑定在同一回执中。

直接传 `--profile/--asset` 的旧入口只为历史候选构建兼容，输出会明确标记
`request_lineage=legacy_unverified`；新数据集不得再使用该模式。

split 的原子单位是 `lineage_group_id`，不是 instance。一个 Rocketbox 角色的
所有配色永远在同一 split；同一动物参考模板产生的所有个体也在同一 split。
当前只有六个 lineage 的小 canary 可能让某个 split 暂时为空，这是样本组数少
的正常结果，不是放宽 lineage 规则的理由；正式构建需增加独立、合规的基础
lineage。

`scene_source_pool.json` 不是“其中每一项都可直接开拍”的白名单。每条记录都
带有 `qa`、`rights` 和：

```json
{
  "scene_readiness": {
    "eligible_for_apartment_generation": false,
    "blocking_qa": ["ue_import_readback", "apartment_media", "audio"],
    "rights_ready": true,
    "policy": "all_scene_qa_passed_and_rights_cleared_v1"
  }
}
```

场景生成器必须只选择 `eligible_for_apartment_generation=true` 的记录。候选
数据集仍可包含尚未完成场景 QA 的资产，以检查属性、哈希和 QA 数据合同；但
它们不会因为出现在 source pool 文件中就被误当成可生产 Apartment 视频的资产。

当前实际编译结果进一步验证了这条规则：

| 数据集 | 资产 | realized pair / question | 当前可用于 Apartment |
|---|---:|---:|---:|
| `animal_static_candidate_dataset_v1_20260713` | 10 动物 | 2 / 3 | 0 / 10 |
| `rocketbox_candidate_dataset_v3_20260713` | 3 人类 | 3 / 6 | 0 / 3 |
| `combined_controlled_candidate_dataset_v1_20260713` | 10 动物 + 3 人类 | 5 / 9 | 0 / 13 |
| `combined_controlled_candidate_dataset_full_v1_20260713` | 42 动物 + 3 人类 | 42 / 86 | 0 / 45 |

合并数据集的权威 manifest 是
`external/SPEAR/tmp/controlled_source_asset_execution_v1/combined_controlled_candidate_dataset_v1_20260713/dataset_manifest.json`，
SHA-256 为
`a433ad26d5c63d5cdb211610998ed4033c53e44ded07d699e249c49aaa4d6e86`；
13 个资产及其 profile、source artifact 和许可证共计全部重新认证通过。

完整 45-request 静态批次完成后的权威合并 manifest 是
`external/SPEAR/tmp/controlled_source_asset_execution_v1/combined_controlled_candidate_dataset_full_v1_20260713/dataset_manifest.json`，
内部 `manifest_sha256` 为
`241e331f5607d23b97c02e3800fb0d60f6734212cec78b1b5fa738356061d54f`。
它包含 42 个动物和 3 个 Rocketbox 人类候选，生成 42 个 realized pair、
86 个问题：`body_build=32`、`coat_color=29`、`coat_tone=5`、
`point_color=14`、`top_color=6`。45 个资产、6 个 profile 及所有引用工件和
许可证再次认证通过；由于动画、UE、Apartment、audio 和 rights 门禁尚未齐全，
`scene_source_pool.json` 正确保持 0/45 eligible。

最初的动物 canary 原本计划了 5 个 pair，但静态阶段只有哈巴狗
`coat_color=apricot/fawn` 和虎斑猫 `body_build=standard/slim` 具备足够证据，
所以只产生 2 个 realized pair。暹罗猫、比格犬和金毛的三个 size pair 都被
自动拦截，因为 `physical_measurements.status=pending`。这正是期望行为：需要
先完成真实厘米和 UE scale 闭环，之后重新注册新 revision，size QA 才会出现。
完整批次的 86 个问题同样不包含任何 `size` 问题；新增问题只来自已经通过 2D
和静态 3D 证据的绝对外观属性。

## 12. 当前实现状态、代码权威与剩余边界

截至 2026-07-13，已实现并有自动测试覆盖的是：

- 严格 profile/request/pair/`source_asset_v2`/dataset 契约：
  `external/SPEAR/tools/controlled_source_asset_schema.py`；
- 每项 1--3 个值检查、禁止相对属性、精确配额平衡采样和确定性 ID；
- 动物完整 prompt 与 Rocketbox MaterialEditPlan 编译；
- profile 依赖工件的实际 size/SHA-256 认证与无覆盖发布；
- authenticated preflight 与两个 route 的 normalized execution adapters；
- FLUX.2 多 GPU 常驻执行、2D 决策、ISNet 输入准备和 Pixal3D 多 GPU 常驻执行；
- Pixal GLB 回读、静态多视图、静态决定与动物候选 `source_asset_v2` 注册；
- planned QA、realized QA、lineage split 和跨 split 泄漏检查；
- 输入编译 CLI：
  `external/SPEAR/tools/build_controlled_source_asset_inputs.py`；
- 已生成资产的数据集编译 CLI：
  `external/SPEAR/tools/build_controlled_source_dataset.py`；
- 数据集规范化输入冻结与 request 逐资产绑定 CLI：
  `external/SPEAR/tools/build_controlled_source_dataset_input_manifest.py`；
- 回归测试：
  `external/SPEAR/tests/tools/test_controlled_source_asset_schema.py`。

当前两条 route 的规范化入口按阶段分别是：

| 职责 | 工具 |
|---|---|
| 认证 `execution_jobs.json` 并发布 preflight | `external/SPEAR/tools/prepare_controlled_source_asset_execution.py` |
| Rocketbox 去重材质执行 | `external/SPEAR/tools/execute_controlled_rocketbox_material_jobs.py` |
| Rocketbox 原生动作 runtime / UE runtime | `external/SPEAR/tools/run_controlled_rocketbox_runtime_handoffs.py`、`external/SPEAR/tools/normalize_controlled_rocketbox_runtimes.py` |
| Rocketbox 候选注册 | `external/SPEAR/tools/register_controlled_rocketbox_source_assets.py` |
| 动物 FLUX.2 多 GPU 执行 | `external/SPEAR/tools/run_controlled_animal_flux2_jobs.py` |
| 动物 2D 决策 | `external/SPEAR/tools/review_controlled_animal_flux2_candidates.py` |
| ISNet/Pixal 输入准备 | `external/SPEAR/tools/prepare_controlled_animal_pixal_inputs.py` |
| 动物 Pixal3D 多 GPU 执行 | `external/SPEAR/tools/run_controlled_animal_pixal_jobs.py` |
| 动物静态多视图/决策 | `external/SPEAR/tools/run_controlled_animal_static_reviews.py`、`external/SPEAR/tools/review_controlled_animal_pixal_static_candidates.py` |
| 动物静态候选注册 | `external/SPEAR/tools/register_controlled_animal_source_assets.py` |
| profile/request/asset 输入冻结 | `external/SPEAR/tools/build_controlled_source_dataset_input_manifest.py` |
| realized QA / split / scene pool | `external/SPEAR/tools/build_controlled_source_dataset.py --input-manifest ...` |

每个工具都重新认证上游哈希、在不存在的新目录中 staging、回读后原子发布，并
拒绝覆盖旧产物。人工判断不是直接修改生成 JSON，而是形成带哈希的新 decision
manifest；后续执行器只消费已批准 decision。

目前必须区分“静态 canary 已完成”和“可正式批量生成 Apartment 资产”：

- 115 个 Rocketbox 原生角色、Walking/Standing Idle、UE 导入/回读和作者身高
  已验证；当前男性 profile 的 blue/green/burgundy 三种上衣材质、runtime 和
  候选数据集也已经执行。其余 114 个模板尚无完整颜色 mask/profile 审计，不能
  仅因原角色能进 UE 就声称它们可随机改色。
- 新受控动物批次已经真实执行 45/45 个 request。2 个多尾虎斑在 2D 阶段拒绝，
  1 个悬浮尾段比格犬在静态 3D 阶段拒绝；42 个通过项已成为静态候选
  `source_asset_v2`。与 3 个 Rocketbox 材质候选合并后，数据合同实际产生
  42 个 realized QA pair 和 86 个问题。
- 这 42 个动物尚未完成 LOD、品种动作族绑定、Walking/Idle、metric size、UE、
  Apartment 和音频 QA。静态批准只授权进入 LOD/绑定，不授权批量场景生产。
- 目标动物尺寸目前是 provisional pipeline target；正式注册前必须补许可证
  快照和可核验品种尺寸来源，再用实际网格/骨架/UE 读回值闭环。
- 现有 `external/SPEAR/data/source_assets_v1` 是旧注册表，仍包含 Hunyuan 历史
  资产和早期字段。必须保留，不就地改写；新资产通过新 ID 和
  `source_asset_v2` 并行迁移。

建议的剩余实现顺序是：

1. 将 Pixal 执行器从静态 round-robin 改成共享抢占队列并做回归测试，消除多
   GPU 尾部空闲；
2. 对当前 42 个动物做安全 LOD，并按猫/狗动作族进行绑定；
3. 完成 Walking/Idle、方向、落地、GLB 回读和真实尺寸测量；
4. 完成 UE、Apartment、物种音频和可观看媒体 QA，发布新的候选 revision；
5. 清除许可证/参考来源 blocker 后，才把通过项提升为
   `formal_dataset_asset`；
6. 扩展女性/儿童/职业 Rocketbox 的独立 mask profile 和更多合规动物 lineage。

旧文档或旧 contract 与本文冲突时，本文关于绝对属性、每项最多三个值、人类
眼睛/附件锁定、Rocketbox 固定几何、FLUX/Pixal 分工和物理测量生命周期的决定
优先。

## 13. 2026-07-13 动态、Apartment 与 QA 闭环结果

本节覆盖第 12 节里“42 个动物都尚未完成动态阶段”的历史状态。静态候选没有被
整批直接放行；只有继续通过 LOD、绑定、动作、UE、Apartment、音频和测量的
31 个实例进入最终动态候选注册表，其余实例保留原 decision/rejected 证据。

| 最终集合 | 实例 | Walk/Idle clips | 完整动作对 | 状态 |
|---|---:|---:|---:|---|
| Rocketbox 原生人类 | 115 | 230 / 230 | 115 / 115 | 原几何/骨架稳定基线完成 |
| Pixal 猫 | 8 | 16 / 16 | 8 / 8 | `research_candidate` |
| Pixal 狗 | 23 | 46 / 46 | 23 / 23 | `research_candidate` |
| 动物合计 | 31 | 62 / 62 | 31 / 31 | 技术 QA 全通过，rights 待清理 |

稳定可点击入口是：

- 人类：`docs/rocketbox_batch_apartment_video_index.md`；
- 动物：`docs/controlled_animal_video_catalog.md`；
- 最终动物解析 manifest：
  `external/SPEAR/tmp/controlled_source_asset_execution_v1/controlled_animal_apartment_specs_dogs_resolved_v1_20260713/spec_manifest.json`；
- 最终动物 `source_asset_v2` 注册表：
  `external/SPEAR/tmp/controlled_source_asset_execution_v1/animal_source_assets_apartment_31_final_v1_20260713/registry_manifest.json`。

### 13.1 物理属性不是 prompt 自证

`measure_controlled_animal_physical_attributes.py` 使用绑定后的网格比例、骨骼加权
前肢/肩部表面和 UE 每帧 bounds/actor scale 计算肩高、整体高度与鼻尾长度。
第一次统一测量发现 Pug 的相对 small/medium/large 顺序虽然正确，但实际肩高为
约 55--81 cm，明显不适合作为普通 Pug。该异常没有因“自动检查顺序通过”而被
放行。

`recalibrate_controlled_animal_apartment_specs.py` 采用：

```text
new_actor_scale = observed_actor_scale
                  * target_shoulder_height_cm
                  / observed_shoulder_height_cm
```

它只挑选超过相对误差阈值的资产，并发布全新的审核目录，不覆盖旧视频。
`resolve_controlled_animal_apartment_manifests.py` 再把新记录显式替换进解析后的最终
manifest。Pug 第二次 UE 回读结果如下：

| size | 实测肩高中位数 cm | 目标 cm | 结果 |
|---|---:|---:|---|
| small | 27.074 | 27 | passed |
| medium | 30.000 | 30 | passed |
| large | 32.997 | 33 | passed |

这套反馈是纯代码步骤。新实例若尺寸误差不超过阈值就沿用第一次结果；明显异常才
自动产生新 revision，不需要人工逐资产调 scale。

### 13.2 最终数据集与问题答案

最终候选数据集位于
`external/SPEAR/tmp/controlled_source_asset_execution_v1/controlled_animal_apartment_candidate_dataset_31_final_v1_20260713`：

| 项目 | 数量/状态 |
|---|---:|
| `source_asset_v2` | 31 |
| realized instance pairs | 92 |
| realized questions | 226 |
| size questions | 68 |
| size answer/实测顺序冲突 | 0 |
| size answer 最小肩高差 | 0.544 cm |
| technical `blocking_qa` 非空 | 0 / 31 |
| `rights_ready=true` | 0 / 31 |
| formal Apartment eligible | 0 / 31 |

数据集 manifest 内部 SHA-256 为
`ceb30767c813b99ce7bd526b71aa1f58d4f15b7b850b5f35584a0cd45ec2cc92`；
注册表内部 SHA-256 为
`caebb9301843f4dfec2c8f7d18bb278a3d97974ad11c98f1d1c94bf7ef1b85d9`。
所有技术 QA 已通过，但许可证与来源风险没有被技术成功自动清除，因此这些媒体
可用于研究候选审核和 QA 合同验证，不能改标为 `formal_dataset_asset`。

当前 31 个资产只有五个动物 lineage，按 lineage 分组后本次小批次全部落在
train 是哈希划分的自然结果。不能为了让 validation/test 非空而把同一模板的
近重复实例拆开；正式数据应增加许可证明确的独立基础 lineage。

### 13.3 人类与动物统一的规范化输入构建

最终统一候选不再从一组松散的 profile 和资产目录直接编译。冻结输入位于：

`external/SPEAR/tmp/controlled_source_asset_execution_v1/controlled_human_animal_dataset_input_34_final_v1_20260713/dataset_input_manifest.json`

其逐字节一致、可由 Git 审计的控制面副本是
`external/SPEAR/data/controlled_source_attributes_v1/dataset_inputs/controlled_human_animal_34_final_v1_20260713.json`；大体积资产和媒体仍只保存在不可覆盖的
证据目录中。

它认证了 8 个 profile、2 个 deterministic request batch、72 条 request 和 34 个
realized asset。34 个资产各自匹配唯一 request，34 个 asset ID 和 request SHA-256
均无重复，全部通过逐字段验证；另外 38 条未实现 request 被明确记录为 unused，
不会被误报成数据集资产。输入 manifest SHA-256 为
`c1b315c590030f4f952f4b876ab976bae665621f41c46b947efe27f9eb5a1c8c`。

只消费上述 manifest 得到的权威统一候选位于：

`external/SPEAR/tmp/controlled_source_asset_execution_v1/controlled_human_animal_normalized_candidate_dataset_34_final_v1_20260713`

| 项目 | 数量/状态 |
|---|---:|
| 动物 `source_asset_v2` | 31 |
| Rocketbox 材质 `source_asset_v2` | 3 |
| realized instance pairs | 95 |
| realized questions | 232 |
| request-lineage bindings | 34 / 34 passed |
| duplicate asset/request IDs | 0 |
| lineage leakage / cross-split QA | 0 / 0 |
| scene-ready | 0 / 34 |

数据集、QA、source pool 和 artifact audit SHA-256 分别为：

- `a7b09ddec1d2d6e72690d686deaa42f9f62ea04db52a26bc31e5838c8a27e3fa`；
- `52a15ee29d285aa9ef19c3ca43edc59284e9e3d33437fae7f40f0ac5d8401e34`；
- `66feb737c35b23285a9d3d4e20c622da72645003ea593267c83781aa8fa12738`；
- `2c28d454a06b98da509ace7bd5ab211852f60111144eb342a5ac99ae4800df45`。

构建回执 SHA-256 为
`bd8e68e529ebd342561386d8e9c492f85f4beae1f774a7d6cd527703a9c989fa`。
新编译的四个核心数据文件与此前直接构建的 34-asset 版本逐字节一致，说明加入
request-lineage 认证没有改变已批准的 QA 内容，只补齐了可复现输入边界。

`scene-ready=0` 有两类原因，必须分别保留：31 个动物技术 QA 已通过但 rights
尚未清理；3 个 Rocketbox 配色候选 rights 已清理，但仍缺该三条受控实例自己的
UE import readback、Apartment Walk/Idle 媒体和音频证据。不能借用 115 个原生角色
批次的成功记录自动提升这三个材质 revision。

## 14. 2026-07-13 动物正侧姿势修正 canary

第 13 节的 31 个旧动物虽然文件门完整，但其 Walking 已被用户视觉审核推翻：猫
斜跑、狗后退或斜跑。因此旧 31 个结果继续作为 `rejected_by_user_visual_review`
诊断证据，不能因为自动骨向量或轨迹检查通过而恢复注册。

根因不只是一个可用 yaw 修复的坐标问题。旧 2D/Pixal 输入普遍存在头颈偏转、
躯干纵轴斜扭、前后腿不在一致平面或四脚不共地。人工把页面调成 -15°、-45°、
-170° 只是在追随歪头轮廓，并不能把错误 rest pose 变成合格绑定输入。新模板把
生成约束前移到 2D：严格正侧视、头颈沿躯干、脊柱笔直、四条腿彼此可见且处于
一致前后平面、四脚同一地面、neutral authored quadruped rest pose。狗 v6 还用
uniform-clay 源图避免灰色遮罩泄漏，再由 Pixal3D 生成最终 PBR。

新的方向门采用以下不可放宽的合同：

- 原始 100k mesh 以 identity 变换显示；
- 禁止自动方向推断、隐藏 mirror 和细角度 yaw；
- 人工只能选择 0°、±90°、180°；
- 若头颈、躯干、腿平面或落地不合格，直接拒绝源姿势并重生，不能靠旋转补偿；
- 静态方向、隔离动作、UE Apartment 和正式注册是相互独立的门。

当前不覆盖 canary 结果如下：

| 资产 | 隔离 Walking/Idle | UE import/readback | Apartment Walk/Idle | 状态 |
|---|---|---|---|---|
| `cat_tabby_four_limb_rest_side_3a1ecde08179` | passed | passed | 2 / 2 passed | `research_candidate`，待人工方向保存 |
| `dog_beagle_four_limb_rest_side_clay_1550ff78df40` | Walking 脚部拉伸/碎片 | 未运行 | 未运行 | `rejected` |
| `dog_beagle_four_limb_rest_side_clay_1b1e63af05c3` | passed | passed | 2 / 2 passed | `research_candidate`，待人工方向保存 |

猫和通过的狗在 Apartment 中均使用“相机右后方 → 左前方 → 绕桌一圈”轨迹，
动态贴地、root roll/pitch、270 帧 GLB/UE 回读均通过。四段 18 秒双耳音频也已
生成：猫 9 个短叫事件，狗 7 个短叫事件，均采用有静音间隔的自适应重复，不是
无缝机械循环。

当前审核入口是 `http://127.0.0.1:8102/`，认证清单是
`external/SPEAR/tmp/controlled_source_asset_execution_v1/controlled_animal_pose_direction_new_canary_review_v4_20260713/review_manifest.json`。
清单同时认证隔离 Walk/Idle、UE Walk/Idle 主视图、同步 Top-down 和组合审核片；
动画 rejected 的狗没有 Apartment 链接。文件 SHA-256 为
`977b74fb08cad3ce2a547b0eb87eafba54ac5bc25e61c03dcabf1fb6109b47bd`，
内部 `manifest_sha256` 为
`0e3efd51ac9ca7bc73f5158c2b037e837764d55b909a6cc8c8be51ebece56ef9`。

该 canary 证明“严格生成姿势 → Pixal3D → 物种绑定 → Walk/Idle → UE Apartment”
可以走通，但只覆盖一个猫实例和一个狗实例，且 Pixal/reference/training provenance
风险仍在。因此它不能恢复旧 31 个资产，也不能自动提升为
`formal_dataset_asset`；下一批必须从同一姿势合同逐品种扩展，并保留每次人工
整 90° 方向结论。
