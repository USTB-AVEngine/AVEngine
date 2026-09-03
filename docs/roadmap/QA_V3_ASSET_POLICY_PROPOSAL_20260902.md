# 题型声明资产政策：方案（2026-09-02，只写方案，不实施）

> 状态：提案，等 Codex 只读审核与 owner 拍板后再改生成器。本文不改变任何现有产物，
> 不宣布任何题型可放量。

## 1. 要解决的问题

现在的生成链是"先有资产，再看能做什么题"：两只狗（黑白边牧、黄色比格）写死在批次
生成器里，题型只在这两只狗上求解。owner 2026-09-02 指出方向应该反过来：**题型先声明
它需要什么样的声源资产，生成器再按声明去注册表里选**。理由是不同题型对资产的要求
本来就不同：

- 有的题只有"同类型的不同个体"才成立，比如按外观区分两只狗、问某一只的毛色；
- 有的题任何单个声源都行，比如首叫时间带、距离变化；
- 有的题需要跨物种混合，但声音内容必须与身份脱钩，比如用音响放狗叫来做"声源在哪"。

资产反过来决定题型，会让人和固定声源进来时无处安放；题型声明资产政策，人和音响
就是按政策自然选进来的。

## 2. 声明放在哪里

每个题型的配置（现在是 `qa_v3_prescale_core_profiles_*.json` 里的一个条目）新增一块
`asset_policy`，与本轮已加的 `visual_requirements` 并列。两块都是"题目的属性"：一块说
要看见什么，一块说用什么演。

```json
"asset_policy": {
  "roles": ["target", "other"],
  "family_rule": "same_family_distinct_instances",
  "allowed_families": ["dog", "human"],
  "motion": {"target": "must_move", "other": "any"},
  "voice": {"shared_clip_across_roles": false, "identity_coupled": true},
  "appearance": {"must_be_distinguishable": true, "referring_expression": "coat_color"},
  "body_geometry_source": "asset_registry",
  "status": "proposal_not_implemented"
}
```

字段含义：

| 字段 | 取值 | 意思 |
| --- | --- | --- |
| `roles` | 角色名列表 | 题目里有几个声源角色，对应现在的 target/other 槽位 |
| `family_rule` | `same_family_distinct_instances` / `any_single_source` / `mixed_families_voice_decoupled` | 三种配对政策，见第 3 节 |
| `allowed_families` | 资产族 | 允许从哪些族里选：dog、human、speaker（音响）等 |
| `motion` | 每个角色 `must_move` / `must_be_still` / `any` | 对路线的要求；card6R 需要一个静止角色，card1 需要目标移动 |
| `voice.shared_clip_across_roles` | 布尔 | 两个角色是否共用同一条声音素材（canonical 音频孪生用） |
| `voice.identity_coupled` | 布尔 | 声音是否必须与身份一致（狗叫必须来自狗）；音响解耦时为假 |
| `appearance.must_be_distinguishable` | 布尔 | 外观是否必须能区分（问毛色的题为真） |
| `appearance.referring_expression` | `coat_color` / `top_color` / `species` / `none` | 题面用什么词指代角色 |
| `body_geometry_source` | `asset_registry` | 预测可见性与净空表的目标带要用的身高体长，从资产注册表读，不再用参数占位 |

## 3. 三种配对政策

1. **同族不同个体**（`same_family_distinct_instances`）：两个角色来自同一族的不同实例，
   外观可区分。适用：card1F/1B、card7、card9、card4R、card5R、card6R、card2/3/5/6/10
   ——所有需要"哪一只"的题。首批人的题按 owner 意见：两个改色的人物、同一把嗓音、
   行走动作，与两只狗完全平行。
2. **任意单声源**（`any_single_source`）：题目只关心一个声源何时/在哪叫，另一角色
   只是干扰。适用：card8（首叫时间带）、card15b（事件计数）。
3. **混族但声音与身份脱钩**（`mixed_families_voice_decoupled`）：允许音响播放狗叫，
   题面不许用"狗"指代声源，只能用"声音"。适用：以后的"声音从哪来"类题；本轮不做。

## 4. 生成器怎么用声明

1. 求解前按 `allowed_families` × `family_rule` 从注册表挑出合法的角色组合；挑不出
   就像现在的扩展题型那样报 `resource_unavailable`，不回退到两只狗。
2. `motion` 决定路线池：`must_move` 的角色从移动路线池抽（本轮的静止路线预过滤就是
   它的特例），`must_be_still` 从静止路线抽，`any` 用全库。
3. `body_geometry_source` 让净空表的目标带与预测可见性按实际身高取值：狗 0.5 m、人
   1.7 m、音响按箱体；净空表已经按 0.5/1.0/1.7 三个高度存了摘要，不需要重渲。
4. `voice` 决定音频调度器怎么选素材：同族不同个体默认各用自己的叫声；共用一条素材
   只在 canonical 孪生里打开。
5. fact 记录实际选中的资产、族、政策名与声明版本，manifest 记每个政策下的候选数。

## 5. 与现有东西的关系

- 不改任何题型的几何约束、锚带/答案带分配、Gate A/Gate B 构造。
- 现有 `pair_assets` 写死两只狗的逻辑保留为 `same_family_distinct_instances` 在 dog 族
  下的特例，历史批次可复现。
- `visual_requirements` 与 `asset_policy` 分开：前者说"要看见谁在哪帧"，后者说"谁来演"。
  两者都由题型声明，都写进 fact。

## 6. 一次性成本与顺序

- 注册表要补：每个资产的族、身高体长、可区分外观标签、指代词、允许的动作（走/站）。
  两只狗与两个改色人先补，音响后补。
- 生成器改动集中在 `build_cell_plan` 的 `pair_assets` 与音频调度器的素材选择，估 1 天；
  测试用合成注册表验证三种政策各能选出/选不出。
- 顺序：Codex 审本方案 → owner 拍板字段 → 注册表补齐 → 生成器改动 → 两房 canary
  （狗）→ 首批人的题（两个改色人）。

## 7. 本方案不声明

不声明任何题型可放量；不声明人的题已可生成；不改占位阈值。

## 8. 静止声源不得当"被问的声源"的通用规则（owner 2026-09-03 已定）

原来的说法是"音响不进动与静类题"，点名 card6、card6R、card10 三个。按同一个逻辑数下来实际不止三个：
**凡是答案取决于声源自身位移的题，静止音响的外观都能替代听音**，因为音响不会走路是常识。

所以规则写成一句通用的：

> **一道题的答案只要取决于某个声源自己的位移，这个声源就不能是静止声源（音响、固定件）。**
> 静止声源仍然可以在同一道题里当另一个声源（干扰项），因为答案不问它。

按现在实现的题型清一遍，受这条规则约束的是八个：

| 受约束（答案取决于被问声源的位移） | 不受约束 |
| --- | --- |
| card1F、card1B（叫声前后的方位带） | card2（此刻方位带） |
| card5、card5R（离你变近还是变远） | card3（第一声来自哪侧） |
| card6、card6R、card10（在动还是站着） | card4R（哪只更近） |
| card16（结尾可见状态，取决于走到哪） | card7、card9（毛色） |
| card17（第二段里在哪） | card8（第几秒） |
| | card11（哪只发的声）、card12（什么声音） |
| | card13、card14（说了什么／谁说的） |
| | card15a、card15b（计数） |

**怎么执行**：这条规则按"被问的声源"判定，不按题型名单判定，所以新增题型自动适用，不用回来改名单。
生成器侧的落点是题型声明里的 `answer_depends_on_source_displacement: true`（占位键名），
求解器给这类题分配角色时，把 `entity_class == "rigid_static_object"` 的资产排除在目标角色之外，
对照角色不受限；fact 里记下这次判定。**本节只是规则，键名与实现待出题侧那一摊落地。**
