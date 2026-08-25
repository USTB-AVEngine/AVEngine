# 毛色轴：品种特征色不是改一张表就能到的（20260826）

owner 20260826 拍板"毛色清单按品种特征色，不再是明度三档"。我按交接文档的指示，
先读了 `COAT_PROFILE_REALIZATION_RULES` 怎么消费这些值再动手。结论是：**这不是
对齐一张域表，是要加一个新的实现算子。** 下面是证据和最小可行方案。

## 1. 现状：毛色轴在代码里就是一个亮度乘数

`src/avengine/appearance/contracts.py`：

```
_OPERATION_BY_AXIS = {
    "coat_profile": "breed_scoped_coat_luminance_v1",   # <- 算子名字就写着 luminance
    ...
}
REALIZER_PARAMETER_BOUNDS = {
    "luminance_gain": (0.65, 1.35),                     # <- 唯一的毛色参数
    ...
}
COAT_PROFILE_REALIZATION_RULES[(species, breed, profile_id)] = {
    "light_level": ..., "neutral_level": ..., "dark_level": ...,
    "baseline_level": ..., "preserve_pattern": ...,      # <- 校验器要求恰好这五个键
}
```

`src/avengine/assets/appearance_realization.py` 实际做的事：对**已烘焙的 albedo**
按像素乘一个亮度增益，且显式要求 `preserve_pattern` 不变（比格那条实现里
`if preserve_pattern != "tricolor": raise`）。也就是说这个算子的语义是
**"同一张贴图、同一个花纹、整体调亮调暗 ±35%"**。

所以：

| owner 要的 | 亮度乘数能做到吗 |
|---|---|
| 拉布拉多 黄 → 黑 | 勉强（同色系纯色，明度差） |
| 缅甸 貂色 → 香槟 → 铂金 | 大致可以（这就是缅甸猫的经典稀释序列） |
| 拉布拉多 黄 → **巧克力** | 不行，是色相变化 |
| 暹罗 海豹点 → **巧克力点 / 蓝点 / 丁香点** | 不行，全是色相变化 |
| 阿比 红褐 → **蓝 / 浅黄** | 不行 |
| 边牧 黑白 → **红白 / 蓝色云石** | 不行，且云石是**花纹**变化 |
| 柴犬 赤 → **黑芝麻**；柯基/杰罗 **三色** | 不行，加了新的色块 |
| 英短 蓝 → **银虎斑** | 不行，虎斑是花纹 |

9 个品种里只有缅甸猫那一列基本落在亮度算子的能力内。

## 2. 还有两个 3 值硬上限（两侧都是三档，不是巧合）

1. **AVEngine 侧**：`_L9_LEVEL_ROWS` 是标准 OA(9,4,3,2)，每根轴**恰好 3 档**
   （levels 0/1/2）。四个值放不进去。
2. **SPEAR 侧**：`controlled_source_asset_schema.validate_attribute_profile` 里
   `not (1 <= len(raw_values) <= 3)` 直接拒绝，**每根采样轴最多 3 个值**。

所以 owner 批的清单里那 4 个四值品种（英短 / 暹罗 / 缅甸 / 阿比）必须各砍一个。
砍哪个我按"孤立渲染里最难分辨的那个"来定：

| 品种 | 建议的三个值 | 砍掉的 | 理由 |
|---|---|---|---|
| 英国短毛猫 | 蓝 / 乳白 / 黑 | 银虎斑 | 虎斑是花纹变化，成本最高，而蓝/乳白/黑已经互相不可能看错 |
| 暹罗 | 海豹点 / 巧克力点 / 蓝点 | 丁香点 | 丁香点和蓝点在一帧里最接近 |
| 缅甸 | 貂色 / 香槟 / 蓝 | 铂金 | 铂金和香槟最接近 |
| 阿比西尼亚 | 红褐 / 蓝 / 浅黄 | 索雷尔 | 索雷尔和红褐最接近 |
| 拉布拉多 | 黄 / 巧克力 / 黑 | — | 已经三个，而且是这个品种的规范三色 |
| 边境牧羊犬 | 黑白 / 红白 / 蓝色云石 | — | 已经三个 |
| 杰克罗素 | 白棕 / 三色 / 白黑 | — | 已经三个 |
| 柯基 | 红白 / 貂白 / 三色 | — | 已经三个 |
| 柴犬 | 赤 / 黑芝麻 / 奶油 | — | 已经三个 |

## 3. 能做到的路子已经在仓里：SPEAR 的多视图毛色投影

`SPEAR/tools/blender_project_animal_multiview_coat.py`：渲四视图 → 在图上改颜色/
花纹 → 按空间对数色比投影回**没有变过的 UV**。配套
`blender_render_generated_animal_coat_views.py` 渲源视图，
`build_animal_coat_reference_board.py` 出参考板。

关键性质（这决定了成本）：**"几何、蒙皮权重、骨架、Walk/Idle 动作一律不重新生成"。**
所以每个新毛色只要一次投影，**不重绑骨、不重跑形变闸门**——两道动物闸门判的是
几何与蒙皮，几何逐字节相同就没有重判的必要。

## 4. 最小可行方案

1. **不动**现有 `breed_scoped_coat_luminance_v1` 和它的 7 个已注册资产。
   它是经过验证的算子，明度三档在它的语义内是正确的。
2. **新增**第二个毛色算子 `breed_scoped_multiview_coat_projection_v1`，
   由上面那个 SPEAR 工具实现，参数不是 `luminance_gain` 而是
   "每个值一份已评审的四视图编辑板"。
3. `COAT_PROFILE_REALIZATION_RULES` 需要按算子分支：投影算子的规则里没有
   light/neutral/dark 这三个角色，只有 `baseline_value` + 每个值的参考板哈希。
   现在的校验器写死了 `expected_rule_keys` 那五个键，要按算子选期望键集。
4. 每个品种一个新的 `<species>_<breed>_coat_v2`（不要覆盖 v1 的 profile_id，
   v1 是 7 个已注册资产的权威）。
5. 顺序上先做一个品种走通（建议**拉布拉多 黄/巧克力/黑**：三色是规范三色、
   纯色无花纹、色相差最大，最容易判成功），再铺开。

## 5. 为什么这条值得做

owner 的原话是"每个属性，最好真的能和之前能看出来不一样"。明度三档在孤立渲染里
是最难看出来的那种差别——这也正是尺寸轴被降级的同一个理由。品种特征色是
**唯一**能让"同一个品种的两个实例"在一帧里明确不同、而且能出 QA 金标答案的轴。
