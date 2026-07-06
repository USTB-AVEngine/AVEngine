# Apartment 家具碰撞检测 — 设计文档

> 日期：2026-07-06
>
> Scope：**只**解决 apartment 场景里动物轨迹穿家具（沙发/门框/餐桌）的问题。
>
> **明确不做**：RoomSpec 抽象重构 · procgen 家具 · procgen 房间 · 家具语义标签 · QA 生成器 · 房间随机化 · Cosmos sim2real。

---

## 1. 问题陈述

`tools/gpurir_scenes/scene_spec.py::check_no_clipping()` 目前只知道抽象房间边界（wall margin），不知道 apartment_0000 地图里摆着的沙发、餐桌、门框等家具。结果：

- 随机采样场景 (`compose_scene(seed)`) 无法在 apartment 里跑，动物会穿家具
- 手写场景 (`scene_two_dogs.py`) 只能靠硬编码走廊约束（如"husky.x ≤ 2.85"），换一个 apartment 变体就废
- 视觉上表现为"哈士奇从沙发/门框里长出来"

SPEAR RPC 支持 `find_actors_by_class + GetActorBounds`（现成代码在 [tools/enumerate_scene_primitives.py](../../../tools/enumerate_scene_primitives.py) / [tools/full_primitive_enum.py](../../../tools/full_primitive_enum.py)），信息本来就在 UE 里，只是我们没读。本次设计把这条路径打通。

---

## 2. 高层方案

三块新代码 + 两处现有代码修改 + 一份数据文件：

```
NEW  tools/gpurir_scenes/dump_apartment_furniture.py   一次性离线 dump 脚本
NEW  tools/gpurir_scenes/furniture_map.py              JSON 加载 + 坐标转换 utils
NEW  data/apartment_furniture_map.json                 dump 产物，入 git
EDIT tools/gpurir_scenes/scene_spec.py                 check_no_clipping / compose_scene 加 furniture_bboxes 参数
EDIT tools/gpurir_scenes/run_render_pass.py            apartment 分支读 JSON 并传入
NEW  tests/tools/test_furniture_collision.py           4 项验收
```

数据流：

```
[UE apartment_0000]
  │  一次性 dump（3-5 分钟）
  ▼
[apartment_furniture_map.json]  (UE cm 坐标 + metadata，入 git)
  │  furniture_map.py 加载
  ▼
[FurnitureBBox list]  (SceneSpec 米坐标)
  │
  ├─→ compose_scene(seed, furniture_bboxes=...)   采样时避开家具
  │
  └─→ check_no_clipping(spec, furniture_bboxes=...)   每帧每动物 vs 每件家具
```

---

## 3. 数据结构

### 3.1 dump 产物 `data/apartment_furniture_map.json`

```json
{
  "meta": {
    "apartment_map_path": "/Game/apartment_0000/Maps/apartment_0000.apartment_0000",
    "dump_date_utc": "2026-07-06T10:00:00Z",
    "spear_commit": "<git rev-parse HEAD>",
    "ue_version": "5.5",
    "apartment_mic_origin_cm_at_dump": [-120.0, 80.0, 120.0],
    "num_actors_seen": 1234,
    "num_actors_after_filter": 87,
    "filter_reasons": {
      "z_ceiling": 3,
      "z_floor": 1,
      "name_wall": 12,
      "name_floor": 4,
      "bbox_too_large": 2,
      "kept": 87
    }
  },
  "furniture": [
    {
      "actor_name": "SM_Sofa_01_C_UAID_1234",
      "uclass": "AStaticMeshActor",
      "bbox_min_ue_cm": [120.5, -80.2, 0.0],
      "bbox_max_ue_cm": [280.5, 70.8, 90.0],
      "actor_location_ue_cm": [200.5, -5.0, 45.0],
      "actor_rotation_deg": [0.0, 0.0, 15.0]
    }
    // ...
  ]
}
```

**关键设计**：`bbox_*_ue_cm` 是**未经转换的 UE world 坐标**（Q7 决策）。运行时用 `furniture_map.py` 把它转成 SceneSpec 米坐标；未来 `APARTMENT_MIC_ORIGIN_CM` 调整时**无需**重新 dump。

### 3.2 运行时 `FurnitureBBox` (dataclass)

在 `furniture_map.py` 里：

```python
@dataclass(frozen=True)
class FurnitureBBox:
    actor_name: str
    xy_min_m: tuple[float, float]   # SceneSpec 坐标（米，含 0.1m margin）
    xy_max_m: tuple[float, float]
    z_min_m: float
    z_max_m: float

    def contains_xy(self, x: float, y: float) -> bool:
        return (self.xy_min_m[0] <= x <= self.xy_max_m[0]
                and self.xy_min_m[1] <= y <= self.xy_max_m[1])
```

**注意**：只使用 AABB（Q2 决策）。actor 旋转 15° 的沙发会得到略大的外接立方体 —— 换来实现简单、O(1) 碰撞检测。0.1m margin 在**加载阶段**膨胀 bbox，运行时不再管。

---

## 4. 组件详细设计

### 4.1 `dump_apartment_furniture.py`（新）

**入口**：

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python \
  tools/gpurir_scenes/dump_apartment_furniture.py \
  --out data/apartment_furniture_map.json
```

**核心步骤**：

1. `configure_instance(rpc_port=39002)` 启动 UE 加载 `apartment_0000`
2. `game.unreal_service.find_actors_by_class(uclass="AStaticMeshActor")` 枚举所有静态 mesh actor
3. 对每个 actor：
   - 调 `actor.GetActorBounds(bOnlyCollidingComponents=False)` 拿 `origin` 和 `boxExtent`
   - 计算 `bbox_min = origin - boxExtent`, `bbox_max = origin + boxExtent`（AABB，UE cm）
   - 拿 `actor_location = actor.K2_GetActorLocation()`
   - 拿 `actor_rotation = actor.K2_GetActorRotation()`（Roll/Pitch/Yaw）
   - 拿 `actor_name = get_stable_name_for_actor(actor)`（参考 [render_in_apartment.py:312-318](../../../examples/render_in_apartment.py#L312-L318)）
4. **三层过滤**（Q6 决策）。**注意**：所有 z 值都是 **UE world Z 坐标**（apartment 原点算起），不是"房间地板算起"。apartment 地板在 UE 里 Z 约 27cm（见 `APARTMENT_FLOOR_Z_CM`），所以以下阈值是"起步值"，**dump 首次跑完后 print 一份 z 分布直方图**，然后 review 是否需要调整：
   - **z 高度上限**：`bbox_min_z_ue_cm > 300` cm → 天花板，跳
   - **z 高度下限**：`bbox_max_z_ue_cm < 5` cm → 地板贴片，跳
   - **名字关键词**：`actor_name` 小写后含 `wall / floor / ceiling / ground` → 跳
   - **bbox 大小兜底**：`(x_extent_cm * y_extent_cm) > 200000` cm²（20m²）→ 结构 mesh，跳
   - dump 脚本必须**先 print z 分布** + 每个 filter reason 计数，再写 JSON。若某个 reason 计数 > 50% 触发 → 停下来 review 阈值再写 JSON
5. 记录每个 filter reason 的计数到 `meta.filter_reasons`
6. 剩余 actor 全部写入 `furniture[]` 列表
7. 关闭 UE，写 JSON

**验收信号**：脚本自己 print 一份 summary，包含 `num_actors_seen`, `num_actors_after_filter`, 每个 filter reason 计数。

### 4.2 `furniture_map.py`（新）

**职责**：JSON ↔ 运行时 FurnitureBBox 转换。

```python
# ---- 加载 ----------------------------------------------------------------
def load_apartment_furniture(
    json_path: str = "data/apartment_furniture_map.json",
    apartment_mic_origin_cm: tuple = APARTMENT_MIC_ORIGIN_CM,
    apartment_mic_pos_scene_m: tuple = (2.6, 2.2),
    y_flip: bool = True,
    margin_m: float = 0.1,
) -> list[FurnitureBBox]:
    """把 UE cm bbox 转成 SceneSpec 米坐标 bbox，带 0.1m margin。

    转换公式 (per Q7)：
      x_scene_m = (x_ue_cm - apartment_mic_origin_cm[0]) / 100 + apartment_mic_pos_scene_m[0]
      y_scene_m = -(y_ue_cm - apartment_mic_origin_cm[1]) / 100 + apartment_mic_pos_scene_m[1]  # y-flip
      z_scene_m = z_ue_cm / 100
    """

# ---- 碰撞查询 ------------------------------------------------------------
def bbox_hits_point(bbox: FurnitureBBox, x: float, y: float) -> bool:
    return bbox.contains_xy(x, y)

def any_bbox_hits_point(bboxes: list[FurnitureBBox], x: float, y: float) -> bool:
    return any(b.contains_xy(x, y) for b in bboxes)

def any_bbox_hits_series(bboxes: list[FurnitureBBox], xy_series: np.ndarray) -> bool:
    """xy_series shape (K, 2). 只要任意一帧在任意 bbox 内即算命中。"""
```

**关键约束**：`load_*` 里就把 0.1m margin 膨胀到 bbox 上（`xy_min -= 0.1`, `xy_max += 0.1`）。运行时 `contains_xy` 就是纯坐标比较，快。

**Y-flip 决策依据**：apartment 场景当前坐标转换有 y-flip（[run_render_pass.py:128-135](../../../tools/gpurir_scenes/run_render_pass.py#L128-L135)）。`load_apartment_furniture` 复用同样的 y-flip 逻辑，保证家具坐标和动物坐标在同一世界系。

### 4.3 `scene_spec.py` 修改

**`check_no_clipping` 新增 `furniture_bboxes` 参数**（Q8 决策）：

```python
def check_no_clipping(
    spec: SceneSpec,
    wall_margin_m: float = WALL_MARGIN_M,
    min_sep_m: float = ANIMAL_MIN_SEP_M,
    furniture_bboxes: list = None,   # NEW
) -> None:
    # ... 原有 wall margin 和 pairwise 检查 ...
    if furniture_bboxes:
        for a in spec.animals:
            xy_series = _placement_xy_series(a)  # 已存在
            hit = any_bbox_hits_series(furniture_bboxes, xy_series)
            assert not hit, (
                f"{a.tag} clips furniture at some frame; "
                f"bbox_count={len(furniture_bboxes)}"
            )
```

**`compose_scene` 新增 `furniture_bboxes` 参数**（Q9 决策）：

```python
def compose_scene(
    seed: int,
    furniture_bboxes: list = None,   # NEW
) -> SceneSpec:
    rng = np.random.default_rng(seed)
    ...
    for _ in range(200):
        # 生成候选轨迹或静态位置
        cand = ...
        if furniture_bboxes and any_bbox_hits_series(furniture_bboxes, cand[:, :2]):
            continue  # 撞家具，重采
        if _min_xy_distance_series(cand, animals) >= ANIMAL_MIN_SEP_M:
            animals.append(...)
            break
```

**向后兼容**：不传 `furniture_bboxes` → 完全等价原行为。shoebox 场景不受影响。

### 4.4 `run_render_pass.py` 修改

在 apartment 分支加载家具并传给下游（compose_scene 由外层调用者处理；render_pass 只需要传给最终 assert）：

```python
def run_render_pass(spec, room, out_dir):
    ...
    if room == "apartment":
        from gpurir_scenes.furniture_map import load_apartment_furniture
        furniture = load_apartment_furniture()
        check_no_clipping(spec, furniture_bboxes=furniture)   # 双重 assert
    ...
```

**为什么这里也 assert**：spec 可能是手写的（如 `scene_two_dogs`），没走 `compose_scene` 的避让路径。在 render 前做最后一道验证，防止手写场景漏检。

### 4.5 `scene_two_dogs.py` 修改

去掉当前硬编码的走廊约束 —— 具体在 [scene_two_dogs.py](../../../tools/gpurir_scenes/scene_two_dogs.py) 里 `_A_START`, `_A_END`, `_C_END` 这批常量（当前 husky 起点 x=4.0、pivot=3.0、终点 x=3.0，全部是为了避 apartment 门框硬调的）。改法：

- 保留 `_A_START` / `_A_END` / `_C_END` 常量骨架，但把默认值改回**更符合视觉构图**的位置（比如 husky 起点在 apartment 走廊入口处）
- 在 `compose_two_dog_scene()` 结尾显式调 `check_no_clipping(spec, furniture_bboxes=load_apartment_furniture())` 做验证
- 若默认布局与家具冲突，用错误信息（"husky at (x, y) clips furniture SM_XX"）指导人手动调常量 —— 因为这是手写 demo scene，不做自动 fallback

---

## 5. 验收标准（Q10 决策：4 项全做）

### V1. Dump 数量 sanity check

```python
# tests/tools/test_furniture_collision.py::test_dump_yields_reasonable_count
def test_dump_yields_reasonable_count():
    with open("data/apartment_furniture_map.json") as f:
        data = json.load(f)
    n = len(data["furniture"])
    assert 30 <= n <= 500, (
        f"apartment 家具数 {n} 不在 [30, 500]。"
        f"过少：过滤太狠 / apartment 空；过多：结构 mesh 没过滤掉。"
        f"filter reasons: {data['meta']['filter_reasons']}"
    )
```

### V2. 回归测试：旧穿模场景必须被拒绝

写一个 fixture，把 husky 静态放到"已知的最大件家具中心"，然后 assert `check_no_clipping` 报错。fixture 位置**从 JSON 动态计算**（不 hardcode 数字），避免 dump 后 apartment_mic_origin 调整导致 fixture 失效：

```python
def test_known_clipping_trajectory_rejected():
    furniture = load_apartment_furniture()
    assert furniture, "V2 depends on V1 having produced non-empty map"
    # 选最大件家具（大概率是沙发/床/大桌），把它的中心作为 clipping fixture
    largest = max(furniture, key=lambda b: (b.xy_max_m[0]-b.xy_min_m[0]) * (b.xy_max_m[1]-b.xy_min_m[1]))
    cx = (largest.xy_min_m[0] + largest.xy_max_m[0]) / 2
    cy = (largest.xy_min_m[1] + largest.xy_max_m[1]) / 2
    spec = _make_static_husky_at(cx, cy)  # 辅助：造一个 husky 全 75 帧站在 (cx,cy) 的 spec
    with pytest.raises(AssertionError, match="clips furniture"):
        check_no_clipping(spec, furniture_bboxes=furniture)
```

**依赖**：V1 必须先跑过并产出非空 JSON，否则 V2 无 fixture 可用。

### V3. 随机 compose 成功率 ≥ 80%

```python
def test_random_compose_success_rate():
    furniture = load_apartment_furniture()
    successes = 0
    for seed in range(50):
        try:
            spec = compose_scene(seed, furniture_bboxes=furniture)
            check_no_clipping(spec, furniture_bboxes=furniture)
            successes += 1
        except AssertionError:
            pass
    assert successes / 50 >= 0.80, (
        f"apartment compose 成功率 {successes}/50 < 80%。"
        f"可能 apartment 可行区域太小，需放宽过滤"
    )
```

**若 V3 失败的处理顺序**（不要一步跳到"放宽物理约束"）：
1. 先看 `filter_reasons` 是否合理（有没有明显误伤，比如"墙"关键词匹配掉了非墙的家具）
2. 若过滤合理，加大 `compose_scene` 内部采样次数上限（`for _ in range(N)` 里的 N，从当前值加大到 500 或 1000）
3. 只有 1、2 都做过 V3 仍 fail，才考虑放宽 `WALL_MARGIN_M` / `ANIMAL_MIN_SEP_M`（这是最后手段）

### V4. 可视化验证（人工）

跑 3 个新 seed 的 apartment 渲染：

```bash
for seed in 42 137 999; do
  /data/jzy/miniconda3/envs/spear-env/bin/python \
    tools/gpurir_scenes/run_scene.py \
    --seed $seed --room apartment \
    --out-root tmp/gpurir_scenes_furniture_v1
done
```

产出 3 组视频，用户肉眼确认动物不再从家具里长出。**这一项只有用户 approve 才算通过**（自动化无法判定）。

---

## 6. 风险与开放问题

| 风险 | 影响 | 缓解 |
|---|---|---|
| **R1**：apartment 里家具太多 → free space 太小 → 随机采样成功率 < 80% | V3 fail | 若发生：降低 `ANIMAL_MIN_SEP_M` 或放宽 wall margin，重跑 V3 |
| **R2**：AABB 对斜摆家具过于保守，拒绝很多本可行的路径 | 采样效率下降 | 若 R1 触发但过滤合理，可以引入 OBB 精确判定作为二次检查 |
| **R3**：`get_stable_name_for_actor` 对某些 Kujiale actor 返回空字符串 | 关键词过滤漏一些墙 | dump 时若 name 空，回退到 `actor.get_class().get_name()` |
| **R4**：`GetActorBounds` 对复合 actor（含多个 mesh component）行为不明确 | bbox 可能不准 | dump 完抽样 5 个 actor，在 UE editor 里对照 bbox，验证准确性 |
| **R5**：三层过滤阈值（z=3m, z=0.05m, 20m², 关键词列表）拍脑袋定的 | 可能漏 or 误伤 | V1 的 filter_reasons 计数会暴露；每次 dump 后 review 一次 |
| **R6**：apartment_mic_origin_cm 未来若调整，V2 的 fixture 位置也变 | V2 fixture 需更新 | fixture 位置从家具 JSON 里动态选（"取第一个沙发的 center"）而非 hardcode |

**开放问题（本 spec 不解决，留给未来）**：

- Q_future_1：多个 apartment 变体 → 每个都得 dump 一次 → 是否需要 `data/apartment_*/furniture_map.json` 目录结构？（先做单 apartment_0000，跑通再说）
- Q_future_2：家具语义标签（沙发 / 桌 / 椅）→ QA 生成器需要 → 本 spec 只存 actor_name
- Q_future_3：Procgen 房间的家具 —— 完全另一条路径，用 rule-based solver，不复用本 spec 的 dump 流程

---

## 7. 工作量估计

| 任务 | 估计（人时） |
|---|---|
| dump_apartment_furniture.py | 2h（含 UE 启动调试） |
| 首次 dump 并 review filter_reasons | 1h |
| furniture_map.py | 1.5h |
| scene_spec.py 修改 + 单测 | 1.5h |
| run_render_pass.py + scene_two_dogs.py 修改 | 1h |
| V1-V3 测试 | 2h |
| V4 视觉验证 + 迭代 | 2h（含跑 3 个 seed 渲染） |
| **合计** | **~11 人时** |

单人一天到一天半可完成。

---

## 8. 依赖与前置

- ✅ SPEAR RPC 已支持 `find_actors_by_class + GetActorBounds`（现成代码在 `tools/enumerate_scene_primitives.py`）
- ✅ apartment_0000 map 已在 UE 项目中并可加载
- ✅ Python env 就绪：`/data/jzy/miniconda3/envs/spear-env/bin/python`
- ✅ 现有 `check_no_clipping` / `compose_scene` 结构清楚，可安全扩展

无阻塞。

---

## 9. Out-of-scope 一览（明确不做）

- ❌ RoomSpec 抽象重构（多房间 / procgen 场景走这条路径的架构）
- ❌ Procgen 家具（sofa / table / chair 自动摆到 shoebox）
- ❌ Procgen 房间（尺寸 / 材质 / T60 随机化）
- ❌ 家具语义标签（category = sofa / table / chair 归类）
- ❌ QA 生成器（"the dog next to the couch"）
- ❌ 家具遮挡在音频侧的建模（GPURIR 目前不考虑家具吸声/遮挡）
- ❌ OBB 精确 bbox（R2 缓解方案，只在必要时上）
- ❌ Cosmos sim2real（Cosmos 3 无该能力；Cosmos 2.5 破坏 GT 对齐）

这些每一项都是**未来独立 spec** 的题目。
