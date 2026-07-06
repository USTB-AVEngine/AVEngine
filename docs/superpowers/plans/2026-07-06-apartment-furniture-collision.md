# Apartment 家具碰撞检测 — 实现 Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 apartment 场景的动物轨迹碰撞检测能考虑家具 bbox，消除"哈士奇从沙发/门框里长出"这一类问题。

**Architecture:** 一次性离线用 SPEAR RPC 从 UE dump apartment_0000 所有 `AStaticMeshActor` 的 AABB（UE cm 坐标），三层过滤（z 高度 + 名字关键词 + bbox 面积）掉墙/地/天花板，结果写 JSON 入 git。运行时用小工具模块把 UE cm 转 SceneSpec 米坐标（含 0.1m margin），`check_no_clipping` 与 `compose_scene` 都接受 `furniture_bboxes` 可选参数，apartment 分支加载并传入。

**Tech Stack:** Python 3.11 + SPEAR RPC + numpy + pytest。UE 5.5 在 `DISPLAY=:99` 上运行；conda env `/data/jzy/miniconda3/envs/spear-env/bin/python`。

## Global Constraints

- Python 解释器：`/data/jzy/miniconda3/envs/spear-env/bin/python`（含 `spear_ext`，不要用其它 env）
- SPEAR 项目根：`/data/jzy/code/SPEAR`
- 启动 UE 前必须 `export DISPLAY=:99` 和 `export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`
- 保持 GPURIR 空间 GT 严格对齐 —— **不改**麦克风位置、`_YAW_WORLD_TO_UE_*`、Y-flip 逻辑
- 保持 4 视角固定相机、5 s / 75 帧 / 15 fps / 640×480 输出规格 —— **不改** [run_render_pass.py](../../../tools/gpurir_scenes/run_render_pass.py) 里 `WIDTH/HEIGHT/CAMERA_FOV_DEG/FPS/N_FRAMES`
- Apartment 世界坐标常量 `APARTMENT_MIC_ORIGIN_CM = (-120.0, 80.0, 120.0)` — **不改**；本 plan 里 dump 与 loader 严格与它对齐
- **不做** RoomSpec 重构 / procgen / 语义标签 / QA 生成器 / OBB 精确 bbox / Cosmos
- 每 Task 独立 commit，每 Task 有可运行验证命令

---

## File Structure

**新增**：
- `tools/gpurir_scenes/dump_apartment_furniture.py` — 一次性 dump 脚本
- `tools/gpurir_scenes/furniture_map.py` — JSON 加载 + UE→SceneSpec 坐标转换 + AABB 碰撞查询
- `data/apartment_furniture_map.json` — dump 产物，入 git
- `tests/tools/test_furniture_map.py` — furniture_map 单测
- `tests/tools/test_furniture_collision.py` — V1/V2/V3 集成测试

**修改**：
- `tools/gpurir_scenes/scene_spec.py` — [line 240](../../../tools/gpurir_scenes/scene_spec.py#L240) `check_no_clipping` 与 [line 265](../../../tools/gpurir_scenes/scene_spec.py#L265) `compose_scene` 各加 `furniture_bboxes=None` 参数
- `tools/gpurir_scenes/run_render_pass.py` — apartment 分支加载家具并调 `check_no_clipping`（预留 double-check）
- `tools/gpurir_scenes/scene_two_dogs.py` — 常量与验证接入家具

---

## Task 1: dump_apartment_furniture.py — 从 UE 抓 raw 数据

**Files:**
- Create: `tools/gpurir_scenes/dump_apartment_furniture.py`
- Create: `data/apartment_furniture_map.json` (通过运行脚本产生)

**Interfaces:**
- Consumes: 无（reads SPEAR RPC + UE apartment_0000 map）
- Produces: `data/apartment_furniture_map.json`。schema：
  ```
  {
    "meta": {
      "apartment_map_path": str,
      "dump_date_utc": str (ISO 8601),
      "spear_commit": str (git rev-parse HEAD),
      "ue_version": str,
      "apartment_mic_origin_cm_at_dump": [x, y, z],
      "num_actors_seen": int,
      "num_actors_after_filter": int,
      "filter_reasons": {"z_ceiling": int, "z_floor": int, "name_wall": int,
                        "name_floor": int, "name_ceiling": int, "name_ground": int,
                        "bbox_too_large": int, "kept": int}
    },
    "furniture": [
      {"actor_name": str, "uclass": "AStaticMeshActor",
       "bbox_min_ue_cm": [x, y, z], "bbox_max_ue_cm": [x, y, z],
       "actor_location_ue_cm": [x, y, z],
       "actor_rotation_deg": [roll, pitch, yaw]}
    ]
  }
  ```

**Sub-step order（含自校准环节，spec §4.1）：**
1. 写脚本骨架（枚举 + AABB + 过滤 + JSON 写盘）
2. **首跑 dry-run 模式**（只 print z 分布 + 每个 filter reason 计数，不写 JSON），review 阈值
3. 确认阈值合理后 full-run 写 JSON

---

- [ ] **Step 1.1: 创建 `dump_apartment_furniture.py` 骨架**

```python
"""One-shot offline dump of apartment_0000 static-mesh actor bboxes.

Runs SPEAR RPC, enumerates every AStaticMeshActor in the apartment,
applies three-layer filter (z-height / name-keyword / bbox-area) to
remove walls/floor/ceiling, writes surviving bboxes as UE cm coords to
data/apartment_furniture_map.json for downstream collision checks.

Usage:
    export DISPLAY=:99
    /data/jzy/miniconda3/envs/spear-env/bin/python \\
        tools/gpurir_scenes/dump_apartment_furniture.py --dry-run
    # review z-histogram and filter reasons, then:
    /data/jzy/miniconda3/envs/spear-env/bin/python \\
        tools/gpurir_scenes/dump_apartment_furniture.py --out data/apartment_furniture_map.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

REPO = "/data/jzy/code/SPEAR"
sys.path.insert(0, os.path.join(REPO, "examples"))
sys.path.insert(0, os.path.join(REPO, "tools"))

from render_in_apartment import APARTMENT_MAP, configure_instance  # noqa: E402


# Filter thresholds (UE world Z is from apartment origin, NOT room floor).
# Apartment floor ~27 cm; ceiling ~280 cm. Thresholds are STARTING values.
# After --dry-run, review the z-histogram and adjust if needed.
Z_CEILING_CM = 300.0     # bbox_min_z > this → drop (ceiling)
Z_FLOOR_CM = 5.0         # bbox_max_z < this → drop (floor patch/decal)
NAME_KEYWORDS = ("wall", "floor", "ceiling", "ground")
BBOX_AREA_MAX_CM2 = 200000.0  # x_extent * y_extent > 20 m² → drop (structural mesh)


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def _classify(actor_name: str, bbox_min_z: float, bbox_max_z: float,
              x_extent_cm: float, y_extent_cm: float) -> str:
    """Return filter reason string ("z_ceiling" / "name_wall" / ... / "kept")."""
    if bbox_min_z > Z_CEILING_CM:
        return "z_ceiling"
    if bbox_max_z < Z_FLOOR_CM:
        return "z_floor"
    nl = actor_name.lower()
    for kw in NAME_KEYWORDS:
        if kw in nl:
            return f"name_{kw}"
    if x_extent_cm * y_extent_cm > BBOX_AREA_MAX_CM2:
        return "bbox_too_large"
    return "kept"


def dump_apartment(out_path: str | None, dry_run: bool = False) -> None:
    instance = configure_instance(rpc_port=39002)
    game = instance.get_game()

    from render_in_apartment import APARTMENT_MAP as MAP_PATH

    try:
        with instance.begin_frame():
            actors = game.unreal_service.find_actors_by_class(uclass="AStaticMeshActor")
            print(f"[dump] found {len(actors)} AStaticMeshActor instances", flush=True)

            records = []
            reasons = {"z_ceiling": 0, "z_floor": 0, "name_wall": 0,
                       "name_floor": 0, "name_ceiling": 0, "name_ground": 0,
                       "bbox_too_large": 0, "kept": 0}
            z_bins = [0] * 12  # 0-50, 50-100, ..., 550+ cm

            for actor in actors:
                try:
                    stable = game.unreal_service.get_stable_name_for_actor(
                        actor=actor, include_unreal_name=True
                    )
                except Exception:
                    stable = "<no-stable-name>"
                if not stable:
                    stable = "<empty-name>"

                try:
                    b = actor.GetActorBounds(bOnlyCollidingComponents=False, as_dict=True)
                    origin = b["Origin"]
                    ext = b["BoxExtent"]
                except Exception:
                    continue  # actor with no valid bounds — skip silently

                bbox_min = [origin["x"] - ext["x"], origin["y"] - ext["y"], origin["z"] - ext["z"]]
                bbox_max = [origin["x"] + ext["x"], origin["y"] + ext["y"], origin["z"] + ext["z"]]

                try:
                    loc = actor.K2_GetActorLocation(as_dict=True)
                    loc_list = [loc["x"], loc["y"], loc["z"]]
                except Exception:
                    loc_list = [origin["x"], origin["y"], origin["z"]]

                try:
                    rot = actor.K2_GetActorRotation(as_dict=True)
                    rot_list = [rot["roll"], rot["pitch"], rot["yaw"]]
                except Exception:
                    rot_list = [0.0, 0.0, 0.0]

                x_ext = ext["x"] * 2.0
                y_ext = ext["y"] * 2.0
                bin_idx = min(int(bbox_min[2] / 50), 11)
                z_bins[bin_idx] += 1

                reason = _classify(stable, bbox_min[2], bbox_max[2], x_ext, y_ext)
                reasons[reason] += 1
                if reason != "kept":
                    continue

                records.append({
                    "actor_name": stable,
                    "uclass": "AStaticMeshActor",
                    "bbox_min_ue_cm": bbox_min,
                    "bbox_max_ue_cm": bbox_max,
                    "actor_location_ue_cm": loc_list,
                    "actor_rotation_deg": rot_list,
                })
        with instance.end_frame():
            pass

        print("\n[dump] z-histogram (bbox_min_z, 50 cm bins):", flush=True)
        for i, c in enumerate(z_bins):
            lo = i * 50
            hi = (i + 1) * 50 if i < 11 else "+"
            print(f"  {lo:4d}-{hi:>4} cm: {c}", flush=True)
        print(f"\n[dump] filter reasons: {reasons}", flush=True)
        print(f"[dump] kept {len(records)} / {len(actors)} actors", flush=True)

        # Guard: if any single reason >50% of total, warn (spec §4.1)
        total = sum(reasons.values())
        for r, c in reasons.items():
            if c > 0.5 * total and r != "kept":
                print(f"[dump] WARNING: reason={r} triggered on {c}/{total} "
                      f"(> 50%). Consider adjusting threshold before writing JSON.",
                      flush=True)

        if dry_run:
            print("\n[dump] --dry-run: not writing JSON", flush=True)
            return

        if out_path is None:
            raise SystemExit("[dump] --out is required unless --dry-run")

        payload = {
            "meta": {
                "apartment_map_path": MAP_PATH,
                "dump_date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "spear_commit": _git_head(),
                "ue_version": "5.5",
                "apartment_mic_origin_cm_at_dump": [-120.0, 80.0, 120.0],
                "num_actors_seen": len(actors),
                "num_actors_after_filter": len(records),
                "filter_reasons": reasons,
            },
            "furniture": records,
        }
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[dump] wrote {out_path}", flush=True)

    finally:
        instance.close(force=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None,
                   help="Output JSON path. Required unless --dry-run.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print z-histogram + filter reasons only, don't write JSON.")
    args = p.parse_args()
    dump_apartment(out_path=args.out, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
```

- [ ] **Step 1.2: 首次 dry-run，review 阈值**

Run:
```bash
export DISPLAY=:99
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
/data/jzy/miniconda3/envs/spear-env/bin/python \
  /data/jzy/code/SPEAR/tools/gpurir_scenes/dump_apartment_furniture.py --dry-run 2>&1 | tail -40
```

Expected: 打印 z-histogram + filter reasons + kept count。**人工 review**：
- kept 是否在 30–500 范围内？
- 是否有某个 reason > 50%？若有，人工判断该阈值需不需要调（e.g., 若 `bbox_too_large` 占 60%，可能真实沙发都 > 20 m²，需放宽 `BBOX_AREA_MAX_CM2`）
- z 分布是否合理（家具大多应集中在 30-250 cm）

若 kept < 30 或 > 500，**停下**：先调 `Z_CEILING_CM / Z_FLOOR_CM / NAME_KEYWORDS / BBOX_AREA_MAX_CM2`，再重跑 --dry-run。**不进 Step 1.3 直到 kept 在 [30, 500]**。

- [ ] **Step 1.3: 正式跑，写 JSON**

Run:
```bash
export DISPLAY=:99
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
/data/jzy/miniconda3/envs/spear-env/bin/python \
  /data/jzy/code/SPEAR/tools/gpurir_scenes/dump_apartment_furniture.py \
  --out /data/jzy/code/SPEAR/data/apartment_furniture_map.json 2>&1 | tail -20
```

Expected: 最后一行 `[dump] wrote /data/jzy/code/SPEAR/data/apartment_furniture_map.json`。

Verify:
```bash
python3 -c "import json; d=json.load(open('/data/jzy/code/SPEAR/data/apartment_furniture_map.json')); print('kept:', d['meta']['num_actors_after_filter']); print('first:', d['furniture'][0]['actor_name'])"
```

- [ ] **Step 1.4: Commit**

```bash
cd /data/jzy/code/SPEAR
git add tools/gpurir_scenes/dump_apartment_furniture.py data/apartment_furniture_map.json
git commit -m "feat(furniture): dump apartment_0000 static-mesh bboxes to JSON

- New tools/gpurir_scenes/dump_apartment_furniture.py: one-shot SPEAR RPC
  script that enumerates AStaticMeshActor in apartment_0000, applies
  z-height + name-keyword + bbox-area filter to drop walls/floor/ceiling,
  writes AABB in UE cm to data/apartment_furniture_map.json.
- Filter thresholds are self-calibrated via --dry-run z-histogram before
  first real write."
```

---

## Task 2: furniture_map.py — 加载 + 坐标转换 + 碰撞查询

**Files:**
- Create: `tools/gpurir_scenes/furniture_map.py`
- Test: `tests/tools/test_furniture_map.py`

**Interfaces:**
- Consumes: `data/apartment_furniture_map.json` (Task 1 产物)
- Produces:
  - `FurnitureBBox` dataclass: `(actor_name: str, xy_min_m: tuple[float,float], xy_max_m: tuple[float,float], z_min_m: float, z_max_m: float)`
  - `load_apartment_furniture(json_path=..., apartment_mic_origin_cm=..., apartment_mic_pos_scene_m=(2.6, 2.2), y_flip=True, margin_m=0.1) -> list[FurnitureBBox]`
  - `any_bbox_hits_point(bboxes: list[FurnitureBBox], x: float, y: float) -> bool`
  - `any_bbox_hits_series(bboxes: list[FurnitureBBox], xy_series: np.ndarray) -> bool` — xy_series shape (K, 2)

- [ ] **Step 2.1: 写 failing test（坐标转换 + 碰撞查询）**

Create `/data/jzy/code/SPEAR/tests/tools/test_furniture_map.py`:

```python
"""Unit tests for furniture_map.py: coord conversion + AABB queries."""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

import sys
sys.path.insert(0, "/data/jzy/code/SPEAR/tools")

from gpurir_scenes.furniture_map import (
    FurnitureBBox,
    load_apartment_furniture,
    any_bbox_hits_point,
    any_bbox_hits_series,
)


def _write_fake_json(path):
    """Create a tiny fixture JSON with one 100x100 cm sofa at UE origin (0,0)."""
    payload = {
        "meta": {
            "apartment_map_path": "/Game/fake/Maps/fake",
            "dump_date_utc": "2026-07-06T00:00:00Z",
            "spear_commit": "test",
            "ue_version": "5.5",
            "apartment_mic_origin_cm_at_dump": [-120.0, 80.0, 120.0],
            "num_actors_seen": 1,
            "num_actors_after_filter": 1,
            "filter_reasons": {"kept": 1},
        },
        "furniture": [
            {
                "actor_name": "SM_FakeSofa",
                "uclass": "AStaticMeshActor",
                # 100x100x90 cm bbox at UE origin
                "bbox_min_ue_cm": [-50.0, -50.0, 0.0],
                "bbox_max_ue_cm": [50.0, 50.0, 90.0],
                "actor_location_ue_cm": [0.0, 0.0, 45.0],
                "actor_rotation_deg": [0.0, 0.0, 0.0],
            }
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f)


def test_load_produces_one_bbox_with_margin():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tf:
        _write_fake_json(tf.name)
        bboxes = load_apartment_furniture(json_path=tf.name)
    os.unlink(tf.name)
    assert len(bboxes) == 1
    b = bboxes[0]
    assert b.actor_name == "SM_FakeSofa"
    # Fake bbox spans UE x ∈ [-50, 50] cm. With apartment_mic_origin_cm[0] = -120,
    # scene_m origin at (2.6, 2.2): x_scene = (x_ue - (-120))/100 + 2.6.
    # ⇒ x=-50 UE → (70/100)+2.6 = 3.3 m ; x=50 UE → (170/100)+2.6 = 4.3 m
    # With 0.1 m margin: xy_min_m[0] = 3.2, xy_max_m[0] = 4.4
    assert b.xy_min_m[0] == pytest.approx(3.2, abs=1e-6)
    assert b.xy_max_m[0] == pytest.approx(4.4, abs=1e-6)
    # y with flip: y=-50 UE → -((-50-80)/100)+2.2 = 1.3+2.2 = 3.5; y=50 UE → -((50-80)/100)+2.2 = 0.3+2.2 = 2.5
    # y_flip swaps min<->max ordering; loader must produce xy_min_m[1] < xy_max_m[1].
    assert b.xy_min_m[1] < b.xy_max_m[1]
    assert b.xy_min_m[1] == pytest.approx(2.4, abs=1e-6)   # 2.5 - 0.1 margin
    assert b.xy_max_m[1] == pytest.approx(3.6, abs=1e-6)   # 3.5 + 0.1 margin


def test_any_bbox_hits_point_inside_and_outside():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tf:
        _write_fake_json(tf.name)
        bboxes = load_apartment_furniture(json_path=tf.name)
    os.unlink(tf.name)
    # Center of fake bbox (in scene m): x ≈ 3.8, y ≈ 3.0
    assert any_bbox_hits_point(bboxes, 3.8, 3.0) is True
    # A point clearly outside: x=0, y=0
    assert any_bbox_hits_point(bboxes, 0.0, 0.0) is False


def test_any_bbox_hits_series_one_frame_inside():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tf:
        _write_fake_json(tf.name)
        bboxes = load_apartment_furniture(json_path=tf.name)
    os.unlink(tf.name)
    # 3-frame trajectory that passes through the bbox at frame 1
    xy = np.array([[0.0, 0.0], [3.8, 3.0], [10.0, 10.0]])
    assert any_bbox_hits_series(bboxes, xy) is True
    # trajectory that never enters
    xy_clean = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    assert any_bbox_hits_series(bboxes, xy_clean) is False


def test_empty_bboxes_never_hit():
    xy = np.array([[3.8, 3.0]])
    assert any_bbox_hits_point([], 3.8, 3.0) is False
    assert any_bbox_hits_series([], xy) is False
```

- [ ] **Step 2.2: 运行测试验证 FAIL**

```bash
cd /data/jzy/code/SPEAR
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest tests/tools/test_furniture_map.py -v 2>&1 | tail -15
```

Expected: 全部 ERROR/FAIL，因为 `furniture_map` 模块还不存在。

- [ ] **Step 2.3: 实现 `furniture_map.py`**

Create `/data/jzy/code/SPEAR/tools/gpurir_scenes/furniture_map.py`:

```python
"""Load apartment furniture bbox dump, convert UE cm → SceneSpec m coords,
and expose fast AABB point/series hit queries.

Data file: data/apartment_furniture_map.json (produced by
dump_apartment_furniture.py).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

# Must match tools/gpurir_scenes/run_render_pass.py::APARTMENT_MIC_ORIGIN_CM.
# If run_render_pass changes it, the loader here needs the same value —
# but only at load-time (not stored in the JSON), so no re-dump required.
DEFAULT_APARTMENT_MIC_ORIGIN_CM = (-120.0, 80.0, 120.0)
DEFAULT_APARTMENT_MIC_POS_SCENE_M = (2.6, 2.2)
DEFAULT_JSON_PATH = "/data/jzy/code/SPEAR/data/apartment_furniture_map.json"


@dataclass(frozen=True)
class FurnitureBBox:
    """AABB in SceneSpec m coordinates, already inflated by margin."""
    actor_name: str
    xy_min_m: tuple  # (x_min, y_min)
    xy_max_m: tuple  # (x_max, y_max)
    z_min_m: float
    z_max_m: float

    def contains_xy(self, x: float, y: float) -> bool:
        return (self.xy_min_m[0] <= x <= self.xy_max_m[0]
                and self.xy_min_m[1] <= y <= self.xy_max_m[1])


def _ue_cm_to_scene_m(
    x_ue_cm: float, y_ue_cm: float, z_ue_cm: float,
    mic_origin_cm: tuple, mic_pos_scene_m: tuple, y_flip: bool
) -> tuple[float, float, float]:
    """Mirror of run_render_pass._world_from_scene() but inverse direction.

    Forward (run_render_pass): scene_m → UE cm.
    Here we invert: UE cm → scene_m.
    """
    x_scene = (x_ue_cm - mic_origin_cm[0]) / 100.0 + mic_pos_scene_m[0]
    if y_flip:
        y_scene = -(y_ue_cm - mic_origin_cm[1]) / 100.0 + mic_pos_scene_m[1]
    else:
        y_scene = (y_ue_cm - mic_origin_cm[1]) / 100.0 + mic_pos_scene_m[1]
    z_scene = z_ue_cm / 100.0
    return x_scene, y_scene, z_scene


def load_apartment_furniture(
    json_path: str = DEFAULT_JSON_PATH,
    apartment_mic_origin_cm: tuple = DEFAULT_APARTMENT_MIC_ORIGIN_CM,
    apartment_mic_pos_scene_m: tuple = DEFAULT_APARTMENT_MIC_POS_SCENE_M,
    y_flip: bool = True,
    margin_m: float = 0.1,
) -> list[FurnitureBBox]:
    """Load JSON, transform UE cm → SceneSpec m, inflate by margin_m.

    Returns list of FurnitureBBox in scene m coordinates, ready for
    scene_spec.check_no_clipping / compose_scene consumption.
    """
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r") as f:
        data = json.load(f)
    bboxes = []
    for rec in data.get("furniture", []):
        bmin_ue = rec["bbox_min_ue_cm"]
        bmax_ue = rec["bbox_max_ue_cm"]
        # Transform each corner independently; y_flip may swap min/max order.
        x1, y1, z1 = _ue_cm_to_scene_m(
            bmin_ue[0], bmin_ue[1], bmin_ue[2],
            apartment_mic_origin_cm, apartment_mic_pos_scene_m, y_flip)
        x2, y2, z2 = _ue_cm_to_scene_m(
            bmax_ue[0], bmax_ue[1], bmax_ue[2],
            apartment_mic_origin_cm, apartment_mic_pos_scene_m, y_flip)
        # Re-order so min < max after possible y-flip.
        xy_min = (min(x1, x2) - margin_m, min(y1, y2) - margin_m)
        xy_max = (max(x1, x2) + margin_m, max(y1, y2) + margin_m)
        z_min = min(z1, z2)
        z_max = max(z1, z2)
        bboxes.append(FurnitureBBox(
            actor_name=rec["actor_name"],
            xy_min_m=xy_min,
            xy_max_m=xy_max,
            z_min_m=z_min,
            z_max_m=z_max,
        ))
    return bboxes


def any_bbox_hits_point(bboxes: list, x: float, y: float) -> bool:
    for b in bboxes:
        if b.contains_xy(x, y):
            return True
    return False


def any_bbox_hits_series(bboxes: list, xy_series: np.ndarray) -> bool:
    """xy_series shape (K, 2). Return True if ANY frame in ANY bbox."""
    if len(bboxes) == 0 or xy_series.size == 0:
        return False
    # Vectorized: build (K,) bool via per-bbox mask, OR together.
    xs = xy_series[:, 0]
    ys = xy_series[:, 1]
    for b in bboxes:
        inside = (
            (xs >= b.xy_min_m[0]) & (xs <= b.xy_max_m[0])
            & (ys >= b.xy_min_m[1]) & (ys <= b.xy_max_m[1])
        )
        if inside.any():
            return True
    return False
```

- [ ] **Step 2.4: 运行测试验证 PASS**

```bash
cd /data/jzy/code/SPEAR
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest tests/tools/test_furniture_map.py -v 2>&1 | tail -15
```

Expected: 4 passed.

- [ ] **Step 2.5: Commit**

```bash
cd /data/jzy/code/SPEAR
git add tools/gpurir_scenes/furniture_map.py tests/tools/test_furniture_map.py
git commit -m "feat(furniture): furniture_map loader + AABB hit queries

- FurnitureBBox dataclass in SceneSpec m coords, pre-inflated by 0.1 m margin.
- load_apartment_furniture() inverts the run_render_pass UE→scene transform
  (including apartment Y-flip); no coord duplication required at consumer sites.
- any_bbox_hits_point / any_bbox_hits_series expose the two queries
  scene_spec needs.
- 4 unit tests cover load, hit-in, hit-out, empty-bboxes."
```

---

## Task 3: scene_spec.py — 接入 furniture_bboxes 参数

**Files:**
- Modify: `tools/gpurir_scenes/scene_spec.py` — [line 240 `check_no_clipping`](../../../tools/gpurir_scenes/scene_spec.py#L240), [line 265 `compose_scene`](../../../tools/gpurir_scenes/scene_spec.py#L265)
- Test: `tests/tools/test_furniture_map.py` (extended — 加集成测试)

**Interfaces:**
- Consumes: `furniture_map.FurnitureBBox`, `any_bbox_hits_series`
- Produces:
  - `check_no_clipping(spec, wall_margin_m=..., min_sep_m=..., furniture_bboxes=None) -> None` — 家具穿模时 raise AssertionError with `"clips furniture"` in msg
  - `compose_scene(seed, furniture_bboxes=None) -> SceneSpec` — 采样时避开家具

- [ ] **Step 3.1: 加 failing test — check_no_clipping 家具分支**

Append to `/data/jzy/code/SPEAR/tests/tools/test_furniture_map.py`:

```python
# ---- integration: scene_spec + furniture ----
from gpurir_scenes import scene_spec
from gpurir_scenes.scene_spec import (
    SceneSpec, AnimalPlacement, ROOM_SIZE_M, MIC_POS_M, T60_S, SOURCE_HEIGHT_M
)


def _static_husky_spec_at(x: float, y: float) -> SceneSpec:
    """Build a SceneSpec with dog_husky standing still at (x, y) for all 75 frames."""
    traj = np.tile(np.array([x, y, SOURCE_HEIGHT_M]), (75, 1))
    yaw = np.zeros(75)
    return SceneSpec(
        seed=0,
        room_size_m=ROOM_SIZE_M,
        t60_s=T60_S,
        mic_pos_m=MIC_POS_M,
        animals=[AnimalPlacement(
            tag="dog_husky", is_animated=True,
            trajectory_m=traj, yaw_deg=yaw,
        )],
    )


def test_check_no_clipping_with_furniture_rejects_inside_bbox():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tf:
        _write_fake_json(tf.name)
        bboxes = load_apartment_furniture(json_path=tf.name)
    os.unlink(tf.name)
    # (3.8, 3.0) is inside the fake sofa bbox
    spec = _static_husky_spec_at(3.8, 3.0)
    with pytest.raises(AssertionError, match="clips furniture"):
        scene_spec.check_no_clipping(spec, furniture_bboxes=bboxes)


def test_check_no_clipping_no_furniture_backward_compat():
    """Not passing furniture_bboxes → old behaviour unchanged (no furniture check)."""
    spec = _static_husky_spec_at(2.6, 2.2)  # at mic center, safely away from walls
    scene_spec.check_no_clipping(spec)  # no exception


def test_check_no_clipping_with_furniture_passes_when_safe():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tf:
        _write_fake_json(tf.name)
        bboxes = load_apartment_furniture(json_path=tf.name)
    os.unlink(tf.name)
    spec = _static_husky_spec_at(1.5, 1.5)  # far from fake sofa
    scene_spec.check_no_clipping(spec, furniture_bboxes=bboxes)  # no exception


def test_compose_scene_avoids_furniture():
    """With a fake sofa blocking (3.8, 3.0), compose_scene shouldn't put
    animal centers there. We check 10 seeds."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tf:
        _write_fake_json(tf.name)
        bboxes = load_apartment_furniture(json_path=tf.name)
    os.unlink(tf.name)
    for seed in range(10):
        try:
            spec = scene_spec.compose_scene(seed, furniture_bboxes=bboxes)
        except RuntimeError:
            continue
        for a in spec.animals:
            xy = scene_spec._placement_xy_series(a)
            assert not any_bbox_hits_series(bboxes, xy), (
                f"seed {seed}: {a.tag} hits fake sofa"
            )
```

- [ ] **Step 3.2: 运行测试验证 FAIL（scene_spec 尚未支持 furniture_bboxes）**

```bash
cd /data/jzy/code/SPEAR
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest tests/tools/test_furniture_map.py -v 2>&1 | tail -20
```

Expected: 前 4 个测试 PASS，后 4 个 FAIL/ERROR（unexpected keyword 'furniture_bboxes' 或家具没被拒绝）。

- [ ] **Step 3.3: 修改 `check_no_clipping`**

在 `/data/jzy/code/SPEAR/tools/gpurir_scenes/scene_spec.py` 顶部（其他 import 附近）加：

```python
from gpurir_scenes.furniture_map import any_bbox_hits_series  # noqa: E402
```

修改 [line 240 `check_no_clipping`](../../../tools/gpurir_scenes/scene_spec.py#L240) 的函数签名和末尾：

将
```python
def check_no_clipping(spec, wall_margin_m=WALL_MARGIN_M, min_sep_m=ANIMAL_MIN_SEP_M):
```
改为
```python
def check_no_clipping(spec, wall_margin_m=WALL_MARGIN_M, min_sep_m=ANIMAL_MIN_SEP_M,
                     furniture_bboxes=None):
```

在函数末尾（`min_clearance` assertion 之后、`return` 之前）添加：

```python
    # Furniture clipping check (only when a room supplies furniture, e.g. apartment)
    if furniture_bboxes:
        for a in spec.animals:
            xy_series = _placement_xy_series(a)
            if any_bbox_hits_series(furniture_bboxes, xy_series):
                raise AssertionError(
                    f"{a.tag} clips furniture at some frame "
                    f"(bbox_count={len(furniture_bboxes)})"
                )
```

- [ ] **Step 3.4: 修改 `compose_scene`**

修改 [line 265 `compose_scene`](../../../tools/gpurir_scenes/scene_spec.py#L265) 的签名与循环体：

将
```python
def compose_scene(seed: int) -> SceneSpec:
```
改为
```python
def compose_scene(seed: int, furniture_bboxes=None) -> SceneSpec:
```

在两处 `cand_series = ...` 之后（现有 `_min_pairwise_clearance_series(...) >= 0.0` 检查之前），加一段家具拒绝逻辑。

**具体位置 1**：动画分支（现有 [line 279-286](../../../tools/gpurir_scenes/scene_spec.py#L279-L286)），将
```python
                traj, yaw = _generate_trajectory(rng, ROOM_SIZE_M, tag)
                cand_series = traj[:, :2]
                if _min_pairwise_clearance_series(tag, cand_series, animals) >= 0.0:
```
改为
```python
                traj, yaw = _generate_trajectory(rng, ROOM_SIZE_M, tag)
                cand_series = traj[:, :2]
                if furniture_bboxes and any_bbox_hits_series(furniture_bboxes, cand_series):
                    continue
                if _min_pairwise_clearance_series(tag, cand_series, animals) >= 0.0:
```

**具体位置 2**：静态分支（现有 [line 288-291](../../../tools/gpurir_scenes/scene_spec.py#L288-L291)），将
```python
                pos = _sample_static_pos(rng, ROOM_SIZE_M, MIC_POS_M, tag)
                cand_series = np.array([[pos[0], pos[1]]])
                if _min_pairwise_clearance_series(tag, cand_series, animals) >= 0.0:
```
改为
```python
                pos = _sample_static_pos(rng, ROOM_SIZE_M, MIC_POS_M, tag)
                cand_series = np.array([[pos[0], pos[1]]])
                if furniture_bboxes and any_bbox_hits_series(furniture_bboxes, cand_series):
                    continue
                if _min_pairwise_clearance_series(tag, cand_series, animals) >= 0.0:
```

**最后一行 `check_no_clipping(spec)` 也传家具**（现有 [line 327](../../../tools/gpurir_scenes/scene_spec.py#L327)），将
```python
    check_no_clipping(spec)
    return spec
```
改为
```python
    check_no_clipping(spec, furniture_bboxes=furniture_bboxes)
    return spec
```

- [ ] **Step 3.5: 运行测试验证 PASS**

```bash
cd /data/jzy/code/SPEAR
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest tests/tools/test_furniture_map.py -v 2>&1 | tail -15
```

Expected: 8 passed.

- [ ] **Step 3.6: 回归 — 现有 scene_spec 测试仍过**

```bash
cd /data/jzy/code/SPEAR
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest tests/tools/ -v -k "not furniture" 2>&1 | tail -20
```

Expected: 现有测试全 PASS，说明向后兼容没破。

- [ ] **Step 3.7: Commit**

```bash
cd /data/jzy/code/SPEAR
git add tools/gpurir_scenes/scene_spec.py tests/tools/test_furniture_map.py
git commit -m "feat(furniture): scene_spec accepts furniture_bboxes param

- check_no_clipping(spec, ..., furniture_bboxes=None): raises AssertionError
  with 'clips furniture' when any animal xy_series enters any bbox.
- compose_scene(seed, furniture_bboxes=None): rejects candidate trajectories
  that hit furniture, forcing re-sample.
- Both signatures backward-compatible (None → old behaviour, verified by
  regression test).
- Integration tests use a fake single-sofa fixture JSON so no dependency
  on the actual apartment dump."
```

---

## Task 4: run_render_pass.py — apartment 分支加载家具并 assert

**Files:**
- Modify: `tools/gpurir_scenes/run_render_pass.py` — `run_render_pass()` 函数体的 apartment 分支

**Interfaces:**
- Consumes: `furniture_map.load_apartment_furniture`, `scene_spec.check_no_clipping`
- Produces: 无新接口；只是在 apartment 场景 render 前做双重 assert（compose_scene 已经查过一次，这里防手写 spec 漏检）

- [ ] **Step 4.1: 找到 apartment 分支的位置**

Run:
```bash
grep -n "room == \"apartment\"\|instance = configure_instance\|_spawn_shoebox" /data/jzy/code/SPEAR/tools/gpurir_scenes/run_render_pass.py | head -5
```

Expected: 输出几行含 apartment 分支位置的行号（`configure_instance` 是 apartment 分支的入口）。

- [ ] **Step 4.2: 修改 `run_render_pass`**

**首先扩展现有的 scene_spec import**（[line 39](../../../tools/gpurir_scenes/run_render_pass.py#L39) 当前只 import 了 `compose_scene, N_FRAMES, FPS, MIC_POS_M`，缺 `check_no_clipping`）：

将
```python
from gpurir_scenes.scene_spec import compose_scene, N_FRAMES, FPS, MIC_POS_M  # noqa: E402
```
改为
```python
from gpurir_scenes.scene_spec import compose_scene, check_no_clipping, N_FRAMES, FPS, MIC_POS_M  # noqa: E402
from gpurir_scenes.furniture_map import load_apartment_furniture  # noqa: E402
```

在 `run_render_pass(spec, room, out_dir)` 函数开头（`assert room in (...)` 之后，`configure_instance` 之前）插入：

```python
    # Apartment renders must respect furniture bboxes. Even when the spec came
    # from compose_scene (which already avoids furniture), hand-authored specs
    # (e.g. scene_two_dogs) skip that path — so double-check here before we
    # spend UE render time on a clipping scene.
    if room == "apartment":
        try:
            furniture_bboxes = load_apartment_furniture()
        except Exception as e:
            print(f"[render] WARN could not load apartment furniture map: {e}",
                  flush=True)
            furniture_bboxes = None
        if furniture_bboxes:
            check_no_clipping(spec, furniture_bboxes=furniture_bboxes)
```

- [ ] **Step 4.3: 验证 apartment two_dogs 仍能跑通**

Run:
```bash
cd /data/jzy/code/SPEAR
export DISPLAY=:99
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
# 只跑 apartment 分支一次做 smoke（skip audio 复用旧 wav）
/data/jzy/miniconda3/envs/spear-env/bin/python \
  tools/gpurir_scenes/scene_two_dogs.py --skip-audio 2>&1 | tail -10
```

Expected: 最后一行 `TWO_DOGS_DONE ...`；中间不应有 `AssertionError: clips furniture`（当前 two_dogs 常量已经调过避开家具）。若失败 → 进入 Task 5 调 two_dogs 常量。

- [ ] **Step 4.4: Commit**

```bash
cd /data/jzy/code/SPEAR
git add tools/gpurir_scenes/run_render_pass.py
git commit -m "feat(furniture): apartment render checks furniture before spawn

- run_render_pass loads apartment_furniture_map.json in the apartment
  branch and double-checks the spec against furniture bboxes.
- Guards against hand-authored specs that bypass compose_scene's
  furniture-avoidance path (e.g. scene_two_dogs).
- Missing/unreadable JSON downgrades to WARN, not fatal — keeps
  apartment usable during initial rollout."
```

---

## Task 5: scene_two_dogs.py — 常量清理 + 显式家具校验

**Files:**
- Modify: `tools/gpurir_scenes/scene_two_dogs.py` — 常量 `_A_START` / `_A_END` / `_C_END` / `_STATIC_XY`（[line 58-73](../../../tools/gpurir_scenes/scene_two_dogs.py#L58-L73)）、`compose_two_dog_scene()` 尾部

**Interfaces:**
- Consumes: 现有的 `check_no_clipping` + `load_apartment_furniture`
- Produces: 一个手写 spec，能通过 apartment 家具检查

- [ ] **Step 5.1: 观察当前常量是否已通过家具检查**

Run（sanity — Task 4 已跑过 two_dogs，这一步只是确认状态）:

```bash
cd /data/jzy/code/SPEAR
/data/jzy/miniconda3/envs/spear-env/bin/python -c "
import sys; sys.path.insert(0, 'tools')
from gpurir_scenes.scene_two_dogs import compose_two_dog_scene
from gpurir_scenes.scene_spec import check_no_clipping
from gpurir_scenes.furniture_map import load_apartment_furniture
spec = compose_two_dog_scene()
furniture = load_apartment_furniture()
print(f'furniture_count = {len(furniture)}')
try:
    check_no_clipping(spec, furniture_bboxes=furniture)
    print('PASS: current two_dogs constants clear of apartment furniture')
except AssertionError as e:
    print(f'FAIL: {e}')
    print('Need to retune constants in scene_two_dogs.py')
"
```

Expected: 若 PASS → 常量已 OK，仍执行 Step 5.2 加显式验证；若 FAIL → 记录错误信息，Step 5.2 会用来指导常量调整。

- [ ] **Step 5.2: 在 `compose_two_dog_scene` 末尾加显式家具校验**

修改 `/data/jzy/code/SPEAR/tools/gpurir_scenes/scene_two_dogs.py`：

在文件顶部 import 段（`from gpurir_scenes.scene_spec import ...` 之后）加：

```python
from gpurir_scenes.furniture_map import load_apartment_furniture  # noqa: E402
```

在 `compose_two_dog_scene()` 尾部（现有 `check_no_clipping(spec)` 调用之后、`return spec` 之前）添加：

```python
    # Extra apartment-furniture check: this hand-authored scene targets both
    # shoebox and apartment; ensure the layout also survives apartment furniture.
    apt_furniture = load_apartment_furniture()
    if apt_furniture:
        try:
            check_no_clipping(spec, furniture_bboxes=apt_furniture)
        except AssertionError as e:
            raise RuntimeError(
                f"scene_two_dogs layout clips apartment furniture: {e}\n"
                f"Retune _A_START / _A_END / _C_END / _STATIC_XY constants "
                f"in scene_two_dogs.py to avoid the reported furniture."
            ) from e
```

- [ ] **Step 5.3: 若 Step 5.1 报 FAIL，手工调常量**

（仅当 5.1 FAIL 时执行）根据 5.1 的错误信息中 `clips furniture` 的动物 tag，人工把常量移到无家具区。可用 dump JSON 的 `bbox_min_ue_cm` / `bbox_max_ue_cm` 转 scene m 后作为参考。调完再跑 5.1 直到 PASS。

- [ ] **Step 5.4: 端到端 smoke —— 两个房间都渲染**

```bash
cd /data/jzy/code/SPEAR
export DISPLAY=:99
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
/data/jzy/miniconda3/envs/spear-env/bin/python \
  tools/gpurir_scenes/scene_two_dogs.py --skip-audio 2>&1 | tail -10
```

Expected: `TWO_DOGS_DONE`。产物应在 `tmp/gpurir_scenes_v1/two_dogs/{apartment,shoebox}/view*.mp4`。

- [ ] **Step 5.5: Commit**

```bash
cd /data/jzy/code/SPEAR
git add tools/gpurir_scenes/scene_two_dogs.py
git commit -m "feat(furniture): scene_two_dogs explicitly validates apartment layout

- compose_two_dog_scene() now runs check_no_clipping against apartment
  furniture at build time, raising a clear RuntimeError telling the
  operator which constants to retune.
- No more silent 'husky x <= 2.85' style hacks — the constants speak
  for themselves and the furniture map enforces correctness."
```

---

## Task 6: 验收测试 V1/V2/V3

**Files:**
- Create: `tests/tools/test_furniture_collision.py`

**Interfaces:**
- Consumes: 全部 Task 1-3 产物
- Produces: 3 项自动化验收（V4 是人工视觉，见 Task 7）

- [ ] **Step 6.1: 写 V1/V2/V3 测试**

Create `/data/jzy/code/SPEAR/tests/tools/test_furniture_collision.py`:

```python
"""V1/V2/V3 acceptance tests for apartment furniture collision (spec §5)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, "/data/jzy/code/SPEAR/tools")

from gpurir_scenes.furniture_map import (
    load_apartment_furniture,
    any_bbox_hits_series,
)
from gpurir_scenes import scene_spec
from gpurir_scenes.scene_spec import (
    SceneSpec, AnimalPlacement, ROOM_SIZE_M, MIC_POS_M, T60_S, SOURCE_HEIGHT_M,
)

FURNITURE_JSON = "/data/jzy/code/SPEAR/data/apartment_furniture_map.json"


def _apartment_json_exists() -> bool:
    return os.path.exists(FURNITURE_JSON)


pytestmark = pytest.mark.skipif(
    not _apartment_json_exists(),
    reason=f"{FURNITURE_JSON} not present — run Task 1 first",
)


# ---- V1: Dump 数量 sanity ---------------------------------------------------
def test_V1_dump_yields_reasonable_count():
    with open(FURNITURE_JSON) as f:
        data = json.load(f)
    n = len(data["furniture"])
    reasons = data["meta"]["filter_reasons"]
    assert 30 <= n <= 500, (
        f"apartment kept-furniture count = {n} not in [30, 500].\n"
        f"Too few → filter too aggressive; too many → structural mesh leaked.\n"
        f"filter_reasons: {reasons}"
    )


# ---- V2: 回归 — 已知穿模场景必须被拒绝 --------------------------------------
def _static_husky_spec_at(x: float, y: float) -> SceneSpec:
    traj = np.tile(np.array([x, y, SOURCE_HEIGHT_M]), (75, 1))
    yaw = np.zeros(75)
    return SceneSpec(
        seed=0, room_size_m=ROOM_SIZE_M, t60_s=T60_S, mic_pos_m=MIC_POS_M,
        animals=[AnimalPlacement(
            tag="dog_husky", is_animated=True,
            trajectory_m=traj, yaw_deg=yaw,
        )],
    )


def test_V2_known_clipping_trajectory_rejected():
    furniture = load_apartment_furniture()
    assert furniture, "V2 requires V1's non-empty dump"
    # Pick the largest-area furniture (probably a sofa/table/bed)
    largest = max(
        furniture,
        key=lambda b: (b.xy_max_m[0] - b.xy_min_m[0])
                    * (b.xy_max_m[1] - b.xy_min_m[1]),
    )
    cx = (largest.xy_min_m[0] + largest.xy_max_m[0]) / 2
    cy = (largest.xy_min_m[1] + largest.xy_max_m[1]) / 2
    spec = _static_husky_spec_at(cx, cy)
    with pytest.raises(AssertionError, match="clips furniture"):
        scene_spec.check_no_clipping(spec, furniture_bboxes=furniture)


# ---- V3: 随机 compose 成功率 ≥ 80% ------------------------------------------
def test_V3_random_compose_success_rate():
    furniture = load_apartment_furniture()
    successes = 0
    total = 50
    failures = []
    for seed in range(total):
        try:
            spec = scene_spec.compose_scene(seed, furniture_bboxes=furniture)
            # double-check: no animal hits furniture
            for a in spec.animals:
                xy = scene_spec._placement_xy_series(a)
                if any_bbox_hits_series(furniture, xy):
                    raise AssertionError(f"seed {seed}: {a.tag} hits furniture")
            successes += 1
        except (AssertionError, RuntimeError) as e:
            failures.append((seed, str(e)))
    ratio = successes / total
    msg = (f"apartment compose success rate = {successes}/{total} = {ratio:.0%}. "
           f"First 3 failures: {failures[:3]}")
    assert ratio >= 0.80, msg
```

- [ ] **Step 6.2: 运行 V1/V2/V3**

```bash
cd /data/jzy/code/SPEAR
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest \
  tests/tools/test_furniture_collision.py -v 2>&1 | tail -20
```

Expected: 3 passed.

**若 V3 fail**（成功率 < 80%），按 spec §5 V3 的处理顺序：
1. 先看 filter_reasons：是否有大面积误伤（比如 `bbox_too_large > 30%`）？若是，调 Task 1 阈值重跑 dump
2. 若过滤合理，加大 `scene_spec.PLACEMENT_TRIES`（当前值查一下：`grep PLACEMENT_TRIES /data/jzy/code/SPEAR/tools/gpurir_scenes/scene_spec.py`），从 X 加到 X*3
3. 只有 1、2 都做过 V3 仍 fail，才动 `WALL_MARGIN_M` / `ANIMAL_MIN_SEP_M`

调好后再跑 6.2 直到 PASS。

- [ ] **Step 6.3: Commit**

```bash
cd /data/jzy/code/SPEAR
git add tests/tools/test_furniture_collision.py
git commit -m "test(furniture): V1/V2/V3 acceptance tests

- V1: dump yields 30-500 furniture actors after filter.
- V2: static husky at largest-furniture center is rejected by
  check_no_clipping (fixture dynamic from JSON, not hardcoded).
- V3: compose_scene with furniture succeeds on ≥ 40/50 seeds."
```

---

## Task 7: V4 视觉验证（人工）

**Files:** 无新文件；只是运行现有 pipeline 并让用户看图。

**Interfaces:** 无。产出的是**用户是否 approve** 的决策，不是代码。

- [ ] **Step 7.1: 跑 3 个 seed 的 apartment 渲染**

```bash
cd /data/jzy/code/SPEAR
export DISPLAY=:99
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
mkdir -p tmp/gpurir_scenes_furniture_v1
for seed in 42 137 999; do
  /data/jzy/miniconda3/envs/spear-env/bin/python \
    tools/gpurir_scenes/run_scene.py \
    --seed $seed \
    --out-root tmp/gpurir_scenes_furniture_v1 2>&1 | tail -5
done
```

Expected: 每个 seed 都 `SCENE_DONE`，产物在 `tmp/gpurir_scenes_furniture_v1/scene_{seed:04d}/{apartment,shoebox}/view*_with_audio.mp4`。

**若某个 seed compose 就失败**（`could not place ... without clipping`）：这是**正常**（apartment 家具挤，某些 tag 组合摆不下），只要 3 个 seed 里至少 2 个成功即可。若 3 个全失败，回 Task 6 检查 V3 阈值。

- [ ] **Step 7.2: 展示给用户看图**

从每个成功 seed 的 apartment view0 抽 4 张关键帧（f0/f25/f50/f74）：

```bash
cd /data/jzy/code/SPEAR/tmp/gpurir_scenes_furniture_v1
for seed_dir in scene_*/; do
  for f in 0 25 50 74; do
    ffmpeg -y -i "${seed_dir}apartment/view0.mp4" \
      -vf "select=eq(n\,$f)" -vframes 1 \
      "${seed_dir}apartment/v4_frame_${f}.png" -loglevel error
  done
done
ls scene_*/apartment/v4_frame_*.png
```

在对话里向用户展示这些帧（Read 每张 PNG），询问：**每一帧里，动物是否都在合理位置（不穿家具、不在墙里）？**

- [ ] **Step 7.3: 用户 approve → V4 通过；用户拒 → 回 Task 5/6 调整**

若用户 approve → plan 完成。

若用户指出某帧仍有穿模：
- 记录穿模的家具（人眼估计位置）
- 检查 Task 1 dump 的 JSON 里该位置是否有对应 bbox
- 若无：说明 Task 1 过滤把这件家具错杀了 → 回 Task 1 调过滤阈值 → 重跑 Task 6
- 若有 bbox 但仍穿：说明 Task 2/3 坐标转换或 hit 检测有 bug → 回 Task 2/3 debug

- [ ] **Step 7.4: 更新 memory（可选）**

若 V4 通过，向 memory 追加一条：

```
- [Apartment furniture map 已启用](feedback-apartment-furniture-active.md) — 从 2026-07-06 起 apartment 场景走 furniture-aware collision，两个位置 (compose_scene + run_render_pass) 都会 assert；改 APARTMENT_MIC_ORIGIN_CM 需同步 furniture_map.py 的默认值
```

---

## Global Definition of Done

所有以下条件同时满足才算 plan 完成：

1. Task 1-6 全部 Step 通过（含所有 pytest 绿）
2. `data/apartment_furniture_map.json` 已 commit 到 git，`kept ∈ [30, 500]`
3. `tests/tools/test_furniture_map.py` + `test_furniture_collision.py` 全 PASS
4. `scene_two_dogs.py --skip-audio` 端到端能跑（apartment + shoebox 都出 mp4）
5. Task 7 V4 用户视觉 approve

---

## Rollback

如果任一 Task 出现无法快速修复的问题：

- `git revert <task-commit-sha>` 回滚该 Task；因为每 Task 独立 commit，回滚粒度就是 Task 粒度
- Task 3 的 `scene_spec.py` 修改是最关键的向后兼容点 —— 若被 revert，其他 Task 会自动降级到"no furniture check"行为，pipeline 仍可用（只是恢复到 spec 前的状态）
- 全部回滚等价于：pipeline 回到 spec 前状态；`data/apartment_furniture_map.json` 保留在 git 历史里，未来重启开发时 `git checkout` 就能拿回来

---

## Non-Goals (Recap)

按 spec §9 out-of-scope，本 plan **不实现**：
- RoomSpec 抽象、procgen 家具/房间、家具语义标签、QA 生成器
- OBB 精确 bbox（AABB + 0.1m margin 足够）
- 音频侧家具遮挡建模（GPURIR 不管家具吸声）
- Cosmos sim2real
- 多 apartment 变体（本 plan 只支持 apartment_0000）
