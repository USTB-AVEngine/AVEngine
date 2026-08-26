# 交接 20260826b：JAEGER 对齐 + FOA 链路（给压缩后的自己）

> 前一份交接是 `HANDOFF_20260826_STATIC_AND_COAT.md`（静态资产），本份接在它后面。
> 静态资产的结论在 `STATIC_REFERENCE_RUN_20260826.md`。这份记的是**声学侧**。

## 0. 状态快照

| 项 | 值 |
|---|---|
| 分支 | `cc-static-sound-sources`，最新提交 `d33c938`，已推 |
| 已发布资产 | 16 个（4 动物 + 12 静态），`/data/avengine_external/assets/sound_source_assets_v1/` |
| 静态资产朝向 | 12 个里 11 个已校平，沙岩智能音箱度量被拒故保持歪着（记录里写明） |
| HM3D 下载 | **进行中**，`/data/datasets/habitat_data/versioned_data/hm3d-1.0/hm3d/val/`，已 7.2 G，三个 uid 里第三个在跑 |
| Codex | 在同一分支并行做家电/固定件；镜像目录里会出现他们的 profile 修订，别当成自己的 |

**owner 已拍板**：①静态物必须水平（已实现并全量重跑）；②毛色要在 FLUX 生图时用 prompt 指定，不是后处理；③塔形不再追三饰面，转做多音箱配置；④HM3D 要下、数据集放公共目录。

## 1. 声学链路：四个真实发现（都已实测）

### 1.1 12 ms 分析窗在真实房间里静默偏差约 18°（最重要）

团队现成脚本（`render_test.py` / `render_real.py`）用**检测到 onset 之后 12 ms** 的窗算强度矢量。
空箱子 `testroom` 里 12 ms 内没有别的东西到达，所以测出 0.000°，这个窗长从没被质疑。
真实房间的早期反射 **3–6 ms 就到**。实测扫窗（skokloster-castle，4 个声源）：

| 窗口 | 0.5 ms | 1 ms | 2 ms | 3 ms | 6 ms | 12 ms |
|---|---|---|---|---|---|---|
| 误差 | 0.0° | 0.0–0.5° | 0.0–1.2° | 0–2.7° | 15.5–17.9° | 17.3–19.9° |

**用 1 ms，不是 2 ms。** 这条修正了本文档原来的结论。

skokloster-castle 是一座空旷的石头大厅，0.5–2 ms 都测出 0.0°，所以 2 ms 当时看着安全。
**在有家具的住宅里不安全。** 2026-08-26 在 HM3D `00804-BHXhpBwSMLh` 上，对一条移动声源
路线的 25 个逐帧响应扫窗：

| 窗口 | 0.25 ms | 0.5 ms | 1 ms | 2 ms | 4 ms | 12 ms |
|---|---|---|---|---|---|---|
| 中位 | 0.00° | 0.00° | 0.00° | 0.12° | 4.06° | 16.46° |
| 最大 | 0.00° | 0.00° | 0.81° | **6.16°** | 24.81° | 26.94° |
| 5° 内 | 25/25 | 25/25 | **25/25** | 21/25 | 13/25 | 0/25 |

而且 2 ms 的污染**随距离增长**：4.6 m 处读 4.8°，1.1 m 处读 0.0°。直达波随距离变弱，
反射不变弱。所以"远处的声源测不准"看起来像遮挡，实际是窗太长。

`tools/audio/render_moving_source_foa.py --window-sweep` 可以在任意路线上复现这张表 ——
**不要信这个数字，在你自己的场景上扫一遍**，因为它显然依赖房间。
**这条要同步给 binding_data 的 owner** —— Phase 1 那 2790 个 IR 是用 `audio_utils.py`
（STFT 域、另一条路径）验的，不一定受影响，但 `render_real.py` 这条肯定受。

### 1.2 `render_real.py` 的真值公式是错的

```python
gaz = atan2(rel[1], rel[0]);  gel = asin(rel[2]/rr)   # 这是 z-up 的算法
```
Habitat 是 **y-up**。它自己的输出把一个水平方向标成"俯仰 +90°"，最大误差 173.86°，
而末尾那句 `=> renderer geometrically correct` 是**无条件打印**的。别信那句。

### 1.3 通道常量 `CH_X,CH_Y,CH_Z = 3,1,2` 是对的（我一度怀疑错了）

我猜"ambisonic 是 z-up 所以要换轴"——**错的**。用 8 个**非轴对齐**方向重新拟合，
最差 0.000°，符号全正、无需换轴。之前的验证之所以可疑，是因为它只用轴对齐方向，
那种情况下多个排列并列、拟合返回碰巧先看到的那个（`render_test.py` 的输出里
同一次运行对不同声源给出了 `chs(1,3,2)` 和 `chs(3,1,2)` 两种答案，就是简并的证据）。

**修正（2026-08-26）：两者其实是同一个排列，不存在通道顺序差异。**

本文档原来写"SoundSpaces = `[W,Y,Z,X]`，JAEGER 发布数据 = `[W,Z,X,Y]`，两者不同"，
依据是 `binding_data/audio_utils.py` 第 35 行。那是**记法冲突，不是数据差异**：

- `audio_utils.py` 按 **front/left/up** 给通道命名（`CH_Z,CH_X,CH_Y = 1,2,3`），
  然后 `front=-Ix, left=-Iy, up=+Iz`。那个"符号翻转"是 x/y/z → front/left/up 的换算，
  不是换轴。
- 换算回 habitat 的 x/y/z：`x=ch3, y=ch1, z=ch2`，**三个符号全正**。

拿 JAEGER 自己发布的数据实测过了(`simulation_ds/test` 的 400 个 task1，
用 `position_local` 当真值，1 ms 窗)：`order (3,1,2) signs (+,+,+)`
**中位 0.00°、p90 0.00°、400/400 在 5° 内**；次优的排列中位 27.8°，毫无歧义。

**所以读 JAEGER 的 FOA 不需要任何转换**，和我们渲染出来的用同一组常量。
反过来说，照原来那句话去加一层转换，会把方向搞错。

JAEGER 的角度标签约定（同一批数据实测，中位和最大误差都是 0.000°）：
`azimuth = atan2(-x, -z)`、`elevation = asin(y)`，x/y/z 是听者局部 habitat 坐标。
即**前方 -Z、上方 +Y 都和 habitat 一致，但方位角朝 habitat 的左手方向为正** ——
听者右侧的声源在它的标签里是**负**方位角（`source_label=Right` 的 132 个样本方位角
中位 -20.11°，`Left` 的 104 个中位 +19.83°）。按 +X=右 去读会每个角都反号。

### 1.4 `cast_ray` 在这个构建里不测静态场景

只测刚体，开不开 physics 都返回 0 命中 —— 拿它做视线遮挡检查是**空转**。
改用 navmesh：`geodesic / straight <= 1.08`。

## 2. 关于"某个场景不能用"——这个说法是错的

我一度说 `apartment_1` 不适合声学仿真，**那是拿 12 ms 那把坏尺子量的**。
用 2 ms 重测：4 个方向里 3 个降到 0.1–5.6°，只有一个在**所有窗长下**都错 103°（那个 5.6° 现在看应该用 1 ms 再测一次）
——那是真正被遮挡的摆位（声源在家具里/墙后）。

**正确结论：不存在"这个场景不能用"，只存在"这个摆位不能用"。**
所以逐摆位的 DoA 校验是那道闸门，HM3D 同样适用。

## 3. 环境事实

- **ss2 的 habitat-sim 相机渲染在这台机器上崩**（`GL::Renderer::Error::InvalidValue` + core dump），
  试过 `gpu_device_id` / `MAGNUM_DEVICE` / `EGL_PLATFORM` / 强制 NVIDIA vendor 全无效。**音频完好。**
- **Blender 的 EEVEE 渲染一直正常** → GPU/GL 栈没问题，坏的是 habitat 的 EGL 路径。
- **AVEngine 自己有 habitat 视觉端**（`src/avengine/rooms/habitat_capture.py`），
  但 `/data/jzy/code/habitat-sim-AVEngine/` **只有源码没有编译产物**，
  `AVENGINE_HABITAT_RUNTIME_ROOT` / `_PREFIX` 也没设，所以那条路现在跑不起来。
  **想要 habitat 出视觉，就得把那个运行时装上**；否则视觉走 Blender。
- 只有 `ss2` 环境有可用的 `habitat_sim`（0.2.2，SoundSpaces 2.0 钉死这个版本）。

## 4. 脚本位置（都在 `/tmp/`，要提交进仓库）

| 脚本 | 作用 |
|---|---|
| `/tmp/plan_from_navmesh.py` | navmesh 采样出接收点+音箱摆位，带测地/直线比值的视线检查 |
| `/tmp/insert_speakers_and_render_foa.py` | 按摆位渲 FOA，逐声源验 DoA，输出 `scene_report.json` |
| `/tmp/fit_foa_axes.py` | 用通用方向拟合 FOA→世界轴映射（用来推翻/确认常量） |
| `/tmp/hm3d_download.sh` | 从 `~/.hm3d_token` 读凭证下载，带三次重试 |
| 已进仓库 | `tools/blender/render_instance_diversity_check.py`（房间尺度看饰面分不分得开） |

**token 在 `~/.hm3d_token`（600 权限，两行：id / secret）。我不把它写进命令行。**
owner 说过可以全权代做，但这条我保留 —— 凭证不从对话里复制进命令。

## 5. 已验证跑通的东西

`skokloster-castle` 里放 4 个外观不同的音箱，每个当声源渲 FOA：

| 资产 | 距离 | DoA 误差 |
|---|---|---|
| 书架箱 胡桃 | 4.88 m | 0.04° |
| 书架箱 白 | 1.99 m | 0.00° |
| 智能音箱 炭灰 | 4.01 m | 0.95° |
| 回音壁 黑 | 2.01 m | 1.46° |

声源放在资产记录的**发声锚点偏移**上（低音单元/栅格中心），不是物体原点。

## 6. 下一步（按顺序）

1. **下载完成后**：在 `sound-spaces/data/scene_datasets/` 建软链指向
   `/data/datasets/habitat_data/versioned_data/hm3d-1.0/hm3d`，核对那 36 个场景
   （`00800-TEEsavR23oF` … `00894-HY1NcmCgn3n`，全在 val 范围）都在。
2. **把 §4 的脚本提交进仓库**（现在还在 `/tmp`，压缩后会找不到）。
3. **在 HM3D 场景上跑通同一条链**，逐摆位 DoA 校验。
4. **视觉**：要么装 AVEngine 的 habitat 运行时，要么用 Blender + 同一份 placement 文件。
5. **把 §1 的四条发现同步给 binding_data 的 owner**（12 ms 窗那条尤其）。

## 5.5 JAEGER 的声学到底怎么做的（2026-08-26 读码+读数据）

**它不发布渲染器。** 整个公开仓库里没有一个文件引用 habitat_sim / soundspaces /
AudioSensor（我们的 `binding_data/` 不算）。`data/data_tools/` 下只有一个
`conv_ir_speaker_foa.py`，做的事就是把**已经渲好的 RIR** 和 LibriSpeech 干声做
`fftconvolve`。README 也只说"下载 RIR，然后卷积"，完全没描述仿真方法。
所以"JAEGER 怎么渲的"这个问题，**从公开材料无法回答**；能拿到的是它的产物。

产物形态（`SpatialSceneQA/hm3d_foa_av_v2`，test split 实测）：

| | |
|---|---|
| 每个 task | `rgb.png` + `depth.png` + `metadata.json` + IR，**单帧，不是视频** |
| task1 `single_source` | 1000 个，一个 `ir.npy`，形状 `(4, 2129)` float64 |
| task2 `dual_source` | 895 个，`ir_male.npy` + `ir_female.npy`，卷积后**直接相加**，不做电平配比 |
| 场景内人数 | 1–3 个（`num_speakers`），其中一个/两个是发声源 |
| 标注 | `visual_objects` 用 **HM3D 语义实例**（`instance_id` + `category`，如 wall/picture），每个都带 DoA |

**全部是静态的。没有移动声源的任务类型。** 一个 task = 一个静止听者 + 一个静止声源
+ 一帧图。

**覆盖范围很窄**（1000 个 single_source 全量统计）：

- 方位角 **严格在 ±40° 内**（min -39.97、max +39.92，100%）
- 俯仰角 ±25° 内
- 距离 **0.86–4.12 m，中位 1.85 m**

也就是说**声源永远在相机视野里、永远很近**。它从不问"身后的声音"、"隔着墙的声音"、
"房间对面的声音"。

顺带一个有意思的对照：我们移动声源那批 episode 里，验证干净的距离是 1.1–4.6 m，
全军覆没的是 4.2–7.2 m —— **验证得干净的那一段基本就是 JAEGER 的距离带**。
在有家具的住宅里超过约 4 m，直达波定位本来就难。

## 6. 移动声源（2026-08-26 追加）

`tools/routes/compile_hm3d_dynamic_source_bank.py` 找路线，
`tools/audio/render_moving_source_foa.py` 渲声音。逐帧一个 FOA 冲激响应，干声跟逐帧响应
做时变卷积 + 重叠相加，所以混响不会每帧重启。

**听者必须靠渲染来选，不能靠距离。** 在距离带里随机取 navmesh 点，约一半时候取到被墙
隔开的位置：同一个场景六条 episode，三条每帧都在 0.27° 内，三条**每帧都失败**，中位
44°、44°、70°，有一帧读到 171°（声强矢量指向声源反方向，因为唯一到达的能量是墙的反射）。
测地距离预测不了这个，navmesh 不知道耳朵高度上站着什么。

所以工具先在路线中点渲一发响应试听者（一两秒），不合格就换。加上这道预筛之后，同样六条
episode **全部 15/15 通过**，最大误差 0.07–0.57°，只花了 1–3 个候选。

## 7. 还开着的决定

- **毛色轴**：owner 已定方向（prompt 指定而非后处理），仓里现成的路是
  `SPEAR/docs/generated_animal_real_reference_coat_workflow.md`
  （渲四视图 → FLUX.2 带 seed 编辑 → 投影回原 UV，不重绑骨）。**还没动手。**
- **`instance_axes.rigid_static_object` 里混进了固定属性**（form_factor / material），
  是 Codex 的代码，读起来像它们在实例间会变。没动，等协调。
