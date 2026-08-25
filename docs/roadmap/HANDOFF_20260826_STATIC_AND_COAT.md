# 交接 20260826：音响参考运行 + 毛色轴对齐 + 尺寸轴降级（给压缩后的自己）

> **2026-08-26 晚更新：这份文档里"待办 A / 待办 C"两节已经过期。**
> - **待办 A 已完成**，结论在 `STATIC_REFERENCE_RUN_20260826.md`，
>   操作步骤在工单 §3.6（已填）。链路跑通了，修掉两个真实集成 bug，
>   发布了 4 个静态资产。**还开着的口子是"没有校平"**，见检查点第 3 节。
> - **待办 C 已完成，但做法和这里写的不一样**：扇出规划器早就把尺寸行标成
>   `runtime_only_derivation`，把 `size` 从 `enabled_axes` 移走会**丢掉**免费的实例
>   多样性。真正错的只有发布器把尺寸写进了叶子目录和 asset_id。四个动物资产已按
>   `<coat_value>` 重发（`cat/siamese/standard_seal_point` 等，下面 §4 里那四个带
>   `medium_`/`small_` 前缀的路径已经不存在）。
> - **待办 B 的性质变了**：毛色轴是个明度乘数（`luminance_gain ∈ [0.65, 1.35]`），
>   做不出品种特征色，而且两侧都有 3 值硬上限。证据和方案在
>   `docs/assets/COAT_AXIS_BREED_COLOUR_FINDING_20260826.md`。**还没动手，等拍板。**

> 这份是**在做的事的状态**，不是项目结论。项目结论在
> `docs/assets/MESH_DENSITY_AND_TEARING_20260825.md` 和记忆文件里。
> 服务器 `ssh 48g-jump`；主仓 `/data/jzy/code/AVEngine-lead-a`
> （`/data/jzy/code/AVEngine` 是指向它的**软链**，`Path.resolve()` 会解析成前者，不是 bug）。
> 生成侧 `/data/jzy/code/SPEAR-lead-b`。

## 0. 当前状态（2026-08-26 记）

| 项 | 状态 |
|---|---|
| `origin/main` | `d637609`（对方刚做完按能力重命名，m1–m7 已不存在） |
| 我的分支 | `cc-static-sound-sources`，`bf245a4`，已推，工作树干净 |
| 上一批工作 | 已合入 main（`c1da4b5`，动物链路 + 两道闸门 + 发布器 + model_roots 契约） |
| 四个动物资产 | 已发布在 `/data/avengine_external/assets/sound_source_assets_v1/` |
| Codex 工单 | `docs/roadmap/WORKORDER_STATIC_SOUND_SOURCES_20260826.md`，已推，**§3.6 故意留白** |

**注意仓库上了分支保护**，正常路径是开 PR，不要直推 main（上次直推成功了但绕过了保护）。

## 1. owner 这一轮拍板的三件事

1. **尺寸轴降级为运行时缩放参数**，不再是要生成的资产版本。理由：孤立渲染里看不出来，
   而且按四层谱系它是纯缩放。（我提的异议，owner 同意。）
2. **毛色清单按品种特征色**，不再是明度三档。清单在 §3。
3. **分工**：我做音响 5 个 + 毛色轴 + 尺寸轴；Codex 做家电 9 + 固定件 5。

## 2. 待办 A：音响参考运行（长杆，解锁 Codex）

**为什么是我做**：静态链路**代码写了但从没跑过**（`/data/avengine_external/review/`
下没有任何 static 产物目录）。路没通就交出去会返工。跑通后要把工单 §3.6 从"待填"
改成真实命令。

### 2.1 静态链路（真实工具名和参数，已从源码读出）

```
# 生成（AVEngine 仓，和动物共用）
tools/assets/generate_canonical_2d.py     # 静态物是文生图，不是图生图
tools/assets/segment_canonical_2d.py
tools/assets/run_pixal3d_mesh.py

# SPEAR 仓
tools/blender_create_watertight_textured_proxy_mesh.py      # 出水密清单
tools/blender_finalize_generated_static_object.py \
  --input-glb --watertight-manifest --static-decision --heading-evidence \
  --output --manifest
tools/blender_measure_generated_static_emitter.py \
  --input-glb --finalization-manifest --anchor-spec --output --marker-glb
```

- 定型工具行为：**刚性 + 一次统一物理缩放**；评审 yaw 映射到 +X 前；
  目标高度取自经认证的 profile 请求；网格最低点落地到 0。工具里没有任何按物体类别的启发式。
- **坐标帧和动物链路不同**：静态物锚点结果在 `+X 前 / +Y 上 / +Z 解剖学右`；
  动物那条链是 **+Z 上**。这个一定会绊人。
- `--static-decision` 和 `--heading-evidence` 是上游产物，具体 schema 还没看，
  参考 `tools/adopt_direct_animal_pixal_attempt.py` 和
  `tools/build_controlled_animal_derived_static_review.py`（这两个也 grep 到了这两个键）。

### 2.2 profile 格式（照现成的抄，别另起炉灶）

现成 5 个在
`/data/jzy/code/SPEAR-lead-b/data/controlled_source_attributes_v1/candidate_profiles/static_object/`：
微波炉 / 闹钟 / 门铃 / 座机 / 水壶（水壶是被砍的厨房四类，不要用）。

静态 profile 与动物 profile 的差别：
- `taxonomy` = `{category, object_type}`，不是 `{species, breed}`
- `base_template.kind = "text_prompt_only"`，`artifact: null`（没有粘土引导图）
- `generation_contract.route = "flux2_pixal3d_static_v1"`
- 策略 `static_object_per_request_one_shot_v1`，
  `sampled_domains_must_be_singleton: false`（动物那边是 true）
- 有 `target_physical_profiles`：真实高度，如微波炉 `28 cm ± 5`，
  **这就是定型工具读的目标高度**；provenance 是 `provisional`，
  备注写了"正式注册前要量实际网格"，这句要带到产物里
- `model_revisions` 钉死 flux2 / pixal3d / dino 三个版本号
- `pose_guard_prompt` 那段很关键：三四分之一产品视角、纯净浅灰无缝背景、单物体居中、
  四周留边、只有正下方柔和阴影、明确声明"这是给图生 3D 的中性重建参考，不是广告图"

### 2.3 我要写的 5 份新 profile

| object_type | category | 形态 |
|---|---|---|
| `bookshelf_speaker` | `audio_playback` | 书架箱 |
| `floorstanding_speaker` | `audio_playback` | 落地箱 |
| `soundbar` | `audio_playback` | 回音壁 |
| `smart_speaker` | `audio_playback` | 智能音箱（圆柱/圆饼） |
| `television` | `audio_playback` | 电视 |

**发声锚点要按物理选，不是几何中心**：音箱取低音单元或号角口，回音壁取正面出声栅格，
电视取底部下向扬声器，智能音箱取顶/侧环形出声孔。理由要写进 anchor-spec 的理由字段。

**为什么音响最值钱**（owner 原话"这次最值钱的设计点"）：一个网格一个锚点可播 573 类内容；
堵住 v1QA 的语义先验漏洞（音响放狗叫、画面里另有一只不叫的真狗，"听到狗叫就指狗"失效）；
是 JAEGER 对比里唯一我们能出他们出不了的题；而且是**仰角多样性最便宜的来源**
（猫狗贴地、人嘴 1.5 m，仰角近乎常数；音箱放书架/桌面/地板就有了）。

### 2.4 静态物的验收（**不要套动物的两道闸门**）

`gate_retopology.py` 判"减面有没有饿死头部"、`gate_rigged_asset.py` 判"蒙皮走路撕不撕"——
刚性物体不绑骨不走路，这两个在这里无意义。静态该判：水密 / 朝向映射到 +X 且有渲图证明 /
最低点=0 / 物理高度量级合理 / 锚点落在物理正确位置 / **人眼看四视图确认像那个东西** /
面数 2.5 万–8 万（刚性没有撕裂问题，面数只影响引擎开销）。

## 3. 待办 B：毛色轴对齐

**现状是明度滑块**，`src/avengine/appearance/contracts.py` 的 `COAT_PROFILE_DOMAINS`：

```
labrador_retriever  → light_yellow / standard_yellow / dark_yellow
british_shorthair   → light_blue / standard_blue / dark_blue
beagle              → light_tricolor / standard_tricolor / dark_tricolor
abyssinian          → light_ruddy / standard_ruddy / dark_ruddy
border_collie       → light_black_white / standard_black_white / dark_black_white
shiba_inu           → light_red / standard_red / dark_red
pembroke_welsh_corgi → light_red_white / standard_red_white / dark_red_white
```

**这是对齐不是新发明**：生成侧的暹罗 profile 里本来就有 `point_color: [seal_point]`
这种真实品种轴，是 AVEngine 侧把它压平了。

owner 批准的目标清单：

| 品种 | 改成 |
|---|---|
| 英国短毛猫 | 蓝 / 乳白 / 银虎斑 / 黑 |
| 暹罗 | 海豹点 / 巧克力点 / 蓝点 / 丁香点 |
| 缅甸猫 | 貂色 / 香槟 / 蓝 / 铂金 |
| 阿比西尼亚 | 红褐 / 红(索雷尔) / 蓝 / 浅黄 |
| 拉布拉多 | 黄 / 巧克力 / 黑 |
| 边境牧羊犬 | 黑白 / 红白 / 蓝色云石 |
| 杰克罗素 | 白棕 / 三色 / 白黑 |
| 柯基 | 红白 / 貂白 / 三色 |
| 柴犬 | 赤 / 黑芝麻 / 奶油 |

**关键：毛色变体不需要重绑骨。** `SPEAR/tools/blender_project_animal_multiview_coat.py`
渲四视图 → 编辑 → 按空间对数色比投影回原 UV，"几何、蒙皮权重、骨架、Walk/Idle 动作
一律不重新生成"。配套 `blender_render_generated_animal_coat_views.py` 渲源视图、
`build_animal_coat_reference_board.py` 出参考板。**所以形变闸门也不用重跑**（几何相同）。

**风险**：改 `COAT_PROFILE_DOMAINS` 会影响 `examples/runtime/source_asset_runtime_profiles.json`
里 7 个已注册资产的毛色值（它们现在叫 `standard_ruddy` 之类）。要么保留旧值当别名，
要么同步改注册表。**先看 `contracts.py` 里 `COAT_PROFILE_REALIZATION_RULES` 怎么用这些值**
再动手，别直接改域。

## 4. 待办 C：尺寸轴降级

改三处，一起改一次提交：

1. `examples/assets/instance_fanout_axes_v1.json`：`size` 从 `enabled_axes` 移到
   运行时参数（现在是 `enabled_axes: ["size","coat_profile"]`，
   `pinned_axes: {body_build, life_stage: inherit_from_source_asset}`）
2. `tools/assets/publish_animal_assets.py`：叶子目录从 `<size>_<coat_value>` 改成
   `<coat_value>`；asset_id 去掉 size 段
3. 把已发布的四个资产按新命名重发（发布器**拒绝覆盖已存在的叶子**，
   所以要先删旧树或换 `--revision`）

现有四个资产的当前路径（改名前）：
```
cat/siamese/medium_standard_seal_point/                0.591%
cat/burmese/medium_standard_sable/                     0.717%
cat/burmese/medium_dark_sable/                         1.867%
dog/jack_russell_terrier/small_standard_white_tan/     0.733%
```

## 5. 给 Codex 的 prompt（已经给过 owner，存这里免得丢）

```
你在 AVEngine 项目上做静态发声资产。工单已经写好了，读它，按它做：
  服务器: ssh 48g-jump
  主仓:   /data/jzy/code/AVEngine-lead-a   (/data/jzy/code/AVEngine 是软链)
  生成侧: /data/jzy/code/SPEAR-lead-b
  分支:   cc-static-sound-sources
  工单:   docs/roadmap/WORKORDER_STATIC_SOUND_SOURCES_20260826.md

任务: 26 个 AudioSet 静态声类 → 14 个网格(家电 9 + 建筑固定件 5)。
第一遍只做所有形态、每个形态一个默认饰面，约 26 个资产。

先做三件事再动手：
1. 完整读一遍工单，包括 §6 那 11 个坑。
2. 检查 §3.6「精确调用」。如果还写着"待填"，就先不要开始 —— 这条流水线从来没跑过，
   要等音响那 5 个的参考运行把它填上。
3. 看 §2.5：14 个网格里有 4 个的 profile 已经写好了（微波炉、闹钟、门铃、座机），
   不要重造。第五个是水壶，属于被砍掉的厨房四类，不要做。

三条最容易做错的：
- 声类 ≠ 网格。门铃响三个声类但只是一个网格。不要做 26 个网格。
- 属性轴形态为主、饰面为次。
- 不要给刚性物体套动物的形变闸门。静态该判什么在 §4。

铁律在 §5，产物位置和发布规范在 §7。拿不准先问，不要猜。
```

## 6. 环境与坑（省得重新踩）

- **解释器**：`/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python` 跑测试和
  依赖 jsonschema 的工具（系统 python3 的 jsonschema 太旧，
  `plan_instance_variants.py` 会挂）；`h3d` 那个 env 有 ruff；
  Blender 在 `/data/jzy/.local/bin/blender`（4.2.1）
- **单测**：`python -m pytest tests/unit -q`，基线 **3169 passed / 65 skipped / 71 subtests**，
  约 4 分 22 秒。全仓 ruff 有 89 个既有错误（origin/main 上一样），
  只需保证自己动的文件干净
- **长任务**：ssh 会话被移到后台会带走远端进程。必须 `nohup` + 日志落盘再轮询日志
- **不要 `pkill -f`**：会杀掉自己的 ssh。用中括号模式或 `fuser -k <端口>/tcp`
- **远端是 zsh**：不做无引号变量分词（`set -- $pair` 不拆），glob 不匹配直接报错退出
- **heredoc 经 ssh 会被改写**：规矩是本地写文件、`scp`、再执行
- **`set -o pipefail` + `grep -q` 会把成功判成失败**（SIGPIPE）。先写日志再 grep 日志
- **任何按边的度量必须先焊接**：glTF 在 UV/法线接缝处拆点。这个坑咬过三次
- **Blender 4.2 移除了 `Mesh.calc_normals_split()`**；`uv.export_layout` 需要 GPU
- **导入场景里有个 `Icosphere` 标记物**（80 面），按面数取最大网格，别按名字
- **权重路径走契约**：`examples/assets/model_roots_v1.json` +
  `tools/assets/model_roots.py`，条目 `flux2_klein_base` / `flux2_klein_tokenizer` /
  `isnet_general_use`，解析顺序 = 显式参数 → `AVENGINE_MODEL_<NAME>` →
  `AVENGINE_MODEL_ROOTS` → 仓内注册表
- **prompt 512 token 硬闸门**：`tools/assets/check_prompt_token_budget.py`，
  跑生成前先过，超了非零退出
