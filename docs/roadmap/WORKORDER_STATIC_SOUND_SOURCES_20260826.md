# 工单 20260826：静态发声资产（家电 + 建筑固定件，14 个网格）

> 执行方：Codex。服务器 `ssh 48g-jump`，主仓 `/data/jzy/code/AVEngine-lead-a`
> （`/data/jzy/code/AVEngine` 是指向它的软链）。生成侧在 `/data/jzy/code/SPEAR-lead-b`。
> 本文里的每个工具参数名都是 2026-08-26 从源码里读出来的，接手后请自己 `--help` 复核一遍。

## 0. 前置条件：不要现在就开始

静态物这条流水线**代码写了但从来没跑过**——我查过，`/data/avengine_external/review/`
下没有任何 static 产物目录。所以我（另一条会话）先用**音响那 5 个网格**把这条路走通，
产出一份参考运行，再把 §3 里"精确调用"那一节填上。

**参考运行落地在 `/data/avengine_external/review/static_speaker_reference_*/`，
并且本文 §3.6 会从"待填"改成实际命令。看到 §3.6 还是"待填"就先不要动手。**

在那之前你可以做的：读 §1–§2 建立背景、读 §5 铁律、读 §6 那些坑（都是真踩过的）。

---

## 1. 背景：这个项目在做什么

AVEngine 造的是**视听问答数据集**：一个房间里有若干个会发声的东西，渲染出画面 +
空间音频，然后出题问模型"谁在发声、在哪一侧、当时有没有在动"。所以每个发声的东西
都要是一个**声源资产**：一份几何 + 一个发声锚点（声音从物体的哪个点发出）+ 一份
可被引擎调用的记录。

### 1.1 五个资产族，以及为什么静态族现在最缺

owner 在 2026-08-24 从 AudioSet 的 609 个非抽象类里逐条筛出 100 个可用类，收敛成五族：

| 族 | 声类数 | 现状 |
|---|---|---|
| 人形（Rocketbox） | 56 | 已有 6 个注册资产 |
| 猫狗四足 | 13 | 已有 8 个注册资产 + 4 个新发布 |
| **静态家电** | **18** | **0** ← 本工单 |
| **建筑固定件** | **8** | **0** ← 本工单 |
| 音响 / 电视 | 5 | 0 ← 另一条会话在做 |

**静态族占 26 个声类，一个资产都没有。** 而且它是最便宜的一族：刚性物体，
不需要绑骨、不需要重定向动画、不需要过形变闸门。动物那条链最贵最不稳的三步
（绑骨是随机的、重定向会撕裂、形变要闸门）在这里全都没有。

### 1.2 一个必须先理解的换算：声类 ≠ 网格

26 个声类只对应 **14 个物体**。同一个物理对象会发出多个 AudioSet 类的声音：
门铃会响"Doorbell"也会响"Ding-dong"也会响"Chime"，但只需要一个门铃网格。
**不要按声类数去做 26 个网格。**

### 1.3 你做的东西下游怎么被用

出题时 `source_endpoint_id`（声源身份）和 `sound_asset_id`（声音内容）是**两个独立字段**。
也就是说一个音响可以播狗叫。这是这批数据的卖点：模型不能靠"听到狗叫就指狗"的语义
先验蒙对，必须真做空间定位。你做的每个静态物都会成为一个可被指代的实体。

---

## 2. 你要做什么

### 2.1 家电：18 声类 → 9 个网格

| 网格 | 覆盖的 AudioSet 声类 |
|---|---|
| 门铃 doorbell | Doorbell, Ding-dong, Chime |
| 座机 landline_phone | Telephone, Telephone bell ringing, Telephone dialing DTMF, Dial tone, Busy signal |
| 手机 cellphone | Ringtone, Cellphone buzz vibrating alert |
| 闹钟 alarm_clock | Alarm clock, Buzzer |
| 烟感 smoke_detector | Smoke detector smoke alarm, Fire alarm |
| 空调 air_conditioner | Air conditioning |
| 搅拌机 blender | Blender |
| 微波炉 microwave_oven | Microwave oven |
| 打印机 printer | Printer |

### 2.2 固定件：8 声类 → 5 个网格

| 网格 | 覆盖的 AudioSet 声类 |
|---|---|
| 马桶 toilet | Toilet flush |
| 洗手池含龙头 sink_with_tap | Water tap faucet, Sink (filling or washing) |
| 浴缸 bathtub | Bathtub (filling or washing) |
| 地漏/明装存水弯 floor_drain | Drip, Gurgling |
| 壁炉 fireplace | Fire, Crackle |

### 2.3 属性轴：形态是主轴，饰面是次轴

**这一条是本工单最容易做错的地方。** owner 的要求是"每个属性最好真的能和之前
能看出来不一样"。对刚性物体来说：

- **形态（form factor）差别大，是主轴。** 空调的壁挂机 / 窗机 / 移动柜机是三个
  完全不同的形状，一眼能分。
- **饰面（finish）差别小，是次轴。** 同一台微波炉换成白色还是不锈钢，差别小得多。

所以：**第一遍只做所有形态、每个形态一个默认饰面。** 第二遍（等 owner 验收第一遍
之后再说）才加第二种饰面。**广度优先于深度。**

建议的形态清单（可以调，调了要在产物里写明理由）：

| 网格 | 形态 |
|---|---|
| air_conditioner | wall_split / window_unit / portable_floor |
| microwave_oven | countertop / over_range |
| printer | desktop_inkjet / office_laser_mfp |
| blender | jug_blender / bullet_blender |
| alarm_clock | digital_cube / twin_bell_analog |
| doorbell | wired_chime_box / video_doorbell |
| landline_phone | desk_corded / wall_mounted |
| cellphone | bar_smartphone（无有意义的形态变化，只做一个） |
| smoke_detector | ceiling_disc / wall_square |
| toilet | floor_close_coupled / wall_hung |
| sink_with_tap | pedestal_basin / counter_vanity |
| bathtub | freestanding / built_in_alcove |
| floor_drain | floor_drain / exposed_p_trap |
| fireplace | masonry_open / wood_stove |

第一遍合计约 **26 个资产**。

### 2.4 固定件必须多声明三样东西

owner 在 2026-08-24 修正过一次：**固定件不能依赖房间语义**（不是所有房间有物体标注），
所以要能人工插入。因此每个固定件资产除了几何和锚点，还要声明：

1. **贴附面** attachment_surface：地面 / 墙面 / 天花板（壁挂马桶是墙面，地漏是地面，
   烟感是天花板）；
2. **朝向** facing：插入时哪个方向朝房间内侧；
3. **占地包围盒** footprint_bbox：放置时的碰撞占地，Studio 拖拽授权要用。

**并且要在资产记录里写明一条代价：插入几何会改变房间声学，该房间的 RIR 缓存必须重算。**
这句话必须落在产物里，不能只写在这份工单里——否则下游会拿着旧 RIR 用。

---

## 2.5 先看：已经写好的 profile 不要重造

`/data/jzy/code/SPEAR-lead-b/data/controlled_source_attributes_v1/candidate_profiles/static_object/`
下**已经有 5 个 `static_object` profile**，写好但从没跑过：

| profile | 对应本工单的网格 |
|---|---|
| `kitchen_appliance_microwave_oven_product_view_v1.json` | 微波炉 ✅ |
| `household_clock_alarm_clock_product_view_v1.json` | 闹钟 ✅ |
| `door_hardware_doorbell_chime_unit_product_view_v1.json` | 门铃 ✅ |
| `communication_device_desk_telephone_product_view_v1.json` | 座机 ✅ |
| `kitchen_appliance_stovetop_kettle_product_view_v1.json` | ❌ 水壶属于被 owner 砍掉的厨房四类，**不要做** |

**14 个网格里有 4 个的 profile 是现成的。** 剩下 10 个照着这个格式写，不要另起炉灶。
一个 profile 里已经有的东西：

- `taxonomy` = `{category, object_type}`（静态物用这两个字段，不是动物的 species/breed）
- `base_template.kind = "text_prompt_only"`（静态物是**文生图**，没有粘土引导图；
  动物才是图生图）
- `fixed_attributes` = 形态 + 材质，`sampled_attribute_domains` = `body_color`
- `generation_contract.route = "flux2_pixal3d_static_v1"`
- `positive_template` / `pose_guard_prompt` / `negative_prompt` 三段
  （pose_guard 那段很关键：三四分之一产品视角、纯净浅灰无缝背景、
  单一物体居中、四周留边、只有物体正下方的柔和阴影、明确写了"这是给图生 3D 的
  中性重建参考，不是广告图"）
- `value_labels`：把枚举值翻成自然语言（`black` → `black stainless`）
- `model_revisions`：flux2 / pixal3d / dino 三个版本号**钉死**
- `target_physical_profiles`：真实世界高度，例如微波炉 `28 cm ± 5`。
  **这就是 §3.4 定型工具说的"目标高度取自经认证的 profile 请求"。**

**注意 profile 里那句 provenance 备注**：`"Provisional typical retail dimension;
measure the approved mesh before formal registration."` ——
这些高度是暂定的零售典型值，正式注册前要量实际网格。这句话要一路带到产物里。

**还要注意 `sampled_domains_must_be_singleton` 在静态物 profile 里是 `false`**
（动物那边是 `true`）。也就是说静态物允许一个 profile 一次产出多个颜色请求，
策略是 `static_object_per_request_one_shot_v1`：**每个请求一次成型、不抽 seed**。
但按 §2.3 的广度优先，第一遍每个形态只取一个颜色。

### 2.5.1 另外 5 个 profile 也已经写好，而且是跑通过的样板

`audio_playback_*` 那 5 个（书架箱 / 落地箱 / 回音壁 / 智能音箱 / 电视）是我
2026-08-26 为参考运行写的，**已经过了契约校验和 512 token 闸门，并且真的跑完了
文生图**。它们不在你的 14 个网格里，所以不要重造，但**照它们抄格式比照微波炉抄更好**
——微波炉那批从没跑过，这 5 个是有实测结果的：

- `audio_playback_bookshelf_speaker_product_view_v1.json`（有 v2 方法修订，见 §6.12）
- `audio_playback_floorstanding_speaker_product_view_v1.json`（一次过，3/3 通过 2D 评审）
- `audio_playback_soundbar_product_view_v1.json`（3/3 通过）
- `audio_playback_smart_speaker_product_view_v1.json`（2/3 通过）
- `audio_playback_television_product_view_v1.json`（有 v2 方法修订，见 §6.11）

两条从这批实测里得到的写 profile 的经验：

1. **`sampled_attribute_domains` 每根轴最多 3 个值**（契约硬限制，
   `1 <= len(values) <= 3`）。所以一个 profile 一次最多 3 个饰面。
2. **`positive_template` 里每个属性占位符只能出现一次**（`_format_placeholders`
   会数重复并报错）。想在描述里再提一次材质，就把词写死，不要再放 `{material}`。
3. **不要在 prompt 里写自己不打算当作验收线的数量约束。** 我写了"至少十倍宽"
   给回音壁，实际出来约六倍；写了"恰好两个单元"给书架箱，两张出来是三个。
   数量写进 `fixed_attributes` 的值（比如 `sealed_two_way_cabinet`）就必须守住，
   否则产物记录会说谎；只是想引导构图的话，别把它写成声明属性。

---

## 3. 流水线

### 3.1 生成（和动物共用，在 AVEngine 仓）

```
tools/assets/generate_canonical_2d.py     # FLUX.2 出一张候选图
tools/assets/segment_canonical_2d.py      # ISNet 抠图
tools/assets/run_pixal3d_mesh.py          # 图生 3D
```

**权重路径不要写死。** 仓里有契约：`examples/assets/model_roots_v1.json` +
`tools/assets/model_roots.py`，解析顺序是「显式命令行参数 → `AVENGINE_MODEL_<NAME>`
环境变量 → `AVENGINE_MODEL_ROOTS` 指向替代注册表 → 仓内注册表」。三个条目现成：
`flux2_klein_base` / `flux2_klein_tokenizer` / `isnet_general_use`。

**prompt 有 512 token 硬闸门。** FLUX.2 的 worker 会把负向提示拼进正向
（`f"{prompt} Avoid: {negative}."`）再按 `max_sequence_length` 截断，超出的尾巴
被静默丢掉——而尾巴正是那些约束。跑生成前先过：

```
python3 tools/assets/check_prompt_token_budget.py --profile-dir <你的 profile 目录>
```

它会枚举所有采样组合，超预算就非零退出。**注意用 `avengine-habitat-runtime` 那个
解释器**，系统 python3 的 jsonschema 太旧。

### 3.2 水密代理网格（SPEAR 仓）

```
tools/blender_create_watertight_textured_proxy_mesh.py
```
产出下一步要的 watertight manifest。

### 3.3 朝向证据

刚性物体也要一个"评审过的源 yaw"。动物那条链用
`tools/blender_estimate_generated_animal_forward.py` 估计再人工确认——
**那个估计器四只错一只，而且置信度反相关**（错的那次 0.82，对的那次 0.17）。
所以不要信它的置信度，渲一张图用眼确认。

### 3.4 定型（SPEAR 仓）

```
tools/blender_finalize_generated_static_object.py \
  --input-glb <水密网格> --watertight-manifest <上一步清单> \
  --static-decision <评审决定记录> --heading-evidence <朝向证据> \
  --output <定型.glb> --manifest <定型清单.json>
```

行为：**刚性 + 一次统一物理缩放**。评审过的源 yaw 映射到 +X 前方，目标高度取自
经认证的 profile 请求，网格最低点落地到 0。**工具里没有任何按物体类别的启发式**——
不要试图给它加"如果是马桶就怎样"的分支。

### 3.5 发声锚点（SPEAR 仓）

```
tools/blender_measure_generated_static_emitter.py \
  --input-glb <定型.glb> --finalization-manifest <定型清单.json> \
  --anchor-spec <锚点规格.json> --output <锚点报告.json> --marker-glb <标记.glb>
```

锚点规格是**按实例、哈希绑定**的：要么给精确重心坐标，要么给归一化包围盒目标
（会被解析到最近表面）。

**坐标帧和动物链路不一样，别搞混：静态物是 `+X 前 / +Y 上 / +Z 解剖学右`，
而动物那条链是 +Z 上。**

锚点选哪里要按物理来，不是几何中心：微波炉的声音从门缝出来，打印机从出纸口，
空调从出风口，马桶从水箱与便池之间。这个判断要写进锚点规格的理由字段里。

### 3.6 精确调用：待填

**这一节由音响参考运行填写。填好之前不要开始 §3.2–§3.5。**

---

## 4. 验收标准

**不要套动物那两道闸门。** `tools/assets/gate_retopology.py` 和
`gate_rigged_asset.py` 判的是"减面有没有饿死头部"和"蒙皮走路时撕不撕"——
刚性物体不绑骨、不走路，这两个判据在这里没有意义。硬套会得到无意义的拒收。

静态物该判的是：

| 判据 | 怎么判 |
|---|---|
| 水密 | §3.2 的清单里有结论 |
| 朝向 | 定型清单里 yaw 映射到 +X，并且有一张渲图证明前方是物体正面 |
| 落地 | 定型清单里最低点 = 0 |
| 物理高度 | 定型清单里的目标高度与真实物体量级一致（微波炉不能有一米高） |
| 锚点在表面上 | §3.5 的报告保证，但要核对它落在物理正确的位置 |
| 看起来像那个东西 | 渲一张四视图，人眼确认。**这一条不能省** |
| 面数预算 | 2.5 万–8 万。刚性物体没有撕裂问题，面数只影响引擎开销 |
| 三个轴的尺寸都要量 | **不只量高度。** 定型是按高度做一次统一缩放，
所以重建出的长宽比误差会被放大：一个 6:1 的回音壁按 7 cm 高定型，长度只有 42 cm，
而真实回音壁是 10:1 以上。量完整包围盒，三个轴都要落在真实物体量级里 |

**另外：两道评审契约自带的判据就是验收标准，不要另发明一套。**
2D 评审的 4 个 check + 8 个硬闸门在
`SPEAR/tools/review_controlled_animal_flux2_candidates.py` 的
`STATIC_CHECK_FIELDS` / `STATIC_HARD_GATE_FIELDS`；3D 评审的 5 个 check 在
`review_controlled_static_object_candidates.py` 的 `CHECK_FIELDS`。
这两套是既有代码强制的，填错字段名直接非零退出。

---

## 5. 铁律（不因这个工单豁免）

- **服务器是唯一代码工作副本。** 不要在本地形成"已完成实现"再搬上去。
- **fresh / no-clobber：** 所有新产物用新的、明确的目录；已存在即失败，绝不覆盖。
  失败即停，保留失败目录和日志当证据，不在原目录重跑。
- **`research_only=true`、`episode_counted=false`、正式数据分母 0**，直到 owner 正式准入。
- **不新增无理由的 hash / gate / contract。** 只有能指出一个具体失败场景、并说明
  git / 版本 / 主键 / 事务 / 唯一约束 / 类型 / 普通测试为什么防不住，才允许加。
  （已经有两次被拦的先例：refusal builder 里多加的 WAV SHA-256 闸门、
  持久 RIR 缓存的 CLI 接口。）
- **不要动正在被长任务读取的工作树。** 跑前 `ps` 看一眼；不能杀、暂停或改动别人的任务。
- 进 main 由 owner 决定，而且**这个仓库上了分支保护**，正常路径是开 PR。

---

## 6. 已知的坑（我这两天真踩过的，别重蹈）

**进程与 shell**

1. **不要 `pkill -f`。** 它会匹配到你自己的 ssh 命令行，把自己杀掉。我干过两次。
   用中括号模式或 `fuser -k <端口>/tcp`。
2. **远端 shell 是 zsh，不做无引号变量分词。** `set -- $pair` 不会拆成两个参数，
   而是整个字符串塞给 `$1`。glob 不匹配时 zsh 直接报错退出而不是原样传递

**静态评审特有的坑（我 2026-08-26 在音响这批上真踩到的）**

11. **`emitter_feature_visible` 这道硬闸门不允许填 `not_applicable`。**
    单测 `test_static_route_rejects_not_applicable_emitter_gate` 明确把
    `not_applicable` 判成非法。这不是漏洞而是设计：看不见出声口，就没法评审锚点
    放得对不对。**后果很直接——出声口不在可见面上的物体，v1 prompt 一定全灭。**
    我的 3 张电视候选全部倒在这条上（现代电视的喇叭朝下、藏在底部风口后面）。
    修法不是放宽闸门，而是改 prompt 方法：让出声口出现在可见面上
    （电视 → 屏幕正下方一条横贯的穿孔栅格，这是中端电视真实的样子）。
    **你的 14 个网格里要提前想这条的**：冰箱（压缩机在背后）、
    洗衣机（滚筒门可见，但脱水噪声其实来自整个箱体）、
    抽油烟机（进风格栅可见，风机在里面）、马桶（水箱与便池之间没有开口）。
    在写 prompt 之前先问一句"这个物体的声音从哪个可见特征出来"，
    答不上来就先改形态或视角，不要生成完再返工。

12. **`target_attribute_only` 是"图像有没有偏离声明属性"那一条。**
    动物那边它是"只有目标属性变了"；静态一次成型没有基准图，所以它的实际用法是
    "渲出来的物体符合 `fixed_attributes` + `sampled_attributes` 声明的那一套，
    没有多出别的"。我用它拒了两张书架箱：图很好看，但是三分频，
    而 profile 声明的固定属性是 `sealed_two_way_cabinet`。
    **拒它的理由不是不好看，是发布后记录会说谎。**

13. **失败的两种正当回应不一样，别混。**
    - **重抽 seed**：同一个 profile 换 `--seed` 再来一批。适用于方法本身能出对的东西、
      只是这一张不巧（我的智能音箱 3 个饰面里 2 个一次过，沙岩色那张出成了藤编凳子）。
    - **方法修订**：改 prompt 方法 + 升 `profile_revision` + 写一份 provenance。
      适用于方法本身守不住声明（书架箱的单元数、电视的可见出声口）。
      **格式照 `data/controlled_source_attributes_v1/candidate_profile_revisions/`
      下现成的 `provenance.json` 抄**，必须写：supersedes 的旧 revision 和它的
      canonical sha256、理由、`this_is_not_a_seed_retry: true`、
      禁止重放的 request/candidate sha256、以及"下一次执行必须用新的 batch seed"。
      我这次新建了 `candidate_profile_revisions/static_object/` 两份，可以直接照抄。
      **仓库里没有工具生成这个记录，是手写约定。**
   （`--include=*.json` 会炸）。
3. **heredoc 经 ssh 会被 zsh 的引号和 glob 规则改写。** 规矩是：本地写文件、`scp`、
   在服务器上执行。我每次偷懒都要多花一轮。
4. **`set -o pipefail` + `grep -q` 会把成功的步骤判成失败**：grep 一匹配就关管道，
   Blender 收到 SIGPIPE 非零退出。改成先写日志再 grep 日志。
5. **后台任务的 ssh 会话断开会带走远端进程。** 长任务要 `nohup` + 日志落盘，
   再轮询日志，不要靠 ssh 会话挂着。

**Blender / 网格**

6. **任何按边的度量必须先焊接。** glTF 导出会在每条 UV / 法线接缝处拆点，
   所以在刚导入的文件上数边界边、碎片数、二面角，量的是文件格式不是表面。
   我的第一版二面角度量在 8 万面里只找到 5 条可用边、五个资产全报 0.01°。
   **这个坑在这个仓库咬了我三次。**
7. **Blender 4.2 移除了 `Mesh.calc_normals_split()`**（4.1 起）。用到会直接报错。
8. **`bpy.ops.uv.export_layout` 需要 GPU**，headless 跑不了。
9. 导入的场景里可能有放置标记物（一个 `Icosphere`，80 面）。按面数取最大的网格，
   不要按名字，也不要假设只有一个 mesh 对象。

**度量方法**

10. **单帧采样会低估最差情况 10–13 倍**（动物那边实测）。刚性物体没有动画所以
    不适用，但同一个道理适用于任何"取一个样本代表整体"的度量。
11. **不要用会随实现细节漂移的代理量当判据。** 动物那边有两个反例：
    「碎面占比」惩罚低面数（最粗的资产分数最差却最好看）；
    「起伏比」随体素分辨率和面数变（同一网格能读出 5.2 和 13.4）。

---

## 7. 产物位置与发布

**产物**（不进 git）：`/data/avengine_external/review/<新目录>/`，每个网格一个子目录。

**发布**到共享资产树：`/data/avengine_external/assets/sound_source_assets_v1/`，
布局是 `<类别>/<型号>/<属性组合>`，例如：

```
appliance/microwave_oven/countertop_stainless/
fixture/toilet/floor_close_coupled_white/
```

顶层是**引擎第一个要问的类别**，不是驱动方式。`index.json` 是程序唯一入口，
它是**可合并的**（gates 按流水线分键、instance_axes 按实体类分键、资产按 id 去重），
所以你写进去不会破坏动物那批。

动物那边的发布器是 `tools/assets/publish_animal_assets.py`，它读生成清单里的
`identity` 和生成溯源。**静态物需要一个平行的发布器**（`publish_static_assets.py`），
写的时候照抄它的结构，特别是这几点：

- 每份资产记录必须带**完整生成溯源**：prompt 原文、token 账目、模型快照哈希、
  seed 与全部采样参数、输入引导图的 sha256、输出图与最终 glb 的 sha256。
  少了这些资产就只能看不能复现。
- 每份资产同时写一份 `asset.json` 放在网格旁边，这样目录被搬出树也能自描述。
- 叶子目录已存在则**报错**，不要静默替换——重复发布同一属性组合是新版本。
- 固定件要额外写 §2.4 那三样声明 + RIR 必须重算那句话。

---

## 8. 不要做什么

- 不要按 26 个声类做 26 个网格（见 §1.2）。
- 不要把饰面当主轴（见 §2.3）。
- 不要给静态物套动物的形变闸门（见 §4）。
- 不要给定型工具加按物体类别的分支（见 §3.4）。
- 不要把权重路径写死成模块级默认值（见 §3.1）。这是 owner 最早提的要求之一。
- 不要在 §3.6 还是"待填"的时候开始。

---

## 9. 相关文档

| 文档 | 为什么要读 |
|---|---|
| `docs/assets/MESH_DENSITY_AND_TEARING_20260825.md` | 动物那条链的全部实测证据。**你不需要它的结论**（那些是蒙皮问题），但§"Review rendering" 一节关于"同一姿态在不同打光下差别极大、必须用柔和无阴影打光看图"对刚性物体同样适用 |
| `docs/roadmap/HANDOFF_20260825_LOGICAL_RENAME.md` | 仓库刚做完按能力重命名，m1–m7 已经不存在了 |
| `docs/roadmap/CORRECTION_DIRECTIVE_20260825.md` | 当前优先级总表，别和它冲突 |
| `AGENTS.md`（仓库根） | 项目铁律 |
