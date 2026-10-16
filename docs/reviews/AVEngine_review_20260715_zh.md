# AVEngine 四足动物 Mesh–Rig–Animation 与音视频同步技术审查

日期：2026-07-15  
审查对象：`AVEngine_code_and_docs_20260716.zip` 解包后的 `/mnt/data/AVEngine`

## 1. 执行结论

当前最值得改的不是 nearest-surface 权重转移参数，而是资产权威关系。

**正式数据链路应采用：审计过的模板拓扑/骨架/权重/动作作为动画权威，Pixal3D、TRELLIS.2 或其他生成器的结果只作为形状与 PBR 外观 guide。**

未知生成拓扑直接套 Quaternius 骨架的路线只能保留为研究支线。它可以借助目标原生骨架拟合、鲁棒体积/无网格权重和 IK 改善，但不应成为批量数据集的默认入口。

音视频方面应引入唯一的 `timeline.json`，由它同时驱动 UE 姿态、相机、多视角渲染、声源骨骼轨迹、音频事件和 mux 验证。当前自由播放动画加“视角外层循环”的实现会让四个视角处于不同动作相位。

## 2. 已检查的项目证据

### 2.1 测试

执行了项目中与四足绑定、模板拟合、动作重定向、形变、足底和稳定模板有关的测试：

```text
59 passed in 0.47s
```

但其中相当一部分是读取脚本文本并检查静态契约的测试，并不等价于 Blender 中真实导入、完整 41 帧形变、BVH 自交或 UE 多视角一致性测试。因此这些测试说明接口约束较完整，不能证明真实资产已经泛化通过。

### 2.2 当前直接换皮链路

`external/SPEAR/tools/run_generated_quadruped_deformation_stabilization.py` 当前默认执行：

1. 生成 mesh 降面；
2. 全局 uniform bbox 对齐；
3. Blender BVH 欧氏最近三角形；
4. 重心插值复制 Quaternius 源 mesh 权重；
5. 缺失区域通过目标 mesh 图传播补权重；
6. 只改权重，不改源骨架 rest matrix 和动作曲线；
7. 运动感知 edge-average 修复；
8. 最终硬门主要检查最大正向边延伸是否变好且低于 4% bbox 对角线。

关键代码位置：

- `run_generated_quadruped_deformation_stabilization.py:239-308`
- `run_generated_quadruped_deformation_stabilization.py:419-437`
- `blender_robust_swap_mesh_keep_rig.py:325-351`
- `blender_robust_swap_mesh_keep_rig.py:995-1092`
- `blender_robust_swap_mesh_keep_rig.py:1143-1157`

### 2.3 项目自己的失败证据

项目文档已经记录：

- Pixal 静态阶段 14 个动物中只有 7 个允许继续；失败包括碎片毛发、地面/背景融合、蹲坐姿势、缺失身体、晶体状四肢。
- 动画阶段出现前腿折叠/拉长、脱离 mesh 岛、后脚拉丝、前腿交叉、后腿错误变形。
- TRELLIS 比格、Pixal 虎斑猫和巴哥经过权重修复后最大延伸从 7.71%–13.75% 降到 2.36%–2.83%，但状态仍明确是 `research_candidate_pending_human_visual_review`。
- 稳定模板文档已经得出同样的主结论：正式默认应是 audited closed template + native actions + controlled appearance variation；Pixal 原生拓扑留在研究路线。

相关文档：

- `docs/generated_quadruped_deformation_stabilization.md`
- `docs/pixal_animal_static_qa.md`
- `docs/pixal_animal_animation_qa.md`
- `docs/stable_animal_template_route_20260714.md`

## 3. 根因分析

### 3.1 全局 bbox 对齐不能解决解剖对应

当前对齐只统一整体尺度和中心。巴哥、比格、灵缇、腊肠、猫、马的肩胛、髋、肘、腕、膝、跗关节位置和肢段比例完全不同。即使 bbox 相同，骨轴和旋转中心仍可能落在皮肤外、关节错误位置或另一条腿附近。

### 3.2 源骨架不动，目标比例却发生变化

当前流程把新 mesh 绑定到旧 armature，并明确保持源骨架 rest matrices。权重修复只能改变“哪些骨影响哪些点”，不能把错误的膝、肘、髋关节移动到目标解剖位置。因此它可以降低拉丝指标，却不能从结构上解决折腿、交叉腿和不正确弯折中心。

### 3.3 欧氏最近表面会跨越四肢和腹部

单图生成动物的左右腿通常距离很近，且可能存在桥面、内壳、非流形边和碎片。欧氏最近三角形没有“左前腿只能对应左前腿”这一硬约束。远侧腿、腹部和近侧腿之间的几何距离很小，权重容易跳到错误链。之后的目标图传播又会把错误影响扩散到邻域。

### 3.4 当前七区语义过粗

项目把 mesh 分成 tail、torso、head 和四条整腿。它没有区分肩/上臂/前臂/腕/掌、髋/大腿/小腿/跗/足，也没有胸、腹、颈、口鼻和耳。整条腿内部仍可能把上臂权重复制到脚，或把膝弯折中心放错。

### 3.5 原始生成拓扑并非动画友好拓扑

项目已有证据显示生成结果包含：开放边、非流形、碎片、内壳、腿间桥、地面融合、遮挡侧肢体不完整。减面和蒙皮会放大这些问题。DQS、bone heat 或简单刚性父骨无法修复错误拓扑和错误解剖。

### 3.6 动作形态与物种/体型不匹配

同一个通用 Dog Walk 不适合巴哥的短腿与宽胸、腊肠的长躯干，也不适合猫、马和牛。即使没有拉丝，也会出现步幅过大、脚滑、腿交叉、躯干摆动不自然和关节超限。

### 3.7 当前最终门禁不充分

独立的 skinned deformation audit 已经能测对称边长比例和三角形面积比例，但当前 stabilization orchestrator 的最终硬门主要依赖最大正向边延伸。它不能拒绝：

- 三角形翻面；
- 局部塌缩但没有正向延伸；
- 自交和左右腿穿插；
- 足底滑动、悬空和穿地；
- 关节弯曲方向错误；
- 局部体积丢失；
- 四个视角动作相位不一致。

## 4. 正式生产架构

```text
属性/品种请求
  -> 参考图与生成 3D guide
  -> guide 静态 QA + 权利 QA
  -> body-plan / morphotype 分类
  -> 选择 audited canonical template
  -> 解剖 landmark + 语义部件对应
  -> 受约束的模板非刚性拟合
  -> 保留模板拓扑、UV、骨架语义、权重和动作族
  -> guide PBR 烘焙到模板固定 UV
  -> 同体型动作重定向 + 接触 IK + 步幅校正
  -> 多层形变/接触/碰撞/视图 QA
  -> timeline 驱动 UE 和音频
  -> 数据资产注册
```

### 4.1 模板银行，而不是一个万能狗模板

建议按 body plan 和 morphotype 建立模板银行。狗的第一版至少包括：

1. 通用中头型/中等比例；
2. 短头宽胸、短腿型（pug/bulldog）；
3. 长躯干短腿型（dachshund/corgi）；
4. 长腿纤细型（greyhound）；
5. 大体型厚重型（mastiff/retriever）。

之后分别建立 cat、fox/wolf、horse/donkey、bovine/deer、pig、sheep/goat 等模板族。所有模板共享相同的数据契约和 QA，但不强求同一骨架比例或动作。

每个模板应冻结并登记：

- topology revision 和顶点/面哈希；
- 固定 UV；
- semantic skeleton graph；
- rest pose 和允许调整范围；
- skin weights；
- action library；
- collision proxies；
- 原生动作的 QA 分布基线；
- 来源、许可证与审核记录。

### 4.2 模板选择

从 guide 和原参考图提取归一化形态向量，例如：

- body length / withers height；
- chest width / height；
- leg length / body length；
- muzzle length / skull length；
- head width；
- pelvis/shoulder width；
- tail length；
- ear类别和置信度。

先做 body-plan 硬分类，再在同族中做最近原型/概率选择。若离所有模板都太远，应拒绝自动生成或进入“创建新模板 revision”流程，不能强行使用最近模板。

### 4.3 解剖 landmarks

狗至少需要：鼻尖、眼/耳锚点、颈根、肩峰/肩胛、胸骨、骨盆、尾根/尾尖、四肢的肩/髋、肘/膝、腕/跗、足中心。建议组合：

- 原始 2D 图关键点和分割；
- 已知相机参数的 ray cast/back projection；
- guide 的对称面和几何截面；
- TokenRig/SkinTokens 或 medial-axis 方法输出的 skeleton candidate；
- 近侧 limb 高置信度、远侧 limb 用对称先验并降低观测权重。

学习模型只负责 proposal，最终语义图、左右/前后、骨序和位置范围由确定性约束检查。

### 4.4 受约束模板拟合

升级 `blender_fit_i23d_to_animal_template.py`。当前实现仍是七区最近表面位移 + 同区平滑；建议改成 coarse-to-fine deformation graph/ARAP 拟合：

```text
E = λsurface Erobust_surface
  + λlandmark Elandmark
  + λarap EARAP
  + λsym Esymmetry
  + λjoint Ejoint_envelope
  + λfoot Efoot_plane
  + λcollision Eself_collision
  + λvolume Eregional_volume
  + λsmooth Esmooth
```

关键要求：

- Surface term 使用 robust loss，拒绝异常碎片和错误侧表面；
- 所有对应必须受语义部件约束，不允许跨腿；
- joint term 限制关节中心始终位于对应 limb volume 中；
- 足底锚在统一地面；
- 对称先验处理远侧肢体缺失；
- ARAP/局部体积避免胸腹和关节附近被压扁；
- 先拟合躯干/头/尾，再拟合四肢；
- 超出模板允许形态域时换模板或拒绝，而不是继续增大位移上限。

`axial-only` 现有模式是正确的稳定方向：单图 guide 不可靠时，保留四条模板原生 limb 几何和权重，只拟合轴向轮廓。后续可在有可靠 landmarks 时有限度调整 limb 长度和粗细。

### 4.5 外观/PBR

保持模板固定 UV，通过语义约束的多视图或 selected-to-active bake 把 guide 的 base color、normal、roughness、metallic/occlusion 烘到模板。不要复制生成拓扑和它的 UV 岛。

当前 `blender_fit_i23d_to_animal_template.py` 已有 vertex-color、region-atlas、bake 和 projected-uv 路线；正式路线应优先固定模板 UV 的 bake/region-atlas，并增加：

- 语义遮罩；
- 正反面/左右侧一致性；
- 无观测区域的对称补全；
- 接缝和空洞修补；
- PBR 数值范围与纹理哈希回读。

## 5. 必须保留生成原始拓扑时的研究支线

若某些任务必须保留生成 mesh 的精细几何，应改成“目标原生 rig”，而不是继续复制 Quaternius 权重：

1. 对原 mesh 做静态 QA；严重桥连、缺肢、地面融合直接拒绝。
2. 建立 watertight/SDF 或低分辨率 proxy，仅用于骨架与权重计算，最终 PBR mesh 不必被体素化替换。
3. 在目标解剖中拟合 semantic skeleton；可用 TokenRig/SkinTokens 产生多个候选，再用确定性评分选择。
4. 在目标上重新计算权重：优先 Geodesic Voxel Binding 或 Robust Biharmonic Skinning；闭合良好时可用 bounded biharmonic weights。
5. 通过 cage/tet/最近语义表面把 proxy 权重回传到高分辨率目标。
6. 对权重施加硬 forbidden matrix：左腿不能受右腿骨影响，前腿不能受后腿骨影响；躯干过渡区域例外且有上限。
7. 使用已有 `blender_retarget_quaternius_to_generated_quadruped.py` 的 target-native 思路：rest-pose change of basis、按 chain arc length 重采样、swing/twist 限制、foot IK、contact lock、root/stride warp。
8. 通过完整门禁后仍只作为研究候选，直到同体型 canary 足够稳定。

当前 repair 脚本若暂时继续使用，至少显式传入：

```text
--cross-limb-authority nearest-rest-chain
```

而不是默认的 `largest-bone`。但这只是降低跨腿错误，不能替代骨架拟合。

## 6. 动作重定向设计

### 6.1 动作族与体型绑定

每个模板族应有自己的 gait variants：

- walk slow/normal/fast；
- idle；
- 可选 trot/run；
- 每个动作记录 stride length、cycle duration、contact phases、joint limits。

不要把跨物种动作仅因为数值门通过就视为视觉通过。

### 6.2 rest-pose correction

对每个语义关节，以源和目标 rest basis 构造 change-of-basis，再把源动作的相对旋转增量映射到目标局部坐标。平移按躯干/肢段比例缩放，旋转用 swing-twist 分解后限制到目标关节允许范围。

### 6.3 速度—步态相位一致

若场景轨迹是世界运动权威，应使用 in-place 动作，并按累计位移驱动相位：

```text
phase(t) = (phase0 + traveled_distance(t) / stride_length) mod 1
```

否则当前“手工移动 actor + 自由循环动画”会产生脚滑。根运动和外部 trajectory 必须二选一，不能同时作为权威。

## 7. 正式 QA 门

### 7.1 静态几何

- connected components、boundary/nonmanifold edges；
- 内壳、地面/背景碎片、窄桥；
- 四肢分离和 standing pose；
- 中心面和左右一致性；
- SDF/体素中的 limb 可分离度；
- guide 与选中模板的 OOD 距离。

### 7.2 骨架与权重

- joint 位于正确 semantic volume；
- chain 顺序、骨长比例、骨轴方向；
- 骨骼与表面最小间距；
- 权重非负、和为 1、top-k；
- forbidden cross-limb influence；
- 权重熵和支持范围；
- 左右镜像一致性；
- 足骨和掌/跗区域对应。

### 7.3 动画

在现有 edge/area metrics 基础上增加：

- 最大伸长与最大压缩；
- signed triangle normal flip / 局部 Jacobian；
- mesh self-intersection 和 limb-limb collision；
- 语义区域体积变化；
- 足底穿透、悬空、滑动、接触时序；
- 左右腿交叉；
- paw yaw 和 lateral excursion；
- 关节角/角速度超限；
- loop closure、root drift；
- silhouette temporal continuity。

软指标阈值应由同模板原生动作分布校准，例如 candidate 不超过 native Q99 加稳健偏差；错误侧权重、翻面、穿地等属于硬不变量。不要继续使用一个跨物种通用的 4% 阈值作为唯一批准依据。

## 8. 音视频同步审查

### 8.1 当前问题

`run_render_pass.py` 当前：

- `PlayAnimation(..., bLooping=True)` 后动画自由推进；
- `_step_animated()` 每帧只设置 actor 世界位置与 yaw，没有设置动画时间；
- 先 warmup 120 帧；
- 以 view 为外层循环，每个 view 再 warmup 40 帧，然后输出 75 帧。

因此 view0、view1、view2、view3 的 frame 0 不是同一 gait phase。`run_audio_pass.py` 却为所有视角生成同一条 80,000-sample 音频，mux 只使用 `-shortest`。这破坏了多视角同一时刻和语义 AV 同步。

### 8.2 唯一主时间轴

建议使用 48,000 Hz 的整数 tick time base：

- 15 fps 每帧正好 3,200 ticks；
- 16 kHz 每个音频 sample 正好 3 ticks；
- 5 秒正好 240,000 ticks；
- 75 帧、80,000 samples 都可精确表示。

每帧音频样本边界可用：

```text
sample_start(f) = round(f       * 16000 / 15)
sample_end(f)   = round((f + 1) * 16000 / 15)
```

不能固定使用 1,067 samples/frame，否则 75 帧总数会漂移。

### 8.3 渲染顺序

改成：

```text
warmup while animation clock frozen
reset authoritative timeline to t=0
for frame f:
    set every actor root transform at exact t_f
    set every skeletal animation to exact action time/phase at t_f
    evaluate pose once and freeze
    render all cameras from this exact pose without advancing simulation
```

可通过 UE Single Node Animation 的显式 SetPosition，或把动作烘成 75 帧 Level Sequence 并逐帧 SetPlaybackPosition/Evaluate。warmup 不得推进正式 action clock。

### 8.4 声源应绑定骨骼

当前运动声源只跟随 `placement.trajectory_m`，高度使用统一的“dog-mouth-ish”常量；这对马、牛和不同体型狗不准确。建议事件级 emitter：

- vocalization：mouth/head bone；
- footstep：发生接触的 paw bone；
- body/gear sound：对应部件 bone。

从同一 timeline 导出 emitter world path，再输入 gpuRIR。足步音由 contact event 触发；叫声若用于语义同步训练，应有 mouth/jaw 动作或明确标记为无口型同步样本。

### 8.5 timeline manifest

建议字段见随报告提供的 `avengine_timeline_v2.schema.json`。关键内容包括：

- rational/tick time base；
- frame PTS 和 sample 边界；
- actor root transform；
- action clip、action time、phase；
- pose hash；
- foot contact 和 mouth state；
- audio event 的精确 sample 区间；
- emitter bone 和 emitter path hash；
- 每个 view 的 frame pose hash。

### 8.6 mux 门禁

不要只依赖 `-shortest`。明确裁剪并回读：

- video = 75 frames；
- WAV = 80,000 samples；
- first PTS = 0；
- duration = 5 秒；
- 四个 view 同 frame 的 pose hash 相同；
- 音频 event onset 与 contact/mouth event 对齐；
- AAC encoder delay 需要在最终训练容器中测量并记录，或保留无损 PCM/独立 WAV 作为训练权威。

## 9. 许可证结论

以下是工程审计，不替代律师意见。

- **Hunyuan3D-2.1**：许可证明确禁止使用模型、输出或结果去改进其他 AI 模型，因此不能作为你的 AI 训练数据生成链路。
- **Pixal3D**：当前官方 master 和项目本地 checkout 的代码、参数、权重、文档为 MIT；第三方组件继续使用各自许可证。
- **TRELLIS.2**：模型和代码为 MIT，但官方实现单独依赖 nvdiffrast/nvdiffrec。
- **nvdiffrast / nvdiffrec**：许可证限制为非商业研究/评估。当前本地 Pixal `inference.py` 调用 `o_voxel.postprocess.to_glb()`，而该函数直接 import/use nvdiffrast 进行 UV-space rasterization；PBR preview 还调用 nvdiffrec。因此 MIT 模型本体不等于当前官方执行路径可直接商用。
- **建议**：替换 `o_voxel/postprocess.py` 的 nvdiffrast UV rasterization，可用 BSD 的 PyTorch3D rasterizer或自研 CPU/CUDA barycentric rasterizer；PBR QA 改用 Blender/UE。固定模板 UV 后，生产拟合分支可直接用 Blender bake，进一步减少此依赖。
- **Quaternius**：官网说明其库为 CC0，动物包可用于个人和商业项目。
- **SkinTokens/TokenRig**：代码/模型卡为 MIT，可作为 skeleton+weights proposal；但推荐 checkpoint 的训练混合包含 ArticulationXL 2.0、VRoid Hub、ModelsResource。对商业数据生成仍应在 provenance ledger 中单独评估训练来源和组织政策，不能只看 checkpoint 的 MIT 标记。
- **RigAnything**：官方 Adobe Research License 为非商业研究，不适合正式商业路线。
- **Step1X-3D**：官方仓库为 Apache-2.0，watertight TSDF geometry 适合做几何 proposal A/B；但仍需对 checkpoint、texture baker 和所有第三方依赖逐项审计，且它本身不会解决 rig/animation authority 问题。

## 10. 按优先级修改

### P0：先让正式路线稳定、同步、可审计

1. 把 `run_generated_quadruped_deformation_stabilization.py` 默认分发改为 template-bank route；raw topology route 显式标记 experimental。
2. 建立狗的 5 个 morphotype 模板，先只覆盖 Dog Walk/Idle。
3. 把 foot-contact、lateral-gait、triangle flip、自交、joint-limit 和 foot sliding 加入最终硬门。
4. 增加 `timeline.json`，并让 render/audio/mux 都只读它。
5. 修复 UE 渲染顺序和显式动画时间；同一帧所有相机必须共享 pose hash。
6. 替换 Pixal/TRELLIS 导出链中的 nvdiffrast/nvdiffrec，或取得独立商业授权。

### P1：提高外形拟合质量

1. 给 `blender_fit_i23d_to_animal_template.py` 增加 template selector、landmarks、deformation graph/ARAP、joint/foot/symmetry/volume/collision constraints。
2. PBR 统一烘到固定模板 UV。
3. 按形态参数生成 gait variant，做 stride warp 和 contact IK。
4. 以每个模板的原生动作建立 QA baseline。

### P2：扩展未知拓扑研究支线

1. TokenRig/SkinTokens 生成多个 target-native skeleton candidates。
2. SDF/voxel proxy + GVB/RBS target-native weights。
3. 使用现有 target-native retargeter，而不是复制源权重。
4. 只有通过确定性硬门和视觉审核的实例才能升级为新 template revision。

## 11. 建议的验收标准

狗 MVP 达标时应满足：

- 同一模板可生成多种毛色/纹理/受限体型，全部复用 topology/rig/action；
- 不同 dog morphotype 自动选模板，不跨域硬拟合；
- Walk/Idle 全周期无翻面、自交、跨腿权重、穿地和明显脚滑；
- 四视角同帧 pose hash 一致；
- 75 帧、80,000 samples、5 秒完全回读；
- 足步事件由接触生成，叫声与口部状态有明确同步标签；
- 每个资产都有模型、依赖、源图、音频、模板和修改链的 provenance/rights manifest；
- Hunyuan3D 输出不进入训练数据；nvdiffrast/nvdiffrec 不出现在商业执行路径。

## 12. 审查限制

本次完成了源码、文档、许可文件和相关静态测试审查，并运行了 59 个项目测试。当前环境未实际启动项目所需的 Blender/UE/SPEAR GPU 渲染链，因此没有重新生成真实 41 帧 GLB canary、UE 四视角视频或 gpuRIR 音频。上述结构性问题来自明确代码路径和项目已有真实 QA 记录；最终阈值仍应通过你的 GPU canary 集校准。

## 13. 2026-07-16 实施补充：固定视角自动调平

审查后已经在真实浅色比格上实现并运行了一个更稳定的绑定前方向门。三分之四
参考图的 30° 相机方位用于保持四条腿分离，但不再交给人工逐只微调，也不从
Pixal mesh 反推应用角度。代码固定应用 profile 在生成前声明的 30°，几何拟合
只负责验收残差；三组原始躯干轴测量为 31.25755°、32.09411°、31.20601°，
固定规范化后的残差为 1.25755°、2.09411°、1.20601°，全部通过 3° 门限。
失败时拒绝该单次结果并修改 pose guide、prompt 或 profile，不允许只换 seed。

浏览器在这个模式下锁死躯干轴调整，只允许人工选择 0° 或 180° 头尾方向。
当前用户接受的比格选择为 180°，与固定 30° 合成后的绑定 yaw 为 -150°。
轴增量以及 ±90° 选择都会返回契约错误；旧审核决定和旧媒体均未覆盖。

可重复实现和真实证据见：

- `/data/jzy/code/AVEngine/external/SPEAR/docs/controlled_animal_declared_view_canonicalization_v1.md`
- `/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/dog_beagle_open_tricolor_direction_canary_v3_declared_axis_20260716/review_manifest.json`
- `/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/dog_beagle_open_tricolor_locked_paw_skeleton_binding_spike_v1_20260716/review_v1/index.html`

当前隔离 Walk/Idle 已被用户认为效果可接受；四脚横摆和脚掌 yaw 审计通过。
但严格变形审计仍保留少量局部最大拉伸异常，因此状态仍是
`research_candidate`，不能据此声称已经完成正式资产注册或吞吐优化。
