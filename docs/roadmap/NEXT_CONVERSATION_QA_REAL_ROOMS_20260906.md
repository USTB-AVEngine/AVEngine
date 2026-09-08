# 新对话执行 Prompt：AVEngine 真实房间统一编排、完整 QA 与双耳音频验证

你接手的是一个已经有代码和真实产物的 AVEngine 任务。请直接在现有成果上完成下面的剩余实现、修复和验证，不停留在设计或能力说明。先读指定文档和实际代码，核实状态后推进；已经验证且未受本次修改影响的工作不要重做。

## 1. 我的研究目标与本次范围

我要用 AVEngine 在真实室内房间中生成符合生活逻辑的视听 QA 数据。上层应按照“题目需要什么条件 → 房间提供什么能力 → 选择适合的活动、轨迹、镜头与发声安排”统一编排。

本次需要完成：

1. 统一 QA 条件与房间能力的表示、匹配和实际调用，而不只是写一张说明表。
2. 优先复用现有真实房间、可行走空间、路径求解器和轨迹模块，支持站立、行走及已有静坐活动。人物/动物、相机、听者、声音时序均由 AVEngine 计划驱动，不写 A/B/C 等房间坐标特例。
3. 按 `QA-01～QA-24` 统一目录逐项补齐缺失的采样、问题及标准答案生成、判分和实际验证。不同题目的缺口不同，优先复用既有实现，不把后十二项一概当成从零开发，也不把目录存在当成实现完成。
4. 核验多个声源混合成双耳两声道的音频链，重点检查真实房间几何缺损、错误穿墙/逃逸、坐标与时序、动态 RIR 和混合是否可信。
5. 交付能检查、重跑和继续扩展的数据、代码、QA 覆盖表与存储统计。

本次不做“补造房间条件”：不为凑题增加椅子、桌子、屏风等家具，不给缺少座点的房间强补坐姿，不为凑条件改变原始房间。按现有资产库实例化人物、动物和声源属于正常 Episode 编排，可以做。某个房间不适合某题时，换现有房间、区域或活动；如果资源确实无法支撑某题，先充分检查已有资源，再明确报告具体缺口，不能造一个无效样本假装完成。

不要求每个房间都覆盖所有题型；完整覆盖由适合的场景池共同实现。没有必要强求每个房间都坐着，真实房间先用站立/行走和已有路径求解即可。

眼镜、手环、衣纹等未登记验证的人物细节，以及个性化、偏好更新、跨片段/长期记忆，已经另存为下一篇论文 ideas，不进入本次范围。不新增坐下过渡、家具交互、IK 或任意家具自适应系统。

## 2. 权威代码与环境

服务器：`48g-jump`，连接示例：

```bash
ssh -o ControlPath=none -o BatchMode=yes -o ConnectTimeout=15 48g-jump
```

权威实现/集成目录：

```text
/data/jzy/tmp/wt-multi-home-activity-integration
```

集成分支：`codex/multi-home-activity-integration`。

2026-09-06 交接时核实的功能基线为 `1ba6294c6cafbd71264775133d3b19d3104cabae`，当时工作树干净。新增本交接文档可能使 HEAD 再有文档提交；开始时读取实际 Git 状态，接续现有成果，不 reset 到这个参考提交，不退回 main。并行需要时可建服务器上的隔离工作树，最终把本任务修改汇入这条集成分支，保留他人修改。

代码只在服务器的真实依赖和保留数据上改、测、修。本地只用于补丁传输、只读审计和交付文件镜像。

不要修改或切换运行中的 Studio：已知其仓库根为 `/data/jzy/tmp/wt-qa-v3-engine-completion`，健康接口为 `http://127.0.0.1:8765/api/health`。开始时核实当前状态。不要动旧工作树、共享源资产和其他人的 GPU 作业。不 push、不合并 main、不正式发布或切换服务。

已授权任务范围内的可逆代码修改、分类器、普通运行时验证，以及必要 UE/GPU/CPU 作业。运行前检查设备和进程；此前本任务使用 GPU2，但必须重新核实。每次生成使用 fresh/no-clobber 输出，失败则停止失败操作及其依赖，保留现场，诊断修复后用新输出重试。

关键运行资源：

```text
Python: /data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python
UE: /data/UE_5.5/Engine/Binaries/Linux/UnrealEditor
Private UE project: /data/avengine_external/workspaces/multi_home_activity_20260905/ue_stage/SpearSim/SpearSim.uproject
SPEAR SDK: /data/avengine_external/spear-host-sdk/avengine-spear-ext-cp312-8a36d4d-20260821T0030Z
Habitat runtime: /data/avengine_external/runtime-prefixes/avengine-habitat-pbr-ibl-c78db29-20260821T0111Z
RLR SDK: /data/avengine_external/rlr-sdk/RLRAudioPropagationPkg-runtime-b-20260820T2010Z
Magnum: /data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages
```

历史测试使用 `PYTHONPATH=src:/data/jzy/tmp/wt-multi-home-activity-integration/tmp/native_python_addons_v1`；soundfile/cffi 等补充依赖位于这个任务目录，没有改共享 Conda。

可使用 subagent 并行完成独立、有边界的工作。GPT-6 Astra 负责整体判断、必要的 3D/视觉审查；普通实现、测试和日志/数据审计可交给 GPT-5.6 Sol。明确文件所有权与集成责任，避免重复调查和互相覆盖。

## 3. 先读哪些文件，哪些已经完成

主阅读入口，均相对于服务器集成仓库。无需重放整个旧对话；历史文档里的当前状态、人物属性和全员入画等表述，以本次用户范围及实际证据校正，既有安全措施仍保留：

- `docs/qa/QA_UNIFIED_CATALOG_20260905.md`：唯一的题目编号、输入条件、输出形式、实现状态及旧 QS/card 映射。
- `docs/roadmap/MULTI_HOME_FURNITURE_POSES_20260905.md`：工作目录、产物和修复历史；顶部最新状态优先于下方历史快照。
- `docs/roadmap/MULTI_HOME_QA_CAPABILITY_AUDIT_20260905.md`：源码审计，旧编号仅作追溯。
- `docs/roadmap/QA_TYPE_DESIGN_V3_20260830.md`：本篇既有题义、对照和验证约定。不要把 17 张卡当成 17 个已实现类型。
- `docs/roadmap/NEXT_PAPER_PERSONALIZED_AV_MEMORY_IDEAS_20260905.md`：仅用来识别本次排除范围。

已经完成的基础：

- 三套自制房间最终资产：A/C 为 `/data/avengine_external/workspaces/multi_home_activity_20260905/living_props_v7_candidate/{room_a,room_c}`；B 为同父目录下 `detailed_v8_linear_materialfix/room_b`。不要误用 B v6/v7 染色诊断版本。
- 三屋各有 4 人、240 帧/15FPS/16 秒/720p 的实际 SPEAR/UE 多模态捕获，以及多源双耳 16kHz 音频、各声源湿声轨和成片。
- 三屋每屋有 9 道 research-only QA 实例：`QA-03 ×1`、`QA-12 ×4`、`QA-02 ×4`，均完成数据/答案检查。不是九种题型，也不是待评模型全答对。
- 当前稳定 QuestionSpec 有 12 类实现和保留的原生样例；统一目录共 24 种问法，包含待接入的时间变体、计数与组合题。
- 上层静坐入口已实现：普通 Studio research template `furnished_seated_visual_episode` 在一个请求内生成 plan 再执行 SPEAR，不需要用户先手写 episode_root。算法在 `src/avengine/rooms/furnished_episode.py`，旧 CLI 仍兼容。
- 该入口已经有真实 15 帧测试，证明调用链、坐姿与读回贯通；它不等于默认 Apartment/HM3D idle/walk 主生成器全面支持静坐。它也不自动理解任意扫描坐标或家具尺寸。
- 默认几何相机曾得到可用的厨房环境远景，但蓝衣人物只有约179个可见像素，不足以可靠辨衣色。不能把 native_pixel pass 当成人物属性可观测性证明。后续条件匹配应解决这种具体问题，同时保留合理的自然遮挡/部分出框。
- 已修复坐姿参考系、人物偏移、听者/相机坐标转换、真实干声总线卷积、时间偏移、B材质导出色彩空间等问题，不要无理由重写。
- 已合入 `591926b`：派生声学几何改变后重新计算可重放的射线 QA，不能继承源网格的旧 pass；缺少重放输入时明确 not_run。历史数据未覆盖。

真实产物集中在：

```text
/data/datasets/avengine_workspaces/multi_home_activity_20260905/root/multi_home_v6_native_review_20260905/
/data/datasets/avengine_workspaces/multi_home_activity_20260905/integration_sol_v6/
```

其中 `integration_sol_v6/run_commands.md` 有准确的三屋计划、声学包、音频、QA 和 mux 命令，是复现参考；它使用已选定相机的手工配方，不等于通用编排已经完成。该文件里的工作目录及相对tmp路径属于当时worker；复用时解析真实输入路径，并使用当前权威代码和新的输出位置，不要无意运行旧工作树代码或覆盖已有结果。上层静坐原生证据在：

```text
/data/datasets/avengine_workspaces/multi_home_activity_20260905/furnished_seated_studio_sol/studio_request_a_seated_15_v2/
```

## 4. 实现顺序与工作流

第一步，做有界的实际源码/资源核对。确认现有真实房间、可用人物/动物/声音、路由、路径求解和音频入口，形成逐项缺口，不做无关的全项目重审。

第二步，落实“题型条件—房间能力—Episode编排”。区分房间潜在能力和某个 Episode 已实际满足的条件；未知或仅推断的能力不能当作验证通过。复用已有房间语义、导航和相机候选，选择适合的房间/区域。不要把缺少坐姿误判成无法生成大部分 QA。

第三步，沿现有上层机制接通通用请求。由 AVEngine 输出明确的人物/声源实例、动作、轨迹、相机/听者轨迹和声音时序，后端执行计划并保存实际读回。避免场景名称分支、固定说话人顺序、固定衣色—声音映射或固定事件时刻成为数据捷径。相机条件应服务具体问题：自然遮挡可以是有效答案，不能用全员完整入画作为房间统一门槛。

第四步，逐项完成 QA-01～QA-24。对每项实现或复用：有效样本采样、题面/选项、由实际证据推导的标准答案、模型回答的判分及必要错误处理。根据目录落实时间窗、数值/分类输出与支持的题目形式，遵循项目已有评测约定。不能只生成模板文字，也不能把“引擎算出标准答案”混同“模型评测通过”。

第五步，先用1～2个现有真实房间完成可检查的纵向贯通，覆盖静态、移动、时间条件和遮挡的代表问题，再扩展题型/房间覆盖。这个首版是里程碑，不是整个24问法任务的完成标志。复用已有模型评测入口；按实际可用模型做合理规模的测试，分别报告数据有效性、模态贡献与模型成绩，不因为模型答错就任意改答案或剔除困难样本。

可能复用的源码入口：

```text
src/avengine/qa/question_spec.py
src/avengine/qa/pixel_visibility.py
src/avengine/rooms/furniture_layout.py
src/avengine/rooms/furnished_episode.py
src/avengine/studio/templates.py
tools/studio/run_furnished_seated_episode.py
src/avengine/timeline/current_mp3d_dynamic_audio.py
src/avengine/capture/acoustics.py
src/avengine/capture/dry_audio.py
src/avengine/acoustics/rir_cache.py
src/avengine/acoustics/semantic_rir_cache.py
src/avengine/acoustics/qa.py
src/avengine/acoustics/research_cleanup.py
tools/acoustics/verify_package_ray_leakage.py
```

## 5. 音频与扫描缺损：不能略过的正确性要求

本次沿用多个声源→双耳两声道，不扩展四声道/阵列。

- 核实实际声源/听者位置、坐标轴、单位、朝向、声道顺序、HRTF、采样率与统一时钟。两个通道的文件存在，不足以证明是正确双耳渲染。
- 移动场景复用已有动态 RIR 关键帧及卷积/混合机制；不要拿一条静态 RIR 冒充移动音频。检查关键帧采样、过渡、混响尾音与多源事件时间，保留增益/混合记录，不用逐源归一化掩盖几何或增益错误。
- 区分视觉网格、声学网格、导航/碰撞表示，记录各自来源与坐标对应；视觉PBR好看不等于声学材料系数准确。
- 当前清理只处理部分退化/极小三角形，不会自动补大洞、缝边或辨别真实门窗。边界边不等于漏声；零逃逸的少量探测也不等于整屋封闭。
- 当前 ray-check pass 可能只表示 CPU/native 对命中和逃逸的判断一致。自制三屋旧包的 pass 仅含一个表面 self-hit，automatic probe 未运行，不能当防泄漏证据。
- 在实际使用的区域与路径布置探测，查看逐起点/逐方向结果，并用已有语义/原始资料识别墙体控制与真实开口。仅“最终撞到某面墙”不能排除错误穿过本应封闭的墙进入邻室。
- 当前任务优先选择几何较完整的房间/区域，做准确诊断与已有工具的合理处理。不要为提高通过率盲目封住门窗，不改变原始扫描，不把任意大洞的自动重建暗中加入本次范围。无法可靠区分开口/缺损时明确报告依据不足。
- 有适用实测或可信参考 RIR 时可以复用现有分析对照；没有参考时，明确区分实现/几何正确性检查与真实混响绝对精度，不能靠“RLR有非零输出”宣称声学准确。

可复查的真实几何案例：

```text
/data/datasets/avengine_workspaces/AVEngine-soundspaces-frl/tmp/soundspaces_mp3d_17DRP5sb8fy_acoustic_package_20260730_01
/data/avengine_external/datasets/mp3d_example_scene_1.1/scene_datasets/mp3d_example/17DRP5sb8fy/
/data/jzy/tmp/wt-multi-home-ac-detail-astra/tmp/scan_boundary_review_20260905/
```

此案例的声学来源为semantic PLY（约301万三角），可读视觉GLB约21.6万三角。实际面集合比较未发现补洞面。旧报告只测1个起点×4方向，native验证未运行，不能当全屋无泄漏。其他保留包也有58/64、45/64射线逃逸的例子，不同包和起点不能作为修复前后直接对比。

声学参考仅在需要时查看主来源：

- https://arxiv.org/pdf/2206.08312 （SoundSpaces 2.0，含扫描孔洞/材料局限及动态音频说明）
- https://github.com/facebookresearch/sound-spaces/blob/main/SoundSpaces2.md

## 6. 数据保存、缓存与存储成本

组织关系必须是：房间共享资源 → Episode完整记录 → 多道QA引用同一Episode。不能为每道题重复复制视频、房间和声学包。

长期交付至少包括：

- 视频母版/可播放视频及独立无损双耳WAV；带声音的MP4可作预览，明确模型实际使用的媒体版本。
- Episode清单：资源引用、配置、时钟、种子、代码/运行时版本、全部输出索引。
- 计划与实际执行记录分开保存：人物/声源/相机/听者位置和朝向、动作/动画相位、时间/事件记录。
- 各源干声来源和处理信息、湿声轨/混合记录，声学材料、HRTF与求解配置。
- 本篇题目所需的像素/几何证据、实例ID映射、可见性/遮挡记录。全量深度/ID/mask等扩展证据单独组织、计量并支持明确的留存配置，不能为了报小体积悄悄丢证据。
- 每题唯一ID、QA-xx类型、题面/选项、观察窗口、标准答案、判分和证据引用。标准答案/引擎真值与模型输入分开。
- 验证结果、不可答/失败原因；实际模型评测的请求配置、原始回答与分数，没有运行就明确not_run。

元数据用JSON/JSONL并可无损压缩，大数组用现有NPZ/NPY等格式；不要把所有像素/RIR数值塞进JSON。原始资源集中引用，不按Episode重复拷贝。

RIR默认是生成期缓存：生成时实际持久化，以便共享与续跑；房间批次中所有依赖它的生成和校验完成、永久产物可读后，清理本任务新创建且确认无后续引用的数值RIR缓存。长期保存求解配置、空间/时钟索引、来源和关键验证记录，必要的少量审查样例可单独保留，不默认永久保存全量RIR。

只能清理本任务自有、可重建、已结束使用的缓存和重复调试产物；不得删除历史保留数据、源资产、其他任务缓存或已交付永久数据。失败现场保留。清理缓存是明确的任务阶段，不依据文件名随意rm：例如当前rir_*.obj是几何文件，不是数值RIR。

优先复用已有RIR cache读写、UE派生数据/已加载场景、声学几何、可行走空间/路径库、几何相机候选。复用需匹配实际依赖：RIR不仅依赖房间，还依赖几何/材料、源和听者姿态、HRTF、布局、采样与求解参数；时序相关状态不能跨不连续Episode乱用。几何改变后不能继承旧QA报告。复用现有完整性机制，不另外造hash锁或冻结contract。

已核对：四人示例入口保存了湿声和报告，但没有单独保存数值RIR；主引擎已有RIR缓存实现，应接入复用，而不是重新发明一套。

每批报告实际空间：永久数据、可清理缓存峰值、共享房间资源，以及按最终有效题数摊销的每题字节数。此前三屋16秒/720p/15FPS、每段9题的静态样例为：成品+gzip完整JSON约0.21MB/题；加全量深度/ID/mask/湿声约16.2MB/题；调试全集约57.1MB/题。它们不是移动真实场景的保证。4～8Mbps视频的轻量预算约1～2MB/题（仍假设16秒和每段9题）。不要用计划生成数、重复改写数或尚未通过有效性检查的题数美化平均成本。

## 7. 验收与工作纪律

最终交付要包含：

1. 已合入服务器集成分支的可运行代码与简洁运行方法。
2. 一份QA-01～QA-24逐项覆盖表：题义、输入条件、采样/生成/判分状态、实际样本、验证结果和确实无法满足的缺口。
3. 多个真实房间上的实际音视频与可核对记录，证明走的是统一上层编排而不是房间专用坐标脚本。不要把1～2房间首版或三种题型的演示当成全部24问法完成。
4. 双耳/动态/多源及所用区域几何的验证结果与限度；模型成绩与生成器正确性分开报告。
5. 符合上述组织和缓存生命周期的导出结果，以及真实存储/耗时统计。

按风险做必要测试与实际运行；未受影响的已验证步骤不要反复全量重跑。默认不新增hash、冻结contract、baseline或gate，不删除已有安全措施。正常使用已有完整性机制和分析对照值允许。保留research_candidate/research_only与正式准入边界，不把研究产物升格为正式数据集。

普通实现、分类器和任务内重试已获授权，不反复问“是否继续”。仅在缺失核心目标、权威数据源或必要授权且无法合理推断时澄清，其他可做部分继续。

进度按实际完成项汇报，说明已知问题和下一步，不用配置存在、进程启动、文件链接或单一pass当完成证明。长任务每约一分钟给一次有意义的可读更新，及时交付可查看的文件/媒体；不要长时间只留工具输出。曾发生模型容量不足、上下文传输中断及历史额度错误，恢复时先说明实际错误并续接已有成果，不重做整项任务，也不要擅自换模型或消耗额度重置信用。

请先给出基于实际核对的简短缺口与执行顺序，然后直接开始实现。把首轮真实房间贯通作为里程碑继续推进到约定完成标准；不要停在计划、文档整理或“可以继续”。
