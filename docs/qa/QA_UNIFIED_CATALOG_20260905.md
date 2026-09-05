# AVEngine 统一 QA 目录

2026-09-06。根据用户要求补充连续编号、完整题目表和当前实现/真实房间覆盖状态。

对外讨论统一使用 **QA-01 至 QA-24**。这是将已有题目和本篇设计问法去重后得到的统一目录。当前 `unified_catalog.py` 已注册 QA-01 至 QA-24，`unified_scoring.py` 提供对应形式的判分；表中的“已实现”表示生成器/判分器具备该题意的明确实现，真实房间仍会因条件不足返回 `deferred`。旧 QS/card 编号只作代码映射，现有接口和数据编号不改。

相比上一版纯文字分组，本版把“发声期间动静”和“声停后动静”等具体时间条件分开编号；重复的旧卡6原形态/卡10仍合并。不同答案形式（例如左右、方向扇区、数值角度）在同一题意下标明实际支持范围，不把尚未实现的细粒度输出算作完成。

**输入记号。**

- **V**：RGB视频/相关图像帧。
- **A**：音频的内容与时间信息；这类对照题通常不需要左右声道差。
- **B**：保留左右声道的双耳音频，包含内容及空间线索。
- **时间序列/时间信息**：视频和音频的顺序、查询时刻或事件时间条件，不是另一种传感器模态。

正式问题的基础交付可以统一为视频+双耳音频+题面；下表“输入依赖”说明该问法拟考察什么、哪些可以作为单模态对照，**不是已经实测证明某模态必不可少**。引擎的深度、实例mask、坐标、事件表等用于产生/核对答案，不作为默认送给被评模型的输入。

**完整题目表。**

表中例句用于说明题意；“已实现”不等于每个房间都满足该题条件，也不等于模型评测通过或正式准入。人物例句只使用当前可控衣色、位置、台词和状态；不使用未登记验证的眼镜、手环或衣纹。

| 编号 | 题目类型与例题 → 答案 | 输入依赖 | 关键条件 / 答案形式 | 当前状态 |
|---|---|---|---|---|
| **QA-01** | **外观→是否发声**：蓝衣人发过声吗？→ 是/否 | V+B | 目标外观可辨；整段事件集合；布尔答案 | 已实现；A/B/C及native N均有pass样本 |
| **QA-02** | **声音/台词→外观**：说“Please call Stella”的人穿什么颜色？→ 白色 | V+B | 唯一受控声音/台词；衣色证据可见；外观枚举 | 已实现；A/B/C及native N均有pass样本；C由final QA-02 override纳入 |
| **QA-03** | **谁先发声**：最先说话的是哪位？→ 蓝衣人物 | A判顺序；回答衣色用V+B | 至少2个候选；首发先后可区分；人物描述/ID | 已实现；A/B/C及native N均有pass样本 |
| **QA-04** | **发声当时的相对方位**：这句话来自听者左边还是右边？→ 左 | B | 查询发声时刻；使用该时刻听者朝向；左右 | 已实现；A/B/C及native N均有左右pass样本；扇区和度数是QA-13的独立形式 |
| **QA-05** | **发声区间重叠**：这两段说话重叠过吗？→ 是/否 | A | 至少2段声音；比较发声区间；布尔答案 | 已实现；A/B/C及native N均有pass样本 |
| **QA-06** | **发声期间是否移动**：说这句话期间，他在动吗？→ 是/否 | V+B；声学运动对照 | 有声音锚；查询发声期间运动；布尔答案 | 已实现；A/B及native N有pass样本，C未请求 |
| **QA-07** | **从哪侧入画**：那个人从画面哪侧进入？→ 左侧 | V+时间序列 | 画外→至少部分可见的转变；入画侧 | 已实现；B有pass样本；A和native N因无明确入画转变deferred |
| **QA-08** | **发声时的可见状态**：说话的人此时是什么状态？→ 部分遮挡 | V+B | 声音定位/绑定目标；当前帧；四状态之一 | 已实现；A/B/C及native N均有pass样本 |
| **QA-09** | **完全遮挡后是否重现**：那个人完全被挡住后又出现了吗？→ 是/否 | V+时间序列 | fully_occluded→可见；单片段；布尔答案 | 已实现；B有pass样本；A和native N因无完整重现转变deferred |
| **QA-10** | **遮挡物身份**：桌子还是椅子挡住了那个人？→ 桌子 | V | 目标确被遮挡；遮挡物有登记语义；实体名/类别 | 已实现；A/C有actor-occluder pass样本；native N仅保留actor occluder identity且无唯一对象deferred |
| **QA-11** | **部分遮挡转清晰可见**：他从部分遮挡变成清晰可见过吗？→ 是/否 | V+时间序列 | visible_occluded→visible_clear；布尔答案 | 已实现；A/B及native N有pass样本 |
| **QA-12** | **外观→说话内容**：蓝衣人说了什么？→ Ask her to bring these things with her from the store. | V+B | 外观可辨；目标唯一受控台词；逐字文本/台词选项 | 已实现；A/B/C及native N均有pass样本 |
| **QA-13** | **发声结束后的方位**：说出第二句话的人，片尾在哪个方向？→ 左 | V+B+时间序列 | 声音锚在前；查询静默后的时刻；扇区/角度 | 已实现；L与native N有pass样本；边界采用±45° half-open sectors，Open角度容差仍是placeholder |
| **QA-14** | **给定时刻的距离比较**：第2秒，蓝衣人和白衣人谁更近？→ 蓝衣人 | V | 题面给定时刻；2个目标可比较；目标描述 | 已实现；A/B有pass样本；其他本批房间因距离margin不足可deferred |
| **QA-15** | **发声期间靠近/远离**：这段声音发出期间，声源在靠近还是远离？→ 靠近 | B；音频运动对照 | 发声期间发生距离变化；靠近/远离 | 已实现；B及native N有pass样本 |
| **QA-16** | **声停后的距离变化**：他停止说话后，到片尾比发声时更近还是更远？→ 更近 | V+B+时间序列 | 发声时锚定人；随后静默；比较两个时刻距离 | 已实现；A/B有pass样本；native N与L因距离变化未达到 margin 而deferred |
| **QA-17** | **声停后是否移动**：说完第二句话的人，随后动过吗？→ 是/否 | V+B+时间序列 | 发声锚与后续静默查询窗分离；布尔答案 | 已实现；A/B/L及native N有pass样本 |
| **QA-18** | **指定时刻谁在发声**：第3秒正在说话的是谁？→ 白衣人/多人/无人 | V+B+时间信息 | 明确视频时钟t；可并发或静音；人物集合/无人 | 已实现；A/B/C及native N均有pass样本 |
| **QA-19** | **目标首次发声时刻**：蓝衣人第一次说话在第几秒？→ 2.4秒 | V+B+时间序列 | 先绑定外观目标；查询首个发声事件；秒/时间区间 | 已实现；A/B/C/L及native N均有pass样本；数值容差由scorer参数控制 |
| **QA-20** | **可见候选谁发声/都不是**：刚才的声音是画面中哪位发出的？→ 都不是，声源在画外 | V+B | 有可见候选及画外/遮挡源；可答负例；人物/都不是 | 已实现；A/B/C及native N均有pass样本 |
| **QA-21** | **外观→声音类别**：深色狗发出什么声音？→ 低吼 | V+B | 多源候选声音真实存在；目标声音类别唯一；类别标签 | 已实现；C有多声音类别pass样本；A/B/L/native N因类别域不足deferred |
| **QA-22** | **出现过的实体数/发声个体数**：这段里出现过几人，其中几人说过话？→ 4人、3人 | V+B+时间序列 | 明确统计窗；按不同个体去重；两个整数 | 已实现；A/B/C及native N均有pass样本 |
| **QA-23** | **发声事件次数**：这段中独立发声开始了几次？→ 4次 | A+时间序列 | 明确事件定义和统计窗；次数不是人数；整数 | 已实现；A/B及native N有pass样本；C因非语音多脉冲segmentation未复核而deferred |
| **QA-24** | **先发声者在片尾的可见状态**：最先说话的人，片尾是什么状态？→ 完全遮挡 | V+B+时间序列 | 先按声音选目标，再看后续状态；四状态之一 | 已实现；A/B/C及native N均有pass样本 |

**本批实际覆盖与状态（2026-09-06）。**

当前实现由 `src/avengine/qa/unified_catalog.py`（`CATALOG_VERSION=20260906`，QA-01 至 QA-24 生成）和 `src/avengine/qa/unified_scoring.py`（closed-set、transcript、angle、time、count、MCQ 判分）组成。真实房间的结果按“一个通过的 unique question item”计数；每个 item 可以同时包含 Open 与 MCQ 两种 form，form 缺证据时单独 `deferred`。A/B/C/L 是四个 authored Episode 变体，N 是 native Apartment episode：

| 代号 | Episode / 实际交付目录 | 请求 | 有效 unique questions | deferred | 形式与边界 |
|---|---|---:|---:|---:|---|
| A | `walk_pair_a_v6/delivery_final_v1` | 24 | 19 | 5 | authored walking；research_only |
| B | `walk_four_b_v2/delivery_final_v1` | 24 | 21 | 3 | authored walking；research_only |
| C | `vocal_classes_c_v2/delivery_final_v2` | 14 | 13 | 1 | authored standing；QA-02 由 final override 纳入；QA-23 非语音多脉冲 segmentation 未复核 |
| L | `late_direction_a_v2/delivery_final_v2` | 5 | 3 | 2 | authored late-direction；QA-13 Open/MCQ 均通过 |
| N | `native_apartment_v6/delivery_final_v1` | 24 | 18 | 6 | native Apartment；QA-13 仅 Open 通过，MCQ 因同为 front sector deferred |

总计为 **74 个有效 unique question items**（A/B/C/L 的 56 加 N 的 18），覆盖 4 个房间、5 个 Episode。按 QA 编号取并集，QA-01 至 QA-24 每一类至少有一个真实 `pass` 样本：QA-01/02/03/04/05/06/08/11/12/18/19/20/22/24 可由 N 直接复核；QA-07/09 由 B；QA-10 由 A/C；QA-13 由 L/N；QA-14/16 由 A/B；QA-15 由 B/N；QA-17 由 A/B/L/N；QA-21 由 C；QA-23 由 A/B/N。QA-13 修复只改变 private `forms.open.convention` 的机器值，公开 `question`/`model_input` 保持不变。修复后，74 个 Open 与 73 个 MCQ form 的 gold roundtrip，以及 73 个故意答错的 MCQ 检查均通过。各 Episode 的生成与导出仍标记 `research_only`，模型评测或正式准入不由这些 engine-valid 样本推出；已有模型 batch 的 35/60 也只作 research-only 结果。

N 的权威 native 产物在 `/data/datasets/avengine_workspaces/multi_home_activity_20260905/root/qa_real_rooms_20260906/native_apartment_v6`：`capture/ue_visual_only.mp4` 为 240 帧、15 Hz、16 秒，`delivery/audio/four_speaker_sequential_mixture.wav` 为双声道 16 kHz、256000 samples，问题结果在 `delivery_final_v1/result.json`，导出计数在 `delivery_final_v1/export/manifest.json`。N 的请求为 QA-01 至 QA-24，结果为 24 requested、18 valid、6 deferred；导出中的 24 answers 含 deferred 行，不能把它们计为 24 个有效题。四个 authored 交付和 N 的完整运行、存储与验收证据见[真实房间交付与验收报告](QA_REAL_ROOMS_DELIVERY_20260906.md)。

**场景输入条件表：遮挡程度等属于条件维度，不是另加一组题型。**

| 条件维度 | 可用值 / 定义 | 当前使用范围 |
|---|---|---|
| 目标类别 | 人物、狗、猫等已登记实体 | 本批四个 authored 房间与 native Apartment 均为已登记人物；动物问答依据已有引擎资产与旧样例，未在本批实际房间重出 |
| 人物外观 | 蓝、粉、绿、白上衣；可观测的位置描述 | 当前控制衣色来自资产标签；眼镜/手环/衣纹不支持本轮题目 |
| 声音内容 | 唯一逐字台词，或有登记类别的非语音声音 | A/B/C 与 N 使用实际录音/台词；QA-21 的多声音类别样本来自 C；一般声音类别仍需登记类别和唯一绑定 |
| 发声结构 | 静音、单源、多源；依次发声或重叠发声 | A 使用 overlap，B/C/N 使用 sequential；dry 事件按实际音频计划计时，wet 混响尾单独保留 |
| 动作 | 静坐、站立、行走；静止→移动/移动→静止 | A/B/N 为 walking，C 为 standing，L 为 late-direction 变体；QA-06/15/17 等只在有对应运动证据的 Episode 通过 |
| 时间条件 | 给定t秒、第n次发声、发声期间、声停之后、片尾 | 本目录是单Episode内的条件；跨段/长期记忆移至下一篇ideas |
| 清晰可见 | visible_clear：当前像素证据判为清晰可见 | 现有四状态之一；不是要求全身每个部位均在画框里 |
| 部分遮挡 | visible_occluded：目标有可见部分，也有被其他几何挡住的部分 | 例如人被桌面或椅背挡住；允许作为合法场景和答案 |
| 完全遮挡 | fully_occluded：目标在视野内，但被遮挡物完全挡住 | 与画外区别处理；QA-08/QA-24的合法答案 |
| 画外 | out_of_view：目标不在当前视野内 | 不等于完全遮挡；该状态的遮挡比例没有有效的入画目标分母，不填成100%遮挡 |
| 遮挡比例 | occlusion_fraction = 1 - visible_pixels / target_pixels，范围0至1；画外为null | 分母是当前视野内target-only像素，不是全身面积；当前没有另设轻/中/重分桶，也未新增比例回归题 |
| 部分出框 | 身体只有一部分落在画面里 | 是构图/视野条件，不能直接说成被桌椅遮挡或全房间失败；具体题目检查所需信息是否可见 |
| 遮挡物 | 有登记身份/类别的桌、椅、墙等对象 | QA-10需要原生像素/深度及对象登记共同支持身份 |
| 相对位置参考系 | 查询时刻的听者位置与朝向；或题面明确指定的目标参考物 | “左/右”“更近”等必须说明相对谁、哪个时刻；引擎坐标不能当模型已获知的信息 |
| 统计窗和事件单位 | 某一帧、明确时间窗、整段内至少一次出现；独立发声事件 | 人数、发声个体数、发声次数分开；复合录音内多次叫声不能直接用素材播放次数替代听感叫声数 |
| 缺少证据 | 没有有效像素、声音、时间锚或唯一目标 | 标记not_run/deferred/不可答原因；不是第五种可见状态，也不能伪造为画外 |

“目标人物可见”不等于“上衣区域可辨”。衣色题要能看到足以判断衣色的证据；目前native mask只证明人物可见/遮挡，不能单独认证细粒度外观。对非目标人物的自然遮挡和画外，不做全房间失败判定。

**可直接看的本轮实际题目。**

A/B/C 三个 authored 房间的原生 SPEAR/UE 视频和真实 RLR 双耳音轨均已完成；显式 all-speaking-targets 模式每屋实际生成 9 道实例，evaluator pass 9、deferred 0，仍为 `research_only`。每屋九题对应 **QA-03 一次、QA-12 四次、QA-02 四次**。这些 27 道实例是展示性重复采样，不改变上面的 74 个有效 unique question item 计数。

| 目录代号 | 实际问答 |
|---|---|
| QA-03 | 谁先发声？→ 蓝衣人物 |
| QA-12 / QA-02 | 蓝衣人说了什么？→ Ask her to bring these things with her from the store.；说出此句的人上衣颜色？→ 蓝色 |
| QA-12 / QA-02 | 粉衣人说了什么？→ She can scoop these things into three red bags, and we will go meet her Wednesday at the train station.；此句对应衣色？→ 粉色 |
| QA-12 / QA-02 | 绿衣人说了什么？→ A new school will be built.；此句对应衣色？→ 绿色 |
| QA-12 / QA-02 | 白衣人说了什么？→ Please call Stella.；此句对应衣色？→ 白色 |

真实反向题通过对应唯一台词/声音绑定目标，表中的“此句”只是展示缩写。A/B/C 的详细题目位于本批四套 authored delivery 目录，N 的 18 个有效题位于上面的 native Apartment delivery 目录；每个结果都包含 `catalog_id`。L 的 QA-13 是本批错时采样样本，Open 数值角度与 MCQ sector 各自按当前 form 判分。

九道实例不是九种新题型。当前四个人、台词和时序仍反复使用，尚未构成充分的跨身份、内容和生活活动多样性评测。

**旧编号对照表，仅用于查代码。**

| 统一编号 | 原实现 / 复用关系 | 旧题卡 |
|---|---|---|
| QA-01 | QS-001 | 卡7是时刻扩展，另见QA-18；卡11另见QA-20 |
| QA-02 | QS-002 | 卡14；卡9的外观绑定步骤 |
| QA-03 | QS-003（衣色答案还用QS-002） | 卡9 |
| QA-04 | QS-004 | 卡3；卡2为扇区/数值角度变体 |
| QA-05 | QS-005 | 旧卡目录未单列，保留音频对照 |
| QA-06 | QS-006 | 卡6原形态、卡10合并 |
| QA-07 | QS-007 | 旧卡目录未单列，保留视觉对照 |
| QA-08 | QS-008 | 卡11使用其可见性事实；卡16另见QA-24 |
| QA-09 | QS-009 | 旧卡目录未单列，保留视觉对照 |
| QA-10 | QS-010 | 卡16可进一步组合此项 |
| QA-11 | QS-011 | 旧卡目录未单列，保留视觉对照 |
| QA-12 | QS-012 | 卡13 |
| QA-13 | 无直接稳定QS；可复用事件和位置事实 | 卡1 |
| QA-14 | 无直接稳定QS | 卡4R |
| QA-15 | 无直接稳定QS | 卡5原形态 |
| QA-16 | 无直接稳定QS | 卡5R |
| QA-17 | QA-06/QS-006的不同时间条件，未直接实现 | 卡6R |
| QA-18 | QS-001的时刻锚定扩展 | 卡7 |
| QA-19 | 无直接稳定QS | 卡8 |
| QA-20 | QS-001与QS-008事实组合 | 卡11 |
| QA-21 | QS-002反向思路；不是QS-012台词的现成功能 | 卡12 |
| QA-22 | 无直接稳定QS | 卡15a |
| QA-23 | 无直接稳定QS | 卡15b |
| QA-24 | QS-003+QS-008；可选进一步用QS-010 | 卡16 |

旧卡17跨段记忆，以及新提出的细粒度感知/个性化/时空记忆建议，单独保留在[下一篇论文ideas](../roadmap/NEXT_PAPER_PERSONALIZED_AV_MEMORY_IDEAS_20260905.md)，不占本篇QA-01至QA-24的实施范围。该ideas文档明确标注眼镜、手环、衣纹需要未来资产支持。

源码依据：src/avengine/qa/unified_catalog.py（QA-01至QA-24、CATALOG_VERSION=20260906）；src/avengine/qa/unified_scoring.py（对应 Open/MCQ 判分及 QA-13 `right_positive` 约定）；src/avengine/qa/question_spec.py（历史12类型）；src/avengine/qa/pixel_visibility.py（四种可见状态与画外比例null）；tools/qa/adapt_four_speaker_research_qa.py（四人三类/九实例）。历史[源码审计](../roadmap/MULTI_HOME_QA_CAPABILITY_AUDIT_20260905.md)和[题卡设计](../roadmap/QA_TYPE_DESIGN_V3_20260830.md)保留作为来源，本文件是统一编号与阅读入口。没有新增QuestionSpec注册、hash锁、冻结contract或准入gate。
