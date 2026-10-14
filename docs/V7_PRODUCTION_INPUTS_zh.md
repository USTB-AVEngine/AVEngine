量产输入接线与边界（2026-10-14）

tools/timeline/sample_v7_production_inputs.py 读取冻结的三类房间清单、逐房计划、实际资产、v13 声音和 24 音色语义清单，生成 v7-production-candidate-inputs-v1。CLI 只准备选择，不能当作原生渲染器的完整 recipe 直接启动。原家族出题器还需生成 listener、原生摆放、事件计划，并逐父场景通过全部原生、声音、真值和题目门禁。

默认逐房种子轮换多人站姿、走路、合法座椅上的坐姿、动物动画/明确滑行、声音/衣服/乐手变体。动作只在 HM3D/MP3D 私有 phase2 路径接入；酷家乐注册出题路径只支持此次静态配方，收到 phase2 motion_requests 时明确拒绝，不能静默忽略。座椅要求原 100px 门禁，坐姿只建议完整登记 VCTK 录音；语义 TTS 保持站姿专用。五种非红上衣不重复，测试房子及 budget_limited 格子不采样。

两个原生生成 CLI 可用 --production-contract 或 runtime.production_input_contract 强制执行清单校验。登记与源素材保持原字节，运行路径视图不改登记权威。63 只动物的“动作准入”仅证明原保存动作的 authority，不证明它们在新房间路线或与其他人物组合已通过；每条新路线仍检查原完整皮肤、脚掌、导航、碰撞等门禁。新 motion 原生 readback 明确标记 animated_walk / intentional_slide_no_animation。

phase2 的 28 题私有集成验收与既有 formal alpha/profile 不同；没有在本轮把 expanded mixed v7 升级为正式训练发布。30 万题量产计划须读取本次隔离试产实际导出的题量、封存交付大小和完整成功调用用时，旧63题/69 MB估计不再沿用。每房及每套房的父场分配见产物PRODUCTION_PLAN_v7_20261014.json；最多6800父场不能满足题量时只报告差额，不扩范围、不改出题配额。tools/timeline/export_v7_trial.py只允许本次30个请求槽位、在pilot_20261014/export写trial_only行，沿用原可读答案转换及判分器，用现有open_delivered_media读回视频与FOA。不并入任何训练集。此分支不授权push、拉取CCI或启动CCI/ACP。

三份房间来源已按最新可用权威冻结（HM3D v2、MP3D final=true v2、酷家乐类别清单）；合并运动准入实数为狗30动画/7滑行、猫16动画/10滑行，总46/17。两条新骨骼分支不改变原17只的选中模式，不能只凭有骨骼把滑行提升为动画。动物初始姿态脚点从原保存native首帧反变换得到，原资产和旧owner093x_final保持只读。台词双角色候选指定原试产0度/31.5度方位目标，仍由原QA-28门禁决定实际是否可出题。v13电话旧名在声音输入与环境实体装配两处规范化，原录音域和物理绑定相等检查保持。

最终默认候选按已有严格试产采用最多3个human（可另有独立乐手），而非自动叠加第4个human和空调。空调仍是只摆不发声的登记资产，默认不自动抽入；仅--silent-decor-count 1显式候选，照常过原墙面和100像素材质等门禁。画面像素不足的材质not_run不等于缺少原登记；本轮未借此降低材质或可见性门槛。
