量产输入接线与边界（2026-10-14）

tools/timeline/sample_v7_production_inputs.py 读取冻结的三类房间清单、逐房计划、实际资产、v13 声音和 24 音色语义清单，生成 v7-production-candidate-inputs-v1。CLI 只准备选择，不能当作原生渲染器的完整 recipe 直接启动。原家族出题器还需生成 listener、原生摆放、事件计划，并逐父场景通过全部原生、声音、真值和题目门禁。

默认逐房种子轮换多人站姿、走路、合法座椅上的坐姿、动物动画/明确滑行、声音/衣服/乐手变体。动作只在 HM3D/MP3D 私有 phase2 路径接入；酷家乐注册出题路径只支持此次静态配方，收到 phase2 motion_requests 时明确拒绝，不能静默忽略。座椅要求原 100px 门禁，坐姿只建议完整登记 VCTK 录音；语义 TTS 保持站姿专用。五种非红上衣不重复，测试房子及 budget_limited 格子不采样。

两个原生生成 CLI 可用 --production-contract 或 runtime.production_input_contract 强制执行清单校验。登记与源素材保持原字节，运行路径视图不改登记权威。63 只动物的“动作准入”仅证明原保存动作的 authority，不证明它们在新房间路线或与其他人物组合已通过；每条新路线仍检查原完整皮肤、脚掌、导航、碰撞等门禁。新 motion 原生 readback 明确标记 animated_walk / intentional_slide_no_animation。

phase2 的 28 题私有集成验收与既有 formal alpha/profile 不同；没有在本轮把 expanded mixed v7 升级为正式训练发布。30 万题计划按 owner 的历史 63 题/父场景估算，先用 100 个真正过门禁的父场景复测去重后题量。此分支准备动作不授权 push、导出训练表、拉取 CCI 或启动 CCI/ACP。
