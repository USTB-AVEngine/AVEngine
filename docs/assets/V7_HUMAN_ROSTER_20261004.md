# v7 人物资产交付，2026-10-04

本任务交付 40 个人物大类、120 个小类的原生 Habitat/UE 静态资产：16 个男女均衡 Rocketbox Adults 各六色（96 小类）、14 个原色职业身份、10 个 lx 静态动作人物。生成资产、登记表和原生证据位于任务书指定的外部输出目录，未写入全局资产库或 v7 生成器检出。

输出目录是 48g 的 /data/jzy/tmp/claude_hardbank_20260923/v7_human_roster_20261004/。
权威入口为 registrations/human_roster.json；中文报告 reports/REPORT_zh.md；逐条清单 reports/INVENTORY_zh.md；接入说明 reports/INTEGRATION_zh.md；单文件内嵌截图 reports/review.html；许可 reports/LICENSES_zh.md。

16 个底模的上衣主色按遮罩替换，保护皮肤、条纹/印花、其他贴图、PBR 细节和 geometry/UV/skin/action。严格 schema 的三值上限通过每底模两个 profile 解决，仍是一种身份。Habitat 两房 192 次颜色测量和 480 组比较，最小 CIE76 ΔE 40.198260，阈值 20；UE 没有采用颜色距离准入。

Habitat 原生站姿 240 条、动作抽样 240 条全部 pass；UE 原生酷家乐静态 120 条全部 pass。120 个原生 Blueprint 将模型部件组合成一个人物 Actor，包含 FOA_mouth。LX 仅静态；坐姿需要匹配座椅。UE 新 /Game/V7SkeletonsRoot 保留非蒙皮动画父节点为第 81 个无权重 joint，270 个原生动作锚点采样对源 glTF 最大误差 0.040718655 厘米，阈值 0.3 厘米。旧 /Game/V7Skeletons 未通过该检查，不作为动作准入。UE 行走仍要求 consumer 按 motion.json 做逐帧地面修正并从骨骼取动态 FOA；另有 120 帧原生 UE 骨骼动作画面全部 pass，动态嘴点最大源误差 0.501712 毫米。120 静态与 120 动作的原生深度支撑底面也通过 3.5 厘米量化/栅格容差复核。完整连续动作视频、cooked/SPEAR episode 和 0–8 人同场未验证。

普通静态 GLB 是 +Y 上/-Z 前、米；静态 UE 组件 yaw +90、厘米。源骨骼动画保持 +Y 上/+Z 前，UE skeletal 组件在独立 SceneRoot 下 yaw -90；直接用 SkeletalMeshActor 时根组件就是骨骼组件，组合 actor yaw=语义场景朝向 yaw-90，并在场景位置上加 Z 落地修正，不能把 root relative location 设成零 XY。不得混用这两个入口。登记中的 pass 指本轮原生静态证据，所有正式 episode_admission 仍是 not_run。

实际 FLUX.2-klein-4B 是 Apache-2.0，Pixal3D 是 MIT；实际 Pixal 特征依赖 DINOv3 自定义许可，不能只照其 NOTICE 的 dinov2 标签。ISNet 实际 isnet-general-use.onnx 已从 lx 213ff79 源运行 status 核实，其本机 SHA256 及 rembg 官方下载 MD5 相符；独立 ONNX 权重许可范围尚未找到明确条款。干声仅提供类别对应，具体音频准入由 v7-families 执行。

早期两个 UE Vulkan canary 曾误建物理 GPU0 context，监视器停止了核对过的自有进程；早期 UE 默认启动也自动用了任务外同用户共享缓存/设置路径。完整日志保留在输出目录，中文报告记录了 PID 与时间。最终使用设备 namespace 只暴露物理 GPU1，任务外 filesystem 只读，缓存与输出均在任务目录。没有删除他人缓存或停止他人作业。

本提交只包含 tools/assets/v7_human_roster_* 工具、必要算法回归、此说明和自动生成 Tool Index。二进制资产与生成媒体不放入 Git。完整 fast-unit 实际结果：3148 passed、117 skipped、2 deselected；原生日志与未运行维度均在外部报告中明确。
