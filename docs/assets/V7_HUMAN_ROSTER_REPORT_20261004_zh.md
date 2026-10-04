人物40个大类、120个小类在Habitat/UE静态可用；30个Rocketbox底模（110个小类）在指定路线可走、可坐，家养鹦鹉/金丝雀仍未完成。

任务产物根目录为48g `/data/jzy/tmp/claude_hardbank_20260923/v7_human_roster_20261004`。仅此任务的独立工作树和输出目录被修改；未修改Rocketbox原库、LX归档、v7-spec/families或共享环境。所有数字及实际机器命令以运行目录证据为准。

|交付|数量/入口|
|---|---|
|便装换色|16底模（女8/男8）×红绿蓝黄白黑=96小类|
|独立职业|14（女7/男7），原制服|
|LX静态人物|10，钢琴/手鼓人物含原始座椅；不加走路|
|狗|比格/德牧2种，可走可站|
|鸟兜底|MIT麻雀+无声鸟架1；不能计入家养鸟|
|补放无声座椅|餐椅/办公椅/扶手椅/凳子/沙发椅5款|

版本化小型引用清单见 [V7_HUMAN_ROSTER_REFERENCE_20261004.json](V7_HUMAN_ROSTER_REFERENCE_20261004.json)。完整登记是运行根目录 `registrations/v7_asset_roster.json`，人工清单、座椅清单、换色/动作验收、许可、接入说明及磁盘记录位于 `reports/`。单文件 `reports/review.html` 含1,715张内嵌JPEG，未发owner。所有媒体、模型权重、GLB和uasset留在Git外。

人物15条动作/底模从Rocketbox官方动作派生：行走按具体原生导航路径重定时、接触IK，嘴部逐帧随MUpperLip；坐着讲话只取官方手臂/颈链，保持下半身接触姿态。站立原长待机序列保留但未完整接触准入，静态首姿可用。指定场景已有座位优先，附近无适合座位才补椅，接入算法见 [接入说明](V7_HUMAN_ANIMAL_SEAT_INTEGRATION_20261004.md)。

颜色192次原生Habitat测量/480对，最小CIE76 ΔE=40.198260，标称颜色0失败（阈值20）；遮罩外/保护区改动0像素。UE不使用贴图色差。人完整原生Habitat行走60/60、配座位60/60、额外动作抽样180/180；UE实际Actor截图258/258。110个可动小类完整PBR外观并入master内存副本：同姿态几何差0m、纹理字节一致，不保存110个重复master。结果的准入范围为指定参考路线/摆放，不能推断新房间或拥挤场景自动通过。

完整fast suite3160 passed、117 skipped、2 deselected；最后定向18 passed。对应运行目录 `round3/logs/fast_suite_final.log` 与 `final_targeted_tests.log`。交付依赖与自有进程检查在 `evidence/final_delivery_checks.json`，阶段记录为 `PROGRESS_zh.md`。

FLUX实际使用klein4B Apache2；Pixal3D MIT+NOTICE；实际DINOv3有独立Meta许可及论文致谢要求；ISNet独立ONNX权重/输出许可范围仍未确认，详见运行目录 `reports/LICENSES_zh.md`。生成家养鸟必须在2D候选图审核后才重建，仓库 [生成动物约定](GENERATED_ANIMAL_ASSET_AND_INSTANCE_CONTRACT.md) 明确“Hard stop for project-owner review.”；无人能确认，本次保留参考图/分割，未自批送入Pixal3D。

资源异常已如实记录：一次自有UE重试与旧进程重叠，精确GPU重叠时长未验证；已停止核对后的自有进程并新增任务内flock互斥锁。共享/data曾低于50GiB，守护停写并只清理自有已验收中间件。最终磁盘数值、文件级清理清单和可再回收项见运行目录 `reports/DISK_zh.md`，不声称共享磁盘全程由本任务保证。UE部分在保存结果后退出double-free/139，不能记为无错误rc0。

本任务未准入干声音频/脚步录音同步、未运行0–8人正式生成器场景，未做全屋UE物理射线验收；酷家乐真实USD三角FCL加原生姿态/边界检查是几何代理。隐藏解剖臀部未单独分割，座面范围最低衣服表面为接触代理。
