# Hunyuan3D 2.1 静态家养鸟交付（2026-10-12）

本轮生成两个带独立木架、可落地的静态资产：
`v7_pet_canary_yellow_perch_hy3d_01` 和
`v7_pet_budgerigar_perch_hy3d_01`。既有 Chris Woods CC-BY 鹦鹉、麻雀、
原登记文件和源数据保持只读。本轮仅写新的私有登记文件，没有改生产总登记或准入录音。

任务产物根目录为 owner 指定的
`tmp/claude_hardbank_20260923/v7_pet_birds_hy3d_20261012/`；
完整结果见该目录的 `REPORT_zh.md`、
`evidence/static_delivery_summary.json` 和
`registrations/v7_asset_roster_birds_hy3d_20261012.json`。

| 资产目录 | 体长近似值 | XYZ 外包围尺寸（m） | 三角面数 |
|---|---:|---|---:|
| `assets/canary/` | 0.13 m | 0.120 × 0.376595 × 0.127704 | 32,817 |
| `assets/budgerigar/` | 0.18 m | 0.160 × 0.399499 × 0.188811 | 35,867 |

坐标为米、+Y 朝上、-Z 正面，原点为圆底座落地中心。发声点记录于各自
`asset.json` 的 `emit_point_local_m`，为人工复核的静态喙部表面点。
木架不发声，不准入行走、飞行或动态动作。

## 生成与材质

新增工具位于 `tools/assets/v7_pet_birds_hy3d_*.py`。CLI 参数提供任务根目录、
原任务参考根目录、模型和本地推理源码目录；产物不进 Git。

1. `reference`：FLUX.2-klein-4B 固定种子候选；`segment`：CPU ISNet 和边界检查。
2. `generate`：本地 Hunyuan3D 2.1 形状和 PBR；`pbr`：UV/颜色与粗糙度、金属度通道导出。
3. `preview`、`landmarks`：孤立资产 CPU 四面预览和原始网格的射线标注。
4. `normalize`：根据喙至尾端近似体长均匀缩放，保留生成的鸟、双脚、横杆和 UV/颜色；
   仅移除经过审核的旧木底座/竖杆范围，重新建立直木杆和圆底座。
5. `habitat` 与 `habitat_geometry`：真实 MP3D/HM3D 图像、发声子节点、
   贴地和两种静态碰撞检查；`ue_geometry`、`ue`、`ue_qa`：
   源场景三角形检查、一个蓝图 Actor 的静态导入与原生截图核验。

基线分支缺少生产材质模块，因此从已落地的材质检出
`486799f3d57caaae44ab692bf8cef41a396defb6` 原样引入
`src/avengine/assets/source_materials.py`、配套测试和 matrix 检查器。
生成 GLB 经 `prepare_source_glb` 处理；Habitat 用
`configure_source_pbr` 在场景加载后配置插入源 PBR，关闭 IBL，metallic=0，
粗糙度下限 0.85，动物高光代理上限 2%。没有手工替代这条材质路径。

Hunyuan 本地推理缺失依赖、扩展和 Blender 导入问题均在任务的
`runtime/`、`cache/` 下处理；没有安装到共享 conda 环境、改权重或改原推理检出。
本地适配的上游提交、修改说明和补丁保存在
`evidence/runtime_adaptations.json`、`runtime_hunyuan_changes.patch`。

## 原生验收与验证边界

每只鸟在 MP3D、HM3D 各有四张 640×640 的未标注原生 JPEG，UE 有两张场景 JPEG
及两张 show-only JPEG；最终四面预览每面 1024×1024。带标签拼图是派生审阅图，
不替代原生截图。素材及几何数字均在外部交付根目录的 JSON 中，不靠单元测试宣称原生通过。

Habitat 默认 Kinematic 组排除 Static 碰撞，默认 4 cm 边距也会使小资产接触检查失真。
本轮只对创建的鸟对象覆盖 Dynamic 碰撞组，保持 Kinematic motion_type，
并把其模板边距设为 1 mm；不改全局组掩码或共享场景。验收包含故意下移 3 cm 的阳性对照、
抬高 6 cm 的阴性检查，以及完整资产与原始场景 GLB 三角形的 FCL 检查。
三角形坐标以原场景 up/front 属性旋转，五点地面射线和原生读回对齐。
地面本身是允许接触的支撑，非地面三角形不得相交。

UE 首轮房间网格晚于截图才构建完成，缺失背景的证据已保留并判为不合格。
复拍等待房间加载，QA 额外要求场景图含足够的真实房间像素，阻止仅鸟的黑背景图通过。
静态导入、bounds、发声点及截图均通过，编辑器在保存完后退出崩溃（139）。
此故障单独记录；未声称干净退出或原生 UE 物理碰撞通过。UE 碰撞依据真实源 USD 三角形 BVH，
不是动态图像或完整场景物理准入。

## 许可与测试

FLUX.2-klein-4B 模型采用 Apache-2.0；Hunyuan3D 2.1 采用腾讯社区许可，
地域排除欧盟、英国、韩国，且第 5(b) 条禁止用 Works、Outputs/results 改进其他 AI 模型。
静态技术验收不赋予其他模型训练权限；新登记显式保留
`downstream_ai_training_admission=not_granted`。完整许可原文、条款位置和 Notice
见任务目录 `licenses/`。没有对外发布或 push。

新增 UE 图像 QA 单元测试覆盖“缺少房间背景仍然低高光”的实际失败、有效场景和过曝案例。
材质回归及全量 `tests/unit` 实际执行记录见任务 `logs/`。
运行命令、PID、终态和 GPU3 采样保存在 `JOBS.json`；GPU 统计区分 GPU 作业墙钟与
采样到的 GPU 上下文时间，不将 CPU 四面预览算作 GPU 时间。
