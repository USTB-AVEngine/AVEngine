# P3 七份房间包与原生地板参照

## 1. 改动与提交

以 8ac2ea1 为本次父复核基线；报告和以下实现同次提交。
examples/rooms/packages/ 保存 native Apartment、A/B/C、Kujiale、MP3D、HM3D
七份包及 catalog、Kujiale pose bindings。新增两个 tools/rooms 地板测量
入口与专属测试。room_package.py 与控制器支持 catalog.path_bindings 和
runtime.path_bindings 覆盖环境根；14 个当前机器路径根保存在 catalog 的
明确配置中，代码不写死私有数据根。

## 2. 验证

项目 Python + PYTHONPATH=src:tmp/native_python_addons_v1：P3/P1 相关测试
30 passed / 0 failed / 0 skipped（0.73 秒）；工具索引 1 passed。
父代理逐包展开配置，检查 floor artifact、实际 vertices/triangles 路径、
声学 manifest schema 和分派：7/7 通过；所有声学包均为
avengine_acoustic_scene_package_v1。

机器核对文件：tmp/p3_parent_package_check_20260907_v1.json。
原生 UE 查询只操作本任务新建进程/输出，既有 Studio 与其他 GPU 任务未动。

## 3. 实际产物与检查

| 包文件（examples/rooms/packages 下） | 原生路径 | 实测地板 y（米） |
|---|---|---:|
| native_apartment.json | UE /Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000 | 0.2711074501 |
| room_a.json | UE /Game/AVEngine/MultiHome/room_a_living_props_v7 | 0.0001953125 |
| room_b.json | UE /Game/AVEngine/MultiHome/room_b_detailed_v8_linear_materialfix | 0.0001953125 |
| room_c.json | UE /Game/AVEngine/MultiHome/room_c_living_props_v7 | 0.0001953125 |
| kujiale_0020_full_home_v1.json | UE /Game/AVEngine/Optional/Kujiale/kujiale_0020_full_home_v1 | 0.0001953125 |
| mp3d_17DRP5sb8fy.json | Habitat 原生 17DRP5sb8fy | 0.0724470019 |
| hm3d_00800_TEEsavR23oF.json | Habitat 原生 00800-TEEsavR23oF | 0.1633791029 |

A/B/C/Kujiale 的当前地图已逐一加载，在缺少 line-trace collision hits 时
使用同一已加载关卡的原生 metric-depth 向下测量。四份新输出：
tmp/p3_room_packages_20260907_v6/ue_floor_measurements/{room_a,room_b,room_c,kujiale}/。
这些替换了初版离线几何值和旧 BakedLit 地图引用；逐点深度、地图 level
读回与家具顶部离群值均保留，未把无碰撞数据写成 collision trace 通过。

Native Apartment 使用同一实际地图的既有 UE depth floor reference
（tmp/qa_v3_floor_reference_apartment_20260903_v2/floor_reference.json），
与原生 room layout 地板差小于 1e-6 米。此次重测进入 UE Zen/asset-idle
长循环，该失败不计为新证据。

MP3D 在安装的 Habitat PathFinder 上查询/snap 128 点，128/128 可行：
tmp/p3_room_packages_20260907_v3/habitat_floor_measurement/mp3d_floor_reference.json。
HM3D 原生查询 256 点，256/256 可行：
tmp/p3_room_packages_20260907_v5/hm3d_floor_measurement/floor_reference.json。
另一个观测楼层为 3.1634 米，楼梯中间高度保留；包按两个楼层列 subrooms。
测值是原生 navmesh 地板参考，不声称已测动态碰撞支撑。

Kujiale 当前地图的独立 RGB/depth 单帧 canary：
tmp/p3_room_packages_20260907_v11/。
使用既有 clearance table 的精确候选 `[-112.5,-227.5] cm`、相机高度
1.47 m、yaw 150°（该点 180/180 个表内 yaw 通过），同一 full-home map
实际 level 正确且 AUsdStageActor=1。fresh 原生 RGB 为 320×240、uint8、
范围 4–255、均值 BGR=[134.64,167.61,191.23]；depth 为 float16、
320×240、finite/positive=1、范围 1.197–9.875 m，object-ID pass 同帧
保留。实际查看 `rgb_preview.png` 可辨认客厅的沙发、餐桌、椅子、窗帘和
天花板；初版 v2 的近黑 RGB 不计入证据。临时进程仅使用 5 个 review lights
（强度 750/2000/2500/1800/1600 lm，位置和 UE readback 写入
`evidence.json`）及自动曝光关闭/Lumen quality 控制，未修改地图资产。

## 4. 缺口与失败边界

房间包与实测地板已交付。HM3D 包驱动的 5 帧多实体捕获、四家族完整的
双声源音频/QA 收口属于 P5/P6/P9 集成验证，当前不据包校验提前判定通过。
所有包保持 research_candidate、qualification_claim=false。

MP3D 使用服务器已有的 /data/avengine_external/datasets/mp3d_example_scene_1.1；
任务书的数据根不包含该场景，未创建挂载或别名。HM3D 使用非 Basis GLB
和派生的 nonbasis dataset config；语义源的 Z-up 与声学数组的 Y-up 分开
声明，不重复转换已在共用坐标的声学网格。

初次 Kujiale canary 缺 avengine_spear_ext 路径，修复显式扩展路径后新输出
通过；Native Apartment 重测失败目录与日志保留。失败没有覆盖历史数据。

## 5. 共用与 Claude 接口

catalog 按 RoomPackage.renderer 分派。路径字段可用 AVENGINE_* 模板，
实际配置在 catalog.path_bindings，请求 runtime.path_bindings 优先。
显式缺少路径根会报出变量名。

Kujiale pose sidecar 复用 A 的精确导入资产/动画/emitter 绑定，删除房间特有
seat_affordance_id；它声明 none_declared_for_this_scene，不捏造椅子、坐标或
坐姿。采样器使用 walkable_grid，执行器使用当前 full-home 地图。

## 6. Owner 决策

无需新增授权。当前包没有正式数据集准入声明；更大批量仍遵守原授权边界。
后续原生整链验证须保留实际数据根、楼层和渲染器来源。

父复核补充：v11 的 sofa、餐桌、椅子、窗帘和天花板清晰可辨；5 个已核对
灯光的 renderer-only 配置已存入包 planning_inputs.review_lights，供 UE
执行器复用。预览核对是 Codex 助手的视觉检查，不是真人准入。原回执的
human_inspection 字样已在 parent_visual_verification.json 明确纠正；不据此
制造任何人审记录。
