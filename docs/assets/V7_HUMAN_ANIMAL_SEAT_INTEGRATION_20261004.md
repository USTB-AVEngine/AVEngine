本包登记入口是 `registrations/v7_asset_roster.json`；人物兼容入口 `human_roster.json` 已更新为 v2。全部路径相对于 `artifact_root`，48g 上的包根为 `/data/jzy/tmp/claude_hardbank_20260923/v7_human_roster_20261004`，以下记作 T。本任务没有修改 v7-spec / v7-families 检出或公共登记库；这里提供接入资料与可复用实现。

| 文件 | 用途 |
|---|---|
| registrations/v7_asset_roster.json / .csv | 人物、狗、鸟的大类/小类、运动能力、两端路径与声源 |
| registrations/human_roster.json | 人物 v2；40大类/120小类，底模 lineage 和颜色 |
| registrations/chair_roster.csv；round3/registrations/chair_roster_final.json | 五种无声座椅，两端资源、几何支撑点与材质 |
| registrations/strict_profiles/index.json；instance_requests/ | 六色按每属性≤3值拆成两个profile，共用同一底模；32profile/96请求 |
| registrations/lx_dry_audio_mapping.json；round3/registrations/lx_sound_anchors.json | 10个LX干声类别，主声源是嘴、乐器或手掌；不包含声音文件 |
| round3/registrations/foot_contact_events.json | 599新接触+90初始支撑；尚未做音轨同步 |
| round2/assets/humans/<id>/final_ue_root.glb | 每人唯一完整动画master，15条动作；两端动作几何共同来源 |
| assets/runtime/<id>/<colour>/habitat.glb | 各小类已验收静态刚性GLB |
| ue_project/Content/ | UE5.5.4私有uasset；迁移时带完整依赖 |
| reports/review.html | 单HTML、所有图内嵌JPEG；未发owner |

人物采样先选不同 `major_id`，再选同一人的颜色小类。衣服颜色是实例属性，不应把一个底模的多色当成不同长相。所有题族只在 `walk_allowed=true` 时将人/狗作为移动声源；LX 和鸟永远静态。椅子与鸟架永远不发声。家养鹦鹉候选的双端状态为false，不进入采样池；麻雀只作为明确允许的兜底。

Habitat使用MP3D/HM3D原生场景；酷家乐使用UE/SPEAR原生地图。本包检查的HM3D为TEEsavR23oF、MP3D为17DRP5sb8fy，UE地图为 `/Game/AVEngine/Optional/KujialeBakedLit/kujiale_0020_full_home_baked_lit_v1`。这些是参考验收场景，并非任意场景的通用通过结论。不要用Blender自造房间或MP3D的UE比较导入替代准入。

单位和朝向：

- 静态Habitat GLB是米、+Y上、-Z前。全体实例尺度沿用底模，地面取真实物理表面；不能把navmesh高度直接当脚底地面。
- 坐标转换 `U(x,y,z)=100*(-z,x,y)`，得到UE世界厘米坐标。静态人物、狗、椅子与鸟的Actor yaw是 `90-degrees(canonical_yaw)`。
- 人物动画master保持库的原始+Z前；导出刚性Habitat姿态时先绕Y旋转180°，UE人物动画Actor yaw是 `-90-degrees(canonical_yaw)`。不可与静态Actor的90°规则混用。
- FOA点要用正在播放的姿态和实例Actor变换；不要把世界坐标转换后的U(point)直接当UE Actor局部坐标。对于UE原始glTF导入局部坐标，位置映射是 `100*(x,z,y)`，然后应用Actor Transform。或者先在规范坐标下求世界点，再用U转换到UE世界。
- Rocketbox人体主声源骨为MUpperLip，狗为Xtra Mouth；用AnimPose/get_socket_location或同帧姿态计算，并逐帧跟随。LX的静态FOA_mouth是次级嘴部点；演奏/鼓掌题必须读 `sound_anchor.primary_local_glb_m`，在乐器或手掌建立主声源，不能默认用旧嘴部组件。鸟使用登记的鸟嘴点。

人物行走的可用动画为 `Walking_Contact_mp3d`、`Walking_Contact_hm3d`、`Walking_Contact_kujiale`。它们是按具体路线、地面采样与速度重定时后的脚部IK动作，不是通用“播放后任意移动”的裸Walking。Habitat消费 `v7_human_roster_pose.py` 的姿态与 `v7_roster_contact_ik.py` 的FootPlant；UE读取 `/Game/V7RosterMotionV3/<id>` 的SkeletalMesh及相应AnimSequence。颜色沿用旧静态导入材质实例，根据 `evidence/ue_skeletal_instance_materials.json` 的槽名称/索引覆盖，新骨骼网格的槽名已与原槽逐项核对。

新场景先用导航路径生成平滑yaw/速度（人体1.0–1.5m/s，本包实测1.05–约1.20），按30Hz采样真实地面；重新编译脚部支撑、摆动脚弧线与根轨迹。墙/家具净空采用完整姿态三角形与场景几何碰撞，不能只检脚点或导航中心线。嘴部的同帧世界位置写入声源时间线。主工具为 `v7_roster_navigation.py`、`v7_roster_walk_compile.py`、`v7_roster_habitat_motion.py`、`v7_roster_scene_collision.py`。本包原3动作仍保留，但裸Walking/长Standing_Idle没有获得任意场景接触准入；站立待机使用已验收静态首帧。原长待机循环不能直接当完整通过。

坐着讲话采用 `Sitting_Talk_Contact`，保留已接触坐姿的下半身和脊柱，只移植官方坐姿手势的手臂与颈部链；站着讲话为 `Standing_Talk_Contact`，转身为 `Turning_Left90_Contact`。各自完整30Hz几何/UE骨骼与四关键帧原生Habitat检查已单独登记。坐着讲话完整动态配对目前限定凳子；其他类型只有静态坐姿兼容证明。不能用未经检查的扶手椅搭配大幅手势。

坐姿采样必须先选座位：

1. 在HM3D/MP3D通过语义对象分类找chair、sofa、bench、stool等；在酷家乐用家具标签中的椅、沙发、长凳、凳子等类型和家具实例ID。语义包围盒仅用来找候选，不能直接将其最高点作为座面。
2. 用原生RGBD/深度表面、原几何射线和水平法线估计座面位置、高度、宽深与朝向。实例标签朝向可靠时优先使用；家具看向桌子/电视的标签或原生表面中心是辅助目标。本任务对MP3D/HM3D有11/7个原生候选；当前酷家乐导入为完整扁平网格，未取得可用分件座位标签，所以参考UE验证采用补放椅子。生成器若有家具标签应优先用，不要把未核实标签写成已验收。
3. 选择人物 `seat_types` 中兼容的类型；用 `chair_body_compatibility.json` 的人体偏移、实例座高缩放和真实接触点配对。可选择椅面实际高度并做坐姿IK，但不要把人物整体上下移动来假装脚也贴地。
4. 用靠近导航可走区域的真实地面射线确定脚底高度；脚底射线命中的点要与附近导航支撑的物理地面在15mm以内，拒绝沙发垫/椅面。参照 `StageFloor` 和 `require_physical_ground`；需要忽略当前人物和新增椅子自己的碰撞。
5. 臀部衣服表面在座面投影范围内与座面间隙≤12mm；两脚/地面≤12mm；人体与座椅、墙/家具检查完整三角形。意图接触和穿透要区分，不能把人体抬高4cm的粗检查当完整臀部接触证明。本包静态椅子组合采用13轴三角形/盒SAT、4mm表面容差；隐藏解剖臀部没有独立分割，登记保留此资格。
6. 没有合适已有座位时，从五款哑光椅子选与室内风格一致的一款，摆到人物同向、座面支撑正确的位置。通过四脚/五轮或底座的真实地面接触检查，不能穿墙、穿家具、穿人体。已有人/椅的保护主路线中心线净空至少是人物半径+椅子支撑范围+余量；保留所需连接点之间的可走路径。
7. 在本episode的私有PathFinder上重算导航（`NavMeshSettings.include_static_objects=True`），比较所有采样器需要连接的路径/岛ID；新增椅子若堵住主通道或导致必要路径不连通就拒绝该位置，再找下一候选。原始场景navmesh只读。本包验证了保护路线重建后的水平误差，尚未遍历全屋所有目标的连通性，因此生成器必须完成这一条。
8. 将人物、座椅、朝向、接触点、脚点、声源点及所用语义座位ID写入episode manifest。反事实只改上衣颜色时固定这组布局和动作轨迹；不重新选座位。

坐姿实现与参考证据：`v7_roster_seat_rgbd.py`、`v7_roster_native_seating.py`、`v7_roster_chairs.py`；`round3/evidence/seated_native/<house>/<id>.json` 有具体既有座位ID/类型、面向目标、根偏移、脚底物理地面参考、座面间隙及保护路线误差。不能单独加载历史 `sitting.glb` 后留下空椅位。

LX只用来源固定姿势。钢琴人物含原琴凳；手鼓人物含原折叠椅，native侧面复核记录在 `lx_built_in_seats_review.json`。不把二者当作能随意换座位的坐姿骨骼人物。其他LX动作保持原静态站姿；适合中远景，近景拓扑/解剖、嘴唇运动与行走未准入。对应干声由families确定真实录音/许可；本包仅给10类对应表及主声源锚点。脚步声启用时用接触事件表对齐音轨，当前未验证音轨同步。

狗沿用原Rocketbox目标网格、骨骼及权重，Quaternius仅作为兼容动作供体。静态GLB在 `round3/assets/dogs/<breed>/idle.glb`，动作master在 `round2/assets/dogs/<breed>/motion_runtime.glb`，UE在 `/Game/V7RosterAnimalMotion/<breed>`。动作family至少Idle/Walking，四脚接触与Xtra Mouth声源随动均有完整回放。比格速度0.3m/s，德牧0.8m/s；德牧在酷家乐使用缩短路线，不能误用旧末段有墙碰撞的路线。

鸟兜底是 `round3/assets/bird_sparrow_fallback/runtime.glb`，鸟与无声鸟架一起建模，鸟高0.18m、总高0.45m，鸟嘴局部坐标写在登记，UE包 `/Game/V7RosterBirdsV3/v7_rocketbox_bird_sparrow_fallback_01`。它是MIT麻雀，不能标为家养鹦鹉。虎皮鹦鹉仅2D候选：项目约定在owner确认品种/解剖前禁止送入Pixal3D，本轮无人可确认；还不能用于Habitat/UE。

UE迁移使用依赖迁移功能或按登记的依赖闭包复制T私有uasset，包括纹理、材质实例、Skeleton、AnimSequence、Blueprint，保持/Game路径或更新登记映射。没有运行目标项目cook，不能只复制网格而漏依赖。UE库适用5.5.4；原生全屋碰撞射线在当前扁平导入上不可用，所以本包用实际源USD三角BVH做几何碰撞，并另验native地图bounds；不宣称是UE物理系统通过。

历史中间件已按磁盘纪律回收：`assets/runtime/*/*/runtime.glb`、`assets/rocketbox/*/runtime.blend`、`assets/ue_joint_root/*/runtime.glb` 和旧动作抽样临时GLB。消费以上保留的habitat.glb、唯一动画master、uasset及新登记；旧证据中的历史源文件名仅表示当时输入，并非现在可加载的交付入口。

STARSS23画像来自72段非评测录音，解析帧中坐58.76%、站39.00%、走1.253%；有60.57%姿态未知，不能把已解析比例当真实总体。采样器读取原 `real_scene_profile_20261004/v7_recommended_config.json` 的已解析/未知分支，结合本包运动准入条件；未知分支独立随机化，不借评测数据安排人物。

所有新场景均要复验；本包没有替v7生成器跑0–8人同场、多人互撞、足音同步、全屋座椅连通性或数据集episode准入。许可副本和第三方致谢必须随迁移交付；ISNet独立ONNX权重/输出许可范围仍待确认。

Habitat动态换色：调用 `v7_roster_colour_motion.apply_appearance(master_d,master_blob,variant_habitat_glb)`，按材质名重映射并将选中小类的完整PBR材质/纹理并入master内存副本，然后执行pose/rigid_glb。不要直接用原色动画master渲染所有颜色，也不要把刚性站姿当皮肤动画。这个函数不保存重复动画master；骨骼、动画和所有几何accessor维持原buffer位置。110小类的真实文件逐项验证见 round3/evidence/motion_colour_binding.json。
