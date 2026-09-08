# P11 全资产双绑定与共同UE舞台

## 1. 文件与提交关联

权威树48g-jump:/data/jzy/tmp/wt-multi-home-activity-integration，集成基线3d81f3d。
本报告随P11集成代码和59条source registry提交。新增三个JSON request驱动工具：
 tools/ue/import_extra_animals_editor.py、create_extra_animals_blueprints_editor.py、
 verify_extra_animals_editor.py。父更新 examples/runtime/source_asset_runtime_profiles.json、
 examples/rooms/packages/catalog.json，以及schema/runtime_profiles的研究元数据支持。
没有更改源GLB、旧UE Content/map、Studio或Claude审计器。

12个带任务常量/历史版本的probe脚本已原字节移至
 /data/jzy/tmp/p11_rigid_import_stage_20260907_v3/scripts/，
 relocation_manifest_v1.json记录旧→新路径，旧native evidence原件不改。

## 2. 验证与失败记录

三个editor工具py_compile通过；38刚体和4动物均完成实际导入、fresh reload、
SPEAR native load/readback。29个新增floor刚体还实际spawn/read/destroy，mesh、
root、emitter parent/world位置、scale、bounds均通过，位置/scale误差0。
最终14个相关测试文件共181 passed/0 failed/0 skipped（12.46s），日志为
 tmp/p11_parent_full_related_tests_20260907_v2.log；完整范围见P11主报告。

4动物首次Blueprint创建因parent_class配置失败；不完整的
 /Game/AVEngine/P11ExtraAnimalSourcesV1/Blueprints保留为ignored诊断，唯一消费
wrapper目录是BlueprintsV2。其余实际导入失败原文与修复日志保留在stage的
import/reload/probe输出及归档scripts，不以改回执方式隐藏失败。

## 3. 59条登记与实际目标验证

并集59条现在全部进入同一个runtime registry（40rigid/12animal/7human），
每条都有Habitat和SPEAR binding。原有记录的旧字段保持；新增38个SPEAR静态
路径、4个双renderer动物，以及Beagle可选bone emitter。四新增动物使用source
自己的animated.glb/Idle/Walking、P12 v9 package与native muzzle/forward参考；
不抄其他asset ID、尺寸或动作。未知body_build/life_stage明确unknown，只有
research记录可采用未知体型和已观察到的coat值；没有创建虚假的L9三水平生成域。

关键源证据目录 /data/jzy/tmp/p11_rigid_import_stage_20260907_v3/：

- import_manifest.json、reload_manifest.json、native_load_probe.json：38/38刚体；
  GLB→UE坐标[x,z,y]*100的最大bounds误差6.50e-6cm，29floor base误差4.51e-6cm。
- p11_floor_rigid_attach_probe_v1.json：29floor实际附着；8wall/1ceiling未假放地面。
- p11_extra_animals_ue_native_probe_v5.json、p11_extra_animals_ue_forward_alignment_probe_v1.json：
  4/4 SkeletalMesh/Blueprint/Idle/Walking与native joint通过，骨骼34/38/34/29。
- p11_extra_animals_ue_binding_delta_v2.json：真实UE路径和source-specific joint_proxy。
  解剖方向用实测body→muzzle水平轴，不把通用坐标forward_axis=-Z当动物朝向。

父登记与来源核对在tmp/p11_parent_external_animal_registry_20260907_v2/；
40rigid Habitat实际load/语义RGB与原点修复在P4报告，18个新增Habitat articulated
包的原生动作/mesh核对在P12及P12_external_animals报告。

共同目标stage（fresh）：
 /data/datasets/avengine_workspaces/multi_home_activity_20260905/root/qa_full_asset_ue_stage_20260907_v1/
，SpearSim/SpearSim.uproject。catalog的AVENGINE_MULTI_HOME_UE_ROOT已指向此目录。
复制自已验证的父Kuja task stage，补入新的P11RigidSources、4个动物和BlueprintsV2、
walnut speaker；补齐实际加载时发现缺少的9个旧articulated Blueprint/Mesh目录。
未保存或改写任何map。五个P3 UE map与原件bytewise一致，四个外部USD引用可读，
项目../plugins关系和插件文件均验证。

该正确目标上的实际证据（不只文件存在）：
 diagnostics/target_wide_native_probe_v2.json：40rigid+4extra animal；
 diagnostics/target_remaining_articulated_probe_v1.json：其余15 articulated逐个spawn、
 live SkeletalMesh、30条Idle/Walking路径读回后destroy。三组两两不交，覆盖全部59，
当前level apartment_0000读回通过；stage_manifest.json v3汇总，旧版保留。

## 4. 未完成分类

9个挂墙/吊顶源是placement interface_not_implemented，不是renderer缺binding，
不能标not_applicable或从分母删掉。四动物muzzle为真实joint_proxy，原源没有额外
显式mouth marker；不宣称人体/动物形态、碰撞支撑或人工外观资格。Dark sable的
黑背景review对比度低，外观可识别性保留evidence_missing，不伪填reviewed。
P7真实试听和46段条件仍按任务书执行。

## 5. Claude/调用接口

最高控制器使用examples/rooms/packages/catalog.json；请求runtime.uproject应使用
上述共同stage（显式旧请求仍保留其原运行目标）。资产经asset_id解析，SPEAR只读
exact paths，Habitat只读各自P12/P4 binding。生产捕获读回接口保持P1/P6/P9契约。
actor scale为真实读取值；animal时间线axis、root-reference与joint-local offset
分开记录。源/导入/原生校验路径在registry集成证据与stage manifest中。

## 6. Owner事项

所有产物research-only，无push/main merge、Studio切换或正式admission。后续
46段仍需十二项前置验收完成；不因有binding或probe就宣布数据集完成。
