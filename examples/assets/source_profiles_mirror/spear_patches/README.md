# 打给 SPEAR 的补丁（因为那个仓提交不了）

`SPEAR-lead-b` 的 git 工作树断链，所以在那边改的代码**一行都提交不了**。
这里放的是**可重放的补丁脚本**：每个脚本用精确字符串匹配来改，匹配不到一次就报错退出，
所以它同时是"改了什么"的记录，也是"能不能重放"的自检。

按顺序跑（在服务器上，`python3 <脚本>`）：

| 脚本 | 改了什么 | 为什么 |
|---|---|---|
| `write_audio_playback_profiles.py` | 新建 5 份 `audio_playback` static_object profile | 参考运行要用 |
| `revise_profiles_v2.py` | 书架箱 / 电视两份方法修订 + provenance | 见工单 §6.11、§6.12 |
| `fix_attribute_input.py` | `run_controlled_static_object_admission.py` 的 `attribute_input` 校验 | **真实集成 bug**：水密工具总会多写一个 `same_as_geometry_input` 键，而 `validate_file_record` 要求字段集精确相等 —— 这条校验从来不可能通过 |
| `fix_admission_test.py` | 上面那条的单测桩 | 桩写的 `attribute_input` 和真工具不一致，正是 bug 能活下来的原因；改完 14 条测试全绿 |
| `fix_finalizer_vertex_count.py` | `blender_finalize_generated_static_object.py` 的受保护字段 | **真实集成 bug**：受保护字段里有 `vertex_count`，但它在 glTF 导出/导入往返下不守恒（导出会在 UV/法线接缝处裂点）。实测 37293 → 37353，而三角面数 60000 不变。改成按位置焊接后的顶点数，实测 30002 → 30002 |

两个 bug 都是**第一次真的跑这条链路**才暴露的：两个工具的单测都把对方桩掉了，
所以它们从没在同一次运行里见过面。


## 参考运行用到的其它脚本

这些不是补丁，是**跑这条链路要用的工具**，工单 §3.6 会引用它们。
它们都只读 GLB 的原始缓冲区，不需要 Blender。

| 脚本 | 做什么 |
|---|---|
| `author_admission.py` | 写朝向证据 + 锚点授权 + 准入计划三份 JSON。加新物体改 `ANCHORS` 表 |
| `write_2d_decisions.py` / `write_3d_decisions.py` | 把人眼判决写成两道评审要的 schema（是模板，判决内容当然要自己填） |
| `contact_sheet.py` | 把一批候选图拼成一张联系表，方便一次看完 |
| `audit_raw_topology.py` | **先按位置焊接**再数边界边 / 非流形边 / 连通分量。glTF 在 UV 与法线接缝处拆点，不焊接的话数出来全是噪声——水密工具自己的 manifest 就报了 374913 条"边界边"，其实焊接后是 0 |
| `audit_components.py` | 主壳外的**面积占比**，以及碎片有没有把包围盒撑大（定型按包围盒高度缩放，一个飘在上方的碎片会把物体本身缩小） |
| `audit_base.py` | 拟合最低那层的支承平面，量它离水平差多少 |
| `audit_upright.py` | 面积加权主轴的仰角。立式物体应当接近 90°，长条应当接近 0°。**这个比 `audit_base.py` 可靠**，因为长条只靠两只小脚着地时平面拟合点太少 |
| `audit_yaw.py` | 从几何量偏航：平面主轴 + 最大竖直面板方位。细长物体准，近似方形的物体这个量没意义 |
