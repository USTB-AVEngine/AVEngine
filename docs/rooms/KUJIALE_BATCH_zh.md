# InteriorAgent 批量转换与 v7 训练候选筛选

`tools/rooms/kujiale_batch.py` 在 CPU 上依次执行盘点、转换和声学验收。
输入是独立下载的 InteriorAgent 目录、当前下载 manifest、冻结选房 YAML，
以及 `3414629` 的 v7 声学 manifest。原始数据不修改；每次转换使用新 build 目录。

盘点先测 USD 地板构件、房间多边形、几何、组合错误与 asset 路径。
封闭地板下表面不是行走楼层。正式房间度量使用朝上的地板面，
并复用 `room_selection.geometry.floor_windows`、`protocol.scope_metrics`、
`measurements.navigation_geometry`、原生 PathFinder 采样和三条网格视线摆放搜索。
面积、短边、可走面积、楼层占比、净空、距离和视角门槛全部从冻结 YAML 读取。
设计类别仅做明确映射：living room→living，study room→office，balcony→outdoor；
未知用途保留 unknown，storage、走廊和户外不能成为训练房。

`--whole-stage` 通过一条 /Root 引用保持整屋跨 scope 材质绑定；旧适配器默认行为不变。
批处理把合成 USD 展平为 `engine/render.usdc`，仅保留生成的 Preview Surface 图，
对实际用到的图像解码、重新编码、写成相对路径并核对依赖闭包。
原始下载包、MDL 源文件和外部 UE 材质路径提示不进入这个闭包。
无面的 Mesh 层级容器仅改为 Xform，保留子节点与变换，不省略真实表面。
自相交 rooms.json 多边形用 make_valid 保留全部正面积分量，操作记入 inventory。
这些操作不构成原生视觉或材质资格证明。

同一份可见表面生成几何 GLB、数组、5cm/1cm 的 Recast navmesh、房间语义和地板记录。
声学快照与语义材料包使用现有 USD extractor、编译器及 RLR 兼容派生器。
`acceptance` 调用 `3414629` 的 measure/classify；不改声音设置或验收门槛。
最多选三间，按小、大、新房型顺序选位置。所有 Kujiale 仅面向训练。
已有 v6 暴露信息保留，但不会凭此把房屋纳入开发或测试。

```bash
python tools/rooms/kujiale_batch.py audit --source-root "$DATA" --output "$OUT" --thresholds "$FROZEN"
python tools/rooms/kujiale_batch.py convert --source-root "$DATA" --output "$OUT" --thresholds "$FROZEN" --baseline-manifest "$V7_MANIFEST" --material-rules examples/acoustics/semantic_materials/residential_material_rules.json
python tools/rooms/kujiale_batch.py acceptance --source-root "$DATA" --output "$OUT" --thresholds "$FROZEN" --baseline-manifest "$V7_MANIFEST"
```

运行时固定 `CUDA_VISIBLE_DEVICES=''`，输出、临时文件、matplotlib 缓存指向任务目录。
`--house` 可重复；`--build-name` 选择新目录；`--resume` 只跳过已有完整 conversion_result。
整个目录已有部分失败输出时，换新 build 名称，保留错误证据。
派生体积先按三倍下载量加 2GB 预留估算，再按实测 pilot 更新；超过 50GB 停止。

CPU 导出是可供 UE/SPEAR 继续导入的研究资产，并非已烘焙的原生 UE 地图。
正式准入还须原生 UE RGB 缺面/穿模/全黑检查、碰撞或深度地板偏移、人物脚底，
以及原生 UE 导出与声学几何的一致性核对。不能复制 0020 的零地板偏移或灯光。
旧 PreviewSurface 缺图或未覆盖的材质参数须核对真实渲染；不能把默认色当作正确材质。
只有既有同负载显存峰值和足够空闲显存证据满足任务约束后才启动 GPU。

原始及派生资产都受 InteriorAgent 非商业研究、不得再分发、须引用的条款约束。
Git 仅提交代码、说明与测试；输出 asset index、sha256 清单仅用于自有 48g/CCI 的内部转移。
未生成 v7 数据集，不自动注册或提升 research_only 状态。

扩展材质步骤为 `kujiale_batch.py refresh-materials`。它保留原生 PreviewSurface 图，
按源 MDL 明确的常量和直接纹理模式映射 BaseMaterial v2，补充 OmniPBR 与 v1
光泽/金属纹理；复杂 falloff、旋转 UV、specular 和法线强度仍记为原生待审。
缺图仅清空 file 并保留原先 authored fallback，不能据此准入。
`material_refresh.json` 记录每一项，并核对新旧几何数组逐项相同。
旧 engine 不覆盖；CCI 清单指向 engine_expanded。

`--resume` 可接着已完成的 adapter、闭包、表面和导航阶段；它先核对数组一致，
不覆盖半成品闭包或未完成的声学包。遇到这些半成品应选择新的 build 名称。
`evidence/build_selection.json` 记录任务内某套的明确恢复目录，其他检出不读取为产物。
0024 的 EXR 读取开关仅设在此转换进程；EXR 保持浮点格式，不改公共环境。

完整转换与材质更新结束后执行：

```bash
python tools/rooms/report_kujiale_batch.py prepare-inputs --output "$OUT"
python tools/rooms/report_kujiale_batch.py report --output "$OUT"
```

`engine_asset_index.json` 用相对路径指定可搬迁的 CPU 派生闭包。
navigation/build_result.json 等旧绝对路径仅为 48g 来源记录。
native_ue_pending/ 仅提供具体导入 request 与实测摆放相机，没有启动 UE、建图或画面资格。
报告和索引仅输出到任务目录，派生几何、纹理、声学数组不进入 Git。

原生 `pxr.Gf.Vec*` 不属于 Python `Sequence`：扩展映射按索引读取 v1 色彩、
光泽、金属和发光通道，避免旧数值辅助函数悄悄使用默认值。
已有输出可用 `fix-native-scalars` 只修派生 USD 数值及纹理 scale/fallback，
不重编码纹理；旧 USD/receipt 另存，逐套重新核对几何数组。
旧 USD 备份不进入 CCI 当前操作闭包清单。
