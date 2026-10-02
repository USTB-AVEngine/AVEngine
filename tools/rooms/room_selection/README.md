# 共享测量上的选房判断

以 `room_screening` 为唯一测量和审核界面实现；本包负责候选登记、楼层窗口、资格判断、CPU 三点摆放、切分建议、房子门禁、校准统计和 v7 清单。真实资产、审核章和已有导航网格均只读，所有派生产物由 `--output` 指定。HM3D 语义扫描面积是诊断代理，不能称经过人工核实的建筑净面积。

- 地面类别和语义映射：共享 `geometry.load_semantic_ground_and_instances`，重复颜色保留区域并标冲突。
- 家具占地：共享 `shape_preserving_projected_footprint`，按已有导航网格的 `agent_height` 裁剪，同 region/未分配实例已知占地；跨 region 不静默扣除。
- 地面面积：共享投影多边形并集；判断层以面积加权、最大 0.3 m 高度窗口分层。
- navmesh 面积：共享 `navmesh_triangles` 连续交集；判断层原生图确定主连通区域，网格支持域裁剪连续面积。旧网格面积另报，避免把吸附后的整个网格单元当连续导航面积。
- 摆放：已有 navmesh + 原始碰撞 GLB 的 CPU 三条射线；有限候选搜索没有声学保证。
- 切分：家具簇、测地距离和窄通道加权最小割；每个子块重新测量、独立人工审核。

## 外部配置与顺序

与 smy 工具一致，运行时由 `AVENGINE_HABITAT_RUNTIME_PREFIX`、`AVENGINE_HABITAT_MAGNUM_PYTHON_SITE`、`AVENGINE_MP3D_ROOT`、可选 `AVENGINE_RLR_SDK_ROOT` 配置。数据入口为 `--inventory` 和 `AVENGINE_ROOM_TASKS_ROOT`、`AVENGINE_ROOM_VERDICT_ROOT`、`AVENGINE_ROOM_MEDIA_ROOT`（也可传同名 CLI 路径）。不在仓库配置私人服务器路径，不启动 GPU Simulator 或重建 navmesh。

```bash
python -m tools.rooms.room_selection.run register --inventory /path/to/assets_inventory.json --analysis-split /path/to/original/house_analysis_split.json --output /path/to/output
python -m tools.rooms.room_selection.run measure --partition calibration --skip-splitting --workers 4 --output /path/to/output
python -m tools.rooms.room_selection.workflow freeze --output /path/to/output
python -m tools.rooms.room_selection.run measure --thresholds /path/to/output/thresholds.frozen.yaml --workers 4 --output /path/to/output
python -m tools.rooms.room_selection.workflow evaluate --output /path/to/output
python -m tools.rooms.room_selection.workflow publish --output /path/to/output
```

`run` 测量从不评估标签。`workflow` 要求先准备 `calibration_plan.json`、`previous_82382e2/`（原 agreement、固定名单和阈值）及 inputs 中的手工参考/clean-house 来源，再完成 calibration。冻结改变结果的参数与代码身份是为防止已经观察 holdout 后反复改规则；重复 evaluation receipt 会拒绝再次评估，不能删除 receipt 以挑选结果。`publish` 只读保存的结果，不重评。沿用原按房子划分，这批 holdout 的本次结果是第二次历史评估，不是假称新的独立留出验证。

`compare` 的声明名单必须来自原 calibration，保留旧地面实现的复算核对、10栋每房每层数值与 >10% 差异分解。真实清单与结果不提交 Git。所有参数及来源/理由见 `thresholds.yaml`；缺少素材、颜色冲突与实质多楼层保留 review，不悄悄漏登记。

## 人工审核

本包不提供第二个 HTML 或 HTTP server。`review_manifest.json` 使用 `schemas/room_screening_review_manifest_v1.schema.json`，交给 `room_screening.review_server`。新字段包括 `automatic` 原因、`split_overlay_files`、`gate`、`second_review`、主层、独立子块和固定理由清单。原图缺失明确标记，不能拿几何建议图替代扫描。

巡房视频可用显式 `--media-root` 指定只读外部根；所有 asset/media 引用都校验路径范围。`--blind-second-reviewer` 在 API 边界隐藏第一次和自动提示，只提供原固定180项及原图/视频。反馈只保存到 `--feedback`，不回写正式审核章。原 region 与新子块分别选 use/skip/unsure、固定理由与审核人代号。

## 验证

```bash
python -m pytest tests/unit/room_screening tests/unit/test_room_selection_protocol.py -q
python tools/build_tool_index.py --check
```

单测覆盖共享语义冲突、多层家具审计、楼层窗口、几何/射线、统计分母、盲审字段和重复 holdout 拒绝；另须按任务许可在真实场景做 CPU 冒烟。当前合并测量后端只接受 HM3D；旧 MP3D pilot 不是这一版的验证。
