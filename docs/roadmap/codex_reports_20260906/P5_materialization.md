# P5 Habitat common-plan materialization

## 1. 修改与提交

共用物化器已随其 P12 实际执行依赖提交于 5e2002f；P5 完整采样器见 P5_conditioned_sampler.md。P4 基线为 8ac2ea1。

- src/avengine/assets/mp3d_region_actor_tracks.py
- tools/capture/materialize_common_plan_habitat.py
- tests/unit/test_mp3d_common_plan_materialization.py
- docs/roadmap/codex_reports_20260906/P5_materialization.md

## 2. 接口与行为

新增函数：

    materialize_common_plan_habitat(
        plan,
        room_manifest,
        runtime_registry,
        output,
        habitat_binding_delta=None,
        base_m1_request=None,
    ) -> dict

函数只消费 plan.visual_plan 的既有实体、逐帧 root_transform、
action_time_ticks、action_phase、moving 和 camera_state，不调用路线规划、
路线评分或路线重采样。clock 六字段逐项保留。

关节实体从 runtime_backends.habitat 的 M2 manifest/request 加载已烘焙
idle/walk action，按计划 action_time_ticks 在 baked sample grid 取姿态；
刚体实体直接生成 static track，保留计划 root_transform 与 floor
resting_pose。单位 scale 以当前 Habitat materializer 的既有能力校验为
[1,1,1]，其它 scale 显式拒绝，避免静默丢失变换。

M1 request 只复制 base_m1_request 的传感器、listener、calibration 和其它
设置；camera 与 sources 总是由 common plan 替换。camera basis 按
Habitat 的 [right, up, -forward] 组成旋转，并核验静态 camera 在所有帧
保持一致。sources 从逐帧第 0 帧 root 和 emitter binding offset 生成，
不使用旧 request 的 camera/source。

输出目录包含：

- common_plan.json
- m1_capture_request.json
- case_manifest.json
- tracks/sourceN.json
- research_receipt.json

输出后会调用现有 capture 的 case/track/M1 输入校验器；receipt 明确记录
route_replanned=false、逐 actor root 最大误差和
capture_input_validation=pass。

## 3. 测试与实际验证

- py_compile：P5 module 和 CLI 通过。
- P5 专属测试：2 passed。
- 既有 MP3D track + P5 测试：7 passed。
- P4 capture、既有 MP3D track、P5 测试合计：10 passed。

实际输入计划：

    tmp/p5_common_plan_20260906_v1/common_plan.json

该计划由已完成的 beagle + black-ash speaker Habitat 30 帧产物构造：

- clock：30 帧、15 Hz、3200 ticks/frame、48 kHz time base、16 kHz sample；
- source1：beagle，articulated_animal，M2 action tick 直接沿用；
- source2：black-ash bookshelf speaker，rigid_object，静止 floor pose；
- camera：来自实际 Habitat camera readback 的 position/basis/FOV。

最终物化产物：

    tmp/p5_materialized_common_plan_20260906_v4/

实际读回/校验：

- case clock 与 common plan 六字段完全相同；
- 2 个 actor、30 帧全部写出；
- source1 root 最大误差 4.44e-16；
- source2 root 最大误差 0；
- camera static validation 通过；
- route_replanned 为 false；
- _resolve_case_track_paths 通过；
- _load_case_and_m1 通过，room 为
  habitat_mp3d_example_17DRP5sb8fy；
- CLI receipt 的 capture_input_validation 为 pass。

## 4. 尚未完成与边界

此函数只做计划物化，没有启动 Habitat capture、RLR 音频或正式 admission；
这些仍由后续执行器/音频/收口阶段负责。common plan 中出现不支持的非 unit
scale、非静态 camera、缺失 M2 baked action、binding identity 不一致或
刚体 moving=true 时会显式失败。

## 5. 给父代理的接入摘要

默认调用只需要 runtime registry：

    PYTHONPATH=src python tools/capture/materialize_common_plan_habitat.py
      --plan <common_plan.json>
      --room-manifest <room_manifest.json>
      --runtime-registry examples/runtime/source_asset_runtime_profiles.json
      --base-m1-request <base_m1_request.json>
      --output <fresh-output>

尚未合入的 P12 资产可额外传 --habitat-binding-delta；已合入的 beagle 和
刚体 speaker 优先从 runtime registry 解析。工具索引由父代理统一重生成。

父 Agent 集成补充：物化测试现为4 passed；research-candidate 显式参数、camera frame/resolution、M1 seed int32映射、package actor/skin root关系已接入。最终A/MP3D/HM3D完整240帧原生结果见 P5主报告和H5附报；本节早期v4仅为物化器历史验证，不是最终原生验收。
