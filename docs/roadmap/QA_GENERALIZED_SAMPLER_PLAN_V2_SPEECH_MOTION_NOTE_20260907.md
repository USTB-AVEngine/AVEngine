# speech_motion 三种模式的语义（v2 旁注，2026-09-07）

这份说明紧挨 `docs/roadmap/QA_GENERALIZED_SAMPLER_PLAN_V2_20260906.md`，只写
`conditioned_static_v2` 已经实现的运动约束，不改 v2 正文。

`speech_motion` 约束的是**锚点可听窗内**谁必须在走，不是全员互斥，也不是整段 16 秒都要走。

- `speaker_moving`：每个锚点发声者在其可听窗内必须在走。其余关节实体**可以**走也可以停，**不要求竞争者静止**。刚体仍不得走。
- `competitor_moving`：至少一个非锚点的关节竞争者在锚点可听窗内在走。锚点本身可以静止。
- `all_still`：锚点可听窗内所有实体静止。

对应实现是 `src/avengine/rooms/conditioned_sampler.py` 的 `_moving_flags`（抽谁走）和
`select_camera_and_schedule` 里对 `moving` 掩码的逐帧约束。规划失败时换种子，不换画像。
