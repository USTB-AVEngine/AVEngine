# 放量就绪报告（G-E，2026-09-07）

分支：`grok/pilot46-fixes-20260907`（HEAD 在提交本报告时更新）。  
对照任务书 `docs/roadmap/GROK_FIX_TASKS_20260907.md` 第 2 节 G-E、第 3 节验收、第 4 节就绪门。  
**结论：还不能写「可以开大规模生产」。** 代码修复与 20 段重跑已经证明 Habitat 音频模式和 A/B/C 曝光可修；先导 46 格仍有冻结缺额、规划用尽、挂装接口和人工试听空着。

## 合并后的 46 格

原批只读：`.../qa_pilot46_background_20260907_v1/`（39 delivered / 5 failed / 2 blocked）。  
重跑：`.../qa_pilot46_rerun_20260907_v1/`（20 段 `attempt_02`：19 delivered / 1 failed）。  
合并表：`.../qa_pilot46_merged_20260907_v1/`。

| 状态 | 段数 | 说明 |
| --- | --- | --- |
| delivered | 42 | `attempt_02` 取代对应 `attempt_01` |
| failed | 2 | B 房 animal_device 规划 200 次用尽；C 房 animal_device 原批规划用尽（未重跑） |
| blocked | 2 | A 房 / 酷家乐 device_device 预分配超预算，身份未换（本 46 格清单冻结） |

42 段 delivered 的曝光闸门均为 **pass**（A/B/C 重跑写入 review；其余 23 段离线闸门写在 `merged/.../exposure_gate_offline/`，未改 `attempt_01`）。A/B/C 重跑帧 0/中/末灰度约 136–192，饱和 ≪ 5%。

Habitat 四段失败已由 `attempt_02` 交出 facts/questions 与 16 s 双耳；AudioProgram 为 `simultaneous_subset` / `one_active_of_n`。

覆盖表 9912 行（合并后）：produced 246 / deferred 976 / N/A 1204 / interface_not_implemented 1260 / evidence_missing_or_unsampled 6226。  
家族 × 人/动物/设备：apartment、authored、kujiale、mp3d、hm3d 每格都有至少一段闸门通过的成片。

## 第 4 节就绪门

1. **46 格五态 + 证据路径：基本满足。** delivered 全过曝光闸门。失败/缺额原因：
   - `authored_b_animal_device`：`failure_stage=planning`，`gap_state=evidence_missing_or_unsampled`，直方图在 outcome。
   - `authored_c_animal_device`：原批只记了 `controller_exit`；审核与 stderr 是规划用尽。**未用新记账器重跑**，不能把测试变绿说成这条 outcome 已改完。
   - 两条 device_device blocked：预分配超预算，`identity_substitution_applied=false`。
2. **四家族 × 三类成片：满足**（上表）。
3. **放量干跑：repeat 缺额 0、交叉表打散、画外 14 段在列。** 另有 2/350 格 `no_distinct_compatible_sound_identity`（便携空调，不是 repeat 超预算）。路径：`tmp/gc_scaleup_dryrun_7x50_20260907/`。
4. **相关单测：** 见文末计数。`docs/TOOL_INDEX.md` 已随 `run_qa_batch.py` CLI 更新。
5. **未完成项（不冒充完成）：** 9 个挂墙/吊顶资产的挂装接口仍是 `interface_not_implemented`；人工校准空；五段抽听与 P7 十条试听仍 `pending_human`；本 46 格两条 repeat 缺额未改身份。

## 任务书逐项

| 项 | 结果 | 证据 |
| --- | --- | --- |
| G-A Habitat 模式 / beagle 可选 / 失败记账 / 契约前验 / 湿尾 | 通过 | `G-A_habitat_batch_accounting.md`；四段 attempt_02 16 s 双耳 |
| G-B 过曝根因 / 闸门 / 地板字段 / 包路径 | 通过 | `G-B_ue_visual_packages.md`；A/B/C 重跑闸门 pass |
| G-C 同层 / repeat 预分配 / 打散 / 画外 / 5° 分档 | 通过（本 46 格 repeat 缺额仍在，放量生成器为 0） | `G-C_sampler_manifest.md` |
| G-D 非人类门槛 / body_color / in_fov 字段 | 通过 | `G-D_appearance_pixel_evidence.md` |
| G-E 重跑 4 Habitat + 16 A/B/C | 19/20 delivered；B animal_device 规划用尽 | 本文件 + merged 表 |
| G-E 放量干跑 | repeat 0；2 条其它身份缺额 | dry-run summary |
| 人工试听 | 未做（禁止代填） | pending_human |

## 测试

合并树上 `import avengine` 指向本 worktree。相关 12 个 unit 文件 **122 passed / 0 failed / 0 skipped**。

## 没做完（三分）

- **题义不适用：** QA-18 湿尾边距按 Codex 现状不动；train/eval 房间预留 owner 已否决；五段抽听不代填。
- **接口未实现：** 9 个挂墙/吊顶摆放；C 房 animal_device 原批 outcome 仍是 `controller_exit`。
- **证据缺失：** 本 46 格两条 device_device repeat 超预算未换身份；B 房 animal_device 200 次未找到合法机位；放量干跑 2 条无兼容声音身份；人工可答性/模态必要性/正式准入都没有。

## 需要 owner 拍板

- 本 46 格两条冻结 repeat 缺额：放量生成器已会换短片，**要不要回写先导清单**。
- 两条 animal_device 规划用尽：放量时丢弃该画像还是加相机预算。
- 放量 2 条便携空调无兼容声音身份：换资产还是允许缺额。
- 挂墙/吊顶 9 个资产继续留在分母，还是从放量范围拿掉。
- 在人工试听仍空时，是否允许「机器闸门通过即可开大规模生产」。

在以上五条未裁定前，**不能**把本报告读成生产放行。
