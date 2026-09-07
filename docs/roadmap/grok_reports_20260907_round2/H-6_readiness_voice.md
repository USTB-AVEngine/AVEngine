# H-6 就绪报告口径

日期：2026-09-07。工作树 `/data/jzy/tmp/wt-grok-pilot46-H6`。分支 `grok/pilot46-fixes-round2-20260907-h6`。对照第二轮任务书第 2 节 H-6、第 3 节就绪门、第 4 节报告格式。文档任务，不改 Python / Studio / 第一轮原文。

## 1. 改了哪些文件（路径），提交号

只写第二轮报告目录（第一轮 `docs/roadmap/grok_reports_20260907/` 未改）：

- `docs/roadmap/grok_reports_20260907_round2/READINESS_20260907_round2.md`
- `docs/roadmap/grok_reports_20260907_round2/H-6_readiness_voice.md`（本文件）

提交号即本文件所在提交。未 push。未合并 main。

## 2. 跑了哪些测试，各自的通过/失败/跳过计数

H-6 是口径文档，本项无新单测。不发明合入后计数。

Pre-round-2 baseline（Claude 审核，合并树 `ba0150e`，H-1～H-7 未合入）：**283 passed / 0 failed / 0 skipped**，23 个文件，14.28 s。日志 `/data/jzy/tmp/claude_audit_20260907_v2/pytest_grok_related.log`（点阵 + 摘要，无文件名）。文件清单与 collect-only 分列见 `READINESS_20260907_round2.md` §4。

合入后计数：**TODO(parent): fill after H-1..H-7 merge**。

## 3. 验收产物的路径，以及亲自核对结果

产物：`docs/roadmap/grok_reports_20260907_round2/READINESS_20260907_round2.md`。

按 H-6 五条核对：

1. 结论句只写机器闸门/契约的已核事实、owner 对缺额的原话、以及第二轮门未合入。不以听感、不以测试变绿当放行条件。
2. 「不要求每个房间跑出每种题」只绑在先导四格；「先导切分 train39/eval0：接受」不扩成「无独立 eval」。
3. 单测附 23 文件表与 pytest 摘要；合入后数字留 TODO。
4. Owner 本机包是五段 human_human（4× attempt_01 + A 房 attempt_02），与 `five_clip_listening_pending.json` 不同；heard/reviewer 不代填；H-1 后建议新包含动物与设备各至少一段。
5. G-B「任务书 -3 EV vs B 房 -4」判为假冲突（任务书无 EV 数值；B -4 见 `walk_four_b_v2/plan/episode_plan.json:791`）。「闸门未接入 batch_delivery」在 `789bc64` 已过期（`:331-338` 已调用 `apply_exposure_gate`）。

仍空：挂装、人工校准、逐条试听、正式准入（24 类题不看 facts 的可答性 + 缺失模态视/听消融 + 人工校准阈值）。正式准入按 owner 书面记录现在不做。

## 4. 没做完的部分和原因

- **题义不适用：** H-1 重渲、H-2 合并脚本入库、H-3 路径绑定、H-4 校验顺序、H-5 外观接线、H-7 干跑请求。H-6 不改那些文件。
- **接口未实现：** 9 个挂墙/吊顶挂装（`static_attachment_surface_not_implemented`）。`except ImportError: pass` 仍在，属 H-4。
- **证据缺失：** 第二轮就绪门第 1–4 条的产物与计数；H-1 后新试听包；逐条 `heard`/`reviewer`；正式准入三项。合入后数字标 TODO，不猜。

## 5. 对后续接口的要求

- 就绪结论句只允许：闸门数值、契约是否在写 facts 前通过、owner 原话、仍空项。禁止把听感或「测试变绿」写成放行支柱。
- 试听 JSON 字段 `heard`、`reviewer`、`notes`、`source_assignment_clear` 保持 null，直到 owner 自己填。
- H-1 新试听包建议至少一段动物、一段设备；听不听是 owner 的决定，不是 H-6 的代填。
- 父代理合入后只替换标了 `TODO(parent):` 的格子，不要改 §2 的原话。

## 6. 需要 owner 拍板的地方

本项不自行扩大已有裁定。仍待定（审核 §6，不是 H-6 能定的）：

- 便携空调：从候选范围拿掉，还是允许裁剪超长设备声。两条干跑缺额 owner 已「接受」。
- 画外锚点不说话时能否入画；设备要不要有画外槽。
- H-1 重渲后要不要听新包。
- 先导两条冻结 repeat 缺额要不要回写清单。
