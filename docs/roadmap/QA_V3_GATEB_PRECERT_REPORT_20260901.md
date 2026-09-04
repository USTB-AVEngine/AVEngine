# QA v3 Gate B 认证前置闭合（2026-09-01）

216 条 Gate B 金标逐题重算：180 条纯事实/几何题全部 pass，36 条像素依赖题保持 pixel_pending，0 条纯事实孪生 reject。

权威产物：
- `/data/jzy/tmp/qa_v3_gateb_gold_recompute_216_20260901_v4/gateb_gold_manifest.json`
- `/data/jzy/tmp/qa_v3_gateb_gold_recompute_216_20260901_v4/augmented_pilot_manifest.json`

augmented manifest 已回填全部 Gate B root、main/gateB gold、状态和音频策略。

路线孪生代表 card2_001 的 main/Gate B 各完成75帧UE与双耳，两份 mixture 不同，符合音频随新路线变化。

外观孪生代表 card7_001 若按换后的整套资产重渲音频，mixture 也变化，原因是资产 emitter anchor 高度不同。因此外观孪生必须复用 main 音频；重渲染只作禁止使用的诊断。manifest 策略为 `appearance_reuse_main_audio_no_rerender`；路线孪生为 `route_audio_must_change_consistently`。

Gate B 像素代表：card11_002 因 source4 仍可见而 reject；card15a_002 的 source4 out_of_view，成立；card16_005 在绑定帧两只可见、片尾状态不同，成立。

本轮是 precert，不是 A-only/V-only 认证、人类实验或正式准入。36 条像素依赖孪生尚未全量渲染，card11 需继续重采/筛选。


## 结构化代表证据

`/data/jzy/tmp/qa_v3_gateb_representative_evidence_20260901_v1/gateb_representative_manifest.json` 绑定两类音频与三类像素输入。结果：appearance 重渲染漂移，策略为复用 main 音频；route 音频随路线变化；card11/card15a Gate B 像素 reject，card16 pass。

测试：新增规则 2 passed；顶层 306 passed。
