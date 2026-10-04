# 第7回合 CCI/测试面板交接

结论：来源口径已修正，尚不可正式生产。只在48g改代码/CPU测试，没有运行CCI或GPU。

生产声音配置为`examples/dataset/v7_audio_sources_v1.json`，禁止eSpeak及程序合成动物/家电/环境/音乐；公开录制mono干声＋仿真RIR允许使用。VCTK库存24人/600段/342句（CC BY4.0），自然自编内容TTS须记录实际代码/权重/参考音频/输出链。逐条原始ID和许可证不可省略。

新的候选CLI要求`--recorded-dry-programs FILE.json`。FILE是scene_id到program的映射，program含：

```json
{
  "schema": "v7-recorded-dry-program-v1",
  "records": [],
  "events": []
}
```

records须为families自动准入记录（保留ID、许可、screens、source origin、实际输入路径），再过来源类型过滤。events每项含event_id、sound_asset_id、crop_start_sample；绑定所有实际scene事件，crop用原clip样本坐标。不允许素材不足时循环、拉伸、填机器人音。自编台词保持整句；人物声音元数据须符合既有场景类别，不改真值。actor内同一speaker保持身份；a1成员交换dry/activity关联时收据endpoint同步交换。原始注册路径迁到AFS时按原asset ID更新runtime路径，来源/许可/既有证据不可移花接木。

新runtime inputs和catalog位于本轮外部产物目录；RIR仍继承第6回合开启衍射/阶10/native单线程/已校准相关尾部crossfade配置。既有input/stage recipe加入普通audio source revision，旧eSpeak完成物不得自动恢复；需要新的输出目录，历史产物保持只读。没有改声学门槛或判分器。

本轮未创建正式30秒场景或14题窗口。families旧128正式干声中42条仍为eSpeak，现剔除，保留86录制片段/19类/78原ID。VCTK旧296自动准入仍0；600库存归因清单不等于技术/裁剪链准入。自然台词TTS准入0。人物线目前selection/许可记录，完整native登记和接入说明未交付。不能调度目前v5种子请求；原生v5布局/完整干声证据绑定、人物运动接地、剩余QA视觉信号还缺。

第6回合本底、80中点和900帧验收证据保持原路径不覆盖；报告已记录80/80（三参考中位数）、保留单参考79/80，仅两屋profile。此数字不意味着所有房屋或正式干声验收。HM3D遮挡抽检not_run、深色缅甸猫预期毛色未验证继续保留。

owner可将本轮提交/REPORT_zh.md转给cci-datagen及测试面板。此交接未发消息、未push；正式生产仍blocked，禁止把单元14题夹具当正式数据。


VCTK旧296条的上游拒绝详情为：license/自动activity/归一化许可链296条、模型receipt或音频hash缺失296条、参考去重扫描缺失296条；其中103条还有evaluation_source_split及evaluation_source_identity_overlap。来源见本轮`evidence/vctk_upstream_blockers.json`。本轮已核实VCTK 0.92公开许可与600条原utterance ID，但未新做其模型读回/归一化/完整评测同源扫描；不能把旧“无permissive逐clip许可”文字继续当VCTK发布许可结论。103条相交对象与STARSS/SAVVY实际同源关系未验证，保留原拒绝；既不能仅凭库split=eval字段判泄漏，也不能凭口径修正解除真实同源排除。上游须按允许录制干声的新口径重算并交付逐条技术证据后才能生产使用。
