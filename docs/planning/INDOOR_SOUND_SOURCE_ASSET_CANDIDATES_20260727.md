# Indoor sound-source asset candidates (AudioSet-grounded), 2026-07-27

Owner direction: sound sources come from AudioSet classes that plausibly
occur indoors; animals go through the hardened generation route (hard),
static objects may be plain static 3D meshes.  This document enumerates the
full candidate set, tiered by feasibility, with registry-mapping notes.

Local resources: AudioSet subset index
`/data/datasets/omniaudio/audioset_out.csv` (22,159 clips, labels column);
owner's prior extraction script `/data/jzy/code/Spatial/2-create_dataset_audioset.py`;
CLAP checkpoints trained on AudioSet.  Registry contract:
`examples/m6/registries/sound_assets_v1.json`
(`semantic_sound_class`, `taxonomy_path`, `dry_audio` + rights provenance,
`allowed_transforms`, `permitted_event_usage`); per-asset
`acoustic_profile.allowed_event_classes` stays independent of appearance
(three-independent-selections rule).

Selection criteria applied: (1) the sound has a VISIBLE physical source
object; (2) indoor-plausible in apartment scenes; (3) reasonably separable
dry clips exist in AudioSet (curation/SNR review still required per clip);
(4) rights review per repo policy before registration (AudioSet clips are
YouTube-sourced; research-use posture, per-clip provenance recorded).

Key simplification for static objects: the FLUX->Pixal3D route works for
them too but SKIPS TokenRig/retarget/gait entirely (no rigging) — only
watertight repair + emitter-anchor measurement + registry.  Alternatively
curated mesh libraries may be used subject to rights review.

## Tier 1 — first wave (low risk, high benchmark value)

### T1-A 动物品种名册（articulated，硬化路线；每品种=契约下独立资产）

犬（四足供体直接兼容；AudioSet: Dog/Bark/Growl/Whimper/Howl/Pant；
事件类 dog_bark/dog_growl/dog_whine/dog_howl）：

| 品种 | 状态/风险（morphotype checklist） |
|---|---|
| 柴犬 Shiba | ✅ 已出货（2026-07-26） |
| 柯基 Corgi | 排队中——短腿=中风险，checklist 预测力实验 |
| 金毛 Golden Retriever | 低-中风险（长毛）；有模板线历史对照价值 |
| 拉布拉多 Labrador（硬化重制） | 低风险；替换被弃用的前契约版 |
| 边牧 Border Collie（硬化重制） | 低-中风险；替换前契约版 |
| 哈士奇 Husky | 低风险；面部 mask 花纹好做毛色变体 |
| 德牧 German Shepherd | 低风险 |
| 比格 Beagle（生成版） | 低风险；可与 Rocketbox 模板版对照 |
| 吉娃娃 Chihuahua | 低风险（小体型，物理档差异大——size 轴价值） |
| 法斗 French Bulldog | 低-中风险（扁脸重建待验证） |
| 贵宾 Poodle | 中风险（卷毛重建） |
| 博美 Pomeranian | 中风险（蓬毛，柴犬毛壳经验适用） |
| 杰克罗素梗 Jack Russell | 低风险 |
| 腊肠 Dachshund | 中-高风险（极端比例，Corgi 之后的加压项） |

猫（同供体家族；AudioSet: Cat/Meow/Purr/Hiss/Caterwaul；
事件类 cat_meow/cat_purr/cat_hiss）：

| 品种 | 状态/风险 |
|---|---|
| 英短 British Shorthair | 排队中——复活测试（历史失败品种） |
| 暹罗 Siamese | 低风险（短毛、历史 research 视频先例） |
| 美短/虎斑 American Shorthair (Tabby) | 低风险；历史 2D 双尾教训已进硬门 |
| 斯芬克斯 Sphynx | 低风险（无毛=重建最友好），外观辨识度高 |
| 俄蓝 Russian Blue | 低风险 |
| 布偶 Ragdoll | 中风险（长毛） |
| 缅因 Maine Coon | 中风险（长毛+大体型） |

### T1-B 人类与静物

| # | AudioSet class(es) | 声学事件类建议 | 3D 形态 | 备注 |
|---|---|---|---|---|
| 1 | Speech (male/female) | human_speech | articulated（Rocketbox 现有） | 已在注册表（LibriTTS CC-BY） |
| 4 | Telephone / Telephone bell ringing / Ringtone | phone_ring, phone_vibrate | static（桌面电话/手机） | 小体积、事件清晰 |
| 5 | Alarm clock | alarm_beep, alarm_bell | static | 床头场景自然 |
| 6 | Doorbell | doorbell_chime | static（墙面模块） | **天然画外声源**（门外→入画叙事） |
| 7 | Door / Knock / Slam / Creak | door_knock, door_slam, door_creak | 建筑构件（静态+可选开合动画） | 画外支柱之二 |
| 8 | Microwave oven | microwave_hum, microwave_beep | static | 厨房区 |
| 9 | Kettle / Whistling kettle | kettle_whistle, water_boil | static | 厨房区 |
| 10 | Vacuum cleaner（含机器人吸尘器） | vacuum_hum | **rigid 可移动**（扫地机器人=移动刚体声源！） | 非关节移动源，运动轴新增覆盖 |
| 11 | Washing machine / Dishwasher | washer_cycle, dishwasher_hum | static | 长时程连续声 |
| 12 | Mechanical fan / Air conditioning | fan_hum, ac_hum | static（落地扇/壁挂空调） | 宽带持续声，好做干扰源 |
| 13 | Clock / Tick-tock | clock_tick | static（挂钟/座钟） | 低幅周期声，弱信号测试 |
| 14 | Water tap / Sink (filling, running) | tap_running | static（水槽装置） | 厨卫区 |
| 15 | Music box | musicbox_melody | static | 自发声合理，旋律类 |

## Tier 2 — second wave（可行但各有注意点）

| # | AudioSet class(es) | 事件类 | 3D 形态 | 注意点 |
|---|---|---|---|---|
| 16 | Television / Radio / Loudspeaker | playback_speech, playback_music | static | **播放悖论**：内容与源类别脱钩——刻意用作 hard negative（"声音来自电视还是真人？"） |
| 17 | Blender / Food processor | blender_whir | static | 短时高能 |
| 18 | Hair dryer | hairdryer_blow | static（可手持→需 agent，先做台面态） | |
| 19 | Toaster | toaster_pop | static | 事件极短，时序定位题好素材 |
| 20 | Printer | printer_whir | static | 办公角 |
| 21 | Computer keyboard / Typing | keyboard_typing | static | 需不需要手指动画待定（无 agent 版先行） |
| 22 | Toilet flush | toilet_flush | static | 卫生间；画外候选 |
| 23 | Frying (food) / Sizzle | frying_sizzle | static（灶+锅组合） | 厨房长时程 |
| 24 | Wind chime | windchime | static（窗边悬挂） | 随机性纹理 |
| 25 | Electric shaver / Toothbrush | shaver_buzz, toothbrush_buzz | static | 本地 CSV 已见 Toothbrush 条目 |
| 26 | Sewing machine | sewing_whir | static | 低频出现但室内合理 |
| 27 | Refrigerator hum | fridge_hum | static | 极低显著度，弱信号/底噪层 |
| 28 | Laughter / Cough / Sneeze / Clapping | human_laugh, human_cough, human_clap | articulated（现有人形+事件动画） | 复用 Rocketbox，加事件类即可 |
| 28a | Rabbit（thump/咀嚼） | rabbit_thump | articulated | 跳跃步态≠步行供体——需动作家族决策或取坐姿+ subtle idle |
| 28b | Ferret | ferret_dook | articulated | 细长脊柱，供体距离中风险 |
| 28c | Hamster / Guinea pig（squeak） | rodent_squeak | articulated 或笼内半静态 | 小体积+笼遮挡，声显著度低但室内合理 |

## Tier 3 — 难/后排（明确风险再决定）

| # | 类 | 风险 |
|---|---|---|
| 29 | 笼养鸟（Parrot/Canary：Chirp, Squawk） | **新 body plan**——四足供体不适用，需鸟类动作家族；栖息态半静可折衷 |
| 30 | Baby cry | 需婴儿资产（生成+审查复杂度高）；伦理/视觉审慎 |
| 31 | 乐器演奏（Piano, Guitar, Violin） | 无人自奏不合理→需演奏者动画（agent 联动）；自动钢琴可例外 |
| 32 | Cutlery/Dishes（餐具碰撞） | 需操作 agent，事件与可见源绑定弱 |
| 33 | Hamster/Guinea pig（笼内啮齿） | 小体积+笼子遮挡，声学显著度低 |

## CVPR 配对价值标注

- **同类不同外观对**（instance-binding hard negative）：两台不同颜色的电话/音箱/闹钟，仅一台发声——静物即可量产，比动物毛色变体便宜得多。
- **画外→入画**：doorbell、door knock、(门外)vacuum、toilet 天然支持。
- **移动源**：动物（关节）+ 扫地机器人（刚体）双运动类型。
- **播放悖论对**（T2#16）：电视播狗叫 vs 真狗叫——反捷径的高级题型。
- **静默可见干扰**：任意静物摆多台只响一台，S2 场景直接扩展。

## 下一步

1. Owner 圈定第一波清单（建议 T1 全量 15 项起步）。
2. 对圈定项跑 AudioSet 干声可得性审计（本地 22k 子集命中率 + 需全量抓取的类；复用 `2-create_dataset_audioset.py`）。
3. 静物 3D 路线选型：FLUX->Pixal3D 免绑定支线 vs 版权可用的现成网格库（逐项 rights 记录）。
4. 声音注册表扩表（每项 dry_audio + rights + acoustic_profile），外观注册表按三独立选择规则并行扩。
