#!/usr/bin/env python3
"""生成事件类到 FSD50K 标签的对照表草案。

映射是人工判断的(自动匹配会把微波炉提示音配成乌鸦叫),条数由脚本
从单标签表里实算,避免手打错。
"""
import csv
import collections
import json
from pathlib import Path

CSV = Path("/data/datasets/omniaudio/tse_data/single_label_output.csv")
counts = collections.Counter(r["labels"] for r in csv.DictReader(CSV.open()))

MAP = [
    ("air_conditioning", ["Air_conditioning", "Mechanical_fan"],
     "空调外机和风扇的持续风声,两者接近,可合用"),
    ("alarm_beep", ["Alarm", "Buzzer", "Buzz"],
     "通用报警的嘀嘀声。Alarm 是个大类,抽听时要剔掉警笛那种"),
    ("alarm_bell", ["Bell", "Church_bell", "Alarm_clock"],
     "铃铛式报警。Church_bell 偏大教堂钟,抽听注意"),
    ("alarm_clock", ["Alarm_clock"], "闹钟专有标签,很干净"),
    ("any_audioset_class_playback", [],
     "音响播放任意内容的占位类,不单独收,从音乐和人声两类复用"),
    ("bathtub_filling_washing", ["Bathtub_(filling_or_washing)", "Fill_(with_liquid)"],
     "浴缸放水。Fill 是通用注水,可能混进烧水壶"),
    ("blender", ["Blender"], "料理机,专有标签"),
    ("busy_signal", ["Busy_signal", "Dial_tone"],
     "电话忙音。单标签池里极少,大概率要去多标签池补"),
    ("buzzer", ["Buzzer", "Buzz"], "蜂鸣器。单标签池里极少,需要补"),
    ("cat_meow", ["Meow", "Cat"], "猫叫。Cat 里可能混着呼噜声,抽听剔除"),
    ("cellphone_vibration_alert", ["Cellphone_buzz_and_vibrating_alert"],
     "手机震动,专有标签"),
    ("chime", ["Chime", "Wind_chime", "Tubular_bells", "Jingle_bell"],
     "风铃和管钟这类清脆的响声"),
    ("clock_tick", ["Tick-tock", "Tick", "Clock"],
     "钟表滴答。Clock 里可能有整点报时,抽听"),
    ("crackle", ["Crackle", "Crack"],
     "噼啪声。Crack 偏断裂的单响,和持续的噼啪不一样,注意区分"),
    ("dial_tone", ["Dial_tone"], "拨号音。单标签池只有个位数,需要补"),
    ("ding_dong", ["Ding-dong", "Doorbell"],
     "门铃的叮咚。Ding-dong 只有 1 条,主要靠 Doorbell 里挑"),
    ("dog_bark", ["Bark", "Dog"], "狗叫。Dog 里可能有呜咽和喘气,抽听剔除"),
    ("doorbell", ["Doorbell", "Ding-dong"], "门铃总类"),
    ("doorbell_chime", ["Doorbell", "Chime"], "带旋律的门铃"),
    ("drip", ["Drip", "Trickle_and_dribble", "Raindrop"],
     "滴水。Raindrop 是户外雨滴,场景不同,谨慎用"),
    ("fire", ["Fire"], "火焰燃烧声,专有标签而且量大"),
    ("fire_alarm", ["Fire_alarm", "Alarm"],
     "火警。专有标签只有 6 条,其余得从 Alarm 里抽听挑"),
    ("gurgling", ["Gurgling", "Slosh"], "咕嘟声,下水或沸腾"),
    ("microwave_beep", ["Microwave_oven"],
     "微波炉提示音。和运行嗡声同一个标签,必须靠抽听把两者分开"),
    ("microwave_hum", ["Microwave_oven"], "微波炉运行嗡声,同上,需要人工分"),
    ("music_playback",
     ["Piano", "Guitar", "Acoustic_guitar", "Violin_and_fiddle", "Flute", "Trumpet"],
     "音响放音乐。乐器类样本很多,随便挑就够"),
    ("phone_ring", ["Telephone", "Ringtone", "Telephone_bell_ringing"], "电话铃"),
    ("printer", ["Printer"], "打印机,专有标签"),
    ("ringtone", ["Ringtone"], "手机铃声,专有标签"),
    ("sink_filling_washing", ["Sink_(filling_or_washing)"], "水槽放水洗涤,专有标签"),
    ("smoke_alarm", ["Fire_alarm", "Alarm", "Buzzer"],
     "烟雾报警。没有专有标签,靠抽听从 Alarm 里挑尖锐间断的那种"),
    ("speech_playback",
     ["Male_speech_and_man_speaking", "Female_speech_and_woman_speaking", "Speech",
      "Conversation"],
     "音响放人声。新方案改用 VCTK 语音库,这里只当备份"),
    ("telephone", ["Telephone"], "电话类的总声"),
    ("telephone_bell_ringing", ["Telephone_bell_ringing", "Bell"], "老式电话的机械铃"),
    ("telephone_dialing_dtmf", ["Telephone_dialing_and_DTMF"],
     "按键拨号音。单标签池只有 7 条,需要补"),
    ("toilet_flush", ["Toilet_flush"], "马桶冲水,专有标签而且量大"),
    ("water_tap_faucet", ["Water_tap_and_faucet"], "水龙头,专有标签"),
]

TARGET = 20
entries = []
for cls, labels, note in MAP:
    avail = sum(counts.get(label, 0) for label in labels)
    entries.append({
        "event_class": cls,
        "fsd50k_labels": labels,
        "available_single_label_clips": avail,
        "target_clips": TARGET,
        "enough": bool(avail >= TARGET or not labels),
        "note_zh": note,
    })

doc = {
    "schema": "avengine_sound_harvest_map_v1",
    "purpose_zh": (
        "把我们的事件类映射到 FSD50K 的标签,供批量采集脚本使用。"
        "映射由人判断,条数由脚本实算。开始采集前请同学 B 逐行核对。"
    ),
    "source_pool": str(CSV),
    "source_pool_note_zh": (
        "FSD50K 里只挂一个标签的干净子集,避免一条录音里混进别的声音"
    ),
    "audio_root": "/data/datasets/omniaudio/source_data/FSD50K",
    "target_clips_per_class": TARGET,
    "review_status": "draft_pending_review",
    "entries": entries,
}
out = Path(__file__).resolve().parents[2] / "examples/assets/sound_harvest_map_v1.json"
out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

thin = [e for e in entries if not e["enough"]]
print("共 %d 类;单标签池就够用(>=%d 条)的有 %d 类" % (
    len(entries), TARGET, len(entries) - len(thin)))
print("不够的 %d 类,要么去多标签池补,要么把目标降下来:" % len(thin))
for e in thin:
    print("   %-28s 现有 %3d 条   <- %s" % (
        e["event_class"], e["available_single_label_clips"],
        "、".join(e["fsd50k_labels"]) or "(占位类,不单独收)"))
print("\n表已写到", out)
