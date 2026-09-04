import json, glob, os
from collections import Counter

DESIGN = "/data/jzy/tmp/qa_v3_design_pilot01"
QROOT = "/data/jzy/tmp/qa_v3_questions_pilot01_v2"
PROGRAMS = DESIGN + "/programs"
TICKS_PER_SAMPLE, TICKS_PER_FRAME, SR = 3, 3200, 16000
COAT = {"generated_border_collie_black_white_medium_standard_adult_research_v1":
        "black-and-white",
        "generated_labrador_yellow_medium_standard_adult_research_v1": "yellow"}


def sector(deg):
    if -45.0 <= deg < 45.0: return "front"
    if 45.0 <= deg < 135.0: return "right"
    if -135.0 <= deg < -45.0: return "left"
    return "back"


def azimuth(cam_xy, yaw, xy):
    import math
    d = math.degrees(math.atan2(xy[1] - cam_xy[1], xy[0] - cam_xy[0])) - yaw
    return (d + 180.0) % 360.0 - 180.0


def slot_of(program, event):
    eps = program["candidate_source_endpoint_ids"]
    return "source1" if event["source_endpoint_id"] == eps[0] else "source2"


def load_pair(pid):
    mains = sorted(glob.glob(f"{PROGRAMS}/qa_v3_*_{pid}_rand_v1.json"))
    gates = sorted(glob.glob(f"{PROGRAMS}/qa_v3_*_{pid}_gateA_rand_v1.json"))
    if len(mains) != 1 or len(gates) != 1:
        return None, None
    return json.load(open(mains[0])), json.load(open(gates[0]))


def frame_span(event):
    t0 = event["start_sample"] * TICKS_PER_SAMPLE
    t1 = event["end_sample_exclusive"] * TICKS_PER_SAMPLE
    return t0 // TICKS_PER_FRAME, -(-t1 // TICKS_PER_FRAME)


structure = Counter()
flips = {c: Counter() for c in ("card1", "card7", "card8", "card9")}
examples = {}

facts = {os.path.basename(f)[len("facts_"):-len(".jsonl")]:
         [json.loads(l) for l in open(f)]
         for f in sorted(glob.glob(QROOT + "/facts_*.jsonl"))}
by_point = {}
for card, rows in facts.items():
    for row in rows:
        by_point.setdefault(row["point_id"], {}).setdefault(card, []).append(row)

for pid, cards in sorted(by_point.items()):
    main, gate = load_pair(pid)
    if main is None:
        continue
    # 结构:非目标变量是否保持
    structure["pairs"] += 1
    structure["event_count_same"] += int(
        len(main["events"]) == len(gate["events"]))
    structure["event_times_same"] += int(
        [e["start_sample"] for e in main["events"]]
        == [e["start_sample"] for e in gate["events"]])
    structure["candidates_same"] += int(
        main["candidate_source_endpoint_ids"]
        == gate["candidate_source_endpoint_ids"])
    structure["sound_asset_same"] += int(
        {e["sound_asset_id"] for e in main["events"]}
        == {e["sound_asset_id"] for e in gate["events"]})
    structure["slot_sequence_changed"] += int(
        [slot_of(main, e) for e in main["events"]]
        != [slot_of(gate, e) for e in gate["events"]])

    timeline = json.loads(open(f"{DESIGN}/{pid}/timeline.json").read())
    cam = timeline["frames"][74]["camera"]
    cam_xy = (cam["translation_ue_cm"][0], cam["translation_ue_cm"][1])
    yaw = cam["yaw_ue_deg"]
    pos = {s["source_slot_id"]: (s["translation_ue_cm"][0],
                                 s["translation_ue_cm"][1])
           for s in timeline["frames"][74]["actor_states"]}

    # 卡①:锚定者(最后一声)归属是否改变,片尾方位答案是否翻转
    if "card1" in cards:
        f = cards["card1"][0]
        m_anchor = slot_of(main, sorted(main["events"],
                                        key=lambda e: e["start_sample"])[-1])
        g_anchor = slot_of(gate, sorted(gate["events"],
                                        key=lambda e: e["start_sample"])[-1])
        m_deg = azimuth(cam_xy, yaw, pos[m_anchor])
        g_deg = azimuth(cam_xy, yaw, pos[g_anchor])
        flips["card1"]["anchor_actor_changed"] += int(m_anchor != g_anchor)
        flips["card1"]["sector_gold_flipped"] += int(
            sector(m_deg) != sector(g_deg))
        band = f.get("azimuth_band") or {}
        if band.get("truth_option"):
            flips["card1"]["band_gold_flipped"] += int(
                abs(m_deg - g_deg) > 1e-9 and
                band.get("band_index") is not None)
        flips["card1"]["total"] += 1
        examples.setdefault("card1", (pid, m_anchor, g_anchor,
                                      round(m_deg, 1), round(g_deg, 1),
                                      sector(m_deg), sector(g_deg)))

    # 卡⑦:查询时刻发声者是否改变,毛色答案是否翻转
    for f in cards.get("card7_main", []) + cards.get("card7", []):
        frame = f["query_time"]["frame"]
        m_call = [slot_of(main, e) for e in main["events"]
                  if frame_span(e)[0] <= frame < frame_span(e)[1]]
        g_call = [slot_of(gate, e) for e in gate["events"]
                  if frame_span(e)[0] <= frame < frame_span(e)[1]]
        coat = f["slot_coat"]
        m_gold = coat[m_call[0]] if len(m_call) == 1 else "not_single"
        g_gold = coat[g_call[0]] if len(g_call) == 1 else "not_single"
        flips["card7"]["total"] += 1
        flips["card7"]["caller_changed"] += int(m_call != g_call)
        flips["card7"]["gold_flipped"] += int(m_gold != g_gold
                                             and "not_single" not in
                                             (m_gold, g_gold))
        examples.setdefault("card7", (pid, frame, m_gold, g_gold))

    # 卡⑧:目标首叫时间是否移动、是否跨带
    for f in cards.get("card8", []):
        slot = f["target_slot"]
        def first_onset(program):
            for e in sorted(program["events"], key=lambda x: x["start_sample"]):
                if slot_of(program, e) == slot:
                    return e["start_sample"] / SR
            return None
        m_on, g_on = first_onset(main), first_onset(gate)
        edges = (f["mcq"] or {}).get("options_space")
        flips["card8"]["total"] += 1
        flips["card8"]["onset_moved"] += int(m_on is not None and g_on is not None
                                             and abs(m_on - g_on) > 1e-9)
        if f["mcq"]:
            band_edges = [0.0, 0.65, 1.3, 1.95, 2.6]
            def band(v):
                for i in range(len(band_edges) - 1):
                    if band_edges[i] <= v < band_edges[i + 1]:
                        return i
                return None
            flips["card8"]["band_gold_flipped"] += int(
                m_on is not None and g_on is not None
                and band(m_on) != band(g_on))
        examples.setdefault("card8", (pid, slot, m_on, g_on))

    # 卡⑨:先叫者是否改变
    for f in cards.get("card9", []):
        def first_slot(program):
            e = sorted(program["events"], key=lambda x: x["start_sample"])[0]
            return slot_of(program, e)
        m_first, g_first = first_slot(main), first_slot(gate)
        coat = f["slot_coat"]
        flips["card9"]["total"] += 1
        flips["card9"]["first_caller_changed"] += int(m_first != g_first)
        flips["card9"]["gold_flipped"] += int(coat[m_first] != coat[g_first])
        examples.setdefault("card9", (pid, m_first, g_first,
                                      coat[m_first], coat[g_first]))

print("== Gate A 结构核验(非目标变量应保持) ==")
n = structure["pairs"]
for k in ("event_count_same", "event_times_same", "candidates_same",
          "sound_asset_same", "slot_sequence_changed"):
    print(f"  {k}: {structure[k]}/{n}")
print()
print("== 逐题金标翻转 ==")
for card, c in flips.items():
    print(f"  {card}: {dict(c)}")
print()
print("== 样例 ==")
for card, ex in examples.items():
    print(f"  {card}: {ex}")
