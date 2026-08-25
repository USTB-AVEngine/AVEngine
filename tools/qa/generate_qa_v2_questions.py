#!/usr/bin/env python3
"""QA v2 question generation for a constraint-driven batch.

Usage:
  generate_qa_v2_questions.py --inputs-root DIR --output DIR \
      [--programs-dir DIR] [--registry PATH]

Works for pilot48 (program derived from the pair combo) and for batch2d+
(spec.json carries program_id, including *_bfirst order variants).

Truth chain is the constraint-driven design itself (owner-approved reverse
fitting): per-point spec (assets/motion), authored timeline (positions), and
the executed audio program (event windows). No fact-table compile needed.

Types (same-species pairs -> the species word leaks nothing):
  T2-ATTR   R-A->Q-V  who sounded first -> appearance attribute
  TA-MOTION R-A->Q-V  who sounded first -> moving or still (motion-diff points)
  T4-SIDE   R-A->Q-S  first sound from left or right (image side)
  T7-DURING R-T->Q-V  during the second sound, was its emitter moving
  T9-CLOSER R-V->Q-S  which appearance is closer at first-sound frame

Gates by construction plus per-question checks: onset margin (program), motion
differential (spec), image-side margin (|bearing| cross >= 0.1 normalized),
distance ratio >= 1.2, referent in-frame not asserted (visibility pass later).
Left/right sign calibrated against two rendered-frame ground truths from the
20260823 smoke (source1 left c<0, source2 right c>0). research_only.
"""
import argparse
import json, math, os, sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
_ap = argparse.ArgumentParser()
_ap.add_argument("--inputs-root", required=True)
_ap.add_argument("--output", required=True)
_ap.add_argument("--programs-dir",
                 default=str(REPOSITORY / "examples/dataset/current_apartment/audio_programs/qa_v2"))
_ap.add_argument("--registry",
                 default=str(REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json"))
_args = _ap.parse_args()
INPUTS = _args.inputs_root
PROGRAMS = _args.programs_dir
REG = _args.registry
OUT = _args.output

CAM = (-70.0, 65.0)
CAM_YAW_DEG = -145.0
SIDE_MARGIN = 0.1
DIST_RATIO_MIN = 1.2

if os.path.exists(OUT):
    print(json.dumps({"error": "output exists", "out": OUT}))
    sys.exit(2)

reg = json.load(open(REG))
ASSETS = {a["asset_id"]: a for a in reg["assets"]}


def appearance(asset_id):
    rec = ASSETS[asset_id]
    attrs = rec.get("realized_attributes", {})
    if "top_color" in attrs:
        color = attrs["top_color"]
        zh = {"blue": "蓝色", "green": "绿色", "burgundy": "酒红色"}[color]
        return {"kind": "top_color", "value": color,
                "en": f"a {color} top", "zh": f"{zh}上衣",
                "label_en": color, "label_zh": zh}
    breed = rec["identity"].get("breed_id") or rec["asset_id"]
    coat = {"border_collie": ("black_white", "black and white", "黑白色"),
            "labrador_retriever": ("yellow", "yellow", "黄色")}[breed]
    return {"kind": "coat", "value": coat[0], "en": f"a {coat[1]} coat",
            "zh": f"{coat[2]}毛", "label_en": coat[1], "label_zh": coat[2]}


def side_of(p):
    theta = math.radians(CAM_YAW_DEG)
    fx, fy = math.cos(theta), math.sin(theta)
    dx, dy = p[0] - CAM[0], p[1] - CAM[1]
    norm = math.hypot(dx, dy)
    c = (fx * dy - fy * dx) / max(norm, 1e-9)
    return ("right" if c > 0 else "left"), abs(c)


def combo_program(spec):
    if spec.get("program_id"):
        return spec["program_id"]
    def short(aid):
        a = ASSETS[aid]
        if "top_color" in a.get("realized_attributes", {}):
            return a["realized_attributes"]["top_color"]
        return {"border_collie": "collie", "labrador_retriever": "labrador"}[a["identity"]["breed_id"]]
    s1, s2 = short(spec["source1_asset"]), short(spec["source2_asset"])
    kind = "two_human" if spec["pair_kind"] == "human" else "two_dog"
    return f"qa_v2_{kind}_{s1}_{s2}_turn_taking_v1"


def actor_positions(timeline, frame):
    states = timeline["frames"][frame]["actor_states"]
    out = {}
    for st in states:
        key = st.get("actor_id") or st.get("source_slot_id")
        slot = "source1" if "1" in str(key) else "source2"
        out[slot] = st["translation_ue_cm"][:2]
    return out


def mover_slots(motion):
    return {"s1_moving_s2_static": {"source1"}, "s1_static_s2_moving": {"source2"},
            "both_moving": {"source1", "source2"}}[motion]


questions, refused = [], []


def emit(row, ok, reason=None):
    (questions if ok else refused).append(row if ok else dict(row, refusal_reason=reason))


for pid in sorted(os.listdir(INPUTS)):
    pdir = os.path.join(INPUTS, pid)
    spec_path = os.path.join(pdir, "spec.json")
    if not os.path.isfile(spec_path):
        continue
    spec = json.load(open(spec_path))
    timeline = json.load(open(os.path.join(pdir, "timeline.json")))
    prog = json.load(open(os.path.join(PROGRAMS, combo_program(spec) + ".json")))
    events = sorted(prog["events"], key=lambda e: e["start_sample"])
    ep_to_slot = {prog["candidate_source_endpoint_ids"][0]: "source1",
                  prog["candidate_source_endpoint_ids"][1]: "source2"}
    first = events[0]
    second = next(e for e in events if ep_to_slot[e["source_endpoint_id"]]
                  != ep_to_slot[first["source_endpoint_id"]])
    slot_a = ep_to_slot[first["source_endpoint_id"]]
    slot_b = "source2" if slot_a == "source1" else "source1"
    f_first = min(74, round(first["start_sample"] * 75 / 80000))
    f_second = min(74, round(second["start_sample"] * 75 / 80000))
    movers = mover_slots(spec["motion_case"])
    ap = {s: appearance(spec[f"{s}_asset"]) for s in ("source1", "source2")}
    noun_en = "person" if spec["pair_kind"] == "human" else "dog"
    noun_zh = "人" if spec["pair_kind"] == "human" else "狗"
    verb_en = "spoke" if spec["pair_kind"] == "human" else "barked"
    verb_zh = "说话" if spec["pair_kind"] == "human" else "叫"
    base = {"point_id": pid, "pair_kind": spec["pair_kind"],
            "motion_case": spec["motion_case"], "twin_of": spec["twin_of"],
            "offscreen_candidate": spec["offscreen_candidate"],
            "first_slot": slot_a, "first_frame": f_first}

    # T2-ATTR
    opts = sorted({ap["source1"]["label_en"], ap["source2"]["label_en"]})
    emit(dict(base, type_id="QV2-T2-ATTR",
              question_en=f"Which {noun_en} {verb_en} first — the one with {ap['source1']['en']} or the one with {ap['source2']['en']}?",
              question_zh=f"先{verb_zh}的是穿戴{ap['source1']['zh']}的那只/位，还是{ap['source2']['zh']}的？"
              if spec["pair_kind"] == "dog" else
              f"先{verb_zh}的那个{noun_zh}穿的是{ap['source1']['zh']}还是{ap['source2']['zh']}？",
              options=opts, answer=ap[slot_a]["label_en"],
              modality_expectation="dual_required"), True)

    # TA-MOTION (needs motion differential)
    if (slot_a in movers) != (slot_b in movers):
        emit(dict(base, type_id="QV2-TA-MOTION",
                  question_en=f"At the moment of the first {('speech' if noun_en=='person' else 'bark')}, was the {noun_en} making it moving or staying still?",
                  question_zh=f"第一声{verb_zh}响起时，发出它的那个{noun_zh}在走动还是静止？",
                  options=["moving", "staying still"],
                  answer="moving" if slot_a in movers else "staying still",
                  modality_expectation="dual_required"), True)

    # T4-SIDE
    pos = actor_positions(timeline, f_first)
    side_a, mag_a = side_of(pos[slot_a])
    side_b, mag_b = side_of(pos[slot_b])
    row = dict(base, type_id="QV2-T4-SIDE",
               question_en=f"Did the first {('speech' if noun_en=='person' else 'bark')} come from the left or the right side of the view?",
               question_zh=f"第一声{verb_zh}来自画面的左侧还是右侧？",
               options=["left", "right"], answer=side_a,
               modality_expectation="audio_sufficient_control",
               side_margins=[round(mag_a, 3), round(mag_b, 3)])
    if mag_a < SIDE_MARGIN:
        emit(row, False, "G-side-margin")
    elif side_a == side_b:
        emit(row, False, "G-differential-same-side")
    else:
        emit(row, True)

    # T7-DURING
    if (slot_b in movers) != (slot_a in movers) or spec["motion_case"] == "both_moving":
        ans = "moving" if slot_b in movers else "staying still"
        ok = (slot_a in movers) != (slot_b in movers)
        emit(dict(base, type_id="QV2-T7-DURING",
                  question_en=f"During the second {('speech' if noun_en=='person' else 'bark')}, was the {noun_en} making it moving or staying still?",
                  question_zh=f"第二声{verb_zh}期间，发出它的那个{noun_zh}在走动还是静止？",
                  options=["moving", "staying still"], answer=ans,
                  modality_expectation="dual_required"),
             ok, None if ok else "G-differential-motion")

    # T9-CLOSER
    d = {s: math.hypot(pos[s][0] - CAM[0], pos[s][1] - CAM[1]) for s in pos}
    ratio = max(d.values()) / max(min(d.values()), 1e-9)
    closer = min(d, key=d.get)
    row = dict(base, type_id="QV2-T9-CLOSER",
               question_en=f"At the first {('speech' if noun_en=='person' else 'bark')}, which {noun_en} is closer to the camera — the one with {ap['source1']['en']} or the one with {ap['source2']['en']}?",
               question_zh=f"第一声{verb_zh}时，离相机更近的是{ap['source1']['zh']}的那个还是{ap['source2']['zh']}的？",
               options=opts, answer=ap[closer]["label_en"],
               modality_expectation="vision_dominant_control",
               distance_ratio=round(ratio, 2))
    emit(row, ratio >= DIST_RATIO_MIN, None if ratio >= DIST_RATIO_MIN else "G-distance-ratio")

# ---- stats + write ----
os.makedirs(OUT)
from collections import Counter
by_type = Counter(q["type_id"] for q in questions)
ans_by_type = {}
for q in questions:
    ans_by_type.setdefault(q["type_id"], Counter())[str(q["answer"])] += 1
doc = {
    "schema": "avengine_qa_v2_questions_v1",
    "status": "research_candidate",
    "qualification_claim": False,
    "claim_boundary": ("Constraint-driven v2 questions compiled from the pilot48 design, "
                       "authored timelines and executed audio programs; visibility/pixel "
                       "gates pending; no dataset admission."),
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "counts": {"accepted": len(questions), "refused": len(refused),
               "by_type": dict(by_type),
               "answers_by_type": {k: dict(v) for k, v in ans_by_type.items()}},
    "gate_params": {"side_margin": SIDE_MARGIN, "distance_ratio_min": DIST_RATIO_MIN},
    "questions": questions,
    "refused": refused,
}
with open(os.path.join(OUT, "questions.json"), "w") as f:
    json.dump(doc, f, ensure_ascii=False, indent=1)
print(json.dumps({"out": OUT, "counts": doc["counts"]}, ensure_ascii=False, indent=1))
