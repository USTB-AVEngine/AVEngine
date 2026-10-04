"""Four-member selection from actual eligible questions and media.

For the current v7-S table, 5 bound items per 14-question quartet window
give a planned aggregate share above 20%. Selection still fails when visual
candidates are missing. Neither four-member answers nor budgets invent facts.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import hashlib
import math
from pathlib import Path

from avengine.qa.v7_families import deduplicate_questions, select_questions

MEMBERS=("v0_a0","v0_a1","v1_a0","v1_a1")

def binding_reservation(seed_rows, *, split, questions_per_window=14, minimum_fraction=.20):
    rows=[r for r in seed_rows if r["split"]==split]
    all_windows=sum(r["window_count"]*len(r["members"]) for r in rows)
    quartet_windows=sum(r["window_count"]*4 for r in rows if set(r["members"])==set(MEMBERS))
    if not quartet_windows:raise ValueError("no four-member windows for binding quota")
    reserve=math.ceil(minimum_fraction*all_windows*questions_per_window/quartet_windows)
    return {"all_member_windows":all_windows,"quartet_member_windows":quartet_windows,
            "binding_items_per_quartet_window":reserve,
            "planned_binding_fraction":reserve*quartet_windows/(all_windows*questions_per_window),
            "status":"planned_unverified_until_questions_and_media_pass"}

def eligible_binding_keys(public_by_member, private_by_member):
    pools={};truths={}
    for member in MEMBERS:
        p,g,_=deduplicate_questions(public_by_member[member],private_by_member[member])
        gold={r["question_id"]:r for r in g}
        pools[member]={}
        truths[member]={}
        for row in p:
            if row["form"]!="open" or row["modality"]!="audiovisual":continue
            key=(row["protocol_panel"],row["qa_id"],row["question_zh"])
            pools[member][key]=row
            truths[member][key]=gold[row["question_id"]]["canonical_answer"]
    common=set.intersection(*(set(pools[m]) for m in MEMBERS))
    result=[]
    for key in sorted(common):
        a,b,c,d=(truths[m][key] for m in MEMBERS)
        if a==d and b==c and a!=b:result.append(key)
    return result,pools

def verify_group_media(public_by_member):
    digests={}
    for member in MEMBERS:
        rows=public_by_member[member]
        if not rows:raise ValueError("empty quartet member")
        media=rows[0]["media"]
        if any(r["media"]!=media for r in rows):raise ValueError("member has conflicting media")
        digests[member]={k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in media.items()}
    for visual in ("v0","v1"):
        if digests[visual+"_a0"].get("video")!=digests[visual+"_a1"].get("video"):
            raise ValueError("audio swap changed video bytes")
    for audio in ("a0","a1"):
        if digests["v0_"+audio].get("audio_foa")!=digests["v1_"+audio].get("audio_foa"):
            raise ValueError("appearance swap changed audio bytes")
    if digests["v0_a0"].get("audio_foa")==digests["v0_a1"].get("audio_foa"):
        raise ValueError("declared audio counterfactual has identical audio bytes")
    return digests

def select_counterfactual_group(public_by_member, private_by_member, allocation_by_member, *,
                                binding_items_per_window=5, quota_index=0, family_counts=None):
    if set(public_by_member)!=set(MEMBERS) or set(private_by_member)!=set(MEMBERS):
        raise ValueError("complete four-member group required")
    media=verify_group_media(public_by_member)
    eligible,pools=eligible_binding_keys(public_by_member,private_by_member)
    if len(eligible)<binding_items_per_window:
        raise ValueError(f"only {len(eligible)} measured binding questions; need {binding_items_per_window}")
    counts=Counter(family_counts or {})
    chosen=sorted(eligible,key=lambda k:(counts[k[1]],k))[:binding_items_per_window]
    selected={};answers={}
    for member in MEMBERS:
        p,g,_=deduplicate_questions(public_by_member[member],private_by_member[member])
        reserved=[pools[member][key] for key in chosen]
        allocation=deepcopy(allocation_by_member[member])
        allocation["audiovisual"]-=len(reserved)
        if allocation["audiovisual"]<0:raise ValueError("binding reservation exceeds AV allocation")
        logicals={r["logical_question_id"] for r in reserved}
        rest=[r for r in p if r["logical_question_id"] not in logicals]
        q,a=select_questions(rest,g,allocation,quota_index=quota_index,family_counts=counts)
        gold={r["question_id"]:r for r in g}
        selected[member]=reserved+q;answers[member]=[gold[r["question_id"]] for r in reserved]+a
    return selected,answers,{"binding_keys":chosen,"binding_items_per_member":len(chosen),
                             "native_media_sha256":media,"status":"pass"}
