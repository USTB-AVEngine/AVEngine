import json, glob, math
DESIGN="/data/jzy/tmp/qa_v3_design_pilot01"; PROGRAMS=DESIGN+"/programs"
QROOT="/data/jzy/tmp/qa_v3_questions_pilot01_v2"
params=json.load(open("/data/jzy/tmp/qa_v3_params_pilot_mcq.json"))
THETA_HALF=float(params["THETA_HALF"]); T_HALF=float(params["T_HALF"])
SR=16000
def azimuth(cam,yaw,xy):
    d=math.degrees(math.atan2(xy[1]-cam[1], xy[0]-cam[0]))-yaw
    return (d+180.0)%360.0-180.0
def circ(a,b): return abs((a-b+180.0)%360.0-180.0)
def sector(d):
    if -45<=d<45: return "front"
    if 45<=d<135: return "right"
    if -135<=d<-45: return "left"
    return "back"
def band3(d,e=(-52.5,-17.5,17.5,52.5)):
    for i in range(len(e)-1):
        if e[i]<=d<e[i+1]: return i
    return None
def slot_of(p,ev):
    eps=p["candidate_source_endpoint_ids"]
    return "source1" if ev["source_endpoint_id"]==eps[0] else "source2"
def pair(pid):
    return (json.load(open(glob.glob(f"{PROGRAMS}/qa_v3_*_{pid}_rand_v1.json")[0])),
            json.load(open(glob.glob(f"{PROGRAMS}/qa_v3_*_{pid}_gateA_rand_v1.json")[0])))

print("== 卡① 按形式拆分 (THETA_HALF=%.0f, 阈值 2xTHETA_HALF=%.0f 度) ==" % (THETA_HALF, 2*THETA_HALF))
c1=[json.loads(l) for l in open(QROOT+"/facts_card1.jsonl")]
anchor_changed=num_changed=mcq_flip=open_disjoint=band_flip=0
gaps=[]
for f in c1:
    pid=f["point_id"]; m,g=pair(pid)
    tl=json.load(open(f"{DESIGN}/{pid}/timeline.json"))
    cam=tl["frames"][74]["camera"]; c=(cam["translation_ue_cm"][0],cam["translation_ue_cm"][1]); yaw=cam["yaw_ue_deg"]
    pos={s["source_slot_id"]:(s["translation_ue_cm"][0],s["translation_ue_cm"][1]) for s in tl["frames"][74]["actor_states"]}
    ma=slot_of(m,sorted(m["events"],key=lambda e:e["start_sample"])[-1])
    ga=slot_of(g,sorted(g["events"],key=lambda e:e["start_sample"])[-1])
    md,gd=azimuth(c,yaw,pos[ma]),azimuth(c,yaw,pos[ga])
    anchor_changed+=int(ma!=ga); num_changed+=int(abs(md-gd)>1e-9)
    mcq_flip+=int(sector(md)!=sector(gd))
    d=circ(md,gd); gaps.append(round(d,1))
    open_disjoint+=int(d>2*THETA_HALF)
    band_flip+=int(band3(md)!=band3(gd))
n=len(c1)
print("  [run01 四扇区 MCQ]  anchor_actor_changed %d/%d | numeric_truth_changed %d/%d | mcq_gold_flipped %d/%d" % (anchor_changed,n,num_changed,n,mcq_flip,n))
print("     -> 答案空间退化,Gate A 语义对 MCQ 无效")
print("  [run01 开放数值版]  numeric_truth_changed %d/%d | open_gold_regions_disjoint %d/%d | open_gateA_structurally_valid %d/%d" % (num_changed,n,open_disjoint,n,open_disjoint,n))
print("     -> 其余 %d 条数值虽不同,但存在中间答案可能同时落进两边容差区" % (n-open_disjoint))
print("     角距分布 min/中位/max: %.1f / %.1f / %.1f" % (min(gaps), sorted(gaps)[len(gaps)//2], max(gaps)))
print("  [事后诊断,非 run01 正式结果] counterfactual_three_band_gold_flipped %d/%d" % (band_flip,n))

print()
print("== 卡⑧ 按形式拆分 (T_HALF=%.1f s) ==" % T_HALF)
c8=[json.loads(l) for l in open(QROOT+"/facts_card8.jsonl")]
band_f=open_sep=moved=0
diffs=[]
for f in c8:
    pid=f["point_id"]; slot=f["target_slot"]; m,g=pair(pid)
    def first(p):
        for e in sorted(p["events"],key=lambda x:x["start_sample"]):
            if slot_of(p,e)==slot: return e["start_sample"]/SR
    mt,gt=first(m),first(g)
    moved+=int(abs(mt-gt)>1e-9)
    diffs.append(round(abs(mt-gt),3))
    open_sep+=int(abs(mt-gt)>T_HALF)
    edges=[0.0,0.65,1.3,1.95,2.6]
    def b(v):
        for i in range(len(edges)-1):
            if edges[i]<=v<edges[i+1]: return i
    band_f+=int(b(mt)!=b(gt))
n8=len(c8)
print("  [MCQ] mcq_band_flipped %d/%d" % (band_f,n8))
print("  [Open] numeric_time_changed %d/%d | abs(main-gateA) > T_HALF: %d/%d" % (moved,n8,open_sep,n8))
print("     时间差 min/中位/max: %.3f / %.3f / %.3f s" % (min(diffs), sorted(diffs)[len(diffs)//2], max(diffs)))
