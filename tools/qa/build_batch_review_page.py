#!/usr/bin/env python3
"""Build a self-contained lazy-loading review page for one QA v2 batch.

Emits into --output:
  index.html          single-file viewer (vanilla JS, no CDN)
  data/points.json    lightweight index (one row per point)
  data/<pid>.json     per-point detail, fetched on demand
  clips/<pid>.mp4     muxed review clip (only when audio exists; skip if present)

Serve read-only from the output directory, e.g.:
  cd <output> && python3 -m http.server 8901
  ssh -L 8901:127.0.0.1:8901 <server>   ->  http://127.0.0.1:8901/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs-root", required=True, type=Path)
    p.add_argument("--captures-root", required=True, type=Path)
    p.add_argument("--audio-root", type=Path)
    p.add_argument("--questions", type=Path, help="questions.json for the batch")
    p.add_argument("--batch-name", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--mux-clips", action="store_true")
    return p.parse_args()


def receipt_summary(receipt: dict) -> dict:
    cap = receipt.get("capture", {})
    root = cap.get("root_readback_summary", {})
    anim = cap.get("animation_readback_summary", {})
    return {
        "frames": cap.get("completed_frame_count"),
        "max_pos_err_cm": max(
            (v.get("maximum_position_error_cm", 0.0) for v in root.values()), default=None
        ),
        "max_yaw_err_deg": max(
            (v.get("maximum_yaw_error_deg", 0.0) for v in root.values()), default=None
        ),
        "anim_status": anim.get("status"),
        "research_only": receipt.get("research_only"),
        "episode_counted": receipt.get("episode_counted"),
    }


def mux_clip(captures_root: Path, audio_root: Path, pid: str, out_path: Path) -> bool:
    mixture = audio_root / pid / "audio/binaural/mixture.wav"
    if out_path.exists():
        return True
    if not mixture.is_file():
        return False
    tool = REPOSITORY / "tools/m5/build_current_mp3d_dynamic_review_clip.py"
    proc = subprocess.run(
        [sys.executable, str(tool),
         "--visual-capture-dir", str(captures_root / pid),
         "--mixture-wav", str(mixture),
         "--channel-order", "bgr",
         "--output", str(out_path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 and out_path.exists():
        out_path.unlink()
    base = out_path.with_name(out_path.stem + ".base.mp4")
    if base.exists():
        base.unlink()
    return proc.returncode == 0 and out_path.is_file()


HTML = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>__BATCH__ · QA v2 review</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{color-scheme:light dark;font-family:system-ui,sans-serif}
body{margin:0;background:#111;color:#ddd}
header{position:sticky;top:0;background:#1b1b1f;padding:10px 16px;border-bottom:1px solid #333;z-index:5}
h1{font-size:16px;margin:0 0 8px}
.bars{display:flex;gap:14px;flex-wrap:wrap;font-size:12px}
.bar{min-width:180px}.bar .track{background:#333;border-radius:4px;height:8px;overflow:hidden}
.bar .fill{background:#4c8dff;height:8px;width:0}
.filters{margin-top:8px;display:flex;gap:6px;flex-wrap:wrap}
.filters button{background:#2a2a30;color:#ccc;border:1px solid #444;border-radius:12px;padding:2px 10px;font-size:12px;cursor:pointer}
.filters button.on{background:#4c8dff;color:#fff;border-color:#4c8dff}
main{padding:10px 16px;max-width:1100px;margin:0 auto}
.point{border:1px solid #333;border-radius:8px;margin:8px 0;background:#1b1b1f}
.row{display:flex;gap:10px;align-items:center;padding:8px 12px;cursor:pointer;flex-wrap:wrap}
.row .pid{font-weight:700;min-width:52px}
.tag{font-size:11px;background:#2a2a30;border-radius:8px;padding:1px 8px;color:#aaa}
.tag.ok{color:#7ad97a}.tag.warn{color:#e8b34c}.tag.miss{color:#e06666}
.detail{border-top:1px solid #2a2a30;padding:10px 14px;display:none}
.detail.open{display:block}
video{width:100%;max-width:640px;border-radius:6px;background:#000}
table{border-collapse:collapse;font-size:12px;margin:8px 0}
td,th{border:1px solid #333;padding:4px 8px;text-align:left}
.q{margin:8px 0;padding:8px 10px;background:#202027;border-radius:6px;font-size:13px}
.q .ans{color:#7ad97a}.q .meta{color:#888;font-size:11px}
#sentinel{height:40px}
#loading{font-size:12px;color:#888;padding:8px}
</style></head><body>
<header>
 <h1>__BATCH__ <span id="counts" style="font-weight:400;color:#999"></span></h1>
 <div class="bars">
  <div class="bar">视觉捕获 <span id="pcapt"></span><div class="track"><div class="fill" id="bcapt"></div></div></div>
  <div class="bar">音频渲染 <span id="paud"></span><div class="track"><div class="fill" id="baud"></div></div></div>
  <div class="bar">成片 <span id="pclip"></span><div class="track"><div class="fill" id="bclip"></div></div></div>
  <div class="bar">题目 <span id="pq"></span><div class="track"><div class="fill" id="bq"></div></div></div>
 </div>
 <div class="filters" id="filters"></div>
</header>
<main><div id="list"></div><div id="loading">加载中…</div><div id="sentinel"></div></main>
<script>
let ALL=[],VIEW=[],shown=0;const PAGE=20;const ROWS={};
const F={pair:null,motion:null,flag:null};
function pct(n,d){return d? (100*n/d).toFixed(1)+"%":"–"}
function bar(id,n,d){document.getElementById("b"+id).style.width=(d?100*n/d:0)+"%";
 document.getElementById("p"+id).textContent=`${n}/${d} (${pct(n,d)})`}
function updateBars(){
 bar("capt",ALL.filter(p=>p.capture).length,ALL.length);
 bar("aud",ALL.filter(p=>p.audio).length,ALL.length);
 bar("clip",ALL.filter(p=>p.clip).length,ALL.length);
 bar("q",ALL.filter(p=>p.nq>0).length,ALL.length);
}
async function refresh(){
 try{
  const j=await(await fetch("data/points.json?t="+Date.now())).json();
  ALL=j.points;updateBars();
  document.getElementById("counts").textContent=`· 更新于 ${new Date().toLocaleTimeString()}`;
  for(const p of ALL){const r=ROWS[p.pid];if(r)setTags(r,p)}
 }catch(e){}
}
async function boot(){
 const r=await fetch("data/points.json");const j=await r.json();ALL=j.points;
 document.getElementById("counts").textContent=`· ${j.batch.generated_at}`;
 updateBars();mkFilters();apply();
 new IntersectionObserver(e=>{if(e[0].isIntersecting)more()}).observe(document.getElementById("sentinel"));
 setInterval(refresh,30000);
}
function mkFilters(){
 const fs=document.getElementById("filters");
 const groups=[["pair",["human","dog"]],["motion",[...new Set(ALL.map(p=>p.motion))]],
  ["flag",["twin","offscreen","no_audio"]]];
 for(const[k,vals] of groups)for(const v of vals){
  const b=document.createElement("button");b.textContent=v;
  b.onclick=()=>{F[k]=F[k]===v?null:v;
   fs.querySelectorAll("button").forEach(x=>x.classList.toggle("on",x.textContent===F.pair||x.textContent===F.motion||x.textContent===F.flag));
   apply()};
  fs.appendChild(b)}
}
function apply(){
 VIEW=ALL.filter(p=>(!F.pair||p.pair===F.pair)&&(!F.motion||p.motion===F.motion)
  &&(!F.flag||(F.flag==="twin"?!!p.twin_of:F.flag==="offscreen"?p.offscreen:!p.audio)));
 shown=0;document.getElementById("list").innerHTML="";more();
}
function more(){
 const slice=VIEW.slice(shown,shown+PAGE);shown+=slice.length;
 for(const p of slice)document.getElementById("list").appendChild(row(p));
 document.getElementById("loading").textContent=shown>=VIEW.length?`全部 ${VIEW.length} 条已加载`:`已加载 ${shown}/${VIEW.length} — 下拉加载更多`;
}
function tag(t,cls){const s=document.createElement("span");s.className="tag "+(cls||"");s.textContent=t;return s}
function setTags(r,p){
 let st=r.querySelector(".status");
 if(!st){st=document.createElement("span");st.className="status";r.append(st)}
 st.innerHTML="";
 st.append(tag(p.capture?"视觉✓":"视觉…",p.capture?"ok":"miss"));
 st.append(tag(p.audio?"音频✓":"音频…",p.audio?"ok":"miss"));
 st.append(tag(p.clip?"成片✓":"成片…",p.clip?"ok":"miss"));
 st.append(tag(p.nq+" 题",p.nq>0?"ok":""));
}
function row(p){
 const d=document.createElement("div");d.className="point";
 const r=document.createElement("div");r.className="row";
 r.append(Object.assign(document.createElement("span"),{className:"pid",textContent:p.pid}));
 r.append(tag(p.pair),tag(p.motion));
 if(p.twin_of)r.append(tag("twin→"+p.twin_of,"warn"));
 if(p.offscreen)r.append(tag("off-screen","warn"));
 setTags(r,p);
 const det=document.createElement("div");det.className="detail";
 r.onclick=()=>{det.classList.toggle("open");if(det.dataset.done!=="1"){det.dataset.done="1";load(p,det)}};
 d.append(r,det);ROWS[p.pid]=r;return d;
}
async function load(p,el){
 el.textContent="加载详情…";
 try{
  const j=await(await fetch(`data/${p.pid}.json`)).json();
  el.innerHTML="";
  if(j.clip){const v=document.createElement("video");v.controls=true;v.preload="none";v.src=j.clip;el.append(v)}
  const t=document.createElement("table");
  const rs=j.receipt||{};
  t.innerHTML=`<tr><th>帧数</th><th>位姿误差cm</th><th>yaw误差°</th><th>动画</th><th>program</th></tr>
  <tr><td>${rs.frames??"–"}</td><td>${rs.max_pos_err_cm??"–"}</td><td>${(rs.max_yaw_err_deg??0).toExponential? (rs.max_yaw_err_deg??0).toExponential(1):rs.max_yaw_err_deg}</td><td>${rs.anim_status??"–"}</td><td>${j.spec.program_id||"(pilot48)"}</td></tr>`;
  el.append(t);
  for(const q of (j.questions||[])){
   const div=document.createElement("div");div.className="q";
   div.innerHTML=`<b>${q.type_id}</b> · ${q.question_zh}<br><span style="color:#9ab">${q.question_en}</span><br>
    <span class="ans">答案：${q.answer}</span> <span class="meta">[${q.options.join(" / ")}] · ${q.modality_expectation}</span>`;
   el.append(div)}
  if(!(j.questions||[]).length)el.append(Object.assign(document.createElement("div"),{className:"q",textContent:"（该点题目待音频批完成后生成）"}));
 }catch(e){el.textContent="加载失败: "+e}
}
boot();
</script></body></html>
"""


def main() -> int:
    args = parse_args()
    out = args.output
    (out / "data").mkdir(parents=True, exist_ok=True)
    clips_dir = out / "clips"
    clips_dir.mkdir(exist_ok=True)

    questions_by_pid: dict[str, list] = {}
    if args.questions and args.questions.is_file():
        qdoc = json.loads(args.questions.read_text(encoding="utf-8"))
        for q in qdoc.get("questions", []):
            questions_by_pid.setdefault(q["point_id"], []).append(q)

    def write_index(points_list):
        index = {
            "batch": {
                "name": args.batch_name,
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            "points": points_list,
        }
        tmp = out / "data/points.json.tmp"
        tmp.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
        tmp.replace(out / "data/points.json")

    # pass 1: skeleton immediately (page usable before any clip exists)
    points, details = [], {}
    for pdir in sorted(args.inputs_root.iterdir()):
        spec_path = pdir / "spec.json"
        if not spec_path.is_file():
            continue
        pid = pdir.name
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        receipt_path = args.captures_root / pid / "research_receipt.json"
        captured = receipt_path.is_file()
        receipt = (
            receipt_summary(json.loads(receipt_path.read_text(encoding="utf-8")))
            if captured else None
        )
        has_audio = bool(
            args.audio_root
            and (args.audio_root / pid / "audio/binaural/mixture.wav").is_file()
        )
        clip_path = clips_dir / f"{pid}.mp4"
        has_clip = clip_path.is_file()
        qs = questions_by_pid.get(pid, [])
        points.append({
            "pid": pid,
            "pair": spec.get("pair_kind"),
            "motion": spec.get("motion_case"),
            "twin_of": spec.get("twin_of"),
            "offscreen": bool(spec.get("offscreen_candidate")),
            "capture": captured,
            "audio": has_audio,
            "clip": has_clip,
            "nq": len(qs),
        })
        details[pid] = {
            "spec": spec, "receipt": receipt, "questions": qs,
            "clip": f"clips/{pid}.mp4" if has_clip else None,
        }
        (out / "data" / f"{pid}.json").write_text(
            json.dumps(details[pid], ensure_ascii=False), encoding="utf-8"
        )
    (out / "index.html").write_text(
        HTML.replace("__BATCH__", args.batch_name), encoding="utf-8"
    )
    write_index(points)

    # pass 2: incremental clip muxing, re-aligning the index after each clip
    if args.mux_clips and args.audio_root:
        for row_entry in points:
            pid = row_entry["pid"]
            if row_entry["clip"] or not (row_entry["capture"] and row_entry["audio"]):
                continue
            if mux_clip(args.captures_root, args.audio_root, pid, clips_dir / f"{pid}.mp4"):
                row_entry["clip"] = True
                details[pid]["clip"] = f"clips/{pid}.mp4"
                (out / "data" / f"{pid}.json").write_text(
                    json.dumps(details[pid], ensure_ascii=False), encoding="utf-8"
                )
                write_index(points)

    print(json.dumps({
        "output": str(out),
        "points": len(points),
        "with_capture": sum(1 for p in points if p["capture"]),
        "with_audio": sum(1 for p in points if p["audio"]),
        "with_clip": sum(1 for p in points if p["clip"]),
        "with_questions": sum(1 for p in points if p["nq"]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
