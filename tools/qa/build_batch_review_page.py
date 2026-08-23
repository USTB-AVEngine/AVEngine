#!/usr/bin/env python3
"""Build a self-contained lazy-loading review page for one QA v2 batch.

Emits into --output:
  index.html          single-file viewer (vanilla JS, no CDN)
  data/points.json    full card index (spec + questions inline, one fetch)
  data/<pid>.json     per-point detail (kept for tooling; page no longer needs it)
  clips/<pid>.mp4     muxed review clip (only when audio exists; skip if present)
  thumbs/<pid>.jpg    lazy-loaded poster frame (scannable without pulling video)

Layout: one card per point with the clip poster on the left and every
question (type, zh/en text, options, answer highlighted) fully visible on
the right - review needs scrolling only, no clicks. Network discipline:
a single index fetch, thumbnails load only when scrolled into view
(native lazy imgs), video bytes move only after an explicit play click,
one video plays at a time, and index polling stops once the batch is
complete.

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

INDEX_REWRITE_EVERY = 10  # clips between incremental points.json rewrites


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


def make_thumb(clip: Path, thumb: Path) -> bool:
    if thumb.is_file():
        return True
    if not clip.is_file():
        return False
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", "2.5", "-i", str(clip),
         "-frames:v", "1", "-vf", "scale=480:-2", "-q:v", "5", str(thumb)],
        capture_output=True, text=True,
    )
    return proc.returncode == 0 and thumb.is_file()


HTML = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>__BATCH__ · QA v2 review</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif}
body{margin:0;background:#101014;color:#ddd}
header{position:sticky;top:0;background:#17171c;padding:8px 16px;border-bottom:1px solid #2c2c34;z-index:5}
h1{font-size:15px;margin:0 0 6px}
.bars{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;color:#999}
.bar{min-width:150px}.bar .track{background:#2c2c34;border-radius:4px;height:6px;overflow:hidden}
.bar .fill{background:#4c8dff;height:6px;width:0}
.ctl{margin-top:6px;display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.ctl button{background:#26262e;color:#bbb;border:1px solid #3a3a44;border-radius:12px;padding:2px 10px;font-size:12px;cursor:pointer}
.ctl button.on{background:#4c8dff;color:#fff;border-color:#4c8dff}
.ctl input{background:#1d1d24;color:#ddd;border:1px solid #3a3a44;border-radius:6px;padding:3px 8px;font-size:12px;width:90px}
main{padding:10px 14px;max-width:1380px;margin:0 auto}
.card{display:grid;grid-template-columns:390px 1fr;gap:0;border:1px solid #2c2c34;border-radius:10px;margin:10px 0;background:#17171c;overflow:hidden}
@media(max-width:900px){.card{grid-template-columns:1fr}}
.media{background:#000;position:relative;min-height:200px;display:flex;align-items:center;justify-content:center}
.media img{width:100%;display:block;cursor:pointer}
.media video{width:100%;display:block;background:#000}
.media .play{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;cursor:pointer;
 font-size:40px;color:#fff;text-shadow:0 0 12px #000;background:rgba(0,0,0,.12)}
.media .noclip{color:#777;font-size:12px;padding:60px 0}
.body{padding:10px 14px;min-width:0}
.head{display:flex;gap:6px;align-items:baseline;flex-wrap:wrap;margin-bottom:6px}
.pid{font-weight:700;font-size:16px}
.tag{font-size:11px;background:#26262e;border-radius:8px;padding:1px 8px;color:#aaa;white-space:nowrap}
.tag.ok{color:#7ad97a}.tag.miss{color:#e06666}.tag.warn{color:#e8b34c}
.tag.link{cursor:pointer;color:#8ab4ff;border:1px solid #3a3a55}
.q{display:flex;gap:8px;margin:5px 0;padding:6px 9px;background:#1d1d24;border-radius:7px;font-size:13px;line-height:1.45;align-items:baseline}
.q .ty{flex:none;font-size:10px;font-weight:700;border-radius:6px;padding:1px 6px;color:#fff}
.ty.T2{background:#7b5cd6}.ty.TA{background:#2f8f5b}.ty.T4{background:#b3762f}.ty.T7{background:#2f7fb3}.ty.T9{background:#b34f6e}
.q .zh{color:#e6e6ec}
.q .en{color:#8a8a99;font-size:11px;display:block;margin-top:1px}
.opt{display:inline-block;font-size:11px;border:1px solid #3a3a44;border-radius:9px;padding:0 8px;margin:2px 3px 0 0;color:#999}
.opt.ans{border-color:#3f8f4f;color:#7ad97a;font-weight:700}
.mod{font-size:10px;color:#666;margin-left:4px}
#tail{font-size:12px;color:#888;padding:10px;text-align:center}
</style></head><body>
<header>
 <h1>__BATCH__ <span id="counts" style="font-weight:400;color:#888;font-size:12px"></span></h1>
 <div class="bars">
  <div class="bar">视觉 <span id="pcapt"></span><div class="track"><div class="fill" id="bcapt"></div></div></div>
  <div class="bar">音频 <span id="paud"></span><div class="track"><div class="fill" id="baud"></div></div></div>
  <div class="bar">成片 <span id="pclip"></span><div class="track"><div class="fill" id="bclip"></div></div></div>
  <div class="bar">题目 <span id="pq"></span><div class="track"><div class="fill" id="bq"></div></div></div>
 </div>
 <div class="ctl" id="ctl"><input id="search" placeholder="点位 id…"></div>
</header>
<main><div id="list"></div><div id="tail">加载中…</div></main>
<script>
let ALL=[],VIEW=[],shown=0;const PAGE=16,CARDS={};
const F={pair:null,motion:null,flag:null,q:""};
const pct=(n,d)=>d?(100*n/d).toFixed(1)+"%":"–";
function bar(id,n,d){document.getElementById("b"+id).style.width=(d?100*n/d:0)+"%";
 document.getElementById("p"+id).textContent=`${n}/${d}`}
function updateBars(){
 bar("capt",ALL.filter(p=>p.capture).length,ALL.length);
 bar("aud",ALL.filter(p=>p.audio).length,ALL.length);
 bar("clip",ALL.filter(p=>p.clip).length,ALL.length);
 bar("q",ALL.filter(p=>p.questions.length>0).length,ALL.length);
}
const complete=()=>ALL.length&&ALL.every(p=>p.capture&&p.audio&&p.clip&&p.questions.length);
async function boot(){
 const j=await(await fetch("data/points.json")).json();ALL=j.points;
 document.getElementById("counts").textContent=`· ${ALL.length} 点 · ${ALL.reduce((s,p)=>s+p.questions.length,0)} 题 · ${j.batch.generated_at}`;
 updateBars();mkFilters();apply();
 new IntersectionObserver(e=>{if(e[0].isIntersecting)more()},{rootMargin:"600px"})
  .observe(document.getElementById("tail"));
 if(!complete()){const t=setInterval(async()=>{ // poll only while incomplete
   try{const j2=await(await fetch("data/points.json?t="+Date.now())).json();
    ALL=j2.points;updateBars();if(complete())clearInterval(t);}catch(e){}},30000)}
}
function chip(k,v){const b=document.createElement("button");b.textContent=v;
 b.onclick=()=>{F[k]=F[k]===v?null:v;
  document.getElementById("ctl").querySelectorAll("button").forEach(x=>
   x.classList.toggle("on",[F.pair,F.motion,F.flag].includes(x.textContent)));
  apply()};
 return b}
function mkFilters(){
 const c=document.getElementById("ctl");
 for(const v of [...new Set(ALL.map(p=>p.pair))].filter(Boolean))c.append(chip("pair",v));
 for(const v of [...new Set(ALL.map(p=>p.motion))].filter(Boolean))c.append(chip("motion",v));
 for(const v of ["main","twin","offscreen"])c.append(chip("flag",v));
 const s=document.getElementById("search");
 s.oninput=()=>{F.q=s.value.trim().toLowerCase();apply()};
}
function apply(){
 VIEW=ALL.filter(p=>(!F.pair||p.pair===F.pair)&&(!F.motion||p.motion===F.motion)
  &&(!F.flag||(F.flag==="twin"?!!p.twin_of:F.flag==="main"?!p.twin_of:p.offscreen))
  &&(!F.q||p.pid.toLowerCase().includes(F.q)));
 shown=0;document.getElementById("list").innerHTML="";
 for(const k in CARDS)delete CARDS[k];
 more();
}
function more(){
 const slice=VIEW.slice(shown,shown+PAGE);shown+=slice.length;
 for(const p of slice){const c=card(p);CARDS[p.pid]=c;document.getElementById("list").appendChild(c)}
 document.getElementById("tail").textContent=
  shown>=VIEW.length?`已显示全部 ${VIEW.length} 点`:`已显示 ${shown}/${VIEW.length} — 继续下拉`;
}
function tag(t,cls){const s=document.createElement("span");s.className="tag "+(cls||"");s.textContent=t;return s}
// video bytes move only after an explicit play click; one plays at a time
function stopOthers(){
 document.querySelectorAll(".media video").forEach(v=>{
  const pid=v.dataset.pid,clip=v.dataset.clip,media=v.closest(".media");
  v.pause();v.removeAttribute("src");v.load();
  media.replaceChildren(...thumbNodes(pid));
 });
}
function playVideo(p,media){
 stopOthers();
 const v=document.createElement("video");
 v.controls=true;v.autoplay=true;v.preload="auto";v.src=p.clip;
 v.dataset.pid=p.pid;v.dataset.clip=p.clip;
 media.replaceChildren(v);
}
function thumbNodes(pid){
 const img=document.createElement("img");
 img.loading="lazy";img.src=`thumbs/${pid}.jpg`;img.alt=pid;
 img.onerror=()=>{img.replaceWith(Object.assign(document.createElement("div"),
  {className:"noclip",textContent:"缩略图待生成（点 ▶ 直接播放）"}))};
 const play=document.createElement("div");play.className="play";play.textContent="▶";
 return [img,play];
}
function card(p){
 const c=document.createElement("div");c.className="card";c.id="pt-"+p.pid;
 const media=document.createElement("div");media.className="media";
 if(p.clip){
  media.append(...thumbNodes(p.pid));
  media.onclick=e=>{if(e.target.tagName!=="VIDEO")playVideo(p,media)};
 }else{
  media.append(Object.assign(document.createElement("div"),
   {className:"noclip",textContent:p.audio?"成片生成中…":"待音频完成"}));
 }
 const body=document.createElement("div");body.className="body";
 const head=document.createElement("div");head.className="head";
 head.append(Object.assign(document.createElement("span"),{className:"pid",textContent:p.pid}));
 if(p.pair)head.append(tag(p.pair));
 if(p.motion)head.append(tag(p.motion));
 if(p.program)head.append(tag(p.program.replace("qa_v2_two_human_","").replace("_turn_taking","")));
 if(p.twin_of){const t=tag("孪生 ⇄ "+p.twin_of,"link");t.onclick=()=>jump(p.twin_of);head.append(t)}
 if(p.twin){const t=tag("孪生 ⇄ "+p.twin,"link");t.onclick=()=>jump(p.twin);head.append(t)}
 if(p.offscreen)head.append(tag("off-screen","warn"));
 if(!p.capture)head.append(tag("视觉缺","miss"));
 if(!p.audio)head.append(tag("音频缺","miss"));
 if(p.receipt&&p.receipt.frames!==75)head.append(tag("帧数 "+p.receipt.frames,"miss"));
 body.append(head);
 for(const q of p.questions){
  const d=document.createElement("div");d.className="q";
  const ty=document.createElement("span");
  ty.className="ty "+(q.type_id.match(/QV2-(T[0-9A]+)/)||[,""])[1];
  ty.textContent=q.type_id.replace("QV2-","");
  const tx=document.createElement("div");
  const opts=q.options.map(o=>`<span class="opt${o===q.answer?" ans":""}">${o===q.answer?"✓ ":""}${o}</span>`).join("");
  tx.innerHTML=`<span class="zh">${q.question_zh||q.question_en}</span>`+
   `<span class="en">${q.question_zh?q.question_en:""}</span>`+
   `<div>${opts}<span class="mod">${q.modality_expectation||""}</span></div>`;
  d.append(ty,tx);body.append(d);
 }
 if(!p.questions.length)body.append(Object.assign(document.createElement("div"),
  {className:"q",textContent:"（题目待生成）"}));
 c.append(media,body);return c;
}
function jump(pid){
 const i=VIEW.findIndex(p=>p.pid===pid);
 if(i<0)return;
 while(shown<=i)more();
 const el=document.getElementById("pt-"+pid);
 if(el){el.scrollIntoView({behavior:"smooth",block:"center"});
  el.style.outline="2px solid #4c8dff";setTimeout(()=>el.style.outline="",1600)}
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
    thumbs_dir = out / "thumbs"
    thumbs_dir.mkdir(exist_ok=True)

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

    # pass 1: full card index immediately (page usable before any clip exists)
    twin_back: dict[str, str] = {}
    for pdir in sorted(args.inputs_root.iterdir()):
        spec_path = pdir / "spec.json"
        if spec_path.is_file():
            s = json.loads(spec_path.read_text(encoding="utf-8"))
            if s.get("twin_of"):
                twin_back[s["twin_of"]] = s["point_id"]

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
            "twin": twin_back.get(pid),
            "offscreen": bool(spec.get("offscreen_candidate")),
            "program": spec.get("program_id"),
            "capture": captured,
            "audio": has_audio,
            "clip": f"clips/{pid}.mp4" if has_clip else None,
            "receipt": receipt,
            "questions": [{
                "type_id": q["type_id"],
                "question_zh": q.get("question_zh"),
                "question_en": q["question_en"],
                "options": q["options"],
                "answer": q["answer"],
                "modality_expectation": q.get("modality_expectation"),
            } for q in qs],
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

    # pass 2: incremental clip muxing; debounced index rewrites (final below)
    if args.mux_clips and args.audio_root:
        pending = 0
        for row_entry in points:
            pid = row_entry["pid"]
            if row_entry["clip"] or not (row_entry["capture"] and row_entry["audio"]):
                continue
            if mux_clip(args.captures_root, args.audio_root, pid, clips_dir / f"{pid}.mp4"):
                row_entry["clip"] = f"clips/{pid}.mp4"
                details[pid]["clip"] = f"clips/{pid}.mp4"
                (out / "data" / f"{pid}.json").write_text(
                    json.dumps(details[pid], ensure_ascii=False), encoding="utf-8"
                )
                pending += 1
                if pending >= INDEX_REWRITE_EVERY:
                    write_index(points)
                    pending = 0
        if pending:
            write_index(points)

    # pass 3: poster thumbnails for every existing clip (cheap, idempotent)
    thumbs = 0
    for row_entry in points:
        if row_entry["clip"] and make_thumb(
            clips_dir / f"{row_entry['pid']}.mp4",
            thumbs_dir / f"{row_entry['pid']}.jpg",
        ):
            thumbs += 1
    write_index(points)

    print(json.dumps({
        "output": str(out),
        "points": len(points),
        "with_capture": sum(1 for p in points if p["capture"]),
        "with_audio": sum(1 for p in points if p["audio"]),
        "with_clip": sum(1 for p in points if p["clip"]),
        "with_questions": sum(1 for p in points if p["questions"]),
        "thumbs": thumbs,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
