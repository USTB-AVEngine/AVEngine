#!/usr/bin/env python3
"""Build a browser-ready full-AV human calibration pack from run02 media.

The public study manifest never contains gold values.  A separate answer key
binds each item to the exact copied MP4 because the media live outside Git and
a filename alone cannot detect accidental replacement.  This pack is research
material only; it does not approve tolerances or dataset admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path


CALIBRATION_PROFILES = {"card1F", "card1B", "card8"}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_measured_floor(fact, point_id):
    """Refuse media whose room floor was never measured.

    On 2026-09-03 every Apartment render turned out to stand on a hand-written
    ground_z_ue_cm (0.0) while the floor sits about 27 cm higher: dogs sunk into
    the floor and a camera 1.20 m above it.  Such frames must not calibrate
    human tolerances.  A fact written after the fix carries the measured floor
    reference in its room block; older facts do not and are refused.
    """
    room = fact.get("room") or {}
    reference = room.get("floor_reference") or {}
    if reference.get("status") != "measured" or reference.get("ground_z_ue_cm") is None:
        raise ValueError(
            f"{point_id}: fact carries no measured floor reference "
            f"(room.floor_reference); media rendered before the room floor was "
            f"measured must not enter a calibration pack")
    declared = room.get("ground_z_ue_cm")
    if declared is None or abs(float(declared) - float(reference["ground_z_ue_cm"])) > 0.5:
        raise ValueError(
            f"{point_id}: room.ground_z_ue_cm={declared} disagrees with the measured "
            f"floor {reference['ground_z_ue_cm']} cm")


# The page must label its azimuth scale with the same convention the stem
# states, and the stem is what both the participant and the model read.  On
# 2026-09-03 owner answered two items on the assumption that right was 0 deg
# and straight ahead was 90: the same three answers scored a 48.97 deg median
# error under the stem's convention and 30.0 deg under his, so an unlabelled
# scale measures the convention guess rather than perception.  Derive the
# convention from the stem and fail closed when no marker is present.
_AZIMUTH_MARKERS = (
    ("right is positive", "right_positive"),
    ("positive values to the right", "right_positive"),
    ("left is positive", "left_positive"),
    ("positive values to the left", "left_positive"),
)


def _azimuth_convention(stem, point_id):
    lowered = str(stem).lower()
    found = sorted({name for marker, name in _AZIMUTH_MARKERS
                    if marker in lowered})
    if not found:
        raise ValueError(
            f"{point_id}: the numeric stem states no azimuth convention; the "
            "page cannot label its scale and the answer would measure the "
            "participant's convention guess")
    if len(found) > 1:
        raise ValueError(
            f"{point_id}: the numeric stem states more than one azimuth "
            f"convention {found}")
    return found[0]


def _view_facts(point_dir, point_id):
    """Frame extent and clip length, read from the render block, never guessed."""

    path = Path(point_dir) / "timeline.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{point_id}: missing timeline.json; the page needs the camera "
            "field of view and the clip length")
    render = (_read(path).get("render") or {})
    missing = [key for key in ("hfov_degrees", "frame_count", "frame_rate_hz")
               if key not in render]
    if missing:
        raise ValueError(f"{point_id}: timeline render block missing {missing}")
    hfov = float(render["hfov_degrees"])
    frames = int(render["frame_count"])
    fps = float(render["frame_rate_hz"])
    if hfov <= 0 or frames <= 0 or fps <= 0:
        raise ValueError(f"{point_id}: render block has non-positive extent")
    return {
        "hfov_degrees": hfov,
        "frame_count": frames,
        "video_fps": fps,
        "clip_seconds": round(frames / fps, 4),
    }


def _ffprobe_media(path: Path) -> dict:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise ValueError("ffprobe is required to validate calibration media")
    command = [
        ffprobe, "-v", "error", "-count_frames",
        "-show_entries", "stream=codec_type,nb_read_frames,nb_frames,r_frame_rate,duration",
        "-of", "json", str(path),
    ]
    try:
        done = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=30)
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"ffprobe timed out for calibration media: {path}") from exc
    if done.returncode != 0:
        raise ValueError(f"ffprobe failed for calibration media {path}: {done.stderr[-400:]}")
    try:
        streams = json.loads(done.stdout)["streams"]
    except (TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"ffprobe returned no streams for calibration media {path}") from exc
    video = next((row for row in streams if row.get("codec_type") == "video"), None)
    audio = next((row for row in streams if row.get("codec_type") == "audio"), None)
    if video is None or audio is None:
        raise ValueError(f"calibration media must contain video and audio: {path}")
    raw_frames = video.get("nb_read_frames", video.get("nb_frames"))
    try:
        frames = int(raw_frames)
        video_duration = float(video["duration"])
        audio_duration = float(audio["duration"])
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"calibration media has an invalid stream clock: {path}") from exc
    return {"frame_count": frames, "video_duration_seconds": video_duration,
            "audio_duration_seconds": audio_duration}


def _validate_calibration_media(path: Path, view: dict, point_id: str) -> dict:
    observed = _ffprobe_media(path)
    expected_frames = int(view["frame_count"])
    expected_duration = float(view["clip_seconds"])
    if observed["frame_count"] != expected_frames:
        raise ValueError(
            f"{point_id}: calibration video has {observed['frame_count']} frames, "
            f"expected {expected_frames}")
    tolerance = max(1.0 / float(view["video_fps"]), 0.05)
    for stream_name in ("video", "audio"):
        duration = observed[f"{stream_name}_duration_seconds"]
        if not math.isfinite(duration) or abs(duration - expected_duration) > tolerance:
            raise ValueError(
                f"{point_id}: calibration {stream_name} duration {duration:.6f}s "
                f"differs from expected {expected_duration:.6f}s")
    return {
        "media_frame_count": observed["frame_count"],
        "media_video_duration_seconds": observed["video_duration_seconds"],
        "media_audio_duration_seconds": observed["audio_duration_seconds"],
    }


def build(selection, facts_root, media_root, output_root, *,
          practice_selection, per_profile_limit=None,
          practice_per_profile_limit=None):
    facts_root = Path(facts_root).resolve()
    media_root = Path(media_root).resolve()
    output_root = Path(output_root).resolve()
    public_root = output_root / "public"
    private_root = output_root / "private"
    media_output = public_root / "media"
    media_output.mkdir(parents=True)
    private_root.mkdir()
    def _collect(records, limit, *, reveal):
        """Build items for one selection.  Practice items reveal their truth."""
        items, answers, counts = [], [], {}
        for record in sorted(records, key=lambda row: row["point_id"]):
            profile_id = str(record["profile_id"])
            if profile_id not in CALIBRATION_PROFILES:
                continue
            if limit is not None and counts.get(profile_id, 0) >= limit:
                continue
            counts[profile_id] = counts.get(profile_id, 0) + 1
            point_id = str(record["point_id"])
            point_dir = facts_root / point_id
            fact_path = point_dir / "fact_record.json"
            source_media = media_root / point_id / "full_main.mp4"
            if not fact_path.is_file() or not source_media.is_file():
                raise FileNotFoundError(
                    f"{point_id}: missing fact or full-main media")
            fact = _read(fact_path)
            _require_measured_floor(fact, point_id)
            view = _view_facts(point_dir, point_id)
            view.update(_validate_calibration_media(source_media, view, point_id))
            copied = media_output / f"{point_id}.mp4"
            shutil.copy2(source_media, copied)
            stem = str(fact["open"]["stem"])
            if profile_id in {"card1F", "card1B"}:
                binding = {
                    "stem": "Which dog is the one that barked last?",
                    "options": ["black-and-white", "yellow"],
                    "truth": str(fact["target_coat"]),
                }
                numeric = {
                    "stem": stem,
                    "unit": "deg",
                    "kind": "azimuth_deg",
                    "convention": _azimuth_convention(stem, point_id),
                    "truth": float(fact["open"]["truth_value"]),
                    "error": "circular_angle_deg",
                }
            else:
                numeric_truth = float(fact["open"]["truth_value"])
                binding = {
                    "stem": ("Did the named dog bark before or after the "
                             "other dog?"),
                    "options": ["before", "after"],
                    "truth": "before" if bool(fact["target_first"]) else "after",
                }
                numeric = {
                    "stem": stem,
                    "unit": "s",
                    "kind": "time_s",
                    "truth": numeric_truth,
                    "error": "absolute_time_s",
                }
            hidden = {"error"} if reveal else {"error", "truth"}
            items.append({
                "item_id": point_id,
                "profile_id": profile_id,
                "media": f"media/{point_id}.mp4",
                "view": view,
                "binding": {key: value for key, value in binding.items()
                            if key not in hidden},
                "numeric": {key: value for key, value in numeric.items()
                            if key not in hidden},
            })
            answers.append({
                "item_id": point_id,
                "profile_id": profile_id,
                "binding_truth": binding["truth"],
                "numeric_truth": numeric["truth"],
                "error_kind": numeric["error"],
                "media_sha256": _sha256(copied),
                "source_fact": str(fact_path),
            })
        return items, answers

    public_items, answer_items = _collect(
        selection["selected"], per_profile_limit, reveal=False)
    practice_items, _ = _collect(
        practice_selection["selected"], practice_per_profile_limit, reveal=True)
    if not practice_items:
        raise ValueError(
            "practice selection contains no card1F/card1B/card8 items; the "
            "study cannot anchor the answer scale without practice feedback")
    shared = ({item["item_id"] for item in public_items}
              & {item["item_id"] for item in practice_items})
    if shared:
        raise ValueError(
            f"practice and study selections share {sorted(shared)}; practice "
            "reveals its own truth, so a shared item leaks a study answer")
    conventions = sorted({item["numeric"]["convention"]
                          for item in public_items + practice_items
                          if "convention" in item["numeric"]})
    if len(conventions) > 1:
        raise ValueError(
            f"items mix azimuth conventions {conventions}; one pack states one")
    if not public_items:
        raise ValueError("selection contains no card1F/card1B/card8 items")
    study = {
        "schema": "qa_v3_human_calibration_study_v1",
        "status": "research_candidate",
        "qualification_claim": False,
        "instructions": (
            "Use headphones. Replay and scrub as much as you like. Do the "
            "practice items first: they show the correct answer so the degree "
            "and second scales stop being a guess. Do not inspect "
            "answer_key.json."),
        "azimuth_convention": conventions[0] if conventions else None,
        "azimuth_convention_source": (
            "derived from each item's own open stem; the build fails when a "
            "stem states no convention"),
        "items": public_items,
        "practice_item_count": len(practice_items),
    }
    answers = {
        "schema": "qa_v3_human_calibration_answer_key_v1",
        "status": "research_candidate",
        "azimuth_convention": conventions[0] if conventions else None,
        "items": answer_items,
        "boundary": (
            "Hidden study answer key; tolerances remain unapproved until "
            "responses are scored and reviewed."),
    }
    _write(public_root / "study_items.json", study)
    _write(public_root / "practice_items.json", {
        "schema": "qa_v3_human_calibration_practice_v1",
        "status": "research_candidate",
        "note": ("Practice items carry their own truth on purpose: the page "
                 "shows it as feedback. They are disjoint from the study "
                 "items, which is enforced at build time."),
        "azimuth_convention": conventions[0] if conventions else None,
        "items": practice_items,
    })
    _write(private_root / "answer_key.json", answers)
    (public_root / "index.html").write_text(_HTML, encoding="utf-8")
    (output_root / "README.md").write_text(
        "# QA v3 human calibration pack\n\n"
        "Serve only `public/` with a static HTTP server. Participants open "
        "`index.html`, use stereo headphones, and download one response JSON. "
        "Participants may replay and scrub freely; the practice round in "
        "`public/practice_items.json` shows its own answers on purpose and is "
        "built disjoint from the study items. The response file records the "
        "azimuth convention the page labelled its scale with. "
        "The answer key stays in `private/answer_key.json`, outside the served "
        "tree. Score collected files "
        "with `tools/qa/score_qa_v3_human_calibration.py`. This is research "
        "calibration, not dataset admission. Example:\n\n"
        "```bash\ncd public\npython3 -m http.server 8767 --bind 127.0.0.1\n```\n",
        encoding="utf-8")
    return study, answers


_HTML = r"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QA v3 试听</title>
<style>
body{font:17px system-ui,-apple-system,sans-serif;max-width:960px;margin:1.5rem auto;padding:0 1rem;background:#f6f7f9;color:#18202a}
.card{background:white;border-radius:14px;padding:1.2rem;box-shadow:0 3px 18px #0001;margin:1rem 0}
video{width:100%;max-height:520px;background:#111;border-radius:10px}label{display:block;margin:.8rem 0}
button{padding:.7rem 1.1rem;margin:.4rem .4rem .4rem 0;border:0;border-radius:9px;background:#1769e0;color:white;font-size:1rem}
button.ghost{background:#e6eaf0;color:#18202a}button:disabled{background:#aeb7c4}
.muted{color:#667085}.notice{background:#fff4d6;padding:.7rem;border-radius:8px}
.good{background:#e6f6ec;padding:.7rem;border-radius:8px}.bad{background:#fdeaea;padding:.7rem;border-radius:8px}
input[type=text],input[type=number]{padding:.55rem;border:1px solid #aeb7c4;border-radius:7px;font-size:1rem}
input[type=range]{width:100%}
textarea{width:100%;box-sizing:border-box;padding:.6rem;border:1px solid #aeb7c4;border-radius:7px;font:13px ui-monospace,monospace}
.scale{display:flex;justify-content:space-between;font:13px ui-monospace,monospace;color:#667085;margin-top:-.4rem}
.ticks{display:flex;justify-content:space-between;font:11px ui-monospace,monospace;color:#98a2b3}
.row{display:flex;gap:.6rem;align-items:center;flex-wrap:wrap}
.big{font:600 1.25rem system-ui;margin:.3rem 0}
</style>
<h1>QA v3 音视频试听</h1>
<p class="notice">请戴<b>双声道耳机</b>。可以任意次数重播，也可以拖动进度条。
先做几道<b>练习题</b>——练习题会告诉你正确答案，好让"多少度""第几秒"这两把尺子不用靠猜。</p>
<p class="muted" id="levelNote" hidden></p>

<div class="card" id="welcome">
  <label>测试编号 <input id="participant" type="text" placeholder="例如 owner-1"></label>
  <div class="row"><button id="start">开始练习</button><button id="resume" class="ghost" hidden>接着上次继续</button></div>
  <p class="muted" id="resumeNote" hidden></p>
</div>

<div class="card" id="stage" hidden>
  <p class="muted"><span id="progress"></span> · <span id="clipLen"></span> · <span id="phase"></span></p>
  <h2 id="mainQuestion"></h2>
  <video id="video" playsinline preload="auto" disablepictureinpicture oncontextmenu="return false"></video>
  <div class="row">
    <button id="play">播放</button>
    <button id="restart" class="ghost">从头播</button>
    <span class="muted" id="playState"></span>
  </div>
  <input id="seek" type="range" min="0" max="1000" value="0" step="1" aria-label="进度">
  <div class="scale"><span>开头</span><span id="seekTime"></span><span>结尾</span></div>

  <div id="answers">
    <h3 id="bindingStem"></h3>
    <div id="bindingOptions"></div>
    <h3 id="numericStem"></h3>
    <p class="muted" id="legend"></p>
    <input id="slider" type="range" min="0" max="1000" value="500" step="1" aria-label="数值答案">
    <div class="ticks" id="tickRow"></div>
    <div class="row">
      <div class="big" id="readout"></div>
      <label style="margin:0">也可以直接输入 <input id="numeric" type="number" step="any"> <span id="unit"></span></label>
    </div>
    <label>置信度（1–5） <input id="confidence" type="number" min="1" max="5"></label>
    <div class="row"><button id="next">保存并下一题</button><button id="export" class="ghost">导出当前进度</button></div>
  </div>
  <div id="feedback" hidden></div>
</div>

<div id="done" class="card" hidden><h2>完成</h2><p>请下载结果文件并交给研究人员。</p>
<button id="download">下载回答 JSON</button><button id="copy" class="ghost">复制 JSON</button>
<textarea id="resultText" rows="10" readonly aria-label="回答 JSON"></textarea></div>
<p class="muted">本页面不加载 answer_key.json。练习题自带答案是有意的，它们与正题不重叠（构建时强制）。</p>
<script>
const $=id=>document.getElementById(id);
let study=null,practice=[],items=[],order=[],at=0,responses=[],phase="practice";
let participant="",plays=0,seeks=0,shown=0,convention=null,storeKey="";
const S=()=>({participant_id:participant,phase,at,order,responses});
function payload(){return JSON.stringify({schema:"qa_v3_human_calibration_responses_v1",
  azimuth_convention:convention,responses},null,2);}
function save(){try{localStorage.setItem(storeKey,JSON.stringify(S()));}catch(e){}}
function half(item){return item.view.hfov_degrees/2;}
function rightSign(){return convention==="left_positive"?-1:1;}
function isAngle(item){return item.numeric.kind==="azimuth_deg";}
function sliderToValue(item,pos){
  if(isAngle(item))return rightSign()*(pos/500-1)*half(item);
  return pos/1000*item.view.clip_seconds;}
function valueToSlider(item,v){
  if(isAngle(item)){const h=rightSign()*half(item);return Math.max(0,Math.min(1000,Math.round(500*(v/h+1))));}
  return Math.max(0,Math.min(1000,Math.round(v/item.view.clip_seconds*1000)));}
function fmt(item,v){return isAngle(item)?v.toFixed(0)+"°":v.toFixed(2)+" 秒";}
function drawScale(item){
  const angle=isAngle(item);
  if(angle){
    const h=half(item),r=rightSign();
    $("legend").textContent=convention==="left_positive"
      ?"0° 正前方，+90° 正左方，−90° 正右方，±180° 正后方。答案永远在画面之内。"
      :"0° 正前方，+90° 正右方，−90° 正左方，±180° 正后方。答案永远在画面之内。";
    const cells=[0,250,500,750,1000].map(p=>{
      const v=r*(p/500-1)*h;const tag=p===500?"正前 0°":(p===0?"最左 ":p===1000?"最右 ":"")+v.toFixed(0)+"°";
      return "<span>"+tag+"</span>";});
    $("tickRow").innerHTML=cells.join("");
    $("unit").textContent="度";
  }else{
    $("legend").textContent="片长 "+item.view.clip_seconds.toFixed(1)+" 秒；报你听到那一声的时刻。";
    const last=Math.ceil(item.view.clip_seconds);
    $("tickRow").innerHTML=Array.from({length:last+1},(_,s)=>s)
      .map(s=>"<span>"+s+"s</span>").join("");
    $("unit").textContent="秒";
  }
}
function syncFromSlider(item){const v=sliderToValue(item,Number($("slider").value));
  $("readout").textContent=fmt(item,v);$("numeric").value=isAngle(item)?v.toFixed(0):v.toFixed(2);}
function current(){return phase==="practice"?practice[at]:items[order[at]];}
function show(){
  const item=current();plays=0;seeks=0;shown=Date.now();
  $("phase").textContent=phase==="practice"?"练习（会告诉你答案）":"正题";
  $("progress").textContent="第 "+(at+1)+" / "+(phase==="practice"?practice.length:order.length)+" 题";
  $("clipLen").textContent="片长 "+item.view.clip_seconds.toFixed(1)+" 秒";
  $("mainQuestion").textContent=item.numeric.stem;
  $("video").src=item.media;$("video").load();
  $("bindingStem").textContent=item.binding.stem;$("numericStem").textContent=item.numeric.stem;
  $("bindingOptions").innerHTML=item.binding.options.map(o=>
    '<label><input type="radio" name="binding" value="'+o+'"> '+o+"</label>").join("");
  $("confidence").value="";$("answers").hidden=false;$("feedback").hidden=true;
  $("playState").textContent="尚未播放";$("seek").value=0;$("seekTime").textContent="";
  $("seek").disabled=false;
  drawScale(item);$("slider").value=500;syncFromSlider(item);
  $("next").textContent=phase==="practice"?"看答案":"保存并下一题";
}
function begin(){
  $("welcome").hidden=true;$("stage").hidden=false;
  if(phase==="practice"&&practice.length===0){phase="main";at=0;}
  show();
}
$("participant").oninput=()=>{
  const key="qa_v3_calib_"+$("participant").value.trim();
  let raw=null;try{raw=localStorage.getItem(key);}catch(e){}
  $("resume").hidden=!raw;$("resumeNote").hidden=!raw;
  if(raw){try{const s=JSON.parse(raw);
    $("resumeNote").textContent="找到上次进度："+s.phase+" 第 "+(s.at+1)+" 题，已答 "+s.responses.length+" 条。";}catch(e){}}
};
async function load(){
  study=await fetch("study_items.json").then(r=>r.json());
  items=study.items;convention=study.azimuth_convention;
  try{const pr=await fetch("practice_items.json").then(r=>r.json());practice=pr.items||[];}
  catch(e){practice=[];}
  if(study.listening_gain_db!==undefined&&study.listening_gain_db!==null){
    $("levelNote").hidden=false;
    $("levelNote").textContent="本包音量已按统一标准抬升 "+study.listening_gain_db.toFixed(1)
      +" dB（整包同一个值，条目之间的电平差和每条的左右耳差不变；渲染产物未改动）。";}
}
$("start").onclick=async()=>{
  participant=$("participant").value.trim();
  if(!participant)return alert("请先填写测试编号");
  storeKey="qa_v3_calib_"+participant;await load();
  order=[...items.keys()];
  for(let i=order.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[order[i],order[j]]=[order[j],order[i]];}
  phase="practice";at=0;responses=[];save();begin();
};
$("resume").onclick=async()=>{
  participant=$("participant").value.trim();storeKey="qa_v3_calib_"+participant;
  await load();
  try{const s=JSON.parse(localStorage.getItem(storeKey));
    order=s.order;at=s.at;responses=s.responses;phase=s.phase;}catch(e){return alert("进度读不出来，请重新开始");}
  begin();
};
$("play").onclick=async()=>{const v=$("video");
  if(v.paused){plays++;try{await v.play();}catch(e){alert("浏览器未能播放，请再点一次");}}
  else v.pause();};
$("restart").onclick=async()=>{const v=$("video");v.currentTime=0;plays++;try{await v.play();}catch(e){}};
$("video").ontimeupdate=()=>{const v=$("video");
  if(!v.duration)return;$("seek").value=Math.round(v.currentTime/v.duration*1000);
  if(isAngle(current()))$("seekTime").textContent=v.currentTime.toFixed(2)+" / "+v.duration.toFixed(2)+" 秒";};
$("video").onplay=()=>{$("play").textContent="暂停";$("playState").textContent="播放中（第 "+plays+" 次）";};
$("video").onpause=()=>{$("play").textContent="播放";};
$("video").onended=()=>{$("playState").textContent="已播放 "+plays+" 次";};
$("seek").oninput=()=>{const v=$("video");if(!v.duration)return;
  seeks++;v.currentTime=Number($("seek").value)/1000*v.duration;};
$("slider").oninput=()=>syncFromSlider(current());
$("numeric").oninput=()=>{const item=current(),v=Number($("numeric").value);
  if($("numeric").value==="")return;$("slider").value=valueToSlider(item,v);
  $("readout").textContent=fmt(item,v);};
function collected(){const item=current();
  const b=document.querySelector("input[name=binding]:checked");
  if(!b)return alert("请先选一个"),null;
  if($("numeric").value==="")return alert("请给一个数值答案"),null;
  if(!$("confidence").value)return alert("请填置信度"),null;
  return {participant_id:participant,item_id:item.item_id,profile_id:item.profile_id,
    presentation_index:at,play_count:plays,seek_count:seeks,
    seconds_on_item:Math.round((Date.now()-shown)/100)/10,
    binding_answer:b.value,numeric_answer:Number($("numeric").value),
    confidence:Number($("confidence").value)};}
function circular(a,b){const d=Math.abs(a-b)%360;return Math.min(d,360-d);}
$("next").onclick=()=>{
  const item=current(),row=collected();if(!row)return;
  if(phase==="practice"&&$("feedback").hidden){
    const bOK=row.binding_answer===item.binding.truth;
    const err=isAngle(item)?circular(row.numeric_answer,item.numeric.truth)
                           :Math.abs(row.numeric_answer-item.numeric.truth);
    $("feedback").hidden=false;$("feedback").className=bOK?"good":"bad";
    $("feedback").innerHTML="<b>"+(bOK?"认对了":"认错了")+"</b>：正确答案是 "+item.binding.truth
      +"。<br>正确数值是 <b>"+fmt(item,item.numeric.truth)+"</b>，你答 "+fmt(item,row.numeric_answer)
      +"，差 "+(isAngle(item)?err.toFixed(0)+"°":err.toFixed(2)+" 秒")+"。";
    $("next").textContent="下一题";return;}
  if(phase==="main")responses.push(row);
  at++;save();
  if(phase==="practice"&&at>=practice.length){phase="main";at=0;save();show();return;}
  if(phase==="main"&&at>=order.length){
    $("stage").hidden=true;$("done").hidden=false;$("resultText").value=payload();
    try{localStorage.removeItem(storeKey);}catch(e){}return;}
  show();};
function dl(text,name){const b=new Blob([text],{type:"application/json"});
  const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download=name;a.click();}
$("export").onclick=()=>dl(payload(),"qa_v3_"+participant+"_partial.json");
$("download").onclick=()=>dl(payload(),"qa_v3_"+participant+".json");
$("copy").onclick=async()=>{try{await navigator.clipboard.writeText(payload());$("copy").textContent="已复制";}
catch(e){$("resultText").focus();$("resultText").select();document.execCommand("copy");$("copy").textContent="已复制";}};
</script></html>
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--practice-selection-manifest", required=True, type=Path,
                        help=("held-out points used for the practice round; "
                              "practice shows its own answer, so these must "
                              "not appear in the study selection"))
    parser.add_argument("--facts-root", required=True, type=Path)
    parser.add_argument("--media-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--per-profile-limit", type=int)
    parser.add_argument("--practice-per-profile-limit", type=int)
    args = parser.parse_args(argv)
    if args.output_root.exists() or args.output_root.is_symlink():
        print(f"refusing to overwrite: {args.output_root}", file=sys.stderr)
        return 2
    args.output_root.mkdir(parents=True)
    for name, value in (("--per-profile-limit", args.per_profile_limit),
                        ("--practice-per-profile-limit",
                         args.practice_per_profile_limit)):
        if value is not None and value <= 0:
            parser.error(f"{name} must be positive")
    study, _ = build(
        _read(args.selection_manifest), args.facts_root,
        args.media_root, args.output_root,
        practice_selection=_read(args.practice_selection_manifest),
        per_profile_limit=args.per_profile_limit,
        practice_per_profile_limit=args.practice_per_profile_limit)
    print(json.dumps({
        "output": str(args.output_root.resolve()),
        "item_count": len(study["items"]),
        "practice_item_count": study["practice_item_count"],
        "azimuth_convention": study["azimuth_convention"],
        "profiles": sorted({item["profile_id"] for item in study["items"]}),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
