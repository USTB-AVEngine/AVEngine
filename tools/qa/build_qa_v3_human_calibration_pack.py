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
import shutil
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


def build(selection, facts_root, media_root, output_root, *,
          per_profile_limit=None):
    facts_root = Path(facts_root).resolve()
    media_root = Path(media_root).resolve()
    output_root = Path(output_root).resolve()
    public_root = output_root / "public"
    private_root = output_root / "private"
    media_output = public_root / "media"
    media_output.mkdir(parents=True)
    private_root.mkdir()
    public_items = []
    answer_items = []
    profile_counts = {}
    for record in sorted(selection["selected"], key=lambda row: row["point_id"]):
        profile_id = str(record["profile_id"])
        if profile_id not in CALIBRATION_PROFILES:
            continue
        if (per_profile_limit is not None
                and profile_counts.get(profile_id, 0) >= per_profile_limit):
            continue
        profile_counts[profile_id] = profile_counts.get(profile_id, 0) + 1
        point_id = str(record["point_id"])
        fact_path = facts_root / point_id / "fact_record.json"
        source_media = media_root / point_id / "full_main.mp4"
        if not fact_path.is_file() or not source_media.is_file():
            raise FileNotFoundError(
                f"{point_id}: missing fact or full-main media")
        fact = _read(fact_path)
        copied = media_output / f"{point_id}.mp4"
        shutil.copy2(source_media, copied)
        if profile_id in {"card1F", "card1B"}:
            target = str(fact["target_coat"])
            binding = {
                "stem": "Which dog is the one that barked last?",
                "options": ["black-and-white", "yellow"],
                "truth": target,
            }
            numeric = {
                "stem": fact["open"]["stem"],
                "unit": "deg",
                "truth": float(fact["open"]["truth_value"]),
                "error": "circular_angle_deg",
            }
        else:
            target_first = bool(fact["target_first"])
            binding = {
                "stem": "Did the named dog bark before or after the other dog?",
                "options": ["before", "after"],
                "truth": "before" if target_first else "after",
            }
            numeric = {
                "stem": fact["open"]["stem"],
                "unit": "s",
                "truth": float(fact["open"]["truth_value"]),
                "error": "absolute_time_s",
            }
        public_items.append({
            "item_id": point_id,
            "profile_id": profile_id,
            "media": f"media/{point_id}.mp4",
            "binding": {key: value for key, value in binding.items()
                        if key != "truth"},
            "numeric": {key: value for key, value in numeric.items()
                        if key not in {"truth", "error"}},
        })
        answer_items.append({
            "item_id": point_id,
            "profile_id": profile_id,
            "binding_truth": binding["truth"],
            "numeric_truth": numeric["truth"],
            "error_kind": numeric["error"],
            "media_sha256": _sha256(copied),
            "source_fact": str(fact_path),
        })
    if not public_items:
        raise ValueError("selection contains no card1F/card1B/card8 items")
    study = {
        "schema": "qa_v3_human_calibration_study_v1",
        "status": "research_candidate",
        "qualification_claim": False,
        "instructions": (
            "Use headphones. Watch and listen to the full clip, answer the "
            "binding question, then enter one numeric answer. Do not inspect "
            "answer_key.json."),
        "items": public_items,
    }
    answers = {
        "schema": "qa_v3_human_calibration_answer_key_v1",
        "status": "research_candidate",
        "items": answer_items,
        "boundary": (
            "Hidden study answer key; tolerances remain unapproved until "
            "responses are scored and reviewed."),
    }
    _write(public_root / "study_items.json", study)
    _write(private_root / "answer_key.json", answers)
    (public_root / "index.html").write_text(_HTML, encoding="utf-8")
    (output_root / "README.md").write_text(
        "# QA v3 human calibration pack\n\n"
        "Serve only `public/` with a static HTTP server. Participants open "
        "`index.html`, use stereo headphones, and download one response JSON. "
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
body{font:17px system-ui,-apple-system,sans-serif;max-width:900px;margin:1.5rem auto;padding:0 1rem;background:#f6f7f9;color:#18202a}
.card{background:white;border-radius:14px;padding:1.2rem;box-shadow:0 3px 18px #0001;margin:1rem 0}
video{width:100%;max-height:520px;background:#111;border-radius:10px}label{display:block;margin:.8rem 0}
button{padding:.7rem 1.1rem;margin:.4rem .4rem .4rem 0;border:0;border-radius:9px;background:#1769e0;color:white;font-size:1rem}
button:disabled{background:#aeb7c4}.muted{color:#667085}.notice{background:#fff4d6;padding:.7rem;border-radius:8px}
input[type=text],input[type=number]{padding:.55rem;border:1px solid #aeb7c4;border-radius:7px;font-size:1rem}
textarea{width:100%;box-sizing:border-box;padding:.6rem;border:1px solid #aeb7c4;border-radius:7px;font:13px ui-monospace,monospace}
</style>
<h1>QA v3 音视频试听</h1>
<p class="notice">请戴双声道耳机。每题最多从头播放两次；页面不显示时间轴，也不能拖动或调速。</p>
<div class="card" id="welcome"><label>测试编号 <input id="participant" type="text" placeholder="例如 owner-test"></label>
<button id="start">开始试听</button></div>
<div id="study" class="card" hidden><p id="progress" class="muted"></p>
<h2 id="mainQuestion"></h2><video id="video" playsinline preload="auto" disablepictureinpicture oncontextmenu="return false"></video>
<div><button id="play">播放视频</button><span id="playState" class="muted">尚未播放</span></div>
<div id="answers" hidden><h3 id="bindingStem"></h3><div id="bindingOptions"></div>
<h3 id="numericStem"></h3><label>数值答案 <input id="numeric" type="number" step="any"> <span id="unit"></span></label>
<label>置信度（1–5） <input id="confidence" type="number" min="1" max="5"></label>
<button id="next">保存并进入下一题</button></div></div>
<div id="done" class="card" hidden><h2>完成</h2><p>请下载结果文件并交给研究人员。</p>
<button id="download">下载回答 JSON</button><button id="copy">复制 JSON</button>
<textarea id="resultText" rows="10" readonly aria-label="回答 JSON"></textarea></div>
<p class="muted">本页面不会加载答案文件。</p>
<script>
let items=[],order=[],at=0,responses=[],plays=0,completed=false; const $=id=>document.getElementById(id);
function payload(){return JSON.stringify({schema:'qa_v3_human_calibration_responses_v1',responses},null,2);}
function show(){const x=items[order[at]];plays=0;completed=false;$('progress').textContent=`第 ${at+1} / ${order.length} 题`;
$('mainQuestion').textContent=x.numeric.stem;$('video').src=x.media;$('video').load();$('answers').hidden=true;
$('bindingStem').textContent=x.binding.stem;$('numericStem').textContent=x.numeric.stem;$('unit').textContent=x.numeric.unit;
$('numeric').value='';$('confidence').value='';$('play').disabled=false;$('play').textContent='播放视频';$('playState').textContent='尚未播放';
$('bindingOptions').innerHTML=x.binding.options.map(o=>`<label><input type="radio" name="binding" value="${o}"> ${o}</label>`).join('');}
$('start').onclick=async()=>{if(!$('participant').value.trim())return alert('请先填写测试编号');
const data=await fetch('study_items.json').then(r=>r.json());items=data.items;order=[...items.keys()];
for(let i=order.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[order[i],order[j]]=[order[j],order[i]];}
$('welcome').hidden=true;$('study').hidden=false;show();};
$('play').onclick=async()=>{if(plays>=2)return;plays++;completed=false;$('answers').hidden=true;$('video').currentTime=0;
$('play').disabled=true;$('playState').textContent=`正在播放（第 ${plays} 次）`;try{await $('video').play();}catch(e){$('play').disabled=false;alert('浏览器未能播放，请再点一次');}};
$('video').onended=()=>{completed=true;$('answers').hidden=false;$('playState').textContent=`已完整播放 ${plays} 次`;
$('play').disabled=plays>=2;$('play').textContent=plays>=2?'已达到两次上限':'从头重播（最后一次）';};
$('next').onclick=()=>{const x=items[order[at]],b=document.querySelector('input[name=binding]:checked');
if(!completed)return alert('请先完整播放视频');if(!b||$('numeric').value===''||!$('confidence').value)return alert('请完成三个答案');
responses.push({participant_id:$('participant').value.trim(),item_id:x.item_id,presentation_index:at,play_count:plays,
binding_answer:b.value,numeric_answer:Number($('numeric').value),confidence:Number($('confidence').value)});
at++;if(at<order.length)show();else{$('study').hidden=true;$('done').hidden=false;$('resultText').value=payload();}};
$('download').onclick=()=>{const blob=new Blob([payload()],{type:'application/json'});
const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`qa_v3_${$('participant').value.trim()}.json`;a.click();};
$('copy').onclick=async()=>{const text=payload();try{await navigator.clipboard.writeText(text);$('copy').textContent='已复制';}
catch(e){$('resultText').focus();$('resultText').select();document.execCommand('copy');$('copy').textContent='已复制';}};
</script></html>"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--facts-root", required=True, type=Path)
    parser.add_argument("--media-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--per-profile-limit", type=int)
    args = parser.parse_args(argv)
    if args.output_root.exists() or args.output_root.is_symlink():
        print(f"refusing to overwrite: {args.output_root}", file=sys.stderr)
        return 2
    args.output_root.mkdir(parents=True)
    if args.per_profile_limit is not None and args.per_profile_limit <= 0:
        parser.error("--per-profile-limit must be positive")
    study, _ = build(
        _read(args.selection_manifest), args.facts_root,
        args.media_root, args.output_root,
        per_profile_limit=args.per_profile_limit)
    print(json.dumps({
        "output": str(args.output_root.resolve()),
        "item_count": len(study["items"]),
        "profiles": sorted({item["profile_id"] for item in study["items"]}),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
