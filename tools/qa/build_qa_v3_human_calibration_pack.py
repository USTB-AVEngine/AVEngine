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


def build(selection, facts_root, media_root, output_root):
    facts_root = Path(facts_root).resolve()
    media_root = Path(media_root).resolve()
    output_root = Path(output_root).resolve()
    media_output = output_root / "media"
    media_output.mkdir(parents=True)
    public_items = []
    answer_items = []
    for record in sorted(selection["selected"], key=lambda row: row["point_id"]):
        profile_id = str(record["profile_id"])
        if profile_id not in CALIBRATION_PROFILES:
            continue
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
    _write(output_root / "study_items.json", study)
    _write(output_root / "answer_key.json", answers)
    (output_root / "index.html").write_text(_HTML, encoding="utf-8")
    (output_root / "README.md").write_text(
        "# QA v3 human calibration pack\n\n"
        "Serve this directory with a static HTTP server. Participants open "
        "`index.html`, use headphones, and download one response JSON. Keep "
        "`answer_key.json` inaccessible to participants. Score collected files "
        "with `tools/qa/score_qa_v3_human_calibration.py`. This is research "
        "calibration, not dataset admission.\n",
        encoding="utf-8")
    return study, answers


_HTML = r"""<!doctype html>
<meta charset="utf-8"><title>QA v3 calibration</title>
<style>body{font:16px system-ui;max-width:900px;margin:2rem auto;padding:0 1rem}
video{width:100%;max-height:520px;background:#111}label{display:block;margin:.8rem 0}
button{padding:.6rem 1rem;margin:.5rem}.muted{color:#666}</style>
<h1>QA v3 full audiovisual calibration</h1>
<p id="instructions"></p><label>Participant ID <input id="participant"></label>
<div id="study" hidden><p id="progress"></p><video id="video" controls></video>
<h3 id="bindingStem"></h3><div id="bindingOptions"></div>
<h3 id="numericStem"></h3><label>Numeric answer <input id="numeric" type="number" step="any"> <span id="unit"></span></label>
<label>Confidence (1–5) <input id="confidence" type="number" min="1" max="5"></label>
<button id="next">Save and continue</button></div><button id="start">Start</button>
<button id="download" hidden>Download responses</button><p class="muted">Gold values are not loaded by this page.</p>
<script>
let items=[],order=[],at=0,responses=[]; const $=id=>document.getElementById(id);
function show(){const x=items[order[at]];$('progress').textContent=`Item ${at+1}/${order.length}`;
$('video').src=x.media;$('bindingStem').textContent=x.binding.stem;$('numericStem').textContent=x.numeric.stem;
$('unit').textContent=x.numeric.unit;$('numeric').value='';$('confidence').value='';
$('bindingOptions').innerHTML=x.binding.options.map((o,i)=>`<label><input type="radio" name="binding" value="${o}">${o}</label>`).join('');}
$('start').onclick=async()=>{if(!$('participant').value.trim())return alert('Participant ID required');
const data=await fetch('study_items.json').then(r=>r.json());items=data.items;$('instructions').textContent=data.instructions;
order=[...items.keys()];for(let i=order.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[order[i],order[j]]=[order[j],order[i]];}
$('start').hidden=true;$('study').hidden=false;show();};
$('next').onclick=()=>{const x=items[order[at]],b=document.querySelector('input[name=binding]:checked');
if(!b||$('numeric').value===''||!$('confidence').value)return alert('Complete all fields');
responses.push({participant_id:$('participant').value.trim(),item_id:x.item_id,presentation_index:at,
binding_answer:b.value,numeric_answer:Number($('numeric').value),confidence:Number($('confidence').value)});
at++;if(at<order.length)show();else{$('study').hidden=true;$('download').hidden=false;}};
$('download').onclick=()=>{const blob=new Blob([JSON.stringify({schema:'qa_v3_human_calibration_responses_v1',responses},null,2)],{type:'application/json'});
const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`qa_v3_${$('participant').value.trim()}.json`;a.click();};
</script>"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--facts-root", required=True, type=Path)
    parser.add_argument("--media-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output_root.exists() or args.output_root.is_symlink():
        print(f"refusing to overwrite: {args.output_root}", file=sys.stderr)
        return 2
    args.output_root.mkdir(parents=True)
    study, _ = build(
        _read(args.selection_manifest), args.facts_root,
        args.media_root, args.output_root)
    print(json.dumps({
        "output": str(args.output_root.resolve()),
        "item_count": len(study["items"]),
        "profiles": sorted({item["profile_id"] for item in study["items"]}),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
