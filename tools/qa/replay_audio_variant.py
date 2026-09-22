#!/usr/bin/env python3
"""Render a bounded audio answer variant over retained native pixels into a fresh episode."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from avengine.qa.audio_variants import prepare_audio_variant


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--speech-manifest",type=Path,required=True)
    p.add_argument("--sound-ids",nargs=2,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--profile",choices=["sequential","overlap","single_first","single_second","three_events"],default="sequential")
    p.add_argument("--swap",action="store_true")
    p.add_argument("--human-nonverbal-classes",nargs="*",default=[])
    p.add_argument("--seed",type=int,default=20260922)
    p.add_argument("--prepare-only",action="store_true")
    a=p.parse_args()
    if not a.prepare_only:
        # Fail before rendering if the caller forgot the existing native
        # soundfile/cffi addon directory in PYTHONPATH.
        import soundfile
    targets=prepare_audio_variant(a.source,a.output,a.speech_manifest,a.sound_ids,repository=ROOT,profile=a.profile,swap=a.swap,seed=a.seed,human_nonverbal_classes=a.human_nonverbal_classes)
    if a.prepare_only:print(json.dumps(targets));return
    from avengine.dataset.binding_group_native import check_requested_visibility
    from avengine.rooms.qa_delivery import finalize_qa_episode
    request=json.loads((a.output/"request.json").read_text());plan=json.loads((a.output/"plan/episode_plan.json").read_text())
    check_requested_visibility(plan,request,a.output/"capture")
    source_review=a.source/"delivery/appearance_review.json"
    if not source_review.is_file():
        refs=json.loads((a.source/"delivery/input_refs.json").read_text())
        if refs.get("appearance_review"):
            source_review=Path(refs["appearance_review"])
    result=finalize_qa_episode(a.output,a.output/"delivery",repository=ROOT,request=request,
        appearance_review=source_review if source_review.is_file() else None)
    (a.output/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    print(json.dumps({"status":result.get("status"),"output":str(a.output)},ensure_ascii=False))

if __name__=="__main__":main()
