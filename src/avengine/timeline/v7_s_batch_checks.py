"""Read sealed v7-S media and coverage; never manufacture missing supervision."""
from pathlib import Path
from collections import Counter,defaultdict
import json,subprocess
import numpy as np
import soundfile as sf


def check_critical_delivery(root,require_full=True):
    root=Path(root);summary=json.loads((root/'summary.json').read_text());counts=Counter();classes=Counter();split_classes=defaultdict(set);split_speakers=defaultdict(set);questions=Counter();errors=[]
    seeds={};window_pairs=[];verified_binding_ids=set()
    for parent in (root/'parents').iterdir():
        manifest=json.loads((parent/'delivery_manifest.json').read_text())
        group=json.loads((parent/'private/counterfactual_audit.json').read_text())
        verified_binding_ids.update(group.get('verified_binding_logical_question_ids',[]))
        for win in manifest['windows']:
            wid=win['window_id'];ev=json.loads((parent/'private'/(wid+'_evidence.json')).read_text());row=ev['seed_row'];requested=row['production_scenario'];gold=json.loads((parent/'private'/(wid+'_answers.json')).read_text());qs=json.loads((parent/'windows'/wid/'questions.json').read_text())
            seeds[row['scene_id']]=row;byid={r['question_id']:r for r in gold}
            window_pairs.append((qs,gold))
            if len({q['logical_question_id'] for q in qs})!=14:errors.append(wid+': not 14 logical questions')
            speaker=[byid[q['question_id']]['canonical_answer'] for q in qs if q['qa_id']=='STARSS-SPEAKERS']
            category=[byid[q['question_id']]['canonical_answer'] for q in qs if q['qa_id']=='V7-CLASS34']
            if len(speaker)!=1 or int(speaker[0])!=requested['speaker_count']:errors.append(wid+': missing or incorrect actor-count supervision')
            else:counts[int(speaker[0])]+=1;split_speakers[row['split']].add(int(speaker[0]))
            if requested['category_focus_class'] not in category:errors.append(wid+': missing actual target sound-class question')
            for c in category:classes[c]+=1;split_classes[row['split']].add(c)
            questions.update(q['qa_id'] for q in qs)
            if row['split']=='test':errors.append(wid+': test physical house accessed')
    if require_full:
        expected=set(summary['actual_sound_class_emission_counts'])
        from avengine.timeline.v7_production_contract import audit_batch_budget
        allocation=audit_batch_budget(window_pairs,{'STARSS-SPEAKERS':len(window_pairs),'V7-CLASS34':len(window_pairs)},verified_binding_ids=verified_binding_ids)
        if allocation['status']!='pass':errors.extend(allocation['failure_reasons'])
        # Live VCTK uses the dry-library playback label upstream; the separate
        # emitted playback probes also cover speech_playback as a true class.
        if len(seeds)!=500:errors.append('not 500 independent worlds')
        for split in ('train','dev'):
            if split_speakers[split]!=set(range(9)):errors.append(split+': incomplete 0..8 live-actor coverage')
            if len(expected)!=34 or split_classes[split]!=expected:errors.append(split+': incomplete admitted 34-class supervision')
        if summary['binding_percent']<20:errors.append('verified binding below 20 percent')
    return {'status':'pass' if not errors else 'blocked','errors':errors,'independent_worlds':len(seeds),'actual_speaker_count_questions':dict(sorted(counts.items())),'actual_sound_class_questions':dict(sorted(classes.items())),'family_counts':dict(sorted(questions.items())),'modality_counts':summary['modality_counts'],'modality_percent':summary['modality_percent'],'binding_percent':summary['binding_percent'],'test_physical_houses':0 if not any(r['split']=='test' for r in seeds.values()) else None}


def open_delivered_media(root):
    """Actually decode a RGB frame and every FOA/mono file on this machine."""
    root=Path(root);media={}
    for split in ('train','dev'):
        manifest=root/(split+'.jsonl')
        if not manifest.is_file():continue
        for line in manifest.open():
            r=json.loads(line);media.setdefault(r['window_id'],r)
    checked=0;errors=[]
    for wid,r in media.items():
        try:
            video=root/r['video'];foa=root/r['audio_foa'];mono=root/r['audio_mono']
            # Paths must be real delivered files, not a capture identity string.
            if any(not p.is_file() or not p.resolve().is_relative_to(root.resolve()) for p in (video,foa,mono)):raise ValueError('unresolved or escaping actual media path')
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,duration','-of','json',str(video)]))['streams'][0]
            if int(probe['nb_read_frames'])!=150 or abs(float(probe['duration'])-10)>1e-6:raise ValueError('10-second native video clock')
            frame=subprocess.check_output(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-threads','1','-i',str(video),'-frames:v','1','-threads','1','-pix_fmt','rgb24','-f','rawvideo','pipe:1'])
            if len(frame)!=int(probe['width'])*int(probe['height'])*3:raise ValueError('actual RGB decode failed')
            x,sr=sf.read(foa,dtype='float32',always_2d=True);y,ysr=sf.read(mono,dtype='float32',always_2d=True)
            if sr!=16000 or ysr!=sr or x.shape!=(160000,4) or not np.isfinite(x).all() or not np.array_equal(x[:,:1],y):raise ValueError('FOA, W mono or clock differs')
            checked+=1
        except Exception as error:errors.append({'window_id':wid,'error':str(error)})
    return {'status':'pass' if checked and not errors else 'blocked','hostname':__import__('socket').gethostname(),'actual_decoded_windows':checked,'unique_windows':len(media),'errors':errors,'GPU_started':False,'method':'ffprobe all frames + ffmpeg actual first RGB frame + soundfile complete FOA and W-mono read'}
