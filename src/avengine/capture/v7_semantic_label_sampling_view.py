"""Private sampling view for malformed labels irrelevant to support proposals.

All valid label lines and all stage geometry remain unchanged. A malformed
label for any support surface still raises; no missing colour is guessed.
"""
from pathlib import Path
import csv,hashlib
from avengine.capture.v7_visible_semantic_support import category_kind

def make_semantic_label_sampling_view(source,destination):
    source=Path(source);destination=Path(destination)
    raw=source.read_bytes();lines=raw.splitlines(keepends=True);kept=[];removed=[]
    for index,line in enumerate(lines,1):
        if index==1:kept.append(line);continue
        row=next(csv.reader([line.decode('utf-8')]))
        if len(row)<3:raise ValueError('malformed semantic label row')
        try:bytes.fromhex(row[1])
        except ValueError:
            if category_kind(row[2])is not None:
                raise ValueError('malformed support-surface colour; original label requires repair')
            removed.append({'line_number':index,'object_id':row[0],'colour':row[1],'category':row[2],'original_support_kind':None});continue
        kept.append(line)
    if not removed:raise ValueError('no irrelevant malformed labels require a private view')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb')as f:f.write(b''.join(kept))
    return {'source':str(source),'destination':str(destination),'source_sha256':hashlib.sha256(raw).hexdigest(),'view_sha256':hashlib.sha256(b''.join(kept)).hexdigest(),'removed_irrelevant_rows':removed,'valid_label_lines_byte_identical':True,'geometry_modified':False,'thresholds_truth_and_scorers_modified':False}
