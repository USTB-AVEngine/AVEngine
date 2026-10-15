"""Append-only route evidence and a replaceable, task-owned progress snapshot."""
from pathlib import Path
import json,uuid


def write_route_summary(path,payload):
    from avengine.timeline.continuous_pilot import write_json
    path=Path(path);identity=uuid.uuid4().hex
    version=path.with_name(path.stem+'_'+identity+path.suffix)
    write_json(version,payload)
    temporary=path.with_name('.'+path.name+'.'+identity+'.latest')
    temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)
