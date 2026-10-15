"""Append-only native layout diagnostics across automatic reconstruction passes."""
from pathlib import Path
from avengine.timeline.continuous_pilot import write_json


def record_placement_rejection(directory,endpoint,search_pass,payload):
    path=Path(directory)/('attachment_rejection_'+str(endpoint)+'_pass_'+str(search_pass).zfill(3)+'.json')
    write_json(path,payload)
    return path
