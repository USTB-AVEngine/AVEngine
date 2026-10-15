import json
from concurrent.futures import ThreadPoolExecutor
import pytest
from avengine.timeline.v7_kujiale_manifest import append_task
pytestmark=pytest.mark.fast_unit

def test_concurrent_coordinators_cannot_overwrite_another_requested_scene(tmp_path):
    path=tmp_path/'manifest.json';path.write_text(json.dumps({'parents':[]}))
    with ThreadPoolExecutor(max_workers=4)as pool:tasks=list(pool.map(lambda i:append_task(path,{'row':{'scene_id':'scene'+str(i)}}),range(16)))
    data=json.loads(path.read_text());assert len(data['parents'])==16 and {t['slot']for t in tasks}==set(range(16))and len({t['row']['scene_id']for t in data['parents']})==16
