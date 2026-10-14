"""A policy recheck cannot override an executed native validation failure."""
import json
from tools.assets.recheck_static_dog_motion import qualify_native

def test_native_failure_is_not_readmitted(tmp_path):
    path=tmp_path/"native_motion.json"
    path.write_text(json.dumps({"frame_count":145,"frames":[{}]*145,
        "native_checks_run":True,"checks":{"status":"fail","failures":[{"kind":"clipped"}]}}))
    assert qualify_native(path) is None

def test_claimed_frame_count_requires_actual_frames(tmp_path):
    path=tmp_path/"native_motion.json"
    path.write_text(json.dumps({"frame_count":145,"frames":[{}]*144,
        "native_checks_run":True,"checks":{"status":"pass"}}))
    assert qualify_native(path) is None
