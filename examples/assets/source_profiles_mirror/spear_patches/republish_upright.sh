#!/bin/zsh
set -e
cd /data/jzy/code/AVEngine-lead-a
PY=/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python
IMG=/data/jzy/miniconda3/envs/avengine-imagegen/bin/python
ROOT=/data/avengine_external/assets/sound_source_assets_v1
S=/data/jzy/code/SPEAR-lead-b

report() {
  local ws=$1
  rm -rf $ws/snapshot_profiles_up; mkdir -p $ws/snapshot_profiles_up
  python3 -c "
import json, sys
from pathlib import Path
s=json.load(open(sys.argv[1]+'/inputs/profile_snapshot.json'))
for e in s['profiles']:
    Path(sys.argv[1],'snapshot_profiles_up',e['profile_schema_id']+'.json').write_text(
        json.dumps(e['profile'],ensure_ascii=False,indent=1)+chr(10),encoding='utf-8')
" $ws
  rm -f $ws/token_budget_report_up.json
  $IMG tools/assets/check_prompt_token_budget.py --profile-dir $ws/snapshot_profiles_up \
    --report $ws/token_budget_report_up.json > /dev/null
}

publish() {
  local ws=$1; shift
  report $ws
  $PY tools/assets/publish_static_source_assets.py --root $ROOT \
    --admission-batch $ws/admission_up/static_object_admission_batch_manifest.json \
    --admission-root $ws/admission_up --flux-root $ws/flux --review-root $ws/static_review \
    --profile-snapshot $ws/inputs/profile_snapshot.json \
    --token-budget-report $ws/token_budget_report_up.json "$@"
}

# Drop every static leaf that is about to be replaced, and its index row.
python3 - "$ROOT" <<'PY'
import json, shutil, sys
from pathlib import Path
root = Path(sys.argv[1])
keep_prefix = "audio_playback/bookshelf_speaker/compact_shelf_cabinet"
index = json.loads((root / "index.json").read_text(encoding="utf-8"))
dropped = []
for asset in list(index["assets"]):
    path = asset["path"]
    if not path.startswith("audio_playback"):
        continue
    if path.startswith(keep_prefix):
        continue
    dropped.append(asset["asset_id"])
    shutil.rmtree(root / path, ignore_errors=True)
index["assets"] = [a for a in index["assets"] if a["asset_id"] not in dropped]
(root / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"dropped {len(dropped)} leaves for replacement")
PY

publish $S/tmp/audio_playback_floor_v2
publish $S/tmp/audio_playback_statics_v2 \
  --instance audio_playback_smart_speaker_product_view_352efe1972a4 \
  --instance audio_playback_television_product_view_8569086c6422 \
  --instance audio_playback_television_product_view_cafcf63bf0e8
publish $S/tmp/audio_playback_level_camera_v3 \
  --instance audio_playback_smart_speaker_product_view_282ee90d2dd9 \
  --instance audio_playback_smart_speaker_product_view_b744cb8818e2 \
  --instance audio_playback_soundbar_product_view_3c57b10560b9
publish $S/tmp/audio_playback_statics_v1 \
  --instance audio_playback_soundbar_product_view_61ea3095f68c \
  --instance audio_playback_soundbar_product_view_fdbb7157722f
echo REPUBLISHED
