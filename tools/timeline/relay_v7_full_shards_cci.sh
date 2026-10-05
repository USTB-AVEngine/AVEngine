#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
H=/mnt/afs/xinyuanqian/jzy/hardbank
A=$H/generation/v7_full_preproduction1500_20261005
S=/data/jzy/tmp/claude_hardbank_20260923/v7_batch1_20261003/round17_full_preproduction1500_20261005
mode=${1:-production}
if [[ "$mode" != production && "$mode" != smoke ]]; then exit 12; fi
mkdir -p "$A/incoming" "$A/receipts" "$A/readback" "$A/tmp"
exec 8>"$A/relay.lock"
flock -n 8 || exit 13
if [[ "$mode" == smoke ]]; then
 out=$A/transfer_smoke
 ready_source=$S/transfer_smoke/TRANSFER_READY.json
else
 out=$A
 ready_source=$S/TRANSFER_PENDING.json
fi
mkdir -p "$out/incoming" "$out/receipts"
if ! ssh -F "$H/tmp/ssh_config" -n -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 lab48 "test -f '$ready_source'"; then exit 11; fi
stamp=$(date -u +%Y%m%dT%H%M%SZ)
ready=$out/incoming/ready_$stamp.json
rsync -rt --no-owner --no-group --partial -e "ssh -F $H/tmp/ssh_config -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=2" "lab48:$ready_source" "$ready"
mapfile -t contract < <(python3 - "$ready" "$S" "$out" <<'PYS'
from pathlib import Path
import json,sys
r=json.loads(Path(sys.argv[1]).read_text());root=Path(sys.argv[2]);out=Path(sys.argv[3]);index=r['shard_id']
assert type(index)is int and 0<=index<30
p=Path(r['archive']);assert p.is_relative_to(root) and not any(q=='..'for q in p.parts)
assert r['test_house_access']==0 and r['GPU_started'] is False and r['AFS_destination']==str(out/'shards'/f'shard_{index:03d}')
assert len(r['archive_SHA256'])==64 and all(x in '0123456789abcdef'for x in r['archive_SHA256'])
print(str(p));print(r['archive_SHA256']);print(str(r['archive_bytes']));print(f'{index:03d}')
PYS
)
if [[ ${#contract[@]} != 4 ]]; then exit 14; fi
archive_source=${contract[0]};digest=${contract[1]};bytes=${contract[2]};index=${contract[3]}
archive=$out/incoming/shard_${index}_${digest:0:16}.tar.gz
receipt=$out/receipts/shard_$index.json
if [[ "$mode" == smoke && -f "$receipt" ]]; then
 archive=$out/incoming/shard_${index}_${digest:0:16}_${stamp}.tar.gz
 receipt=$out/receipts/shard_${index}_${stamp}.json
fi
started=$(date +%s)
quota_code=$(python3 - "$A/scripts/QUOTA_SOURCE.json" <<'PYS'
from pathlib import Path
import json,sys
p=Path(json.loads(Path(sys.argv[1]).read_text())['code']);assert p.is_relative_to(Path('/mnt/afs/xinyuanqian/jzy/hardbank/generation/v7_s_20261005_batch1/code'));assert (p/'tools/timeline/v7_quota_cache.py').is_file();print(p)
PYS
)
quota_python=$H/generation/cci_datagen_20261003_codex/env/bin/python
PYTHONPATH="$quota_code/src:$quota_code" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
 nice -n 10 "$quota_python" "$quota_code/tools/timeline/v7_quota_cache.py" \
 --owner-root /mnt/afs/xinyuanqian/jzy --task-root "$A" \
 --cache "$A/quota_cache/personal_allocation.json" --expected-bytes "$((3*bytes))" \
 --quota-bytes 2000000000000 --reserve-bytes 80000000000 \
 --receipt "$out/readback/quota_$stamp.json"
transfer_started=$(date +%s)
LC_ALL=C rsync -rt --no-owner --no-group --stats --partial --append-verify -e "ssh -F $H/tmp/ssh_config -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=2" "lab48:$archive_source" "$archive" > "$out/readback/rsync_$stamp.log" 2>&1
finished=$(date +%s)
python3 - "$archive" "$digest" "$bytes" "$started" "$finished" "$out/readback/transfer_$stamp.json" "$transfer_started" "$out/readback/rsync_$stamp.log" <<'PYS'
from pathlib import Path
import hashlib,json,sys,time,re
p=Path(sys.argv[1]);h=hashlib.sha256()
with p.open('rb')as f:
 for b in iter(lambda:f.read(4194304),b''):h.update(b)
assert h.hexdigest()==sys.argv[2] and p.stat().st_size==int(sys.argv[3]);seconds=max(1,int(sys.argv[5])-int(sys.argv[7]));out=Path(sys.argv[6]);out.parent.mkdir(parents=True,exist_ok=True);stats=Path(sys.argv[8]).read_text();match=re.search(r'Total transferred file size: ([0-9,]+) bytes',stats);assert match;transferred=int(match.group(1).replace(',',''))
out.write_text(json.dumps({'actual_archive_bytes':p.stat().st_size,'actual_payload_bytes_transferred':transferred,'actual_rsync_wall_seconds':seconds,'payload_bytes_per_second':transferred/seconds,'quota_preflight_seconds':int(sys.argv[7])-int(sys.argv[4]),'quota_plus_transfer_seconds':int(sys.argv[5])-int(sys.argv[4]),'archive_SHA256_verified':True,'rsync_stats':sys.argv[8],'measured_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2)+'\n')
PYS
P=$H/generation/cci_datagen_20261003_codex/env/bin/python
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PATH=$H/generation/v7_s_20261005_batch1/portable_media_tools_v1/bin:$H/generation/cci_datagen_20261003_codex/env/bin:/usr/local/bin:/usr/bin:/bin
code=$H/generation/v7_s_20261005_batch1/code/d18c0e48ece192dd0214d119335cc3a57f1c98a4
if [[ "$mode" == production ]]; then
 code=$(python3 - "$A/RECEIVER_SOURCE.json" <<'PYS'
from pathlib import Path
import json,sys
d=json.loads(Path(sys.argv[1]).read_text());p=Path(d['code']);assert p.is_relative_to(Path('/mnt/afs/xinyuanqian/jzy/hardbank/generation/v7_full_preproduction1500_20261005/code'));print(p)
PYS
)
fi
export PYTHONPATH=$code/src:$code:$H/generation/cci_datagen_20261003_codex/overlay TMPDIR=$A/tmp
args=()
if [[ "$mode" == smoke ]]; then args+=(--smoke); fi
nice -n 10 "$P" "$A/scripts/verify_v7_transferred_shard.py" --ready "$ready" --archive "$archive" --receipt "$receipt" "${args[@]}" > "$out/readback/media_$stamp.log" 2>&1
if [[ "$mode" == production ]]; then
 ssh -F "$H/tmp/ssh_config" -n -o BatchMode=yes -o ConnectTimeout=20 lab48 "mkdir -p '$S/CCI_RECEIPTS'"
 rsync -rt --no-owner --no-group -e "ssh -F $H/tmp/ssh_config -o BatchMode=yes -o ConnectTimeout=20" "$receipt" "lab48:$S/CCI_RECEIPTS/shard_$index.json"
else
 ssh -F "$H/tmp/ssh_config" -n -o BatchMode=yes -o ConnectTimeout=20 lab48 "mkdir -p '$S/transfer_smoke/CCI_RECEIPTS'"
 rsync -rt --no-owner --no-group -e "ssh -F $H/tmp/ssh_config -o BatchMode=yes -o ConnectTimeout=20" "$receipt" "lab48:$S/transfer_smoke/CCI_RECEIPTS/$(basename "$receipt")"
fi
printf '%s\n' "VERIFIED shard=$index mode=$mode receipt=$receipt"
