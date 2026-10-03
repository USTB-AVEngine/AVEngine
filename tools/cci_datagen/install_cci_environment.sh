#!/usr/bin/env bash
set -euo pipefail
BASE=${CCI_DATAGEN_BASE:-/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_datagen_20261003_codex}
[[ "$BASE" == /mnt/afs/xinyuanqian/jzy/* ]] || exit 2
INPUT=$BASE/mirror/data/jzy/tmp/claude_hardbank_20260923/cci_datagen_20261003_codex
mkdir -p "$BASE/evidence" "$BASE/tmp" "$BASE/configs"
cd "$BASE/mirror"
sed 's@/data/@data/@' "$INPUT/evidence/environment.sha256" > "$BASE/evidence/environment_relative.sha256"
sha256sum -c "$BASE/evidence/environment_relative.sha256" > "$BASE/evidence/environment_sha256_verification.txt"
for entry in 'env env' 'overlay overlay' 'native native' 'code_base code'; do
  read -r archive dest <<< "$entry"
  if [[ ! -f "$BASE/evidence/unpacked-${archive}.json" ]]; then
    [[ ! -e "$BASE/$dest" ]] || { echo "partial $dest exists; inspect task-owned installation before retry" >&2; exit 3; }
    mkdir "$BASE/$dest"
    tar --no-same-owner -xzf "$INPUT/bundles/$archive.tar.gz" -C "$BASE/$dest"
    printf '{"archive":"%s","destination":"%s"}\n' "$archive" "$BASE/$dest" > "$BASE/evidence/unpacked-${archive}.json"
  fi
done
tar --no-same-owner -xzf "$INPUT/bundles/datagen_tools.tar.gz" -C "$BASE/code"
# Remove only editable import hooks in the copied environment, leaving the
# source environment untouched. All AVEngine source resolves through PYTHONPATH.
"$BASE/env/bin/python" -B - "$BASE" <<'PY'
import json,sys
from pathlib import Path
base=Path(sys.argv[1]);moves=[]
for folder in [base/'env/lib/python3.12/site-packages',base/'overlay']:
 for p in list(folder.glob('*.pth'))+list(folder.glob('*.egg-link')):
  if '/data/jzy/code/' in p.read_text(errors='replace') or 'editable' in p.name:
   target=base/'evidence/disabled-import-hooks'/folder.name/p.name;target.parent.mkdir(parents=True,exist_ok=True)
   p.rename(target);moves.append({'source':str(p),'saved':str(target)})
(base/'evidence/disabled_import_hooks.json').write_text(json.dumps(moves,indent=2)+'\n')
PY
printf "import sys; sys.path.insert(0, '%s/overlay')\n" "$BASE" > "$BASE/env/lib/python3.12/site-packages/cci_generation_overlay.pth"
cat > "$BASE/activate_generation.sh" <<ACTIVATE
export CCI_DATAGEN_BASE='$BASE'
export DATAGEN_PYTHON='$BASE/env/bin/python'
export PATH='$BASE/env/bin':\$PATH
export PYTHONPATH='$BASE/code/src:$BASE/code:$BASE/overlay'
export LD_LIBRARY_PATH='$BASE/private_glvnd:$BASE/env/lib':\${LD_LIBRARY_PATH:-}
export PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR='$BASE/tmp/numba'
export MPLCONFIGDIR='$BASE/tmp/matplotlib'
export XDG_CACHE_HOME='$BASE/tmp/cache'
export TMPDIR='$BASE/tmp'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ACTIVATE
if [[ -f "$INPUT/bundles/private_glvnd.tar.gz" ]]; then
  mkdir -p "$BASE/private_glvnd"
  tar --no-same-owner -xzf "$INPUT/bundles/private_glvnd.tar.gz" -C "$BASE/private_glvnd"
fi
source "$BASE/activate_generation.sh"
"$DATAGEN_PYTHON" -B -m pip freeze --all > "$BASE/evidence/cci_python_packages.txt"
"$DATAGEN_PYTHON" -B "$BASE/code/tools/cci_datagen/native_preflight.py" --output "$BASE/evidence/native_import.json"
