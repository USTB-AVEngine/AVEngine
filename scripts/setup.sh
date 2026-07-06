#!/bin/bash
# AVEngine setup: read manifest.yaml, populate external/ dir with
# either symlinks (to local pre-existing clones) or fresh git clones.
#
# Does NOT: install conda envs, download datasets, run sudo.
#
# Usage:
#   bash scripts/setup.sh               # idempotent; populate missing deps
#   bash scripts/setup.sh --update      # for cloned deps, git-checkout to
#                                       # manifest-declared commit
#   bash scripts/setup.sh --force-clone <dep>  # ignore local_hint for <dep>
#   bash scripts/setup.sh --dry-run     # print planned actions, do nothing

set -uo pipefail

# ---- locate AVEngine root ------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AVENGINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFEST="$AVENGINE_ROOT/manifest.yaml"

# ---- parse args -----------------------------------------------------------
UPDATE=0
DRY_RUN=0
FORCE_CLONE_DEP=""
while [ $# -gt 0 ]; do
    case "$1" in
        --update) UPDATE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --force-clone) FORCE_CLONE_DEP="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--update] [--dry-run] [--force-clone <dep>]"
            exit 0 ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done

# ---- helpers --------------------------------------------------------------
say() { echo "[setup] $*"; }
err() { echo "[setup] ERROR: $*" >&2; }

require_manifest() {
    if [ ! -f "$MANIFEST" ]; then
        err "manifest.yaml not found at $MANIFEST"; exit 1
    fi
    if ! python3 -c "import yaml" 2>/dev/null; then
        err "python3 pyyaml not available. Install: pip install pyyaml"; exit 1
    fi
}

# Read a scalar from manifest via python3.
# Usage:  manifest_get <section> <key1> [<key2> ...]
# Example: manifest_get dependencies SPEAR url
# We use argv (not a dotted string) because dep names like "Hunyuan3D-2.1"
# contain literal dots that would be mis-split.
manifest_get() {
    MANIFEST_PATH="$MANIFEST" python3 -c "
import yaml, os, sys
d = yaml.safe_load(open(os.environ['MANIFEST_PATH']))
for k in sys.argv[1:]:
    if isinstance(d, dict) and k in d:
        d = d[k]
    else:
        sys.exit(0)  # missing key -> empty output
print(d if d is not None else '')
" "$@"
}

# List all dep names under dependencies.
manifest_deps() {
    MANIFEST_PATH="$MANIFEST" python3 -c "
import yaml, os
d = yaml.safe_load(open(os.environ['MANIFEST_PATH']))
for k in d.get('dependencies', {}).keys():
    print(k)
"
}

# Compare two git URLs case-insensitively, ignoring trailing '.git' and
# treating ssh+https equivalents as matching.
url_matches() {
    normalize() {
        # Strip trailing .git, lowercase, and normalize ssh vs https form:
        # git@github.com:Org/Repo → github.com/Org/Repo
        # https://github.com/Org/Repo → github.com/Org/Repo
        echo "$1" \
            | sed 's|\.git$||' \
            | sed 's|^git@\([^:]*\):|\1/|' \
            | sed 's|^https\?://||' \
            | sed 's|^ssh://git@||' \
            | tr 'A-Z' 'a-z'
    }
    [ "$(normalize "$1")" = "$(normalize "$2")" ]
}

# ---- per-dep resolution ---------------------------------------------------
resolve_dep() {
    local name="$1"
    local url="$(manifest_get dependencies "$name" url)"
    local commit="$(manifest_get dependencies "$name" commit)"
    local rel_path="$(manifest_get dependencies "$name" path)"
    local upstream="$(manifest_get dependencies "$name" upstream)"
    local hint="$(manifest_get dependencies "$name" local_hint)"
    local target="$AVENGINE_ROOT/$rel_path"

    say ""
    say "==== resolving $name ===="
    say "  url:        $url"
    say "  commit:     $commit"
    say "  target:     $target"
    say "  local_hint: ${hint:-<none>}"

    # --- Case 1: target already exists ---
    if [ -e "$target" ] || [ -L "$target" ]; then
        if [ -L "$target" ]; then
            local link_dst="$(readlink -f "$target")"
            say "  status:     symlink → $link_dst"
        elif [ -d "$target/.git" ]; then
            local origin_url="$(git -C "$target" remote get-url origin 2>/dev/null || echo '<none>')"
            local head_sha="$(git -C "$target" rev-parse HEAD 2>/dev/null || echo '<none>')"
            say "  status:     git repo, origin=$origin_url, HEAD=$head_sha"
            if [ "$UPDATE" = "1" ] && [ "$head_sha" != "$commit" ]; then
                if [ "$DRY_RUN" = "1" ]; then
                    say "  DRY-RUN:    would git fetch && git checkout $commit"
                else
                    (cd "$target" && git fetch --all && git checkout "$commit")
                    say "  updated to $commit"
                fi
            fi
        else
            err "  target exists but is neither symlink nor git repo; refusing to touch. rm -rf $target if you want setup to re-populate."
            return 1
        fi
        return 0
    fi

    # --- Case 2: symlink to local hint (if hint valid AND not forced-clone) ---
    if [ "$FORCE_CLONE_DEP" != "$name" ] && [ -n "$hint" ] && [ -d "$hint/.git" ]; then
        local hint_origin="$(git -C "$hint" remote get-url origin 2>/dev/null || echo '')"
        # Accept if hint origin matches url, upstream, or ANY dep's url/upstream
        # in manifest (last case handles pre-fork state).
        local matches=0
        for candidate in "$url" "$upstream"; do
            if [ -n "$candidate" ] && url_matches "$hint_origin" "$candidate"; then
                matches=1; break
            fi
        done
        if [ "$matches" = "0" ]; then
            while read -r other_dep; do
                local other_url="$(manifest_get dependencies "$other_dep" url)"
                if [ -n "$other_url" ] && url_matches "$hint_origin" "$other_url"; then matches=1; break; fi
                local other_up="$(manifest_get dependencies "$other_dep" upstream)"
                if [ -n "$other_up" ] && url_matches "$hint_origin" "$other_up"; then matches=1; break; fi
            done < <(manifest_deps)
        fi
        if [ "$matches" = "1" ]; then
            if [ "$DRY_RUN" = "1" ]; then
                say "  DRY-RUN:    would symlink $target → $hint"
            else
                mkdir -p "$(dirname "$target")"
                ln -s "$hint" "$target"
                say "  action:     symlinked $target → $hint (local origin=$hint_origin)"
            fi
            return 0
        else
            say "  local_hint exists but origin ($hint_origin) doesn't match url/upstream. Falling through to clone."
        fi
    fi

    # --- Case 3: clone from url and checkout commit ---
    if [ "$DRY_RUN" = "1" ]; then
        say "  DRY-RUN:    would git clone $url $target && git checkout $commit"
        return 0
    fi
    mkdir -p "$(dirname "$target")"
    if ! git clone "$url" "$target"; then
        err "  clone failed. Fix and rerun."
        return 1
    fi
    if ! (cd "$target" && git checkout "$commit"); then
        err "  checkout $commit failed. The commit may not exist upstream yet."
        return 1
    fi
    say "  action:     cloned + checked out $commit"
}

# ---- .setup_state.json ---------------------------------------------------
write_state() {
    local state_file="$AVENGINE_ROOT/.setup_state.json"
    local now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    local user="${USER:-unknown}"
    local host="$(hostname)"
    AVENGINE_ROOT="$AVENGINE_ROOT" MANIFEST="$MANIFEST" \
    STATE_FILE="$state_file" NOW="$now" USR="$user" HOSTNAME_="$host" \
    python3 - <<'PYEOF'
import json, os, yaml
root = os.environ["AVENGINE_ROOT"]
manifest = yaml.safe_load(open(os.environ["MANIFEST"]))
resolved = {}
for name, meta in manifest.get("dependencies", {}).items():
    rel = meta.get("path", os.path.join("external", name))
    p = os.path.join(root, rel)
    if os.path.islink(p):
        resolved[name] = "symlink"
    elif os.path.isdir(os.path.join(p, ".git")):
        resolved[name] = "clone"
    else:
        resolved[name] = "missing"
with open(os.environ["STATE_FILE"], "w") as f:
    json.dump({
        "last_run_utc": os.environ["NOW"],
        "user": os.environ["USR"],
        "host": os.environ["HOSTNAME_"],
        "dependencies_resolved": resolved,
    }, f, indent=2)
print(f"wrote {os.environ['STATE_FILE']}")
PYEOF
}

# ---- next-step hints ------------------------------------------------------
print_next_steps() {
    cat <<EOF

============================================================
setup.sh done.

Next steps (setup.sh does NOT do these — do them manually):

1) Create conda envs:
   for env in spear-env sao-env hunyuan3d-env; do
     conda env create -f envs/\$env.yml
   done

2) Provide external data (see manifest.yaml external_data section):
   - /data/datasets/omniaudio/train-data-az-360-large  (~40 GB AudioSet wavs)
   - /data/datasets/omniaudio/stable-audio-open        (~5 GB SAO model)
   - /data/jzy/code/Hunyuan3D-2.1/pretrained_models    (~20 GB Hunyuan weights)

3) SPEAR internal paths still expect Spatial mesh_library at
   /data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_animalpack
   Until Spec 2 (path parameterization) lands, run manually on collaborator machines:
     sudo mkdir -p /data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library
     sudo ln -s $AVENGINE_ROOT/assets/mesh_library/quaternius_animalpack /data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_animalpack
     sudo ln -s $AVENGINE_ROOT/assets/mesh_library/quaternius_farm       /data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_farm

4) Run first demo:
   conda activate spear-env
   export DISPLAY=:99 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
   python external/SPEAR/tools/gpurir_scenes/scene_two_dogs.py --skip-audio
EOF
}

# ---- main -----------------------------------------------------------------
require_manifest
while read -r dep; do
    resolve_dep "$dep" || exit 1
done < <(manifest_deps)

if [ "$DRY_RUN" = "0" ]; then
    write_state
fi
print_next_steps
