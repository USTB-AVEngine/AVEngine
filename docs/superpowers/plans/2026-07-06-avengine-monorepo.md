# AVEngine Monorepo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭建 `/data/jzy/code/AVEngine/` 顶层 monorepo，让合作者能一键 setup 拿到跑 audio-visual pipeline 所需的所有代码 + 资产。

**Architecture:** 新 git repo 作为 wrapper：`manifest.yaml` 单一真相源声明 2 个 external 依赖（SPEAR fork + Hunyuan3D-2.1 upstream），`scripts/setup.sh` 读 manifest 自适应处理（本机 symlink / 别人机器 clone），`assets/mesh_library/` 直接 add 进 git（实际只 4.6 MB），3 个 conda env yml 用 `--from-history + pip freeze` 生成，README 单页 quickstart。setup.sh 只拉仓，env 和数据集靠用户手动。

**Tech Stack:** Bash 4+ / Python 3.11 pyyaml (系统已装) / git 2.x。无新 runtime 依赖。

## Global Constraints

- 顶层目录名：`AVEngine`，位于 `/data/jzy/code/AVEngine/`（新建）
- 不改现有 `/data/jzy/code/SPEAR/`、`/data/jzy/code/Hunyuan3D-2.1/`、`/data/jzy/code/Spatial/` 的内容或位置
- `assets/mesh_library/` 从 `/data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/` cp（**仅 quaternius_animalpack + quaternius_farm 两个子目录**，共 4.6 MB / 12 个 GLB 文件）
- SPEAR fork 位置：`https://github.com/Eastforward/spear.git`（用户已 fork，尚未 push commits）
- Hunyuan3D-2.1 upstream：`https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git`
- 三个 conda env：`spear-env` (py3.11), `sao-env` (py3.10), `hunyuan3d` (py3.10) — 都已在 `/data/jzy/miniconda3/envs/` 存在
- setup.sh **不** 装 conda env、**不** 下 external data、**不** sudo
- 每 Task 独立 commit（AVEngine repo 内部）
- 平台限定 Linux（UE + Vulkan 需求）
- Out-of-scope：SPEAR 硬编码路径 refactor（Spec 2）、CI、LFS、mac/win 支持

---

## File Structure

**新增文件树**（都在 `/data/jzy/code/AVEngine/` 内）：

```
AVEngine/
├── README.md                        # Task 10 (quickstart)
├── LICENSE                          # Task 2 (占位)
├── manifest.yaml                    # Task 3
├── .gitignore                       # Task 2
├── scripts/
│   ├── setup.sh                     # Task 4
│   └── update.sh                    # Task 5
├── envs/
│   ├── spear-env.yml                # Task 6
│   ├── sao-env.yml                  # Task 6
│   └── hunyuan3d-env.yml            # Task 6
├── assets/
│   └── mesh_library/                # Task 7 (cp from Spatial)
│       ├── README.md                # Task 7
│       ├── quaternius_animalpack/   # 5 GLB, 2.6 MB
│       └── quaternius_farm/         # 7 GLB, 2.0 MB
├── docs/
│   ├── pipeline_zh.md               # Task 8 (cp from SPEAR)
│   ├── pipeline_en.md               # Task 8
│   ├── quickstart.md                # Task 10
│   ├── troubleshooting.md           # Task 10
│   └── superpowers/
│       ├── specs/                   # Task 8 (cp)
│       └── plans/                   # Task 8 (cp)
├── external/                        # git-ignored；setup.sh 填充
│   └── .gitkeep                     # Task 2
└── .setup_state.json                # git-ignored；setup.sh 写
```

**依赖矩阵**：Task 1 (前置 push) → Tasks 2-8 (repo 搭建) → Task 9 (setup.sh 本机验证) → Task 10 (README) → Task 11 (V1-V6 acceptance)

---

## Task 1: 前置动作 — SPEAR push 到 fork + 记录 SHAs

**Files:**
- Modify: `/data/jzy/code/SPEAR/.git/config`（加 remote，非 tracked file）
- 产出：两个 SHA 字符串，供 Task 3 manifest 填入

**Interfaces:**
- Consumes: 无
- Produces:
  - `SPEAR_HEAD_SHA`：40-hex，Task 3 manifest.yaml 里 SPEAR.commit
  - `HUNYUAN_HEAD_SHA`：40-hex，Task 3 manifest.yaml 里 Hunyuan3D-2.1.commit

- [ ] **Step 1.1: 加 Eastforward remote 到 SPEAR**

Run:
```bash
cd /data/jzy/code/SPEAR
git remote add eastforward https://github.com/Eastforward/spear.git 2>&1 || \
  git remote set-url eastforward https://github.com/Eastforward/spear.git
git remote -v | grep eastforward
```

Expected: 输出两行
```
eastforward	https://github.com/Eastforward/spear.git (fetch)
eastforward	https://github.com/Eastforward/spear.git (push)
```

- [ ] **Step 1.2: Push 49 commits 到 Eastforward/spear main**

Run:
```bash
cd /data/jzy/code/SPEAR
git push -u eastforward main 2>&1 | tail -5
```

Expected: 无 `[rejected]` 字样；末尾类似 `To https://github.com/Eastforward/spear.git   * [new branch]      main -> main` 或 push 成功。

若 rejected（因为 fork 本身 default branch 也有内容）：
```bash
git push -u eastforward main --force-with-lease 2>&1 | tail -5
```

- [ ] **Step 1.3: 记录 SPEAR HEAD SHA**

Run:
```bash
SPEAR_HEAD_SHA=$(cd /data/jzy/code/SPEAR && git rev-parse HEAD)
echo "SPEAR_HEAD_SHA=$SPEAR_HEAD_SHA"
```

Expected: `bc8ce3236cc8c9f06b84d3764d5211cad747fd8c` 或更新（如你 push 前又新加了 commit）。**记下这个 SHA**，Task 3 要用。

- [ ] **Step 1.4: 记录 Hunyuan3D-2.1 HEAD SHA**

Run:
```bash
HUNYUAN_HEAD_SHA=$(cd /data/jzy/code/Hunyuan3D-2.1 && git rev-parse HEAD)
echo "HUNYUAN_HEAD_SHA=$HUNYUAN_HEAD_SHA"
```

Expected: `82920d643c0dc2f7bfd7255f45f62d386edfe60c`（当前值；若 upstream 后续有更新可能不同）。**记下这个 SHA**。

- [ ] **Step 1.5: 无 commit（Task 1 只是准备工作）**

跳过 —— 本 task 无 AVEngine repo 内容修改。SHA 值传递到 Task 3。

---

## Task 2: 新建 AVEngine 空目录骨架 + git init

**Files:**
- Create: `/data/jzy/code/AVEngine/` (目录)
- Create: `/data/jzy/code/AVEngine/.gitignore`
- Create: `/data/jzy/code/AVEngine/LICENSE`
- Create: `/data/jzy/code/AVEngine/external/.gitkeep`

**Interfaces:**
- Consumes: 无
- Produces: 一个空 git repo，含 .gitignore 和 LICENSE 占位

- [ ] **Step 2.1: 创建目录树**

Run:
```bash
mkdir -p /data/jzy/code/AVEngine/{scripts,envs,assets/mesh_library,docs/superpowers/{specs,plans},external}
touch /data/jzy/code/AVEngine/external/.gitkeep
ls /data/jzy/code/AVEngine
```

Expected: 输出 `assets  docs  envs  external  scripts`

- [ ] **Step 2.2: git init**

Run:
```bash
cd /data/jzy/code/AVEngine
git init -b main 2>&1 | tail -3
```

Expected: `Initialized empty Git repository in /data/jzy/code/AVEngine/.git/`

- [ ] **Step 2.3: 写 .gitignore**

Create `/data/jzy/code/AVEngine/.gitignore`:

```gitignore
# external repos are populated by scripts/setup.sh, not tracked here
external/*
!external/.gitkeep

# setup state (per-machine)
.setup_state.json

# python
__pycache__/
*.py[cod]
*.egg-info/

# editor/OS
.DS_Store
.idea/
.vscode/
*.swp

# any tmp/ inside AVEngine (mirrors SPEAR convention)
tmp/
```

- [ ] **Step 2.4: 写 LICENSE 占位**

Create `/data/jzy/code/AVEngine/LICENSE`:

```
Copyright (c) 2026 Ziyang Ji (Eastforward)

All rights reserved. This project is currently under private research
development and is NOT yet released as open source.

Contact: <ziyang's email> for research collaboration or reuse questions.
```

- [ ] **Step 2.5: 初次 commit**

Run:
```bash
cd /data/jzy/code/AVEngine
git add .gitignore LICENSE external/.gitkeep
git commit -m "chore: initial AVEngine skeleton (empty dirs + gitignore + LICENSE)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" 2>&1 | tail -3
```

Expected: `[main <sha>] chore: initial AVEngine skeleton ...`

---

## Task 3: manifest.yaml — 依赖单一真相源

**Files:**
- Create: `/data/jzy/code/AVEngine/manifest.yaml`

**Interfaces:**
- Consumes: `SPEAR_HEAD_SHA`, `HUNYUAN_HEAD_SHA` (Task 1)
- Produces: YAML file that Task 4 (setup.sh) will parse via `python3 -c "import yaml; ..."`

**Schema fields** (per spec §4):
- `version: 1`
- `dependencies.{SPEAR,Hunyuan3D-2.1}`: url / commit / path / upstream / local_hint
- `external_data.{omniaudio_wavs,stable_audio_open,hunyuan3d_weights}`: expected_path / description / source / size_estimate / used_by
- `python_environments.{spear-env,sao-env,hunyuan3d-env}`: yml / python / critical_packages

- [ ] **Step 3.1: 创建 manifest.yaml**

Create `/data/jzy/code/AVEngine/manifest.yaml`（**把 `<SPEAR_SHA>` 和 `<HUNYUAN_SHA>` 替换成 Task 1 记下的实际 SHA**）:

```yaml
# AVEngine dependency manifest — single source of truth.
# scripts/setup.sh reads this. If you add a new external repo here,
# rerun `bash scripts/setup.sh` and it will pick it up.

version: 1

dependencies:
  SPEAR:
    url: https://github.com/Eastforward/spear.git
    commit: <SPEAR_SHA>                                # replace with Task 1.3 output
    path: external/SPEAR
    upstream: https://github.com/spear-sim/spear.git
    local_hint: /data/jzy/code/SPEAR                   # AUTHOR'S LOCAL PATH.
                                                       # setup.sh checks: if this
                                                       # absolute path exists AND its
                                                       # git origin matches `url` or
                                                       # `upstream`, symlink instead
                                                       # of cloning. On any other
                                                       # machine (path won't exist) →
                                                       # clone. Collaborators leave
                                                       # this field untouched.

  Hunyuan3D-2.1:
    url: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
    commit: <HUNYUAN_SHA>                              # replace with Task 1.4 output
    path: external/Hunyuan3D-2.1
    upstream: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
    local_hint: /data/jzy/code/Hunyuan3D-2.1

external_data:
  # These are NOT downloaded by setup.sh. Collaborators must obtain and
  # place them at the paths below (or symlink to their own copies) BEFORE
  # running pipeline. Loader code in SPEAR reads from these paths.

  omniaudio_wavs:
    expected_path: /data/datasets/omniaudio/train-data-az-360-large
    description: "~58k AudioSet .wav clips indexed by keyword"
    source: "Contact <author> or reproduce from AudioSet dumps"
    size_estimate: "~40 GB"
    used_by: SPEAR/tools/gpurir_scenes/audio_registry.py

  stable_audio_open:
    expected_path: /data/datasets/omniaudio/stable-audio-open
    description: "Stable Audio Open 1.0 model weights"
    source: https://huggingface.co/stabilityai/stable-audio-open-1.0
    size_estimate: "~5 GB"
    used_by: SPEAR/tools/gpurir_scenes/gen_sao_clip.py

  hunyuan3d_weights:
    expected_path: /data/jzy/code/Hunyuan3D-2.1/pretrained_models
    description: "Hunyuan3D-2.1 shape + paint weights"
    source: https://huggingface.co/Tencent-Hunyuan/Hunyuan3D-2.1
    size_estimate: "~20 GB"
    used_by: SPEAR/tools/hy3d_bake_diffuse.py

python_environments:
  # setup.sh does NOT create these. README instructs manual creation.
  spear-env:
    yml: envs/spear-env.yml
    python: "3.11"
    critical_packages: [spear_ext]
  sao-env:
    yml: envs/sao-env.yml
    python: "3.10"
    critical_packages: [torch>=2.7, stable-audio-tools, gpuRIR==1.2.0, numpy<2]
  hunyuan3d-env:
    yml: envs/hunyuan3d-env.yml
    python: "3.10"
    critical_packages: [diffusers, hy3dgen]
```

- [ ] **Step 3.2: 验证 YAML 语法**

Run:
```bash
python3 -c "import yaml; d=yaml.safe_load(open('/data/jzy/code/AVEngine/manifest.yaml')); print('deps:', list(d['dependencies'].keys())); print('SPEAR commit:', d['dependencies']['SPEAR']['commit'])"
```

Expected:
```
deps: ['SPEAR', 'Hunyuan3D-2.1']
SPEAR commit: <bc8ce323... 或你 Task 1 记的 SHA>
```

**若 SPEAR commit 输出还是 `<SPEAR_SHA>` 字面量** — 忘替换了。回 Step 3.1 换值。

- [ ] **Step 3.3: Commit**

Run:
```bash
cd /data/jzy/code/AVEngine
git add manifest.yaml
git commit -m "feat: add manifest.yaml declaring SPEAR + Hunyuan3D-2.1 deps + external_data provenance

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" 2>&1 | tail -3
```

Expected: `[main <sha>] feat: add manifest.yaml ...`

---

## Task 4: scripts/setup.sh — 拉仓核心逻辑

**Files:**
- Create: `/data/jzy/code/AVEngine/scripts/setup.sh`

**Interfaces:**
- Consumes: `manifest.yaml` (Task 3)
- Produces:
  - `external/<dep>` filled per manifest (symlink or clone)
  - `.setup_state.json` with per-machine record
  - CLI flags: no flag / `--update` / `--force-clone <dep>` / `--dry-run`

- [ ] **Step 4.1: 创建 setup.sh**

Create `/data/jzy/code/AVEngine/scripts/setup.sh`:

```bash
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
# usage: manifest_get "dependencies.SPEAR.url"
manifest_get() {
    python3 -c "
import yaml, sys
d = yaml.safe_load(open('$MANIFEST'))
for k in '$1'.split('.'):
    d = d[k]
print(d)
"
}

# List all dep names under dependencies.
manifest_deps() {
    python3 -c "
import yaml
d = yaml.safe_load(open('$MANIFEST'))
for k in d.get('dependencies', {}).keys():
    print(k)
"
}

# Compare two git URLs case-insensitively, ignoring trailing '.git'.
url_matches() {
    local a="$(echo "$1" | sed 's|\.git$||' | tr 'A-Z' 'a-z')"
    local b="$(echo "$2" | sed 's|\.git$||' | tr 'A-Z' 'a-z')"
    [ "$a" = "$b" ]
}

# ---- per-dep resolution ---------------------------------------------------
resolve_dep() {
    local name="$1"
    local url="$(manifest_get "dependencies.$name.url")"
    local commit="$(manifest_get "dependencies.$name.commit")"
    local rel_path="$(manifest_get "dependencies.$name.path")"
    local upstream="$(manifest_get "dependencies.$name.upstream" 2>/dev/null || echo "")"
    local hint="$(manifest_get "dependencies.$name.local_hint" 2>/dev/null || echo "")"
    local target="$AVENGINE_ROOT/$rel_path"

    say ""
    say "==== resolving $name ===="
    say "  url:        $url"
    say "  commit:     $commit"
    say "  target:     $target"
    say "  local_hint: $hint"

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
        # Accept if hint origin matches url, upstream, or ANY dep's url in manifest
        # (last case handles pre-fork state: local SPEAR still points at spear-sim/spear
        # while manifest already points at Eastforward/spear).
        local matches=0
        for candidate in "$url" "$upstream"; do
            if [ -n "$candidate" ] && url_matches "$hint_origin" "$candidate"; then
                matches=1; break
            fi
        done
        if [ "$matches" = "0" ]; then
            # Also check all other dep URLs (safety net for renamed forks)
            while read -r other_dep; do
                local other_url="$(manifest_get "dependencies.$other_dep.url")"
                if url_matches "$hint_origin" "$other_url"; then matches=1; break; fi
                local other_up="$(manifest_get "dependencies.$other_dep.upstream" 2>/dev/null || echo '')"
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
    # All state-derivation happens in one self-contained Python block that
    # reads manifest.yaml itself — no bash-side variable injection into
    # Python source, no fragile quoting.
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
```

- [ ] **Step 4.2: 赋可执行位**

Run:
```bash
chmod +x /data/jzy/code/AVEngine/scripts/setup.sh
ls -l /data/jzy/code/AVEngine/scripts/setup.sh
```

Expected: `-rwxr-xr-x ... setup.sh`

- [ ] **Step 4.3: 首次 --dry-run 验证脚本能解析 manifest**

Run:
```bash
cd /data/jzy/code/AVEngine
bash scripts/setup.sh --dry-run 2>&1 | tail -20
```

Expected: 输出含 `==== resolving SPEAR ====` 和 `==== resolving Hunyuan3D-2.1 ====`，每个都说 `DRY-RUN: would symlink ... → /data/jzy/code/...`（因为你本机 local_hint 都有效）。**不应有 ERROR 或 Traceback**。

**若报 Python 或 yaml 错**：确认 `python3 -c "import yaml"` 能 import；本机已确认可以，若不行 `pip install --user pyyaml`。

- [ ] **Step 4.4: Commit setup.sh**

Run:
```bash
cd /data/jzy/code/AVEngine
git add scripts/setup.sh
git commit -m "feat: scripts/setup.sh — read manifest, symlink or clone external deps

- 4 scenarios: your machine (symlink to local_hint), collaborator machine
  (clone), idempotent re-run, --update.
- Does not install conda envs, does not download datasets, does not sudo.
- Prints next-step hints for env creation + data provisioning + SPEAR path
  workaround (until Spec 2 lands).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" 2>&1 | tail -3
```

Expected: `[main <sha>] feat: scripts/setup.sh ...`

---

## Task 5: scripts/update.sh — 便捷别名

**Files:**
- Create: `/data/jzy/code/AVEngine/scripts/update.sh`

**Interfaces:**
- Consumes: `scripts/setup.sh` (Task 4)
- Produces: 一行 wrapper，`bash scripts/update.sh` == `bash scripts/setup.sh --update`

- [ ] **Step 5.1: 创建 update.sh**

Create `/data/jzy/code/AVEngine/scripts/update.sh`:

```bash
#!/bin/bash
# Convenience alias for `bash scripts/setup.sh --update`.
# Only affects cloned external repos (symlinks are left alone; use
# `git pull` in the linked directory instead).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/setup.sh" --update "$@"
```

- [ ] **Step 5.2: 赋可执行位 + verify**

Run:
```bash
chmod +x /data/jzy/code/AVEngine/scripts/update.sh
cd /data/jzy/code/AVEngine
bash scripts/update.sh --dry-run 2>&1 | tail -5
```

Expected: 与 Task 4.3 同样输出（因为 update 就是 setup 加 --update flag）。

- [ ] **Step 5.3: Commit**

Run:
```bash
cd /data/jzy/code/AVEngine
git add scripts/update.sh
git commit -m "feat: scripts/update.sh — alias for 'setup.sh --update'

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" 2>&1 | tail -3
```

---

## Task 6: envs/*.yml — 3 个 conda env 导出

**Files:**
- Create: `/data/jzy/code/AVEngine/envs/spear-env.yml`
- Create: `/data/jzy/code/AVEngine/envs/sao-env.yml`
- Create: `/data/jzy/code/AVEngine/envs/hunyuan3d-env.yml`

**Interfaces:**
- Consumes: 三个 activate-able conda envs (`/data/jzy/miniconda3/envs/{spear-env,sao-env,hunyuan3d}`)
- Produces: 3 yml files that `conda env create -f <yml> --dry-run` can parse

- [ ] **Step 6.1: 写 helper 一次性导出 3 个 env**

Create temporary helper `/tmp/dump_env.sh`:

```bash
#!/bin/bash
# Usage: bash /tmp/dump_env.sh <env-name> <out-yml> <canonical-name>
ENV="$1"
OUT="$2"
CANONICAL="$3"

CONDA_BIN=/data/jzy/miniconda3/bin/conda

# Base env: `conda env export --from-history` — user-declared conda pkgs only
CONDA_BIN=$CONDA_BIN "$CONDA_BIN" env export -n "$ENV" --from-history > "$OUT.tmp"

# Rewrite 'name:' line to canonical
sed -i "s/^name: .*/name: $CANONICAL/" "$OUT.tmp"

# Append pip freeze (skip editable and comments); place under existing
# pip section if present, else create one.
PIP_LIST=$(/data/jzy/miniconda3/envs/"$ENV"/bin/pip freeze 2>/dev/null | \
    grep -vE "^-e |^#|^@|^git\+" | sort -u)

if grep -q "^  - pip:" "$OUT.tmp"; then
    :   # existing pip block; user must merge manually
else
    # Ensure last line is not the auto-generated 'prefix:' — strip and re-add
    grep -v "^prefix:" "$OUT.tmp" > "$OUT.tmp2" && mv "$OUT.tmp2" "$OUT.tmp"
    echo "  - pip:" >> "$OUT.tmp"
    while IFS= read -r pkg; do
        echo "    - $pkg" >> "$OUT.tmp"
    done <<< "$PIP_LIST"
fi

mv "$OUT.tmp" "$OUT"
echo "wrote $OUT"
```

Run:
```bash
chmod +x /tmp/dump_env.sh
```

- [ ] **Step 6.2: 导出 spear-env**

Run:
```bash
bash /tmp/dump_env.sh spear-env /data/jzy/code/AVEngine/envs/spear-env.yml spear-env
wc -l /data/jzy/code/AVEngine/envs/spear-env.yml
head -5 /data/jzy/code/AVEngine/envs/spear-env.yml
```

Expected: `name: spear-env` at top, some conda deps, then `- pip:` section, total 50-300 lines.

- [ ] **Step 6.3: 导出 sao-env**

Run:
```bash
bash /tmp/dump_env.sh sao-env /data/jzy/code/AVEngine/envs/sao-env.yml sao-env
head -5 /data/jzy/code/AVEngine/envs/sao-env.yml
```

Expected: `name: sao-env` at top.

- [ ] **Step 6.4: 导出 hunyuan3d → 命名为 hunyuan3d-env.yml**

Run:
```bash
bash /tmp/dump_env.sh hunyuan3d /data/jzy/code/AVEngine/envs/hunyuan3d-env.yml hunyuan3d-env
head -5 /data/jzy/code/AVEngine/envs/hunyuan3d-env.yml
```

Expected: `name: hunyuan3d-env` at top.

- [ ] **Step 6.5: 手动清理 editable installs 和硬编码本机路径**

对每个 yml：搜索并处理危险 pip 行。

Run:
```bash
for f in /data/jzy/code/AVEngine/envs/*.yml; do
  echo "==== $f ===="
  grep -nE "^    - -e|/data/jzy|/home/jzy|@ file://" "$f" || echo "  (clean)"
done
```

Expected: 若有输出，需要手动处理（打开 yml 删除或注释掉这些行）。以下是**必须处理**的模式：

- `- -e /path/to/xxx` — editable install，本机 dev 用；改成注释 `# -e /path/to/xxx  # editable install; skipped for portability`
- 任何 `@ file:///` 开头的 pip 行 — 硬编码本机路径
- `- prefix: ...` line — 若还有（不该有），删掉

若 spear-env 里含 `spear_ext` 的 pip 行且指向本机路径，把它替换成一个注释：
```yaml
    # NOTE: spear_ext is built from SPEAR/cpp/. Install via:
    #   cd external/SPEAR && ./tools/build_extension.sh   (see SPEAR README)
```

- [ ] **Step 6.6: Dry-parse verify 3 个 yml**

Run:
```bash
for env in spear-env sao-env hunyuan3d-env; do
  yml="/data/jzy/code/AVEngine/envs/$env.yml"
  echo "==== $env ===="
  python3 -c "
import yaml, sys
try:
    d = yaml.safe_load(open('$yml'))
    print('OK name:', d.get('name'))
    print('   deps:', len(d.get('dependencies', [])))
except Exception as e:
    print('FAIL:', e); sys.exit(1)
"
done
```

Expected: 3 个 `OK name:` 行，且每个 `deps:` count > 0。

- [ ] **Step 6.7: Commit**

Run:
```bash
cd /data/jzy/code/AVEngine
git add envs/
git commit -m "feat: envs/*.yml — three conda envs (spear-env, sao-env, hunyuan3d-env)

Generated via 'conda env export --from-history' + 'pip freeze' filter.
Editable installs and local-path pip refs stripped or annotated.
spear_ext is annotated as build-from-source (not pip-installable).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" 2>&1 | tail -3
```

---

## Task 7: assets/mesh_library — cp from Spatial + license readme

**Files:**
- Create: `/data/jzy/code/AVEngine/assets/mesh_library/quaternius_animalpack/*.glb` (5 files, cp)
- Create: `/data/jzy/code/AVEngine/assets/mesh_library/quaternius_farm/*.glb` (7 files, cp)
- Create: `/data/jzy/code/AVEngine/assets/mesh_library/README.md`

**Interfaces:**
- Consumes: `/data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/{quaternius_animalpack,quaternius_farm}/`
- Produces: 12 GLB files + README under AVEngine/assets/mesh_library/

- [ ] **Step 7.1: cp mesh_library**

Run:
```bash
cd /data/jzy/code/AVEngine
cp -r --preserve=all /data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_animalpack assets/mesh_library/
cp -r --preserve=all /data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_farm assets/mesh_library/
```

- [ ] **Step 7.2: 验证 12 GLB + size sanity**

Run:
```bash
find /data/jzy/code/AVEngine/assets/mesh_library -name '*.glb' | wc -l
du -sh /data/jzy/code/AVEngine/assets/mesh_library
ls /data/jzy/code/AVEngine/assets/mesh_library/quaternius_animalpack/
ls /data/jzy/code/AVEngine/assets/mesh_library/quaternius_farm/
```

Expected:
- File count: `12`
- Size: `~4.6M`（不超过 200 MB）
- animalpack: `Cat.glb  Dog.glb  Eagle.glb  Piranha.glb  Wolf.glb`
- farm: `Cow.glb  Horse.glb  Llama.glb  Pig.glb  Pug.glb  Sheep.glb  Zebra.glb`

- [ ] **Step 7.3: 写 mesh_library/README.md**

Create `/data/jzy/code/AVEngine/assets/mesh_library/README.md`:

```markdown
# mesh_library

Rigged 3D animal models used by the AVEngine pipeline.

## Contents

- `quaternius_animalpack/` — 5 GLB: Cat, Dog, Eagle, Piranha, Wolf (2.6 MB)
- `quaternius_farm/` — 7 GLB: Cow, Horse, Llama, Pig, Pug, Sheep, Zebra (2.0 MB)

Total: 12 GLB, ~4.6 MB.

## Provenance

All meshes are from **Quaternius** free 3D asset packs:
- Animal Pack: https://quaternius.com/packs/animatedanimals.html
- Farm Animal Pack: https://quaternius.com/packs/animatedfarmanimals.html

## License

Both packs are released by Quaternius under **CC0 1.0 Universal (Public Domain)**:
https://creativecommons.org/publicdomain/zero/1.0/

Redistribution as part of AVEngine is unrestricted.

## Usage by AVEngine pipeline

Referenced from `external/SPEAR/tools/species_rig_map.py::QUATERNIUS_DIR`.
Currently only `Cat.glb` and `Dog.glb` are actively used (map to 5 animated
tags: dog_golden, dog_husky, cat_persian, cat_tabby, chipmunk). Other GLBs
kept for future rig-family expansion (e.g., Horse for future animated
ungulates once robust_skin_transfer supports semantic bone names).

## ⚠ Hardcoded path caveat

Until Spec 2 (SPEAR path parameterization) lands, SPEAR reads these from
`/data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_*`.
On collaborator machines, symlink after `scripts/setup.sh` (see setup.sh
"Next steps" output).
```

- [ ] **Step 7.4: Commit**

Run:
```bash
cd /data/jzy/code/AVEngine
git add assets/mesh_library/
git commit -m "feat: assets/mesh_library — Quaternius animal + farm GLBs (12 files, 4.6 MB, CC0)

Copied verbatim from /data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/.
Cat.glb + Dog.glb are actively used by SPEAR pipeline (5 animated tags).
Others kept for future rig-family expansion.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" 2>&1 | tail -3
```

Expected: 一个 commit 含 13 个文件（12 GLB + 1 README）。

---

## Task 8: docs/ — 搬 pipeline 文档 + spec + plan

**Files:**
- Create: `/data/jzy/code/AVEngine/docs/pipeline_zh.md` (cp)
- Create: `/data/jzy/code/AVEngine/docs/pipeline_en.md` (cp)
- Create: `/data/jzy/code/AVEngine/docs/superpowers/specs/2026-07-06-avengine-monorepo-design.md` (cp)
- Create: `/data/jzy/code/AVEngine/docs/superpowers/specs/2026-07-06-apartment-furniture-collision-design.md` (cp)
- Create: `/data/jzy/code/AVEngine/docs/superpowers/plans/2026-07-06-apartment-furniture-collision.md` (cp)
- Create: `/data/jzy/code/AVEngine/docs/superpowers/plans/2026-07-06-avengine-monorepo.md` (cp — 本 plan 自身)

**Interfaces:**
- Consumes: SPEAR/docs/ 的文档
- Produces: AVEngine/docs/ 的完整文档树

**注意**：**cp 而不是 mv**。SPEAR 里的原始文件保留，让 SPEAR repo 依然完整。

- [ ] **Step 8.1: cp pipeline 文档**

Run:
```bash
cd /data/jzy/code/AVEngine
cp /data/jzy/code/SPEAR/docs/current_animal_audio_visual_dataset_pipeline_zh.md docs/pipeline_zh.md
cp /data/jzy/code/SPEAR/docs/current_animal_audio_visual_dataset_pipeline.md docs/pipeline_en.md
# Also cp the assets/pipeline/ image dir referenced by pipeline_zh.md
cp -r /data/jzy/code/SPEAR/docs/assets docs/
ls docs/
ls docs/assets/pipeline/ | head
```

Expected: `docs/` 含 `pipeline_zh.md pipeline_en.md assets superpowers`。`docs/assets/pipeline/` 含 6 张 png/jpg + 2 mp4。

- [ ] **Step 8.2: cp superpowers spec + plan**

Run:
```bash
cd /data/jzy/code/AVEngine
cp /data/jzy/code/SPEAR/docs/superpowers/specs/2026-07-06-avengine-monorepo-design.md docs/superpowers/specs/
cp /data/jzy/code/SPEAR/docs/superpowers/specs/2026-07-06-apartment-furniture-collision-design.md docs/superpowers/specs/
cp /data/jzy/code/SPEAR/docs/superpowers/plans/2026-07-06-apartment-furniture-collision.md docs/superpowers/plans/
cp /data/jzy/code/SPEAR/docs/superpowers/plans/2026-07-06-avengine-monorepo.md docs/superpowers/plans/
ls docs/superpowers/specs docs/superpowers/plans
```

Expected: `specs` 有 2 个 md，`plans` 有 2 个 md。

- [ ] **Step 8.3: 写 quickstart.md 和 troubleshooting.md 骨架**

Create `/data/jzy/code/AVEngine/docs/quickstart.md`:

```markdown
# Quickstart

See top-level [`README.md`](../README.md) first.

This doc drills into details: env creation troubleshooting, data set
provisioning, running individual pipeline stages.

## Environment creation gotchas

TODO: filled after first collaborator runs setup.
```

Create `/data/jzy/code/AVEngine/docs/troubleshooting.md`:

```markdown
# Troubleshooting

## setup.sh reports "target exists but is neither symlink nor git repo"

Something (not setup.sh) put a plain directory at `external/<dep>`. Delete
it and rerun:
```bash
rm -rf external/<dep>
bash scripts/setup.sh
```

## setup.sh reports "local_hint exists but origin doesn't match"

Your local pre-existing clone at `local_hint` has a different git remote
than manifest expects. Either (a) update that clone's origin, or (b) use
`--force-clone <dep>` to clone into `external/` instead.

## conda env create fails on spear_ext

`spear_ext` is a compiled C++ extension. Build it manually:
```bash
cd external/SPEAR
# See SPEAR README for build instructions.
```

## Pipeline reports "apartment_furniture_map.json not found"

Confirm `external/SPEAR/data/apartment_furniture_map.json` exists.
This file ships with SPEAR at commit >= bc8ce323.

## Pipeline reports "Quaternius rig not found at /data/jzy/code/Spatial/..."

Until Spec 2 lands, SPEAR expects this absolute path. Symlink:
```bash
sudo mkdir -p /data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library
sudo ln -s $(pwd)/assets/mesh_library/quaternius_animalpack /data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_animalpack
sudo ln -s $(pwd)/assets/mesh_library/quaternius_farm       /data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_farm
```
```

- [ ] **Step 8.4: Commit**

Run:
```bash
cd /data/jzy/code/AVEngine
git add docs/
git commit -m "feat: docs/ — pipeline zh+en + assets + superpowers specs & plans + quickstart + troubleshooting

Copied from SPEAR/docs/ (originals preserved). AVEngine is now the canonical
home for pipeline documentation going forward.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" 2>&1 | tail -3
```

---

## Task 9: 本机 setup 幂等验证

**Files:** 无新文件。运行 setup.sh 并验证产物。

**Interfaces:**
- Consumes: Task 4 setup.sh + Task 3 manifest
- Produces: `external/SPEAR`, `external/Hunyuan3D-2.1` 两个 symlink + `.setup_state.json`

- [ ] **Step 9.1: 首跑（真跑，非 dry）**

Run:
```bash
cd /data/jzy/code/AVEngine
bash scripts/setup.sh 2>&1 | tail -30
```

Expected: 输出含
```
==== resolving SPEAR ====
  action:     symlinked ...external/SPEAR → /data/jzy/code/SPEAR (local origin=...)
==== resolving Hunyuan3D-2.1 ====
  action:     symlinked ...external/Hunyuan3D-2.1 → /data/jzy/code/Hunyuan3D-2.1 ...
wrote /data/jzy/code/AVEngine/.setup_state.json
...
Next steps: ...
```

- [ ] **Step 9.2: 验证 symlink 正确**

Run:
```bash
readlink /data/jzy/code/AVEngine/external/SPEAR
readlink /data/jzy/code/AVEngine/external/Hunyuan3D-2.1
ls /data/jzy/code/AVEngine/external/SPEAR/tools/gpurir_scenes/furniture_map.py
cat /data/jzy/code/AVEngine/.setup_state.json
```

Expected:
- `readlink SPEAR` → `/data/jzy/code/SPEAR`
- `readlink Hunyuan3D-2.1` → `/data/jzy/code/Hunyuan3D-2.1`
- `furniture_map.py` 存在
- `.setup_state.json` 含 `"SPEAR": "symlink"` 和 `"Hunyuan3D-2.1": "symlink"`

- [ ] **Step 9.3: 二跑（幂等）**

Run:
```bash
cd /data/jzy/code/AVEngine
bash scripts/setup.sh 2>&1 | grep -E "status|action"
```

Expected: 每个 dep 都是
```
  status:     symlink → /data/jzy/code/...
```
无 `action:` 行（因为已 resolve）。**不应创建重复 symlink**。

- [ ] **Step 9.4: Commit .gitignore 若有更新（可能不需要）**

Run:
```bash
cd /data/jzy/code/AVEngine
git status --short
```

Expected: 无 tracked 修改。`.setup_state.json` 因 .gitignore 应被忽略。若显示为 untracked，说明 .gitignore 有 bug — 回 Task 2.3 修。

若一切干净：Task 9 无 commit（仅验证）。

---

## Task 10: README.md — 单页 quickstart

**Files:**
- Create: `/data/jzy/code/AVEngine/README.md`

**Interfaces:**
- Consumes: 前 9 个 Task 全部产物（内容引用它们）
- Produces: 单页文档

- [ ] **Step 10.1: 写 README.md**

Create `/data/jzy/code/AVEngine/README.md`:

```markdown
# AVEngine — Audio-Visual Engine

Research infrastructure for **Attribute-Conditioned Spatial Audio-Visual
Reasoning (ASAR)**. Combines Unreal Engine 5 (via SPEAR RPC) for
photorealistic multi-view rendering with gpuRIR for 4-channel
first-order-ambisonic room-impulse-response simulation, yielding audio-video
scenes where the **spatial ground truth is exact** (mic position, source
positions, camera intrinsics all deterministic).

Currently supports 5 animated animal tags (dogs, cats, chipmunk) + 7 static
ungulate tags in two rooms (apartment_0000 real Kujiale scan; procedural
shoebox). Outputs 640×480 15 fps 5s MP4 with muxed stereo audio.

⚠ **Private research project (not open source yet).** Contact author before
redistribution.

## Directory layout

```
AVEngine/
├── README.md                # this file
├── manifest.yaml            # single source of truth for deps + data
├── scripts/setup.sh         # `bash scripts/setup.sh` populates external/
├── envs/*.yml               # 3 conda env recipes (create manually)
├── assets/mesh_library/     # Quaternius rigged animal GLBs (CC0)
├── docs/                    # pipeline docs, specs, plans, image assets
└── external/                # git-ignored; populated by setup.sh
    ├── SPEAR/               # pipeline main; fork of spear-sim/spear
    └── Hunyuan3D-2.1/       # 3D asset generator (Tencent, upstream)
```

## Setup (Linux, GPU)

**Prereqs**: bash 4+, python3 + pyyaml, git, conda (miniconda/anaconda),
NVIDIA GPU with driver 550+, UE 5.5 build tools if you'll re-cook the
SpearSim project.

**Step 1 — Clone + populate deps**

```bash
git clone <AVEngine repo url> /data/jzy/code/AVEngine
cd /data/jzy/code/AVEngine
bash scripts/setup.sh
```

`setup.sh` is idempotent. On the author's machine (with pre-existing
`/data/jzy/code/SPEAR` etc.) it creates symlinks; on your machine it clones
into `external/`.

**Step 2 — Create 3 conda envs**

```bash
for env in spear-env sao-env hunyuan3d-env; do
    conda env create -f envs/$env.yml
done
```

Notes:
- `spear-env` needs `spear_ext` (SPEAR's compiled C++ RPC extension). Build
  it separately per SPEAR's build docs (`external/SPEAR/cpp/`).
- `sao-env` needs CUDA + torch 2.7. Ensure driver ≥ 550.
- Env creation can take 30-60 min due to CUDA torch downloads.

**Step 3 — Provide external data**

`setup.sh` does NOT download data. Place these at the paths listed in
`manifest.yaml` `external_data`:

| Path | Size | Source |
|------|------|--------|
| `/data/datasets/omniaudio/train-data-az-360-large` | ~40 GB | AudioSet wavs (contact author) |
| `/data/datasets/omniaudio/stable-audio-open` | ~5 GB | https://huggingface.co/stabilityai/stable-audio-open-1.0 |
| `/data/jzy/code/Hunyuan3D-2.1/pretrained_models` | ~20 GB | https://huggingface.co/Tencent-Hunyuan/Hunyuan3D-2.1 |

**Step 4 — Symlink mesh_library to SPEAR's expected path** ⚠

Until Spec 2 (SPEAR path parameterization) lands, SPEAR hardcodes
`/data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_*`.
On collaborator machines, run (needs sudo):

```bash
sudo mkdir -p /data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library
sudo ln -s $(pwd)/assets/mesh_library/quaternius_animalpack /data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_animalpack
sudo ln -s $(pwd)/assets/mesh_library/quaternius_farm       /data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_farm
```

## First demo — two dogs in a room

```bash
conda activate spear-env
export DISPLAY=:99
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
python external/SPEAR/tools/gpurir_scenes/scene_two_dogs.py --skip-audio
```

Expected outputs (after ~5-10 min UE render):
- `external/SPEAR/tmp/gpurir_scenes_v1/two_dogs/shoebox/view0.mp4`
- `external/SPEAR/tmp/gpurir_scenes_v1/two_dogs/apartment/view0.mp4`

Each MP4 is 640×480, 5s, 75 frames, no audio. Add `--skip-audio` off (i.e.
omit the flag) to also generate GPURIR 4-channel audio and mux
`view*_with_audio.mp4` files.

## Directory cheat sheet — where does X live?

| Feature | Path |
|---------|------|
| Pipeline main entrypoints | `external/SPEAR/tools/gpurir_scenes/` |
| Species → rig map | `external/SPEAR/tools/species_rig_map.py` |
| Furniture collision map | `external/SPEAR/data/apartment_furniture_map.json` |
| Rigged 3D animal meshes | `assets/mesh_library/` |
| Chinese pipeline doc | `docs/pipeline_zh.md` |
| English pipeline doc | `docs/pipeline_en.md` |
| Design specs | `docs/superpowers/specs/` |
| Implementation plans | `docs/superpowers/plans/` |

## Troubleshooting

See `docs/troubleshooting.md`. Common:
- "conda must be spear-env" — do NOT use `thu`, RPC silently fails
- "DISPLAY=:99 required" — UE needs an X server, headless X counts
- "furniture_map.json missing" — SPEAR must be at commit ≥ bc8ce323

## Contact

Ziyang Ji — Eastforward on GitHub. Research collaboration welcome; please
ping before redistributing.

## License

See `LICENSE`. Currently proprietary; open-source release pending.
Third-party components (Quaternius rigs, SPEAR upstream, Hunyuan3D-2.1)
retain their own licenses; see `manifest.yaml` `upstream` fields.
```

- [ ] **Step 10.2: Commit**

Run:
```bash
cd /data/jzy/code/AVEngine
git add README.md
git commit -m "feat: README.md — single-page quickstart

Covers: what AVEngine is, directory layout, 4-step setup, first demo,
cheat sheet, troubleshooting pointers, contact.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" 2>&1 | tail -3
```

---

## Task 11: V1-V6 Acceptance

**Files:** 无。运行 spec §10 定义的 6 项验收。

**Interfaces:**
- Consumes: 所有 Task 1-10 产物
- Produces: 6 项 pass/fail 报告

- [ ] **Step 11.1: V1 — 目录完整性**

Run:
```bash
cd /data/jzy/code/AVEngine
if command -v tree >/dev/null; then
    tree -L 2 -I 'external|__pycache__'
else
    find . -maxdepth 2 -not -path '*/\.*' | sort
fi
```

Expected 结构含（不必字字对应）：
- `README.md`, `LICENSE`, `manifest.yaml`, `.gitignore`
- `scripts/{setup.sh, update.sh}`
- `envs/{spear-env.yml, sao-env.yml, hunyuan3d-env.yml}`
- `assets/mesh_library/{quaternius_animalpack, quaternius_farm, README.md}`
- `docs/{pipeline_zh.md, pipeline_en.md, quickstart.md, troubleshooting.md, assets, superpowers}`
- `external/` (若已 setup 过，含 `SPEAR` 和 `Hunyuan3D-2.1` 两个 symlink)

- [ ] **Step 11.2: V2 — 本机 setup 幂等 (再验一次)**

Run:
```bash
cd /data/jzy/code/AVEngine
bash scripts/setup.sh 2>&1 | grep -E "status|action|ERROR"
readlink external/SPEAR
readlink external/Hunyuan3D-2.1
```

Expected: `status: symlink → ...` × 2；无 `action:`；无 `ERROR`；两个 readlink 成功。

- [ ] **Step 11.3: V3 — 模拟合作者机器（--force-clone）**

Run:
```bash
cd /data/jzy/code/AVEngine
# 备份现有 symlink
mv external/SPEAR /tmp/AVEngine_SPEAR_bak_$$
bash scripts/setup.sh --force-clone SPEAR 2>&1 | tail -10
ls external/SPEAR/tools/gpurir_scenes/furniture_map.py
# 恢复
rm -rf external/SPEAR
mv /tmp/AVEngine_SPEAR_bak_$$ external/SPEAR
readlink external/SPEAR   # 应恢复到 /data/jzy/code/SPEAR
```

Expected:
- setup.sh 输出 `action: cloned + checked out <sha>`
- `furniture_map.py` 存在（说明 clone 拿到了含最新 commit 的代码）
- 恢复后 readlink 回归 symlink

**若 clone 失败** = SPEAR 49 commits 没 push 上去。回 Task 1 重跑 `git push`.

- [ ] **Step 11.4: V4 — mesh_library 大小 sanity**

Run:
```bash
du -sh /data/jzy/code/AVEngine/assets/mesh_library
find /data/jzy/code/AVEngine/assets/mesh_library -name '*.glb' | wc -l
ls /data/jzy/code/AVEngine/assets/mesh_library/quaternius_animalpack/Cat.glb
ls /data/jzy/code/AVEngine/assets/mesh_library/quaternius_animalpack/Dog.glb
```

Expected: size ~4.6M, count 12, both Cat.glb + Dog.glb exist.

- [ ] **Step 11.5: V5 — 三个 env yml 可 parse**

Run:
```bash
for env in spear-env sao-env hunyuan3d-env; do
    yml="/data/jzy/code/AVEngine/envs/$env.yml"
    echo "==== $env ===="
    python3 -c "
import yaml
try:
    d = yaml.safe_load(open('$yml'))
    print('name:', d.get('name'))
    print('deps:', len(d.get('dependencies', [])))
    # Check no editable installs remain
    pip_pkgs = []
    for dep in d.get('dependencies', []):
        if isinstance(dep, dict) and 'pip' in dep:
            pip_pkgs = dep['pip']
    editable = [p for p in pip_pkgs if isinstance(p, str) and (p.startswith('-e ') or 'file://' in p)]
    if editable:
        print('WARN editable:', editable)
    else:
        print('pip clean')
except Exception as e:
    print('FAIL:', e)
"
done
```

Expected: 三个都 `name:`, `deps:` count > 0, `pip clean`.

- [ ] **Step 11.6: V6 — 端到端 demo (人工看视频)**

Run:
```bash
cd /data/jzy/code/AVEngine
export DISPLAY=:99
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
/data/jzy/miniconda3/envs/spear-env/bin/python \
  external/SPEAR/tools/gpurir_scenes/scene_two_dogs.py --skip-audio 2>&1 | tail -5
```

Expected: `TWO_DOGS_DONE ...`；产物在 `external/SPEAR/tmp/gpurir_scenes_v1/two_dogs/{apartment,shoebox}/view*.mp4`.

**人工验收**：
```bash
ls -la external/SPEAR/tmp/gpurir_scenes_v1/two_dogs/shoebox/view0.mp4
ls -la external/SPEAR/tmp/gpurir_scenes_v1/two_dogs/apartment/view0.mp4
```

两个视频文件都存在且 > 50 KB → V6 pass。

**若失败**：回 Task 9 / 4 debug（symlink 问题）或 Task 1 debug（SPEAR 内容问题）。

- [ ] **Step 11.7: Commit acceptance report**

Create `/data/jzy/code/AVEngine/docs/superpowers/plans/2026-07-06-avengine-monorepo-acceptance.md`:

```markdown
# AVEngine Monorepo Acceptance Report — <YYYY-MM-DD>

## V1 — Directory completeness
- [x] README.md, LICENSE, manifest.yaml, .gitignore
- [x] scripts/{setup.sh, update.sh}
- [x] envs/{spear-env,sao-env,hunyuan3d-env}.yml
- [x] assets/mesh_library with 12 GLB + README
- [x] docs/{pipeline_zh,pipeline_en,quickstart,troubleshooting}.md + assets + superpowers

## V2 — Local symlink idempotency
- [x] `bash scripts/setup.sh` twice — second run reports "status: symlink → ..." with no action

## V3 — Force-clone simulation
- [x] `bash scripts/setup.sh --force-clone SPEAR` produced a working clone

## V4 — mesh_library size sanity
- [x] size ~4.6 MB, count 12, Cat.glb + Dog.glb present

## V5 — env yml parseability
- [x] all three yml parse; no editable installs

## V6 — End-to-end demo
- [x] `scene_two_dogs.py --skip-audio` produced view0.mp4 for both apartment and shoebox

## Sign-off
User approved on <YYYY-MM-DD>.
```

Run:
```bash
cd /data/jzy/code/AVEngine
git add docs/superpowers/plans/2026-07-06-avengine-monorepo-acceptance.md
git commit -m "docs: V1-V6 acceptance report

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" 2>&1 | tail -3
```

---

## Global Definition of Done

以下**同时**成立 = plan 完成：

1. Task 1-11 每 Step 打勾（含所有 commit）
2. `git log --oneline` in AVEngine 应约 8-10 commits
3. `bash scripts/setup.sh` 两次执行都成功且幂等
4. V1-V6 全 pass（V6 是人工看视频）
5. `du -sh /data/jzy/code/AVEngine` < 20 MB（无 external，无 tmp/）

---

## Rollback

**per-Task**：`git revert <task-commit>`（每 Task 独立 commit）

**整体**：
```bash
cd /data/jzy/code/AVEngine
git log --oneline   # 记下 initial commit sha (Task 2)
# 若想完全回滚：rm -rf /data/jzy/code/AVEngine
# 因为 AVEngine 是**新** repo（未 push），rm -rf 无痕
```

⚠ **不可回滚**：Task 1.2 `git push eastforward main` 是网络操作；即使 rm AVEngine 也不会 unpush 那 49 commits。若要 undo push，`git push eastforward main --force` 回滚到上游 HEAD（需要网络）。

---

## Non-Goals (Recap)

不做（本 plan 明确不涉及）：
- SPEAR 硬编码路径 → env var 化（Spec 2）
- 装 conda env（用户手动）
- 下 external_data（用户手动）
- Spatial / JAEGER / 30+ 其他项目
- git LFS
- CI / GitHub Actions
- Mac/Windows
- 开源前的隐私裁剪

若某项做了或漏了，抛错，回 spec review。
