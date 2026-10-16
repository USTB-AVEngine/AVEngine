# AVEngine 顶层 monorepo 设计

> 日期：2026-07-06
>
> Scope：**只**搭建 `/data/jzy/code/AVEngine/` 顶层目录 —— 让合作者能一键 setup 拿到跑 audio-visual pipeline 需要的所有代码 + 资产。
>
> **明确不做**（本 spec 不涉及，后续独立 spec 处理）：
> - SPEAR / mesh 引用侧的硬编码路径参数化（"$AVENGINE_ROOT" 化）—— 独立 Spec 2
> - 自动装 conda env
> - 自动下载数据集
> - Spatial 主 repo / JAEGER / Cosmos / 30+ 语音项目
> - Git LFS
> - CI

---

## 1. 问题陈述

当前 audio-visual pipeline 分散在 `/data/jzy/code/` 下多个目录：`SPEAR/`（主 pipeline，fork of `spear-sim/spear`，49 未 push commits），`Spatial/v77_4ch_S2L/assets/mesh_library/`（Quaternius rigs，100 MB GLB），`Hunyuan3D-2.1/`（第三方 stable）。合作者要用这套 pipeline 时：

1. 拿不到你 SPEAR 49 未 push commits（含家具 dump、动画 fix）
2. 不知道 Quaternius rigs 从哪来（`Spatial/` 32 GB 无法 push）
3. 不知道该装哪 3 个 conda env
4. 不知道 pipeline 入口在哪、demo 怎么跑

**AVEngine 是解决"合作者体验"的顶层 wrapper**：一个新 repo，从中 `git clone && bash scripts/setup.sh && conda env create ...` 就能拿到全部代码 + rigs，能跑 demo。

---

## 2. 高层方案

单个新 git repo，位于 `/data/jzy/code/AVEngine/`，含：

- `manifest.yaml` — 声明式的依赖单一真相源（哪个 external repo，pin 哪个 commit）
- `scripts/setup.sh` — 读 manifest，把 external repo clone 到 `external/`（本机则 symlink 到 `/data/jzy/code/{SPEAR, Hunyuan3D-2.1}`）
- `assets/mesh_library/` — Quaternius GLB 直接 add 进 git（100 MB，不用 LFS）
- `envs/*.yml` — 3 个 conda env（`--from-history` + `pip freeze`）
- `docs/` — 中文/英文文档 + spec + plan 归档
- `README.md` — 单页 quickstart

setup.sh 只拉仓，不装 env、不下数据。conda env 和外部数据在 README 里教。

**前置动作**（本 spec 执行前必须完成）：
- 把 `/data/jzy/code/SPEAR` 里的 49 commits `git push` 到 `github.com/Eastforward/spear`

---

## 3. 目录布局

```
AVEngine/
├── README.md                        # 单页 quickstart (≈150 行)
├── LICENSE                          # 未开源前留空占位
├── manifest.yaml                    # 依赖单一真相源
├── .gitignore                       # 排除 external/, cache 等
├── scripts/
│   ├── setup.sh                     # 拉仓（symlink 本机 / clone 别人机器）
│   └── update.sh                    # 更新到 manifest 声明的新 commit
├── envs/
│   ├── spear-env.yml                # SPEAR + spear_ext (py3.11)
│   ├── sao-env.yml                  # torch 2.7 + stable-audio-tools + gpuRIR
│   └── hunyuan3d-env.yml            # Hunyuan3D 依赖
├── assets/
│   └── mesh_library/
│       ├── quaternius_animalpack/   # cp from Spatial (≈50 MB)
│       │   ├── Cat.glb
│       │   ├── Dog.glb
│       │   └── ...
│       └── quaternius_farm/         # cp from Spatial (≈50 MB)
│           ├── Horse.glb
│           └── ...
├── docs/
│   ├── quickstart.md                # setup 详细步骤
│   ├── pipeline_zh.md               # 中文 pipeline 文档（从 SPEAR 拷来）
│   ├── pipeline_en.md               # 英文版
│   ├── troubleshooting.md           # 常见坑
│   └── superpowers/
│       ├── specs/                   # 本 spec + 未来 spec
│       └── plans/                   # 本 plan + 未来 plan
├── external/                        # git-ignored，clone/symlink 目标
│   ├── SPEAR/                       # symlink 本机 → /data/jzy/code/SPEAR
│                                    # 或 clone from github.com/Eastforward/spear
│   └── Hunyuan3D-2.1/               # symlink 本机 → /data/jzy/code/Hunyuan3D-2.1
│                                    # 或 clone from github.com/Tencent-Hunyuan/Hunyuan3D-2.1
└── .setup_state.json                # git-ignored，记录 setup 是何时哪个用户跑的
```

### 说明

- **`external/` 是 git-ignore 的**（`.gitignore` 里 `external/`），因为它是每台机器上 setup.sh 现场决定的（symlink or clone）。
- **`assets/mesh_library/` 进 git**（`git add`）。100 MB 一次性 clone 成本可接受。
- **`docs/superpowers/specs+plans` 从 SPEAR 里搬进来**：因为设计文档从此归 AVEngine 管，SPEAR 只管代码。

---

## 4. `manifest.yaml` schema

```yaml
# AVEngine dependency manifest — single source of truth.
# scripts/setup.sh reads this. If you add a new external repo here,
# rerun `bash scripts/setup.sh` and it will pick it up.

version: 1

dependencies:
  SPEAR:
    url: https://github.com/Eastforward/spear.git
    commit: <SHA-to-fill-after-first-push>  # bc8ce323 or later
    path: external/SPEAR
    upstream: https://github.com/spear-sim/spear.git   # for provenance / future contribution
    local_hint: /data/jzy/code/SPEAR                    # AUTHOR'S LOCAL PATH. Setup.sh
                                                        # checks: if this absolute path
                                                        # exists AND its git origin matches
                                                        # `url` or `upstream`, symlink
                                                        # instead of cloning. On any other
                                                        # machine (path won't exist) → clone.
                                                        # Collaborators can leave this
                                                        # field untouched.

  Hunyuan3D-2.1:
    url: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
    commit: <SHA-to-pin>   # locked to a known-good version
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
    python: "3.11"
    critical_packages: [torch>=2.7, stable-audio-tools, gpuRIR==1.2.0, numpy<2]
  hunyuan3d-env:
    yml: envs/hunyuan3d-env.yml
    python: "3.10"
    critical_packages: [diffusers, hy3dgen]
```

---

## 5. `scripts/setup.sh` 行为规范

### 5.1 4 个使用场景（幂等）

| 场景 | 状态 | setup.sh 行为 |
|---|---|---|
| **A. 你本机首跑** | `external/SPEAR` 空 · `/data/jzy/code/SPEAR` 存在且 `git remote origin` 匹配 manifest.url 或 upstream | 创建 symlink `external/SPEAR → /data/jzy/code/SPEAR`；不 clone；不改任何 external 侧 git 状态 |
| **B. 合作者首跑** | `external/SPEAR` 空 · local_hint 路径不存在 | `git clone <url> external/SPEAR && cd external/SPEAR && git checkout <commit>` |
| **C. 重跑（幂等）** | `external/SPEAR` 已 symlink 或 clone 且 commit 匹配 manifest | 只报状态 "already at correct commit"，不动 |
| **D. 更新 commit** | 通过 `--update` flag | `git fetch && git checkout <new-commit>` (只对 clone 情况，symlink 情况警告 "external repo is a symlink, update it manually") |

### 5.2 CLI

```bash
bash scripts/setup.sh              # 幂等：拉齐所有依赖
bash scripts/setup.sh --update     # 拉齐后额外把 clone 类的 checkout 到 manifest 声明的 commit
bash scripts/setup.sh --force-clone <dep>  # 忽略 local_hint，强制 clone（e.g., 想在本机测别人体验）
bash scripts/setup.sh --dry-run    # 只 print 将做什么，不执行
```

### 5.3 每个依赖的决策流程（伪代码）

```
for dep in manifest.dependencies:
    target = manifest[dep].path         # e.g. external/SPEAR
    url    = manifest[dep].url
    commit = manifest[dep].commit
    hint   = manifest[dep].local_hint

    if target exists:
        if it's a symlink or a git repo with matching origin:
            report "OK: already at <target>"
            if --update and not a symlink:
                git fetch && git checkout $commit
            continue
        else:
            error "target exists but doesn't look right; refusing to overwrite"

    if hint exists and (hint's git origin matches url OR upstream OR any other manifest
                        dep's url — to allow pre-fork state where local SPEAR still
                        points at spear-sim/spear even though manifest already points
                        at Eastforward/spear):
        ln -s $hint $target
        report "symlinked $target → $hint (local; origin was <observed url>)"
        continue

    git clone $url $target
    cd $target && git checkout $commit
    report "cloned $target from $url @ $commit"
```

### 5.4 失败处理

- Clone 失败：`echo "FAIL: clone $url" && exit 1`。用户自己 `rm -rf $target` 重跑。
- Symlink 目标不存在：报错。
- Git origin 不匹配：**不覆盖**，报错让用户手动介入（因为可能是他们自己搞的 repo，不能自动删）。

### 5.5 写 `.setup_state.json`

setup 结束后写：
```json
{"last_run_utc": "2026-07-06T14:00:00Z", "user": "jzy", "host": "gpu01",
 "dependencies_resolved": {"SPEAR": "symlink", "Hunyuan3D-2.1": "symlink"}}
```

用于 debug"这台机器 setup 走的是 clone 还是 symlink"。

### 5.6 setup.sh **不做**：
- 不装 conda env
- 不下 external_data（omniaudio / hunyuan weights / stable audio）
- 不改 shell profile
- 不 `sudo`

只是 print 提示："Next steps: (1) `conda env create -f envs/spear-env.yml` ...  (2) put external_data at paths listed in manifest.external_data ...  (3) `python external/SPEAR/tools/gpurir_scenes/scene_two_dogs.py`"。

---

## 6. `envs/*.yml` 生成规则

用 **`conda env export --from-history`**（只导出用户明确装的包）+ **`pip freeze`**（补 pip-only 包）。

**具体命令**（每个 env 各跑一次，写进 spec 里以便未来 regenerate）：
```bash
conda activate spear-env
conda env export --from-history > envs/spear-env.yml
# 手动追加 pip section:
echo "  pip:" >> envs/spear-env.yml
pip freeze | grep -v "^-e " | grep -v "^#" | sed 's/^/    - /' >> envs/spear-env.yml
```

三个 env 同样处理，产出 3 个 yml。

**Yml 里手动确认的关键字段**：
- `name:` 用 canonical 名（spear-env / sao-env / hunyuan3d-env）
- `channels:` 列 conda-forge / pytorch / nvidia
- `dependencies:` conda 部分 + pip 部分

**注意**：pip 里的 `-e /local/path/to/xxx` 类 editable install 要**手动删掉或改写**（合作者机器没有该路径）。已知需处理：
- `spear-env` 里 `spear_ext` 是 SPEAR 项目 build 出来的编译扩展，不在 pip index。yml 里注释 "Install spear_ext by building SPEAR/cpp/. See SPEAR README."
- `sao-env` 里 gpuRIR 是 pip 装的（`gpuRIR==1.2.0`），OK

---

## 7. `mesh_library` 迁移

从 `/data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/{quaternius_animalpack,quaternius_farm}` 复制到 `AVEngine/assets/mesh_library/{quaternius_animalpack,quaternius_farm}`。

**Method**：`cp -r --preserve=all`（保留时间戳/权限）。**不改任何 filename**。

**License**：Quaternius 是 CC0，可以随便打包。在 `AVEngine/assets/mesh_library/README.md` 说明来源与 license。

**大小验证**：`du -sh assets/mesh_library` 应 < 200 MB。若超（比如意外把 Deer/Alpaca 等未用到的目录也带进来），只 cp `Cat.glb`, `Dog.glb`, `Horse.glb`, `Cow.glb`, ... 具体见 species_rig_map.py 引用的文件。

⚠ **依然存在的问题**：SPEAR 代码里 `species_rig_map.py` 硬编码路径 `QUATERNIUS_DIR = "/data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_animalpack"`。合作者机器上这个路径不存在！

**本 spec 的 workaround**：setup.sh **不 sudo**（§5.6），但会**打印一段 sudo 命令让用户自己执行**：
```
# On collaborator's machine, run manually (needs sudo because /data/jzy/... is not owned):
sudo mkdir -p /data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library
sudo ln -s $(pwd)/assets/mesh_library/quaternius_animalpack /data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_animalpack
sudo ln -s $(pwd)/assets/mesh_library/quaternius_farm /data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_farm
```
如果合作者所在服务器**没 sudo 或不能碰 `/data/jzy`**，那**只有 Spec 2**（硬编码路径 → env var 化）能真正解掉这问题。本 spec 只提供 print 指导，不代跑。

**真正解**：**Spec 2** 会把 SPEAR 硬编码换成 env var，届时这个 workaround 可以拆掉。

---

## 8. README 内容清单（150 行左右）

必须覆盖：

1. **What is AVEngine**（2 段）
2. **Directory tree**（树状图）
3. **Setup**（3 步：clone / bash setup.sh / conda env create × 3）
4. **External data requirements**（omniaudio + stable-audio-open + Hunyuan weights，附下载/挂载指令）
5. **Quickstart demo**：
   ```bash
   conda activate spear-env
   export DISPLAY=:99
   python external/SPEAR/tools/gpurir_scenes/scene_two_dogs.py --skip-audio
   # 输出：tmp/gpurir_scenes_v1/two_dogs/{apartment,shoebox}/view0.mp4
   ```
6. **Where to look at output**（说明 view*_with_audio.mp4 vs view*.mp4 的区别）
7. **Troubleshooting**：
   - "conda activate spear-env" 必须（不能用 thu 或别的 env）
   - `DISPLAY=:99` 必须（UE 需要 X server）
   - 若 setup.sh 报"origin mismatch"，检查 `git remote get-url origin`
   - 若渲染时报"apartment_furniture_map.json not found"，检查 external/SPEAR/data/ 是否存在
8. **Directory map cheat sheet**（"哪个功能在哪里"表）
9. **Contact / issues**（你的邮箱/GitHub）

---

## 9. 前置步骤（本 spec 执行前你必须完成）

**Blocker 1：SPEAR 49 commits push 到 fork**
```bash
cd /data/jzy/code/SPEAR
git remote add eastforward https://github.com/Eastforward/spear.git
git push eastforward main
# 记录 HEAD SHA，写进 manifest.yaml
git rev-parse HEAD
```

**Blocker 2：确认 Hunyuan3D-2.1 upstream commit**
```bash
cd /data/jzy/code/Hunyuan3D-2.1
git rev-parse HEAD
# 记录并写进 manifest.yaml
```

这两个 SHA 是 manifest.yaml 里的必填项，spec 无法回填 —— 由 plan 阶段的第一步收集。

---

## 10. 验收标准

Spec 完成 = 以下都通过：

**V1. 目录完整性**
```bash
tree AVEngine -L 2
```
应包含：README.md, LICENSE, manifest.yaml, .gitignore, scripts/{setup.sh, update.sh}, envs/{spear-env.yml, sao-env.yml, hunyuan3d-env.yml}, assets/mesh_library/{quaternius_animalpack, quaternius_farm}, docs/, external/（空但存在）

**V2. setup.sh 本机幂等**
```bash
cd AVEngine
bash scripts/setup.sh          # 首跑，创建 symlink
bash scripts/setup.sh          # 二次跑，报"already at correct commit"
readlink external/SPEAR        # → /data/jzy/code/SPEAR
readlink external/Hunyuan3D-2.1  # → /data/jzy/code/Hunyuan3D-2.1
```

**V3. setup.sh 在"模拟别人机器"下 clone**
```bash
mkdir /tmp/fake_home && cd /tmp/fake_home
git clone <AVEngine-repo-url> && cd AVEngine
bash scripts/setup.sh
# Expected: 因为 /data/jzy/code/SPEAR 存在但当前 cwd 无关，local_hint 检测通过（因为 hint 是绝对路径）
# —— 用 --force-clone 强制 clone 到 external/
bash scripts/setup.sh --force-clone SPEAR
ls external/SPEAR/tools/gpurir_scenes/  # 应含 furniture_map.py 等
```

**V4. mesh_library 大小 sanity**
```bash
du -sh AVEngine/assets/mesh_library  # < 200 MB
ls AVEngine/assets/mesh_library/quaternius_animalpack/Cat.glb  # exists
```

**V5. 三个 env yml 可加载**
```bash
for env in spear-env sao-env hunyuan3d-env; do
  # Dry-parse only, don't actually create
  conda env create --file AVEngine/envs/$env.yml --dry-run
done
```

**V6. Demo 可跑（human verification）**
```bash
cd AVEngine
conda activate spear-env
export DISPLAY=:99 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
python external/SPEAR/tools/gpurir_scenes/scene_two_dogs.py --skip-audio
# Expected: tmp/gpurir_scenes_v1/two_dogs/shoebox/view0.mp4 exists
```

（V6 依赖 workaround #7 里的 mesh_library symlink 在合作者机器上，本机应已 OK）

---

## 11. 风险与开放问题

| # | 风险 | 缓解 |
|---|---|---|
| R1 | SPEAR fork push 时遇 upstream 冲突 | 用 `git push --force-with-lease eastforward main`（fork 是你自己的，不用担心） |
| R2 | Hunyuan3D-2.1 upstream 后续大改，pin 的 commit 变过时 | 提供 `--update` flag；README 提醒定期 review |
| R3 | mesh_library 复制丢文件（GLB 或 texture） | V4 加 file count + hash check（可后加，本 spec 只 sanity size） |
| R4 | 合作者机器无 `/data/jzy/code/Spatial/...` 但 SPEAR 硬编码期望这个路径 | Workaround: setup.sh 帮建符号链接（§7）；真正解在 Spec 2 |
| R5 | env yml 里含 editable install 指向本机路径 | 手动检查并删除；README warn |
| R6 | 合作者用 Mac/Windows | 本 spec **只支持 Linux**（UE + Vulkan 需求）；README 明说 |

**开放问题（future spec）**：
- Q_future_1：Spec 2 硬编码路径 → `$AVENGINE_ROOT` refactor
- Q_future_2：加 CI（GitHub Actions），至少跑 `setup.sh --dry-run` 验证 manifest 语法
- Q_future_3：加自动数据集下载（当有 HF Datasets 化 后）
- Q_future_4：JAEGER / Spatial 大 repo 什么时候接入（可能永不）

---

## 12. 未来开源准备（Spec 完成后不做，只记录）

开源前需要额外做：
1. LICENSE 从占位改成真实（推 Apache 2.0 或 MIT）
2. `docs/` 里内部 HANDOFF/status 文档删掉，只保留公开文档
3. `manifest.external_data` 里的私有路径改为公开来源（HF Datasets URL）
4. README 加 citation, acknowledgment
5. 检查 `assets/mesh_library` 是否合规（Quaternius CC0 应该 OK）
6. 检查 `envs/*.yml` 是否含 token、path
7. 检查 `docs/superpowers/plans` 是否含敏感讨论（feedback、内部人名）

这些都不在本 spec scope，将来单独 spec。

---

## 13. Out-of-scope 明单

**明确不做**：
- ❌ SPEAR 硬编码路径 refactor（Spec 2）
- ❌ Spatial / JAEGER / Cosmos / Wan2.1 / 30+ 语音项目（不入 AVEngine）
- ❌ 自动装 conda env
- ❌ 自动下载 external_data
- ❌ Git LFS
- ❌ CI / GitHub Actions
- ❌ Mac/Windows 支持
- ❌ 开源前的隐私裁剪（未来独立 spec）
- ❌ 真正的多 apartment 变体支持（另一条线）

**明确要做**：
- ✅ AVEngine 目录 + git init
- ✅ manifest.yaml
- ✅ scripts/setup.sh（4 场景 + 幂等 + --update + --dry-run + --force-clone）
- ✅ 3 个 conda env yml（`--from-history` + `pip freeze`）
- ✅ assets/mesh_library cp
- ✅ docs/ 移动（superpowers spec/plan、pipeline 中英文文档）
- ✅ README.md 单页 quickstart
- ✅ V1-V6 验收（V6 人工看视频）
