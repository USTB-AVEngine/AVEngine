# 危险/不可逆操作记录 — 2026-07-07 session

> 本 session 建立了 AVEngine monorepo 并做了大量文件系统 + git 操作。以下是**破坏性/不可逆**操作清单，方便你审计和万一发现问题时回滚。

---

## 1. Git remote 变更

### 1.1 SPEAR 新增 `eastforward` remote + force-push
```bash
cd external/SPEAR
git remote add eastforward git@github.com:Eastforward/spear.git
git push -u eastforward main --force-with-lease
```
- **影响**：`github.com/Eastforward/spear` 的 `main` 从 upstream `c2cea1be` 强制覆盖成本地 `bc8ce323`（49 未 push commits）
- **不可逆**：GitHub 上 fork 的 main branch 内容已被覆盖
- **回滚**：`git push eastforward main --force`（reset 回 `c2cea1be`，从上游 sync 一次），但这样会丢我们所有 pipeline 代码
- **随后又 push 了一次**：`a5168b8c` (hardcoded-path 参数化 refactor)

### 1.2 AVEngine 新 repo 首次 push
```bash
cd /data/jzy/code/AVEngine
git remote add origin git@github.com:Eastforward/AVEngine.git
git push -u origin main
```
- **影响**：`github.com/Eastforward/AVEngine` 从空仓变成 12 commits + 全部 pipeline 代码/资产/文档
- **可逆**：`git push origin main --delete` 或在 GitHub 上删仓再建

---

## 2. 文件系统 mv（物理迁移）

### 2.1 SPEAR + Hunyuan3D-2.1 从 `/data/jzy/code/` 迁到 `/data/jzy/code/AVEngine/external/`
```bash
mv /data/jzy/code/SPEAR /data/jzy/code/AVEngine/external/SPEAR
mv /data/jzy/code/Hunyuan3D-2.1 /data/jzy/code/AVEngine/external/Hunyuan3D-2.1
```
- **影响**：以下路径**不再存在**：
  - `/data/jzy/code/SPEAR` → 变为 `/data/jzy/code/AVEngine/external/SPEAR`
  - `/data/jzy/code/Hunyuan3D-2.1` → 变为 `/data/jzy/code/AVEngine/external/Hunyuan3D-2.1`
- **副作用**：
  - 一个用户 shell (PID 3549938) 之前 cwd 在 `Hunyuan3D-2.1`，现在指向不存在路径。**不 fatal**（Linux 支持），但那个 shell 用 `pwd` 或 `ls .` 会 confusion。请在那个 shell 里 `cd /data/jzy/code/AVEngine/external/Hunyuan3D-2.1` 修正
- **可逆**：`mv /data/jzy/code/AVEngine/external/SPEAR /data/jzy/code/SPEAR` 等（但会破坏 AVEngine 结构）
- **验证 mv 干净**：SPEAR 49 commits + 85 dirty 文件 + 双 remote（origin, eastforward）**全部保留**在新位置

---

## 3. pip uninstall + reinstall

### 3.1 spear-env 的 `spear-sim` 和 `spear-ext` 卸载后重装
```bash
/data/jzy/miniconda3/envs/spear-env/bin/pip uninstall -y spear-sim spear-ext
# then reinstall from new location:
/data/jzy/miniconda3/envs/spear-env/bin/pip install -e /data/jzy/code/AVEngine/external/SPEAR/python
# spear-ext 需要 UE clang，用了两条命令：
# (1) SPEAR 官方 installer:
/data/jzy/miniconda3/envs/spear-env/bin/python \
  /data/jzy/code/AVEngine/external/SPEAR/tools/install_python_extension.py \
  --unreal-engine-dir /data/UE_5.5 --conda-env spear-env
# → 由于 shell PATH 优先 thu env，误装到 /data/jzy/miniconda3/envs/thu/
# (2) 手工用 spear-env pip 装到正确 env：
/data/jzy/miniconda3/envs/thu/bin/pip uninstall -y spear-ext  # 从错的 env 卸
UE=/data/UE_5.5
CLANG=$UE/Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/v23_clang-18.1.0-rockylinux8/x86_64-unknown-linux-gnu/bin/clang++
LIBCXX=$UE/Engine/Source/ThirdParty/Unix/LibCxx
CXX_FLAGS="-std=c++20 -O3 -D_LIBCPP_ENABLE_EXPERIMENTAL -nostdinc++ -I$LIBCXX/include/c++/v1 -Wno-reserved-macro-identifier -stdlib=libc++ -L$LIBCXX/lib/Unix/x86_64-unknown-linux-gnu -lc++"
/data/jzy/miniconda3/envs/spear-env/bin/pip install -e /data/jzy/code/AVEngine/external/SPEAR/python_ext \
    -C cmake.define.CMAKE_CXX_COMPILER="$CLANG" \
    -C cmake.define.CMAKE_CXX_FLAGS="$CXX_FLAGS"
```
- **影响**：
  - `spear-env` 里 `spear-sim` 和 `spear-ext` 从旧路径重装到 `AVEngine/external/SPEAR/{python, python_ext}`
  - `thu` env 意外多装过一次 `spear-ext-cp312` 然后卸掉。thu env 应该没被永久污染
- **验证**：`/data/jzy/miniconda3/envs/spear-env/bin/python -c "import spear; import spear_ext"` 均 OK
- **可逆**：`pip uninstall -y spear-sim spear-ext` + 从原位置（若还在）重装

---

## 4. 修改 SPEAR 内文件（多处硬编码路径 → 参数化）

在 `/data/jzy/code/AVEngine/external/SPEAR/` 内以下文件被 patch（**已 commit 到你 fork**，SHA `a5168b8c`）：

- `tools/species_rig_map.py` — `QUATERNIUS_DIR` / `QUATERNIUS_FARM` / `HY3D_*` 全部改为 `__file__` 相对路径 or env var
- `tools/gpurir_scenes/dump_apartment_furniture.py` — `REPO` 改 `__file__` 相对
- `tools/gpurir_scenes/run_render_pass.py` — 同上
- `tools/gpurir_scenes/furniture_map.py` — `DEFAULT_JSON_PATH` 改相对
- `tools/gpurir_scenes/audio_registry.py` — `AUDIO_CORPUS` 读 env `AVENGINE_AUDIO_CORPUS`；`DEFAULT_SAO_CACHE` 改相对
- `tools/gpurir_scenes/scene_two_dogs.py` — `--out-root` 默认改相对
- `tools/gpurir_scenes/run_scene.py` — 同上
- `tools/gpurir_scenes/run_all_scenes.py` — 同上
- `tools/gpurir_scenes/run_audio_pass.py` — 同上
- `examples/render_in_apartment.py` — `EXECUTABLE`（SpearSim.sh 路径）改相对
- `examples/render_in_gpurir_room.py` — `SPEARSIM_EXECUTABLE` + `DEFAULT_TMP_ROOT` + `DEFAULT_META_DIR` 改相对

**影响**：改动全部**向后兼容**（相对路径 == 旧的绝对路径 in-place）；env var overrides 也是可选。
**已 commit + push 到 Eastforward/spear main**。

---

## 5. 中间产物（可以随时删）

以下路径是本 session 产生的**验证/临时**文件，可随时 `rm -rf`：

- `/data/jzy/code/AVEngine/external/SPEAR/tmp/gpurir_scenes_v1/two_dogs/` — 最新 e2e demo 产物（8 mp4）
- `/data/jzy/code/AVEngine/docs/assets/e2e/*` — 端到端文档 hero 图/视频（这些**已 commit** 到 AVEngine repo）
- `/tmp/dump_env.sh` — env yml 导出脚本（一次性使用，可删）
- `/data/jzy/cosmos_outputs/dog_smoke/` — 之前的 Cosmos smoke（本 session 未动，仅提及）

---

## 6. 短暂存在过的 symlinks（已删）

在我 mv 后为让 demo 能跑，短暂创建了：
- `/data/jzy/code/SPEAR → /data/jzy/code/AVEngine/external/SPEAR`
- `/data/jzy/code/Hunyuan3D-2.1 → /data/jzy/code/AVEngine/external/Hunyuan3D-2.1`

**已在本 session 结束前 `rm`**（因为你明确说"不想在 AVEngine 外有依赖"）。现在 `/data/jzy/code/SPEAR` 和 `/data/jzy/code/Hunyuan3D-2.1` **不存在**。

---

## 7. 未做的可能你要知道的操作

以下 **未做**（等你决定）：

- ❌ 未删 `/data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_*`（原始 mesh 还在，AVEngine 里是拷贝副本）
- ❌ 未删你其它 `/data/jzy/code/` 下的项目（30+ 个语音分离/TTS 旧仓库都还在）
- ❌ 未装 pyroomacoustics / Steam Audio（音频升级还在调研阶段）
- ❌ 未清理 SPEAR 里那 85 个 untracked/modified 文件（HANDOFF*.md, examples/my_*.py 等）—— 这些在 fork push 时**没被 push**（git 只 push commit），本机仍留着

---

## 一句话总结

**破坏性操作只有 2 类**：
1. 物理 mv 两个 repo 到 AVEngine/external/（可逆但麻烦）
2. force-push Eastforward/spear main 覆盖 fork 默认分支（可逆但要重 sync upstream）

其它都是可以 `git reset` / `pip reinstall` / `rm` 恢复的常规操作。
