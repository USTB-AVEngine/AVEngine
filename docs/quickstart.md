# 同服务器协作与个人环境构建

本文面向在同一台服务器上参与 AVEngine 开发的协作者。每个人必须拥有
独立的个人目录、Miniconda、Conda 环境、源码副本、构建目录和输出目录。
公共目录只用于只读共享场景数据和系统资源。

开始前先阅读顶层 [`README.md`](../README.md)、
[`功能与使用指南`](usage_guideline.md) 和
[`仓库职责边界`](architecture/REPOSITORY_BOUNDARIES.md)。

## 强制隔离规则

每位协作者必须遵守：

- 只能在本人拥有并可写的目录中下载、安装和使用 Miniconda；
- 只能在本人的 Miniconda 下创建 Conda 环境；
- 禁止使用或修改项目所有者、其他成员、服务器公共目录中的 Conda、
  Python、`.venv`、编译结果和包缓存；
- 禁止使用 `sudo` 安装 Python 包或修改服务器系统环境；
- 修改任何仓库前，必须先将该仓库 Fork 到自己的 GitHub 账号；
- 必须克隆自己的 Fork，并为每项任务创建一个全新的独立分支；
- 禁止直接在 `main`、固定运行时分支或其他人的分支上开发和提交；
- 禁止直接向 `Eastforward` 上游仓库推送代码；
- 必须使用自己的 `tmp`、RIR 缓存、媒体、证据和构建目录；
- MP3D、ReplicaCAD、HRTF 等大文件可以只读共享，但不得覆盖；
- 系统编译器、显卡驱动或 EGL 库缺失时，应报告给服务器管理员，不得
  自行修改其他人的环境来绕过问题。

同服务器不等于共用工作区。只要 Git 权限、个人环境和输出隔离都满足，
协作者就可以正常拉取代码、运行测试、推送个人分支并参与代码评审。

## 修改代码前的强制 Git 规则

以下步骤必须发生在第一次代码修改之前：

1. 在 GitHub 上将目标 `Eastforward` 仓库 Fork 到自己的账号；
2. 克隆自己的 Fork，使 `origin` 指向个人 Fork；
3. 将 `Eastforward` 原仓库添加为只读同步用的 `upstream`；
4. 获取最新 `upstream`，从本文规定的基线创建一个新的任务分支；
5. 确认当前分支不是 `main` 或固定运行时分支后，才能修改文件。

一项任务对应一个分支。分支合并或关闭后，下一项任务必须重新获取
`upstream`，再创建新分支，禁止复用旧任务分支。本地 `main` 仅用于
对照和同步，不用于开发、提交或直接推送。

## 规划个人目录

下面的变量只作为示例。协作者应确认该路径属于自己的账号，且其他成员
不会写入：

```bash
export AVENGINE_PERSONAL_ROOT="${HOME}/avengine-local"
export AVENGINE_MINICONDA_PREFIX="${AVENGINE_PERSONAL_ROOT}/miniconda3"
export AVENGINE_ENV_PREFIX="${AVENGINE_PERSONAL_ROOT}/conda-envs/avengine-runtime"
export AVENGINE_CODE_ROOT="${AVENGINE_PERSONAL_ROOT}/code"
export AVENGINE_OUTPUT_ROOT="${AVENGINE_PERSONAL_ROOT}/output"
export AVENGINE_CONDA_CACHE="${AVENGINE_PERSONAL_ROOT}/conda-pkgs"
export AVENGINE_PIP_CACHE="${AVENGINE_PERSONAL_ROOT}/pip-cache"
export AVENGINE_BUILD_TMP="${AVENGINE_PERSONAL_ROOT}/build-tmp"
export AVENGINE_MINICONDA_INSTALLER="${AVENGINE_PERSONAL_ROOT}/downloads/Miniconda3-latest-Linux-x86_64.sh"

mkdir -p \
  "${AVENGINE_PERSONAL_ROOT}/downloads" \
  "${AVENGINE_PERSONAL_ROOT}/conda-envs" \
  "${AVENGINE_CODE_ROOT}" \
  "${AVENGINE_OUTPUT_ROOT}" \
  "${AVENGINE_CONDA_CACHE}" \
  "${AVENGINE_PIP_CACHE}" \
  "${AVENGINE_BUILD_TMP}"

export CONDA_PKGS_DIRS="${AVENGINE_CONDA_CACHE}"
export PIP_CACHE_DIR="${AVENGINE_PIP_CACHE}"
export TMPDIR="${AVENGINE_BUILD_TMP}"
```

项目相关的安装、源码、构建、缓存和输出都必须位于
`AVENGINE_PERSONAL_ROOT` 下。

## 自行下载并安装 Miniconda

当前服务器是 Linux x86-64。安装步骤以
[Anaconda 官方 Miniconda Linux 安装说明](https://www.anaconda.com/docs/getting-started/miniconda/install/linux-install)
为准。

每位协作者必须自己下载安装器：

```bash
if [ -e "${AVENGINE_MINICONDA_INSTALLER}" ]; then
  echo "安装器已经存在，请先核对，不要盲目覆盖。" >&2
else
  curl --fail --location \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    --output "${AVENGINE_MINICONDA_INSTALLER}"
fi

sha256sum "${AVENGINE_MINICONDA_INSTALLER}"
```

执行安装前，将 SHA-256 与
[官方 Miniconda 安装器归档](https://repo.anaconda.com/miniconda/)
中对应文件的记录进行比较。校验一致后才安装：

```bash
if [ -e "${AVENGINE_MINICONDA_PREFIX}" ]; then
  echo "个人 Miniconda 目录已经存在，请停止并检查，禁止覆盖。" >&2
else
  bash "${AVENGINE_MINICONDA_INSTALLER}" \
    -b \
    -p "${AVENGINE_MINICONDA_PREFIX}"
fi

source "${AVENGINE_MINICONDA_PREFIX}/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS="${AVENGINE_CONDA_CACHE}"
export PIP_CACHE_DIR="${AVENGINE_PIP_CACHE}"
export TMPDIR="${AVENGINE_BUILD_TMP}"

"${AVENGINE_MINICONDA_PREFIX}/bin/conda" --version
```

不要运行其他用户目录下的 `conda`，也不要把个人环境安装进公共数据目录。

## 创建个人 Conda 环境

使用显式 `--prefix`，确保环境一定落在个人目录：

```bash
source "${AVENGINE_MINICONDA_PREFIX}/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS="${AVENGINE_CONDA_CACHE}"
export PIP_CACHE_DIR="${AVENGINE_PIP_CACHE}"
export TMPDIR="${AVENGINE_BUILD_TMP}"

"${AVENGINE_MINICONDA_PREFIX}/bin/conda" create \
  --yes \
  --prefix "${AVENGINE_ENV_PREFIX}" \
  --channel conda-forge \
  python=3.12 \
  cmake=3.27 \
  ninja \
  pip

conda activate "${AVENGINE_ENV_PREFIX}"

test "${CONDA_PREFIX}" = "${AVENGINE_ENV_PREFIX}"
"${AVENGINE_ENV_PREFIX}/bin/python" -c \
  'import sys; print(sys.executable); print(sys.version)'
```

后续安装命令都显式调用
`"${AVENGINE_ENV_PREFIX}/bin/python"`，防止终端激活失败后误写系统 Python
或其他人的环境。

## 需要克隆的仓库

| 仓库 | 职责 | 当前入口 |
| --- | --- | --- |
| `Eastforward/AVEngine` | 任务、时间线、资产/房间配置、音频组装、质量检查、命令行 | `main` |
| `Eastforward/habitat-sim-AVEngine` | Habitat 原生视觉、动作和 RLR 适配 | `feature/m6-release-state` |
| `Eastforward/spear` | 可选的 UE 控制、`SpearSim` 项目和 Apartment 来源 | `7fbf3632fdb63cc2eceea564811c9597cabfb199`，仅 `legacy_optional` |

RLR Audio Propagation 已经是 Habitat 派生仓库的递归子模块，不要单独
克隆。SPEAR、UE、Blender、生成模型仓库和模型权重都不是默认构建依赖；
只有正式 Apartment 溯源或 UE 重导出任务才需要单独准备 SPEAR。

当前固定版本为：

```text
habitat-sim-AVEngine:
  e9c81c10834f7e89f33f4e0602c75535a84e054b
rlr-audio-propagation:
  4fd446b4abb5c71fb7a232a083bbddd65f25fc6f
spear (legacy_optional):
  7fbf3632fdb63cc2eceea564811c9597cabfb199
```

固定版本的机器可读依据是
[`manifest.yaml`](../manifest.yaml)。更新原生运行时时，必须同时更新
版本清单、相关测试和本文档。

### 48g-jump 的可选 Apartment 公共输入

同一服务器上的协作者应按只读方式使用以下共享输入：

```bash
export AVENGINE_UNREAL_ENGINE_ROOT=/data/UE_5.5
export AVENGINE_SPEAR_ROOT=/data/datasets/avengine_workspaces/shared/SPEAR-7fbf3632
export AVENGINE_LEGACY_APARTMENT_EXPORT_ROOT=/data/datasets/avengine_workspaces/AVEngine-habitat-native/tmp/m1/legacy_apartment_export
export AVENGINE_LEGACY_APARTMENT_PACKAGE_ROOT=/data/datasets/avengine_workspaces/shared/legacy_apartment_0000_v2

git config --global --add safe.directory \
  /data/datasets/avengine_workspaces/shared/SPEAR-7fbf3632
```

`/data/UE_5.5` 只是 UE 引擎根目录，绝不是 SPEAR root。公共 SPEAR
checkout 是固定 commit 的 sparse 溯源输入；共享目录权限可能允许组内
写入，但消费者仍必须把它当作只读输入。它可以用于打包、正式采集和验证，
但不能在其中启动 UE、生成 `Saved/Intermediate` 或修改项目。
`safe.directory` 是每个协作者自己的 Git 信任决定，不得改成系统级配置。

需要 UE 重导出时，必须在个人目录中创建完整 SPEAR clone：

```bash
export AVENGINE_SPEAR_ROOT="${AVENGINE_CODE_ROOT}/spear"
git clone git@github.com:Eastforward/spear.git "${AVENGINE_SPEAR_ROOT}"
git -C "${AVENGINE_SPEAR_ROOT}" checkout --detach \
  7fbf3632fdb63cc2eceea564811c9597cabfb199
```

若任务只是用既有 GLB 做 render-only 预览，则不需要 UE 或 SPEAR。
AVEngine 的正式 M1 package/capture/evidence 路线仍会检查 clean pinned
SPEAR、仓库相对路径和内容 hash。完整命令见
[`M1_EXECUTION.md`](roadmap/M1_EXECUTION.md)。

## 克隆仓库

先确认本人拥有两个 `Eastforward` 仓库的读取权限、自己的 GitHub 账号
以及可用的 SSH 密钥。不需要、也不应依赖 `Eastforward` 上游仓库的
直接写权限。

在 GitHub 页面先完成以下两个 Fork：

```text
Eastforward/AVEngine
  → <自己的 GitHub 账号>/AVEngine

Eastforward/habitat-sim-AVEngine
  → <自己的 GitHub 账号>/habitat-sim-AVEngine
```

然后克隆自己的 Fork，并添加上游远端：

```bash
export AVENGINE_GITHUB_USER="your-github-user"

cd "${AVENGINE_CODE_ROOT}"

git clone \
  "git@github.com:${AVENGINE_GITHUB_USER}/AVEngine.git"
git -C AVEngine remote add upstream \
  git@github.com:Eastforward/AVEngine.git
git -C AVEngine fetch upstream --prune

git clone \
  --recurse-submodules \
  "git@github.com:${AVENGINE_GITHUB_USER}/habitat-sim-AVEngine.git"
git -C habitat-sim-AVEngine remote add upstream \
  git@github.com:Eastforward/habitat-sim-AVEngine.git
git -C habitat-sim-AVEngine fetch upstream --prune

git -C habitat-sim-AVEngine checkout --detach \
  e9c81c10834f7e89f33f4e0602c75535a84e054b
git -C habitat-sim-AVEngine submodule sync --recursive
git -C habitat-sim-AVEngine submodule update \
  --init --recursive --jobs 8
```

核对远端职责：

```bash
git -C AVEngine remote -v
git -C habitat-sim-AVEngine remote -v
```

两个仓库都必须满足：

```text
origin    自己的 GitHub Fork，用于推送个人任务分支
upstream  Eastforward 原仓库，只用于获取基线和提交 Pull Request
```

核对原生运行时和 RLR：

```bash
test "$(git -C "${AVENGINE_CODE_ROOT}/habitat-sim-AVEngine" rev-parse HEAD)" = \
  "e9c81c10834f7e89f33f4e0602c75535a84e054b"

test "$(git -C "${AVENGINE_CODE_ROOT}/habitat-sim-AVEngine/src/deps/rlr-audio-propagation" \
  rev-parse HEAD)" = \
  "4fd446b4abb5c71fb7a232a083bbddd65f25fc6f"
```

`habitat-sim-AVEngine` 的远端默认 `main` 目前仍是上游 Habitat 基线，
不能替代上述固定版本。全新环境暂时不要使用
`./scripts/setup.sh --clone-runtime`：当前脚本不会自动从原生运行时的
默认 `main` 切换到固定功能分支，也不会完成全部递归子模块初始化。

## 仅安装 AVEngine 上层

只修改数据结构约束、任务、时间线、注册表或质量检查逻辑时，可以先只
安装 AVEngine：

```bash
"${AVENGINE_ENV_PREFIX}/bin/python" -m pip install --upgrade pip
"${AVENGINE_ENV_PREFIX}/bin/python" -m pip install \
  -e "${AVENGINE_CODE_ROOT}/AVEngine[test]"

cd "${AVENGINE_CODE_ROOT}/AVEngine"
"${AVENGINE_ENV_PREFIX}/bin/python" scripts/load_paths.py \
  --validate \
  --layer fast_unit
"${AVENGINE_ENV_PREFIX}/bin/python" scripts/validate_schemas.py
"${AVENGINE_ENV_PREFIX}/bin/python" -m pytest -q \
  tests/unit \
  -m 'not integration and not canary'
"${AVENGINE_ENV_PREFIX}/bin/avengine" --help
```

这一层不需要 Habitat、RLR、场景数据、HRTF、UE 或模型权重。协作者
交接路线不再创建第二套共享或隐式 `.venv`；统一使用上面明确位于个人
目录的 Conda 环境。

## 构建 Habitat/RLR 原生环境

需要运行 Habitat 视觉、动作或 RLR 声学时，在同一个个人环境中继续：

```bash
"${AVENGINE_ENV_PREFIX}/bin/python" -m pip install \
  -r "${AVENGINE_CODE_ROOT}/habitat-sim-AVEngine/requirements.txt"
"${AVENGINE_ENV_PREFIX}/bin/python" -m pip install \
  'scikit-build-core>=0.10' \
  'pybind11>=2.10'

cd "${AVENGINE_CODE_ROOT}"

CMAKE_BUILD_PARALLEL_LEVEL=4 \
HABITAT_BUILD_GUI_VIEWERS=OFF \
HABITAT_WITH_BULLET=ON \
HABITAT_WITH_AUDIO=ON \
HABITAT_WITH_CUDA=OFF \
"${AVENGINE_ENV_PREFIX}/bin/python" -m pip install \
  -e ./habitat-sim-AVEngine \
  --no-build-isolation
```

参考环境是 Python 3.12、CMake 3.27、Ninja 和 Linux x86-64。原生构建
约需 8 GB，建议个人目录至少预留 10 GB。该构建不需要 CUDA toolkit。
系统层需要可用的 C/C++ 编译器、EGL/Mesa 开发库和 `pkg-config`；若
缺失，应联系管理员。

设置当前终端使用的仓库路径：

```bash
export AVENGINE_HABITAT_RUNTIME_ROOT="${AVENGINE_CODE_ROOT}/habitat-sim-AVEngine"
export AVENGINE_SOUNDSPACES_ROOT="${AVENGINE_HABITAT_RUNTIME_ROOT}/src/deps/rlr-audio-propagation/RLRAudioPropagationPkg"
```

当前 SoundSpaces MP3D 材质 JSON 已包含在固定 RLR 子模块中，因此默认
AVEngine 路线不需要额外克隆 SoundSpaces。只有运行 SoundSpaces 自身的
官方基线和示例时，才需要其独立仓库。

## 验证原生环境

固定原生运行时存在已知导入顺序问题，必须先导入 `quaternion`：

```bash
"${AVENGINE_ENV_PREFIX}/bin/python" - <<'PY'
import quaternion
import habitat_sim

print("habitat_sim:", habitat_sim.__file__)
print("audio:", habitat_sim.audio_enabled)
print("bullet:", habitat_sim.built_with_bullet)
print("cuda:", habitat_sim.cuda_enabled)

assert habitat_sim.audio_enabled
assert habitat_sim.built_with_bullet
PY

cd "${AVENGINE_CODE_ROOT}/AVEngine"
"${AVENGINE_ENV_PREFIX}/bin/python" scripts/load_paths.py \
  --validate \
  --layer native_habitat \
  --layer rlr_audio
"${AVENGINE_ENV_PREFIX}/bin/python" scripts/validate_schemas.py
"${AVENGINE_ENV_PREFIX}/bin/python" -m pytest -q \
  tests/unit \
  -m 'not integration and not canary'
```

构建成功或快速单元测试通过，不代表原生场景、RLR、媒体回读或发布冒烟
验证已经通过。应根据改动范围运行对应测试层，并如实记录未执行的测试。

## 个人输出与只读共享数据

以下内容不随 Git 克隆提供：

- MP3D、ReplicaCAD、导航网格和语义场景数据；
- 编译后的 Habitat/RLR 动态库；
- `tmp/` 下的 RIR、视频、证据和编译后的声学包；
- 干声、生成动物、模型权重和 UE 资产。

所有可写输出必须位于个人目录：

```bash
mkdir -p "${AVENGINE_OUTPUT_ROOT}/AVEngine/tmp"

cd "${AVENGINE_CODE_ROOT}/AVEngine"
if [ -e tmp ] || [ -L tmp ]; then
  echo "tmp 已存在，请先检查归属和指向，禁止直接覆盖。" >&2
else
  ln -s "${AVENGINE_OUTPUT_ROOT}/AVEngine/tmp" tmp
fi
```

多人不能共用同一个 RIR 缓存、媒体输出目录或禁止覆盖的证据目录。公共
场景数据只能通过环境变量或场景配置只读引用。

正式双耳路线还需要系统提供的 HRTF；参考安装位置是：

```text
/usr/share/libmysofa/MIT_KEMAR_normal_pinna.sofa
```

构建和快速单元测试不需要场景数据或 HRTF。运行具体房间时，再按对应
操作手册设置 `AVENGINE_REPLICACAD_ROOT`、MP3D 场景路径和声学包根目录。

## 代码协作流程

开始修改前，任务负责人必须给出以下五项：

1. 任务 ID；
2. 目标上游仓库；
3. 基线分支和 base SHA；
4. PR 目标分支；
5. 改动范围与最低验收。

协作者不能自行猜测基线或 PR 目标。没有联合集成目标的普通任务默认从
最新 `upstream/main` 建分支；联合开发任务必须以负责人明确给出的
integration 分支为基线和 PR 目标，不能自行改投 `main`。本轮联合开发
使用 `upstream/integration/lifelike-engine-v1`。

先设置只对应当前任务的分支名和负责人提供的基线：

```bash
export AVENGINE_TASK_ID="lifelike-t1"
export AVENGINE_GIT_BASE_REF="upstream/integration/lifelike-engine-v1"
export AVENGINE_GIT_BASE_SHA="<负责人提供的精确 SHA>"
export AVENGINE_GIT_BRANCH="${AVENGINE_GITHUB_USER}/lifelike-t1-scenario-composer"

cd "${AVENGINE_CODE_ROOT}/AVEngine"
git fetch upstream --prune
test "$(git rev-parse "${AVENGINE_GIT_BASE_REF}")" = \
  "${AVENGINE_GIT_BASE_SHA}"
git switch -c "${AVENGINE_GIT_BRANCH}" "${AVENGINE_GIT_BASE_SHA}"

test "$(git branch --show-current)" = "${AVENGINE_GIT_BRANCH}"
test "$(git branch --show-current)" != "main"
git merge-base --is-ancestor \
  "${AVENGINE_GIT_BASE_SHA}" \
  "${AVENGINE_GIT_BRANCH}"
```

只有以上检查通过后才能开始修改代码。一个任务对应一个分支和一个 PR；
不得夹带其他任务、全仓库格式化、顺手重构或未分配的研究代码。优先扩展
现有 schema、registry、runner、validator 和 CLI。新增平行入口时，
PR 必须说明现有入口为何不能扩展。

禁止提交以下内容：

- `tmp/`、RIR、视频、音频和数据集；
- 模型权重、Conda 环境、编译目录和包缓存；
- 第三方仓库或外部数据的副本；
- 私有绝对路径、密钥和未经授权的数据。

大型 canary 产物保留在个人输出目录，PR 只报告仓库相对路径、状态和
必要的轻量证据。完成后检查改动、运行实际相关测试、明确选择要提交的
文件，并把分支推送到自己的 Fork：

```bash
git status --short
git diff --check
"${AVENGINE_ENV_PREFIX}/bin/python" -m pytest -q \
  tests/unit \
  -m 'not integration and not canary'

git add path/to/changed-file
git commit -m "说明本次改动"
git push -u origin "${AVENGINE_GIT_BRANCH}"
```

禁止使用 `git add -A` 代替范围检查。提交前应查看 staged diff，并确认
没有其他人的文件或生成产物。

然后按任务负责人给出的目标创建 Pull Request。本轮联合开发使用：

```text
base: Eastforward/AVEngine:integration/lifelike-engine-v1
head: <自己的 GitHub 账号>:<个人任务分支>
```

PR 必须按 [Pull Request 模板](../.github/PULL_REQUEST_TEMPLATE.md)
填写：

- 任务 ID、base SHA、head SHA 和目标分支；
- 改动范围、复用的现有模块和明确未修改的相邻模块；
- 实际运行的命令以及 `pass`、`skip`、`not_run`；
- canary、视频和证据的仓库相对路径；
- claim boundary、已知未完成项和工作树状态。

只给视频不能替代机器检查，只给单元测试也不能宣称 native Habitat、
RLR、Blender 或媒体层已通过。无法运行的层应如实写 `not_run`，不得
伪造或降低验证条件。

协作者不得自行合并 PR。审核通过后由 integration 维护者 Squash Merge；
冲突应由原作者在自己的 Fork 分支解决，审核者不得进入作者工作树直接
修改。只有项目负责人完成组合回归、真实 canary 和最终视频审核后，
才可以决定是否创建 integration 到 `main` 的最终 PR。

禁止直接向 `upstream` 推送，禁止覆盖别人的分支、工作树、环境或生成
目录。PR 合并或关闭后，下一项任务必须重新获取负责人指定的最新基线，
再创建另一个新分支，禁止复用旧任务分支。

只有必须发生在 Habitat C++/Python 运行时内部、且无法通过稳定接口实现
的变化，才修改原生运行时仓库。其功能分支应从固定提交创建：

```bash
export AVENGINE_RUNTIME_GIT_BRANCH="${AVENGINE_GIT_BRANCH}-runtime"

cd "${AVENGINE_CODE_ROOT}/habitat-sim-AVEngine"
git fetch upstream --prune
git switch -c "${AVENGINE_RUNTIME_GIT_BRANCH}" \
  e9c81c10834f7e89f33f4e0602c75535a84e054b

test "$(git branch --show-current)" = \
  "${AVENGINE_RUNTIME_GIT_BRANCH}"
test "$(git branch --show-current)" != \
  "feature/m6-release-state"

git push -u origin "${AVENGINE_RUNTIME_GIT_BRANCH}"
```

原生运行时分支同样只能推送到个人 Fork 的 `origin`，再向
`Eastforward/habitat-sim-AVEngine:feature/m6-release-state` 提交 Pull
Request。不能从该仓库的 `main` 创建 AVEngine 运行时改动，因为当前
权威基线是上面的固定提交。

跨仓库改动按以下顺序交付：

1. 原生运行时改动单独评审并形成新提交；
2. AVEngine 更新 `manifest.yaml` 的运行时固定版本和相应测试；
3. AVEngine 功能分支通过测试后再合并到 `main`。

提交协作结果时至少报告：两个仓库的提交、工作树是否干净、实际运行的
测试、未运行的原生/数据测试层，以及生成产物的仓库相对路径。

## 使用与许可边界

AVEngine 当前保留所有权利，协作者必须获得项目所有者授权。RLR Audio
Propagation 当前采用 CC BY-NC 4.0；默认声学路线只能用于符合其许可
条件的研究用途。
