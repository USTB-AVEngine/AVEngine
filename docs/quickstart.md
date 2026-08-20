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

## 当前只克隆 AVEngine

当前 bootstrap 的源码输入只有 AVEngine：

| 仓库 | 职责 | 当前入口 |
| --- | --- | --- |
| `USTB-AVEngine/AVEngine` | 任务、时间线、资产/房间配置、已精选纳入的适配源码、音频组装、质量检查和命令行 | `main` |

在 GitHub 页面只需 Fork AVEngine，并在自己的工作目录克隆该 Fork：

```bash
export AVENGINE_GITHUB_USER="your-github-user"
cd "${AVENGINE_CODE_ROOT}"
git clone "git@github.com:${AVENGINE_GITHUB_USER}/AVEngine.git"
git -C AVEngine remote add upstream \
  git@github.com:USTB-AVEngine/AVEngine.git
git -C AVEngine fetch upstream --prune
```

`habitat-sim-AVEngine`、SPEAR 和 RLR 的 checkout-era 工作树仅保留为迁移
历史或只读对照，不能作为新的 build、setup 或 run 输入。
scripts/setup.sh --clone-runtime 已撤销并以退出码 2 停止；它不会下载、clone、
fetch 或初始化任何外部仓库。

## Fast-unit bootstrap

只修改数据结构约束、任务、时间线、注册表或质量检查逻辑时，先选择项目
指定的 Conda 环境，再使用默认 fast_unit profile：

~~~bash
source "${AVENGINE_MINICONDA_PREFIX}/etc/profile.d/conda.sh"
conda activate "${AVENGINE_ENV_PREFIX}"
test "$CONDA_PREFIX" = "${AVENGINE_ENV_PREFIX}"

cd "${AVENGINE_CODE_ROOT}/AVEngine"
./scripts/setup.sh --profile fast_unit
~~~

非交互 shell 可以保留同一个 Conda prefix、但明确指向其解释器：

~~~bash
./scripts/setup.sh --profile fast_unit --python "${AVENGINE_ENV_PREFIX}/bin/python"
~~~

bootstrap 会验证所选解释器的 sys.prefix/conda-meta，拒绝系统 Python、普通
venv 及与已激活 CONDA_PREFIX 不一致的解释器。它直接向所选 Conda 环境安装
AVEngine/test 依赖、校验路径/schema 并运行普通 unit tests；不会创建 .venv。
可先加 --dry-run --skip-tests 查看精确命令；dry-run 不安装包或写输出。

## 显式 native-external 输入

需要 current installed-prefix 路线时，bootstrap 不替用户构建或选择一个
checkout，而是要求四个明确输入：

- AVENGINE_HABITAT_RUNTIME_PREFIX：非 Git 的已安装 Habitat facade、binding
  和 physics-config prefix；
- AVENGINE_HABITAT_MAGNUM_PYTHON_SITE：与当前 Python ABI 兼容的外置
  Corrade/Magnum site；
- AVENGINE_MP3D_ROOT：授权的外部 MP3D 数据根（其中有 scene_datasets/）；
- AVENGINE_RLR_SDK_ROOT：非 Git 的用户安装 RLRAudioPropagationPkg SDK；官方 Git checkout 仅用于获取或来源核对，不能作为 runtime root。

~~~bash
source "${AVENGINE_MINICONDA_PREFIX}/etc/profile.d/conda.sh"
conda activate "${AVENGINE_ENV_PREFIX}"
test "$CONDA_PREFIX" = "${AVENGINE_ENV_PREFIX}"

export AVENGINE_HABITAT_RUNTIME_PREFIX="/path/to/installed-habitat-runtime"
export AVENGINE_HABITAT_MAGNUM_PYTHON_SITE="/path/to/magnum-python/site-packages"
export AVENGINE_MP3D_ROOT="/path/to/licensed-mp3d-data"
export AVENGINE_RLR_SDK_ROOT="/path/to/RLRAudioPropagationPkg"

cd "${AVENGINE_CODE_ROOT}/AVEngine"
./scripts/setup.sh --profile native_external
~~~

该 profile 会在任何 pip/editable install 之前（包括 dry-run）进行只读输入预检：它拒绝把 runtime、Magnum site
或 RLR SDK 放在 Git checkout 内，但不会执行原生渲染、RLR 作业、UE、下载或
覆盖既有输出。具体 writer 还会在启动前验证 prefix 的 module、binding、SDK
布局、MP3D 场景和所需的房间资产；预检成功不等同于 native capture 或等价验证。

旧 --profile habitat_native 仍可解析为 native_external，以便现有脚本调用
得到清晰的显式输入错误，而不是回落到 sibling checkout。旧 v1 reader、schema
和历史 artifact 保持可读；它们不授权新运行重用旧 checkout 路径。

## Checkout-era 历史记录

此前的双仓库 clone、固定 Habitat commit、RLR submodule 和
`AVENGINE_HABITAT_RUNTIME_ROOT` 命令是迁移前的留档。需要审计其来源时查看
[`migration/LEGACY_SOURCE_LOCATIONS.md`](migration/LEGACY_SOURCE_LOCATIONS.md)
和 Git history；不要把这些命令复制到新的机器或当前 runbook。当前原生执行
请使用对应里程碑的 installed-prefix 命令，例如
[`M5.1 execution`](roadmap/M5_1_EXECUTION.md)。

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

### Checkout-era 原生运行时协作记录（归档）

原先要求在 habitat-sim-AVEngine fork、新分支和 RLR submodule 中修改原生
运行时的步骤，只描述迁移前的协作历史。它们不构成当前入口，不能被复制到
新的机器、bootstrap 或新的 native run。历史来源、固定提交和已留存 artifact
可通过 docs/migration/LEGACY_SOURCE_LOCATIONS.md 与 Git history 审计。

当前任何 AVEngine 所需的纯 C++、Python binding、RLR adapter 或 UE/SPEAR
集成代码，都应选择性迁入 AVEngine 自有路径并记录来源/许可证；不得以新增
外部 checkout、submodule 或 checkout-relative runtime root 代替迁移。若所需
代码尚未迁入或合法外置 runtime 尚未可用，相关 native 阶段应保持 not_run，
而不是在旧仓库新建分支。报告时列出 AVEngine commit、工作树状态、实际测试、
未运行的 native/data 层和仓库相对输出路径即可。

## 使用与许可边界

AVEngine 当前保留所有权利，协作者必须获得项目所有者授权。RLR Audio
Propagation 当前采用 CC BY-NC 4.0；默认声学路线只能用于符合其许可
条件的研究用途。
