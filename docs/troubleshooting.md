# 故障排查

本文只覆盖默认 AVEngine + Habitat/RLR 路线。个人目录和环境规则见
[`同服务器协作与个人环境构建`](quickstart.md)。不要通过使用其他人的
Conda 环境、工作树或输出目录来绕过错误。

## 找不到 conda 命令

不要调用服务器上其他用户的 `conda`。重新加载本人安装的 Miniconda：

```bash
source "${AVENGINE_MINICONDA_PREFIX}/etc/profile.d/conda.sh"
conda activate "${AVENGINE_ENV_PREFIX}"
```

核对实际路径：

```bash
test "${CONDA_PREFIX}" = "${AVENGINE_ENV_PREFIX}"
command -v conda
"${AVENGINE_ENV_PREFIX}/bin/python" -c \
  'import sys; print(sys.executable)'
```

如果个人 Miniconda 不存在或安装不完整，按照
[`quickstart.md`](quickstart.md) 重新下载并核对安装器。禁止覆盖一个
来源不明的现有目录。

## Conda 环境被建到了错误位置

协作环境必须使用显式前缀：

```bash
"${AVENGINE_MINICONDA_PREFIX}/bin/conda" create \
  --yes \
  --prefix "${AVENGINE_ENV_PREFIX}" \
  --channel conda-forge \
  python=3.12 cmake=3.27 ninja pip
```

不要使用不带路径约束的共享环境，也不要修改他人的 `.condarc`。

## 无法克隆 AVEngine

当前入口只需要 AVEngine 的读取权限和自己的 SSH 密钥：

```bash
ssh -T git@github.com
git ls-remote git@github.com:USTB-AVEngine/AVEngine.git HEAD
```

同服务器账号权限不会自动赋予 GitHub 仓库权限。不要为了 bootstrap 另外
clone `habitat-sim-AVEngine`、SPEAR 或 RLR；`scripts/setup.sh --clone-runtime`
已撤销并会以退出码 2 停止。

## Native external 输入缺失或被拒绝

当前 bootstrap 必须使用项目指定的 Conda Python；它不会创建 venv。先激活
对应环境（或在非交互 shell 中使用同一 prefix 的 --python）：

~~~bash
source "${AVENGINE_MINICONDA_PREFIX}/etc/profile.d/conda.sh"
conda activate "${AVENGINE_ENV_PREFIX}"
test "$CONDA_PREFIX" = "${AVENGINE_ENV_PREFIX}"

export AVENGINE_HABITAT_RUNTIME_PREFIX="/path/to/installed-habitat-runtime"
export AVENGINE_HABITAT_MAGNUM_PYTHON_SITE="/path/to/magnum-python/site-packages"
export AVENGINE_MP3D_ROOT="/path/to/licensed-mp3d-data"
export AVENGINE_RLR_SDK_ROOT="/path/to/RLRAudioPropagationPkg"

cd /path/to/AVEngine
./scripts/setup.sh --profile native_external --dry-run --skip-tests
~~~

如果报 selected Python must resolve to a Conda environment，选择/激活正确的
Conda prefix；不要安装 python3-venv 或让 bootstrap 创建新环境。去掉
--dry-run 后，预检仍先确认目录存在；这一检查始终发生在 pip/editable install 之前，并拒绝 runtime prefix、Magnum site 或
RLR SDK 位于 Git checkout 内（包括经符号链接进入旧 checkout 的情况）。MP3D
是授权外部数据而非源码 checkout；其具体场景闭包由 writer 在运行前验证。
不要设置已退役的 AVENGINE_HABITAT_RUNTIME_ROOT；setup 会忽略它。

## RLR SDK 缺失

不要初始化旧 Habitat 子模块。本服务器批准的 runtime SDK 根是 /data/avengine_external/rlr-sdk/RLRAudioPropagationPkg；它是非 Git 目录，含 header、library、LICENSE 和 README。其他用户可从官方 Git 获取或核对来源，但必须把运行 SDK 放在非 Git 根。`AVENGINE_RLR_SDK_ROOT` 必须指向用户合法安装
的 `RLRAudioPropagationPkg`，其中应有 `headers/RLRAudioPropagation.h` 和
`libs/linux/x64/libRLRAudioPropagation.so`。M3/M4 current runtime 会在 native
调用前验证该布局和动态加载来源；缺失时应保持 `not_run`/`blocked` 边界，而
不是回退到一个 Git checkout 的 `.so`。

## Habitat 安装 prefix 不能导入或能力不足

不要执行 `pip install -e` 指向旧 Habitat checkout。确认 prefix 是一次显式
安装产物，并按具体 current writer 的 `--runtime-prefix`、`--magnum-python-site`
及外部 SDK 参数运行。若 prefix 缺少 facade、binding、physics config 或所需
adapter，应重新在 AVEngine 的 native-source 迁移路径中构建/安装，或将该 native
阶段记录为未运行；快速单元测试不能替代它。

## 直接导入 habitat_sim 时进程崩溃

已安装 Habitat prefix 仍可能有已知导入顺序问题。始终先：

```python
import quaternion
import habitat_sim
```

不要把一次直接 `import habitat_sim` 的崩溃误判为整个构建不存在。

## 找不到 cmake、ninja 或编译器

先核对个人环境：

```bash
"${AVENGINE_ENV_PREFIX}/bin/cmake" --version
"${AVENGINE_ENV_PREFIX}/bin/ninja" --version
c++ --version
```

CMake 和 Ninja 安装在个人 Conda 环境中。C/C++ 编译器、EGL/Mesa 库和
显卡驱动属于服务器系统层；缺失时联系管理员，不要使用 `sudo` 或借用
其他人的环境。

## 原生构建内存不足

不要通过删除或覆盖其他人的 build 目录、或重新启用旧 checkout 的 editable
install 来绕过内存不足。记录可用内存、并发度和首次真实错误；当前 bootstrap
可以继续运行 fast-unit，native 阶段则保持未运行，直到 AVEngine 的 native-source
构建路径和合法外置输入可用。

## 原生视觉无法创建 EGL 上下文

先记录当前显卡和 EGL 环境：

```bash
nvidia-smi
eglinfo 2>/dev/null | head
```

该问题属于驱动、设备可见性或系统 EGL 库时，应交给管理员处理。不要
修改全局驱动或其他用户的 `LD_LIBRARY_PATH`。

## 找不到 HRTF

双耳 RIR 默认需要：

```text
/usr/share/libmysofa/MIT_KEMAR_normal_pinna.sofa
```

如果系统未提供该文件，双耳测试应记录为 `blocked` 并联系管理员。FOA
或不依赖 HRTF 的上层检查仍可单独运行。

## 找不到 MP3D 或 ReplicaCAD

这些数据不随 Git 克隆提供。确认项目所有者提供了只读共享路径，并按
房间注册表设置对应环境变量。不要把大数据复制进 Git 仓库，也不要改写
公共数据。

仅运行快速单元测试和仓库内合同验证时不需要这些场景。

## 输出目录已经存在

正式工具采用禁止覆盖策略。应选择新的个人运行标识：

```bash
export AVENGINE_RUN_ID="your_new_run_id"
```

不要删除旧证据，也不要让两个人写同一个输出目录。先检查旧目录的归属和
状态，再决定是否使用新的输出路径。

## tmp 已经存在或指向未知位置

先检查：

```bash
cd "${AVENGINE_CODE_ROOT}/AVEngine"
ls -ld tmp
readlink -f tmp
```

如果它不属于当前协作者的个人输出目录，停止操作并确认归属。禁止直接
覆盖、删除或重新链接未知 `tmp`。

## UE、SPEAR、Blender 或生成模型问题

这些不是默认构建依赖。先确认任务是否真的需要可选后端：

- 只开发任务、轨迹、RLR、音频和 Habitat 路线时，不安装 UE/SPEAR；
- 只运行上层检查时，不安装 Blender 或生成模型；
- 需要历史/可选后端时，阅读 [`legacy/OPTIONAL_BACKENDS.md`](legacy/OPTIONAL_BACKENDS.md)，
  并单独记录未满足的外部依赖。

不要让可选后端安装失败阻塞默认 AVEngine + Habitat/RLR 环境验证。
