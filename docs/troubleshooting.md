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

## 无法克隆 Eastforward 仓库

确认当前账号拥有两个仓库的 GitHub 权限，并配置了自己的 SSH 密钥：

```bash
ssh -T git@github.com
git ls-remote git@github.com:Eastforward/AVEngine.git HEAD
git ls-remote \
  git@github.com:Eastforward/habitat-sim-AVEngine.git \
  refs/heads/feature/m6-release-state
```

同服务器账号权限不会自动赋予 GitHub 仓库权限。

## Habitat 运行时提交不正确

`habitat-sim-AVEngine` 的默认 `main` 当前仍是上游基线。必须检出
`manifest.yaml` 固定的提交：

```bash
git -C "${AVENGINE_CODE_ROOT}/habitat-sim-AVEngine" fetch origin
git -C "${AVENGINE_CODE_ROOT}/habitat-sim-AVEngine" checkout --detach \
  e9c81c10834f7e89f33f4e0602c75535a84e054b
```

不要使用 `git reset --hard` 清理不属于自己的改动。

## RLR 或其他子模块缺失

初始化 Habitat 的全部递归子模块：

```bash
git -C "${AVENGINE_CODE_ROOT}/habitat-sim-AVEngine" \
  submodule sync --recursive
git -C "${AVENGINE_CODE_ROOT}/habitat-sim-AVEngine" \
  submodule update --init --recursive --jobs 8
```

核对 RLR：

```bash
git -C "${AVENGINE_CODE_ROOT}/habitat-sim-AVEngine/src/deps/rlr-audio-propagation" \
  rev-parse HEAD
```

预期固定提交是
`4fd446b4abb5c71fb7a232a083bbddd65f25fc6f`。

## Habitat 能导入但没有音频能力

原生构建必须显式设置：

```bash
HABITAT_BUILD_GUI_VIEWERS=OFF \
HABITAT_WITH_BULLET=ON \
HABITAT_WITH_AUDIO=ON \
HABITAT_WITH_CUDA=OFF \
"${AVENGINE_ENV_PREFIX}/bin/python" -m pip install \
  -e "${AVENGINE_CODE_ROOT}/habitat-sim-AVEngine" \
  --no-build-isolation
```

验证：

```bash
"${AVENGINE_ENV_PREFIX}/bin/python" - <<'PY'
import quaternion
import habitat_sim
assert habitat_sim.audio_enabled
assert habitat_sim.built_with_bullet
print(habitat_sim.__file__)
PY
```

## 直接导入 habitat_sim 时进程崩溃

当前固定运行时存在已知导入顺序问题。始终先：

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

降低并行度后重新执行个人构建：

```bash
CMAKE_BUILD_PARALLEL_LEVEL=1 \
HABITAT_BUILD_GUI_VIEWERS=OFF \
HABITAT_WITH_BULLET=ON \
HABITAT_WITH_AUDIO=ON \
HABITAT_WITH_CUDA=OFF \
"${AVENGINE_ENV_PREFIX}/bin/python" -m pip install \
  -e "${AVENGINE_CODE_ROOT}/habitat-sim-AVEngine" \
  --no-build-isolation
```

不要删除或覆盖别人的 build 目录来释放空间。

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
