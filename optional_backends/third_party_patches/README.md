# 可选第三方源码补丁保存区

本目录保存 AVEngine 历史或可选资产生成路线对第三方源码工作区所做的精确
文本修改。补丁来自对应工作区在表列基线 commit 上执行的标准
`git diff`；补丁文件本身的 SHA-256 同时也是本次冻结的 diff SHA-256。

这里的“已保存”不表示“已批准”“已合并上游”或“已通过端到端验证”。
Habitat-native 默认安装、导入和测试均不得自动下载、应用或加载这些补丁、
模型、权重或生成工具。

## 保存内容

| 组件 | 上游基线 | 补丁 | 当前结论 |
| --- | --- | --- | --- |
| Pixal3D | `cdbb2bbffbf4e6f298b5f2af3d1d76a8d823d2af` | [`pixal3d-cdbb2bb-persistent-pipeline.patch`](pixal3d-cdbb2bb-persistent-pipeline.patch) | 候选补丁；为调用方提供常驻 pipeline 复用接口，待 Fork、测试和真实双任务 canary |
| Hunyuan3D-2.1 | `82920d643c0dc2f7bfd7255f45f62d386edfe60c` | [`hunyuan3d-2.1-82920d6-local-only-compat.patch`](hunyuan3d-2.1-82920d6-local-only-compat.patch) | 仅研究保存；当前本地路径策略与既有相对路径及权重符号链接不兼容，不得默认应用 |
| MaterialAnything | `be3d6b32a195f968540abc2ee106dc02d4b07479` | [`materialanything-be3d6b3-mesh-mask-compat.patch`](materialanything-be3d6b3-mesh-mask-compat.patch) | 候选兼容补丁；待 Blender 与低分辨率 PBR canary |

SkinTokens 工作区没有 tracked 源码 diff，只有外部 checkpoint、环境和输出
链接，因此本目录不为它生成空补丁。

机器可读的 URL、完整 commit、补丁哈希、状态、许可证边界与测试结果见
[`manifest.yaml`](manifest.yaml)。

## 校验与应用

先取得对应上游仓库并精确 checkout manifest 中的 `base_commit`。不要在
含有其他修改的工作区直接应用：

```bash
git checkout --detach <base_commit>
git status --short
git apply --check /path/to/AVEngine/optional_backends/third_party_patches/<patch>
git apply /path/to/AVEngine/optional_backends/third_party_patches/<patch>
```

应用后仍需运行 manifest 列出的未完成测试。`git apply --check` 只证明
文本补丁可施加到固定基线，不证明模型能加载、GPU 推理正确、Blender/PBR
输出合格，也不改变任何许可证或数据准入状态。

## 禁止进入 Git 的内容

本目录只保存小型源码 diff 和元数据，禁止加入：

- checkpoint、模型权重、Hugging Face snapshot 或其符号链接；
- Conda/venv、编译扩展、CUDA cache 和下载 cache；
- 输入图片、AudioSet 下载图片、GLB/OBJ、纹理、turntable 和网页图库；
- `tmp/`、`outputs/`、`demo/` 生成结果和其他实验工作区内容。

这些外部内容如确有保留需要，应存放在批准的模型、数据或输出根中，以独立
manifest 记录来源、许可证、revision、大小和内容哈希；补丁目录不得成为
权重或生成资产的分发通道。

## 许可证边界

- Pixal3D 源码标为 MIT，但必须保留其 `NOTICE`，并分别遵守模型及嵌套依赖
  条款。
- Hunyuan3D-2.1 使用 Tencent Hunyuan 3D 2.1 Community License，存在地域、
  使用和分发限制。分发修改时还要求附带当前许可证、Notice，并在修改文件
  中作显著修改声明。本保存动作不解除 `THIRD_PARTY_NOTICES.md` 中的 release
  hold。
- MaterialAnything 源码标为 MIT；其模型、ControlNet 等嵌套组件仍按各自
  条款处理。

本目录是工程可复现记录，不是法律意见或重新授权。
