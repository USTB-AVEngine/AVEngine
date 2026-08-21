## 任务与基线

- 任务 ID：
- PR 目标分支：
- base SHA：
- head SHA：

## 改动范围

- 本 PR 解决的问题：
- 主要修改文件：
- 明确未修改的相邻模块：

## 复用说明

- 复用的现有 schema、registry、runner、validator 或 CLI：
- 如新增入口，现有入口无法扩展的原因：

## 验证

- 实际运行的命令与结果：
- `pass`：
- `skip`：
- `not_run` 及原因：
- canary/视频/证据的仓库相对路径：

## 边界与交接

- 当前不能宣称的内容：
- 已知未完成项：
- 工作树是否干净：

## 提交前检查

- [ ] `origin` 是个人 Fork，`upstream` 是 `USTB-AVEngine/AVEngine`。
- [ ] 分支和 PR base 符合任务分配，没有直接使用 `main`。
- [ ] 一个分支只包含一个任务，没有无关重构或全仓库格式化。
- [ ] 没有提交 `tmp`、RIR、媒体、数据集、权重、环境、构建目录或未经选择和来源/许可证审计的第三方代码。
- [ ] 没有加入私有绝对路径、密钥或未经允许的数据。
- [ ] 已运行 `git diff --check`，并显式检查了待提交文件。
- [ ] 没有重写已有 M7、AudioProgram、RIR cache 或场景 selector；如确有必要，已在复用说明中给出证据。
