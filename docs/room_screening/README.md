# 房间筛选工具

本目录包含房间范围/可行域筛选方法文档的仓库副本、初步几何诊断脚本和一个本地抽查页。

## 当前状态

- [主方案](ROOM_SCREENING_PLAN.md) 定义研究范围、面积口径、核验和后续筛选流程。
- [操作规则](ROOM_SCREENING_RULES.md) 将地面、家具、范围和连通性判断写成可执行检查。
- 命令行代码目前提供 HM3D 语义地面与 navmesh 初步交叠、家具实例占地诊断、输入清单生成和 manifest 驱动的本地抽查页。
- 这些脚本输出的是**初步几何诊断**。它们尚未实现 Plan 中的全套边界人工核验、正式主面积、抽样/阈值验证、形状自动筛选或自动弃用。不能将其输出称为最终准确面积或正式房间准入结果。
- 真实 HM3D 场景、房屋列表、渲染图片/视频、运行清单、抽查意见和生成结果应放在外部数据目录或仓库 `tmp/`，不提交 Git。

## 环境

从 AVEngine 仓库根目录安装项目及房间筛选可选依赖：

```bash
python -m pip install -e '.[room-screening]'
```

准备 AVEngine 支持的 Habitat 运行时和有权访问的 HM3D 数据。可用环境变量配置：

```bash
export AVENGINE_HABITAT_RUNTIME_PREFIX=/path/to/habitat-runtime
export AVENGINE_HABITAT_MAGNUM_PYTHON_SITE=/path/to/magnum/site-packages
export AVENGINE_MP3D_ROOT=/path/to/licensed/habitat-data-root
# 仅当所用 Habitat 安装需要外部 RLR 共享库时设置：
export AVENGINE_RLR_SDK_ROOT=/path/to/RLRAudioPropagationPkg
```

也可以在命令行逐次传入这些参数（`--rlr-sdk-root` 仅在运行时需要 RLR 库时使用）。不要将真实数据根路径写进仓库文件。

## 计算流程

先在外部文本文件中列出本次允许处理的 HM3D 场景 ID，每行一个，格式为 `hm3d_<split>_<5位场景序号>_<scene-id>`。这个文件属于输入数据，不要提交。

```bash
python -m tools.rooms.room_screening.build_inventory \
  --houses-file /path/to/private-houses.txt \
  --dataset-root /path/to/hm3d \
  --output tmp/room_screening/input_inventory.json

python -m tools.rooms.room_screening_candidates \
  --inventory tmp/room_screening/input_inventory.json \
  --dataset-root /path/to/hm3d \
  --output tmp/room_screening/candidate_area_inventory.json

python -m tools.rooms.room_screening_furniture_audit \
  --inventory tmp/room_screening/input_inventory.json \
  --raw-area-path tmp/room_screening/candidate_area_inventory.json \
  --dataset-root /path/to/hm3d \
  --output-dir tmp/room_screening/furniture_area_audit
```

第一步从显式场景清单、HM3D 语义标注和已保存 navmesh 生成输入清单；缺少语义标注、navmesh 或无法解析的场景会标为不可评估，不会伪造面积。第二步计算语义地面投影与 navmesh 的候选交集；第三步对该交集做家具实例交叠诊断。家具审计的试算修正仍是临时代理指标，并非 Plan 定义的最终主面积。运行参数和命令产生的清单保存在本机 `tmp/`，不要复制到公开提交中。

## 本地抽查页

先准备外部抽查项索引 JSON。每项至少有 `id`、`image_path`；可选 `geometry_path`、`video_path` 和展示字段。路径可以相对 `--asset-root`，也可以是该目录内的绝对路径。路径越出 asset root 会被拒绝。

```bash
python -m tools.rooms.room_screening_manifest \
  --input /path/to/private-review-items.json \
  --asset-root /path/to/private-review-assets \
  --output tmp/room_screening/review_manifest.json

python -m tools.rooms.room_screening_review \
  --manifest tmp/room_screening/review_manifest.json \
  --asset-root /path/to/private-review-assets
```

默认仅监听 `127.0.0.1:8772`；浏览器打开终端打印的本机地址。抽查意见自动保存到 `tmp/room_screening/review_feedback.json`，不写正式 verdict。`--feedback` 可指定其他本地输出路径。不要将服务器绑定到公网接口。

Manifest 结构见 [JSON Schema](../../schemas/room_screening_review_manifest_v1.schema.json)。叠加几何文件为 JSON，包含可选 GeoJSON 几何键 `ground`、`raw`、`corrected`、`blockers`、`gap`、`unknown`；坐标为 `world_xz` 时须提供 `view`、`projection`、`floor_y_m` 和 `image_size`，为 `image_pixels` 时几何坐标直接对应底图像素。`bbox_xz_m` 仅作为数据 AABB 参考框。抽查页只展示已生成的图片/几何，不负责 Habitat 渲染、房间边界判定或面积计算。

## 验证

```bash
python -m unittest tests.unit.room_screening.test_geometry tests.unit.room_screening.test_review_server tests.unit.room_screening.test_inventory_contract -v
python tools/build_tool_index.py --check
```

上述单测使用合成几何和临时文件；不需要 HM3D 文件，也不测试 Habitat 原生场景加载。项目此前已用真实 HM3D 场景进行过面积初算和样本抽查。2026-10-02，本分支的移植版本又在一栋真实 HM3D 场景上完成了端到端冒烟回归：清单识别 11 个语义区域，候选面积步骤为其中 10 个区域生成面积，家具诊断完成且没有场景级运行失败。该测试验证了真实数据上的启动、路径和执行链路，**不代表这些面积已被人工确认，也不代表 181 栋已全量重算或结果准确率已得到证明**。具体运行输出和场景标识保留在外部工作区，没有提交到本仓库；此前的详细运行结果和审核记录也不随代码公开。

## 公开与数据边界

本分支不包含真实场景清单、扫描资产、审核图像/视频、审核文本、运行输出或反馈 JSON。方法文档是去除样本编号和个人工作区追溯信息后的仓库副本。任何真实场景 ID、任务记录或审阅产物仅在本地外部输入中使用。
