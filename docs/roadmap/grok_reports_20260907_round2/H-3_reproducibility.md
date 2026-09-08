# H-3 复现性：生产 catalog、path_bindings、cwd 无关的房间包

按 `docs/roadmap/GROK_FIX_TASKS_ROUND2_20260907.md` 第 4 节（同第一轮
第 5 节报告格式）。工作树 `/data/jzy/tmp/wt-grok-pilot46-H3`，分支
`grok/pilot46-fixes-round2-20260907-h3`。未 push。未合并 main。未动
Studio。第一轮产物只读。未改 Claude 审计器、Codex/Claude 原文、阈值、
声学渲染、`qa_evidence.py`、`exposure_gate.py`、`batch_coverage.py`、
合并脚本。未改 `_looks_like_interface_defect`（H-2）、未改
validate-vs-resolve 顺序（H-4）、未给 scaleup-dry-run 补 `sound_pool`
或改打散距离（H-7）。

## 1. 改了哪些文件（路径），提交号

实现提交 `1be7715f1cd5102e12bb15f4607baf8657855946`（短号 `1be7715`）。
本句中的哈希由后续 docs 提交写入，避免 amend 后自指失效。

- `tools/dataset/build_qa_batch_manifest.py`：清单写入本工作树绝对
  `examples/rooms/packages/catalog.json`（Codex
  `wt-multi-home-activity-integration` 路径被替换）；把 catalog 的
  `path_bindings` 全量写入 `request.runtime.path_bindings`（请求侧已有
  键保留并覆盖）。相对 catalog 相对仓库根解析，不跟进程 cwd。`--catalog`
  仍是显式覆盖。
- `tools/dataset/run_qa_batch.py`：`producer.json` 增加 `argv`（完整 CLI）
  与 `avengine_environ`（全部 `AVENGINE_*`）。未改失败分类。
- `tools/studio/run_qa_episode.py`：`request_package_runtime` 以请求的
  `runtime.path_bindings` 为准、catalog 只补缺键；把 catalog 路径传给
  `package_from_catalog_entry`；plan 目录写展开后的房间包与所用绑定。
- `src/avengine/rooms/room_package.py`：`configured_path_bindings` 只吃
  显式 `runtime.path_bindings` / `mp3d_root`，不再 `dict(os.environ)`；
  `resolve_catalog_room_package_path` 相对 catalog 目录解析（生产 catalog
  里 `examples/rooms/packages/*.json` 按同目录 basename）；
  `write_room_package_plan_snapshot` 写 `plan/room_package.json` 与
  `plan/path_bindings.json`。`package_from_catalog_entry(..., catalog_path=)`
  仍是先 `validate_room_package` 再 `resolve_room_package_paths`，
  存在性检查顺序留给 H-4。
- 单测：`tests/unit/test_room_package_reproducibility.py`（新）、
  `tests/unit/test_build_qa_batch_manifest.py`（新）、
  `tests/unit/test_qa_batch_runner.py`（producer 记录）。

未改：`docs/TOOL_INDEX.md`（`tools/build_tool_index.py --check` 通过）。

## 2. 跑了哪些测试

项目 Python `/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python`，
`PYTHONPATH=src:tmp/native_python_addons_v1`。`import avengine` 解析到
`/data/jzy/tmp/wt-grok-pilot46-H3/src/avengine/__init__.py`。

```
pytest -q tests/unit/test_room_package_reproducibility.py \
          tests/unit/test_build_qa_batch_manifest.py \
          tests/unit/test_qa_batch_runner.py \
          tests/unit/test_room_package_path_existence.py \
          tests/unit/test_room_packages_p3.py \
          tests/unit/test_qa_production_contracts.py \
          tests/unit/test_qa_batch_manifest.py \
          tests/unit/test_qa_conditioned_transition.py
```

| 文件 | passed | failed | skipped |
|---|---:|---:|---:|
| `tests/unit/test_room_package_reproducibility.py` | 7 | 0 | 0 |
| `tests/unit/test_build_qa_batch_manifest.py` | 5 | 0 | 0 |
| `tests/unit/test_qa_batch_runner.py` | 17 | 0 | 0 |
| `tests/unit/test_room_package_path_existence.py` | 2 | 0 | 0 |
| `tests/unit/test_room_packages_p3.py` | 10 | 0 | 0 |
| `tests/unit/test_qa_production_contracts.py` | 20 | 0 | 0 |
| `tests/unit/test_qa_batch_manifest.py` | 16 | 0 | 0 |
| `tests/unit/test_qa_conditioned_transition.py` | 5 | 0 | 0 |
| **合计** | **82** | **0** | **0** |

`tools/build_tool_index.py --check`：exit 0。

覆盖：换 cwd 结果不变；缺绑定报错列出变量名；进程环境不能充当绑定源；
Codex catalog 被换成生产工作树；`producer.json` 记录 `AVENGINE_*` 与 argv；
请求 `path_bindings` 覆盖 catalog。

## 3. 验收产物与亲自核对

本项是生成器/装载复现性，没有新的 GPU 批次。第一轮产物只读。核对：

1. 生产 catalog 绝对路径是本树
   `/data/jzy/tmp/wt-grok-pilot46-H3/examples/rooms/packages/catalog.json`，
   含 15 条 `path_bindings`，其中包括
   `AVENGINE_MULTI_HOME_AUTHORING_ROOT`（Codex HEAD 2659048 没有这项）。
2. 在 cwd=`/tmp` 下对 `room_a` 条目调用
   `package_from_catalog_entry(..., catalog_path=生产 catalog)`：
   `room_id=aea_loc3_social_rebuild_v1`，`acoustic_package` 已展开为绝对路径
   且不再含 `${`，`acoustic_package_template` 仍保留模板。换 cwd 到仓库根
   结果相同（单测
   `test_production_catalog_relative_package_is_cwd_independent`）。
3. 把 decoy `room.json` 放进另一目录并 chdir 过去，仍加载 catalog 旁边的包，
   不会吃到 cwd 里的 decoy。
4. 缺 `${AVENGINE_MISSING_A}` / `${AVENGINE_MISSING_B}` 时，即使这两项在
   `os.environ` 里，报错仍是
   `RoomPackage missing configured path roots: AVENGINE_MISSING_A, AVENGINE_MISSING_B`。
5. 旧 request 没有 `runtime.path_bindings` 时，控制器仍从 catalog 文件补齐
   （`request_package_runtime`）；新清单把绑定写进 request 后不再依赖 shell。

未重出 46 格清单或放量 350 条（H-7 的声音池/距离配置尚未接上；第一轮产物
只读）。H-7 重出放量清单时应走本生成器。

## 4. 没做完的部分

- **题义不适用**：未重跑 attempt、未渲音频、未改覆盖表。复现性修的是
  生成器与装载，不需要 GPU。
- **接口未实现**：无。本项要求的 catalog 绝对路径、`path_bindings` 写全、
  producer 记录、catalog 相对解析、plan 快照均已接线。
- **证据缺失**：没有用新生成器重出一份可跑的 46 格/放量清单产物（留给
  H-7 干跑，且要等 H-7 把 `sound_pool` 与距离配置补上之后才有意义）。
  第一轮 `qa_pilot46_rerun_20260907_v1` 的 request 仍指向 Codex catalog、
  仍无 `runtime.path_bindings`，那是只读旧产物，不是本项回归范围。

## 5. 对后续接口的要求

- `package_from_catalog_entry(entry, *, runtime=None, catalog_path=None)`：
  相对 `room_package` 必须带 `catalog_path`，否则 `ValueError`。
- `resolve_room_package_paths` 只使用 `runtime.path_bindings` 与
  `runtime.mp3d_root`。缺根时报
  `RoomPackage missing configured path roots: <VAR>, ...`。
- 清单请求：`request.room_catalog` 为绝对路径；
  `request.runtime.path_bindings` 为 string 到 string 的全表。
- 控制器 plan 目录新增 `path_bindings.json`：
  `{"path_bindings": {...}, "catalog_path": "..."}`。既有
  `plan/room_package.json` 仍是展开后的包。
- `run_qa_batch.py` 的 `producer.json` 新增 `argv`（list of str）与
  `avengine_environ`（只含 `AVENGINE_*`）。`execute_batch(..., argv=)`
  可选；`main()` 写入完整 CLI。
- H-4 改存在性检查顺序时，请保持先加载 catalog 相对路径、再用显式绑定
  展开这两步；不要再引入 `os.environ` 作为绑定源。
- H-7 重出放量清单必须走本生成器，这样 `room_catalog` / `path_bindings`
  才会完整。

## 6. 需要 owner 拍板的地方

无。Codex 工作树 catalog 替换为本工作树、以及不再用 shell 环境补绑定，
都是任务书与审核 R1 的修法，没有另作产品裁定。`--catalog` 仍可作为显式
覆盖（测试或非生产 catalog）。
