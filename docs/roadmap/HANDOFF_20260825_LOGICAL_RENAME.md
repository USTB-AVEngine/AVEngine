# HANDOFF 20260825：仓库按逻辑重命名（给新会话）

> 任务：把 AVEngine 仓库里按流水线阶段编号命名的东西（`m1`…`m7`）重构成
> **按能力命名**，并保证**每项功能仍然正确**。
> 服务器 `ssh 48g-jump`；主仓 `/data/jzy/code/AVEngine-lead-a`；
> 推 main 用 `git push origin HEAD:main`（owner 有 bypass）。
> 本文所有数字都是 2026-08-25 在服务器上实测的，接手后请自己复核一遍。

---

## 0. 为什么做这件事（不是洁癖，是在止损）

`m1`…`m7` 编码的是**流水线依赖顺序**，那是实现视角；而人找工具时想的是
**"我要做什么"**。一天之内因此造成六次真实返工：

1. Codex 写了个"只支持双比格"的 MP3D 演员生成器——而 `m5_1/human_runtime.py`
   早就把人物编译进 Habitat 了，他没找到；
2. MP3D 音频批明确"不用持久 RIR 缓存"——而 `m6x/rir_cache.py` 是现成的，
   白烧约 12 小时 GPU/CPU 时间；
3. 我断言"人物只有 UE 那条腿"——错的，owner 当场纠正；
4. 公寓渲染器还在走两点直线——而 `m5/mp3d_region_source_planner.py` 和
   Kujiale 后端**早就会走折线**，同一个仓库里一半代码会、一半不会；
5. `navigation_service` 的 Python 封装躺在 `backends/spear_ue/client/services/`
   里从没人调用过，我花了 8 轮探针才摸清用法；
6. 帧事务的正确写法 `run_frame_transaction` 就在 `research_runtime.py` 里，
   我照猜写成 `with begin_frame(), end_frame():` 直接死锁 15 分钟。

**所以重构的目标不是"名字好看"，是让"这个能力有没有、在哪、怎么用"三个问题
能在一次 grep 内回答。**

---

## 1. 现状盘点（实测）

### 1.1 代码规模

| 位置 | 阶段式 | 逻辑式 | 合计 |
|---|---|---|---|
| `src/avengine/` | m1:4, m2:30, m3:18, m4:10, m5:14, m5_1:16, m6:15, m6x:19, m7:10 = **136** | 顶层 15 个模块 + `appearance/ backends/ contracts/ motion/ optional_backends/ qa/ security/ studio/` | 225 |
| `tools/` | m1:3, m2:34, m3:6, m5:1, m5_1:8, m6:3, m6x:13, m6y:13, m6z:9, m7:15 = **105** | qa:84, assets:11, studio:6, blender:5, motion:4, ue:3, mesh:1, release:1 = **115** | 220 |
| `examples/` | m1…m7 目录 | `assets/ qa/ qa_v2/` | — |

**关键事实：一半以上的工具已经是逻辑命名的**（`qa` 一个目录就有 84 个）。
迁移已经自发做了一半，这次是把剩下的一半补齐。

### 1.2 引用规模（改名要动多少地方）

```
avengine.m1  被  56 个文件引用      avengine.m6   被  52
avengine.m2  被 114 个文件引用      avengine.m6x  被  72
avengine.m3  被  61 个文件引用      avengine.m7   被  41
avengine.m4  被  57 个文件引用      avengine.m5   被  47
avengine.m5_1 被 76 个文件引用
```

另外 `examples/` 有 4 个、`docs/` 有 17 个文件里**写死了** `tools/m*` 或
`avengine.m*` 路径。CLI 里还有按阶段命名的子命令：`m1` / `m3` / `m4`
（另有逻辑式的 `capture` / `aggregate` / `verify`）。

### 1.3 最大的风险：34 个工作树、32 条 codex 分支

```bash
git worktree list | wc -l     # 34
git branch -a | grep -c codex # 32
```

**一次性全量改名会让这 32 条未合并分支每一条都产生成百上千处冲突。**
这是本任务最重要的约束，方案必须正面回答它（见 §3）。

---

## 2. 目标命名（起点，不是圣旨）

我已经按能力把 210 个工具分了 12 组，并写了一个从 docstring 自动生成索引的
生成器。**它已经提交但按 owner 要求没有推远端**：

```
分支 feature/tool-capability-index，commit 0fa2252
  docs/TOOL_INDEX.md          （303 行，自动生成）
  tools/build_tool_index.py   （148 行，带 --check 可在回归里发现索引过期）
工作树：/data/jzy/tmp/wt-index
```

**先把它推上去**（它是零冲突的纯新增），然后以它的分组作为改名的目标结构：

| 能力组 | 现在散落在 | 建议归到 |
|---|---|---|
| 资产生成与装配 | `m2/` 大部分、`blender/`、`assets/` | `tools/assets/`、`src/avengine/assets/` |
| 资产验收与探针 | `m2/audit_*`、`m2/probe_*` | `tools/assets/audit/` |
| 房间与声学场景包 | `m1/`、`m3/` | `tools/rooms/`、`src/avengine/rooms/`、`.../acoustics/` |
| 相机、路径与时间线 | `m6x/`、新的 `route_sampling` | `tools/routes/`、`src/avengine/routes/` |
| 视觉捕获与渲染 | `m5/`、`m5_1/`、`m7/` | `tools/capture/`、`src/avengine/capture/` |
| 空间音频 | `m4/`、`m7/` 的音频部分 | `tools/audio/`、`src/avengine/audio/` |
| 出题与认证 | 已是 `qa/` ✅ | 不动 |
| 评测与训练数据 | `qa/` 里混着 | `tools/eval/` |
| 审阅与可视化 | `m6y/`、`m6z/`、`studio/` | `tools/review/`、`studio/` 保持 |
| 注册表、准入与发布 | `m6/`、`release/` | `tools/registry/` |
| UE / stage 工程 | `ue/`、`m6x` 的一部分 | 已是 `ue/` ✅ |

**接手后请先自己核对这张表**——分组是关键词自动分的，肯定有错分，尤其
"其他/诊断"那一组。

---

## 3. 必须遵守的实施策略（针对 32 条分支的约束）

### 3.1 分阶段，不要一次全改

建议顺序（每阶段独立提交、独立回归、独立推 main）：

1. **阶段 0：推索引**（零冲突，立刻见效）；
2. **阶段 1：只加不改**——新建逻辑包，用 `from ... import *` 或显式再导出
   做**转发层**，旧路径继续可用。此时 32 条分支毫发无损；
3. **阶段 2：迁移调用点**——把 `src/`、`tools/`、`tests/` 里的引用逐组改到
   新路径（一组一提交），旧模块保留转发。分支合并时冲突面只有"改了引用的
   那几行"，不是整文件移动；
4. **阶段 3：等 codex 分支收敛后**，再删掉转发层、真正移动文件。

**如果 owner 明确要求一次性硬改**：那必须先协调 Codex 那条线停手并把分支合掉，
否则代价会转嫁成几十次冲突解决。这一点要在动手前跟 owner 确认，不要自己决定。

### 3.2 不要动的东西（改了会砸掉现成数据）

- **外部产物路径**：`/data/avengine_external/m6x-canary-inputs/`、
  `/data/datasets/avengine_workspaces/.../tmp/m5_1/`、
  `ue-package-stages/`、`review/` 下的一切。这些路径写在**已生成的
  receipt、manifest、注册表**里，改名等于让历史证据失效；
- **注册表里的 schema 名与 id**（`avengine_m6_entity_asset_registry_v1` 之类）：
  它们参与 `registry_content_sha256` 的计算，改名会让所有哈希失配；
- **`/Game/...` UE 内容路径**：那是 cook 进 stage 的，改名要重新打包；
- **正在跑的作业**：动手前 `ps` 一遍，别改别人正在读的工作树。

### 3.3 铁律（不因重构豁免）

全链 `research_only=true` / `episode_counted=false` / 正式分母 0；
fresh/no-clobber；不新增无理由的 hash/gate；`blender_custom` 与 Skokloster
不进生产；进 main 由 owner 决定。

---

## 4. 怎么保证"每项功能都还正确"

### 4.1 基线（先跑一遍，记下来）

```bash
cd /data/jzy/code/AVEngine-lead-a   # 或自己的 worktree
PYTHONPATH=src /data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python \
  -m pytest tests/unit -q
```

**当前基线：3007 passed / 118 skipped / 52 subtests / 0 failed（约 3.5 分钟）。**
每个阶段结束后必须回到这个数字（passed 只能增不能减，skipped 必须**一模一样**）。

### 4.2 单测覆盖不到的地方（必须额外验证）

单测**不会**发现这些断裂，改名最容易在这里出事：

| 风险面 | 怎么验 |
|---|---|
| `docs/` 17 个文件、`examples/` 4 个文件里写死的路径 | `grep -rn "tools/m[0-9]\|avengine\.m[0-9]" docs/ examples/` 必须归零或指向新路径 |
| CLI 子命令 `m1/m3/m4` | 改名要**保留旧子命令作为别名**，并各跑一次 `--help` |
| 服务器上 `/data/jzy/tmp/*.sh|*.py` 的一次性 runner | 它们引用 `tools/...` 绝对路径；`grep -l "tools/m" /data/jzy/tmp/*` 逐个修或标注废弃 |
| `tools/studio/studio_config_48g.json` 的 task_templates | 里面全是工具路径；改完必须重启 Studio 并跑一次 `/api/health` + 提交一个任务 |
| 动态导入 / 字符串拼出来的模块名 | `grep -rn "importlib\|__import__\|getattr(avengine" src/ tools/` |
| 已安装包 vs 源码 | 环境里的 `avengine` 指向 canonical worktree；**跑测试必须带 `PYTHONPATH=src`**，否则你测的是别人的代码（我踩过） |

### 4.3 功能级冒烟（比单测更能说明问题）

每个阶段至少跑一次这三条，**产物必须与改名前逐字节或逐字段一致**：

1. **出题**：`tools/qa/generate_qa_v2_questions.py` 对 pilot48 输入重跑，
   194 题应与 `review/qa_v2_pilot48_questions_v1/questions.json` **完全一致**
   （这是现成的回归基准，之前用过）；
2. **路径库**：`tools/m6x/plot_route_bank.py` 用现成的
   `review/apartment_route_bank_20260825T0700Z/route_bank.json` 重画一张图，
   能出图即说明 `route_sampling` 链路完好；
3. **Studio**：起服务 + `curl /api/health` + `/api/templates` 返回模板列表。

**不要**为了验证去跑 UE 捕获或 RLR 音频（贵且占卡）；上面三条足够覆盖导入链。

---

## 5. 已知的坑（今天现踩的，别重蹈）

1. **`PYTHONPATH=src` 必须带**，否则 import 的是 canonical worktree 里的旧代码，
   你会看到"改了没生效"的幻觉；
2. **别在工作树里跑 SpearSim**：它会往 cwd 写 `tmp/spear_instance_<port>/`，
   而保留工作区的测试判断条件正是"仓库里有没有 `tmp`"——一跑就从 skip 变成
   **28 failed + 21 errors**，看起来像代码回归，其实是环境污染。
   `tools/m6x/build_apartment_route_bank.py` 里的 `_scratch_working_directory()`
   是正确做法，可以照抄；
3. **`pkill -f` 会匹配到自己的 ssh 命令行**，把自己杀掉；用 `ps -p` 精确杀或
   `fuser -k <port>/tcp`；
4. **zsh**：`===` 会被当 glob 报错；`--include=*.py` 未加引号会 no matches；
   grep 零命中 exit 1 会断 `&&` 链；
5. **复杂脚本别用 heredoc 塞进 ssh**（转义会炸），写本地文件 `scp` 过去跑；
6. **不要用脆弱的"重新缩进"式补丁改 Python**（我今天把一个文件改坏两次），
   要么整文件重写，要么用 AST 工具。

---

## 6. 推荐的第一天动作

1. 只读复核：跑一遍基线单测，确认 3007/118；
2. 推索引分支（`feature/tool-capability-index` 已提交，commit `0fa2252`）；
3. **人工校对索引的 12 个分组**，尤其"其他/诊断"组，产出一份最终的
   "旧路径 → 新路径"映射表，**先给 owner 过目再动手**；
4. 跟 owner 确认 §3.1 的策略选择（渐进转发层 vs 一次性硬改 + 协调分支）；
5. 从**影响面最小的一组**开始做阶段 1（建议 `m6y`/`m6z` → `tools/review/`，
   它们只有 13+9 个工具、引用面小），完整走一遍"改 → 回归 → 冒烟 → 提交"，
   把流程验证过再铺开。

---

## 7. 相关文档索引

| 文档 | 为什么要读 |
|---|---|
| `docs/TOOL_INDEX.md`（未推，在 `feature/tool-capability-index`） | 210 个工具的能力分组，改名的起点 |
| `docs/roadmap/APARTMENT_ROUTE_BANK_20260825.md` | 最新的模块结构范例（`route_sampling` 三处共用），可作为"逻辑命名应该长什么样"的样板 |
| `docs/roadmap/CORRECTION_DIRECTIVE_20260825.md` | 当前优先级总表，别和它冲突 |
| `docs/roadmap/DATA_DIVERSIFICATION_WORKORDER_20260823.md` | Codex 那条线在做什么（32 条分支的来源） |
| `AGENTS.md`（仓库根） | 项目铁律 |
| `docs/quickstart.md` | 环境与协作约定 |

---

## 8. 交接时的仓库状态

- `main` = `9f8f0ee`（折线路径库刚合入，全量单测 3007 passed）；
- 未推远端：`feature/tool-capability-index` @ `0fa2252`（索引，owner 说先不推）；
- 已推但**不应合 main**：`spatial-omni-avengine-av`（Spatial-Omni 的代码，借仓托管）；
- Codex 的 32 条 `codex/*` 分支全部未合，**动手前必须与 owner 确认如何处理**。
