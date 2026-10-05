# v7 连续场景 CPU 试点与扩展运行

本工具是独立的研究试点，不改变已有 Episode 生产器或旧数据。代码入口为
`tools/timeline/pilot_continuous_scenes.py`，算法在
`src/avengine/timeline/continuous_scene.py` 和 `continuous_pilot.py`。
运行需要现有的 Habitat/RLR CPU 适配器、已验收声学包、原始视觉 GLB、navmesh、
eSpeak NG 库、FFmpeg、NumPy/SciPy/Pillow；不安装或修改共享环境。

## 运行与产物

在 AVEngine 工作区设置 `PYTHONPATH=src:.`，同时设置
`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1`。
`TMPDIR` 指向任务自己的产物目录。输入 JSON 含
`simulation`、`runtime`、`source_radius_m`、`listener_radius_m` 和 `houses`。
每屋明确指定 `house`、`split`、声学包、只读原始 GLB/navmesh 和已检查房间的摆放点。
服务器具体参数保存在本轮产物 `pilot_inputs.json`，不写入通用代码默认值。

```bash
python tools/timeline/pilot_continuous_scenes.py run --inputs INPUT.json --output OUTPUT --workers 2
python tools/timeline/pilot_continuous_scenes.py review-windows --inputs INPUT.json --pilot-root OUTPUT --output REVIEW
python tools/timeline/pilot_continuous_scenes.py benchmark --inputs INPUT.json --output BENCH --workers 1 8 32 64 --rirs-per-worker 6
```

本轮 `run` 每屋生成三条 30 秒场景及三条配对的 60 秒对照，各切三窗。
固定相机下有两位几何人物代理：一位多次讲话并有持续段、另一位沿有停止段的轨迹
移动中讲话/停声/再次讲话；另有同 actor 的程序拍手。活动标签在完整场景上以
100 ms 格划段，之后才与窗口求交。窗口起点用 200 ms 格，使 10 Hz 标注、15 fps
视频和 16 kHz 音频同时对齐。三窗和两个长度版本都保留共同父场景，不冒充独立经验。

每条场景的 `completed.json` 指向一个新 attempt，包含真实墙钟、进程 CPU、子进程
CPU、分阶段耗时、RIR 缓存增量和窗口清单。公开题面与私有答案分文件；公共题面没有
真值、正确选项、私有 source/actor ID 或 teacher 答案。v0.2 面板实现 QA-04、QA-25、
QA-21 严格映射、QA-23、STARSS-SPEAKERS、STARSS-FRUSTUM；本轮两人讲话/拍手场景
不产生没有严格映射的 QA-21。其余 v6 题型的正式生产接线未实现。

## 声学正确性与复用

沿用 v6 的采样率、频带、射线、深度、材质、声速和衍射设置，只把 RLR 的工作线程数
设为 1，以免 64 个进程各开 8 线程。静止声源同一位置只有一个 RIR；位置、听者、
材质包、半径、仿真配置和原生运行时身份必须相同才可复用。位置使用精确参数键，
不以粗空间量化替代原生传播。

移动默认在 0.25 秒或 0.5 米上限较先到达时更新，所有运动拐点与停止段边界保留。
相邻 RIR 按发声时刻作线性交叉淡化，权重和为 1；加权干声分别与完整 RIR 卷积再
相加，先保存尾部，再切窗。不会在切窗或停声时重置混响，不压缩声学时钟，也不
分别归一化 FOA 通道。静态复用是同条件的精确复用；移动插值是有测量依据的近似，
不能声称与逐样本连续传播完全相等。

每个试点路径在首个完整往返上以 50 ms 原生 RIR 做独立对照，比较 0.25 秒、单独
0.5 米和选定组合规则；记录每点方向、RT60、DRR 和尾部状况，并保留原生重复采样
的随机混响差异。方向采用 v6 的原生 `W*[X,Y,Z]` 0.5 ms 直达窗和
`[forward;-right;up]` 相机变换，最大 FOA 误差要求 ≤5°。新材质、新路径、近场或
遮挡变化仍需重新检查；本轮结果不是全空间误差保证。

验证 RIR 与生产 RIR 使用不同缓存，避免密采样验证提前把生产缓存填满而低报生成
成本。缓存以 SQLite 主键和逐 RIR 不覆盖文件提交进度；JSON与背景数据先写完整临时兄弟
文件，再发布完成指针；未完成临时文件保留，孤立背景不覆盖而重建。已有文件保留，未完成的
记录可继续计算。每条已完成场景检查计划和必需文件后跳过；中断 attempt 留在原地，
下次使用新的 attempt。相同固定相机的 CPU 背景/深度单独缓存，重启不用重新读网格。

## 图像和监督范围

图像是原始扫描网格的 CPU BVH 视图，纹理采用面质心近似采样；人物为自有球/椭球
几何代理，按背景深度遮挡，停止时不播放步行摆动。背景内部为 320×180，编码为
1280×720/15 fps。它能验证时钟、投影与原生 FOA 方向，不是正式 Habitat RGB 或
已验收的人物外观/衣服/姿态资产。审阅脚本从实际编码窗口读取像素，独立检测人物
头部质心，与原生及插值 FOA 比较；原始干声和单源音轨只作私有诊断。

讲话由现有 eSpeak NG formant 引擎按作者文字合成，指定两个 voice 配置；拍手为
自有带限噪声和衰减模型。未读取真实录音、STARSS 波形/标注音频、目标域 RIR，未跑
GPU/TTS 模型。记录 eSpeak 的版本、数据路径、GPL-3.0-or-later 代码许可和文字来源；
合成声音的可辨认性、拟真性、听觉性别及输出再分发许可没有在本轮作最终认证。

## 48g 与 CCI 的扩大运行计划

48g 的 112 核按逻辑 CPU 预算处理：RLR 每进程 1 线程，工作进程最多 64。
本轮真实测量 1/8/32/64 个 RIR 进程；场景试点按房屋两个进程并行，先把相同场景内
的 RIR、背景和材质编译复用。不能假定 64 进程能给原来四个八线程进程带来 16 倍
声学加速。未来 64 槽队列需按场景/RIR 分片；原始网格背景初始化单独限流，否则
同时加载 64 套重网格可能超过内存。当前 `run` 按屋分片，其实际并行上限还受屋数
限制；64 槽端到端生产队列未接线。

CCI 开发机仅规划，未登录、同步或运行。建议 128 个 CPU 工作进程、每进程 1 线程，
给其他任务留 64 核。规划的所有持久产物位于
`/mnt/afs/xinyuanqian/jzy/hardbank/v7_batch1_append1_20261003/`：
`code/` 保存48g已测提交的只读同步副本，`packages/`、`rir_cache/`、`scenes/`、
`progress/`、`logs/` 随任务分片；CCI 不修改代码或共享环境。

CCI规划的恢复单位为一条 RIR、一个场景和一个窗口，不能只在整批结束时写收据。
目前实现RIR及完整场景恢复；中断场景的媒体attempt重新生成，逐窗口复用尚未接线。
每 3 小时 30 分钟停止接新任务，正在计算的单条 RIR/场景落盘；到重启时未完成的
attempt 保留。由任务自己的外部启动器在机器可达后扫描完成记录并重启缺失任务，
不修改系统服务/cron，也不清理别人的任务。任务记录 PID、命令和 boot ID；只对
已核对的本任务 PID 停止，不做全局杀进程。

AFS 的原子提交、SQLite/锁行为、配额、吞吐以及 CCI 的库/数据可用性均未验证。
部署前要用任务自有目录验证持久化；不能直接把本机 SQLite 并发假设搬到 AFS。
优先每个 RIR/任务分片单写者，不用网络盘上的全局 WAL；如锁语义不成立，改为
独立 RIR 文件与逐任务 JSON 完成记录，重建索引。CCI 性能表按与48g相同单核速度
及 87.5% 重启可用率（每四小时先接单三小时半）估算，尚无 CCI 实测依据。

吞吐数字、分母、场景明细、方向图和验证结果以本轮汇总报告为准。正式人物资产、
原生 RGB、完整 v6 QuestionSpec 接线、素材盲听和全批生产仍是后续工作。
