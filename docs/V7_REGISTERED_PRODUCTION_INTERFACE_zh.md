# v7 登记资产生成器交接（第10/11回合）

本版本只验收了登记站姿人物、录制讲话/笑声和三角色自然TTS的有限场景。完整 v7-S 的全部声源机制、移动对象、总体画像原生布局和全题族门禁尚未完成；禁止把画像请求表降级为旧两人模板。生产状态与计数以本轮 REPORT_zh.md 和实际导出的 summary.json 为准。CCI 仅同步冻结代码及冒烟测试；未将本轮复核加入共享 P19 队列。

## 输入与启动

入口 `tools/timeline/generate_v7_registered.py` 接受 `--houses`（已验收的物理房屋/房间/划分）、`--seeds`（每行一个登记场景，固定 scene_id 与独立因素种子）、`--runtime-inputs`（原生场景文件、材质、RLR配置、自然台词清单）、`--assets`（人物实例目录）、`--source-admission`（录音与许可/活动读数）、`--human-admission`（原始登记及逐资产许可与SHA）、`--output`、`--offset`、`--limit`、`--workers`、`--render-backend cpu_llvmpipe`。48g 必须另传 `--storage-budget-root`，限定本任务根目录；复核最多8父场景。

登记种子 schema 为 `v7-recorded-registered-standing-v1`。`production_scenario` 当前支持0–8名讲话actor、已准入讲话/笑声、站姿、相机轨迹、实际台词 scenario/sound IDs。台词的完整波形、原始ID、盲ASR和许可链必须同时通过原有准入。自然台词支持逐角色起点 `semantic_role_start_offsets_s`；模拟验收可显式指定 `binding_camera_yaw_offset_deg` 和 `camera_sweep_peak_deg`，这些改变必须触发全新物理渲染，不可沿用旧证据。

第9回合 `profile_seed_tables_v1/v7_S_train.jsonl` 的400条与完整版训练表前400条仍为同一固定请求；开发表100条随开发屋划分。它们不是本版本已经实现的登记场景输入，尚缺原生布局/发声机制编译。入口继续拒绝未经完整实现的画像/配额表，不暗中缩减人数或类别。测试屋不得传入任何复核/量产请求。

声音性别与实际画面人物一致；音色和外观使用独立种子。仅采用本轮冻结的原始 human_roster v1 字节与SHA；共享登记表后来更新至 v2 时不替换既有验收版本。精确SHA路径的只读输入快照不修改原登记字段或资产根目录，仍核对原资产、许可和逐帧mouth。

## CPU后端与物理验收

设 `CUDA_VISIBLE_DEVICES=''`、`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`LP_NUM_THREADS=1`、`LIBGL_ALWAYS_SOFTWARE=1`、`MESA_LOADER_DRIVER_OVERRIDE=llvmpipe`、`EGL_PLATFORM=surfaceless`。CPU入口拒绝NVIDIA/OpenGL硬件渲染器。RLR保持校准的一线程、diffraction=true、阶数10；每个静止口型声源位置只算一条RIR，方向误差>5°的位置拒绝并在候选位置中重采样。实际视频、FOA旋转、v0.2协议和统一QA均使用测得的逐帧相机轨迹。

材质使用既有 d790e8c/486799f 修复；全蒙皮人物碰撞、脚底、口型、完整450帧RGB/depth/semantic恢复、材质高光/色差和RIR方向逐个核对。高光/色差不足100可见像素的行标 `not_run`，不能写成通过。未知遮挡像素保留真实计数；不完整逐帧唯一遮挡ID不提供给严格QA-10。现场演奏及设备对象尚无本版本的完整动作/发声关联见证，相关内容绑定两类不生成。

本轮修复统一QA的固定listener覆盖逐帧相机读数问题：只在生产者的证据入口绑定完整native camera frames，不修改QA判分、谓词或角度死区。原导出保留，旧错误方向证据不得送训练，使用新目录重新出题封存。反事实只核对相同公开绑定问题；纯V/A仍按实际查询谓词比较；不把调换问法的选项顺序当作物理变化。

## 目录、交付、恢复

`parents/<scene_id>/input_recipe.json` 固定全部输入；每阶段生成独立 `<stage>_attempt_<uuid>/`，只有全部required_files存在时才以 `<stage>_completed.json` 发布。恢复必须逐字相同recipe，已经完成的stage只读；输入、材质、问题证据版本改变时使用新输出目录。每个父场景只允许一个写者，AFS调度层必须提供独占分片归属；本地SQLite缓存锁不能代替分布式独占。

封存位于 `alpha_seal_attempt_*/windows/<window_id>/`：10秒/150帧视频、16kHz/160000样本的四通道ACN WYZX SN3D FOA、FOA W派生单声道、14条公开题目。私有答案、原生逐帧证据、实际干声/RIR/材质/许可/反事实见证在 `private/`；答案不混入公开metadata。`alpha_seal_completed.json` 是封存门禁与全部必需文件的指针。

`export_alpha` 仅导出明确选定的已封存父场景，写 `train.jsonl`、`parents.csv`、`windows.csv`、`summary.json`、媒体及private证据；逻辑题数不以多表单或重复记录加倍。`train.jsonl` 格式沿用第8回合，每窗14题和V/A/AV整数轮换由原门禁核对。交付链接若属于原始/别线资料必须复制，任务自有不可变文件可硬链接。

不得删除封存所列required_files。封存后可先列清单，再删未列入required_files的重复PNG/NPZ调试帧，至少保留每视图首/中/末帧；完整视频和450帧JSON必须保留。失败尝试的重型二进制可在确认无活进程引用、保留错误/必要RIR见证后删除；失败JSON可校验SHA后无损gzip；只在本任务目录内按实物SHA硬链接去重材质。48g每次大写入之前查df，>=50十进制GB，任务新增分配块<=10GB；不用du推断XFS reflink实际回收量。

## CCI交接与未完成项

AFS新目录 `/mnt/afs/xinyuanqian/jzy/hardbank/generation/v7_s_round10_20261004_codex_2023/`。`p19_queue.txt` 已备份，`P19_DRY=1` 使用备份做只读预演，`p19_run.sh` 未改；此预演不是新生成计划已入队。正式窗口/题为0，生成用时0；目前没有本任务CCI生成PID。

量产计划保留8训练屋×50=400父场景、2开发屋共100父场景。建议20个25父场景分片（16训练、4开发），一个P19调度器限定全体生成最多32CPU，全部native/RLR/ffmpeg/BLAS线程1。应先记录同一训练作业吞吐基线，按稳定时间窗比较，下降>10%减少本任务并行；这一联动尚未在生产P19验证。4小时重启时只有已经发布的完整stage可以继续复用，未封存attempt不能算交付。开发机只做CLI/文件系统冒烟，不跑500父场景。

开量产前仍需：全部声音类别对应真实对象/发声机制、至少一个已验收移动对象、4与6人层及完整画像覆盖、现场演奏动作与两类缺失内容绑定、严格遮挡对象登记、将500画像请求编译为原生登记场景、AFS绝对路径/原资产SHA可读映射和全局32CPU与训练吞吐联动。AFS的代码/CLI冒烟不证明这些事项完成。


已有单成员世界可用 `v7_native_requalification.requalify_native_single_parent` 复核；只接受已经封存且 train/dev 的完整 standing world，拒绝测试屋和反事实组。必须重新检查视频/FOA、完整 450 帧 native 读回、材质、许可、RIR 和真实逐帧相机，重新出题并封存到新目录；原物理阶段不覆盖，成本注明重用。

场景 recipe 可用 `semantic_sound_subset_ids` 显式选择已准入完整语句场景中至少两个不同角色，`semantic_sound_ids` 指定顺序，`semantic_role_start_offsets_s` 指定起点，`camera_policy=binding_static` 及 `binding_camera_yaw_offset_deg` 控制固定相机。任何源 ID、性别、许可或完整音频 SHA 不满足仍拒绝。运行结束不代表所有题族/覆盖配额已达标；以生产准入收据为准。

第一阶段设备/静态演奏者/动物可见声源接入试产通过显式 `--integration-pilot` 启用，接口、严格检查和断点规则见 [V7_VISIBLE_INTEGRATION_PILOT_zh.md](V7_VISIBLE_INTEGRATION_PILOT_zh.md)。该接口只封存任务自有试产目录，不能视作已具备完整量产或训练表导出资格。
