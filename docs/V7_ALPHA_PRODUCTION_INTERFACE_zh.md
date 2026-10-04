# v7-S alpha 生产和 CCI 交接

可生产范围为 `v7-S-alpha-recorded-static-native-14-v2`：30 秒连续世界，两段 10 秒窗口；四个完成原生接地/碰撞检查的站姿人体，其中两个使用不同 VCTK 讲话身份、两个沉默；合格录制片段整段播放，允许自然停顿、重复发声和重叠。每个父场景以独立 appearance seed 随机分配四种已验收衣色到四个身体，音频 seed 独立；组内只改预定的两位发声者衣色分配或干声关联。完整 v5 的 0–8 人、13/34 类、运动、噪声和严格空间四格配额仍单列未完成。QA-26–28 不进入 alpha；不用 eSpeak 或程序合成动物/电器声。

输入的实际位置是 48g 的 `/data/jzy/tmp/claude_hardbank_20260923/v7_batch1_20261003/round8_alpha_20261004/`：

| 输入 | 文件 | 约束 |
|---|---|---|
| 房屋 | `houses_alpha_registered.json` | 已通过整屋声学；只用自动房间规则和原暴露政策；房屋不跨 split |
| 固定父种子 | `seeds_alpha_500_v1/v7_S_alpha_train.jsonl`、`v7_S_alpha_dev.jsonl` | 400/100；父 seed、分量 seeds、house/split、窗口起点和原单成员/四成员计划不变 |
| 完整版固定前缀 | `v7_full_alpha_train_prefix400.jsonl`、`v7_full_alpha_dev_prefix100.jsonl` | 和同目录 S 表逐字节一致；尚未分配房屋的完整储备表不能直接调度 |
| 原生站姿布局 | `native_static_layouts_10.json` | 10 屋均有 4 个实际全皮肤、双足、碰撞和故意下沉 10cm 阴性控制通过的站位；未授予 walking/sitting 准入 |
| RLR/渲染 | `runtime_inputs_alpha.json` | RLR 1 线程、衍射 true/10、16kHz、4 秒 RIR；CPU llvmpipe；接触 15mm、方向 5°及材质原阈值不变 |
| 人物资产 | `assets_alpha_static.json` | 4 个许可通过的 M2 衣色变体，底模 1 个；来自当前许可通过的 7 项子集；运行时重新校验原许可/源文件绑定 |
| 干声 | `SOURCE_ADMISSION_alpha.json` | 86 条/19 类已准入录制非讲话声加 600 条 VCTK；本 alpha 采样器实际只使用合格男声 VCTK，不能把可用池计作已生成类别 |
| 人物许可 | 运行脚本 `--human-admission` 指向的 families 正式许可收据 | CCI 路径迁移只能改文件位置，保持 ID、许可、源 hash、类别和姿态证据 |

`tools/timeline/project_v7_alpha_seeds.py` 把已有固定 S 种子投影到测得的布局；布局不足直接报错，不能把未通过房屋替换进开发集。生产表保留 100/400 和 25/100 个四成员父场景，其余单成员：训练 1400 窗/19600 题，开发 350 窗/4900 题，合计 1750 窗/24500 题。每窗 14 个逻辑题，MCQ/open 不重复计。模态累计 V/A/AV 30%/25%/45%；四成员窗口每窗至少 5 个真实绑定题，计划绑定 5000/24500=20.408%。这些是计划数量，只有实际封存后才计正式数量。

## 后端和入口

CCI 只同步已冻结代码。本线不登录或启动 CCI 作业。P19 可调用 `avengine.timeline.v7_alpha_generation.generate_alpha_parent(row, house, inputs, assets, source_admission, human_admission, output)`，或下列 CLI；一个父场景只有一个写者，分片的 offset/limit 不得重叠。

```bash
# 以下变量是交接示例，路径迁移和 CCI 环境建立由 cci-datagen 完成。
V7_CODE=/mnt/afs/xinyuanqian/jzy/hardbank/code/AVEngine-v7-alpha
V7_ROOT=/mnt/afs/xinyuanqian/jzy/hardbank/v7_S_alpha_20261004
V7_PY=/path/to/cci/python
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
export LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export EGL_PLATFORM=surfaceless MESA_SHADER_CACHE_DISABLE=true
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export LD_PRELOAD="$V7_ROOT/cpu_egl/libegl_cpu_only.so"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$V7_CODE/src:$V7_CODE"
export TMPDIR="$V7_ROOT/tmp" XDG_CACHE_HOME="$V7_ROOT/tmp/cache"
export MPLCONFIGDIR="$V7_ROOT/tmp/mpl" NUMBA_CACHE_DIR="$V7_ROOT/tmp/numba"
# CPU EGL 选择器是本仓库已有工具；CCI 只做平台编译，不另改源文件。
bash "$V7_CODE/tools/capture/build_cpu_egl.sh" "$V7_ROOT/cpu_egl"
"$V7_PY" "$V7_CODE/tools/timeline/generate_v7_alpha.py" \
  --houses "$V7_ROOT/inputs/houses.json" \
  --seeds "$V7_ROOT/inputs/v7_S_alpha_train.jsonl" \
  --runtime-inputs "$V7_ROOT/inputs/runtime_inputs.json" \
  --assets "$V7_ROOT/inputs/assets.json" \
  --source-admission "$V7_ROOT/inputs/source_admission.json" \
  --human-admission "$V7_ROOT/inputs/human_admission.json" \
  --output "$V7_ROOT/train_generation" \
  --render-backend cpu_llvmpipe --workers 16 --offset 0 --limit 16
```

`--render-backend cpu_llvmpipe` 和上述 Mesa/CPU EGL 环境必须同时提供。原生收据检查 `/proc/self/maps` 中软件光栅器实际已加载，NVIDIA GL 未加载；不能只在配置里写 CPU。48g 代码及测试已用该后端。该 alpha 入口不提供 GPU 回退。

CCI 的 Python、Mesa、Habitat/RLR 原生 ABI、许可/纹理/数据集路径和 AFS 硬链接/持久化需要 cci-datagen 用冻结版预检。路径迁移保留 original ID、原许可文字和内容 hash；`local_source_bindings.read_path`、`retained_text`、`normalized_path`、房屋 geometry/config 路径指向 AFS 中同内容文件。布局里的 48g probe 路径只作为历史 provenance，实际渲染会重新生成自身站姿碰撞证明。不能用相同文件名替代 hash 不同的源资产。

## 恢复和交付

阶段是 `input_recipe` → `dry` → `visual_v0[/v1]` → `audio_<member>` → `windows_<member>` → 整组选题 → `selected_windows_<member>` → `alpha_seal`。完成指针只在全部所需文件存在时独占发布；相同 recipe 可恢复，recipe 改变必须用新输出目录。4 小时重启后复用完成阶段、原生精确姿态 RIR 缓存和封存；未完成 RGB 阶段重跑该 30 秒世界，原失败 attempt 保留。单成员只渲染 v0，四成员渲染两个世界；同一视频跨音频成员硬链接，同一音频跨画面成员逐样本复用，静止位置一条 RIR 跨成员/窗口复用。

```text
train_generation/parents/<scene_id>/
  input_recipe.json
  *_completed.json
  rir_cache/                 # 每父场景自有，不在 AFS 多写者共用 SQLite
  alpha_seal_attempt_<id>/
    delivery_manifest.json
    windows/<window_id>/video.mp4
                        audio_foa.wav       # 160000×4, 16k, ACN W/Y/Z/X SN3D 相机坐标
                        audio_mono.wav      # FOA W 逐样本相同
                        questions.json      # 14 个公开题；无私有证据/答案
    private/<window_id>_answers.json
            <window_id>_evidence.json
            native/                    # RGB/masks、接地、材质、RIR 和相机读回证据
            dry_program.json
            dry_emission_receipt.json
            human_license_admission.json
            counterfactual_audit.json
```

`tools/timeline/export_v7_alpha.py` 只接受选定的完整合格父封存；重复/旧题面版本通过显式 `--seal-parent-map` 排除，绝不“取最新”掩盖失败。导出包含 `train.jsonl`/`dev.jsonl`、相对媒体路径、逐窗原答案/证据和候选拒绝明细、精确 RIR payload、使用的归一化干声、原始 ID、逐来源许可文字。训练目标 `model_answer` 由原 `authority_truth.label` 生成，并经原判分器满分确认；原 `canonical_answer` 和判分器不改。E/S 预测不由本生成器提供，训练若接 E/S，只能接部署预测，不能把 private evidence 当模型输入。

封存并成功导出、重新验证媒体时钟/全部题数与答案/许可/精确 RIR 后，父场景的未选题池、RGB PNG/debug mask、原完整 30 秒视频和完整尾部临时音频可以作为生产临时文件清理；所需证据、原干声/许可、精确 RIR、封存和失败审计须保留，未完成阶段不能边写边删。本轮没有执行清理，也没有删除既有产物。

初步分片为 50 个不重叠的 10 父场景分片（训练 40、开发 10）；按冻结 input/代码版本在 AFS 发布。每屋先完成一个完整原生父场景的 14 题预检，再放行该屋的其余 49 个。可用 64 个 worker；CCI 192 核建议两组互不重叠的 64 worker，合计 128，保留其他核。128 workers 不是 48g 的测试并行度。墙钟预算和完整验收的 CPU 统计见本回合 `REPORT_zh.md`/`production_plan.json`；CCI 的 106 秒是该线的裸 RGB 实测，不能替代完整 masks/碰撞/声学/出题耗时。
