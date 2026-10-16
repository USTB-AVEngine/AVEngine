# v7 物件外观与原生验收

本分支只生成、处理、验收与登记派生资产，不接生产采样器，不绑定声音。只写本 worktree 和 `/data/jzy/tmp/claude_hardbank_20260923/v7_object_variants_20261012`。旧物件、旧名册、模型、公共 Python 环境及其他 worktree 均只读。

owner 去掉浴缸、地漏、洗手池、马桶、壁炉共十个。旧证据保留，当前 A 和新名册没有这十个。A 当前范围是三十个；空调三件均静音且 emitter 为 null，无新增空调。门铃发声，原生墙面位置必须靠近语义或几何确认的门洞。

## 数据流与复现

所有命令在 48g 运行。复杂逻辑都先写任务目录里的 bash 文件再用 bash 执行。后台 PID、命令、GPU 门槛与时间在 JOBS.json；PROGRESS.md 在每块完成后追加。

新资产采用常驻 FLUX.2 klein → CPU ISNet → 常驻 Pixal3D。每对象至多三个种子，连续三个实际失败后才用 TRELLIS.2；没有使用 Hunyuan 模型。ISNet 的 ONNX CPU runtime 使用已有 hunyuan3d 环境，其模型不是 Hunyuan。已发现的 TRELLIS 导出扩展 OOM 有原日志及一次降到 512 的重试记录。

只使用物理 GPU 0、1；每作业启动前要求空闲至少 30720 MiB。全部作业 nice 10，限自己的进程；generation_cutoff 与 hard_end 在 budget.json。模型缓存放任务目录。每次启动查 /data 可用空间至少 150 GiB，任务新增大小不超过 60 GB。

`v7_object_collect.py` 汇总三份 plan 和每个资产的最终 geometry receipt；`v7_object_collect_native.py` 只接受当前 GLB 哈希匹配的 v5 原生证据。兼容入口 merge_batches 已使用同一汇总，避免旧的分区汇总遗漏 TRELLIS、门铃补充或第二次补量。运行中的作业不重复启动，已登记名字不由新 supervisor 覆盖。

生产 PBR 代码 `src/avengine/assets/source_materials.py` 与材质复查目录 code 中对应文件 cmp 相同；生产 code_commit.txt 为 11c4b99819fc9646469d4549f0f3c0f6d48d54de。几何、姿态或材质修复仅写任务目录的派生副本。

## 机器指标

每个资产分别在 MP3D 和 HM3D 原生 Habitat PBR 渲染四视图。用真实场景的地面、台面、墙或天花板，五个接触探针误差不超过 15 mm；埋入 3 cm 的正控制应碰撞，离面 6 cm 的负控制应不碰撞。四图可见像素超过 700 且留边至少 3 px，高光比例不超过 8%。尺寸、实际面数 <=100000、GLB <=40000000 字节和支撑姿态均机器检查。发声点跟随世界变换误差 <=0.1 mm；空调没有发声点。

颜色 ROI 由原生 semantic/depth/camera 矩阵反投影及局部表面法线得到；只在真实功能正面排除屏幕、按键、网罩和钟面，背面与侧面保留。平躺手机的屏幕正面沿本地 +Y。每房主体颜色合并占比至少 55%，至少两视图同色且两房标签一致；对象的意图色不覆盖实测标签。阈值先用已知人物上衣校准，160 样本，90 留出样本；精确结果见 color_calibration_refined.json。灰银、米和木棕没有对应已知上衣样本，工程分界明示，未声称已知色校准覆盖这些族。

`v7_object_delivery.py` 输出 summary、A/B 表、完整交叉表、两份新名册及所有图片内嵌的中文 REVIEW.html。只把全部指标通过的个体写入 assets；失败及未执行请求另列，不拿方案数或几何预检数代替可用数量。`v7_object_validate.py` 从实际 GLB 和无损 RGB/主体 mask 重算尺寸、面数、字节和颜色，再核对当前 lineage、原生指标、两份名册、交叉计数及 HTML 内嵌图片。

## 尚未证明的内容

门铃与原生门洞的距离经过机器检查，但整栋建筑真正入户门身份未验证。窗机必须有原生窗框定位，缺少窗框时不会按普通高墙挂装算通过。尺寸参考产品只用于常识区间，不宣称复刻；未逐类取得制造商规格的物件采用 profile 目标尺寸和工程范围，明确标注依据。

FLUX.2 klein 模型 Apache-2.0，Pixal3D 和 TRELLIS.2 本地模型 LICENSE 为 MIT。生成资产独立再分发权利未验证，登记为内部科研候选。声音录音数量只沿用输入清单与 owner 门铃追加说明，本轮没有核验音频、声音门槛或绑定。

## 验证命令

```bash
bash /data/jzy/tmp/claude_hardbank_20260923/v7_object_variants_20261012/refresh_counts.sh
/data/jzy/envs/room-selection-merged-20261003/bin/python tools/assets/v7_object_validate.py --task-root /data/jzy/tmp/claude_hardbank_20260923/v7_object_variants_20261012 --worktree /data/jzy/code/AVEngine-v7-object-variants-20261012
bash /data/jzy/tmp/claude_hardbank_20260923/v7_object_variants_20261012/run_unit.sh
```

完整单元测试记录在 logs/unit_release_final.log：3217 通过、118 跳过、52 个子测试通过，300.00 秒。跳过项原因由 pytest 原日志列出。只提交本 worktree 的新代码、配置、测试、文档与新名册，不 push。


## 机壳材质修复

原生颜色失败后，仅对自己的派生 GLB 修正机壳 PBR 的真实介电颜色；屏幕、按键和网罩功能面保留，闭合安防键盘的塑料盖算机壳。木纹材质保留原始 UV 亮度/纹理变化，去除错误背壳色和烘焙照明。颜色阈值仍来自上衣校准，没有按物件结果调整阈值。

初次导出全部被拦下，没有启用。诊断发现两种实际问题：浮点坐标排序会误配三角形；重复 wrapper 节点名会让 Trimesh 图丢失父级变换。临时 GLB 副本只把节点名改为唯一值，节点索引和变换不改。实际几何在导出前后按三角形双向匹配，误差不超过 1 微米；还由交付 validator 从真实 GLB 再核对一次。v1 失败副本与诊断留在原地，v2 另建。材质修复的 m/receipt 不能替代新 GLB 的 MP3D、HM3D 全套原生验收。

## 最终机器汇总

A 当前 30 个均有结论：27 通过、3 失败。owner 去掉的 10 个不计入结论或新名册，旧证据保留。200 个新请求中 165 个通过完整机器验收、35 个剔除。名册共 192 个通过个体，其中 190 个发声候选、2 个空调陈设。门铃已有 2 个通过、新增 10 个通过，合计 12 个／5 造型／3 主色族；空调移动和分体通过，窗机失败。

13 个发声类别共覆盖 19 个物件种类，每种至少 6 外观、3 造型、3 主色族，每个实际标签色至少横跨两个造型。完整物件种类 × 标签颜色 × 造型矩阵在 B_counts.md；13 个声音类别共享物件的计数见 summary.json 的 class_coverage。造型名称来自独立生成配置，不宣称机器已做逐类语义造型识别。

最后一个灰银平板仅改自己的派生机壳介电 PBR 反射率，由 finish_v4/receipt.json 核对真实几何不变，再以 native_new_batch12.json 重验两房四视图。没有调整上衣校准阈值、场景灯光或曝光。此前加固平板的一次暗屏状态尝试仍因 MP3D 高光失败而剔除，保留相关记录。

bundle_validation.json 当前 evidence_integrity_status 和 task_requirements_status 均为 pass，独立核验 192 个入册 GLB、1824 张内嵌图。该机器标准的通过仍受上文三项未验证内容限制，尤其门洞邻近通过不等于整栋建筑真正入户门身份已验证。

最后复核确认原银色子弹搅拌机在固定黑／灰边界附近，finish_v4 将自己派生机壳的介电灰银反射率设为 0.25，实际三角形不变，两房全套重验均通过。没有改变已知上衣校准阈值。最终失败是窗机、黑色喷墨打印机和双脚电视；35 个新候选仍剔除，完整清单在 failure_list.json。此补交仅更新名册和文档，源代码与完整单测记录一致。
