# 酷家乐原生 UE 光度修复（2026-10-04）

过曝主因是 UE 5.5 USD 光源转换未把父级缩放计入发光尺寸和光通量；四间亮度小批均值为 112.19、112.81、126.81、109.83，44 间完整验收已完成，19 套、33 间物理准入；原 29 间全部保留，量产未启动。

私有证据根目录：`/data/jzy/tmp/claude_hardbank_20260923/v7_kujiale_20261003/ue_round11/`。原始 InteriorAgent、旧地图和共享 Content 均未修改；新地图仅在此前已隔离的 SpearSim 项目内以 `NativeR11_*` 命名。

安装版本的权威转换源码是 `/data/UE_5.5/Engine/Plugins/Runtime/USDCore/Source/USDUtilities/Private/USDLightConversion.cpp`：`ConvertLightIntensityAttr` 只乘一次 `2^exposure`；矩形灯转换为 `π × 面积(m²) × intensity × 2^exposure`，球灯转换为 `4π × 球表面积(m²) × intensity × 2^exposure`。尺寸来自 USD width/height/radius 属性，未计入其祖先 Xform 的缩放。317 个设计灯的 `normalize` 均为 false。归一化语义参见 [OpenUSD LightAPI](https://openusd.org/release/api/class_usd_lux_light_a_p_i.html)，此处没有经验亮度倍数。

0009 的 L0002 原生半径原为约 29.04 m、400000064 lm；修复后的原生回读为半径 29.0702 cm、40084.5898 lm、Source Length=0。L0000 的原生 500×2000 cm、6283185.5 lm 变为 33.7777×40.0000 cm、8489.2734 lm。证据：`evidence/native_source_size_readback.json` 和每套的 `native_lighting_audit.json`。球灯继承非均匀缩放时，UE 没有椭球发光体，适配为同世界表面积的球；剪切矩形保留原发光平面、宽度和面积，近似方式逐灯记入侧车回执。镜像灯具按逆转置运输发光法线，避免翻转正反面。可见灯具网格完全未改。

统一相机仍为 2560×1440、85° 水平视角，交付降至 1280×720。曝光使用 Histogram，EV100 −4..16、直方图范围 −10..20、百分位 10/90、速度上行 3/下行 1 f-stop/s、曝光补偿 0、白平衡 4000K。不逐房间或按单段录音调参。每次换机位实际读取 300 帧 RGB，保留首帧和末 15 帧亮度轨迹；小批末 15 帧的均值波动约 0.01..0.07 灰度单位，已观察到稳定。EyeAdaptation 通过 UE Blueprint setter 打开，保持持久 ViewState，仅作用于 RGB；深度和 ID 通道不改。

新配置名为 `kujiale-native-world-lights-histogram-v2`。`v7_alpha_ue` 的统一绑定函数会选择新配置。旧 histogram 配置继续进入光度审计，缺少世界尺寸回读时明确失败，不退回 EV3 或补光。正式生成需要新地图的光度回读和新房间 QA；本轮仍等待正式 v7 生成器的新资产版本，未生成数据窗口。

自动转换调用 `tools/ue/build_kujiale_authored_light_sidecar.py`，输入已转换的 `engine_expanded/render.usdc`，输出新的灯光侧车和回执。加 `--material-output /own/new/house/source_world_baked.usda` 会另存原 Preview Surface 材质图供 QA 追溯，不导入或修改任何材质。元数据文件 stem 必须与旧 UE 导入命名空间相同。新原生请求同时声明 `base_native_map_path`、`base_native_editor_audit`、`geometry_material_reference_usd`、`authored_light_sidecar_receipt`；导入时复制并重新载入新地图，仅替换该新地图的灯。实际几何仍由原生 LOD0 导出进行比对。

四间首次完整 QA 的材质来源误指灯光侧车，导致“未解析材质”误报。这些失败记录保留，没有改写为通过。改正来源后的 1270 个原生材质字段逐项与旧验收相同；44 间已从新的原生捕获重跑原 QA，最终名单只引用新完成的全量结果。`native_ue_qa.py` 和 `finalize_v7_kujiale_ue.py` 与冻结基线 `f9db982aab3e2a0dd8e124840328d845174af2b6` 字节相同。黑像素、过曝、缺面、穿模、缺图和玻璃判据均未降低。

验证：专项目标测试 8 passed；fast_unit 373 passed、36 skipped、7029 deselected，日志为 `logs/tests_focused_verified.log`、`logs/tests_fast_verified.log`。跳过项依赖此检出未包含的历史事实文件。CPU 世界光度预检覆盖 317 灯；4 间小批有 12 张原生相机图，源尺寸回读用 NullRHI，峰值 0 MiB。小批图形进程树峰值为 3812 MiB（GPU1），用 `nvidia-smi -q -x` 按秒采样，包含图形进程。

存储启动要求 /data ≥55 GB，渲染停止要求 /data ≥50 GB，本轮新增上限 5 GB，图存 JPEG。独立控制器只删自身新建、验收结束后的深度和几何中间件；保留 JPEG、回读、结果、日志和失败见证。过程与进程身份持续写入根目录 `PROGRESS_zh.md`，最终中文报告及离线对照页以本轮运行目录交付为准。


## 全量验收封存（第11轮）

21套44间、同机位129相机视角的全量原生验收已完成：19套、33间物理准入；原29间保留29间。新增kujiale_0022/R1、kujiale_0042/R0、kujiale_0043/R0、kujiale_0066/R0；掉出无。判据、几何与材料均未更改。

同720p JPEG灰度均值123.46→113.55，5–95%分位43.21–186.04→107.81–120.23；STARSS总体106.08、64.76–149.52。RMS对比度均值0.1979、5–95% 0.1329–0.2721。这是静态取景与真实视频帧总体参照，不能声称完全相同。

317个原设计灯原生位置／朝向／尺寸／光通量断言通过；21套新地图的LOD0数组与原地图逐套字节相同。原0.5mm声学射线对齐重新执行，候选不足3个仍保持拒绝。单进程树峰值3812MiB；连续采样段最低空闲及收尾复核66.420GB、新增峰值0.703GB；启动闸门空闲58.040GB。

正式交付目录 `/data/jzy/tmp/claude_hardbank_20260923/v7_kujiale_20261003/ue_round11/delivery_v1`；其中 `REPORT_zh.md`、`room_brightness_before_after.csv`、`view_brightness_before_after.csv`、`train_houses.json`、`BEFORE_AFTER_ue_round11.html` 为完整报告／名单／离线前后图；未发owner。来源 `summary.json`、`../alignment/result.json`、每套guard与QA。59资产／7人物沿用原R7见证并复核75个原MIC SHA，并非本轮另行重渲59资产。

源码提交 `7d23a4dc2f2e62d71429cf0c4afd02705f8b4dc5`。生成输入全部通过新灯光和当前房间QA的接口断言；按用户指令等待共同生成器新资产版本，不进行量产或CCI传输。


## 旧验收页逐张对应与最终复核

原R9统计基线仅125/129相机图与R7离线页逐张相同。0037/R0 camera2、0042/R0 camera1/camera3、0066/R3 camera2已用相同新灯图、相同统一曝光补渲，原准入QA及候选不变。最终HTML的44俯视和129相机图逐张SHA及姿态断言通过；旧页实际显示机位的灰度均值122.52→113.58，5–95% 43.21–186.04→107.81–120.23，新RMS均值0.1991。原R9基线123.46保留在原CSV，严格旧页机位见original_page开头两份CSV。

两种统计口径均显示新亮度分布明显窄于STARSS：不可声称总体完全对齐，未为了拟合参考压缩EV范围。实际129相机中126个通过原像素门槛，3个黑像素失败都在0065/R1，该房间仍拒绝；全部44间的物理最终准入仍为19套33间。

源代码与五个冻结判据文件再次逐字节复核；8个定向与373个fast_unit通过的结果仍适用于未改动源代码。本次只改测量文档，不重复跑相同测试。离线页350张图已逐张解码、44间名单与33个生成接口断言通过。资源采样停止、自己的原生作业全部结束；无需留后台作业。来源delivery_v1/ORIGINAL_PAGE_EXACT_PAIRING.json与FINAL_VERIFICATION.json。
