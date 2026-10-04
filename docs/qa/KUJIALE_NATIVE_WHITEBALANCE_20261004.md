偏蓝根因是相机白平衡4000K与原设计主日光白点不匹配；改用统一主光源规则后，相机画面R/G均值1.101、B/G均值0.893，保留19套33间准入，原33间全部保留。

同机位129张相机JPEG的灰度5–95%由107.81–120.23扩到74.72–153.43；真实总体64.76–149.52。宽度达到真实的92.9%。不声称两个总体完全一致；44间静态设计画面与10,860帧真实视频的布局／时间权重不同。来源summary.json及view_brightness_chroma_before_after.csv。

|指标（均值；括号5–95%）|R11|R13|STARSS非评测总体|
|---|---:|---:|---:|
|灰度0–255|113.580（107.814–120.233）|112.808（74.717–153.426）|106.084（64.762–149.524）|
|RMS对比度|0.199（0.133–0.272）|0.195（0.122–0.281）|0.190（0.104–0.262）|
|R/G|0.893（0.800–0.994）|1.101（1.039–1.210）|1.108（0.879–1.295）|
|B/G|1.189（1.104–1.284）|0.893（0.792–0.958）|0.866（0.633–1.005）|
|Lab a*|0.897（-0.702–3.457）|2.283（0.872–4.984）|2.571（-4.604–6.524）|
|Lab b*|-13.206（-20.034–-6.700）|7.851（2.928–13.201）|8.897（-0.262–21.653）|

统一规则：有效主日光按lux加权的倒色温求白点，无主日光时才用同单位lm的局部灯；不把Sky强度与lux/lm混加。禁用色温时使用D65基准并保留原RGB设计色调。这21套规则结果均6500K，WhiteTint0、TEMP_WhiteBalance；未逐房手调、未改灯色或补灯。317灯色度导入正常，240灯色温启用和值无差异；EyeAdaptation／TemporalAA旗标与R11相同。来源WHITE_BALANCE_ROOT_CAUSE.json。

全局随机曝光：Uniform[−1,+1] EV，一个完整场景以scene_id固定抽一次，所有视角／帧共享。用四间总体响应与真实总体确定范围；更暗−1.25EV在0009/R1黑16.285%超过原15%门槛，保留失败、不使用该范围。EV100−4～16、直方图−10～20、10/90%、速度3/1及300帧预热保持。来源EXPOSURE_RANDOMIZATION_DECISION.json、../pilot和../lower_probe。

48g GPU2原生全量44间完成；单UE进程树峰值3814MiB，每秒nvidia-smi -q -x采样。当前采样段/data最低63.659GB、本轮新增峰值0.442GB，底线50GB／增量5GB。来源disk_curve.csv及../full/*/guard/result.json；最终收尾以后续FINAL_VERIFICATION.json为准。

源码提交766f7ed4b15a4d639c6676b3f807b142c54f05f1；定向29通过，fast_unit 385通过、36跳过。名单train_houses.json、room_results.csv；离线页/data/jzy/tmp/claude_hardbank_20260923/v7_kujiale_20261003/ue_round13/delivery_v1/BEFORE_AFTER_ue_round13.html，未发owner。生成后端NativeUEAlphaVideo已接主光源白平衡和scene_id曝光抽样；正式共同版本窗口尚未重跑，量产0。

完整证据目录：/data/jzy/tmp/claude_hardbank_20260923/v7_kujiale_20261003/ue_round13/delivery_v1。全部44间与原名单对照、精确命令、保留失败试验及未验证事项见该目录REPORT_zh.md；收尾身份见COMMIT.json／FINAL_VERIFICATION.json。
