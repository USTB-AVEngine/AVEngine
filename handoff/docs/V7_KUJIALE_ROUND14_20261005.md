酷家乐已用33b5b12正式管线封存单屋2窗口/28题，并在48g启动132父场景的受限量产；这里只记录开跑交接，全部132完成前不能宣称最终数据完成。

- 本任务 worktree：/data/jzy/code/AVEngine-codex-v7-kujiale-round14-20261005；产物及PROGRESS在 /data/jzy/tmp/claude_hardbank_20260923/v7_kujiale_20261003/ue_round14（PROGRESS在上一级）。
- 生成冻结33b5b12df1a45ca265bf0c4fbe9b26b664577a7c，规格8910727；R13原生白平衡/灯位/自动曝光沿用766f7ed。QA、真值、配额代码保持33b5b12原样。
- 19套33间train-only，每间4父场景；同34类干声/120登记人物变体输入。预计498窗口/6972题只是计划，已封存数量看PRODUCTION_STATUS.json及batch_evidence。
- GPU2，nice10，affinity80–95，共最多16核心，最多4原生实例；每批按实测峰值乘实例数加4GiB检查，使用nvidia-smi -q -x记录图形进程树。GPU0只有SPEAR已知的6MiB/实例Vulkan初始化，保护上限64MiB，不用GPU1/3。
- /data至少100GB空闲，本回合新增至多30GB；CCI仅接收封存窗口/录音/许可/证据，不传原始InteriorAgent和UE项目。CCI命名空间 /mnt/afs/xinyuanqian/jzy/hardbank/generation/v7_s_kujiale_20261005。
- 控制源：v7_kujiale_production.py是48g协调器；v7_kujiale_transfer_local.py是本机转存源副本，实际部署为本机任务目录round14_transfer_driver.py；v7_kujiale_CCI_index.py的实际副本是round14_cci_aggregate.py，转存器在CCI执行，只写自己的追加索引。
- 先原生成器封存，再原export_alpha导出、原check_critical_delivery(require_full=False)/open_delivered_media读取，再CCI端拉取与全数据SHA核验，回执到达后才删该批48g副本。留下SHA、summary、parents/windows.csv、native guard和输入登记证据。
- 不能把追加132伪装成原500主批次。现family25等待脚本强制400 train+100 dev；同数据行格式的小批读取已通过，但自动追加接入仍由训练侧增入口。CCI总索引完成132之前标partial，完成后封存；原等待脚本、队列和别的检出没有修改。
- 首单屋wall912.524s，GPU2峰4013MiB；54数据文件37272765B，CCI传输/全SHA回读33.682s，两窗口在CCI也完成150帧视频、RGB、FOA/W读回。来源是ue_round14/evidence、transfer_receipts和batch_evidence。完整四实例新版本吞吐看该批guard/result.json与parents.csv，不冒用旧版数据。
- 继续运行PID和停止方法见运行目录REPORT_zh.md；stop_round14.sh只核对自己的PID/create_time/argv，由guard清理自己的子进程。重启前先核对该身份，不自动偷锁或重启别人。

集成提交ef57840（合并33b5b12）；原生扩展导入/UE指标出口修复79f2456；受限量产a7fbc33。本分支无push。生成配额、所有正式门槛、测试分割与原素材均不改变。
