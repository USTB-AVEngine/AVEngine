"""Create a truthful Round14 production snapshot from own receipts."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,json,time,subprocess,psutil,re

ROOT=Path('/data/jzy/tmp/claude_hardbank_20260923/v7_kujiale_20261003')
RUN=ROOT/'ue_round14'
REPO=Path('/data/jzy/code/AVEngine-codex-v7-kujiale-round14-20261005')

def read(path):return json.loads(Path(path).read_text())
def gbr(n):return f'{n/1e9:.2f}'

def main():
 state=read(RUN/'PRODUCTION_STATUS.json');queue=read(RUN/'transfer_queue.json')
 snap=read(RUN/'resource_sampler/resource_snapshot.json')
 canary=read(RUN/'evidence/canary_result.json')
 input_inventory=read(RUN/'evidence/input_inventory.json')
 rows=[json.loads(s) for s in (RUN/'inputs/kujiale_v7_S_train.jsonl').read_text().splitlines() if s.strip()]
 planned=Counter(r['house'] for r in rows)
 rooms=defaultdict(set)
 for r in rows:rooms[r['house']].add(r['room_id'])
 withdrawn=set(queue.get('withdrawn_batches',[]))
 verified=[b for b in state['batches'] if b['batch_id'] not in withdrawn and b['status']=='CCI_SHA_verified_local_copies_removed']
 actual=defaultdict(lambda:[0,0,0]);families=Counter();speakers=Counter();verified_receipts=[]
 for batch in verified:
  evidence=RUN/'batch_evidence'/batch['batch_id']
  summary=read(evidence/'summary.json');proof=read(evidence/'CCI_verified.json')
  assert proof['batch_id']==batch['batch_id'] and proof['status']=='verified'
  verified_receipts.append(proof)
  families.update(summary['family_counts']);speakers.update(summary['actual_speakers_per_window'])
  with (evidence/'parents.csv').open() as stream:
   for parent in csv.DictReader(stream):
    value=actual[parent['house']];value[0]+=1;value[1]+=int(parent['formal_windows']);value[2]+=int(parent['logical_questions'])
 parents=sum(x[0] for x in actual.values());windows=sum(x[1] for x in actual.values());questions=sum(x[2] for x in actual.values())
 assert parents==state['transferred_parents']
 assert windows==sum(b['formal_windows'] for b in verified)
 assert questions==sum(b['formal_questions'] for b in verified)
 now=time.time();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
 coordinator_receipt_path=max((RUN/'evidence').glob('production_coordinator_v2*process.json'),key=lambda path:path.stat().st_mtime)
 coordinator_receipt=read(coordinator_receipt_path)
 try:
  coordinator_process=psutil.Process(coordinator_receipt['pid'])
  coordinator_live=coordinator_process.create_time()==coordinator_receipt['create_time'] and coordinator_process.status()!=psutil.STATUS_ZOMBIE
 except psutil.NoSuchProcess:coordinator_live=False
 sampler_receipt=read(RUN/'evidence/sampler_v2_process.json')
 watcher_receipt_path=RUN/'evidence/transfer_watcher_v4_process.json'
 watcher_receipt=read(watcher_receipt_path) if watcher_receipt_path.is_file() else None
 completed_guards=[]
 for phase in [RUN/'canary_v3',RUN/'production_batches/batch_001',*(RUN/'production_batches'/key for key,v in state.get('dispatches',{}).items() if v.get('status')=='processed')]:
  f=phase/'guard/result.json'
  if f.is_file():
   v=read(f);completed_guards.append({'path':str(f),'wall_seconds':v.get('elapsed_seconds'),'peak_mib':v.get('peak_mib'),'status':v.get('status'),'stop_reason':v.get('stop_reason')})
 measured=sum(x['wall_seconds'] or 0 for x in completed_guards)
 four_peaks=[read(RUN/'production_batches'/name/'guard/result.json')['peak_mib']
             for name,dispatch in state.get('dispatches',{}).items()
             if dispatch.get('status')=='processed' and dispatch.get('workers')==4
             and (RUN/'production_batches'/name/'guard/result.json').is_file()]
 pending_interface=sorted((set(state.get('preflight_retry_history',{}))|set(state.get('frame_retry_history',{})))
                          -set(state.get('attempted_parent_ids',[])))
 test_path=RUN/'logs/tests_v6.log' if (RUN/'logs/tests_v6.log').is_file() else RUN/'logs/tests_v5.log'
 test_match=re.search(r'(\d+) passed, (\d+) skipped, (\d+) deselected in ([\d.]+)s',test_path.read_text(errors='replace'))
 assert test_match, f'no completed fast_unit receipt in {test_path}'
 report={'observed_epoch':now,'source_commit':head,'active_native_backend_commit':state.get('native_backend_commit'),'frozen_generator':'33b5b12df1a45ca265bf0c4fbe9b26b664577a7c','status':state['status'],'coordinator_current_run_wall_seconds':now-state['started_epoch'],'completed_UE_guard_wall_seconds':measured,'completed_UE_guard_receipts':completed_guards,'formal_sealed_parents':state['formal_parents'],'formal_sealed_windows':state['formal_windows'],'formal_sealed_questions':state['formal_questions'],'CCI_verified_parents':parents,'CCI_verified_windows':windows,'CCI_verified_questions':questions,'rejected_original_requests':len(state.get('rejections',{})),'attempted_original_requests':len(set(state.get('attempted_parent_ids',[]))),'interface_retry_pending':pending_interface,'original_requests':132,'family_counts_CCI_verified':dict(families),'speakers_per_window_CCI_verified':dict(speakers),'free_bytes':snap['shared_data_free_bytes'],'new_allocated_bytes':snap['new_allocated_bytes'],'resource_snapshot_epoch':snap['sample_epoch'],'gpu2_free_mib':snap['gpus'][2]['free_mib'],'four_instance_peak_mib':max(four_peaks,default=None),'fast_unit_log':str(test_path),'CCI_receipts':verified_receipts,'withdrawn_batches':sorted(withdrawn),'dispatch_finished':queue.get('dispatch_finished',False)}
 (RUN/'evidence/report_status_snapshot_v2.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 work_status='继续量产中' if state['status']=='running' and coordinator_live else ('固定请求已处理完' if queue.get('dispatch_finished') else '已停止或待恢复，状态'+state['status'])
 start=(f'酷家乐 v7-S 通过 48g 的 UE/SPEAR {work_status}：截至 {time.strftime("%Y-%m-%d %H:%M UTC",time.gmtime(now))}，CCI 全 SHA 核验 {parents} 个父场景、{windows} 个窗口、{questions} 道题；'
        f'已完成 UE 批次实测机时 {measured/3600:.3f} 小时，固定 132 个请求{"已全部处理" if queue.get("dispatch_finished") else "尚未全部处理"}。')
 lines=[start,'',f'正式生成器冻结提交 `33b5b12df1a45ca265bf0c4fbe9b26b664577a7c`，规格书 `8910727`；自有 UE 分支当前提交 `{head}`，本次生成会话启动时捕获的原生后端提交 `{state.get("native_backend_commit", "未记录")}`。本报告由 `evidence/report_status_snapshot_v2.json`、`PRODUCTION_STATUS.json`、`batch_evidence/*` 和 `resource_sampler/resource_snapshot.json` 实际读回生成。','',
 '### 结果与范围','',f'R13 房间准入为 19 套、33 间，均 train-only。每间固定 4 个请求，共 132 个父场景请求。当前已有终态 {report["attempted_original_requests"]} 个原请求，其中原 QA 拒收 {report["rejected_original_requests"]} 个；接口修复后待同请求重验 {len(pending_interface)} 个。已封存 {state["formal_parents"]} 父、{state["formal_windows"]} 窗口、{state["formal_questions"]} 题；CCI 已全 SHA 核验 {parents} 父、{windows} 窗口、{questions} 题。旧两批撤回数据不计入以上数字。',
 f'初始输入画像预计 {input_inventory["expected_windows"]} 窗口、{input_inventory["expected_questions"]} 题，仅是计划配额；QA 拒收后不能以换种子、补题或放宽门槛补足。原 500 父场景等待脚本只读 400 train + 100 dev；本任务的 train-only 追加批次保留原行和媒体格式，但训练侧还需显式追加入口。CCI `ADDITIONAL_DELIVERY_PROGRESS.json` 在完整性审计之前标记为 `partial_not_training_ready`。','',
 '| 房子 | 准入房间 | 固定请求 | CCI 核验父场景 | 窗口 | 题 |','|---|---:|---:|---:|---:|---:|']
 for house in sorted(planned):
  a=actual[house];lines.append(f'| {house} | {len(rooms[house])} | {planned[house]} | {a[0]} | {a[1]} | {a[2]} |')
 lines+=['','固定请求原门禁拒收（只列实际证据，不按计划补题）：','',
         '| 请求 | 房子 | 原因 | 批次或证据 |','|---|---|---|---|']
 for scene_id,item in sorted(state.get('rejections',{}).items()):
  source=next((row for row in rows if row['scene_id']==scene_id),None)
  lines.append(f'| {scene_id} | {source["house"] if source else "未知"} | {item.get("reason", "未记录").replace("|", "/")} | {item.get("batch", "CPU重新验收")} |')
 lines+=['','原请求接口修复待重验（未计作原门禁拒收）：','',
         '| 请求 | 房子 | 接口错误与处理 | 证据 |','|---|---|---|---|']
 for scene_id,item in sorted(state.get('preflight_retry_history',{}).items()):
  source=next((row for row in rows if row['scene_id']==scene_id),None)
  resolution='终态另见原门禁表或交付表' if scene_id not in pending_interface else '同请求导航初始化修复后重验中'
  lines.append(f'| {scene_id} | {source["house"] if source else "未知"} | 旧 UE 构造早于原候选搜索；{resolution} | {item["batch"]} 原见证 |')
 for scene_id,item in sorted(state.get('frame_retry_history',{}).items()):
  source=next((row for row in rows if row['scene_id']==scene_id),None)
  resolution='终态另见原门禁表或交付表' if scene_id not in pending_interface else '原种子与候选不变，帧前预检修复后重验中'
  lines.append(f'| {scene_id} | {source["house"] if source else "未知"} | SPEAR 帧事务被旧导航异常污染；{resolution} | {item["batch"]} 原见证 |')
 lines+=['','### 题族、人数与机时','',
 'CCI 核验题族题数：`'+json.dumps(dict(sorted(families.items())),ensure_ascii=False)+'`。',
 'CCI 核验窗口实际人数：`'+json.dumps(dict(sorted(speakers.items())),ensure_ascii=False)+'`。',
 '计划父场景请求人数：`'+json.dumps(input_inventory['speaker_requested_counts'],ensure_ascii=False)+'`；计划数不能作实际分布。',
 f'已完成 UE 守护批次的墙钟时间合计 {measured:.3f} 秒（{measured/3600:.3f} 小时）；每次批次来源见 `evidence/report_status_snapshot_v2.json` 的 `completed_UE_guard_receipts`。当前续跑协调器自启动已运行 {(now-state["started_epoch"])/3600:.3f} 小时，包含转存等待，不能与 UE 批次墙钟简单相加。修复后 000033 的原 UE 渲染/音频时间 {canary["wall_seconds"]:.3f} 秒，利用同 SHA 媒体重新编题时未重新占用 GPU（`evidence/sealed_media_requalification.json`）。GPU 2 单实例实测峰值 4013 MiB、已完成四实例批次最大合计峰值 {max(four_peaks,default=0):.0f} MiB；各值见相应 `guard/result.json`。','',
 '### 输入、验收与资源','',
 '源端桥接错误使旧封存环境声错绑人物 1；旧 3 父/6 窗/84 题已撤回。新提交把实际 UE 音频组件位姿登记为环境声源，重新编题沿用原 QA、选题、门槛和 AST 核对相同的原封存函数。固定请求 000032、000034、000035、000036、000037 因合法候选不足被拒收；000038/39 两轮旧接口错误保留见证，v3 只把已有 15cm 导航预检移到 SPEAR 开帧之前，仍从原候选列表选点，以原种子走完整 GPU 验收。证据见 `evidence/endpoint_fix_requalification.json`、`evidence/sealed_media_requalification.json`、`evidence/native_preflight_CPU_test.json`、`evidence/frame_autofix_status.json` 与批次见证。000033 新封存 2 窗口/28 题，原 family25 读取函数的小批模式通过（`evidence/consumer_fixed_canary_readback.json`）。','',
 '使用同 34 类准入干声及 VCTK、同 `v7_asset_roster.json` 登记人物，保持 R13 的原生材质、统一白平衡/自动曝光/灯位。UE 原生可见源与音频读回绑定；当前正式静态场景的站立、脚底与胶囊沿原门禁复核，不把输入 roster 的 120 变体写成 120 个实拍通过。未接触测试房。','',
 f'GPU 只在 48g 2 号卡上运行；CPU affinity 80–95，nice 10，最多 4 实例。每次起 UE 前按已测峰值×实例数+4096 MiB 校验；共享 `/data` 最新空闲 {gbr(snap["shared_data_free_bytes"])} GB，本回合新增 {gbr(snap["new_allocated_bytes"])} GB，硬约束分别为至少 100 GB 和至多 30 GB。监控数据为 `resource_sampler/samples.jsonl`、每批 `guard/samples.jsonl`；离线磁盘曲线为 `evidence/disk_curve.svg`，1分钟数据为 `evidence/disk_curve_1min.csv`。CCI全 SHA 回执后仅清理本任务已核验副本。','',
 '### 文件、验证与继续运行','',
 '只修改 48g 自有 worktree `/data/jzy/code/AVEngine-codex-v7-kujiale-round14-20261005`、运行根 `/data/jzy/tmp/claude_hardbank_20260923/v7_kujiale_20261003`、本机自有控制脚本 `/home/neos/jzy/codex_runs/20261003-kujiale` 与 CCI 自有派生交付目录 `/mnt/afs/xinyuanqian/jzy/hardbank/generation/v7_s_kujiale_20261005`。未修改原始资产、其他检出、共享 Content、全局环境或测试集；未 push，未向 owner 发验收页。','',
 f'CPU 回归测试 {test_match.group(1)} passed / {test_match.group(2)} skipped / {test_match.group(3)} deselected，{test_match.group(4)} 秒（`{test_path.relative_to(RUN)}`）；本次重新验收 000033 的原生产检查、封存、原消费侧读取以及 49 文件 CCI SHA 通过。每批的原门禁拒收、媒体读回、RIR/FOA 和源端位姿证据留在自有 `scene_witnesses/`、`batch_evidence/` 与 CCI 封存目录。','',
 f'进程回执：48g 当前 v2 协调器 PID {coordinator_receipt["pid"]}（核对时{"存活" if coordinator_live else "已退出"}），日志 `{coordinator_receipt["log"]}`；48g 资源采样器 PID {sampler_receipt["pid"]}，日志 `logs/resource_sampler_v2.log`；本机转存监控上次启动 PID {watcher_receipt["pid"] if watcher_receipt else "无回执"}，日志 `/home/neos/jzy/codex_runs/20261003-kujiale/round14_transfer_watcher_v4.log`，进程实时状态以同名回执核对为准。PID 和 create_time/start_ticks 的回执分别见 `{coordinator_receipt_path.relative_to(RUN)}`、`evidence/sampler_v2_process.json`、本机 `round14_transfer_v4_process.json`。停止协调器用 `bash {RUN}/scripts/stop_current.sh`，脚本先核对最新 PID/create_time/命令再发送 SIGTERM；待自身守护子进程退出后，触碰 `resource_sampler/STOP`。停止本机转存监控用 `touch /home/neos/jzy/codex_runs/20261003-kujiale/ROUND14_STOP_TRANSFER`，完成当前转存后退出。','',
 f'未完成与限制：固定 132 请求{"已全部处理" if queue.get("dispatch_finished") else "仍在处理"}；原门禁拒收后实际封存配额为 {state["formal_parents"]}/132，不能以补题、换种子或降门槛凑足。CCI 虽有兼容行 ABI，现有 family25 500 父场景入口仍不识别 train-only 追加批次，训练侧接入尚未完成。不能把部分索引当成训练就绪交付。']
 (RUN/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({k:report[k] for k in ['status','completed_UE_guard_wall_seconds','formal_sealed_parents','formal_sealed_windows','formal_sealed_questions','CCI_verified_parents','CCI_verified_windows','CCI_verified_questions','attempted_original_requests','rejected_original_requests','free_bytes','new_allocated_bytes']},ensure_ascii=False))

if __name__=='__main__':main()
