"""Assemble machine-only object admission, crossed counts and a self-contained Chinese evidence page."""
import argparse,base64,collections,datetime,html,json,os,hashlib
from pathlib import Path
from v7_object_prepare import dump

NAMES={'air_conditioner':'空调','alarm_clock':'闹钟','bathtub':'浴缸','blender':'搅拌机','bookshelf_speaker':'书架音箱','cellphone':'手机','desk_telephone':'电话','doorbell_chime_unit':'门铃','fireplace':'壁炉','floor_drain':'地漏/排水件','floorstanding_speaker':'落地音箱','landline_phone':'壁挂电话','microwave_oven':'微波炉','printer':'打印机','sink_with_tap':'洗手池','smart_speaker':'智能音箱','smoke_detector':'烟感','soundbar':'回音壁','television':'电视','toilet':'马桶','wall_clock':'挂钟','mantel_clock':'台钟','carbon_monoxide_alarm':'一氧化碳报警器','security_keypad':'安防键盘','kitchen_timer':'厨房计时器','alarm_beep_device':'其他嘀声设备','radio':'收音机','tablet':'平板'}
COLORS={'black':'黑','white':'白','gray_silver':'灰银','red':'红','blue':'蓝','green':'绿','beige':'米','wood_brown':'木棕','yellow':'黄/黄铜色','unknown':'未定'}
SURFACES={'floor':'地面','table':'台面/桌面','wall':'墙面','ceiling':'天花板'}
PLAY=['bookshelf_speaker','floorstanding_speaker','smart_speaker','soundbar','television','radio','tablet']
CLASSES=['alarm_beep','alarm_clock','blender','clock_tick','fire_alarm','microwave_beep','microwave_hum','music_playback','printer','speech_playback','telephone','telephone_bell_ringing','doorbell']

def allowed(meta):
 k=meta['object_type'];form=meta.get('form_factor','')
 if meta.get('silent_decor'):return []
 if k=='alarm_clock':return ['alarm_clock']+(['clock_tick'] if any(x in form for x in ['analog','retro_rectangular']) else [])
 if k in ['wall_clock','mantel_clock']:return ['clock_tick']
 if k=='doorbell_chime_unit':return ['doorbell']
 if k=='blender':return ['blender']
 if k=='microwave_oven':return ['microwave_beep','microwave_hum']
 if k=='printer':return ['printer']
 if k in PLAY:return ['music_playback','speech_playback'] if k not in ['radio','tablet'] else ['speech_playback','music_playback']
 if k in ['desk_telephone','landline_phone','cellphone']:return ['telephone','telephone_bell_ringing']
 if k=='smoke_detector':return ['fire_alarm','alarm_beep']
 if k in ['carbon_monoxide_alarm','security_keypad','kitchen_timer']:return ['alarm_beep']
 return []

def reason_zh(value):
 if 'no_mounting_plane_found' in value:return '网格没有可可靠测量的安装面'
 if 'support_pose:leaning' in value:return '支撑姿态仍超出 8°，不能当作正常摆放'
 if 'no_native_' in value and '_support_with_complete_views' in value:return '没有找到贴合指定附着面且四视图完整可见的位置'
 if 'no_dominant_body_color' in value:return '主体色族占比没有达到 55%，或不足两个视图一致'
 if 'excess_highlights' in value:return '高光像素比例超过 8%'
 if 'collision_after_moving_6cm' in value:return '离开支撑面 6 cm 后仍撞到场景物件'
 if 'measured_color_outside_declared_kind_palette' in value:return '实测颜色不在该物件预先声明的标签色组中'
 if 'rendered_color_disagrees' in value:return 'MP3D 与 HM3D 测得的主色标签不一致'
 if 'not_run' in value:return '原生验收未执行，不能算通过'
 return value

def embed(path):
 p=Path(path)
 if not p.is_file():return '<div class="missing">缺少原生图：未验证</div>'
 mime='image/png' if p.suffix=='.png' else 'image/jpeg';return '<img loading="lazy" alt="原生场景渲染" src="data:'+mime+';base64,'+base64.b64encode(p.read_bytes()).decode()+'">'

def main():
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--worktree',type=Path,required=True);p.add_argument('--sound-classes',type=Path,required=True);p.add_argument('--inventory-v6',type=Path,required=True);a=p.parse_args();T=a.task_root;metas=[];plan=json.loads((a.worktree/'examples/assets/v7_object_variants_plan_20261012.json').read_text());supp_path=a.worktree/'examples/assets/v7_object_supplemental_plan_20261012.json';plan['requests']+=json.loads(supp_path.read_text())['requests'] if supp_path.exists() else [];topup_path=a.worktree/'examples/assets/v7_object_topup_plan_20261012.json';plan['requests']+=json.loads(topup_path.read_text())['requests'] if topup_path.exists() else []
 for name in ['prepared_existing.json','prepared_new.json']:
  q=T/name
  if q.exists():metas+=json.loads(q.read_text())
 metas=[m for m in metas if m['object_type'] not in {'bathtub','floor_drain','sink_with_tap','toilet','fireplace'}]
 amendment=json.loads((T/'owner_amendment.json').read_text());removed=amendment['owner_removed']
 known={m['asset_id'] for m in metas}
 for request in plan['requests']:
  q=T/'assets'/request['asset_id']/'asset.json'
  if request['asset_id'] not in known and q.exists():
   m=json.loads(q.read_text());m.setdefault('counting_kind',request['counting_kind']);metas.append(m);known.add(m['asset_id'])
 records=[]
 for name in [('native_existing_body_v5_remeasured.json' if (T/'native_existing_body_v5_remeasured.json').exists() else ('native_existing_owner30.json' if (T/'native_existing_owner30.json').exists() else 'native_existing.json')),'native_new.json']:
  q=T/name
  if q.exists():records+=json.loads(q.read_text())
 eligible_color_families={}
 aliases={'matte_black':'black','black_ash':'black','charcoal':'black','white_satin':'white','silver':'gray_silver','gray':'gray_silver','light_gray':'gray_silver','light_gray_fabric':'gray_silver','walnut_veneer':'wood_brown','sandstone':'beige'}
 for item in list(plan['requests'])+metas:
  family=item.get('counting_kind',item.get('object_type'))
  if family in ['landline_phone','cellphone']:family='desk_telephone'
  color=aliases.get(item.get('intended_color'),item.get('intended_color'))
  if color in COLORS and color not in ['unknown']:eligible_color_families.setdefault(family,set()).add(color)
 by={(r['asset_id'],r['house']):r for r in records};rows=[];admitted=[];failures=[];cross=collections.defaultdict(lambda:collections.Counter());kind_ids=collections.defaultdict(list)
 inv=json.loads(a.inventory_v6.read_text());used={r['asset_id']:r.get('used_plan_count') for r in inv['assets']}
 for meta in metas:
  aid=meta['asset_id'];reasons=list(meta.get('preflight_issues',[]));native={h:by.get((aid,h)) for h in ['mp3d','hm3d']};labels=[]
  for h,r in native.items():
   if r is None:reasons.append(h+':not_run')
   elif r.get('status')!='pass':reasons += [h+':'+x for x in r.get('reasons',['native_failed'])]
   elif r.get('color_label'):
    if r.get('derived_glb')!=meta.get('derived_path') or r.get('derived_sha256')!=hashlib.sha256(Path(meta['derived_path']).read_bytes()).hexdigest():
     reasons.append(h+':stale_native_mesh_evidence');continue
    labels.append(r['color_label'])
    if meta['object_type']=='air_conditioner' and r.get('emitter_policy')!='not_applicable_silent_decor':reasons.append(h+':owner_silent_air_conditioner_native_rerun_pending')
    support=r.get('placement',{}).get('support',{})
    if meta['object_type']=='doorbell_chime_unit' and not support.get('native_door_anchor'):reasons.append(h+':owner_door_nearby_native_check_pending')
    if meta['object_type']=='air_conditioner' and meta.get('form_factor')=='wall_split' and support.get('height_above_floor_m',0)<2.2:reasons.append(h+':owner_high_wall_native_check_pending')
    if meta['object_type']=='air_conditioner' and meta.get('form_factor')=='window_unit' and not support.get('native_window_anchor'):reasons.append(h+':owner_window_frame_native_check_pending')
  if len(labels)==2 and labels[0]!=labels[1]:reasons.append('rendered_color_disagrees_between_houses:'+','.join(labels))
  family=meta.get('counting_kind',meta['object_type'])
  if family in ['landline_phone','cellphone']:family='desk_telephone'
  if len(labels)==2 and labels[0]==labels[1] and labels[0] not in eligible_color_families.get(family,set()):reasons.append('measured_color_outside_declared_kind_palette:'+labels[0])
  status='pass' if not reasons and len(labels)==2 else ('fail' if meta.get('preflight_issues') else ('blocked' if any(x.endswith(':not_run') for x in reasons) else 'fail'));label=labels[0] if len(labels)==2 and labels[0]==labels[1] else None;kind=meta.get('counting_kind',meta['object_type'])
  # Existing phone forms share the owner's declared telephone object family.
  if kind in ['landline_phone','cellphone']:kind='desk_telephone'
  form=meta.get('form_factor','unknown')
  if meta['object_type']=='television':
   suffix='central_pedestal' if 'central_pedestal' in aid else ('two_splayed_feet' if 'two_splayed_feet' in aid else '')
   if suffix:form=form+'_'+suffix
  shares=[r.get('main_color_share') for r in native.values() if r and r.get('main_color_share') is not None];measured=float(sum(shares)/len(shares)) if shares else None
  row={'asset_id':aid,'object_type':meta['object_type'],'counting_kind':kind,'form_factor':form,'origin':meta.get('origin','existing'),'status':status,'reasons':reasons,'label_color':label,'intended_color':meta.get('intended_color'),'measured_main_color_share':measured,'silent_decor':meta.get('silent_decor',False),'used_plan_count_v6':used.get(aid),'allowed_v7_classes_candidate':allowed(meta),'asset_metadata':str(T/'assets'/aid/'asset.json'),'derived_glb':meta.get('derived_path'),'dimensions_m':meta.get('geometry',{}).get('dimensions_m'),'attachment_surface':meta.get('attachment_surface'),'license':meta.get('license'),'generation':meta.get('generation'),'geometry_checks':meta.get('geometry'),'size_basis':meta.get('size_basis'),'emitter':meta.get('emitter'),'material_receipt':meta.get('material_receipt'),'finish_repairs':meta.get('finish_repairs',[]),'native_results':native,'production_sampler_status':'not_connected','sound_binding_status':'not_run'}
  intent_map={'matte_black':'black','black_ash':'black','charcoal':'black','white_satin':'white','silver':'gray_silver','gray':'gray_silver','light_gray':'gray_silver','light_gray_fabric':'gray_silver','walnut_veneer':'wood_brown','sandstone':'beige'}
  expected=intent_map.get(meta.get('intended_color'),meta.get('intended_color'))
  row['color_label_decision']='reject' if status!='pass' else ('measured_label' if expected in [None,'unspecified'] or expected==label else 'machine_relabel')
  row['color_label_rationale']='CIELAB on native body ROIs; pooled main share >=55%, at least two agreeing views per house; both house labels equal'
  rows.append(row)
  if status=='pass':
   admitted.append(row)
   if not row['silent_decor']:cross[(kind,label)][form]+=1;kind_ids[kind].append(row)
  else:failures.append({'asset_id':aid,'status':status,'reasons':reasons})
 stages=[];row_by={r['asset_id']:r for r in rows}
 for request in plan['requests']:
  aid=request['asset_id'];D=T/'new'/aid;proofs={}
  for stage,filename in [('flux','flux.json'),('isnet','segmentation.json'),('pixal','pixal.json'),('trellis','trellis.json')]:
   q=D/filename
   if q.exists():proofs[stage]={'receipt':str(q),'record':json.loads(q.read_text())}
  explicit=[stage+':'+str(value['record'].get('reason',value['record'].get('issues',value['record'].get('attempts','generation_failed')))) for stage,value in proofs.items() if value['record'].get('status')=='fail']
  status=row_by[aid]['status'] if aid in row_by else ('fail' if explicit else 'not_run')
  stages.append({'asset_id':aid,'object_kind':request['counting_kind'],'form_factor':request['form_factor'],'intended_color':request['intended_color'],'status':status,'stage_receipts':proofs,'not_run_reason':'native geometry and/or admission missing' if status=='not_run' else None})
  if aid not in row_by and explicit:failures.append({'asset_id':aid,'status':'fail','reasons':explicit,'stage_receipts':[x['receipt'] for x in proofs.values()]})
 dump(T/'generation_request_status.json',stages)
 existing=[r for r in rows if r['origin']=='existing'];new=[r for r in rows if r['origin']=='new'];plan_path=a.worktree/'examples/assets/v7_object_variants_plan_20261012.json';groups=sorted(set(r['counting_kind'] for r in plan['requests']));counts=[]
 for kind in groups:
  kr=kind_ids[kind];forms=sorted({r['form_factor'] for r in kr});colors=sorted({r['label_color'] for r in kr});cells=[{'color':color,'forms':dict(cross[(kind,color)]),'count':sum(cross[(kind,color)].values()),'distinct_forms':len(cross[(kind,color)])} for color in colors];cross_ok=bool(colors) and all(x['distinct_forms']>=2 for x in cells);counts.append({'object_kind':kind,'usable_appearances':len(kr),'distinct_forms':len(forms),'distinct_colors':len(colors),'colors':cells,'crossing_status':'pass' if cross_ok else 'fail','status':'pass' if len(kr)>=6 and len(forms)>=3 and len(colors)>=3 and cross_ok else 'fail'})
 cls_doc=json.loads(a.sound_classes.read_text());cls_counts=collections.Counter(r['sound_class'] for r in cls_doc['records']);cls_counts['doorbell']=amendment['doorbell_recordings_owner_reported'];class_coverage=[{'sound_class':c,'recordings':cls_counts[c],'recordings_source':'owner 追加任务书，音频未核验' if c=='doorbell' else str(a.sound_classes),'usable_object_appearances':sum(c in r['allowed_v7_classes_candidate'] for r in admitted),'kinds':sorted({r['counting_kind'] for r in admitted if c in r['allowed_v7_classes_candidate']})} for c in CLASSES]
 class_ok=all(c['usable_object_appearances']>=6 for c in class_coverage);missing_special={'clock_tick_new_wall_or_mantel_min4':sum(r['origin']=='new' and r['object_type'] in ['wall_clock','mantel_clock'] for r in admitted),'alarm_beep_new_non_smoke_min4':sum(r['origin']=='new' and r['object_type'] in ['carbon_monoxide_alarm','security_keypad','kitchen_timer'] for r in admitted),'telephone_bell_new_legacy_min4':sum(r['origin']=='new' and r['object_type']=='desk_telephone' and any(s in r['form_factor'] for s in ['rotary','bell']) for r in admitted),'speech_playback_new_radio_tablet':sum(r['origin']=='new' and r['object_type'] in ['radio','tablet'] for r in admitted)}
 summary={'schema':'avengine_v7_static_object_admission_report_v1','artifact_root':str(T),'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'A_existing_total':len(existing),'A_pass':sum(r['status']=='pass' for r in existing),'A_fail':sum(r['status']=='fail' for r in existing),'A_blocked':sum(r['status']=='blocked' for r in existing),'A_has_30_conclusions':len(existing)==30 and not any(r['status']=='blocked' for r in existing),'owner_removed_count':len(removed),'owner_removed':removed,'B_status':'pass' if all(c['status']=='pass' for c in counts) and class_ok and min(list(missing_special.values())[:3])>=4 and missing_special['speech_playback_new_radio_tablet']>0 else 'incomplete','new_declared':len(plan['requests']),'new_prepared':sum(not m.get('preflight_issues') for m in metas if m.get('origin')=='new'),'new_geometry_candidates':len(new),'new_not_run':sum(r['status']=='not_run' for r in stages),'new_generation_failures':sum(r['status']=='fail' for r in stages if r['asset_id'] not in row_by),'new_pass':sum(r['status']=='pass' for r in new),'counts':counts,'eligible_color_families':{k:sorted(v) for k,v in eligible_color_families.items()},'tag_palettes':{c['object_kind']:[x['color'] for x in c['colors']] for c in counts},'class_coverage':class_coverage,'special_additions':missing_special,'failure_list':failures,'silent_decor_total':sum(r['silent_decor'] for r in existing),'silent_decor_pass':sum(r['silent_decor'] and r['status']=='pass' for r in existing),'metrics_sources':['prepared_existing.json','native_existing_body_v5_remeasured.json','prepared_new.json','native_new.json','color_calibration_refined.json'],'geometry_size_limits':{'faces_max':100000,'glb_bytes_max':40_000_000},'native_contact_error_max_m':.015,'emitter_readback_error_max_m':.0001,'highlight_ratio_max':.08,'color_main_share_min':.55,'all_decisions':'machine metrics; no human review required','native_review_method':'object-attachment-body-v5','known_limitations':['native doorway proximity verified; whole-building entrance identity unverified','known-shirt calibration covers black/white/red/green/blue; gray/beige/wood families use disclosed engineering boundaries','generated asset redistribution rights unverified; internal research candidates only']}
 dump(T/'summary.json',summary);roster={'schema':'avengine_v7_static_object_roster_v1','artifact_root':str(T),'generated_utc':summary['generated_utc'],'base_worktree_commit':'455875b1d644ca861a8337aaa487cca6b30aeb6a','assets':admitted,'candidates':rows,'generation_requests':stages,'failures':failures,'counts':counts,'class_coverage':class_coverage,'integration':'asset registration candidates only; production sampler and audio bindings unchanged','source_index_readonly':'/data/avengine_external/assets/sound_source_assets_v1/index.json'};name='v7_asset_roster_object_variants_20261012.json';dump(T/'registrations'/name,roster);dump(a.worktree/'registrations'/name,roster)
 md='# A：owner 变更后 30 个现有物件的机器验收\n\n'+f"通过 {summary['A_pass']}，失败 {summary['A_fail']}，未完成/阻塞 {summary['A_blocked']}。只有 MP3D、HM3D 原生图和全部指标齐全的个体才计入通过。\n\n|物件|结论|附着面|测得颜色|原因|\n|---|---|---|---|---|\n"+''.join('|'+r['asset_id']+'|'+r['status']+'|'+str(r['attachment_surface'])+'|'+str(r['label_color'])+'|'+'；'.join(r['reasons'])+'|\n' for r in existing);(T/'A_admission.md').write_text(md)
 bmd='# B：物件种类 × 测得颜色 × 造型\n\n仅统计全部验收通过的个体。出图方案、几何预处理和旧库许可均不能代替原生验收。\n\n|种类|可用外观|造型|颜色|颜色交叉|结论|\n|---|---:|---:|---:|---|---|\n'+''.join('|'+c['object_kind']+'|'+str(c['usable_appearances'])+'|'+str(c['distinct_forms'])+'|'+str(c['distinct_colors'])+'|'+c['crossing_status']+'|'+c['status']+'|\n' for c in counts)+'\n详细颜色 × 造型计数见 summary.json 的 counts[].colors。\n';bmd+='\n|物件种类|测得标签颜色|造型及个体数|\n|---|---|---|\n'+''.join('|'+c['object_kind']+'|'+cell['color']+'|'+', '.join(form+': '+str(n) for form,n in sorted(cell['forms'].items()))+'|\n' for c in counts for cell in c['colors']);(T/'B_counts.md').write_text(bmd)
 count_table=[{'object_kind':c['object_kind'],'label_color':cell['color'],'form_factor':form,'count':n} for c in counts for cell in c['colors'] for form,n in sorted(cell['forms'].items())];dump(T/'B_crossed_counts.json',count_table)
 cells=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v7 物件资产机器验收</title><style>body{font:16px/1.65 system-ui,sans-serif;background:#f2f3f5;color:#222;margin:0}main{max-width:1500px;margin:auto;padding:28px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:18px}.card{background:white;padding:16px;border-radius:12px;overflow-wrap:anywhere}.card img{width:48%;height:auto;margin:1%}.pass{color:#166637}.fail,.blocked{color:#a43322}table{border-collapse:collapse;width:100%;background:white;margin:20px 0}td,th{border:1px solid #ccc;padding:8px;text-align:left}small{color:#555}.missing{padding:16px;background:#eee}code{font-size:12px}</style><main>']
 cells.append('<h1>v7 物件资产：机器验收页</h1><p><b>'+('A 的 30 个现有物件结论已齐，B 达到外观数量要求。' if summary['A_has_30_conclusions'] and summary['B_status']=='pass' else f"本轮还没全部达标。A：{summary['A_pass']}/30 通过，{summary['A_fail']} 失败，{summary['A_blocked']} 未完成；B：{summary['B_status']}。")+'</b></p>')
 cells.append('<p>标准：每个物件要有 MP3D、HM3D 原生渲染和正常支撑姿态；附着面接触误差 ≤15 mm，高光比例 ≤8%，主色族占比 ≥55%，发声点本地/世界坐标跟随误差 ≤0.1 mm，面数 ≤10 万，单个 GLB ≤40 MB。每种物件至少 6 种可用外观、3 种造型、3 种颜色，每个颜色出现在至少 2 种造型上。只有全部指标通过才入“可用”列表。</p><p>颜色阈值先用已知人物上衣校准；160 份已知颜色样本，其中 90 份留出验证全部正确，详情见 color_calibration_refined.json。灰银、米、木棕没有对应已知上衣样本，使用本轮明示的 CIELAB 工程分界，列明这个校准范围限制。<b>所有结论来自机器测量；图片用于展示，不要求你看片或听声判定。</b></p>')
 cells.append('<h2>owner 去掉</h2><p>以下 10 个不参与本次 A 结论或登记；已经生成的旧验收证据留在原地。</p><p>'+html.escape('；'.join(x['asset_id'] for x in removed))+'</p>')
 cells.append('<h2>可用数量</h2><table><tr><th>物件种类</th><th>外观</th><th>造型</th><th>颜色</th><th>交叉</th><th>结论</th></tr>'+''.join('<tr><td>'+NAMES.get(c['object_kind'],c['object_kind'])+'</td><td>'+str(c['usable_appearances'])+'</td><td>'+str(c['distinct_forms'])+'</td><td>'+str(c['distinct_colors'])+'</td><td>'+c['crossing_status']+'</td><td>'+c['status']+'</td></tr>' for c in counts)+'</table>')
 cells.append('<h2>13 个发声类别的可用外观</h2><p>声音类别可共用真实物件；下面的数字不能替代上一张按物件种类统计的数量和交叉检查。门铃的 21 条录音仅来自 owner 任务书，本轮没有核验或配音。</p><table><tr><th>发声类别</th><th>可用外观</th><th>录音数及来源</th><th>物件种类</th></tr>'+''.join('<tr><td>'+c['sound_class']+'</td><td>'+str(c['usable_object_appearances'])+'</td><td>'+str(c['recordings'])+'；'+html.escape(c['recordings_source'])+'</td><td>'+html.escape('、'.join(NAMES.get(k,k) for k in c['kinds']))+'</td></tr>' for c in class_coverage)+'</table>')
 cells.append('<p>门铃挂在原生门洞附近，门洞位置由语义框或原生几何探测确认；整栋建筑的真正入户门身份尚未验证，登记中保留这一限制。</p>')
 cells.append('<h2>颜色 × 造型的计数</h2><table><tr><th>种类</th><th>测得颜色</th><th>造型：通过数量</th></tr>'+''.join('<tr><td>'+NAMES.get(c['object_kind'],c['object_kind'])+'</td><td>'+COLORS.get(cell['color'],cell['color'])+'</td><td>'+html.escape('；'.join(form+'：'+str(n) for form,n in sorted(cell['forms'].items())))+'</td></tr>' for c in counts for cell in c['colors'])+'</table>')
 for kind in sorted({r['object_type'] for r in rows}):
  cells.append('<h2>'+NAMES.get(kind,kind)+'</h2><div class="grid">')
  for row in [r for r in rows if r['object_type']==kind]:
   aid=row['asset_id'];dims=row['dimensions_m'];dim=' × '.join(f'{x*100:.1f}' for x in dims)+' cm（深 × 高 × 宽）' if dims else '未验证';share=f"{row['measured_main_color_share']:.1%}" if row['measured_main_color_share'] is not None else '未测';label=COLORS.get(row['label_color'],'未定');cells.append('<article class="card"><b class="'+row['status']+'">'+row['status']+'</b> '+html.escape(aid)+'<p>造型：'+html.escape(row['form_factor'])+'<br>标签颜色：'+label+'；主色占比：'+share+'<br>尺寸：'+dim+'<br>尺寸依据：旧出图配置的目标高度和本轮明示的常识范围；参考型号见页面末尾。<br>附着面：'+SURFACES.get(row['attachment_surface'],'未验证')+'<br>来源：'+('现有 AVEngine 库的派生副本' if row['origin']=='existing' else '本轮独立出图、抠图和 3D 生成')+'<br>模型许可：FLUX.2 klein Apache-2.0；3D 模型 MIT。资产再分发许可未验证，登记限项目内部科研用途。<br>'+('不发声陈设候选。' if row['silent_decor'] else '发声部位按类别和既有登记定位；本轮未配声音。')+'</p>')
   if row.get('finish_repairs'):cells.append('<p>派生修复：机壳 PBR 颜色和哑光材质已修正，功能面状态按修复记录登记；几何实测一致，已按新 GLB 重新验收。木纹保留原 UV 的亮度变化。</p>')
   if any(r.get('dark_inactive_functional') for r in row.get('finish_repairs',[])):cells.append('<p>屏幕/控制面状态：关闭的暗屏，使用不自发光的哑光 PBR；这一状态改动已单独登记。</p>')
   generation=row.get('generation') or {};pipeline=generation.get('pipeline','flux2_pixal3d_static_v1')
   backend='TRELLIS.2（Pixal3D 三个种子失败后回退）' if 'trellis' in pipeline else 'Pixal3D'
   cells.append('<p>3D 来源：'+backend+'；'+html.escape(pipeline)+'</p>')
   if row.get('emitter'):
    cells.append('<small>发声点：机身或前部真实网格表面；本地坐标 '+html.escape(str([round(v,4) for v in row['emitter']['local_m']]))+' m；到最近网格顶点 '+str(row['emitter'].get('nearest_vertex_distance_m','未测'))+' m。两房中的跟随误差见下方。</small>')
   geometry=row.get('geometry_checks') or {};cells.append('<p>面数与文件：'+str(geometry.get('faces','未测'))+' 面；'+(f"{geometry['glb_bytes']/1e6:.2f} MB" if geometry.get('glb_bytes') is not None else '未测')+'</p>')
   for house,r in row['native_results'].items():
    cells.append('<p><b>'+house.upper()+'</b>：'+(r.get('status','未验证') if r else '未执行/未验证')+'</p>')
    if r:
     shots=r.get('screenshots',[])
     values=[q.get('production_appearance',{}).get('highlight_ratio') for q in shots if q.get('production_appearance',{}).get('highlight_ratio') is not None]
     shares=r.get('main_color_share');pixels=[q.get('visible_pixels',0) for q in shots]
     cells.append('<small>最大高光比例：'+(f'{max(values):.2%}' if values else '未测')+'；最少可见像素：'+(str(min(pixels)) if pixels else '未测')+'；场景主体色占比：'+(f'{shares:.1%}' if shares is not None else '未测')+'</small>')
     for shot in shots:cells.append(embed(shot['rgb']))
     contact=r.get('placement',{}).get('contact_deltas_m');cells.append('<small>接触误差：'+(str([round(x*1000,3) for x in contact])+' mm' if contact else '未测')+'；颜色：'+str(r.get('color_label'))+'；发声点误差：'+('不适用：空调没有发声点' if row['silent_decor'] else str(r.get('emitter_native_readback_error_m','未测')))+' m</small>')
    else:cells.append('<div class="missing">尚无原生图，不计为通过。</div>')
   if row['reasons']:cells.append('<p class="fail">原因：'+html.escape('；'.join(reason_zh(x) for x in row['reasons']))+'</p>')
   cells.append('</article>')
  cells.append('</div>')
 cells.append('<h2>失败与未完成清单</h2><p>未执行的生成请求：'+str(summary['new_not_run'])+' 个。每个请求的出图、抠图、3D 和原生验收状态都在 generation_request_status.json 中；方案数量不计入可用数量。</p><table><tr><th>个体</th><th>状态</th><th>原因</th></tr>'+''.join('<tr><td>'+html.escape(r['asset_id'])+'</td><td>'+r['status']+'</td><td>'+html.escape('；'.join(reason_zh(x) for x in r['reasons']))+'</td></tr>' for r in failures)+'</table>')
 cells.append('<h2>3 个空调陈设</h2><p>空调均不发声、没有发声点，也没有新增外观。'+str(summary['silent_decor_pass'])+'/3 完整通过，逐项结论在上面的空调组内。</p><p>原始资产、AVEngine index、已有名册和动物任务保持只读；派生修复、日志和证据都写进本任务目录。</p></main></html>');refs_path=T/'size_references.json'
 if refs_path.exists():
  refs=json.loads(refs_path.read_text());source_html='<h2>尺寸依据</h2><p>这里只用外形尺寸作常识核对，没有声称复制某个型号。没有逐类制造商依据的物件，仍明确使用旧 profile 的临时尺寸依据与本轮工程范围。</p><ul>'+''.join('<li><a href="'+html.escape(r['url'])+'">'+html.escape(r['title'])+'</a>：'+html.escape(r['example'])+'</li>' for r in refs['references'])+'</ul>'
  cells[-1]=cells[-1].replace('</main></html>',source_html+'</main></html>')
 appearance_path=T/'appearance_references.json'
 if appearance_path.exists():
  references=json.loads(appearance_path.read_text())['references'];details='<h2>真实外观的依据</h2><ul>'+''.join('<li><a href="'+html.escape(r['url'])+'">'+html.escape(r['title'])+'</a>：'+html.escape(r['evidence'])+'</li>' for r in references)+'</ul>';cells[-1]=cells[-1].replace('</main></html>',details+'</main></html>')
 (T/'REVIEW.html').write_text(''.join(cells));print(json.dumps({k:summary[k] for k in ['A_existing_total','A_pass','A_fail','A_blocked','B_status','new_declared','new_pass','silent_decor_total']}))
if __name__=='__main__':main()
