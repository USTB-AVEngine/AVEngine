"""Self-contained flat Chinese review HTML with viewport-only image decoding."""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import json,csv,base64,io,html,subprocess
import numpy as np
import shapely
from shapely.geometry import shape,box,mapping
from PIL import Image,ImageDraw,ImageFont
from tools.rooms.room_selection.media import polygon_mask,project_xz,polygons
from .pipeline import dump,overhead_for,now
from .evaluation import load_regions
from .atlas import floor_overhead


def ref_map(root):
    split=json.loads((root/'house_hash_split_v1.json').read_text());refs={}
    for ref in split['reference_rooms']:
        d=json.loads(Path(ref['path']).read_text());d['_json_path']=ref['path'];d['_dagger']=d['id'] in split['dagger_ids']
        refs.setdefault((d['house'],d['source_room_label']),[]).append(d)
    return refs


def fallback_frame(region):
    floors=region['source_geometry']['floors'];scope=shapely.union_all([shape(f['floor_polygon']) for f in floors])
    if scope.is_empty:scope=box(-1,-1,1,1)
    x0,z0,x1,z1=scope.bounds;cx=(x0+x1)/2;cz=(z0+z1)/2;span=max(x1-x0,z1-z0,2)*1.1
    entry={'view':[[-1,0,0,cx],[0,0,1,-cz],[0,1,0,0],[0,0,0,1]],'orientation':'image_right=-X,image_up=+Z'}
    im={'projection':np.diag([2/span,2/span,1,1]).tolist(),'span_m':span}
    image=Image.new('RGB',(768,768),(35,40,47));return entry,im,image,True


def overlay_image(region,manuals,cache,root,floor):
    row=region['source_geometry'];fid=floor['floor_id']
    overhead=floor_overhead(row,floor,cache,root/'overhead_cpu_atlas_v1',overhead_for(row,cache))
    if overhead:entry,im,image,_=overhead;geometry_only=False
    else:entry,im,image,geometry_only=fallback_frame(dict(region,source_geometry=dict(row,floors=[floor])))
    base=image.convert('RGBA');layer=Image.new('RGBA',image.size,(0,0,0,0))
    colours={'retain':(20,215,115),'discard':(250,70,70),'unresolved':(255,180,35),'unchanged':(55,175,255)}
    for index,b in enumerate(region.get('blocks',[])):
        if b['floor_id']!=fid:continue
        g=shape(b['floor_polygon_xz_m']);c=colours[b['decision']]
        mask=Image.fromarray((polygon_mask(g,b['floor_y_m'],entry,im,image.size)*65).astype('uint8'));paint=Image.new('RGBA',image.size,(*c,0));paint.putalpha(mask);layer.alpha_composite(paint)
        draw=ImageDraw.Draw(layer)
        for poly in polygons(g):draw.line([tuple(p) for p in project_xz(poly.exterior.coords,b['floor_y_m'],entry,im,image.size)],fill=(*c,255),width=3)
        q=g.representative_point();xy=project_xz([(q.x,q.y)],b['floor_y_m'],entry,im,image.size)[0]
        draw.text(tuple(xy),f'S{index}',fill='white',stroke_width=2,stroke_fill='black')
    draw=ImageDraw.Draw(layer)
    for manual in manuals:
        if manual['_dagger'] or abs(manual['floor_y_m']-floor['floor_y_m'])>.3:continue
        rect=box(*manual['bbox_xz_m'][0],*manual['bbox_xz_m'][1]);pts=project_xz(rect.exterior.coords,manual['floor_y_m'],entry,im,image.size)
        # Purple boxes are references; their floor-clipped area owns the IoU.
        for a,b in zip(pts,pts[1:]):
            a=np.asarray(a);b=np.asarray(b);length=np.linalg.norm(b-a)
            for start in np.arange(0,length,10):
                p1=a+(b-a)*start/max(length,1e-9);p2=a+(b-a)*min(start+5,length)/max(length,1e-9);draw.line([tuple(p1),tuple(p2)],fill=(235,90,255,255),width=4)
    for cut in region.get('cut_lines',[]):
        if cut.get('floor_id')!=fid or not cut.get('active_in_final_partition',True):continue
        if cut.get('final_interface_geometry_xz_m'):geometry=shape(cut['final_interface_geometry_xz_m'])
        elif cut.get('line_geometry_xz_m'):geometry=shape(cut['line_geometry_xz_m'])
        elif cut.get('line_xz_m'):geometry=shapely.LineString(cut['line_xz_m'])
        else:continue
        components=list(geometry.geoms) if hasattr(geometry,'geoms') else [geometry]
        for line in components:
            if not hasattr(line,'coords'):continue
            xy=project_xz(line.coords,floor['floor_y_m'],entry,im,image.size)
            draw.line([tuple(p) for p in xy],fill=(255,255,255,255),width=4)
    result=Image.alpha_composite(base,layer).convert('RGB')
    # The cached Oct-03 camera is opposite to smy's documented display.
    if entry.get('orientation')=='image_right=+X,image_up=-Z':result=result.rotate(180)
    result.thumbnail((768,768));draw=ImageDraw.Draw(result)
    draw.text((8,8),'GEOMETRY ONLY / RGB AND SCAN BLACK UNVERIFIED' if geometry_only else 'CPU SPLIT / PURPLE: SMY / +X LEFT, +Z UP',fill='white',stroke_width=2,stroke_fill='black')
    buffer=io.BytesIO();result.save(buffer,format='JPEG',quality=88)
    return buffer.getvalue(),geometry_only


def lazy_image(data,label):
    encoded=base64.b64encode(data).decode('ascii')
    return '<figure><img class="lazy" width="768" height="768" loading="lazy" decoding="async" data-mime="image/jpeg" data-b64="'+encoded+'" alt="'+html.escape(label,quote=True)+'"><figcaption>'+html.escape(label)+'</figcaption></figure>'


def summarize(root,regions):
    blocks=[b for r in regions for b in r.get('blocks',[])];new=[b for b in blocks if b['new_room']]
    reasons=Counter(x for b in new if b['decision']=='discard' for x in b['discard_reasons']);primary=Counter(b['discard_reasons'][0] for b in new if b['decision']=='discard' and b['discard_reasons'])
    training_path=Path(json.loads((root/'processing_plan_v1.json').read_text())['training_csv']);training_houses=set()
    with training_path.open(newline='') as f:
        for row in csv.DictReader(f):
            if row['family']=='hm3d' and row['training_eligible'].lower()=='true':training_houses.add(row['house'])
    retained=[b for b in new if b['decision']=='retain'];retained_houses={b['house'] for b in retained}
    source_house={r['house'] for r in regions}
    tune=json.loads((root/'tune_selected_scores_v1.json').read_text()) if (root/'tune_selected_scores_v1.json').exists() else None
    hold=json.loads((root/'holdout_scores_v1.json').read_text()) if (root/'holdout_scores_v1.json').exists() else None
    minimum_receipts=[b['visibility_partition_solver'] for b in new if b.get('visibility_partition_solver') and b['visibility_partition_solver'].get('status')=='partitioned']
    return {'created_at_utc':now(),'processed_source_regions':len(regions),'source_houses':len(source_house),'sources_requiring_split':sum(r.get('requires_split',False) for r in regions),'unchanged_source_regions':sum(not r.get('requires_split',False) for r in regions),'blocks_total':len(blocks),'new_partition_blocks':len(new),'retained_new_rooms':len(retained),'discarded_new_blocks':sum(b['decision']=='discard' for b in new),'unresolved_new_blocks':sum(b['decision']=='unresolved' for b in new),'source_status_counts':dict(Counter(r['status'] for r in regions)),'new_block_decision_counts':dict(Counter(b['decision'] for b in new)),'discard_reason_counts_multiple':dict(reasons),'discard_primary_reason_counts_disjoint':dict(primary),'retained_new_room_houses':len(retained_houses),'retained_room_houses_already_in_training':len(retained_houses&training_houses),'processed_houses_already_in_training':len(source_house&training_houses),'training_reference_csv':str(training_path),'training_list_modified':False,'max_partition_area_error_m2':max([r.get('area_partition_error_m2',0) for r in regions]+[0]),'floor_minimum_certificates':sum(len(r.get('floor_minimum_certificates',[])) for r in regions),'floor_minimum_verified_count':sum(c['minimum_piece_count_verified'] for r in regions for c in r.get('floor_minimum_certificates',[])),'visibility_partition_room_records':len(minimum_receipts),'visibility_minimum_verified_room_records':sum(r['minimum_piece_count_verified'] for r in minimum_receipts),'semantic_door_cut_count':sum(c['type']=='door' and c.get('active_in_final_partition',True) for r in regions for c in r.get('cut_lines',[])),'narrow_cut_count':sum(c['type']=='narrow' and c.get('active_in_final_partition',True) for r in regions for c in r.get('cut_lines',[])),'visibility_cut_count':sum(c['type']=='visibility' and c.get('active_in_final_partition',True) for r in regions for c in r.get('cut_lines',[])),'scan_black_unverified_block_count':sum(b['black_fraction'] is None for b in blocks),'tune':tune,'holdout':hold,'code_commit_at_export':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'all_reference_sources_le_50':all(r['source_floor_area_m2']<=50 for r in regions if 'smy_hand_cut_33' in r.get('origins',[])),'reference_parameter_identifiability':'none: all 33 native semantic floor source areas <=50, so frozen preservation makes partition IoU invariant to parameters','cpu_only':True,'acoustics':'not_run_per_task','standards':'new rooms 6–50 m², short side >=2.4 m; original <=50 preserved; floor-height windows <=0.3; both structural cues; visibility partition >= selected coverage within selected distance; door-protected wide merge; frozen placement witness','minimum_partition_global_limit':'only area-lower-bound constructions certify global minimum; other candidate-set partitions explicitly unverified','body_width_limit':'scan-contour proxy; subject to missing floor and texture segmentation','scan_black_limit':'RGB all channels <8 is a scan-quality proxy, may flag dark texture'}


def publish(root,run_name,scope):
    root=Path(root);param=json.loads((root/'selected_parameter_v1.json').read_text())['selected_parameter'];regions=load_regions(root,run_name,param['id']);regions=list(regions.values())
    if scope=='reference':regions=[r for r in regions if 'smy_hand_cut_33' in r['origins']]
    regions=sorted(regions,key=lambda r:(r['house'],int(r['source_region'][1:])))
    out=root/('delivery_'+run_name);out.mkdir(exist_ok=False)
    summary=summarize(root,regions);refs=ref_map(root);plan=json.loads((root/'processing_plan_v1.json').read_text());cache=plan['overhead_cache']
    rows=[];fields=['id','house','source_region','floor_id','floor_y_m','floor_area_m2','short_side_m','nav_walkable_area_m2','black_fraction','visibility_coverage_fraction','room_type','decision','discard_reasons','unresolved_reasons','new_room']
    for r in regions:
        dump(out/'regions'/(r['house']+'__'+r['source_region']+'.json'),r)
        for b in r.get('blocks',[]):
            record=dict(b,cut_lines=[dict(c,type_zh={'door':'门','narrow':'窄口','visibility':'看得见'}[c['type']]) for c in r['cut_lines'] if c['id'] in b['cut_ids']],coordinate_axes=['world_X_m','world_Z_m'],source_region_json=str(out/'regions'/(r['house']+'__'+r['source_region']+'.json')))
            dump(out/('rooms' if b['new_room'] else 'unchanged_diagnostics')/(b['id']+'.json'),record)
            rows.append({k:json.dumps(record.get(k),ensure_ascii=False) if isinstance(record.get(k),(dict,list)) else record.get(k) for k in fields})
    with (out/'rooms_summary.csv').open('x',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    dump(out/'summary.json',summary)
    esc=lambda x:html.escape(str(x))
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HM3D CPU 自动切房审阅</title><style>body{margin:24px;background:#101720;color:#e7edf5;font:16px/1.5 system-ui}article{border-top:2px solid #526172;padding:20px 0;display:grid;grid-template-columns:minmax(340px,770px) 1fr;gap:20px}figure{margin:0 0 16px}img{display:block;width:100%;height:auto;aspect-ratio:1;background:#253343}table{border-collapse:collapse;width:100%}td,th{padding:6px;border:1px solid #536171;text-align:left}.retain{color:#35ed8b}.discard{color:#ff7373}.unresolved{color:#ffc35c}.unchanged{color:#65c5ff}pre{white-space:pre-wrap;overflow-wrap:anywhere}figcaption{font-size:13px;color:#b8c5d5}@media(max-width:1100px){article{grid-template-columns:1fr}}</style><body><h1>HM3D CPU 自动切房审阅</h1>']
    parts.append('<p>原始语义区域的结构地面精确并集；绿色保留、红色丢弃、黄色未决、蓝色原范围不动；紫色虚框为 smy 坐标参考。平铺全部来源，图片只在进入视口时加载，文件无需网络。</p>')
    metrics={k:v for k,v in summary.items() if k not in ('tune','holdout','code_commit_at_export')}
    labels={'processed_source_regions':'处理来源区域','source_houses':'来源房子','sources_requiring_split':'超过 50 m² 的来源','unchanged_source_regions':'按冻结规则不动的来源','blocks_total':'全部地面块（含未切诊断）','new_partition_blocks':'大区域切出的块','retained_new_rooms':'新保留房间（6–50 m²）','discarded_new_blocks':'丢弃新块','unresolved_new_blocks':'未决新块','discard_reason_counts_multiple':'丢弃原因（可重复）','discard_primary_reason_counts_disjoint':'首要丢弃原因（互斥）','retained_new_room_houses':'新保留房间所在房子','retained_room_houses_already_in_training':'其中已在训练名单的房子','processed_houses_already_in_training':'全部处理房子中已在训练名单的数量','semantic_door_cut_count':'语义门切线','narrow_cut_count':'地面窄口切线','visibility_cut_count':'看得见切线','scan_black_unverified_block_count':'黑区比例未验证的块','max_partition_area_error_m2':'最大面积守恒误差 m²'}
    parts.append('<table><tbody>'+''.join('<tr><th>'+esc(labels.get(k,k))+'</th><td>'+esc(v)+'</td></tr>' for k,v in metrics.items())+'</tbody></table>')
    for fold in ('tune','holdout'):
        s=summary[fold]
        if s:parts.append('<p><b>'+('选参数一半' if fold=='tune' else '留出唯一一次')+'</b>：来源宏平均 IoU '+f'{s["mean_iou_source_macro"]:.4f}'+f'；逐参考微平均 {s["mean_iou_manual_micro"]:.4f}；{s["scorable_source_rooms"]} 个有效来源／{s["manual_rooms_scored"]} 间参考；† 排除 {s["dagger_rooms_excluded"]} 间。</p>')
    parts.append('<p><b>口径冲突：</b>33 个手切来源按指定原 region 量法全部 ≤50 m²，冻结规则要求不切；两折 IoU 反映原范围与手工框裁剪的关系，不能解释为大区域切分算法的泛化验证。V20 候选清单的框范围与原 region 语义地面面积单独对账，不据候选父面积扩大来源。</p>')
    for r in regions:
        manual=refs.get((r['house'],r['source_region']),[])
        parts.append('<article><div>')
        floors=r['source_geometry']['floors']
        largest=max(floors,key=lambda f:f['floor_area_m2']) if floors else None
        displayed=[f for f in floors if f['floor_area_m2']>=6 or f is largest]
        if not displayed:displayed=[{'floor_id':'missing','floor_y_m':0,'floor_polygon':mapping(box(-1,-1,1,1)),'floor_area_m2':0}]
        for floor in displayed:
            data,geo=overlay_image(r,manual,cache,root,floor)
            parts.append(lazy_image(data,r['house']+'/'+r['source_region']+'/'+floor['floor_id']+'；楼层 '+f'{floor["floor_y_m"]:.3f}'+' m'+('；无匹配 RGB，仅几何' if geo else '；原俯视图叠加切块和 smy')))
        if len(displayed)<len(floors):parts.append('<p>其余小于 6 m² 的高度层逐块列在表格和 JSON 中；该行显示主层及全部 ≥6 m² 的楼层。</p>')
        for d in manual:
            if not d['_dagger']:
                path=Path(d['_json_path']).parent/(d['id']+'-overhead-rgb.png')
                if path.exists():
                    image=Image.open(path).convert('RGB');image.thumbnail((768,768));buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=85);parts.append(lazy_image(buffer.getvalue(),d['id']+'：smy 留存俯视图（只读副本）'))
                continue
            path=Path(d['_json_path']).parent/Path(d.get('scope_preview',d['id']+'-scope-preview.png')).name
            if path.exists():
                im=Image.open(path).convert('RGB');im.thumbnail((768,768));buf=io.BytesIO();im.save(buf,format='JPEG',quality=85);parts.append(lazy_image(buf.getvalue(),d['id']+' †：仅展示手工划线图；坐标未收窄，不参与 IoU'))
        parts.append('</div><div><h2>'+esc(r['house']+'/'+r['source_region'])+'</h2><p>来源：'+esc(r['origins'])+'；原地面 '+f'{r["source_floor_area_m2"]:.3f}'+' m²；状态 '+esc(r['status'])+'</p>')
        if r.get('unresolved_reasons'):parts.append('<p class="unresolved">'+esc(r['unresolved_reasons'])+'</p>')
        parts.append('<table><tr><th>块</th><th>地面 m²／短边 m</th><th>类型／决定</th><th>可走 m²／黑区／可见</th><th>原因</th></tr>')
        for i,b in enumerate(r.get('blocks',[])):
            reasons=b['discard_reasons']+b['unresolved_reasons']+b['diagnostic_reasons_without_new_rejection'];parts.append('<tr class="'+b['decision']+'"><td>S'+str(i)+' '+esc(b['floor_id'])+'</td><td>'+f'{b["floor_area_m2"]:.3f} / {b["short_side_m"]:.3f}'+'</td><td>'+esc(b['room_type'])+' / '+esc(b['decision'])+'</td><td>'+f'{b["nav_walkable_area_m2"]:.3f} / '+esc(b['black_fraction'])+' / '+esc(b['visibility_coverage_fraction'])+'</td><td>'+esc(reasons or '通过冻结摆放见证；声学未跑' if b['decision']=='retain' else reasons or '原范围保留')+'</td></tr>')
        parts.append('</table><p>切线：'+esc([{k:c.get(k) for k in ('id','type','width_m','semantic_instance_id')} for c in r['cut_lines']])+'</p></div></article>')
    parts.append('<script>const io=new IntersectionObserver(entries=>{for(const e of entries){if(!e.isIntersecting)continue;const im=e.target;im.src="data:"+im.dataset.mime+";base64,"+im.dataset.b64;delete im.dataset.b64;io.unobserve(im)}},{rootMargin:"0px",threshold:0.01});document.querySelectorAll("img.lazy").forEach(im=>io.observe(im));</script></body></html>')
    with (out/'review.html').open('x',encoding='utf8') as f:f.write(''.join(parts))
    report=['结论：已完成本批来源几何与 CPU 切分审阅；IoU 是否达标及全部未决按实际记录，声学未运行。','',json.dumps({k:v for k,v in summary.items() if k not in ('tune','holdout')},ensure_ascii=False,indent=2),'','冻结标准未改。33 个参考来源全部 ≤50 m²，本轮保持不动；因此切分参数无法由这组参考辨识，IoU 不构成对大区自动切分的验证。V20 名单父面积并非本轮原 region 地面面积，具体对账见 scope_area_reconciliation_v1.csv。','']
    for fold in ('tune','holdout'):
        s=summary[fold]
        if s:report.append(f'{fold} 来源宏平均 IoU={s["mean_iou_source_macro"]:.6f}，参考微平均={s["mean_iou_manual_micro"]:.6f}，有效来源={s["scorable_source_rooms"]}，参考={s["manual_rooms_scored"]}，排除†={s["dagger_rooms_excluded"]}。出处：'+str(root/('holdout_scores_v1.json' if fold=='holdout' else 'tune_selected_scores_v1.json')))
    report+=['','未验证：无匹配楼层/取景 RGB 的黑区比例；有限候选相机解未达到面积下界时的全局最少块数；扫描地面孔洞对应墙还是纹理/缺损；户外/走廊几何代理；原生 Habitat 软件渲染失败的场景；所有声学验收。','每间 JSON、来源区域 JSON、CSV、summary.json 与内嵌图片 review.html 同目录。既有名单未改，新保留数量及房子关系见 summary.json。']
    with (out/'REPORT_zh.md').open('x',encoding='utf8') as f:f.write('\n'.join(report)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('tune','holdout')},ensure_ascii=False,indent=2));return out


def validate(root,run_name):
    root=Path(root);out=root/('delivery_'+run_name);summary=json.loads((out/'summary.json').read_text());errors=[];rooms=[]
    for path in (out/'rooms').glob('*.json'):
        b=json.loads(path.read_text());rooms.append(b);g=shape(b['floor_polygon_xz_m'])
        if abs(g.area-b['floor_area_m2'])>1e-7:errors.append('area mismatch '+b['id'])
        if b['decision']=='retain':
            if not (6-1e-8<=g.area<=50+1e-8 and b['short_side_m']>=2.4-1e-8 and b['black_fraction'] is not None and b['black_fraction']<=.15 and b['placement_witness']['found']):errors.append('retained standard failure '+b['id'])
    for path in (out/'regions').glob('*.json'):
        region=json.loads(path.read_text());blocks=region.get('blocks',[])
        for floor in region['source_geometry']['floors']:
            gs=[shape(b['floor_polygon_xz_m']) for b in blocks if b['floor_id']==floor['floor_id']]
            if not gs and region['status']=='unresolved':continue
            union=shapely.union_all(gs);source=shape(floor['floor_polygon'])
            if source.symmetric_difference(union).area>1e-6:errors.append('source partition coverage failure '+str(path))
            if abs(sum(g.area for g in gs)-union.area)>1e-6:errors.append('source partition overlap failure '+str(path))
        if not region.get('requires_split') and any(b['decision']!='unchanged' for b in blocks):errors.append('new rejection of preserved original '+str(path))
        by_id={b['id']:b for b in blocks}
        for block in blocks:
            for other in block['adjacent_rooms']:
                neighbour=by_id.get(other['id'])
                if neighbour is None or block['id'] not in {a['id'] for a in neighbour['adjacent_rooms']}:errors.append('non-reciprocal adjacency '+block['id'])
            solver=block.get('visibility_partition_solver')
            if block['decision']=='retain' and solver and not block['visibility']['meets_visibility']:errors.append('retained visibility construction failure '+block['id'])
    content=(out/'review.html').read_text();from html.parser import HTMLParser
    class Parser(HTMLParser):
        def __init__(self):super().__init__();self.images=[]
        def handle_starttag(self,tag,attrs):
            if tag=='img':self.images.append(dict(attrs))
    parser=Parser();parser.feed(content)
    if any('src' in im or im.get('loading')!='lazy' or 'data-b64' not in im for im in parser.images):errors.append('strict lazy image contract failed')
    if 'rootMargin:"0px"' not in content:errors.append('viewport-only loading margin failed')
    result={'validated_at_utc':now(),'status':'pass' if not errors else 'fail','errors':errors,'new_room_json_count':len(rooms),'retained_count':sum(b['decision']=='retain' for b in rooms),'html_embedded_lazy_image_count':len(parser.images),'html_bytes':(out/'review.html').stat().st_size,'partition_error_max_m2':summary['max_partition_area_error_m2'],'actual_browser_render_test':'not_run; HTML parser verified sources deferred; browser rendering separate','acoustics':'not_run'}
    dump(out/'validation.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
    if errors:raise ValueError('Delivery validation failed')
    return result
