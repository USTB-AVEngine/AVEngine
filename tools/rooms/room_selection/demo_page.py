"""Offline, flat, lazy-image HM3D/MP3D demonstration from real evidence."""
from __future__ import annotations
import html
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import shape, box, Point
from tools.rooms.room_screening.geometry import union_projected_polygons
from .demo_render import dominant_metrics, visual_bbox
from .media import polygons, project_xz, polygon_mask
from .run import write_json, now

COLOURS = {"pass": "#16865b", "fail": "#c3424e", "review": "#d79b19", "excluded": "#707c89"}
STATE_ZH = {"pass": "通过", "fail": "不通过", "review": "待人工审", "excluded": "不是房间", "not_run": "未跑", "completed": "已完成", "measured": "已测量"}
TYPE_ZH = {"bathroom": "卫生间", "bedroom": "卧室", "storage": "储物间", "dining": "餐厅", "entryway": "玄关", "living": "起居室", "garage": "车库", "corridor": "走廊", "library": "书房", "laundry": "洗衣房", "kitchen": "厨房", "meeting": "会议室", "office": "办公室", "outdoor": "室外", "recreation": "娱乐室", "stairs": "楼梯", "gym": "健身房", "scan_junk": "扫描杂项", "unknown": "用途未知"}
REASON_ZH = {"NON_RESIDENTIAL_OR_TRANSIT": "原生用途在冻结排除清单内", "FLOOR_AREA_SMALL": "语义地面面积不足", "SHORT_SIDE_SMALL": "最小旋转矩形短边不足", "NAV_MAIN_AREA_SMALL": "可走主连通区面积不足", "SCAN_BLACK_HIGH": "俯视黑像素比例过高", "PLACEMENT_NO_WITNESS": "有限搜索未找到三点摆放", "BLACK_FRACTION_UNAVAILABLE": "缺少校准俯视图的黑像素测量", "OVERHEAD_UNAVAILABLE": "缺少校准俯视图", "MULTILEVEL_REGION_REQUIRES_REVIEW": "主地面层占比不足 98%", "NO_SEMANTIC_FLOOR_GEOMETRY": "无语义地面几何", "SEMANTIC_COLOUR_CONFLICT": "语义颜色对应有冲突", "UNASSIGNED_SEMANTIC_BUCKET": "未分配语义桶", "SEMANTIC_REGION_MAPPING_MISSING": "语义 region 对应缺失", "TRANSIT_OR_STAIRS": "过道或楼梯", "SCAN_INCOMPLETE": "扫描残缺", "TOO_SMALL": "过小", "MULTIROOM_UNSPLITTABLE": "多房间不可切", "NON_RESIDENTIAL": "非居住空间", "CLUTTER_OCCLUSION": "杂乱遮挡", "FLOOR_OR_BOUNDARY": "楼层或边界不明确", "PLACEMENT_UNAVAILABLE": "摆放尚未证实", "SPLIT_APPROVE": "切分可接受", "SPLIT_CORRECT": "切分需修正", "MEDIA_MISSING": "素材缺失", "OTHER": "其他"}
HUMAN_ZH = {"use": "可用", "skip": "跳过", "unsure": "未定"}

# Source: niessner/Matterport/data_organization.md. Labels are case-sensitive.
# These names affect display only; qualification keeps the frozen room_type.
NATIVE_ZH = {"a": "卫生间", "b": "卧室", "c": "衣帽间", "d": "餐厅", "e": "玄关",
             "f": "家庭起居室", "g": "车库", "h": "走廊", "i": "图书室", "j": "洗衣房",
             "k": "厨房", "l": "客厅", "m": "会议室", "n": "休息室", "o": "办公室",
             "p": "门廊 / 露台", "r": "娱乐室", "s": "楼梯", "t": "厕所", "u": "工具间",
             "v": "影音室", "w": "健身房", "x": "室外", "y": "阳台", "z": "其他房间",
             "B": "酒吧", "C": "教室", "D": "餐饮卡座", "S": "水疗 / 桑拿",
             "Z": "扫描杂项", "-": "未标注"}


def display_type(row):
    native = (row.get("native_region") or {}).get("region_label")
    return NATIVE_ZH.get(native, TYPE_ZH.get(row.get("room_type"), "用途未知"))


def esc(s): return html.escape(str(s), quote=True)

def val(x, unit="", digits=2):
    return "未测" if x is None else f"{x:.{digits}f}{unit}"

def percent(x): return "未测" if x is None else f"{100*x:.1f}%"

def display_state(row, p):
    return "excluded" if row.get("room_type") in p["excluded_types"] else row["stage1"]["status"]

def comparison(row):
    verdict=(row.get("human") or {}).get("verdict"); auto=row["stage1"]["status"]
    if verdict is None:return "unlabelled", "无人工记录"
    if auto=="pass" and verdict=="use":return "agree", "一致通过"
    if auto=="pass" and verdict=="skip":return "fp", "误放 · 自动通过 / 人工跳过"
    if auto=="fail" and verdict=="use":return "fn", "误杀 · 自动不通过 / 人工可用"
    if auto=="review":return "review", "自动待审 / 人工"+HUMAN_ZH[verdict]
    if auto=="fail" and verdict=="skip":return "agree", "一致跳过"
    return "unsure", "人工结论未定"

def human_reason(row):
    human=row.get("human") or {}
    reasons=human.get("reasons", human.get("reason_codes", []))
    if isinstance(reasons,str): reasons=[reasons]
    codes="、".join(REASON_ZH.get(r,r) for r in reasons)
    note=human.get("note") or human.get("comment") or ""
    return "；".join(s for s in [codes,note] if s) or "原审核记录未写具体理由"


def rule_rows(row,p):
    floor=dominant_metrics(row);m=floor["metrics"] if floor else {}
    room_type=row.get("room_type","unknown")
    rules=[]
    def add(label,value,threshold,ok,code):
        rules.append(dict(label=label,value=value,threshold=threshold,state="review" if ok is None else ("pass" if ok else "fail"),code=code))
    add("用途", TYPE_ZH.get(room_type,room_type), "排除楼梯 / 走廊 / 车库 / 储物 / 室外", room_type not in p["excluded_types"], "NON_RESIDENTIAL_OR_TRANSIT")
    for field,label,minimum,code,unit in [
        ("floor_area_m2","语义地面面积",p["floor_area_min_m2"],"FLOOR_AREA_SMALL"," m²"),
        ("short_side_m","短边",p["short_side_min_m"],"SHORT_SIDE_SMALL"," m"),
        ("nav_main_area_m2","可走主区面积",p["bathroom_nav_main_area_min_m2"] if room_type=="bathroom" else p["nav_main_area_min_m2"],"NAV_MAIN_AREA_SMALL"," m²")]:
        x=m.get(field);add(label,val(x,unit),f"≥ {minimum:g}{unit}",None if x is None else x>=minimum,code)
    black=m.get("black_fraction")
    add("黑像素比例",percent(black),f"≤ {100*p['black_fraction_max']:g}% · RGB 三通道均 < {p['black_pixel_channel_lt']}",None if black is None else black<=p["black_fraction_max"],"SCAN_BLACK_HIGH")
    fraction=(row.get("floor_selection") or {}).get("dominant_area_fraction")
    add("主地面层占比",percent(fraction),f"≥ {100*p['dominant_floor_area_fraction_min']:g}% · 每层窗口 ≤ {p['floor_height_separation_m']:g} m",None if fraction is None else fraction>=p["dominant_floor_area_fraction_min"],"MULTILEVEL_REGION_REQUIRES_REVIEW")
    w=m.get("placement")
    if w and w.get("found"):
        value="找到三点 · 间距 "+" / ".join(f"{d:.2f}" for d in w["pairwise_distances_m"])+f" m · 夹角 {w['horizontal_angle_deg']:.1f}°"
    elif w:value=f"未找到 · 相机 {w.get('camera_candidates',0)} / 声源 {w.get('source_candidates',0)} 个候选"
    else:value="未搜索"
    add("摆放可行性",value,f"三条视线畅通；两两 {p['distance_min_m']:g}–{p['distance_max_m']:g} m；夹角 ≤ {p['camera_hfov_deg']:g}°；相机 / 声源净空 ≥ {p['placement_camera_clearance_m']:g} / {p['placement_source_clearance_m']:g} m",None if w is None else bool(w["found"]),"PLACEMENT_NO_WITNESS")
    return rules


def first_block(row,rules):
    codes=row["stage1"].get("reason_codes",[])
    if not codes:return None
    first=codes[0]
    if first=="BLACK_FRACTION_UNAVAILABLE":first="SCAN_BLACK_HIGH"
    return next((r for r in rules if r["code"]==first),None)


def font(size=22,bold=False):
    candidates=["/usr/share/fonts/opentype/noto/NotoSansCJK-"+("Bold" if bold else "Regular")+".ttc","/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf","/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    return next((ImageFont.truetype(p,size) for p in candidates if Path(p).is_file()),ImageFont.load_default())

def rgb(c):return tuple(int(c[i:i+2],16) for i in (1,3,5))

def geom_lines(g):
    if g.geom_type in {"LineString","LinearRing"}:yield g
    elif hasattr(g,"geoms"):
        for child in g.geoms:yield from geom_lines(child)


def draw_geometry(image,geometry,floor_y,entry,record,colour,fill_alpha=24,width=3):
    if geometry.is_empty:return
    paint=Image.new("RGBA",image.size,(*rgb(colour),0))
    mask=Image.fromarray((polygon_mask(geometry,floor_y,entry,record,image.size)*fill_alpha).astype(np.uint8))
    paint.putalpha(mask);image.alpha_composite(paint)
    d=ImageDraw.Draw(image)
    for poly in polygons(geometry.simplify(0.025,preserve_topology=True)):
        for ring in [poly.exterior,*poly.interiors]:
            d.line([tuple(p) for p in project_xz(ring.coords,floor_y,entry,record,image.size)],fill=colour,width=width)


def dashed(draw,points,colour,width=4,dash=10):
    for a,b in zip(points[:-1],points[1:]):
        a=np.asarray(a,float);b=np.asarray(b,float);length=np.linalg.norm(b-a)
        if not length:continue
        for t in np.arange(0,length,2*dash):
            p1=a+(b-a)*(t/length);p2=a+(b-a)*(min(length,t+dash)/length)
            draw.line([tuple(p1),tuple(p2)],fill=colour,width=width)


def camera_marks(canvas,row,floor_y,entry,record):
    pictures=(row.get("demo_visual") or {}).get("interiors",[])
    if not pictures:return
    pic=pictures[0];camera=np.asarray(pic["camera_m"])
    if abs(camera[1]-1.5-floor_y)>0.8:return
    if pic["kind"] != "placement_witness":
        x, y = project_xz([(camera[0], camera[2])], floor_y, entry, record, canvas.size)[0]
        d = ImageDraw.Draw(canvas)
        d.ellipse((x-9, y-9, x+9, y+9), outline="#f3a335", width=4)
        d.text((x+10, y-10), "参考", font=font(16), fill="#f3a335", stroke_width=2, stroke_fill="#18222b")
        return
    w=pic["witness"]
    positions=[camera,np.asarray(w["source_1_m"]),np.asarray(w["source_2_m"])]
    xy=project_xz([(v[0],v[2]) for v in positions],floor_y,entry,record,canvas.size)
    d=ImageDraw.Draw(canvas)
    for i,target in enumerate(xy[1:],1):
        d.line([tuple(xy[0]),tuple(target)],fill=(35,190,220,190),width=2)
        x,y=target;d.ellipse((x-6,y-6,x+6,y+6),fill="white",outline="#00556e",width=2)
        d.text((x+7,y-10),str(i),font=font(17),fill="white",stroke_width=2,stroke_fill="#00556e")
    x,y=xy[0];d.polygon([(x,y-9),(x-8,y+7),(x+8,y+7)],fill="#28c3e0",outline="#00556e")


def split_marks(canvas,floor,floor_y,entry,record):
    parts=[shape(s["metrics"]["floor_polygon"]) for s in floor.get("split_parts",[])]
    cuts=[]
    for i,a in enumerate(parts):
        for b in parts[i+1:]:cuts.append(a.boundary.intersection(b.boundary))
    if cuts:
        from shapely import union_all
        d=ImageDraw.Draw(canvas)
        for line in geom_lines(union_all(cuts)):
            points=project_xz(line.coords,floor_y,entry,record,canvas.size)
            d.line([tuple(p) for p in points],fill="white",width=7)
            d.line([tuple(p) for p in points],fill="#087eaf",width=4)


def maps_for_house(data,out,p):
    levels=data["demo_floor_images"];ys=np.asarray([l["floor_y_m"] for l in levels])
    for li,level in enumerate(levels):
        canvas=Image.open(level["path"]).convert("RGBA");entry=level["entry"];record=level["image_record"]
        labels=[]
        for row in data["rows"]:
            scopes=[]
            for floor in row.get("floors",[]):
                y=floor["metrics"]["floor_y_m"]
                if int(np.argmin(np.abs(ys-y)))==li:
                    scopes.append(shape(floor["metrics"]["floor_polygon"]))
                    split_marks(canvas,floor,level["floor_y_m"],entry,record)
            reference_box = False
            if not scopes and row["room_label"] in level["room_labels"]:
                bounds, origin = visual_bbox(row)
                if bounds:
                    region = row.get("native_region") or {}
                    g = shape(region.get("native_floor_polygon", {"type": "GeometryCollection", "geometries": []}))
                    if g.is_empty:
                        g = box(*np.asarray(bounds).ravel())
                    scopes = [g]
                    reference_box = True
            if not scopes:continue
            g=union_projected_polygons(scopes);state=display_state(row,p)
            if reference_box:
                for polygon in polygons(g):
                    points = project_xz(polygon.exterior.coords, level["floor_y_m"], entry, record, canvas.size)
                    dashed(ImageDraw.Draw(canvas), points, COLOURS[state], width=4)
            else:
                draw_geometry(canvas,g,level["floor_y_m"],entry,record,COLOURS[state])
            point=g.representative_point();xy=project_xz([(point.x,point.y)],level["floor_y_m"],entry,record,canvas.size)[0]
            native=(row.get("native_region") or {}).get("region_label")
            label=row["room_label"]+" "+display_type(row)+(f" [{native}]" if native else "")
            if reference_box:
                label += " · 旧框" if (row.get("legacy_room") or {}).get("bbox_xz_m") else " · 原生参考"
            labels.append((row,label,xy,COLOURS[state]))
            camera_marks(canvas,row,level["floor_y_m"],entry,record)
        used=[];d=ImageDraw.Draw(canvas);fnt=font(21,bold=True)
        for row,label,xy,colour in labels:
            b=d.textbbox((0,0),label,font=fnt);w=b[2]+14;h=b[3]-b[1]+12;x,y=xy
            choices=[(x-w/2,y-h/2),(x+8,y-h),(x-w-8,y-h),(x-w/2,y+10),(x-w/2,y-h-15),(x+12,y+15)]
            best=None
            for cx,cy in choices:
                cx=max(5,min(canvas.width-w-5,cx));cy=max(45,min(canvas.height-h-6,cy));rect=(cx,cy,cx+w,cy+h)
                penalty=sum(max(0,min(rect[2],a[2])-max(rect[0],a[0]))*max(0,min(rect[3],a[3])-max(rect[1],a[1])) for a in used)
                if best is None or penalty<best[0]:best=(penalty,rect)
            rect=best[1];used.append(rect)
            d.line([(x,y),(rect[0]+w/2,rect[1]+h/2)],fill=colour,width=2)
            disagreement=comparison(row)[0] in {"fp","fn"} or (comparison(row)[0]=="review" and (row.get("human") or {}).get("verdict") in {"use", "skip"})
            d.rounded_rectangle(rect,radius=5,fill=(255,255,255,232),outline="#a32d76" if disagreement else colour,width=3)
            d.text((rect[0]+7,rect[1]+3-b[1]),label,font=fnt,fill=colour)
            if disagreement:d.text((rect[2]+3,rect[1]-5),"!",font=font(27,True),fill="#a32d76",stroke_width=1,stroke_fill="white")
        d.rounded_rectangle((12,10,canvas.width-12,42),radius=6,fill=(255,255,255,228))
        d.text((22,10),f"高度切面 {level['level_id']+1} · 地面 y={level['floor_y_m']:.2f} m · 青色三角=摆放相机，1/2=声源 · 橙圈=参考相机 · 蓝线=切分 · 虚线=登记参照，未测地面",font=font(17),fill="#213343")
        path=out/"demo/assets"/data["house"]/(f"level_{level['level_id']}_annotated.jpg")
        canvas.convert("RGB").save(path,quality=93)
        level["annotated_path"]=str(path);level["drawn_labels"]=[r[0]["room_label"] for r in labels]


def relative(path,bundle):
    return Path(path).resolve().relative_to(bundle.resolve()).as_posix()

def img(path,bundle,alt,cls="",width=960,height=540):
    # PNGs remain the exact calibration evidence. This independently saved
    # JPEG is only a lighter display preview; never used for any rule value.
    source = Path(path)
    if source.suffix.lower() == ".png":
        preview = source.with_name(source.stem + ".preview.jpg")
        if not preview.is_file() or preview.stat().st_mtime_ns < source.stat().st_mtime_ns:
            with Image.open(source) as original:
                picture = original.convert("RGB")
                picture.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                picture.save(preview, quality=86, optimize=True)
        path = preview
    return f'<img class="{esc(cls)}" src="{esc(relative(path,bundle))}" alt="{esc(alt)}" loading="lazy" decoding="async" width="{width}" height="{height}">'


def region_card(row,p,bundle,family):
    state=display_state(row,p);rules=rule_rows(row,p);block=first_block(row,rules)
    cmp,cmp_text=comparison(row);human=row.get("human") or {}
    chips=f'<span class="pill {state}">{STATE_ZH[state]}</span>'
    if state == "excluded":
        chips += f'<span class="pill">原协议：{STATE_ZH[row["stage1"]["status"]]}</span>'
    if family=="hm3d":chips+=f'<span class="pill human-{human.get("verdict","none")}">smy：{HUMAN_ZH.get(human.get("verdict"),"无记录")}</span>'
    native=(row.get("native_region") or {}).get("region_label")
    title=f'{row["room_label"]} · {display_type(row)}'
    subtitle=f'MP3D 原生类别 {native} · 冻结规则用途：{TYPE_ZH.get(row.get("room_type"), "用途未知")}' if native else ('未分配语义桶' if row['region_id']<0 else 'HM3D 语义类别推断')
    if family=="hm3d" and cmp in {"fp","fn","review","unsure"}:
        if cmp=="fp":explanation="已测规则通过；人工原因："+human_reason(row)
        elif cmp=="fn":explanation="自动首条失败："+(block["label"]+" "+block["value"]+"，阈值 "+block["threshold"] if block else "、".join(REASON_ZH.get(c,c) for c in row["stage1"].get("reason_codes",[])))+"。人工原因："+human_reason(row)
        else:explanation="自动原因："+"、".join(REASON_ZH.get(c,c) for c in row["stage1"].get("reason_codes",[]))+"。人工原因："+human_reason(row)
        note=f'<div class="difference {cmp}"><strong>{esc(cmp_text)}</strong><p>{esc(explanation)}</p></div>'
    elif family=="hm3d":note=f'<p class="comparison">{esc(cmp_text)}'+(f' · {esc(human_reason(row))}' if human else '')+'</p>'
    else:note='<p class="comparison">MP3D 无人工结论；自动通过仅表示预筛候选。</p>'
    table=''
    for rule in rules:
        symbol={"pass":"✓","fail":"✗","review":"?"}[rule["state"]]
        first=block is rule
        table+=f'<tr class="{"first-block" if first else ""}"><td class="mark {rule["state"]}">{symbol}</td><th>{esc(rule["label"])}'+('<small>第一条阻断</small>' if first else '')+f'</th><td>{esc(rule["value"])}<small>{esc(rule["threshold"])}</small></td></tr>'
    floor=dominant_metrics(row);extra=''
    if row['stage1'].get('reason_codes') and block is None:
        first = row['stage1']['reason_codes'][0]
        extra += '<p class="first-evidence"><strong>首个阻断 / 待审原因：</strong>' + esc(REASON_ZH.get(first, first)) + '。</p>'
    if floor and (row.get("floor_selection") or {}).get("selected_floor_id") is None:
        extra=f'<p class="scope-note">规则值展示面积最大的 {esc(floor["floor_id"])}；主层未达到门槛，协议仍保留全部 {len(row.get("floors",[]))} 个地面高度窗口待审。</p>'
    visual=row.get("demo_visual") or {};media=''
    quality=visual.get("quality_images",[])
    if quality:
        chosen=next((q for q in quality if q.get("floor_id")==((row.get("floor_selection") or {}).get("selected_floor_id"))),quality[0])
        caption='原校准俯视图预览 · 原测量保持原样' if family=='hm3d' else '原 GLB 校准俯视图预览 · 黑像素按原 PNG 统计'
        media+='<figure class="quality">'+img(chosen["path"],bundle,title+"俯视图",width=1536,height=1536)+f'<figcaption>{caption}</figcaption></figure>'
    interiors=visual.get("interiors",[])
    if interiors:
        for pic in interiors:
            caption='摆放见证机位 · 水平视野 85°' if pic["kind"]=="placement_witness" else '参考视点 · 85° · 未证明三点摆放'
            if pic["kind"] == "semantic_floor_reference_no_navmesh":
                caption = '语义地面参考机位 · 85° · 原导航网格不支持，未证明摆放'
            media+='<figure class="interior">'+img(pic["path"],bundle,title+"内景")+f'<figcaption>{caption}</figcaption></figure>'
        pose=interiors[0]
        camera=' / '.join(f'{x:.2f}' for x in pose['camera_m'])
        extra+=f'<p class="camera-info">相机 X / Y / Z：<code>{camera} m</code>'+(f' · {esc(pose.get("floor_id",""))}' if pose['kind']=='placement_witness' else ' · 参考机位')+'</p>'
    elif row['stage1']['status'] in {'pass','review'}:
        extra+='<p class="missing-view">内景缺口：无已证实摆放机位，也无可核对的参考相机。</p>'
    split=[f for f in row.get('floors',[]) if f.get('stage2',{}).get('status')=='proposed']
    if split:
        extra+='<p class="split-note">切分建议：'+esc('；'.join(f['floor_id']+' → '+str(len(f.get('split_parts',[])))+' 个子块' for f in split))+'。子块已重新测量，均须独立人审；不替代原 region 判定。</p>'
    if not row.get("floors") and (row.get("legacy_room") or {}).get("bbox_xz_m"):
        extra += '<p class="scope-note">缺少可用语义地面；虚线只显示合并产物的旧登记框。参考内景仅依赖这个框与原导航网格，不证明面积或摆放通过。</p>'
    elif not row.get("floors") and (row.get("native_region") or {}).get("bbox_xz_m"):
        extra += '<p class="scope-note">缺少可用语义地面；虚线来自原 .house 的原生地面轮廓或登记框，仅作 region 定位，不用于面积、黑比例或摆放资格判定。</p>'
    codes=row['stage1'].get('reason_codes',[])
    if codes:extra+='<p class="reason">'+esc(' · '.join(REASON_ZH.get(c,c) for c in codes))+'</p>'
    return f'<article class="room-card {state}" id="{esc(row["house"]+"-"+row["room_label"])}"><div class="room-head"><div><h4>{esc(title)}</h4><p>{esc(subtitle)}</p></div><div class="pills">{chips}</div></div>{note}<table class="rules"><tbody>{table}</tbody></table>{extra}<div class="room-media">{media}</div></article>'


def counts(data,p):
    rows=data['rows'];raw=Counter(r['stage1']['status'] for r in rows);display=Counter(display_state(r,p) for r in rows)
    for s in ['pass','fail','review']:raw.setdefault(s,0)
    for s in COLOURS:display.setdefault(s,0)
    human=Counter((r.get('human') or {}).get('verdict','unlabelled') for r in rows)
    cmp=Counter(comparison(r)[0] for r in rows)
    return dict(region_count=len(rows),automatic_raw=dict(raw),display=dict(display),human=dict(human),comparison=dict(cmp),house_gate=data.get('demo_house_gate'),native_render_seconds=data.get('demo_native_render',{}).get('seconds'),measurement_seconds=data.get('seconds'),floor_image_count=len(data.get('demo_floor_images',[])))


def house_section(data,p,bundle,index):
    c=counts(data,p);d=c['display'];family=data['family']
    summary=''.join(f'<span class="counter {s}"><b>{d[s]}</b>{STATE_ZH[s]}</span>' for s in COLOURS)
    gate=c['house_gate'] or {};status=gate.get('status','not_run')
    gate_text='原门禁记录：'+STATE_ZH.get(status,status)
    if family=='mp3d':gate_text='正式房子门禁未跑；登记、既有导航网格读取、真实纹理 RGB 已通过诊断。音频 / RIR 闭环仍缺。'
    key=data['house'].replace('hm3d_','').replace('mp3d_','')
    layer_description = '原产物主地面高度分组，仅用于展示'
    if family == 'mp3d':
        native_count = len({r['native_region']['level_index'] for r in data['rows']})
        layer_description = f'原生层编号共 {native_count} 个；含必要的相机高度补充切面'

    header=f'<div class="house-header"><span class="house-number">{index:02d}</span><div><h3>{esc(key)}</h3><p>{c["region_count"]} 个登记单元 · {c["floor_image_count"]} 个展示高度切面 · '+layer_description+f'</p></div><div class="house-counts">{summary}</div></div><p class="gate">{esc(gate_text)}</p>'
    if family=='hm3d':header+=f'<p class="house-human">smy 已审 {sum(v for k,v in c["human"].items() if k!="unlabelled")} 个 · 自动 / 人工一致 {c["comparison"].get("agree",0)} 个 · 误放 {c["comparison"].get("fp",0)} · 误杀 {c["comparison"].get("fn",0)} · 自动待审 {c["automatic_raw"]["review"]}</p>'
    header+=f'<p class="raw-count">原协议三态（含灰色排除类）：通过 {c["automatic_raw"]["pass"]} / 不通过 {c["automatic_raw"]["fail"]} / 待审 {c["automatic_raw"]["review"]}。</p>'
    plans=''
    for level in data['demo_floor_images']:
        labels=[]
        slice_note = (f"相机高度补充 · {level['anchor_region']} / {level['anchor_floor_id']} · 非新增建筑楼层。 " if level.get('kind') == 'witness_camera_height_supplement' else '')
        for row in data['rows']:
            if row['room_label'] in level.get('drawn_labels',[]):
                labels.append(f'<span style="--chip:{COLOURS[display_state(row,p)]}">{esc(row["room_label"])} {esc(display_type(row))}</span>')
        plans+='<figure class="floor-plan">'+img(level['annotated_path'],bundle,key+'俯视轮廓',width=1536,height=1536)+f'<figcaption>{esc(slice_note)}高度切面 {level["level_id"]+1} · 地面代表高度 {level["floor_y_m"]:.2f} m · 纹理来自原 GLB；近裁面为地面上方 1.60 m。白底紫边 / ! 表示需对照人工结论。</figcaption><div class="map-labels">'+''.join(labels)+'</div></figure>'
    cards=''.join(region_card(r,p,bundle,family) for r in data['rows'])
    return f'<section class="house" id="{esc(data["house"])}">{header}<div class="floor-grid">{plans}</div><div class="room-grid">{cards}</div></section>'


CSS='''
:root{--ink:#172a36;--muted:#5b6d78;--line:#dbe3e6;--paper:#f4f6f5;--blue:#185d83;--pass:#16865b;--fail:#c3424e;--review:#d79b19;--excluded:#707c89}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:var(--paper);font-family:Inter,"Noto Sans CJK SC","Microsoft YaHei",system-ui,sans-serif;line-height:1.6}a{color:var(--blue)}main{max-width:1500px;margin:auto;padding:0 28px 80px}.hero{background:#18323f;color:#fff;padding:48px 0 40px}.hero-inner{max-width:1500px;margin:auto;padding:0 28px}.eyebrow{letter-spacing:.16em;font-size:12px;color:#b3d5dc;font-weight:700}.hero h1{font-size:clamp(30px,4vw,52px);line-height:1.25;margin:12px 0 20px;max-width:1050px;letter-spacing:-.04em}.hero p{max-width:1090px;color:#d5e6e9;font-size:17px}.hero-metrics{display:flex;flex-wrap:wrap;gap:16px;margin-top:26px}.hero-metric{min-width:210px;border-top:1px solid #52727d;padding:12px 20px 0 0}.hero-metric b{display:block;font-size:35px;line-height:1.3;color:#e5f0dc}.hero-metric span{font-size:13px;color:#c5dadd}.source-line{font-size:12px!important;color:#9db8c2!important;margin-bottom:0}.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:26px 0}.flow article{background:white;border:1px solid var(--line);padding:17px 20px;border-radius:12px}.flow b{color:var(--blue);font-size:13px}.flow h3{font-size:16px;margin:3px 0 8px}.flow p{font-size:13px;color:var(--muted);margin:0}.note-box{background:#fff8e7;border:1px solid #e9dcad;border-radius:12px;padding:17px 22px;font-size:14px;margin:20px 0}.legend{display:flex;flex-wrap:wrap;gap:9px 20px;font-size:13px;padding:20px 0}.legend span{display:flex;align-items:center;gap:7px}.dot{width:12px;height:12px;display:inline-block;border-radius:50%}.family-heading{display:flex;align-items:flex-end;justify-content:space-between;gap:25px;margin:56px 0 22px}.family-heading h2{font-size:31px;letter-spacing:-.04em;margin:0}.family-heading p{color:var(--muted);font-size:14px;max-width:700px;margin:8px 0 0}.family-badge{font-size:12px;font-weight:700;background:#d9e8ef;color:#1f5c7c;padding:7px 12px;border-radius:50px;white-space:nowrap}.house{background:white;border:1px solid var(--line);border-radius:20px;margin:0 0 40px;padding:24px;box-shadow:0 3px 20px #172a3605}.house-header{display:flex;align-items:center;gap:16px;flex-wrap:wrap}.house-number{font-size:34px;font-weight:800;color:#98aeb8;line-height:1}.house-header h3{font-size:23px;line-height:1.2;margin:0 0 5px;overflow-wrap:anywhere}.house-header p{font-size:12px;margin:0;color:var(--muted)}.house-counts{display:flex;gap:12px;margin-left:auto}.counter{font-size:11px;display:flex;flex-direction:column;align-items:center;min-width:52px}.counter b{font-size:27px;line-height:1.3}.pass{color:var(--pass)}.fail{color:var(--fail)}.review{color:var(--review)}.excluded{color:var(--excluded)}.gate{background:#edf3f5;border-radius:8px;font-size:13px;padding:10px 14px;margin:17px 0 8px;color:#39505d}.house-human,.raw-count{font-size:12px;color:var(--muted);margin:7px 0}.floor-grid{display:grid;grid-template-columns:minmax(0,1fr);gap:18px;margin:22px 0 30px}.floor-plan{margin:0 auto;width:100%;max-width:1100px;background:#eef1f1;border-radius:12px;overflow:hidden;border:1px solid #dce4e6}.floor-plan>img{display:block;width:100%;height:auto;aspect-ratio:1;object-fit:contain;background:#182026}.floor-plan figcaption{padding:12px 14px;font-size:12px;color:#405660}.map-labels{display:flex;flex-wrap:wrap;gap:4px 12px;padding:0 14px 14px}.map-labels span{font-size:11px;display:flex;align-items:center;gap:4px}.map-labels span:before{content:"";width:7px;height:7px;background:var(--chip);border-radius:50%}.room-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(345px,1fr));gap:16px;align-items:start}.room-card{color:var(--ink);border:1px solid #dce4e6;border-top:4px solid currentColor;border-radius:11px;padding:14px;overflow:hidden;background:#fff}.room-card.pass{border-top-color:var(--pass)}.room-card.fail{border-top-color:var(--fail)}.room-card.review{border-top-color:var(--review)}.room-card.excluded{border-top-color:var(--excluded);background:#fafbfc}.room-head{display:flex;align-items:flex-start;justify-content:space-between;gap:8px}.room-head h4{margin:0;font-size:17px}.room-head p{font-size:10px;color:var(--muted);margin:3px 0}.pills{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:4px;max-width:47%}.pill{border-radius:5px;padding:2px 6px;font-size:10px;font-weight:700;background:#eef2f3;white-space:nowrap}.pill.pass,.human-use{background:#e1f2e9;color:#157144}.pill.fail{background:#fae7e9;color:#a03142}.pill.review,.human-unsure{background:#fff0d0;color:#986505}.pill.excluded,.human-skip,.human-none{background:#e9edf0;color:#536372}.comparison{font-size:11px;margin:8px 0 10px;color:var(--muted)}.difference{padding:9px 11px;border-radius:6px;background:#fbedf3;color:#813152;margin:8px 0 10px;border-left:3px solid #ab3378}.difference.review,.difference.unsure{background:#fff5df;color:#83600e;border-left-color:#d79b19}.difference strong{font-size:11px}.difference p{font-size:11px;margin:4px 0 0;overflow-wrap:anywhere}.rules{border-collapse:collapse;width:100%;font-size:12px}.rules td,.rules th{padding:7px 4px;border-bottom:1px solid #e9edef;vertical-align:top}.rules th{text-align:left;font-size:12px;font-weight:600;width:27%}.rules td:last-child{width:64%}.rules small{display:block;font-size:11px;font-weight:400;color:var(--muted);line-height:1.5;margin-top:2px}.rules .mark{font-size:16px;font-weight:700;width:22px}.rules .first-block{background:#fce8ea}.rules .first-block th small{color:#a83243;font-weight:700}.scope-note,.camera-info,.split-note,.reason,.missing-view{font-size:10px;line-height:1.6;margin:8px 0;color:var(--muted)}.missing-view{color:#967215;background:#fff6df;padding:7px}.scope-note{background:#fff5df;padding:7px}.split-note{color:#236686;background:#eaf4fa;padding:7px}.reason{color:#925064}.camera-info code{font-family:monospace;font-size:11px}.first-evidence{font-size:12px;background:#fff0d0;border-left:3px solid #d79b19;padding:8px 10px;border-radius:5px}.room-media{display:grid;grid-template-columns:1fr;gap:7px;margin-top:12px}.room-media figure{margin:0;min-width:0}.room-media img{display:block;width:100%;height:auto;border-radius:6px;background:#1c272c;aspect-ratio:16/9;object-fit:cover}.room-media .quality img{object-fit:contain}.room-media .interior:last-child:nth-child(odd){grid-column:1/-1}.room-media figcaption{font-size:9px;line-height:1.45;margin:4px 0 0;color:var(--muted)}.comparison-table{width:100%;border-collapse:collapse;font-size:13px;background:white}.comparison-table th,.comparison-table td{text-align:left;border-bottom:1px solid var(--line);padding:10px 12px}.table-scroll{overflow-x:auto;border:1px solid var(--line);border-radius:10px;margin-top:18px}.issue-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px;margin-top:25px}.issue{background:white;border:1px solid var(--line);border-radius:12px;padding:17px}.issue h3{font-size:17px;margin:0 0 9px}.issue p{font-size:13px;margin:9px 0;color:var(--muted)}.issue img{width:100%;height:auto;display:block;border-radius:8px;background:#15242a;max-height:450px;object-fit:contain}.issue figcaption{font-size:11px;color:var(--muted);margin:6px 0}.issue figure{margin:10px 0}.suggestions{background:#e7f0f2;border-radius:13px;padding:20px 25px;margin:26px 0}.suggestions h3{font-size:18px;margin:0 0 8px}.suggestions p{font-size:14px;margin:8px 0;color:#3c5866}.footer{font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:20px;margin-top:36px;overflow-wrap:anywhere}.footer a{margin-right:16px}@media(max-width:850px){.flow{grid-template-columns:repeat(2,1fr)}.floor-grid{grid-template-columns:1fr}.house-counts{margin-left:0}.house{padding:16px}.family-heading{align-items:flex-start}.room-grid{grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}}@media(max-width:520px){main,.hero-inner{padding-left:14px;padding-right:14px}.hero{padding-top:30px}.hero p{font-size:14px}.hero-metric{min-width:135px;padding-right:5px}.hero-metric b{font-size:29px}.flow{gap:8px}.flow article{padding:12px}.family-heading{display:block}.family-badge{display:inline-block;margin-top:12px}.house-counts{width:100%;justify-content:space-around}.room-grid{grid-template-columns:1fr}.house{padding:12px}.room-card{padding:12px}.house-header h3{font-size:19px}.family-heading h2{font-size:26px}.issue-grid{grid-template-columns:1fr}.house-number{font-size:26px}}
'''


def issues_for(hm,mp,p,bundle,out):
    candidates=[];all_mp=[]
    for data in mp:
        for row in data['rows']:
            all_mp.append((data,row))
            f=dominant_metrics(row);native=row.get('native_region')
            if f and native and f['metrics']['floor_area_m2']>0:
                g=shape(f['metrics']['floor_polygon']);b=box(*np.asarray(native['bbox_xz_m']).ravel());outside=float(g.difference(b).area)
                row['demo_bbox_diagnostic']=dict(outside_native_bbox_area_m2=outside,outside_fraction=outside/g.area,scope='dominant semantic floor union; native box is metadata only')
                candidates.append((outside/g.area,outside,data,row))
    issue_html=[];receipts=[]
    reasonable=[c for c in candidates if .03<c[0]<.8]
    leak=max(reasonable or candidates,key=lambda t:t[0],default=None)
    if leak:
        fraction,area,data,row=leak;f=dominant_metrics(row)
        q=next((q for q in row.get('demo_visual',{}).get('quality_images',[]) if q.get('floor_id')==f['floor_id']),None)
        if q:
            camera=json.loads(Path(q['path']).with_suffix('.camera.json').read_text());record=camera['images'][0]
            canvas=Image.open(q['path']).convert('RGBA');geometry=shape(f['metrics']['floor_polygon']);b=box(*np.asarray(row['native_region']['bbox_xz_m']).ravel())
            draw_geometry(canvas,geometry.difference(b),f['metrics']['floor_y_m'],camera,record,'#d2384b',fill_alpha=110,width=4)
            d=ImageDraw.Draw(canvas);points=project_xz(b.exterior.coords,f['metrics']['floor_y_m'],camera,record,canvas.size);dashed(d,points,'#ffe82c',width=6)
            path=bundle/'assets/issues/native_bbox.png';path.parent.mkdir(parents=True,exist_ok=True);canvas.convert('RGB').save(path)
            caption=f'{data["house"]} / {row["room_label"]}：语义地面 {area:.2f} m²（{100*fraction:.1f}%）落在原生框外。黄色虚线为原生框，红色为框外的测得地面。'
            issue_html.append('<article class="issue"><h3>原生 region 框 ≠ 测得地面边界</h3><figure>'+img(path,bundle,caption,width=1536,height=1536)+f'<figcaption>{esc(caption)}</figcaption></figure><p>本次测量没有用框裁掉这些地面。原生框只作诊断；轮廓、机位与切分采用语义地面及既有导航网格。框外像素也不能直接视为另一个房间。</p></article>')
            receipts.append(dict(kind='native_bbox',house=data['house'],room=row['room_label'],fraction=fraction,area_m2=area,path=str(path)))
    stairs=next(((d,r) for d,r in all_mp if r.get('room_type')=='stairs' and r.get('demo_visual',{}).get('quality_images')),None)
    if stairs:
        data,row=stairs;q=row['demo_visual']['quality_images'][0];fraction=(row.get('floor_selection') or {}).get('dominant_area_fraction')
        caption=f'{data["house"]} / {row["room_label"]}：原生 s（楼梯），主层 {percent(fraction)}，共 {len(row.get("floors",[]))} 个测量窗口；图中按用途显示灰色。'
        issue_html.append('<article class="issue"><h3>楼梯与真实建筑层要分开解释</h3><figure>'+img(q['path'],bundle,caption,width=1536,height=1536)+f'<figcaption>{esc(caption)}</figcaption></figure><p>原生 s 已按冻结用途清单排除。地面高度窗口不是建筑楼层：台阶、地毯及语义混入都可能造成多窗口。未知 z / 复合用途是否含楼梯，仍须人工核对。</p></article>')
        receipts.append(dict(kind='stairs',house=data['house'],room=row['room_label'],path=q['path']))
    quality=[]
    for data,row in all_mp:
        f=dominant_metrics(row)
        if f and f['metrics'].get('black_fraction') is not None and row.get('demo_visual',{}).get('quality_images'):
            quality.append((f['metrics']['black_fraction'],data,row,f))
    if quality:
        black,data,row,f=max(quality,key=lambda t:t[0]);q=next((q for q in row['demo_visual']['quality_images'] if q.get('floor_id')==f['floor_id']),row['demo_visual']['quality_images'][0])
        caption=f'{data["house"]} / {row["room_label"]}：黑像素 {100*black:.1f}%，冻结门槛 ≤ {100*p["black_fraction_max"]:g}%。'
        issue_html.append('<article class="issue"><h3>黑像素是扫描质量代理</h3><figure>'+img(q['path'],bundle,caption,width=1536,height=1536)+f'<figcaption>{esc(caption)}</figcaption></figure><p>RGB 三通道都小于 8 才计黑像素，且只计语义地面多边形内、排除孔洞。深色材料、遮挡与未覆盖扫描可能产生同样的代理信号；这里没有把它称为真实缺失率。MP3D 新视图的迁移有效性未有人审验证。</p></article>')
        receipts.append(dict(kind='black_proxy',house=data['house'],room=row['room_label'],black_fraction=black,path=q['path']))
    uncommon = next(((d, r) for d, r in all_mp
                     if (r.get("native_region") or {}).get("region_label") == "S"), None)
    if uncommon:
        data, row = uncommon
        pictures = row.get("demo_visual", {}).get("interiors", []) or row.get("demo_visual", {}).get("quality_images", [])
        if pictures:
            picture = pictures[0]
            caption = f'{data["house"]} / {row["room_label"]}：原生 S = 水疗 / 桑拿；冻结判断用途仍是 {TYPE_ZH.get(row["room_type"], row["room_type"])}。'
            issue_html.append('<article class="issue"><h3>原生类别与冻结用途词表有缺口</h3><figure>' + img(picture['path'], bundle, caption) + f'<figcaption>{esc(caption)}</figcaption></figure><p>MP3D 原生标签区分大小写：S 是水疗 / 桑拿，s 才是楼梯。图和卡片显示官方原生名称；规则仍用已登记的冻结用途，没有因展示补词表而重判。建议在铺开前审核 B / S / Z 等类别以及复合 region。</p></article>')
            receipts.append(dict(kind='native_category_gap', house=data['house'], room=row['room_label'], native_code='S', frozen_type=row['room_type'], path=picture['path'], ontology_source='https://github.com/niessner/Matterport/blob/master/data_organization.md'))
    fp=next(((d,r) for d in hm for r in d['rows'] if comparison(r)[0]=='fp' and r.get('demo_visual',{}).get('interiors')),None)
    if fp:
        data,row=fp;pic=row['demo_visual']['interiors'][0];rules=rule_rows(row,p)
        numerical='；'.join(r['label']+' '+r['value'] for r in rules if r['label'] in ['语义地面面积','短边','可走主区面积'])
        caption=f'{data["house"]} / {row["room_label"]}：{numerical}。smy 跳过原因：{human_reason(row)}。'
        issue_html.append('<article class="issue"><h3>HM3D 误放说明规则覆盖的边界</h3><figure>'+img(pic['path'],bundle,caption)+f'<figcaption>{esc(caption)}</figcaption></figure><p>自动预筛满足数值门槛仍不等于人审可用。MP3D 没有人审记录，因此只能报告判定分布，不能声称沿用 HM3D 的精度。</p></article>')
        receipts.append(dict(kind='hm3d_false_positive',house=data['house'],room=row['room_label'],path=pic['path'],human_reason=human_reason(row)))
    fn=next(((d,r) for d in hm for r in d['rows'] if comparison(r)[0]=='fn' and r.get('demo_visual',{}).get('quality_images')),None)
    if fn:
        data,row=fn;q=row['demo_visual']['quality_images'][0];rule=first_block(row,rule_rows(row,p));reason=(rule['label']+' '+rule['value']+'，门槛 '+rule['threshold']) if rule else str(row['stage1']['reason_codes'])
        caption=f'{data["house"]} / {row["room_label"]}：自动 {reason}；smy 判可用。'
        issue_html.append('<article class="issue"><h3>HM3D 误杀也需要保留给 owner 看</h3><figure>'+img(q['path'],bundle,caption,width=1536,height=1536)+f'<figcaption>{esc(caption)}</figcaption></figure><p>该例来自已评估留出结果，展示规则与人工参照之间的差异。它没有被用于重新选阈值；MP3D 也保持同一冻结参数。</p></article>')
        receipts.append(dict(kind='hm3d_false_negative',house=data['house'],room=row['room_label'],rule=reason,path=q['path']))
    leak_stats=dict(region_count=len(candidates),arithmetic_mean_fraction=sum(t[0] for t in candidates)/len(candidates) if candidates else None,area_weighted_fraction=sum(t[1] for t in candidates)/sum(dominant_metrics(t[3])['metrics']['floor_area_m2'] for t in candidates) if candidates else None,maximum_fraction=max((t[0] for t in candidates),default=None),source='demo rendered_houses/*.json; dominant semantic floor union against native region bbox')
    write_json(out/'bbox_diagnostics.json',dict(summary=leak_stats,examples=receipts))
    return ''.join(issue_html),leak_stats,receipts


def build(args):
    out=args.output;bundle=out/'demo';bundle.mkdir(exist_ok=True)
    inputs=json.loads((out/'inputs.json').read_text());p=inputs['parameters']
    hm=[json.loads((out/'rendered_houses'/(h['house']+'.json')).read_text()) for h in inputs['hm3d']]
    mp=[json.loads((out/'rendered_houses'/(h['house']+'.json')).read_text()) for h in inputs['mp3d']]
    for data in hm+mp:maps_for_house(data,out,p)
    issue_html,leak_stats,examples=issues_for(hm,mp,p,bundle,out)
    merged=Path(inputs['merged_results']);agreement=json.loads((merged/'agreement.json').read_text());holdout=agreement['holdout']
    def family_counts(houses):
        raw=Counter();display=Counter();cmp=Counter();human=Counter()
        for data in houses:
            c=counts(data,p);raw.update(c['automatic_raw']);display.update(c['display']);cmp.update(c['comparison']);human.update(c['human'])
        return dict(houses=len(houses),regions=sum(len(d['rows']) for d in houses),automatic_raw=dict(raw),display=dict(display),comparison=dict(cmp),human=dict(human))
    hc=family_counts(hm);mc=family_counts(mp)
    summary=dict(created_at_sgt=now(),inputs_source=str(out/'inputs.json'),threshold_source=inputs['threshold_source'],threshold_sha256=inputs['threshold_sha256'],baseline_commit=inputs['baseline_commit'],hm3d_reference=dict(source=str(merged/'agreement.json'),historical_holdout_evaluation_ordinal=2,human_count=holdout['human_count'],precision=holdout['precision'],recall=holdout['recall'],confusion_matrix=holdout['confusion_matrix']),hm3d=hc,mp3d=mc,houses={d['house']:counts(d,p) for d in hm+mp},bbox_diagnostics=leak_stats,issue_examples=examples,mp3d_accuracy=None,accuracy_note='MP3D has no human labels; demonstration samples are purposively selected and not used to reestimate precision or recall.',interiors={})
    for family,houses in [('hm3d',hm),('mp3d',mp)]:
        stats=Counter()
        for data in houses:
            for row in data['rows']:
                pictures=row.get('demo_visual',{}).get('interiors',[])
                if row['stage1']['status'] in {'pass','review'}:
                    stats['pass_review_regions']+=1
                    if pictures:stats['with_interiors']+=1
                    else:stats['without_interiors']+=1
                    if pictures and pictures[0]['kind']=='placement_witness':stats['with_placement_witness']+=1
                    elif pictures:stats['reference_only']+=1
                if display_state(row, p) in {'pass', 'review'}:
                    stats['display_pass_review_regions'] += 1
                    stats['display_with_interiors' if pictures else 'display_without_interiors'] += 1
                stats['total_perspective_images']+=len(pictures)
        summary['interiors'][family]=dict(stats)
    write_json(bundle/'summary.json',summary)
    with (out/'rooms_registry.jsonl').open('w') as stream:
        for data in mp:
            for row in data['rows']:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    with (bundle/'regions.tsv').open('w') as stream:
        stream.write('family\thouse\tregion\tautomatic_raw\tdisplay_state\thuman\tfirst_reason\n')
        for data in hm+mp:
            for row in data['rows']:stream.write('\t'.join([data['family'],data['house'],row['room_label'],row['stage1']['status'],display_state(row,p),(row.get('human') or {}).get('verdict',''),','.join(row['stage1'].get('reason_codes',[]))])+'\n')
    legend=''.join(f'<span><i class="dot" style="background:{colour}"></i>{STATE_ZH[state]}</span>' for state,colour in COLOURS.items())+'<span>青色三角：摆放找到的相机；1 / 2：两个声源</span><span>橙色空心圈：参考相机，未证明摆放</span><span>虚线：无可用语义地面时的登记参照</span><span>蓝色线：切分建议</span><span>紫边 / !：对照人工结论</span>'
    flow=f'''<div class="flow"><article><b>00 · 登记与用途</b><h3>一个 region 也不隐藏</h3><p>读取语义实例及原生类别；排除楼梯、走廊、车库、储物和室外。用途未知保留证据，不编造房间标签。</p></article><article><b>01 · 几何资格</b><h3>面积、短边与可走主区</h3><p>语义地面 ≥ {p['floor_area_min_m2']:g} m²；短边 ≥ {p['short_side_min_m']:g} m；导航主区 ≥ {p['nav_main_area_min_m2']:g} m²，卫生间 ≥ {p['bathroom_nav_main_area_min_m2']:g} m²。面积是扫描代理。</p></article><article><b>01 · 质量、楼层与摆放</b><h3>每条规则都留下实测值</h3><p>黑比例 ≤ {100*p['black_fraction_max']:g}%；主层 ≥ {100*p['dominant_floor_area_fraction_min']:g}%、窗口 ≤ {p['floor_height_separation_m']:g} m。相机与双声源间距 {p['distance_min_m']:g}–{p['distance_max_m']:g} m，三条视线畅通，水平视野 {p['camera_hfov_deg']:g}°；净空 ≥ {p['placement_camera_clearance_m']:g} / {p['placement_source_clearance_m']:g} m。</p></article><article><b>02 → 人审 → 04</b><h3>建议切分，再看房子门禁</h3><p>面积 > {p['split_area_trigger_m2']:g} m²、家具中心相隔 > {p['split_cluster_separation_m']:g} m 或凸性 < {p['split_convexity_trigger']:g} 会触发切分建议；每个子块独立测量和人审。正式音视频闭环门禁另列。</p></article></div>'''
    def distribution(label,c):
        raw=c['automatic_raw'];total=c['regions']
        return f'<tr><th>{label}</th><td>{c["houses"]} / {total}</td><td>{raw.get("pass",0)}（{100*raw.get("pass",0)/total:.1f}%）</td><td>{raw.get("fail",0)}（{100*raw.get("fail",0)/total:.1f}%）</td><td>{raw.get("review",0)}（{100*raw.get("review",0)/total:.1f}%）</td><td>{c["display"].get("excluded",0)}</td></tr>'
    comparison_table='<div class="table-scroll"><table class="comparison-table"><thead><tr><th>演示样本</th><th>房子 / 登记单元</th><th>原协议通过</th><th>原协议不通过</th><th>原协议待审</th><th>其中灰色排除类</th></tr></thead><tbody>'+distribution('HM3D · 人工参照',hc)+distribution('MP3D · 无人工参照',mc)+'</tbody></table></div>'
    source='room_selection_merged_20261003/agreement.json · 原 91 套留出，715 条人工记录；同一留出集的第二次历史评估'
    hero=f'''<header class="hero"><div class="hero-inner"><div class="eyebrow">V7 ROOM SELECTION · FROZEN MERGED PROTOCOL</div><h1>自动规则是怎样选房间的？<br>先用 HM3D 对照，再看 MP3D。</h1><p>自动规则先把每个 region 测量、判定并提出切分，再交给人工审核。绿色是预筛通过，红色有已知失败，黄色证据不足或需人工确认，灰色属于冻结排除用途。下面将所有房子、规则值与真实内景直接铺开，无需点击展开。</p><div class="hero-metrics"><div class="hero-metric"><b>{100*holdout['recall']:.1f}%</b><span>HM3D 留出召回 · TP {holdout['confusion_matrix']['TP']} / 人工可用 {holdout['confusion_matrix']['TP']+holdout['confusion_matrix']['FN']}</span></div><div class="hero-metric"><b>{100*holdout['precision']:.1f}%</b><span>HM3D 留出精度 · TP {holdout['confusion_matrix']['TP']} / 已定人工标签中自动通过 {holdout['confusion_matrix']['TP']+holdout['confusion_matrix']['FP']}</span></div><div class="hero-metric"><b>6 + 6 套</b><span>{hc['regions']} 个 HM3D 登记单元 / {mc['regions']} 个 MP3D region</span></div><div class="hero-metric"><b>阈值不变</b><span>面积 {p['floor_area_min_m2']:g} m² · 短边 {p['short_side_min_m']:g} m · 主层 {100*p['dominant_floor_area_fraction_min']:g}%</span></div></div><p class="source-line">{esc(source)}</p></div></header>'''
    body=hero+'<main>'+flow+'<div class="note-box"><strong>看图时请保留三点区别：</strong>地面多边形是语义扫描测量，不是人工核实的建筑净面积；找到摆放见证不代表声学已验；新增纹理图不替换 HM3D 原有测量。MP3D 黑像素采用保存原生相机矩阵的新校准俯视图，其迁移有效性仍未人工验证。</div><div class="legend">'+legend+'</div>'
    body+='<div class="family-heading"><div><h2>HM3D：先看规则与人工是否一致</h2><p>6 套均来自原 91 套留出，包含单层、多层、一致通过、误放、误杀及待审。自动结论、region 边界和规则值直接沿用合并产物；smy 有记录的逐项显示，没有记录的明确留空。</p></div><span class="family-badge">有人工参照 · 不重评留出</span></div>'
    body+=''.join(house_section(data,p,bundle,i+1) for i,data in enumerate(hm))
    body+='<div class="family-heading"><div><h2>MP3D：同一冻结规则会筛出什么</h2><p>包括原 pilot 的 3 套，另加单层小户型、大型单层及四层房子。每个原生 region 都登记；测量复用合并后的共享面积、家具、导航、射线摆放与切分算法。</p></div><span class="family-badge">无人工参照 · 不计算精度</span></div>'
    body+=''.join(house_section(data,p,bundle,i+1) for i,data in enumerate(mp))
    body+='<div class="family-heading"><div><h2>MP3D 和 HM3D 的差别</h2><p>以下比例仅描述刻意选出的演示样本。HM3D 为展示误差而选样；MP3D 没有人工标签。它们不是随机样本，不能据此外推 90 套房子的精度或召回。</p></div></div>'+comparison_table
    body+=f'<p class="source-line" style="color:#5b6d78">灰色排除类属于原协议三态的一部分，不能与后三态再相加。MP3D 有地面几何的 {leak_stats["region_count"]} 个 region：框外地面比例均值 {percent(leak_stats["arithmetic_mean_fraction"])}，面积加权 {percent(leak_stats["area_weighted_fraction"])}；出处：bbox_diagnostics.json（主语义层口径）。</p>'
    body+='<div class="issue-grid">'+issue_html+'</div><div class="suggestions"><h3>改进建议：本轮只列建议，不改变冻结规则</h3><p>先人工核对 MP3D 的类别与边界，用导航支撑域、语义地面及原生框一起审，不以框直接裁房间；unknown / 复合用途要单独排队。</p><p>抽样核对新俯视切面、黑像素与真实缺损的关系，并核实台阶 / 地毯是否造成 98% 主层规则待审。需要调整参数时，在独立 MP3D 人工校准样本上定规则，再留出整套房子评估。</p><p>切分建议继续独立人审；补齐 MP3D 的正式房子级音频 / RIR 与下游题库闭环后，再确定可进入题库的房间。</p></div>'
    body+='<footer class="footer"><p>源基线：b2d625f · 冻结参数：'+esc(inputs['threshold_source'])+'<br>HM3D 原产物：'+esc(inputs['merged_results'])+'<br>生成时间（新加坡）：'+esc(summary['created_at_sgt'])+'</p><a href="summary.json">数值与出处</a><a href="regions.tsv">逐 region 清单</a><a href="REPORT_zh.md">交付说明</a><p>所有图片均为原始场景的 Habitat-Sim 原生渲染或原 HM3D 校准图；离线相对路径、全部懒加载。纹理解码兼容使用独立 GLB 副本，仅把原 Basis 图像解码为 PNG；网格、UV、材质参数和场景节点均核对未改。此页为展示页，未写回人工审核记录。</p></footer></main>'
    document='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V7 自动选房 · HM3D 与 MP3D 演示</title><style>'+CSS+'</style></head><body>'+body+'</body></html>'
    (bundle/'index.html').write_text(document)
    for data in hm+mp:write_json(out/'rendered_houses'/(data['house']+'.json'),data)
    print('DEMO_PAGE',str(bundle/'index.html'),json.dumps(dict(hm3d=hc,mp3d=mc,interiors=summary['interiors']),ensure_ascii=False),flush=True)
    return summary
