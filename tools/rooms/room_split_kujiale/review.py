"""Build one embedded-JPEG review page with strict viewport image loading."""
from __future__ import annotations
import argparse, base64, html, json
from pathlib import Path
from PIL import Image
from tools.rooms.room_split_kujiale.adapter import read, dump


def build(root):
    root=Path(root);summary=read(root/'kujiale_room_list_draft_v1/summary.json')
    outcomes={r['house']+'/'+r['source_region']:r for r in read(root/'comparison_old_33_rooms_v1.json')['old_admitted']}
    regs=[read(p) for p in sorted((root/'kujiale_delivery_v1/final_v1/regions').glob('*.json'))]
    overlays=read(root/'owner_overlays_all_sources_v1/index.json')
    images={e['region']:e['images'] for e in overlays}
    assets={};sections=[[],[],[]];seen=set()
    for reg in regs:
        key=reg['house']+'/'+reg['source_region'];old=outcomes.get(key);kept=[b for b in reg['blocks'] if b['decision']=='retain']
        section=0 if reg['requires_split'] else 1 if old and old['change']!='unchanged' else 2 if kept else None
        if section is None:continue
        imagefiles=images.get(key,[])
        pics=[]
        for file in imagefiles:
            path=Path(file);aid=str(len(assets));assets[aid]='data:image/jpeg;base64,'+base64.b64encode(path.read_bytes()).decode('ascii')
            with Image.open(path) as im:width,height=im.size
            pics.append(f'<figure><img data-asset="{aid}" width="{width}" height="{height}" alt="{html.escape(key)} 实际网格俯视叠图"><figcaption>真实原始 USD 网格 · CPU 正交光栅 · 材质颜色</figcaption></figure>')
        if not pics:pics=['<p class="missing">缺少渲染，未验证</p>']
        reasons=sorted({x for b in reg['blocks'] for x in b.get('discard_reasons',[])+b.get('unresolved_reasons',[])})
        blocks=''.join('<tr><td>'+html.escape(b['id'])+'</td><td>'+html.escape(b['decision'])+'</td><td>%.2f</td><td>%.2f</td><td>%s</td></tr>'%(b['floor_area_m2'],b['short_side_m'],'找到并核查' if b['placement_witness'].get('found') else '未找到或未运行') for b in reg['blocks'])
        change=old['change'] if old else '新增候选来源'
        text=f'{key} {change} '+ ' '.join(reasons)
        sections[section].append(f'<article data-search="{html.escape(text.lower(),quote=True)}"><h3>{html.escape(key)} · {html.escape(reg["source_geometry"]["source_room_type"])}</h3>'
            f'<p>原面积 {reg["source_floor_area_m2"]:.2f} m²；保留 {len(kept)} 间；旧准入对照：{html.escape(change)}。</p>'+''.join(pics)+
            '<div class="scroll"><table><thead><tr><th>最终 ID</th><th>去向</th><th>面积 m²</th><th>短边 m</th><th>放置见证</th></tr></thead><tbody>'+blocks+'</tbody></table></div>'+
            ('<p>原因：'+html.escape('；'.join(reasons))+'</p>' if reasons else '')+'</article>')
        seen.add(key)
    headings=['被切的客厅（全部 14 个大面积来源）','与旧 33 间准入相比有变化的其余房间','其余保留房间（含新增候选）']
    body=''.join(f'<section><h2>{i+1}. {title} · {len(sections[i])} 个来源</h2>'+''.join(sections[i])+'</section>' for i,title in enumerate(headings))
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>酷家乐房间切分审图 · 2026-10-10</title>
<style>body{font:16px/1.55 system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:20px;background:#f3f4f6;color:#18202a}header{position:sticky;top:0;background:#f3f4f6ee;padding:12px 0;z-index:3}h1{font-size:24px;margin:0}h2{margin-top:36px}article{background:white;border:1px solid #ddd;border-radius:8px;padding:18px;margin:18px 0;content-visibility:auto;contain-intrinsic-size:auto 1250px}img{width:100%;height:auto;display:block;background:#eee;cursor:zoom-in}figure{margin:0}figcaption{color:#666;font-size:13px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;border-bottom:1px solid #ddd;padding:8px}.scroll{overflow:auto}input{padding:8px;width:min(500px,90%)}.missing{color:#a00}dialog{max-width:95vw;max-height:95vh;border:0;padding:8px}dialog img{width:auto;max-width:90vw;max-height:87vh}button{padding:6px 14px}article[hidden]{display:none}</style>
<header><h1>酷家乐房间切分审图</h1><p>SUMMARY</p><input id="search" placeholder="按房屋、房间或原因筛选" aria-label="筛选房间"></header>
<p>蓝/橙等色为保留房；斜纹 × 为丢弃；交叉纹 ? 为未判定。面积使用真实交付多边形。漏声待另一任务测量，当前为名单草稿。≤5% 可测试，5–15% 仅训练，>15% 不用。生产仍渲染整屋、计算整屋声音。图片仅供本地非商业研究审查，不许再分发。</p>
BODY
<dialog id="viewer"><button id="close">关闭</button><img id="large" alt="放大叠图"></dialog>
<script id="embedded-jpegs" type="application/json">ASSETS</script>
<script>
'use strict';
const payload=document.getElementById('embedded-jpegs');let cache;
const getAsset=id=>{if(!cache)cache=JSON.parse(payload.textContent);return cache[id];};
const load=img=>{if(!img.hasAttribute('src'))img.setAttribute('src',getAsset(img.dataset.asset));};
const observer='IntersectionObserver' in window?new IntersectionObserver(entries=>{for(const entry of entries){if(entry.isIntersecting&&entry.intersectionRatio>0){load(entry.target);observer.unobserve(entry.target);}}},{rootMargin:'0px',threshold:0.01}):null;
for(const img of document.querySelectorAll('img[data-asset]')){if(observer)observer.observe(img);else{img.insertAdjacentHTML('beforebegin','<button class="manual">点击加载图片</button>');img.previousElementSibling.addEventListener('click',()=>load(img));}img.addEventListener('click',()=>{load(img);document.getElementById('large').src=getAsset(img.dataset.asset);document.getElementById('viewer').showModal();});}
document.getElementById('close').onclick=()=>{document.getElementById('viewer').close();document.getElementById('large').removeAttribute('src');};
document.getElementById('search').addEventListener('input',event=>{const q=event.target.value.toLowerCase();for(const card of document.querySelectorAll('article'))card.hidden=!card.dataset.search.includes(q);});
</script><noscript>为保证严格懒加载，禁用 JavaScript 时不解码图片。</noscript></html>'''
    info=f'151 个来源 · 保留 {summary["total_rooms"]} 间 · 面积 {summary["total_area_m2"]:.1f} m² · 未判定 {summary["pending"]} 块 · 全程 CPU'
    document=document.replace('SUMMARY',html.escape(info)).replace('BODY',body).replace('ASSETS',json.dumps(assets,separators=(',',':')))
    path=root/'KUJIALE_REVIEW_v1.html'
    with path.open('x',encoding='utf8') as f:f.write(document)
    # Static evidence is distinct from an actual browser run.
    dump(root/'evidence/review_page_static_v1.json',dict(path=str(path),bytes=path.stat().st_size,embedded_jpegs=len(assets),source_cards=len(seen),
         section_source_counts=[len(x) for x in sections],initial_image_src_attributes=0,
         lazy_policy='empty img src at HTML load; base64 stays in JSON text; viewport IntersectionObserver rootMargin=0 threshold=.01 decodes images on demand; dialog loads on click',
         external_image_requests=0,browser_execution_status='not_run'))
    print('REVIEW',str(path),'bytes',path.stat().st_size,'images',len(assets),'sections',[len(x) for x in sections])

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',required=True);v=a.parse_args();build(v.root)
