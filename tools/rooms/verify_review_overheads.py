"""Verify every overhead mapping/file and read-only browser interaction."""
import json,io,urllib.request,argparse
from pathlib import Path
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--base',default='http://127.0.0.1:8789')
    args=parser.parse_args();root=args.root;base=args.base
    inv=json.loads((root/'inventory.json').read_text())
    entries=json.loads((root/'manifest.json').read_text())['entries']
    expected={h['house']+'/'+r['label'] for h in inv['houses'] for r in h['rooms']}
    assert set(entries)==expected
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));fail=[];checked=0;paths=set();sparse=[]
    sources={}
    for key,e in entries.items():
        assert key==e['house']+'/'+e['room_label']
        assert Path(e['source_glb']).is_file() and not e['source_glb'].endswith('.basis.glb')
        if e['rooms_source'] not in sources:
            sources[e['rooms_source']]={f"R{r['region_id']}":r for r in json.loads(Path(e['rooms_source']).read_text())['rooms']}
        original=sources[e['rooms_source']][e['room_label']]
        assert original['bbox_xz_m']==e['bbox_xz_m'] and original['floor_y_m']==e['floor_y_m']
        assert e['orientation']=='image_right=-X,image_up=+Z'
        lo,hi=np.array(e['bbox_xz_m']);centre=(lo+hi)/2
        for im in e['images']:
            try:
                rel=im['path'];assert rel not in paths;paths.add(rel)
                data=opener.open(base+'/overheads/'+rel,timeout=20).read()
                assert data==(root/rel).read_bytes()
                image=Image.open(io.BytesIO(data));image.load();assert image.size==(1024,1024)
                assert np.asarray(image).std()>1
                assert im['span_m']>=max(hi-lo)+1.-.0001
                mat=np.array(im['projection']);v=np.array(e['view'])
                transform=np.linalg.inv(v)
                assert np.linalg.norm(transform[:3,:3]@np.array([0,0,-1])-np.array([0,-1,0]))<1e-5
                assert abs(abs(mat[0,0])-abs(mat[1,1]))<1e-6
                assert abs(2/abs(mat[0,0])-im['span_m'])<.001
                assert np.allclose(mat[3],[0,0,0,1])
                points=np.array([[x,e['floor_y_m'],z,1] for x in [lo[0],hi[0]] for z in [lo[1],hi[1]]])
                clip=(mat@v@points.T).T;assert np.max(np.abs(clip[:,:2]/clip[:,3,None]))<1
                checked+=1
            except Exception as ex:fail.append([key,im['path'],repr(ex)])
        if min(im['black_fraction'] for im in e['images'])>.75:sparse.append(key)
    browser_errors=[];samples=[];writes=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page(viewport={'width':1600,'height':1000})
        def readonly(route):
            if route.request.method not in ('GET','HEAD'):
                writes.append(route.request.url);route.abort()
            else:route.continue_()
        page.route('**/*',readonly)
        page.on('pageerror',lambda error:browser_errors.append(str(error)))
        page.goto(base+'/curation_review.html');page.wait_for_function("typeof DATA !== 'undefined' && DATA && document.querySelectorAll('.room-row').length>0")
        for key in [list(entries)[0],'hm3d_train_00006_HkseAnWCgqk/R8','hm3d_train_00250_U3oQjwTuMX8/R1',list(entries)[-1]]:
            house,label=key.split('/')
            page.evaluate('([h,r])=>selectRow([...document.querySelectorAll(".room-row")].find(row=>row.dataset.house===h&&row.dataset.label===r))',[house,label])
            page.wait_for_function("key=>{let i=document.getElementById('pv-img');return !i.hidden&&i.complete&&i.naturalWidth===1024&&i.src.includes(key+'_overview.png')}",arg=key)
            page.select_option('#pv-overhead-view','1')
            page.wait_for_function("key=>{let i=document.getElementById('pv-img');return !i.hidden&&i.complete&&i.naturalWidth===1024&&i.src.includes(key+'_lower.png')}",arg=key)
            assert page.locator('#pv-img').evaluate("i=>getComputedStyle(i).objectFit")=='contain'
            samples.append(key)
        # Rapid changes must not leave the previous room's picture displayed.
        page.evaluate("()=>{renderPreview('hm3d_train_00006_HkseAnWCgqk','R8');renderPreview('hm3d_train_00250_U3oQjwTuMX8','R1')}")
        page.wait_for_function("()=>{let i=document.getElementById('pv-img');return !i.hidden&&i.complete&&i.src.includes('00250_U3oQjwTuMX8/R1_overview.png')}")
        page.screenshot(path=str(root/'browser_verified.png'),full_page=False);browser.close()
    result={'expected_entries':len(expected),'mapped_entries':len(entries),'verified_png_http_decode_count':checked,'failures':fail,'browser_errors':browser_errors,'unexpected_write_attempts':writes,'browser_samples':samples,'sparse_views_for_review':sparse,'note':'Black fraction is whole-image, includes unified framing margins; not a room rejection. No verdict POST made.'}
    (root/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False))
    assert not fail and not browser_errors and not writes

if __name__=='__main__':main()
