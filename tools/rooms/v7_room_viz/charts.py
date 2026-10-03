"""Small responsive SVG plots derived directly from immutable acceptance JSON."""
from __future__ import annotations
import math
from html import escape
from statistics import median
from collections import defaultdict

COLORS = ['#118d8c', '#bf7042', '#6666bb']


def number(value, digits=2):
    return '未测' if value is None else f'{value:.{digits}f}'


def text(x,y,value,size=15,anchor='start',color='#425466'):
    size = max(size, 18)
    return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" fill="{color}">{escape(str(value))}</text>'


def line(x1,y1,x2,y2,color='#dce5e9',width=1,dash=''):
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>'


def rect(x,y,w,h,color,opacity=1):
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(w,0):.2f}" height="{max(h,0):.2f}" fill="{color}" opacity="{opacity}" rx="3"/>'


def svg(content,title,width=480,height=270):
    return f'<svg class="plot" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title,quote=True)}"><title>{escape(title)}</title>{content}</svg>'


def empty(title):
    return svg(rect(15,35,450,165,'#eef3f4')+text(240,112,'尚未完成这项测量',20,'middle')+
               text(240,144,'缺失值不按 0 或通过显示',15,'middle'),title)


def band_stats(result):
    bands=defaultdict(list)
    for rir in result.get('rirs',[]):
        for octave in rir.get('octaves',[]):
            fit = octave.get('t30', {})
            if fit.get('status') != 'valid':
                fit = octave.get('t20', {})
            if fit.get('status') == 'valid':
                bands[octave['center_hz']].append(fit['rt60_s'])
    return {str(hz):dict(median_s=median(values),min_s=min(values),max_s=max(values),n=len(values))
            for hz,values in sorted(bands.items())}


def rt60(result):
    stats=band_stats(result)
    title='分频带 RT60；色带为 0.2–1.5 秒目标范围'
    if not stats:
        return empty(title)
    y_max=max(1.6,math.ceil(max(v['max_s'] for v in stats.values())*2)/2)
    py=lambda v:218-v/y_max*175
    s=rect(52,py(1.5),410,py(.2)-py(1.5),'#daf1e7')
    ticks={0,.2,.5,1,1.5}
    if y_max-1.5>=.35:
        ticks.add(y_max)
    for value in sorted(ticks):
        if value>y_max:continue
        s+=line(52,py(value),462,py(value))+text(44,py(value)+5,f'{value:g}',13,'end')
    s+=text(52,23,'秒',14)+text(454,39,'目标带',14,'end','#197457')
    for i,hz in enumerate([125,250,500,1000,2000,4000]):
        x=87+i*65
        stat=stats.get(str(hz))
        if stat:
            v=stat['median_s']
            s+=rect(x-18,py(v),36,218-py(v),'#168d8e')
            s+=line(x,py(stat['min_s']),x,py(stat['max_s']),'#164c59',2)
            s+=line(x-6,py(stat['max_s']),x+6,py(stat['max_s']),'#164c59',2)
            s+=text(x,max(42,py(v)-10),f'{v:.2f}',14,'middle','#183d50')
        s+=text(x,242,f'{hz/1000:g}k' if hz>=1000 else str(hz),14,'middle')
    s+=text(453,263,'Hz',13,'end')
    return svg(s,title)


def drr(result):
    title='直达混响比；负值表示混响能量更强'
    groups=defaultdict(list)
    for rir in result.get('rirs',[]):
        if rir.get('drr_db') is not None:groups[rir['room_label']].append(rir['drr_db'])
    if not groups:return empty(title)
    low=min(-20,5*math.floor(min(min(v) for v in groups.values())/5))
    high=max(5,5*math.ceil(max(max(v) for v in groups.values())/5))
    px=lambda v:65+(v-low)/(high-low)*350
    s=text(65,28,'dB · 每个检查房间的中位数与范围',15)
    for tick in range(int(low),int(high)+1,5):
        s+=line(px(tick),40,px(tick),210)+text(px(tick),238,str(tick),13,'middle')
    s+=line(px(0),40,px(0),210,'#8098a4',2,'4 4')
    for i,(name,values) in enumerate(groups.items()):
        y=70+i*51
        s+=text(14,y+5,name,16)+line(px(min(values)),y,px(max(values)),y,COLORS[i%3],5)
        m=median(values)
        s+=f'<circle cx="{px(m):.2f}" cy="{y}" r="7" fill="{COLORS[i%3]}"/>'
        s+=text(450,y+5,f'{m:.1f}',15,'end')
    s+=text(65,263,'描述性指标，无单独的通过阈值',14)
    return svg(s,title)


def distance_series(result):
    groups=defaultdict(list)
    for rir in result.get('rirs',[]):
        suffix=rir['probe'].removeprefix(rir['room_label']+'_')
        if suffix in {'range_near','range_middle','source_1'} and rir.get('direct_energy',0)>0:
            groups[rir['room_label']].append(rir)
    return {room:sorted(values,key=lambda x:x['distance_m']) for room,values in groups.items() if len(values)>=2}


def attenuation(result):
    title='距离衰减；各房间以最近位置为 0 dB'
    groups=distance_series(result)
    if not groups:return empty(title)
    xmax=max(x['distance_m'] for v in groups.values() for x in v)*1.12
    all_y=[10*math.log10(x['direct_energy']/v[0]['direct_energy']) for v in groups.values() for x in v]
    ymin=min(-20,math.floor(min(all_y)/5)*5)
    ymax=max(3,math.ceil(max(all_y)/5)*5)
    px=lambda v:62+v/xmax*390
    py=lambda v:218-(v-ymin)/(ymax-ymin)*166
    s=text(62,26,'相对直达能量 dB',14)
    for tick in range(ymin,1,5):
        s+=line(62,py(tick),452,py(tick))+text(52,py(tick)+5,str(tick),13,'end')
    for tick in range(0,int(xmax)+1,max(1,int(xmax/5))):
        s+=text(px(tick),241,str(tick),13,'middle')
    for i,(room,points) in enumerate(groups.items()):
        color=COLORS[i%3]
        baseline=points[0]['direct_energy']
        values=[(p['distance_m'],10*math.log10(p['direct_energy']/baseline)) for p in points]
        reference=[(values[0][0]+(values[-1][0]-values[0][0])*j/32,
                    -20*math.log10((values[0][0]+(values[-1][0]-values[0][0])*j/32)/values[0][0]))
                   for j in range(33)]
        for a,b in zip(reference,reference[1:]):
            s+=line(px(a[0]),py(a[1]),px(b[0]),py(b[1]),color,1.5,'3 3')
        for a,b in zip(values,values[1:]):s+=line(px(a[0]),py(a[1]),px(b[0]),py(b[1]),color,3)
        for x,y in values:s+=f'<circle cx="{px(x):.2f}" cy="{py(y):.2f}" r="5" fill="{color}"/>'
        s+=text(66+i*125,263,room,14,color=color)
    s+=text(454,263,'距离 m',14,'end')
    return svg(s,title)


def leakage(result):
    title='漏声比例；所有射线原点中的最差值'
    fraction=result.get('max_escape_fraction')
    if fraction is None:return empty(title)
    value=fraction*100
    maximum=max(20,10*math.ceil(value/10))
    px=lambda v:35+v/maximum*410
    color='#108872' if value<=5 else '#bc8432' if value<=15 else '#d6455e'
    s=text(35,54,f'{value:.2f}%',35,color=color)+text(35,83,'最差原点的逃逸射线比例',16)
    s+=rect(35,122,px(5)-35,37,'#d8eee7')+rect(px(5),122,px(15)-px(5),37,'#faeccc')
    s+=rect(px(15),122,445-px(15),37,'#f8e0e5')
    s+=rect(35,131,px(value)-35,19,color)
    for v,label in [(5,'5% 通过线'),(15,'15% 不通过线')]:
        s+=line(px(v),115 if v==5 else 104,px(v),177,'#566a74',2,'4 3')
        s+=text(px(v),110 if v==5 else 196,label,14,'middle')
    s+=text(35,228,'0%',14)+text(445,228,f'{maximum:g}%',14,'end')
    s+=text(35,258,'≤5% 通过；5–15% 待审；>15% 不通过',14)
    return svg(s,title)


def distribution(results):
    values=[r['rt60_mid_median_s'] for r in results if r['status']=='pass' and r.get('rt60_mid_median_s') is not None]
    title='全部通过房屋的 RT60 分布；每屋一个中頻中位数'
    counts=[sum(i*.25<=v<(i+1)*.25 for v in values) for i in range(10)]
    ymax=max(5,max(counts)+1)
    px=lambda v:65+v/2.5*660
    py=lambda v:218-v/ymax*168
    s=rect(px(.2),50,px(1.5)-px(.2),168,'#dcefe6')+rect(px(1.5),50,px(2.5)-px(1.5),168,'#fff0d9')
    for tick in range(ymax+1):s+=line(65,py(tick),725,py(tick))+text(54,py(tick)+5,tick,14,'end')
    for i,n in enumerate(counts):
        s+=rect(px(i*.25)+5,py(n),56,218-py(n),'#178d8d')
        s+=text(px(i*.25+.125),py(n)-8,n,17,'middle')
    for tick in [0,.2,.5,1,1.5,2,2.5]:s+=text(px(tick),244,f'{tick:g}',14,'middle')
    s+=text(65,28,'房屋数 · n='+str(len(values)),16)+text(px(.8),43,'主目标 0.2–1.5 s',16,'middle','#197457')
    s+=text(px(2),43,'长尾 1.5–2.5 s：0 套',16,'middle','#a26520')+text(725,272,'RT60 秒',14,'end')
    return svg(s,title,790,285)
