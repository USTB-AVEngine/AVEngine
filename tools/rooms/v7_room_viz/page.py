"""Flat, offline, lazy-loaded Chinese owner review with traceable small charts."""
from __future__ import annotations
import json
import re
import zipfile
from collections import Counter
from html import escape as esc
from pathlib import Path
from .common import read_json, write_json, now
from . import charts

STATUS={'pass':'通过','review':'待审','fail':'不通过'}
SPLIT={'train':'训练','development':'开发','test':'测试'}
ROOM={'living':'起居室','bedroom':'卧室','kitchen':'厨房','dining':'餐厅','bathroom':'卫浴',
      'unknown':'用途未明','office':'书房','hall':'走廊','stairs':'楼梯','laundry':'洗衣间'}
STARSS=['女性现场讲话','男性现场讲话','拍手','电话','笑声','特定家务设备','步行脚步','门／柜门',
        '播放音乐','现场乐器','水龙头','手动铃','敲击']
NAMES={'air_conditioning':'空调','alarm_beep':'电子报警','alarm_bell':'报警铃','alarm_clock':'闹钟',
       'any_audioset_class_playback':'任意类别录音播放','bathtub_filling_washing':'浴缸用水','blender':'搅拌机',
       'busy_signal':'电话忙音','cat_meow':'猫叫','cellphone_vibration_alert':'手机振动提示','chime':'碰铃／钟声',
       'clock_tick':'时钟滴答','crackle':'噼啪声','ding_dong':'叮咚门铃','dog_bark':'狗叫','doorbell':'门铃',
       'doorbell_chime':'门铃钟声','drip':'水滴','fire':'火声','fire_alarm':'火警','gurgling':'冒泡声',
       'microwave_beep':'微波炉提示','microwave_hum':'微波炉机械声','music_playback':'扬声器播放音乐',
       'phone_ring':'电话铃','printer':'打印机','ringtone':'手机铃声','sink_filling_washing':'洗槽用水',
       'smoke_alarm':'烟雾报警','speech_playback':'扬声器播放讲话','telephone':'电话提示',
       'telephone_bell_ringing':'电话机械铃','telephone_dialing_dtmf':'电话拨号音','toilet_flush':'冲厕',
       'water_tap_faucet':'水龙头','human_speech_female':'女性现场讲话','human_speech_male':'男性现场讲话',
       'hand_clapping':'拍手','human_laughter':'笑声','vacuum_cleaner':'吸尘器','water_boiling':'烧水',
       'walking_footsteps':'步行脚步','door_open_close':'门／柜门开关','instrument_live':'现场乐器',
       'bell_manual':'手动铃','surface_knock':'敲击'}

CSS='''
:root{--ink:#1d3441;--muted:#5b707b;--line:#dbe5e7;--teal:#098579;--amber:#a87122;--red:#ba3c52;--paper:#fff;--bg:#f3f5f1}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;line-height:1.6;font-size:16px;overflow-wrap:anywhere}
main{max-width:1320px;margin:auto;padding:38px 28px 55px}h1,h2,h3,h4,p{margin:0}h1{font-size:clamp(30px,4.2vw,52px);line-height:1.2;letter-spacing:-1px;margin:10px 0 18px}h2{font-size:28px;margin-bottom:8px}h3{font-size:22px;line-height:1.35}h4{font-size:17px;margin-bottom:5px}.kicker{color:var(--teal);font-size:13px;font-weight:750;letter-spacing:2px}.muted,.note,figcaption{color:var(--muted)}.note{font-size:14px;line-height:1.7}.section{margin-top:35px}.panel,.house{background:var(--paper);border:1px solid var(--line);border-radius:18px;padding:26px;margin:22px 0;box-shadow:0 3px 14px #153c4c04}
.hero{border-left:5px solid var(--teal);padding:0 0 2px 22px}.hero p{max-width:900px}.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:22px}.stat{padding:16px 18px;background:#fff;border:1px solid var(--line);border-radius:12px}.stat strong{display:block;font-size:29px;line-height:1.3}.stat span{font-size:14px;color:var(--muted)}.stat.pass strong{color:var(--teal)}.stat.review strong{color:var(--amber)}.stat.fail strong{color:var(--red)}
.flow{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:18px;margin:22px 0}.step{position:relative;background:#eef6f4;border:1px solid #d1e9e2;border-radius:12px;padding:20px 15px;text-align:center}.step svg{width:44px;height:44px;display:block;margin:0 auto 9px;stroke:var(--teal);fill:none;stroke-width:2}.step b{display:block;font-size:18px}.step small{color:var(--muted);display:block;margin-top:6px}.step:not(:last-child)::after{content:'→';position:absolute;right:-17px;top:42%;font-size:23px;color:var(--teal)}
.timelines{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px;margin-top:22px}.time-item{border:1px solid var(--line);border-radius:12px;padding:16px}.axis{height:65px;position:relative;margin:9px 5px 3px;background:linear-gradient(to right,#dce6e6 1px,transparent 1px);background-size:10% 100%;border-bottom:1px solid #b4c8cc}.bar{position:absolute;top:9px;height:17px;border-radius:4px;background:var(--teal)}.bar.alt{top:34px;background:#6976bb}.bar.pause{background:#dce5e6}.bar.move{background:repeating-linear-gradient(45deg,#098579,#098579 6px,#58aa9f 6px,#58aa9f 10px)}.ticks{display:flex;justify-content:space-between;font-size:12px;color:var(--muted)}.time-item:last-child{grid-column:1/-1}.time-item:last-child .axis{height:49px}.time-item:last-child .bar{height:25px;text-align:center;color:#fff;font-size:13px;top:8px}.quota{color:var(--teal);font-weight:650;font-size:13px}
.split-grid{display:grid;grid-template-columns:2fr 1fr 1fr;gap:16px;margin-top:20px}.split-box{padding:18px;border:1px solid var(--line);border-radius:12px;background:#fbfcfa}.split-box h3{margin-bottom:12px}.house-list{display:grid;gap:5px}.house-list span{display:block;font-size:13px;overflow-wrap:anywhere;padding:6px 9px;background:#edf4f2;border-radius:5px}.split-box:first-child .house-list{grid-template-columns:repeat(2,minmax(0,1fr))}.rules{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:20px}.rule{border:1px solid var(--line);border-radius:10px;padding:15px}.rule strong{display:block;font-size:22px;color:var(--teal);margin:2px 0}.rule p{font-size:14px;color:var(--muted)}
.taxonomy{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin:18px 0}.class-pair{display:grid;grid-template-columns:minmax(0,1fr) 34px minmax(0,1fr);align-items:center;padding:15px;background:#eaf6f2;border-radius:10px;font-size:14px}.class-pair>div{min-width:0}.class-pair.playback{background:#fff1df}.class-pair b{display:block;font-size:17px}.class-pair .arrow{text-align:center;font-size:25px;color:var(--teal)}.native-classes{display:flex;flex-wrap:wrap;gap:6px;margin:15px 0}.native-classes span{background:#eef3f7;border-radius:5px;padding:5px 8px;font-size:13px}.mapping-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.mapping-grid table{table-layout:fixed;width:100%;border-collapse:collapse;font-size:13px}.mapping-grid th,.mapping-grid td{padding:9px 7px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;overflow-wrap:anywhere}.mapping-grid th{background:#eef4f3}.mapping-grid td:nth-child(1){font-weight:600}.mapping-grid th:nth-child(1){width:34%}.mapping-grid th:nth-child(2){width:22%}.mapping-grid code{display:block;font-size:11px;font-weight:400;color:#66808a;overflow-wrap:anywhere}.mapping-new{grid-column:1/-1}.mapping-new th:nth-child(1){width:31%}.mapping-new th:nth-child(2){width:22%}
.group-intro{display:flex;justify-content:space-between;gap:20px;align-items:flex-end;margin-top:40px}.group-intro p{font-size:14px;color:var(--muted)}.house{padding:27px;margin-bottom:26px}.house-head{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}.house-id{font-size:13px;color:var(--muted);overflow-wrap:anywhere;margin-top:5px}.number{font-size:34px;color:#8aa2a8;min-width:45px;font-weight:600}.house-name{display:flex;gap:16px;align-items:flex-start;min-width:0}.house-name>div{min-width:0}.badges{display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end}.badge{white-space:nowrap;border-radius:20px;padding:5px 12px;font-size:13px;background:#edf1f3;font-weight:650}.badge.pass{color:#087c68;background:#ddf2e9}.badge.review{color:#946111;background:#fff0d3}.badge.fail{color:#ab3149;background:#ffe3e9}.badge.exposed{color:#745369;background:#f0e9ee}.verdict{margin:17px 0;padding:12px 15px;border-left:4px solid var(--teal);background:#f0f7f4;border-radius:0 7px 7px 0;font-size:17px}.house.review .verdict{border-color:#bf913c;background:#fff7e9}.house.fail .verdict{border-color:#cd5670;background:#fff1f3}.meta{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:18px 0 22px}.meta div{padding:11px 12px;border:1px solid var(--line);border-radius:9px}.meta dt{color:var(--muted);font-size:12px}.meta dd{margin:3px 0 0;font-size:16px;font-weight:650}.legend{display:flex;gap:8px 17px;flex-wrap:wrap;font-size:13px;margin:11px 0;color:var(--muted)}.legend i{display:inline-block;vertical-align:middle;margin-right:5px}.legend .scope{width:15px;height:12px;border:2px solid #10ad9a}.legend .listener{width:0;height:0;border-left:6px solid transparent;border-right:6px solid transparent;border-bottom:12px solid #20bcd1}.legend .source{width:10px;height:10px;background:#f5a52e;border-radius:50%}.legend .hole{width:8px;height:8px;border-radius:50%;background:#f43c59}.plans{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:17px}.plans.single{grid-template-columns:minmax(0,1fr);max-width:930px;margin:auto}.plans figure{min-width:0;margin:0}.plans img{background:#020405;width:100%;height:auto;display:block;border-radius:10px;aspect-ratio:1}.plans figcaption{font-size:13px;padding:8px 1px;line-height:1.7}.rooms-scope{margin:13px 0 21px;font-size:14px}.rooms-scope span{display:inline-block;padding:5px 9px;border-radius:6px;background:#eaf5f1;margin:4px 5px 0 0}.interiors{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:14px 0 24px}.interiors figure{margin:0;min-width:0}.interiors img{width:100%;height:auto;display:block;border-radius:8px;aspect-ratio:16/9}.interiors figcaption{font-size:13px;padding-top:6px;line-height:1.6}.metrics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.metric{background:#f8faf9;border:1px solid var(--line);border-radius:11px;padding:13px;min-width:0}.metric h4{font-size:16px}.metric .note{font-size:12px}.plot{display:block;width:100%;height:auto;font-family:inherit}.acoustic-notes{display:flex;flex-wrap:wrap;gap:7px;margin-top:13px}.acoustic-notes span{font-size:13px;background:#eef2f5;border-radius:5px;padding:5px 8px}.provenance{font-size:11px;color:#71858e;overflow-wrap:anywhere;margin-top:18px;border-top:1px solid var(--line);padding-top:11px}.missing{padding:16px;background:#fff2db;border:1px solid #ead8b5;border-radius:9px;color:#865c25}.footer-grid{display:grid;grid-template-columns:1.5fr 1fr;gap:24px;margin-top:22px}.summary-table{width:100%;border-collapse:collapse;font-size:14px}.summary-table th,.summary-table td{border-bottom:1px solid var(--line);padding:10px 8px;text-align:left}.todo{padding-left:22px;margin:15px 0 0}.todo li{margin:10px 0}.footer-note{padding:18px;background:#edf5f2;border-radius:12px;margin-top:20px;color:var(--muted);font-size:14px}a{color:var(--teal);text-decoration:none}code{font-family:ui-monospace,monospace;overflow-wrap:anywhere}.no-break{white-space:nowrap}
@media(max-width:1100px){.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.split-grid{grid-template-columns:repeat(3,minmax(0,1fr))}.split-box:first-child .house-list{grid-template-columns:1fr}}
@media(max-width:700px){body{font-size:15px}main{padding:24px 14px 35px}.panel,.house{padding:18px 15px;border-radius:13px}.hero{padding-left:14px}h2{font-size:23px}.stats{grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.stat{padding:12px}.flow{grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.step:nth-child(2)::after{display:none}.timelines,.split-grid,.mapping-grid,.taxonomy,.plans,.plans.single,.footer-grid{grid-template-columns:minmax(0,1fr)}.time-item:last-child{grid-column:auto}.rules{grid-template-columns:repeat(2,minmax(0,1fr))}.split-box:first-child .house-list{grid-template-columns:1fr}.mapping-new{grid-column:auto}.house-head{display:block}.badges{justify-content:flex-start;margin-top:12px}.number{font-size:26px;min-width:30px}.house-name{gap:9px}.meta{grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.meta div{padding:9px}.meta dd{font-size:15px}.interiors{grid-template-columns:minmax(0,1fr);gap:17px}.metrics{grid-template-columns:minmax(0,1fr)}.metric{padding:16px}.metric .note{font-size:13px}.group-intro{display:block}.mapping-grid table{font-size:12px}.mapping-grid code{font-size:10px}.mapping-grid th,.mapping-grid td{padding:8px 4px}.verdict{font-size:15px}.house-id{font-size:11px}.provenance{font-size:10px}.legend{font-size:12px}.class-pair{font-size:13px}.class-pair b{font-size:15px}}
'''


def icon(kind):
    shapes={'house':'<path d="M4 21 22 5l18 16M8 18v21h28V18M18 39V27h9v12"/>',
            'scene':'<rect x="4" y="7" width="35" height="29" rx="3"/><path d="m17 15 12 7-12 7zM9 39h25"/>',
            'window':'<path d="M5 12h34M5 32h34M13 6v32M31 6v32M18 19h8M18 25h8"/>',
            'qa':'<rect x="5" y="5" width="34" height="34" rx="4"/><path d="M17 16c0-8 13-8 13 0 0 5-7 4-7 9M23 30v3"/>'}
    return '<svg viewBox="0 0 44 44" aria-hidden="true">'+shapes[kind]+'</svg>'


def timeline(title,subtitle,bars,quota=''):
    s='<div class="time-item"><h4>'+title+'</h4><p class="note">'+subtitle+'</p><div class="axis">'
    for start,end,kind,label in bars:
        s+=f'<span class="bar {kind}" style="left:{start*10}%;width:{(end-start)*10}%">{label}</span>'
    return s+'</div><div class="ticks"><span>0 秒</span><span>5 秒</span><span>10 秒</span></div><div class="quota">'+quota+'</div></div>'


def spec_overview(recommendations,mapping):
    s='<section class="panel"><div class="kicker">01 / 规格总览</div><h2>从房屋到题目，中间发生什么</h2><div class="flow">'
    steps=[('house','物理房屋','12 套推荐 · 整屋划分'),('scene','30 / 60 秒连续场景','规格默认 60 秒；30 秒变体待验证'),
           ('window','随机切 10 秒窗口','保留窗前声活动与混响历史'),('qa','沿用 v6 题型','按 STARSS v0.2 界定活动段')]
    for kind,title,sub in steps:s+='<div class="step">'+icon(kind)+'<b>'+title+'</b><small>'+sub+'</small></div>'
    s+='</div><p class="note">设计目标：3000 个独立场景／主窗口，约 4050 个反事实成员窗口。当前已完成规格和房屋诊断，尚未完成这批数据生产；同场景多题不计成更多独立样本。</p>'
    s+='<div class="timelines">'
    s+=timeline('持续声穿过片头','声音在截窗前已开始；片头不重置事件。',[(0,10,'','')],'目标 ≥40% 窗口有片头持续声')
    s+=timeline('移动、发声与停顿','条纹表示移动中发声；空隙是停声，停步另有状态。',[(0,3,'move',''),(4.1,7.2,'move',''),(7.5,9.8,'','')],'目标 ≥40% 窗口含移动发声')
    s+=timeline('同一人，多次发声','相同颜色与同一身份；每个不连续活动段分别计数。',[(.4,2.2,'',''),(3.4,5.6,'',''),(7.2,9.4,'','')],'目标 ≥50% 窗口同人至少 2 段')
    s+=timeline('多个声源重叠','上下两行是两个声源；重叠段不合并为一个源。',[(.6,6.1,'',''),(4,9.3,'alt','')],'目标 ≥40% 窗口有至少 0.5 秒重叠')
    s+=timeline('超过 6 个先后事件','示意 8 个活动段；顺序事件数与同时活动源数分开。',[(i*1.25+.06,i*1.25+1.1,'',str(i+1)) for i in range(8)],'目标 ≥25% 窗口有 7–12 段；≥5% 压力窗有 21–30 段')
    s+='</div><p class="note" style="margin-top:12px">以上时间轴是设计示意，配额可重叠；完整时间线以 10 Hz 活动标注切分，缺一帧即分段，混响尾声不另增事件。</p></section>'
    s+='<section class="panel"><h2>推荐的 12 套，按整屋留出</h2><p class="note">屋数 8／2／2（66.7%／16.7%／16.7%）；独立窗口目标另为 2100／450／450。开发、测试来自干净的原 HM3D val。</p><div class="split-grid">'
    for split in ['train','development','test']:
        houses=[x for x in recommendations['houses'] if x['split']==split]
        s+=f'<div class="split-box"><h3>{SPLIT[split]} · {len(houses)} 套</h3><div class="house-list">'
        for house in houses:s+='<span>'+esc(house['house'])+'</span>'
        s+='</div></div>'
    s+='</div></section>'
    s+='<section class="panel"><h2>验收判据，一眼看懂</h2><div class="rules">'
    rules=[('漏声','≤5% / >15%','最差射线原点 ≤5% 通过，>15% 不通过，中间待审。'),
           ('混响时间','0.2–1.5 秒','屋中频 RT60 以此为主，少量 1.5–2.5 秒长尾。'),
           ('FOA 方向','≤15°','固定 0.5 ms 直达窗：≤15° 通过，15–30° 待审，>30° 不通过。'),
           ('距离衰减','p = 0.7–1.3','距离跨度 ≥1.8 倍；压力衰减指数异常或缺数据则待审。'),
           ('RIR 覆盖','≥6 / ≥70%','至少 6 个 RIR、至少 2 个听者位置，≥70% 有有效中频 RT。'),
           ('尾部与直达','4 秒 / 3 ms','达 4 秒且尾部仍明显，或直达时延偏差 >3 ms，需待审。')]
    for name,value,rule in rules:s+='<div class="rule"><h4>'+name+'</h4><strong>'+value+'</strong><p>'+rule+'</p></div>'
    s+='</div><p class="note" style="margin-top:14px">DRR 是描述性指标。通过是本轮声学与 CPU 投影诊断通过，不能替代正式 Episode、人物或视频验收；本页的 Habitat RGB 是新增静态预览。</p></section>'
    s+='<section class="panel"><h2>35 个 v6 类，如何对应 STARSS 13 类</h2><p class="note">按物理发声机制与内容对应，不能只按声音名字强行合并；新增 11 类后，v7 拟定词表共 46 类。</p><div class="taxonomy">'
    s+='<div class="class-pair"><div><b>现场人声</b>画面中人体发声<br>human_speech_female / male</div><div class="arrow">→</div><div><b>STARSS 0 / 1</b>计为现场讲话者<br>同人再次讲话仍是同一个人</div></div>'
    s+='<div class="class-pair playback"><div><b>播放人声</b>扬声器发声<br>speech_playback</div><div class="arrow">→</div><div><b>不映射现场讲话</b>保留 v6 扩展类<br>不计为画面中的现场说话人</div></div>'
    s+='</div><div class="native-classes">'+''.join(f'<span>{i} · {name}</span>' for i,name in enumerate(STARSS))+'</div>'
    s+='<div class="mapping-grid">'
    classes=mapping['v6_classes']
    for group,title in [(classes[:18],'v6 原有类 · 1–18'),(classes[18:],'v6 原有类 · 19–35'),(mapping['new_classes'],'补齐原生类别的 11 个新增类（设计，素材尚未就绪）')]:
        new=group is mapping['new_classes']
        s+='<div'+(' class="mapping-new"' if new else '')+'><h4>'+title+'</h4><table><thead><tr><th>v6 / v7 类</th><th>STARSS 对应</th><th>对应条件</th></tr></thead><tbody>'
        for item in group:
            name=item['sound_class'];index=item['starss_class_id']
            target='— 不对应' if index is None else f'{index} · {STARSS[index]}'
            s+='<tr><td>'+esc(NAMES.get(name,name))+'<code>'+esc(name)+'</code></td><td>'+target+'</td><td>'+esc(item['rule'])+'</td></tr>'
        s+='</tbody></table></div>'
    s+='</div><p class="note" style="margin-top:14px">旧声音池有 35 类，QA-21 固定答案域有 34 类：复合占位 any_audioset_class_playback 不在旧固定域内；新增原生 13 类面板单独统计。来源：V7_SPEC_zh.md §3 与 examples/acoustics/v7_class_mapping.json。</p></section>'
    return s


def verdict(result):
    reasons=result.get('reasons',[])
    leak=result.get('max_escape_fraction')
    if result['status']=='pass':
        return '漏声、混响、距离衰减与方向检查满足本轮判据，纳入首批推荐。'
    if 'GEOMETRIC_ESCAPE_ABOVE_15_PERCENT' in reasons:
        return f'漏声不通过：最差位置 {leak*100:.2f}%，超过 15% 不通过线。'
    if any('zero-area triangle' in r for r in reasons) and not result.get('rirs'):
        return '声学输入含退化面，RLR 拒绝加载；漏声、RIR 和 FOA 未完成。'
    if 'DEGENERATE_INPUT_GEOMETRY_DIAGNOSTIC_ONLY' in reasons:
        return '仅有省略退化面的诊断结果，原始几何仍待审；不能据此准入生产。'
    if 'GEOMETRIC_ESCAPE_5_TO_15_PERCENT' in reasons:
        return f'漏声待审：最差位置 {leak*100:.2f}%，处于 5%–15% 之间。'
    if 'DIRECT_DISTANCE_ATTENUATION_OUTLIER' in reasons:
        bad=[x for x in result.get('distance_attenuation',[]) if x.get('pressure_exponent') is not None and not .7<=x['pressure_exponent']<=1.3]
        groups=charts.distance_series(result)
        for fit in bad:
            points=groups.get(fit['room_label'],[])
            for a,b in zip(points,points[1:]):
                if b['direct_energy']>a['direct_energy']:
                    return f"距离衰减异常：{fit['room_label']} 中 {b['distance_m']:.2f} 米处比 {a['distance_m']:.2f} 米处更响，需复核。"
        if bad:
            fit=bad[0]
            return f"距离衰减异常：{fit['room_label']} 的压力指数 p={fit['pressure_exponent']:.2f}，超出 0.7–1.3。"
        return '距离衰减异常或有效距离跨度不足，需要复核实际直达路径。'
    if 'RAW_WITNESS_VS_ACOUSTIC_DIRECT_PATH_DISAGREEMENT' in reasons:
        return '原始视觉网格与声学网格的直达路径不一致，相关直达指标仍待审。'
    return '部分验收检查缺失或存在软异常，需要复核后再决定。'


def reason_tags(result):
    translations={'DIRECT_DISTANCE_ATTENUATION_OUTLIER':'距离衰减异常',
    'GEOMETRIC_ESCAPE_5_TO_15_PERCENT':'漏声 5–15%', 'GEOMETRIC_ESCAPE_ABOVE_15_PERCENT':'漏声 >15%',
    'RAW_WITNESS_VS_ACOUSTIC_DIRECT_PATH_DISAGREEMENT':'视觉／声学直达路径不一致',
    'DEGENERATE_INPUT_GEOMETRY_DIAGNOSTIC_ONLY':'仅退化面诊断', 'MULTIPLE_LISTENER_POSITIONS_MISSING':'多听者检查缺失',
    'CPU_VISUAL_CHECK_PENDING':'原 CPU 投影未完成', 'VISUAL_CHECK_MISSING':'原 CPU 视图缺失'}
    return [translations.get(r,'原始网格退化面待审' if 'zero-area triangle' in r else r) for r in result.get('reasons',[])]


def picture(path,caption,output,kind):
    relative=Path(path).relative_to(output/'page').as_posix()
    alt=re.sub(r'<[^>]*>', ' ', caption)
    return f'<figure><img src="{esc(relative,quote=True)}" loading="lazy" decoding="async" width="{1536 if kind=="top" else 1280}" height="{1536 if kind=="top" else 720}" alt="{esc(alt,quote=True)}"><figcaption>{caption}</figcaption></figure>'


def house_card(job,result,native,index,split,output):
    house=job['house'];name=house['house'];status=result['status']
    scan=house.get('scan_id',name.split('_')[-1])
    s=f'<article class="house {status}" id="{name}" data-status="{status}" data-house="{name}"><div class="house-head"><div class="house-name"><span class="number">{index:02d}</span><div><h3>{esc(scan)}</h3><div class="house-id">{esc(name)}</div></div></div><div class="badges">'
    s+=f'<span class="badge {status}">{STATUS[status]}'+(' · 首批推荐' if split else '')+'</span>'
    s+='<span class="badge">'+(SPLIT.get(split,'未纳入划分'))+'</span>'
    if house.get('known_exposure'):s+='<span class="badge exposed">已知 SO 暴露 · 仅诊断</span>'
    s+='</div></div><p class="verdict">'+verdict(result)+'</p>'
    regular=[x for x in job['levels'] if x['kind']=='registry_level']
    story=(f"原生层级 {len(regular)}" if house['family']=='mp3d' else f"登记高度层 {len(regular)}")
    room_count=house['candidate_room_count'];checked=len(house['acoustic_rooms'])
    s+='<dl class="meta">'+''.join('<div><dt>'+label+'</dt><dd>'+value+'</dd></div>' for label,value in [
       ('候选房间面积和',f"{house['candidate_area_sum_m2']:.1f} m²"),('房间',f'{room_count} 候选 / {checked} 检查'),
       ('楼层信息',story),('来源 / 划分',house['family'].upper()+' / '+SPLIT.get(split,'未分配'))])+'</dl>'
    s+='<p class="note">面积不是建筑总面积；灰线为登记范围，绿线仅表示本轮检查房间。'+(f"原规格楼层代理为 {house.get('floor_count_proxy','未测')}；高度层与补充切面不等于已核实的建筑楼层数。" if house['family']=='hm3d' else '楼层读取 MP3D 登记原生层级；仅部分房间做了声学检查。')+'</p>'
    s+='<h4 style="margin-top:20px">带原始纹理的分层俯视图</h4><div class="legend"><span><i class="scope"></i>检查范围</span><span><i class="listener"></i>L 听者／相机</span><span><i class="source"></i>S1 / S2 源位；小点为近中距离源位</span><span><i class="hole"></i>漏声候选边界位置</span></div>'
    if native and native.get('status')=='complete':
        tops=native['topdowns'];s+='<div class="plans'+(' single' if len(tops)==1 else '')+'">'
        for top in tops:
            cal=top['calibration'];rooms=', '.join(top['acoustic_room_labels']) or '本层未做声学检查'
            caption=(f"{'补充高度切面' if top['kind']=='witness_height' else '登记层'} {top['level_id']+1} · 地面 y={cal['floor_y_m']:.2f} m · 检查 {rooms} · 红点 {top['hole_candidate_count']} 个")
            s+=picture(top['path'],caption,output,'top')
        s+='</div>'
    else:s+='<div class="missing">Habitat 原生俯视图尚未生成；详见本任务逐屋渲染日志。</div>'
    s+='<p class="note">红点是逃逸射线附近原始网格边界的候选位置投影，可能是合法开口或上下边界；不代表已确认破洞。无红点时也须结合漏声比例和是否完成射线检查。</p>'
    s+='<div class="rooms-scope">本轮检查：'+''.join('<span>'+esc(r['room_label'])+' · '+esc(ROOM.get(r['room_type'],r['room_type']))+f" · {r['floor_area_m2']:.1f} m²</span>" for r in house['acoustic_rooms'])+'</div>'
    s+='<h4>在验收听者位置看房间 · Habitat RGB · 85°</h4><p class="note">橙色圈标注发声位置；虚线圈表示中心被遮挡。这些标记是测量点投影，未在扫描房间中添加人物或设备。</p>'
    if native and native.get('status')=='complete':
        s+='<div class="interiors">'
        for i,pic in enumerate(native['interiors'],1):
            cal=pic['calibration']
            planned=cal['camera_role']=='planned_listener_acoustics_not_completed'
            coordinates=', '.join(f'{v:.2f}' for v in cal['camera_m'])
            markers=[]
            for m in pic['markers']:
                markers.append('S'+str(m['source_index'])+(' 不在此视野' if not m['inside_frustum'] else ' 中心被遮挡' if not m['center_depth_visible'] else ''))
            caption=f"视角 {i} · {cal['room_label']} · {'计划验收听者，声学未完成' if planned else '验收听者'} ({coordinates}) m<br>"+'；'.join(markers)
            s+=picture(pic['path'],caption,output,'rgb')
        s+='</div>'
    else:s+='<div class="missing">原生内景尚未生成，未用 CPU 点云代替。</div>'
    s+='<div class="metrics">'
    for title,chart,note in [('分频带 RT60',charts.rt60(result),'柱：有效位置中位数；细线：最小–最大；优先 T30，否则 T20。'),
      ('直达 / 混响比 DRR',charts.drr(result),'点：房间中位数；线：所有该房间位置的范围。'),
      ('距离衰减',charts.attenuation(result),'实线：近、中、远位置；虚线：1/r 理想参考，各房间分别归一化。'),
      ('漏声比例',charts.leakage(result),'原声学网格，每原点 512 条球面射线；不由本页纹理图估计。')]:
        s+='<div class="metric"><h4>'+title+'</h4>'+chart+'<p class="note">'+note+'</p></div>'
    s+='</div><div class="acoustic-notes">'
    s+='<span>中频 RT60 '+charts.number(result.get('rt60_mid_median_s'),3)+' s</span>'
    s+='<span>FOA 最大误差 '+charts.number(result.get('max_foa_angle_deg'),2)+'°</span>'
    s+='<span>RIR '+str(len(result.get('rirs',[])))+' 个</span>'
    for tag in dict.fromkeys(reason_tags(result)):s+='<span>'+esc(tag)+'</span>'
    s+='</div>'
    if 'RAW_WITNESS_VS_ACOUSTIC_DIRECT_PATH_DISAGREEMENT' in result.get('reasons',[]):
        s+='<p class="note">直达路径不一致，因此 DRR／距离曲线仅供诊断，不能作为有效直达见证。</p>'
    s+='<div class="provenance">数值：'+esc(job['input_paths']['result'])+'<br>结论：'+esc(str(Path(job['input_paths']['manifest']).parent/'report_final/all_house_results.json'))+'<br>范围：'+esc(job['input_paths']['registry'])+'；漏声候选：'+esc(job['input_paths']['boundary'])+'</div></article>'
    return s


def footer(results,summary,rendered):
    passed=[x for x in results if x['status']=='pass']
    valid=[x['rt60_mid_median_s'] for x in passed if x.get('rt60_mid_median_s') is not None]
    s='<section class="panel"><div class="kicker">03 / 覆盖与缺口</div><h2>12 套通过房屋，有没有覆盖够宽</h2><div class="footer-grid"><div>'+charts.distribution(results)
    s+=f'<p class="note">每屋一票，取有效 RIR 的中频屋中位数；实测 {min(valid):.3f}–{max(valid):.3f} 秒。1.5–2.5 秒的长尾屋为 0，宽覆盖目标未满足。</p></div><div><h4>状态汇总</h4><table class="summary-table"><tr><th>来源</th><th>通过</th><th>待审</th><th>不通过</th></tr>'
    for family,label in [('hm3d','HM3D'),('mp3d','MP3D'),(None,'合计')]:
        counts=Counter(x['status'] for x in results if family is None or x['family']==family)
        s+='<tr><td>'+label+'</td>'+''.join('<td>'+str(counts[k])+'</td>' for k in ['pass','review','fail'])+'</tr>'
    s+=f'</table><p class="note" style="margin-top:13px">原验收：{summary["rirs"]} 个 RIR，{summary["baseline_ray_count"]:,} 条球面射线；27 套完成 RIR。<br>本页原生预览：{len(rendered)} 套，'+str(sum(len(x['topdowns']) for x in rendered.values()))+' 张俯视图，'+str(sum(len(x['interiors']) for x in rendered.values()))+' 张内景。</p></div></div>'
    s+='<h3 style="margin-top:25px">还缺什么</h3><ul class="todo">'
    missing=['1.5–2.5 秒长尾房屋为 0；开发、测试的声学覆盖也偏窄，下一批需补结构与材质多样性。',
             'MP3D 1LXtFkjw3qL 的声学输入仍未被 RLR 接受，漏声、RIR、FOA 都未测；本页可看原始扫描，不能补称声学通过。',
             '待审房屋的漏声、距离衰减、退化面或视觉／声学直达路径不一致，仍需逐项复核；新增 RGB 预览不改变旧判定。',
             '6 套 MP3D 均有已知 SO 暴露，不能纳入这批干净划分；匿名历史场景及跨数据集物理别名仍未完全核实。',
             '面积和楼层仍是登记范围或代理；未检查房间不能随整屋结论获得验收资格，扫描合法开口与真正破洞也需人工区分。',
             'v7 连续场景生成、时变卷积、人物与干声资产许可、反事实绑定、v0.2 出题适配、事件容量和完整视频验收仍需接入；3000 窗口未生产。']
    if len(rendered)<28:missing.insert(0,f'本页仍有 {28-len(rendered)} 套未完成原生渲染，见逐屋日志。')
    s+=''.join('<li>'+item+'</li>' for item in missing)+'</ul>'
    s+='<div class="footer-note">用途：供 owner 先看规格、房间和验收问题，再决定是否交给 smy 审。本页未发送给他人，未推送分支。数据依据：v7_batch1_20261003/V7_SPEC_zh.md、report_final/summary.json、all_house_results.json、recommended_houses.json；本页不重算或改写验收阈值。</div></section>'
    return s


def build(input_root,output):
    input_root,output=Path(input_root),Path(output)
    summary=read_json(input_root/'report_final/summary.json')
    recommendations=read_json(input_root/'report_final/recommended_houses.json')
    results=read_json(input_root/'report_final/all_house_results.json')
    mapping=read_json(Path(__file__).resolve().parents[3]/'examples/acoustics/v7_class_mapping.json')
    by_house={x['house']:x for x in results}
    recommended={x['house']:x['split'] for x in recommendations['houses']}
    ordering=[x['house'] for x in recommendations['houses']]
    ordering+=sorted(x['house'] for x in results if x['house'] not in recommended and x['status']=='review')
    ordering+=sorted(x['house'] for x in results if x['house'] not in recommended and x['status']=='fail')
    ordering+=sorted(x['house'] for x in results if x['house'] not in ordering)
    rendered={}
    for path in (output/'rendered_houses').glob('*.json'):
        item=read_json(path)
        if item.get('status')=='complete':rendered[item['house']]=item
    s='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v7 房屋规格与声学 · owner 可视化审阅</title><style>'+CSS+'</style></head><body><main>'
    s+='<header class="hero"><div class="kicker">HARDBANK V7 / 房屋规格与声学</div><h1>先看清房屋，再决定送审</h1><p class="muted">把时间结构、整屋划分、真实扫描内景与声学结论放在同一页。28 套全部平铺，推荐的 12 套排在最前；图片按需懒加载，离线打开即可。</p></header>'
    counts=Counter(x['status'] for x in results)
    s+='<div class="stats">'+''.join(f'<div class="stat {kind}"><strong>{value}</strong><span>{label}</span></div>' for kind,value,label in [('',len(results),'候选房屋'),('pass',counts['pass'],'通过 · 首批推荐'),('review',counts['review'],'待审 · 不凑入名单'),('fail',counts['fail'],'不通过')])+'</div>'
    s+=spec_overview(recommendations,mapping)
    groups=[('pass','02 / 推荐房屋','12 套诊断通过 · 训练 8 / 开发 2 / 测试 2'),
            ('review','待审房屋','12 套 · 问题与缺失检查均保留'),('fail','不通过房屋','4 套 · 漏声超过 15%')]
    index=0
    evidence=[]
    for status,title,sub in groups:
        s+='<div class="group-intro"><div><div class="kicker">逐屋审阅</div><h2>'+title+'</h2></div><p>'+sub+'</p></div>'
        for name in ordering:
            result=by_house[name]
            if result['status']!=status:continue
            index+=1
            job=read_json(output/'jobs'/(name+'.json'))
            s+=house_card(job,result,rendered.get(name),index,recommended.get(name),output)
            evidence.append(dict(house=name,status=status,split=recommended.get(name),
                                 band_rt60=charts.band_stats(result),source_paths=job['input_paths']))
    s+=footer(results,summary,rendered)
    at=now()
    s+='<p class="note">生成时间（新加坡）：'+at+' · 图例中的测量点来自原验收记录。所有数值来源已列在各卡底部。</p></main></body></html>'
    target=output/'page/index.html'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(s)
    references=sorted(set(re.findall(r'<img[^>]+src="([^"]+)"',s)))
    total_bytes=target.stat().st_size
    archive=output/'v7_room_viz_20261003.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.write(target,'index.html')
        for ref in references:
            path=target.parent/ref
            if not path.is_file():raise FileNotFoundError(path)
            z.write(path,ref)
            total_bytes+=path.stat().st_size
    write_json(output/'evidence/page_data.json',dict(at_sgt=at,houses=evidence))
    write_json(output/'evidence/page_build.json',dict(at_sgt=at,page=str(target),zip=str(archive),
               zip_bytes=archive.stat().st_size,referenced_images=len(references),uncompressed_bytes=total_bytes,
               houses=index,native_houses=len(rendered),topdowns=sum(len(x['topdowns']) for x in rendered.values()),
               interiors=sum(len(x['interiors']) for x in rendered.values()),status_counts=dict(counts),
               source_summary=str(input_root/'report_final/summary.json'),references=references))
    print('PAGE_BUILT',target,'ZIP_BYTES',archive.stat().st_size,'IMAGES',len(references),flush=True)
