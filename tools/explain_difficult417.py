"""Visual explanation only: frozen forecasts against native cached ON/OFF.

No training, inference changes, reference correction or source selection for
scoring. Crops are display-only; every metric remains the original410 score.
"""
from pathlib import Path
import sys,json,hashlib,html
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from pure_visibility170 import window
from extend_inverse321 import references
from physical_publication import evaluate

OUT=WORK/'output/difficult_clips_20261110';OUT.mkdir(exist_ok=True)
FONT='C:/Windows/Fonts/msyh.ttc'
def font(size):return ImageFont.truetype(FONT,size)
CACHE={'3':WORK/'output/frozen_tangent287_20261108/cache','4':WORK/'output/frozen324_20261109/cache',
       '16':WORK/'output/phase185_20261107/cache_recording16','13':WORK/'output/phase185_20261107/cache_recording13'}
ROWS={n:[json.loads(s) for s in (WORK/f'output/combined410_20261110/{n}_joint.jsonl').read_text('utf8').splitlines()] for n in CACHE}
REFS={n:json.loads((WORK/'output/reference323_20261109/references.json' if n=='4' else references(n)).read_text('utf8'))['records'] for n in CACHE}
SCORES=json.loads((WORK/'output/combined410_20261110/results.json').read_text('utf8'))
COL=dict(pred='#ce3434',actual='#087e53',body='#235bc9',trail='#797e89')


def old(n,t):return max((r for r in ROWS[n] if r['t']<=t+1e-9),key=lambda r:r['t'])
def ref(n,t):return REFS[n].get(f'{t:.1f}')
def center(q):return (np.asarray(q['box'][:2])+q['box'][2:])/2 if q and q.get('box') else None


def native(n,t,us=20000):
    e=window(CACHE[n],t);e=e[e['t']>=round(t*1e6)-us]
    counts=np.bincount(e['y'].astype(np.int64)*1280+e['x'],minlength=1280*720).reshape(720,1280)
    # Both polarities counted, fixed display mapping. NOT G reconstruction or
    # a new detection threshold; no raw event is removed by this renderer.
    return Image.fromarray(np.uint8(255-np.minimum(counts*20,255))).convert('RGB')


def cross(d,p,color,size=11,width=3):
    x,y=map(float,p);d.line((x-size,y-size,x+size,y+size),fill=color,width=width);d.line((x-size,y+size,x+size,y-size),fill=color,width=width)


def panel(n,t,roi,scale=2,forecast=None,origin=None):
    im=native(n,t).crop(roi).resize(((roi[2]-roi[0])*scale,(roi[3]-roi[1])*scale),Image.Resampling.NEAREST)
    d=ImageDraw.Draw(im)
    def pt(p):return tuple((np.asarray(p)-roi[:2])*scale)
    q=ref(n,t);c=center(q)
    trail=[center(q) for q in REFS[n].values() if origin is not None and origin-.4-1e-5<=q['t']<=t+1e-5 and q.get('box')]
    if len(trail)>1:d.line([pt(p) for p in trail],fill=COL['trail'],width=2)
    if q and q.get('box'):
        b=q['box'];d.rectangle((*pt(b[:2]),*pt(b[2:])),outline=COL['actual'],width=3)
        x,y=pt(c);d.ellipse((x-5,y-5,x+5,y+5),fill=COL['actual'])
    r=old(n,t)
    if r['body'].get('bbox') and (origin is None or t<origin+.4-1e-5):
        b=r['body']['bbox'];d.rectangle((*pt(b[:2]),*pt(b[2:])),outline=COL['body'],width=2)
    if forecast is not None:
        p=pt(forecast);cross(d,p,COL['pred'])
        if origin is not None and abs(t-origin-.4)<1e-5 and c is not None:d.line((p,pt(c)),fill=COL['pred'],width=3)
    return im


def case_card(n,t,roi,title,name):
    scored=next(q for q in SCORES['cases'][n]['joint']['records'] if q['valid'] and abs(q['t']-t)<1e-6)
    r=old(n,t);future=np.asarray(scored['prediction']);w=(roi[2]-roi[0])*2;h=(roi[3]-roi[1])*2
    im=Image.new('RGB',(3*w+40,h+170),'#f5f7fb');d=ImageDraw.Draw(im)
    d.text((20,7),title,font=font(30),fill='#172b44')
    for i,stamp in enumerate([t,round(t+.2,1),round(t+.4,1)]):
        x=10+i*(w+10);d.text((x,54),f'{stamp:.1f}s'+('  作出预测' if i==0 else '  后续原生画面'),font=font(24),fill='#26374e')
        im.paste(panel(n,stamp,roi,forecast=future if i in (0,2) else None,origin=t),(x,90))
    y=h+105;d.text((16,y),'绿框/绿点：该帧弱参考   蓝框：起点/中间算法区域；终点仅显示参考与冻结预测',font=font(22),fill='#21334c')
    d.text((16,y+33),f'红叉：始终冻结在{t:.1f}s作出的{t+.4:.1f}s预测；终点误差 {scored["error_px"]:.2f}px（原始像素）',font=font(22),fill=COL['pred'])
    im.save(OUT/f'{name}.png')
    current=evaluate(r,[t-r['t']])[str(float(t-r['t']))];now=center(ref(n,t))
    result=dict(recording=n,origin=t,target_t=round(t+.4,1),native_display_accumulation_us=20000,
        crop_xyxy=roi,crop_for_display_only=True,error_px=scored['error_px'],prediction=future.tolist(),actual=scored['target'],
        now_output_error_px=None if now is None else float(np.linalg.norm(np.asarray(current)-now)),
        internal_material_reference_error_px=scored.get('current_position_error_px'),current_body_detected=r['body']['detected'],
        current_motion_observation=r['motion'].get('current_observation'),counts=r['counts'],structure_mode=r['structure']['mode'],
        view_evidence=r['structure'].get('view_evidence'),velocity=r['coefficient']['velocity'],acceleration=r['coefficient']['acceleration'])
    return result


def gif(n,start,end,roi,name):
    # Cached source frames retain33.333ms spacing. Playback deliberately slowed
    # to120ms/frame; window stays20ms and no frame is interpolated.
    stamps=[r['t'] for i,r in enumerate(ROWS[n]) if start-1e-5<=r['t']<=end+1e-5 and (i==0 or r['t']-ROWS[n][i-1]['t']>.001)];frames=[];scale=2
    for t in stamps:
        raw=native(n,t).crop(roi).resize(((roi[2]-roi[0])*scale,(roi[3]-roi[1])*scale),Image.Resampling.NEAREST)
        im=Image.new('RGB',(raw.width,raw.height+50),'#eef2f8');im.paste(raw,(0,50));r=old(n,t);d=ImageDraw.Draw(im)
        d.text((8,8),f'{n}号 {t:.3f}s  '+('当前已检测' if r['body'].get('detected') else '当前未检测'),font=font(23),fill='#235bc9' if r['body'].get('detected') else '#bd3232')
        frames.append(im)
    frames[0].save(OUT/f'{name}.gif',save_all=True,append_images=frames[1:],duration=120,loop=0,optimize=False)


if __name__=='__main__':
    evidence=[]
    for spec in [('3',15.6,[880,15,1280,155],'① 桨盘清楚，但0.4秒内发生反向：147.27px','case3_turn'),
                 ('4',12.,[840,355,1210,475],'② 侧线已被检测，未来向左位移仍被低估：116.80px','case4_rollout'),
                 ('4',13.2,[850,380,1210,495],'③ 画面有目标，当前观测没有进入算法：117.78px','case4_gap'),
                 ('16',14.4,[350,0,800,190],'④ 目标贴近上边界、算法框触顶：62.67px','case16_border')]:
        evidence.append(case_card(*spec));print('card',spec[-1],flush=True)
    for n,t in [('3',15.6),('4',13.2)]:
        im=native(n,t);d=ImageDraw.Draw(im);q=ref(n,t)
        if q and q.get('box'):d.rectangle(q['box'],outline=COL['actual'],width=3)
        im.save(OUT/f'full_{n}_{t:.1f}.png')
    gif('3',15.2,16.,[880,15,1280,155],'recording3_turn_slow')
    gif('4',11.6,13.8,[800,350,1280,500],'recording4_side_slow')
    # Motion plot: observed references and predictions, never future inputs.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    plt.rcParams['font.family']=FontProperties(fname=FONT).get_name();plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,n,ts,origins in [(axs[0],'3',(14.6,16.),[15.6]),(axs[1],'4',(11.6,13.8),[12.,13.2])]:
        qs=sorted([q for q in REFS[n].values() if ts[0]<=q['t']<=ts[1] and q.get('box')],key=lambda q:q['t'])
        ax.plot([q['t'] for q in qs],[center(q)[0] for q in qs],'o-',color=COL['actual'],label='实际弱参考x位置')
        for t in origins:
            r=old(n,t);hh=np.linspace(t-r['t'],t-r['t']+.4,25);pred=evaluate(r,hh)
            ax.plot(r['t']+hh,[pred[str(float(h))][0] for h in hh],'--',color=COL['pred'],label=f'{t:.1f}s冻结预测')
            ax.axvline(t,color='#a4a9b3',lw=.8)
        ax.set(title=f'录像{n}：横向位置随时间',xlabel='时间（秒）',ylabel='原图x坐标（px，越小越靠左）');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(OUT/'motion_traces.png',dpi=150);plt.close(fig)
    r3=[q for q in SCORES['cases']['3']['joint']['records'] if q['valid']];other=[q['error_px'] for q in r3 if q['t']!=15.6]
    diagnostics=dict(cases=evidence,recording3_all5_errors=[dict(t=q['t'],error_px=q['error_px']) for q in r3],
        recording3_all5_mean=SCORES['cases']['3']['joint']['error']['mean'],recording3_other4_mean_for_diagnosis_only=float(np.mean(other)),
        official_scores_origins_and_references_unchanged=True,display_grayscale_is_event_count_not_G_reconstruction=True,
        no_new_forecast_or_training=True,GT_is_weak_native_visible_envelope_not_independent_COM=True,
        input_sha256={f'{n}_joint.jsonl':hashlib.sha256((WORK/f'output/combined410_20261110/{n}_joint.jsonl').read_bytes()).hexdigest() for n in CACHE})
    (OUT/'evidence.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2),'utf8')
    cards=[('录像3：清晰画面下的图像运动变向','case3_turn','录像3不是“全程画质差”。在15.6s，当前轮廓中心定位只差约'+f'{evidence[0]["now_output_error_px"]:.1f}'+'px，但模型将图像内向左运动延续到了16.0s。弱参考x坐标15.6→15.8→16.0为1066.5→1034.5→1067.5，表明预测窗内图像位置先左移、再右移；模型终点却在x=921.2。不能仅据此断言变向绝对不可预测：桨盘姿态/周期是否已提前发出信号，还需另行验证。也尚未独立分离相机自身运动，因此不把图像反向直接说成世界坐标的飞行反向。'),
           ('录像4：不是没找到，而是延续不足','case4_rollout','12.0s当前检测和位移观测都有效。状态估计vx≈−182px/s、ax≈+759px/s²，即认为向左运动正在减速；0.4秒预测x≈1072，实际弱参考x≈955.5，向左位移明显不足。它是运动延续失败；现有0.2秒弱参考不足以唯一判断起点加速度偏差和随后重新加速各占多少，不能全部归咎侧视看不清。'),
           ('录像4：真有当前观测断档','case4_gap','13.2s原生画面能看到无人机，但保存的body.detected=false、motion.current_observation=false。该包有约90.6万事件、14898个密度像素，却只有7个通过严格周期筛选的像素；这是全画面计数，不等于7个桨叶像素。当前没有确认目标，算法沿短缺测状态继续预测；周围多帧也间歇失去当前观测。此处应查周期/支撑筛选和续接，而不能说画面本来就没信息。13.6s的弱参考还受树枝影响，但终点横向仍差约114.6px，不能全用标注问题解释。'),
           ('录像16：边界造成信息不完整','case16_border','14.4s算法框为[393,0,589,77]，触到画面上边界；目标轮廓与视角在变化。14.8s预测终点误差62.67px。边界会增加参考点/轮廓的不确定性，但不是大误差的充分条件：13号13.4s同样靠近上边界，该起点只错1.95px。')]
    articles=''.join(f'<section><h2>{title}</h2><p>{text}</p><a href="{name}.png"><img src="{name}.png" alt="{title}"></a></section>' for title,name,text in cards)
    t3=''.join(f'<tr><td>{q["t"]:.1f}s</td><td>{q["error_px"]:.2f}px</td></tr>' for q in r3)
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>哪些片段困难：原生画面与预测逐项复盘</title><style>body{{font:17px/1.8 system-ui,"Microsoft YaHei";max-width:1380px;margin:30px auto;padding:0 22px;background:#f3f6fb;color:#22344b}}section{{background:white;padding:25px;margin:20px 0;border-radius:14px}}img{{max-width:100%;height:auto}}a{{color:#176b8b}}table{{border-collapse:collapse}}td,th{{padding:7px 25px;border-bottom:1px solid #dbe2eb;text-align:right}}.note{{border-left:5px solid #c4493c;background:#fff4ee;padding:18px}}.two{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}@media(max-width:800px){{.two{{grid-template-columns:1fr}}}}</style>
    <h1>“困难”是当前算法表现，不是给视频判坏</h1><p>2026-11-10 · 对照冻结410候选 · 仅复盘，没有重训、改检测阈值、改参考或删除起点</p>
    <div class="note">此前把3/4号概括成困难片段过于笼统。3号主要被一次变向大误差拉高；4号包含侧线、背景干扰、当前观测断档和运动延续不足。这些不是同一种问题，也不能都说成“像素没有信息”。</div>
    <section><h2>具体是哪些原录像</h2><p>3号：<code>E:\\dataset\\experiment\\3\\media\\event.raw</code>，主要评分14.8–15.6s；4号：<code>E:\\dataset\\experiment\\4\\media\\event.raw</code>，主要评分11.8–13.8s。此处时间与项目记录一致。完整事件缓存范围10–16s。</p><p>图片由原生ON/OFF事件累计20ms直接显示（固定计数灰度，不是G通道重建），无新滤波阈值。检测器原输入通常约33.3ms，末尾还有不足1ms残包。局部放大仅为看清，原始px误差没有乘放大倍数。绿框是该时刻<strong>弱参考</strong>；蓝框仅显示起点/中间时刻的算法区域，终点不画算法框，避免把末尾残包缺少事件误说成对清晰整窗的漏检。红叉一直是起点作出的0.4秒预测，未来画面仅用于复盘。<a href="full_3_15.6.png">3号全画幅</a> · <a href="full_4_13.2.png">4号全画幅</a></p></section>
    {articles}
    <section><h2>3号58.29px是怎样产生的</h2><table><tr><th>预测起点</th><th>0.4秒误差</th></tr>{t3}</table><p>只有5个可评分起点，其中15.6s的147.27px把均值拉到58.29px。其余4点均值{np.mean(other):.2f}px——这只是解释均值构成，<strong>正式评分仍保留全部5点</strong>，没有剔除异常点。3号10.8–14.4s人工复盘记录为目标不在视野，不应把大部分无目标时间算成检测漏检。</p><img src="motion_traces.png" alt="实际横向弱参考轨迹与起点冻结的预测"></section>
    <section><h2>看运动过程，比静态标签更直观</h2><p>下列原生画面来自约30Hz缓存，20ms累计；每帧回放120ms，约慢放3.6倍，无插帧、无灰度重建；不足1ms的末尾残包不用于GIF回放，正式评分不变。“当前未检测”表示该包没有当前目标区域，不意味着无人机不存在。显示20ms窗与算法包的窗长并不完全相同。</p><div class="two"><div><h3>3号15.2–16.0s：左移、减速、回右</h3><img src="recording3_turn_slow.gif" alt="3号原生事件慢放"></div><div><h3>4号11.6–13.8s：侧线及间歇检测</h3><img src="recording4_side_slow.gif" alt="4号原生事件慢放"></div></div></section>
    <section><h2>现在可以确定与还不能确定的事</h2><p>可以确定：3号大误差不是全程图像模糊；4号13.2s有肉眼可见目标却没有当前观测；4号12.0s检测有效但运动延续仍错。尚不能确定：每个误差分别有多少来自当前加速度估计、未观测控制变化、背景/桨叶旋转误匹配。单峰相关歧义仍是待验证假设，不能先写成已证实根因。</p><p>这些都是同一个算法在不同条件下的不足。改进应处理可见性、周期证据断续、缺测时状态与未来响应，不能按录像ID写补丁，也不能用事后未来轨迹作为预测输入。</p></section>
    <p><a href="evidence.json">可核验数值与源文件哈希</a> · <a href="../precision_update_20261110/index.html">上一轮全部102起点评分</a></p></html>'''
    (OUT/'index.html').write_text(page,'utf8');print('DONE',str(OUT/'index.html'),json.dumps(evidence,ensure_ascii=False))
