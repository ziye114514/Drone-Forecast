"""Post-hoc failure decomposition only; no parameter tuning or new annotations."""
from pathlib import Path
import collections,hashlib,json
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2];INPUT=ROOT.parent/'output/stage31_cross_recording_physics_20261018'
OUT=ROOT.parent/'output/stage32_cross_failure_audit_20261018'


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not (OUT/'audit.json').exists();path=INPUT/'train3/observations_forecasts.jsonl';labelpath=ROOT.parent/'output/raw_only_project/evaluation_data/train3.json'
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'protocol.json',dict(posthoc=True,source_hashes={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))},
        input_sha256=sha(path),labels_sha256=sha(labelpath),video_sha256=sha(INPUT/'train3/forecasts.mp4'),
        selection='All12 present labels25–31s; fixed display indices0,1,2,3,5,7,9,11. Existing rendered event-count MP4 for visualization, no reconstructed G/new RAW reads/new truth.',
        purpose='Separate current localization, future maneuver error and sensor association capacity. Diagnostic not new evaluation or independent alignment calibration.'))
    rows=list(map(json.loads,path.read_text(encoding='utf-8').splitlines()));tt=np.array([r['t'] for r in rows]);labels=[l for l in load(labelpath)['labels'] if 25<l['t']<=31]
    current=[];counters=collections.Counter();reasons=collections.Counter();activity=[]
    for r in rows:
        o=r['observation'];s=r['sensing'];v=sum(x['observed_now'] for x in s['rotors']);n=len(o['rotors'])
        counters['published']+=o['detected'];counters['four_frontend']+=n==4;counters['four_sensed_now']+=s['four_visible'];counters['four_frequency']+=s['four_frequencies']
        counters['frontend_rotors_not_all_represented']+=n>v
        counters['four_slots_occupied_with_unrepresented_current']+=bool(len(s['rotors'])==4 and n>v)
        activity.append(dict(t=r['t'],frontend_rotors=n,visible_tracks=v,stored_tracks=len(s['rotors']),fresh_frequencies=sum(x['frequency_valid'] for x in s['rotors'])))
        for x in s['rotors']:
            if x['observed_now']:reasons[(x.get('frequency_diagnostics') or {}).get('reason','measurement_payload')] += 1
    for l in labels:
        i=int(np.searchsorted(tt,l['t'],side='right')-1);assert i>=0 and l['t']-tt[i]<=.025
        r=rows[i];o=r['observation'];b=np.asarray(l['bbox']);target=(b[:2]+b[2:])/2
        error=float(np.linalg.norm(np.asarray(o['center'])-target)) if o['detected'] else None
        current.append(dict(target_t=l['t'],anchor_t=float(tt[i]),index=i,bbox=l['bbox'],detected=o['detected'],current_error_px=error,
            target_center=target.tolist(),current_center=o.get('center'),box_diagonal_px=float(np.linalg.norm(b[2:]-b[:2]))))
    analysis=load(INPUT/'analysis.json');future=[d for d in analysis['details'] if d['model']=='cv' and d['nominal_h']==.4 and d['forecast_valid']]
    result=dict(current_targets=current,counts=dict(counters),frequency_diagnostic_record_ticks=dict(reasons),
        reason_count_scope='Currently observed track records carrying each stored diagnostic, NOT independent frequency estimation attempts.',
        conditional_current_mean_px=float(np.mean([r['current_error_px'] for r in current if r['detected']])),future_cv_04=future,
        inference='Current first3 positions roughly agree with coarse boxes, so large future errors cannot all be blamed on a global alignment offset. Later9 current misses and large maneuver extrapolation are distinct. Four-slot blocking is a mechanism suggested by observed counts and source, not true rotor-ID accuracy.')
    save(OUT/'audit.json',result);save(OUT/'activity.json',activity)
    fig,axs=plt.subplots(3,1,figsize=(13,10),sharex=True,constrained_layout=True)
    centers=np.array([r['observation'].get('center',[np.nan,np.nan]) for r in rows])
    for j in range(2):
        axs[j].plot(tt,centers[:,j],label='current detector');axs[j].scatter([r['target_t'] for r in current],[r['target_center'][j] for r in current],marker='x',label='archived box center',color='#c64c63')
        axs[j].scatter([d['target_t'] for d in future],[d['predicted'][j] for d in future],marker='o',label='0.4s CV endpoint',color='#d28b23');axs[j].set_ylabel(['x (px)','y (px)'][j]);axs[j].legend(fontsize=9);axs[j].grid(alpha=.2)
    axs[2].plot(tt,[r['frontend_rotors'] for r in activity],label='frontend rotors');axs[2].plot(tt,[r['visible_tracks'] for r in activity],label='tracks seen now');axs[2].plot(tt,[r['fresh_frequencies'] for r in activity],label='fresh frequency tracks');axs[2].set(xlabel='RAW time (s)',ylabel='count');axs[2].legend();axs[2].grid(alpha=.2)
    fig.savefig(OUT/'failure_decomposition.png',dpi=150);plt.close(fig)
    cap=cv2.VideoCapture(str(INPUT/'train3/forecasts.mp4'));pictures=[]
    for k in [0,1,2,3,5,7,9,11]:
        d=current[k];cap.set(cv2.CAP_PROP_POS_FRAMES,d['index']);ok,frame=cap.read();assert ok
        b=np.rint(d['bbox']).astype(int);cv2.rectangle(frame,tuple(b[:2]),tuple(b[2:]),(190,60,180),2)
        lo=np.maximum(b[:2]-70,0);hi=np.minimum(b[2:]+70,[1280,720]);crop=frame[lo[1]:hi[1],lo[0]:hi[0]]
        scale=min(620/crop.shape[1],300/crop.shape[0]);small=cv2.resize(crop,None,fx=scale,fy=scale)
        canvas=np.full((350,640,3),246,np.uint8);canvas[45:45+small.shape[0],10:10+small.shape[1]]=small
        cv2.putText(canvas,f'{d["target_t"]:.3f}s | current={d["detected"]}',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(50,65,85),1);pictures.append(canvas)
    cap.release();cv2.imencode('.png',np.vstack([np.hstack(pictures[i:i+4]) for i in [0,4]]))[1].tofile(str(OUT/'aligned_targets.png'))
    headers=['标签时刻','当前发布','当前误差px','框对角px'];table='<table><tr>'+''.join('<th>'+v+'</th>' for v in headers)+'</tr>'
    for d in current:table+='<tr>'+''.join('<td>'+str(v)+'</td>' for v in [round(d['target_t'],3),d['detected'],None if d['current_error_px'] is None else round(d['current_error_px'],2),round(d['box_diagonal_px'],2)])+'</tr>'
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第三十二阶段：跨录像失败拆解</title><style>body{font:17px/1.75 "Microsoft YaHei",sans-serif;background:#f3f6fa;color:#243342;margin:0}main{max-width:1400px;margin:auto;background:white;padding:32px}h1{font-size:30px}img{max-width:100%}table{border-collapse:collapse;width:100%}td,th{padding:8px;text-align:left;border-bottom:1px solid #dae4ee}.note{padding:18px;background:#edf6fb;border-left:5px solid #3787a7}a{color:#176c99}</style><main><h1>第三十二阶段：不要把跨录像失败归成一个原因</h1><p>阶段31保存结果的事后诊断；没有新算法、没有新增真值。</p><div class="note">前三个有当前检测的粗框标签，位置相差约十几像素；0.4s预测却有186/303/137px误差。随后9个标签当前检测都缺失。错误未来运动假设与定位中断同时存在，不能只归咎于频率、目标小或时间对齐。</div>'''
    text+=table+'</table><img src="failure_decomposition.png"><h2>实际画面与粗框对齐</h2><img src="aligned_targets.png"><p>紫框是存档近似框，绿色是原当前检测；画面来自已保存原生事件计数视频，不是灰度重建。裁剪只用于看清目标，不用于重评分或重新定义标签。前三个当前位置大致匹配支持基本时空对齐，但没有精确标定，不能据此排除所有时延/标框误差。</p><h2>频率前面的关联瓶颈</h2><p>前端给四桨30时刻，RotorSensing同时看见四轨迹仅1时刻；四频有效0。旧实现总轨迹最多4，未匹配新桨仅在有空位时建立，旧轨迹最长占位300ms。记录显示有当前桨盘未被活跃轨迹接住且四槽占满，说明需要独立验证旧占位/门限的作用，而非直接给频率算法判死刑。</p><p>下一步：复用现成跟踪器的出生/失效思路，做不改当前几何的关联消融，明确每次新ID必须清空旧频率与总推力参考；另外查事件特征跟踪/整机补测，解决后9个漏定位。关联指标只是可用性，仍没有独立电机ID/转速真值。强转向0.4秒也需要更短视野和多假设预测，不能保证猜中未来飞控输入。</p><p><a href="protocol.json">诊断来源</a>、<a href="audit.json">具体数据</a>、<a href="activity.json">逐时刻计数</a>。无G/yolov10-main访问，原算法/结果不改。</p></main></html>'
    (OUT/'第三十二阶段跨录像失败拆解.html').write_text(text,encoding='utf-8')
    print(json.dumps(dict(counts=dict(counters),conditional_current_mean_px=result['conditional_current_mean_px']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
