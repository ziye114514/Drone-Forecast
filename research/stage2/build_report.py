"""Post-inference analysis, real-future videos and self-contained Stage2 report."""
from pathlib import Path
import csv,html,json,sys
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study import ROOT,OUT,FIRST,save,csvsave,camera_circle,project,ellipse_from_cv,normal_error
from event_drone_forecast.geometry import fit_rotor_geometry
from event_drone_forecast.motion import extrapolate
from event_drone_forecast.raw_io import open_raw,raw_chunks

def readcsv(path):return list(csv.DictReader(Path(path).open(encoding='utf-8-sig')))
def traces(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]
def number(x):return '—' if x is None else f'{x:.2f}'
def write_image(path,image):
    ok,encoded=cv2.imencode('.png',image);assert ok
    Path(path).write_bytes(encoded.tobytes())
def table(headers,rows):
    return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</table>'

def geometry_analysis():
    rows=readcsv(OUT/'geometry_cases.csv');aggregate=[]
    for method in sorted({r['method'] for r in rows}):
        for arc in (360,240,180,120,60):
            q=[r for r in rows if r['method']==method and int(r['arc_deg'])==arc]
            poses=[r for r in q if r.get('best_calibrated_candidate_error_deg')]
            errors=[float(r['best_calibrated_candidate_error_deg']) for r in poses]
            aggregate.append(dict(method=method,arc_deg=arc,cases=len(q),shape_accepted=sum(r['accepted']=='True' for r in q),
                pose_available=len(poses),median_best_candidate_deg=float(np.median(errors)),p95_best_candidate_deg=float(np.percentile(errors,95)),
                high_quality_pose_count=sum(bool(r.get('quality')) and float(r['quality'])>.45 for r in poses),
                high_quality_wrong_over10deg=sum(bool(r.get('quality')) and float(r['quality'])>.45 and float(r['best_calibrated_candidate_error_deg'])>10 for r in poses)))
    ambiguity=[]
    for off in ('False','True'):
        q=[r for r in rows if r['method']=='cv2_fitEllipseAMS' and r['arc_deg']=='360' and r['off_axis']==off]
        ambiguity.append(dict(off_axis=off,normal_candidates=len(q),
            median_best_candidate_deg=float(np.median([float(r['best_calibrated_candidate_error_deg']) for r in q])),
            median_deployed_up_prior_deg=float(np.median([float(r['initialized_up_prior_error_deg']) for r in q])),
            median_weak_up_prior_deg=float(np.median([float(r['weak_up_prior_error_deg']) for r in q]))))
    save(OUT/'geometry_analysis.json',dict(aggregate=aggregate,ambiguity=ambiguity,
        limits='Three noise seeds per setting, synthetic static circles; no inference about independent flights. Pose rows exclude line-only outputs. Best candidate is an oracle bound.'))
    csvsave(OUT/'geometry_aggregate.csv',aggregate)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for method in sorted({r['method'] for r in aggregate}):
        q=[r for r in aggregate if r['method']==method];x=[r['arc_deg'] for r in q]
        axes[0].plot(x,[r['median_best_candidate_deg'] for r in q],'-o',label=method)
        axes[1].plot(x,[100*r['pose_available']/r['cases'] for r in q],'-o',label=method)
    axes[0].set(xlabel='Visible arc(deg)',ylabel='Median best-candidate normal error(deg)',title='Oracle candidate bound, accepted non-line cases only')
    axes[1].set(xlabel='Visible arc(deg)',ylabel='Full-pose output coverage(%)',title='Line outputs do not count as full pose')
    for ax in axes:ax.grid(alpha=.2);ax.invert_xaxis();ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(OUT/'figures/geometry_arc_tradeoff.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(15,3))
    for ax,arc in zip(axes,[360,240,180,120,60]):
        world=camera_circle(45,40,36,False);rng=np.random.default_rng(20);phase=np.linspace(.3,.3+np.deg2rad(arc),180,endpoint=False)
        points=project(world['H'],np.column_stack((np.cos(phase),np.sin(phase))))+rng.normal(0,.75,(180,2))
        ax.scatter(points[:,0],points[:,1],s=2,c='#6e7d8a')
        estimates=[('truth',world['ellipse'],'#28a06a'),('AMS',ellipse_from_cv(points),'#3683b9'),('frozen',fit_rotor_geometry(points,640,None,True),'#dc7749')]
        for name,m,color in estimates:
            if m is None:continue
            if isinstance(m,dict):
                if m['axis_only']:continue
                x,y=m['center'];a,b=m['major']/2,m['minor']/2;angle=np.deg2rad(m['angle'])
            else:x,y,a,b,angle=m
            pp=np.linspace(0,2*np.pi,180);r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            xy=np.column_stack((a*np.cos(pp),b*np.sin(pp)))@r.T+[x,y];ax.plot(xy[:,0],xy[:,1],c=color,label=name)
        ax.set_title(f'{arc}deg,36px,noise0.75px',fontsize=9);ax.set_aspect('equal');ax.invert_yaxis();ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'figures/predeclared_arc_examples.png',dpi=160);plt.close(fig)
    return aggregate,ambiguity

def raster_analysis():
    rows=readcsv(OUT/'raster_frequency_cases.csv');aggregate=[]
    for tilt in (20,60,82,88):
        for diameter in (24,96):
            q=[r for r in rows if int(r['tilt_deg'])==tilt and int(r['diameter_px'])==diameter];ok=[r for r in q if r['valid']=='True']
            aggregate.append(dict(tilt_deg=tilt,diameter_px=diameter,cases=len(q),valid=len(ok),
                correct_5pct=sum(r.get('correct_within_5pct')=='True' for r in q),
                median_relative_error_pct=float(np.median([float(r['relative_frequency_error'])*100 for r in ok])) if ok else None,
                wrong_but_quality_over045=sum(float(r['quality'])>.45 and float(r['relative_frequency_error'])>.05 for r in ok)))
    save(OUT/'raster_analysis.json',dict(aggregate=aggregate,correct=sum(r['correct_5pct'] for r in aggregate),total=len(rows),
        caveat='Ground-truth projected ellipse supplied; ideal raster threshold events. Not end-to-end RAW detection or real flight RPM validation.'))
    return aggregate

def score_own(folder=None,run_label='hybrid_experimental'):
    folder=OUT/'integrated/own_train6' if folder is None else Path(folder)
    current=traces(folder/'forecasts.jsonl');oldroot=ROOT.parent/'output/paper_reproduction_v030'
    baseline=traces(oldroot/'runs/train6/forecasts.jsonl');labels=json.loads((oldroot/'evaluation_data/train6.json').read_text())['labels']
    runs={'v030':baseline,run_label:current};details=[];false_activity=[]
    for run_name,records in runs.items():
        times=np.array([r['t'] for r in records])
        for label in labels:
            if not 0<label['t']<=30:continue
            if label['state']=='absent':
                k=np.searchsorted(times,label['t'],side='right')-1
                if k>=0:false_activity.append(dict(run=run_name,t=label['t'],active=records[k]['active']))
                continue
            if label['state']!='present':continue
            box=np.array(label['bbox']);center=(box[:2]+box[2:])/2
            for h in (.1,.2,.4):
                k=np.searchsorted(times,label['t']-h,side='right')-1
                if k<0:continue
                q=records[k]
                for method in ('cv','paper_nominal','paper_adaptive','paper_fused'):
                    row=dict(run=run_name,target_t=label['t'],horizon_s=h,method=method,available=q['active'],physics_active=q.get('physics_active',False))
                    if q['active']:row['error_px']=float(np.linalg.norm(extrapolate(q['states'][method],label['t']-times[k])-center))
                    details.append(row)
    metrics=[]
    for h in (.1,.2,.4):
        for method in ('cv','paper_nominal','paper_adaptive','paper_fused'):
            subsets={run:[r for r in details if r['run']==run and r['horizon_s']==h and r['method']==method] for run in runs}
            valid={run:{r['target_t'] for r in q if r['available']} for run,q in subsets.items()};common=set.intersection(*valid.values())
            for run,q in subsets.items():
                ok=[r for r in q if r['available']];co=[r for r in ok if r['target_t'] in common]
                metrics.append(dict(run=run,horizon_s=h,method=method,targets=len(q),available=len(ok),coverage=len(ok)/max(1,len(q)),
                    error_px=float(np.mean([r['error_px'] for r in ok])) if ok else None,common_available=len(co),
                    common_error_px=float(np.mean([r['error_px'] for r in co])) if co else None))
    absent=[dict(run=run,absent_labels=len(q),active_when_absent=sum(r['active'] for r in q)) for run in runs for q in [[r for r in false_activity if r['run']==run]]]
    csvsave(folder/'endpoint_details.csv',details);csvsave(folder/'annotated_absence.csv',false_activity)
    save(folder/'evaluation.json',dict(metrics=metrics,absence_diagnostic=absent,gt_interpolated=False,
        caveat='Existing sparse human boxes on a previously used development clip; no test-set or real RPM/pose accuracy claim.'))
    return metrics,absent

def video(sequence):
    folder=OUT/f'activity_fallback_fixed/sequence{sequence}';records=traces(folder/'forecasts.jsonl');observations=traces(folder/'observations.jsonl')
    times=np.array([r['t'] for r in records]);f=np.load(folder/'preview_frames.npz');pt=f['times'];images=f['images']
    raw=Path(json.loads((folder/'completion.json').read_text())['raw']);labels=[]
    for line in (raw.parent/'coordinates.txt').read_text().splitlines():
        stamp,values=line.split(':',1);labels.append((float(stamp),list(map(float,values.split(',')[:4]))))
    writer=cv2.VideoWriter(str(folder/'真实未来对照.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1920,420));count=0
    methods=['original_rotor_cv','rotor_or_activity_cv'];colors=[(40,155,250),(215,100,240)]
    for j,t in enumerate(pt):
        if t<10 or t>19.5:continue
        future=j+8
        if future>=len(pt):continue
        k=np.searchsorted(times,t+1e-8,side='right')-1;q=records[k];panels=[];predictions=[]
        for method,color in zip(methods,colors):
            canvas=np.zeros((420,640,3),np.uint8);canvas[60:]=cv2.cvtColor(images[j],cv2.COLOR_GRAY2BGR)
            cv2.putText(canvas,f'{method} t={t:.2f}s',(12,23),0,.62,color,1)
            cv2.putText(canvas,'Forecast made now; +0.4s',(12,47),0,.55,(220,220,220),1)
            o=observations[k]['observations'][method]
            if o['detected']:
                box=np.round(np.asarray(o['bbox'])/2).astype(int);cv2.rectangle(canvas,(box[0],box[1]+60),(box[2],box[3]+60),color,1)
            prediction=extrapolate(q['states'][method],.4) if q['active'][method] else None;predictions.append(prediction)
            if prediction is not None:
                p=np.round(prediction/2).astype(int);cv2.drawMarker(canvas,(p[0],p[1]+60),color,cv2.MARKER_CROSS,16,2)
            else:cv2.putText(canvas,'NO ACTIVE FORECAST',(12,87),0,.55,(150,150,150),1)
            panels.append(canvas)
        canvas=np.zeros((420,640,3),np.uint8);canvas[60:]=cv2.cvtColor(images[future],cv2.COLOR_GRAY2BGR)
        cv2.putText(canvas,f'Actual RAW at t={pt[future]:.2f}s',(12,23),0,.62,(80,220,100),1)
        cv2.putText(canvas,'Orange: original | Purple: new position fallback',(12,47),0,.48,(220,220,220),1)
        for prediction,color in zip(predictions,colors):
            if prediction is not None:
                p=np.round(prediction/2).astype(int);cv2.drawMarker(canvas,(p[0],p[1]+60),color,cv2.MARKER_CROSS,18,2)
        label=min(labels,key=lambda r:abs(r[0]-pt[future]))
        if abs(label[0]-pt[future])<=.018:
            box=np.round(np.array(label[1])/2).astype(int);cv2.rectangle(canvas,(box[0],box[1]+60),(box[2],box[3]+60),(60,220,80),1)
        panels.append(canvas);joined=np.hstack(panels);writer.write(joined)
        if abs(t-15)<.001:write_image(folder/'未来对照_15秒.png',joined)
        count+=1
    writer.release();save(folder/'video_manifest.json',dict(frames=count,fps=20,horizon_s=.4,anchor_interval_s=[10,19.5],
        real_future_raw=True,labels_read_only_for_posthoc_overlay=True,box_tolerance_s=.018,no_gt_interpolation=True))

def own_video(folder=None):
    folder=OUT/'integrated/own_train6' if folder is None else Path(folder)
    current=traces(folder/'forecasts.jsonl');times=np.array([r['t'] for r in current])
    metadata=json.loads((folder/'run_manifest.json').read_text());reader=open_raw(metadata['input_raw']);images=[];pt=[]
    for i,(t,e) in enumerate(raw_chunks(reader,22,6.5,10000)):
        if (i+1)%5:continue
        image=np.bincount(e['y'].astype(int)*1280+e['x'],minlength=720*1280).reshape(720,1280)
        image=np.uint8(np.clip(np.log1p(image)/4,0,1)*255);images.append(cv2.resize(image,(640,360),interpolation=cv2.INTER_AREA));pt.append(t)
    writer=cv2.VideoWriter(str(folder/'自有录像_真实未来对照.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1280,420));frames=0
    for j,t in enumerate(pt):
        if t>28 or j+8>=len(images):continue
        q=current[np.searchsorted(times,t+1e-8,side='right')-1];canvas=np.zeros((420,1280,3),np.uint8)
        canvas[60:,:640]=cv2.cvtColor(images[j],cv2.COLOR_GRAY2BGR);canvas[60:,640:]=cv2.cvtColor(images[j+8],cv2.COLOR_GRAY2BGR)
        cv2.putText(canvas,f'Current RAW t={t:.2f}s; source={q["position_source"]}',(12,23),0,.52,(230,230,230),1)
        cv2.putText(canvas,f'Physics active={q["physics_active"]}; future horizon0.4s',(12,47),0,.52,(230,230,230),1)
        cv2.putText(canvas,f'Actual RAW t={pt[j+8]:.2f}s',(652,23),0,.6,(80,220,100),1)
        cv2.putText(canvas,'Purple: fused | Orange: CV',(652,47),0,.55,(230,230,230),1)
        if q['active']:
            for method,color in [('cv',(40,155,250)),('paper_fused',(215,100,240))]:
                p=np.int32(np.round(extrapolate(q['states'][method],.4)/2));cv2.drawMarker(canvas,(p[0]+640,p[1]+60),color,cv2.MARKER_CROSS,18,2)
        writer.write(canvas)
        if abs(t-24)<.001:write_image(folder/'自有录像_24秒.png',canvas)
        frames+=1
    writer.release();save(folder/'video_manifest.json',dict(frames=frames,fps=20,real_future_raw=True,horizon_s=.4,anchor_interval_s=[22.05,28]))

def main():
    (OUT/'figures').mkdir(exist_ok=True);aggregate,ambiguity=geometry_analysis();raster=raster_analysis()
    scores=[]
    for sequence in ('1','2','3'):
        d=json.loads((OUT/f'activity_fallback_fixed/sequence{sequence}/scores.json').read_text())
        scores.extend(dict(sequence=sequence,**r) for r in d['metrics']);video(sequence)
    integrated=OUT/'integrated';fred=traces(integrated/'fred3/forecasts.jsonl');cv_ref=traces(OUT/'activity_fallback_fixed/sequence3/forecasts.jsonl')
    for a,b in zip(fred,cv_ref):
        assert a['active']==b['active']['rotor_or_activity_cv']
        if a['states']:
            for key in ('position','velocity','acceleration'):np.testing.assert_array_equal(a['states']['cv'][key],b['states']['rotor_or_activity_cv'][key])
        assert not a['physics_active']
    own,absent=score_own();own_video()
    public=json.loads((OUT/'public_frequency_results.json').read_text());public_rows=[]
    for r in public:
        # Provided GT is mechanical full-cycle rate; blade-pass rate is a
        # different quantity. Do not mistake f*blade_count for an RPM error.
        mechanical_hz=r.get('blade_pass_hz',0)/r['assumed_blade_count']
        public_rows.append([r['sequence'],r['assumed_blade_count'],r['valid'],number(r.get('blade_pass_hz')),number(mechanical_hz),r['gt_hz'],number(r.get('phase_coherence'))])
    rows=[[r['sequence'],r['method'],f"{r['available']}/{r['targets']}",f"{r['coverage']*100:.1f}%",number(r['mean_error_px']),r['common_available'],number(r['common_mean_error_px'])] for r in scores if r['horizon_s']==.4]
    geo_rows=[[r['method'],r['arc_deg'],f"{r['pose_available']}/{r['cases']}",number(r['median_best_candidate_deg']),number(r['p95_best_candidate_deg']),f"{r['high_quality_wrong_over10deg']}/{r['high_quality_pose_count']}"] for r in aggregate]
    raster_rows=[[r['tilt_deg'],r['diameter_px'],f"{r['valid']}/{r['cases']}",f"{r['correct_5pct']}/{r['cases']}",number(r['median_relative_error_pct'])] for r in raster]
    ownrows=[[r['run'],r['method'],f"{r['available']}/{r['targets']}",number(r['error_px']),r['common_available'],number(r['common_error_px'])] for r in own if r['horizon_s']==.4]
    summary=json.loads((integrated/'own_train6/summary.json').read_text());diagnosis=json.loads((OUT/'fred_diagnosis/completion.json').read_text())
    save(OUT/'stage2_summary.json',dict(geometry_cases=2160,raster_cases=96,raster_within5pct=92,public_sequences=3,
        fixed_activity_scores=scores,own_sparse_scores=own,own_absence=absent,integrated_own=summary,
        integrated_fred3_cv_exact_match=True,new_algorithm_claimed=False,independent_real_flight_rpm_pose_validated=False))
    document=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第二阶段：模块验证与二维跟踪回退</title>
<style>body{{margin:0;background:#f3f6f8;color:#1b2b36;font-family:"Microsoft YaHei",sans-serif;line-height:1.8}}main{{max-width:1220px;margin:auto;padding:28px}}section{{background:white;border:1px solid #dce4e9;border-radius:10px;padding:22px;margin:20px 0}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;padding:8px;border-bottom:1px solid #dde4e8}}th{{background:#eef3f6}}img,video{{max-width:100%}}.scroll{{overflow:auto}}.note{{background:#fff0d7;border-left:4px solid #d4a24a;padding:18px}}a{{color:#096d9b}}code{{background:#edf2f5}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style><main>
<h1>第二阶段：测量是否可靠，目标能否持续跟住？</h1><p>2026-10-16 · 独立几何/像素事件生成 · 公开RAW · 无同步日志路线 · 新增独立实验入口run_hybrid.py</p>
<div class="note"><b>结论：</b>二维位置跟踪的覆盖率有明显进展，但尚未证明桨叶物理模型提高了预测精度。完整椭圆的解析反投影可以正确给出候选；残弧拟合与姿态候选选择仍有显著错误。新增普通活动回退解决一部分漏检，也会跟踪非无人机活动，不能当作语义无人机检测器。</div>
<p><a href="protocol.json">首轮事前协议</a> · <a href="activity_fallback_fixed/protocol.json">重捕获修复后冻结协议</a> · <a href="integrated/protocol.json">组合运行冻结协议</a> · <a href="stage2_summary.json">全部结论与数值</a> · <a href="../stage1_observation_errors_20261016/第一阶段实验报告.html">第一阶段</a></p>
<section><h2>先定位FRED失败</h2><p>原前端逐帧重跑并插入只读计数，2000帧的检测与中心和原v0.3逐项一致。317个标注时刻中，262个时刻没有任何候选，另21个时刻目标框内仅1个候选，34个时刻至少2个。原本要求多个桨盘分组的前端难以覆盖这些目标。此统计使用事后标注诊断，未把框送入检测器。</p><p>新增活动回退使用30毫秒事件密度、连通区域和历史关联，三帧确认、丢失后重新发现；只有位置。初版重置逻辑使确认计数每帧被清空，已保留失败运行并用回归检查复现。修复仅增加一次清空last_seen，没有根据测试标签改阈值。</p></section>
<section><h2>真实RAW二维预测：看误差，也看覆盖率</h2><p>每条序列固定前20秒，三个方案共用原CV预测实现，0.1/0.2/0.4秒均已评分，下表展示0.4秒。sequence1是开发片段；sequence2已经用于失败版诊断；sequence3是在修复冻结后才运行的额外迁移检查。三者都是官方训练数据，不是官方测试成绩。帧与端点相关，不宣称统计显著或SOTA。</p>
<div class="scroll">{table(['序列','方案','可用/目标端点','覆盖率','所有可用误差px','三方共同端点','共同端点误差px'],rows)}</div>
<p><b>负结果：</b>sequence2在共同46个端点上的0.4秒误差，原版137.89px，组合146.12px；尽管覆盖率提高，也不能称为全面精度改进。活动CV与组合接近，说明这一轮收益主要来自位置跟踪；不能归功于桨叶推力模型。各方案观测置信度/协方差按其原设定保留，亦非只改变一个门槛的消融。</p>
<h3>预测和真实0.4秒后画面对照</h3><p>左：原桨盘前端+CV；中：四桨合格时用桨盘中心、其余用活动中心+CV；右：真实未来RAW。橙色/紫色十字是当前时刻已经发布的预测。绿色框仅在未来时刻附近有提供标注时事后叠加，未插值标签。视频固定展示10–19.5秒，包含成功和失败。</p>
<p>sequence1：</p><video controls preload="metadata" poster="activity_fallback_fixed/sequence1/未来对照_15秒.png" src="activity_fallback_fixed/sequence1/真实未来对照.mp4"></video>
<p>sequence3，原前端无可用预测：</p><video controls preload="metadata" poster="activity_fallback_fixed/sequence3/未来对照_15秒.png" src="activity_fallback_fixed/sequence3/真实未来对照.mp4"></video></section>
<section><h2>自己的RAW：组合后物理分支仍能运行吗？</h2><p>另建实验入口，只输入RAW。四个当前可见桨盘且几何质量&gt;0.45时用原桨盘位置；其余使用活动位置，同时禁止活动检测更新物理模型或启用推力。原v0.3源码和既往成绩未改。</p>
<p>train6前30秒：检测{summary['detected_frames']}/{summary['frames']}帧，物理分支{summary['physics_active_frames']}帧，物理在线更新{summary['physical_updates']['paper_adaptive']}次。下面仅用原来已有的稀疏人工框评分，不补密集伪真值。这是先前使用过的开发录像。</p>
<div class="scroll">{table(['版本','预测分支','可用/端点','可用误差px','共同端点','共同误差px'],ownrows)}</div>
<p><b>背景误报必须保留：</b>{html.escape(json.dumps(absent,ensure_ascii=False))}。active_when_absent表示人工标为目标不在画面时，程序仍发布了活动目标预测。它没有无人机语义分类能力，背景运动也可能被锁定；不能只展示有无人机的片段来声称完整检测成功。</p>
<video controls preload="metadata" poster="integrated/own_train6/自有录像_24秒.png" src="integrated/own_train6/自有录像_真实未来对照.mp4"></video>
<p>左当前，右真实0.4秒后；紫色融合，橙色CV。显示物理是否激活，画面不使用未来信息驱动预测。</p></section>
<section><h2>几何：残弧可以拟合，但置信度仍不可靠</h2><p>2160组独立三维圆投影，覆盖12/36/120像素直径、0–87°倾斜、全弧至60°可见弧、离轴位置和两种噪声。真值由三维平面单应和解析圆锥曲线产生，未调用椭圆拟合生成真值。每设置仅3个噪声种子，没有实飞含义。过去先验来自独立加噪的上一帧完整圆。</p>
<p>所有无噪解析椭圆反投影均包含正确法向候选；但单圆有候选歧义。“最接近真值候选”用到了评分真值，属于不可部署的误差下界。完整圆离轴时，AMS最佳候选误差中位数{ambiguity[1]['median_best_candidate_deg']:.2f}°，当前图像向上初始化实际选出的候选却有{ambiguity[1]['median_deployed_up_prior_deg']:.2f}°中位误差。不能将候选中存在正确解写成实际姿态已经测准。</p>
<img src="figures/geometry_arc_tradeoff.png"><img src="figures/predeclared_arc_examples.png">
<p>下表姿态覆盖率不把近线段算作完整姿态；误差只在各自接受且非线段的样本上计算，方法间接受集合不同，不是公平共同样本排名。完整逐例CSV支持复核。质量&gt;0.45仍可能对应&gt;10°错误。</p>
<div class="scroll">{table(['方法','可见弧°','姿态输出/总数','最佳候选中位°','P95°','高质量但错误/高质量输出'],geo_rows)}</div>
<p>直接提供的圆弧不是RAW分割出的闭合遮挡轮廓；这里测的是拟合器对点支持不足的反应，不能替代真实遮挡全流程实验。现有曲率/假设评分在这个协议下经常不如直接AMS，不能继续无条件叫它精度增强。近侧视只评轴线，不把轴方向当飞行速度方向。</p></section>
<section><h2>测速：已验证的条件与尚未验证的部分</h2><p>96组二维像素强度阈值事件测试：两叶片由已知三维平面生成，每25微秒采样亮度、触发对数对比度阈值事件，并加背景噪声。提供真值投影ROI以隔离测速，默认160毫秒窗口，不把角度或频率真值输入估计器。</p>
<div class="scroll">{table(['倾斜角°','直径px','有效/总数','误差≤5%/总数','有效相对误差中位%'],raster_rows)}</div>
<p>92/96组在5%以内；4组错误都出现在24像素、88°极端侧视。该模拟没有真实相机带宽、阈值分散、遮挡物新事件或飞行控制，不是ESIM复现，不能据此声称真实飞行RPM精度。本项目测速核心至少在这些独立图像生成条件下通过了条件性验证。</p>
<h3>公开周期目标的模型不匹配检查</h3><p>使用作者ROI内切圆作代理，窗口扩展为1秒、频率下限5Hz。目标不是已知双叶桨。提供真值是完整旋转周期频率；程序输出的blade_pass_hz与机械频率不同，必须除以假定叶片数，不能把它们直接相减当作RPM误差。</p>
{table(['序列','假定叶片数','门控有效','输出重复Hz','假定机械Hz','提供周期Hz','相位相干'],public_rows)}
<p>正视两种假定均被拒绝；侧视在双叶假定下换算约26.5Hz接近26.3Hz，但没有独立确认目标叶片/纹理周期，不能事后挑假定来报测速成功。原EEPPR的已有结果属于原作者算法，仍与本模块验证分开。</p></section>
<section><h2>现在能运行什么，下一步做什么</h2><p>实验入口如下，除RAW外无传感器输入；不将当前版本替换为原v0.3正式成绩。新程序的活动位置不等于已验证的无人机身份。该分支尚未实现实飞真值验证或保证实时，本轮部分运行并发执行，耗时不可拿来作公平算法速度比较。</p>
<pre>&amp; 'D:\\drone forecast\\.venv-event311\\Scripts\\python.exe' 'D:\\drone forecast\\event_drone_forecast\\run_hybrid.py' --raw 'E:\\dataset\\train\\6\\media\\event.raw' --duration 30 --output 'D:\\drone forecast\\output\\my_hybrid_run'</pre>
<p>下一轮优先处理两个明确瓶颈：一是活动目标的身份确认与背景误报，二是残弧的过高置信度及多法向候选的选择。候选选择可先比较多桨共同平面约束和已有时间先验，随后才研究高置信度频率歧义与在线参数互相补偿。它们当前是待验证改进方向，不作创新性声明。</p>
<p>来源与定位：<a href="https://github.com/miccunifi/FRED">FRED</a>、<a href="https://github.com/JackPieCZ/EEPPR">EEPPR</a>、<a href="https://github.com/radimspetlik/helixtrack">HelixTrack</a>、<a href="https://github.com/uzh-rpg/rpg_esim">ESIM（相关工具；本轮没有运行）</a>。原物理结构来自2604.18289论文，当前均为图像像素模型，未唯一恢复质量、推力系数或风速。</p></section></main></html>'''
    (OUT/'第二阶段实验报告.html').write_text(document,encoding='utf-8')
    (OUT/'阶段报告.md').write_text('# 第二阶段结果\n\n已完成2160组几何、96组像素事件测速、三条FRED序列二维预测及自有train6前30秒物理分支组合运行。\n\n'+
        '新回退主要提高位置跟踪覆盖率，不能归因于推力模型；sequence2共同端点误差变差。残弧与法向候选选择仍存在明显错误；活动跟踪存在背景误报。\n\n'+
        '完整数值、真实未来对照视频、失败运行、冻结协议和证据边界见第二阶段实验报告.html与stage2_summary.json。\n',encoding='utf-8')
    print('Stage2 report, scored evidence and real future videos built',flush=True)

if __name__=='__main__':main()
