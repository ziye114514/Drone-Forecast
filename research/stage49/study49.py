"""Frozen synthetic + cached self-recording diagnosis of rotation vs translation."""
from pathlib import Path
import argparse,datetime,hashlib,html,json,os,sys
import numpy as np
import rigid49 as g

ROOT=Path(__file__).resolve().parents[2];PROJECT=ROOT.parent;BASE=PROJECT/'output'
OUT=BASE/'stage49_rigid_center_20261020'
SENSORS=BASE/'stage34_slot_frequency_forecast_20261018'
POSITIONS=BASE/'stage45_cuda_graph_20261019/sequence'
CASES={'clear6':(22.,2.),'side6':(108.,5.),'train3':(25.,6.)}
PARAMS=dict(seed=49,association_margin_fraction=.25,real_jitter_repeats=128,synthetic_noise_sigma_px=.75,
    synthetic_angles_deg=[0,30,60,75,82,87,89],synthetic_noise_repeats=1000,ill_conditioned_diagonal_sine=.15,
    interpretation='Four current fitted ellipse/line centers are candidate hubs, not certified motor axes. Convex hull pairing is a geometric hypothesis. No true pose/COM/future error scored. Body-only neural box center is a consistency reference, not geometric-center ground truth.',
    scope='Only diagnose current center vs attitude deformation, not a new detector or proof of future improvement. Stationary and translating synthetic controls both required. No stationary or periodic-motion prior applied to real UAVs.')
REFERENCES=[
    ['Projective geometry (Hartley and Zisserman, author chapter)','https://www.robots.ox.ac.uk/~vgg/hzbook/hzbook2/HZintroduction.pdf'],
    ['IPPE author implementation, integrated in OpenCV','https://github.com/tobycollins/IPPE'],
    ['Direct 3D model-based event object tracking, ICRA 2024 author code','https://github.com/BluecatLi/direct_event_object_tracker'],
    ['Unbiased Estimator for Distorted Conics, CVPR 2024','https://openaccess.thecvf.com/content/CVPR2024/papers/Song_Unbiased_Estimator_for_Distorted_Conics_in_Camera_Calibration_CVPR_2024_paper.pdf'],
    ['Relative State Estimation using Event-Based Propeller Sensing','https://arxiv.org/abs/2604.18289']]

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sources():return {'research/stage49/'+n:sha(Path(__file__).with_name(n)) for n in ['rigid49.py','study49.py']}
def inputs():
    paths=[p for c in CASES for p in [SENSORS/c/'sensing.jsonl',POSITIONS/c/'observations.jsonl']]
    return {p.relative_to(PROJECT).as_posix():sha(p) for p in paths}
def freeze():
    assert not OUT.exists(),'New directory required; preserve prior runs.'
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),input_hashes=inputs(),parameters=PARAMS,cases=CASES,prior_art=REFERENCES,
        design='Synthetic exact pinhole projection and controlled localization noise; cached current-only self recordings, all650 native20 positions. Save current diagnostics; never read evaluation labels or future forecast files. No selection by results. Standard geometry, not new algorithm claim.'))
    print('Stage49 frozen; no previous files changed.',flush=True)
def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==sources();assert p['input_hashes']==inputs();return p
def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).resolve()
        if p.name in ['targets.json','forecasts.jsonl'] or p.is_relative_to((BASE/'raw_only_project/evaluation_data').resolve()):raise RuntimeError('Truth/future file forbidden: '+str(p))

def synthetic():
    rng=np.random.default_rng(PARAMS['seed']);records=[];moving_errors=[];affine_errors=[]
    for t in np.linspace(0,4,241):
        R=g.rotation(18*np.sin(2*np.pi*t/3),75*np.sin(2*np.pi*t/4),12*np.sin(2*np.pi*t/2.7))
        H=g.homography(R,[0.,0.,3.]);p=g.project(H,g.TEMPLATE);true=g.project(H,[[0,0]])[0];cross,sine=g.intersection(p);ell=g.ellipse_centers(H);ecross,_=g.intersection(ell)
        affine=g.TEMPLATE@R[:2,:2].T*300+true
        affine_errors.append(float(np.linalg.norm(affine.mean(0)-true)))
        moved=g.homography(R,[.3*np.sin(t),.15*np.cos(t),3.+.1*np.sin(2*t)])
        mc,_=g.intersection(g.project(moved,g.TEMPLATE));moving_errors.append(float(np.linalg.norm(mc-g.project(moved,[[0,0]])[0])))
        records.append(dict(t=float(t),hubs=p.tolist(),ellipse_centers=ell.tolist(),true_center=true.tolist(),diagonal_center=cross.tolist(),hub_mean=p.mean(0).tolist(),
            mean_error_px=float(np.linalg.norm(p.mean(0)-true)),diagonal_error_px=float(np.linalg.norm(cross-true)),ellipse_diagonal_error_px=float(np.linalg.norm(ecross-true)),
            max_ellipse_vs_hub_px=float(np.max(np.linalg.norm(ell-p,axis=1))),diagonal_sine=float(sine)))
    assert max(r['diagonal_error_px'] for r in records)<1e-8
    assert max(moving_errors)<1e-8 and max(affine_errors)<1e-8
    noisy=[]
    for angle in PARAMS['synthetic_angles_deg']:
        p=g.project(g.homography(g.rotation(angle,0),[0.,0.,3.]),g.TEMPLATE)
        jitter=p[None]+rng.normal(size=(PARAMS['synthetic_noise_repeats'],4,2))*PARAMS['synthetic_noise_sigma_px']
        c,sine=g.intersection(jitter);error=np.linalg.norm(c-[640,360],axis=1)
        noisy.append(dict(tilt_deg=angle,p50_error_px=float(np.median(error)),p95_error_px=float(np.quantile(error,.95)),finite_fraction=float(np.isfinite(error).mean())))
    result=dict(records=records,noisy=noisy,summary=dict(stationary_true_center=[640.,360.],max_hub_mean_false_motion_px=max(r['mean_error_px'] for r in records),
        max_diagonal_error_px=max(r['diagonal_error_px'] for r in records),max_ellipse_diagonal_error_px=max(r['ellipse_diagonal_error_px'] for r in records),
        max_ellipse_center_vs_projected_hub_px=max(r['max_ellipse_vs_hub_px'] for r in records),moving_control_max_error_px=max(moving_errors),
        affine_mean_max_error_px=max(affine_errors),limits='Chosen synthetic f=900, arm components0.45, radius0.16, range3; not calibrated to actual aircraft, not a measured dataset error.'))
    save(OUT/'synthetic.json',result);return result

def real():
    result=[];rng=np.random.default_rng(PARAMS['seed'])
    with (OUT/'current_diagnostics.jsonl').open('x',encoding='utf-8') as stream:
        for case,(start,duration) in CASES.items():
            sensing={round(r['t'],8):r for r in rows(SENSORS/case/'sensing.jsonl')}
            position=[r for r in rows(POSITIONS/case/'observations.jsonl') if start<r['t']<=start+duration]
            case_rows=[]
            for row in position:
                t=row['t'];s=sensing[round(t,8)]['methods']['replace_stale'];rotors=[r for r in s['rotors'] if r['observed_now']]
                body=row['methods']['body_only']['observation'];box=body.get('bbox') if body.get('detected') else None
                candidates={'all_current':rotors}
                if box is None:candidates['body_associated']=[]
                else:
                    b=np.array(box,float);margin=(b[2:]-b[:2])*PARAMS['association_margin_fraction']
                    candidates['body_associated']=[r for r in rotors if np.all(np.array([r['cx'],r['cy']])>=b[:2]-margin) and np.all(np.array([r['cx'],r['cy']])<=b[2:]+margin)]
                branches={}
                for method,rr in candidates.items():
                    d=g.diagnostics([[r['cx'],r['cy']] for r in rr],[r.get('center_sigma_px',2.) for r in rr],rng,PARAMS['real_jitter_repeats'])
                    d.update(current_hubs=len(rr),all_ellipse=bool(len(rr)==4 and all(r['mode']=='ellipse' for r in rr)))
                    if d['accepted_geometry'] and box is not None:
                        diagonal=np.linalg.norm(np.array(box[2:])-box[:2]);d['distance_to_body_box_center_px']=float(np.linalg.norm(np.array(d['center'])-body['center']));d['body_box_diagonal_px']=float(diagonal)
                    branches[method]=d
                record=dict(case=case,t=t,branches=branches);case_rows.append(record);stream.write(json.dumps(record,allow_nan=False)+'\n')
            for method in ['all_current','body_associated']:
                rr=[r['branches'][method] for r in case_rows];accepted=[r for r in rr if r['accepted_geometry']];dist=[r['distance_to_body_box_center_px'] for r in accepted if 'distance_to_body_box_center_px' in r]
                result.append(dict(case=case,method=method,frames=len(rr),four_current=sum(r['current_hubs']==4 for r in rr),convex_quad=len(accepted),
                    ill_conditioned=sum(r['diagonal_sine']<PARAMS['ill_conditioned_diagonal_sine'] for r in accepted),
                    median_center_vs_hub_mean_px=float(np.median([r['centroid_difference_px'] for r in accepted])) if accepted else None,
                    median_assumed_noise_p95_px=float(np.median([r['assumed_noise_center_p95_px'] for r in accepted])) if accepted else None,
                    median_distance_to_body_box_center_px=float(np.median(dist)) if dist else None,
                    interpretation='Availability/consistency only. No hub identity, geometric-center ground truth or future accuracy claim.'))
    save(OUT/'real_summary.json',dict(table=result,total_frames=sum(r['frames'] for r in result if r['method']=='all_current'),evaluation_labels_read=False));return result

def report(s,r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained');rr=s['records'];sample=max(rr,key=lambda x:x['mean_error_px']);p=np.array(sample['hubs']);closed=np.vstack([p,p[0]])
    axes[0].plot(closed[:,0],closed[:,1],'o-',color='#277da1',label='Projected hubs')
    for a,b in [(0,2),(1,3)]:axes[0].plot(p[[a,b],0],p[[a,b],1],color='#8ecae6')
    axes[0].scatter(*sample['true_center'],s=90,c='#1b4332',marker='+',label='True / diagonal center',zorder=5)
    axes[0].scatter(*sample['hub_mean'],s=45,c='#d55e00',label='Mean of four hubs',zorder=5)
    axes[0].invert_yaxis();axes[0].set_aspect('equal');axes[0].set_title('Fixed center, changing attitude');axes[0].legend(fontsize=8);axes[0].set_xlabel('Image x (px)');axes[0].set_ylabel('Image y (px)')
    t=[x['t'] for x in rr]
    for key,label,color in [('mean_error_px','Hub mean','#d55e00'),('ellipse_diagonal_error_px','Ellipse-center diagonals','#7b2cbf'),('diagonal_error_px','True hub diagonals','#1b4332')]:axes[1].plot(t,[x[key] for x in rr],label=label,color=color)
    axes[1].set_title('Apparent center displacement');axes[1].set_xlabel('Synthetic time (s)');axes[1].set_ylabel('Error to fixed projected center (px)');axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    n=s['noisy'];axes[2].plot([x['tilt_deg'] for x in n],[x['p95_error_px'] for x in n],'o-',color='#277da1');axes[2].set_yscale('log');axes[2].set_title('Near edge-on: noise amplification');axes[2].set_xlabel('Plane tilt from front view (degrees)');axes[2].set_ylabel('95th percentile center error (px)');axes[2].grid(alpha=.2)
    fig.suptitle('Controlled geometry only: 0.75 px hub noise; not measured UAV accuracy',fontsize=12)
    fig.savefig(OUT/'rigid_center_diagnostic.png',dpi=160);plt.close(fig)
    cols=['case','method','frames','four_current','convex_quad','ill_conditioned','median_center_vs_hub_mean_px','median_assumed_noise_p95_px','median_distance_to_body_box_center_px']
    table='<table><tr>'+''.join('<th>'+x+'</th>' for x in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(round(row[x],3) if isinstance(row[x],float) else row[x]))+'</td>' for x in cols)+'</tr>' for row in r)+'</table>'
    summary=s['summary'];refs=' · '.join('<a href="'+url+'">'+title+'</a>' for title,url in REFERENCES)
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>49：五圆刚体与中心分离诊断</title><style>body{max-width:1250px;margin:30px auto;font:16px/1.8 Arial;color:#233e4c}img{width:100%}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border:1px solid #cbd5dd;padding:5px}</style><h1>49：中心不动，四桨仍可移动——先分开平移和姿态</h1><p>依据用户五圆简化模型做受控几何验证，并检查650时刻自采旧缓存的当前四桨候选。不是新检测器、不是新的未来预测提升，也没有把真实无人机强制设为静止。</p><h2>受控验证</h2><p>同一3D中心固定，平面转动；四轴心为对称矩形。真实轴心投影的两对角线交点恢复中心，透视下四点平均会产生假移动；弱透视均值控制与真实平移控制均通过。此例相机/尺寸/距离是人为设定，不能当实采误差。</p><img src="rigid_center_diagnostic.png"><pre>'''+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'''</pre><p>透视投影圆的椭圆中心不等于真实轴心投影，连用椭圆中心求交点也有偏差，需把圆锥投影和轴心区别开。侧视退化时两对角线接近重合，定位噪声被放大；不能保证侧视成功。</p><h2>现有真实数据：可用程度而非真值精度</h2>'''+table+'''<p>all_current使用当前实际可见四候选；body_associated仅保留当前NN整机框扩大25%内的候选。对四点凸包配对只是几何假设，不证明四点确为四个不同转子。任意四个一般位置点都能拟合一个单应性，零拟合残差本身没有证据力。框中心只是可见外观一致性参照，不是机架几何中心/质心真值。噪声P95使用旧拟合自报sigma，未校准，不是概率置信保证。</p><h2>实现方向</h2><p>状态分为中心平移和机架平面姿态；四桨相对中心的运动用于姿态，共同位移用于平移。四桨清楚时建立布局，遮挡时用刚体模型短时预测并增大不确定性；出现当前新证据才恢复观测，禁止把推测点当真实测量反复训练。中心圈可独立测到时才提供额外校验，四点加任意单应性不够。中心静止只是诊断场景，不是运行先验；姿态变化不必周期。下步需先提高真实轴心与对应关系的可靠性，再测未来预测收益。</p><p>复用标准射影几何/NumPy/OpenCV凸包；未执行IPPE或ICRA完整事件三维追踪器（它们需标定/对应或三维模型），不称其复现或新颖算法。</p><p>'''+refs+'''</p><p><a href="protocol.json">冻结协议</a> · <a href="real_summary.json">实采诊断</a> · <a href="synthetic.json">受控数据</a></p></html>'''
    (OUT/'第四十九阶段刚体中心分离诊断.html').write_text(page,encoding='utf-8')
    save(OUT/'completion.json',dict(source_hashes_unchanged=sources()==load(OUT/'protocol.json')['source_hashes'],total_real_current_frames=650,no_evaluation_truth_read=True,
        stationary_control=True,moving_control=True,weak_perspective_control=True,limits='No future forecast improvement or true hub identity measured.'))
    print(json.dumps(dict(synthetic=summary,real=r),ensure_ascii=False),flush=True)
def run():
    sys.addaudithook(guard);verify();assert not (OUT/'synthetic.json').exists(),'Do not overwrite experiment.'
    s=synthetic();r=real();report(s,r);verify()
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','run']);globals()[a.parse_args().action]()
