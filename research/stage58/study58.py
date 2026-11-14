"""Pinned-author center correction and intrinsic-free joint-conic diagnostics."""
from pathlib import Path
import argparse,datetime,hashlib,html,json,os,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage57'),str(ROOT/'research/stage49')]
import study57 as previous
import rigid49 as rigid
import geometry58 as geom
OUT=ROOT.parent/'output/stage58_projected_centers_20261021'
CASES=['clear6','side6','train3']
PARAMS=dict(seed=58,angles_deg=[0,30,60,75,82,87,89],noise_trials=[dict(sigma_px=0.,repeats=1),dict(sigma_px=.75,repeats=3)],
    contour_points=200,radius=.16,translation=[0.,0.,3.],assumed_focals=[600.,900.,1200.],real_sample_stride=10,
    stress='Deterministic last ellipse major*1.4/minor*.6/angle+30deg, centers unchanged; generated inconsistency, not actual false-detection truth.',
    author_selector='For primary circle0 use farthest other ellipse center for the existing equal-radius homography selector. No true center/normal branch oracle.',
    real_scope='Every10th native current tick, all cases. Full-ellipse/conic fit only; line/missing support remains explicitly in denominator. No real-K author solver, production pose, RPM, or forecast claim.')
load=previous.load;save=previous.save;sha=previous.sha;rows=previous.rows

def sources():
    result=previous.sources()
    for p in [Path(__file__),Path(__file__).with_name('geometry58.py'),ROOT/'research/stage49/rigid49.py',geom.VENDOR/'UPSTREAM_SNAPSHOT.json']:
        result[p.relative_to(ROOT).as_posix()]=sha(p)
    snapshot=geom.verify_vendor()
    for r in snapshot['files']:
        p=geom.VENDOR/r['path'];result[p.relative_to(ROOT).as_posix()]=r['sha256']
    return result

def inputs():return {c:sha(previous.OUT/c/'rotor_ticks.jsonl') for c in CASES}

def freeze():
    assert not OUT.exists();previous.verify();upstream=geom.verify_vendor()
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),input_hashes=inputs(),
        parameters=PARAMS,geometry_parameters=geom.PARAMS,upstream_commit=upstream['commit'],upstream_license='Apache-2.0, LICENSE and NOTICE retained; optional thirdparty AAMED not downloaded or executed.',
        design='Two separate tests: unchanged upstream center solver ONLY on known-K undistorted synthetic circles; independent-K joint homography+unknown radius/frame aspect fitted to all four conics on controlled and cached actual current data. No manual true center or future data enters either estimator.',
        missing_calibration='Checked self-recording metadata and dataset filename inventory. train6 calibration/homography.json maps event to AVI, not intrinsic K/distortion. External evperiscope640x480 K is not this1280x720 camera and is not used.',
        diagnostics='Record all failures, incomplete optimization and near-edge-on conditioning; do not use internal residual as certified identity/pose accuracy. Uniform real sampling, original candidates untouched. Synthetic perturbation keeps four centers so point-homography residual is uninformative.',
        prior_art=['https://arxiv.org/html/2511.06611v2','https://github.com/xiahaa/circular-center-calibration/tree/168b85c1cb241937a0c3e563ef382c20ad0be456','https://link.springer.com/article/10.1186/s41074-018-0050-y'],
        scope='Geometry prototype, not full new RAW forecast. Current54/57 deployment and thresholds unchanged. Unknown actual lens rectification prevents claiming a calibrated real-circle-center correction.'))
    print('Stage58 frozen: pinned author solver, known-K controls and current-only conic diagnostic',flush=True)

def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==sources() and p['input_hashes']==inputs();previous.verify();return p

def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).resolve()
        if p.name in ['observations_forecasts.jsonl','forecasts.jsonl','targets.json'] or p.is_relative_to(ROOT.parent/'output/raw_only_project/evaluation_data'):
            raise RuntimeError('Geometry experiment cannot read forecasts/evaluation truth '+str(p))

def distort_shape(ellipses):
    result=list(ellipses);e=result[-1];result[-1]=(e[0],(e[1][0]*1.4,e[1][1]*.6),e[2]+30.);return result

def synthetic():
    verify();assert not (OUT/'synthetic.json').exists();sys.addaudithook(guard);rng=np.random.default_rng(PARAMS['seed']);results=[]
    phase=np.linspace(0,2*np.pi,PARAMS['contour_points'],endpoint=False);circle=np.c_[np.cos(phase),np.sin(phase)]*PARAMS['radius']
    for cfg in PARAMS['noise_trials']:
        for angle in PARAMS['angles_deg']:
            H=rigid.homography(rigid.rotation(angle,0.,12.),PARAMS['translation']);truth=rigid.project(H,rigid.TEMPLATE)
            for trial in range(cfg['repeats']):
                contours=[rigid.project(H,center+circle)+rng.normal(size=circle.shape)*cfg['sigma_px'] for center in rigid.TEMPLATE]
                ellipses=[cv2.fitEllipse(p.astype(np.float32)) for p in contours]
                centers=np.array([e[0] for e in ellipses]);farthest=int(np.argmax(np.linalg.norm(centers[1:]-centers[0],axis=1)))+1
                joint=geom.joint_fit(ellipses);stress=geom.joint_fit(distort_shape(ellipses));author=[]
                for focal in PARAMS['assumed_focals']:
                    K=rigid.K.copy();K[0,0]=K[1,1]=focal
                    try:
                        a=geom.author_center(ellipses[0],ellipses[farthest],K,contours[0],contours[farthest])
                        a.update(focal=focal,error_px=float(np.linalg.norm(np.asarray(a['center'])-truth[0])),
                            best_candidate_oracle_error_px=float(np.min(np.linalg.norm(np.asarray(a['candidates'])-truth[0],axis=1))))
                    except Exception as error:a=dict(focal=focal,failed=True,error=type(error).__name__+': '+str(error))
                    author.append(a)
                if joint.get('valid'):
                    joint['hub_error_mean_px']=float(np.mean(np.linalg.norm(np.asarray(joint['projected_hubs'])-truth,axis=1)))
                    joint['true_center_error_px']=float(np.linalg.norm(np.asarray(joint['geometric_center'])-rigid.project(H,[[0.,0.]])[0]))
                results.append(dict(noise_sigma_px=cfg['sigma_px'],rx_deg=angle,trial=trial,ellipse_center_error_mean_px=float(np.mean(np.linalg.norm(centers-truth,axis=1))),
                    primary_ellipse_center_error_px=float(np.linalg.norm(centers[0]-truth[0])),joint=joint,stress_joint=stress,author=author,
                    true_hubs=truth.tolist(),ellipse_centers=centers.tolist(),input_second_index=farthest))
                print(f'synthetic noise={cfg["sigma_px"]} rx={angle} trial={trial} joint_rms={joint.get("rms_px")} success={joint.get("optimizer_success")}',flush=True)
    assert len(results)==28
    save(OUT/'synthetic.json',dict(records=results,rows=len(results),source_hashes_unchanged=sources()==load(OUT/'protocol.json')['source_hashes'],oracle_used_for_evaluation_only=True))

def real():
    verify();assert not (OUT/'real.json').exists();sys.addaudithook(guard);result=[]
    for case in CASES:
        rr=rows(previous.OUT/case/'rotor_ticks.jsonl')
        for index in range(PARAMS['real_sample_stride']-1,len(rr),PARAMS['real_sample_stride']):
            r=rr[index];rotors=r['observation'].get('rotors',[]);record=dict(case=case,t=r['t'],current_rotor_candidates=len(rotors),actual_K_known=False)
            if len(rotors)!=4:record.update(eligible=False,reason='not_four_current')
            elif any(x.get('mode')!='ellipse' for x in rotors):record.update(eligible=False,reason='not_four_full_ellipses')
            else:
                ellipses=[geom.from_rotor(x) for x in rotors]
                record.update(eligible=True,joint=geom.joint_fit(ellipses),stress_joint=geom.joint_fit(distort_shape(ellipses)),
                    original_hubs=[[x['cx'],x['cy']] for x in rotors],interpretation='Pinhole/copanar equal-circle hypothesis on current fitted ellipses; no distortion calibration or actual center truth. Author known-K correction not applied.')
            result.append(record)
        print(f'{case}: current samples={sum(r["case"]==case for r in result)}',flush=True)
    assert len(result)==130
    save(OUT/'real.json',dict(records=result,sampled_current_ticks=130,no_future_truth_read=True,new_frequency_or_forecast_executed=False,source_hashes_unchanged=sources()==load(OUT/'protocol.json')['source_hashes']))

def report():
    verify();s=load(OUT/'synthetic.json');r=load(OUT/'real.json');summary=[]
    for noise in [0.,.75]:
        for angle in PARAMS['angles_deg']:
            selected=[x for x in s['records'] if x['noise_sigma_px']==noise and x['rx_deg']==angle]
            ok=[x for x in selected if x['joint'].get('valid') and x['joint'].get('optimizer_success') and x['joint'].get('same_projective_halfspace')]
            known=[a for x in selected for a in x['author'] if a['focal']==900. and not a.get('failed')]
            summary.append(dict(noise_sigma_px=noise,rx_deg=angle,trials=len(selected),joint_usable=len(ok),
                ellipse_hub_error_px=float(np.mean([x['ellipse_center_error_mean_px'] for x in selected])),
                joint_hub_error_px=float(np.mean([x['joint']['hub_error_mean_px'] for x in ok])) if ok else None,
                author_primary_knownK_error_px=float(np.mean([a['error_px'] for a in known])) if known else None,
                author_failed=len(selected)-len(known),joint_rms_px=float(np.mean([x['joint']['rms_px'] for x in ok])) if ok else None,
                stress_rms_px=float(np.mean([x['stress_joint']['rms_px'] for x in selected if x['stress_joint'].get('valid')]))))
    real=[]
    for case in CASES:
        rr=[x for x in r['records'] if x['case']==case];eligible=[x for x in rr if x['eligible']];ok=[x for x in eligible if x['joint'].get('valid') and x['joint'].get('optimizer_success') and x['joint'].get('same_projective_halfspace')]
        real.append(dict(case=case,sampled_ticks=len(rr),four_full_ellipse_ticks=len(eligible),joint_usable=len(ok),
            median_rms_px=float(np.median([x['joint']['rms_px'] for x in ok])) if ok else None,
            median_stress_rms_px=float(np.median([x['stress_joint']['rms_px'] for x in ok if x['stress_joint'].get('valid')])) if ok else None,
            median_processing_ms=float(np.median([x['joint']['elapsed_ms'] for x in eligible if x['joint'].get('valid')])) if eligible else None))
    save(OUT/'analysis.json',dict(synthetic=summary,real=real,actual_upstream_executed=True,production_changed=False,known_K_only_synthetic=True,limits=load(OUT/'protocol.json')['scope']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for noise,color in [(0.,'#277da1'),(.75,'#e76f51')]:
        selected=[x for x in summary if x['noise_sigma_px']==noise]
        for key,label,style in [('ellipse_hub_error_px','Ellipse centers','--'),('joint_hub_error_px','Joint four conics','-')]:
            axes[0].plot([x['rx_deg'] for x in selected],[x[key] if x[key] is not None else np.nan for x in selected],style+'o',label=f'{label}, noise{noise}',color=color)
        axes[1].plot([x['rx_deg'] for x in selected],[x['joint_usable']/x['trials'] for x in selected],'o-',label=f'noise{noise}',color=color)
    axes[0].set_ylabel('Mean true projected-hub error (px)');axes[0].set_yscale('symlog',linthresh=.1);axes[1].set_ylabel('Finite/converged/same-halfspace fraction');axes[1].set_ylim(-.05,1.05)
    for ax in axes[:2]:ax.set_xlabel('Synthetic plane tilt (degrees)');ax.legend(fontsize=7);ax.grid(alpha=.2)
    for focal,color in [(600.,'#277da1'),(900.,'#43aa8b'),(1200.,'#e76f51')]:
        selected=[(x['rx_deg'],a['error_px']) for x in s['records'] if x['noise_sigma_px']==0 for a in x['author'] if a['focal']==focal and not a.get('failed')]
        axes[2].plot([x[0] for x in selected],[x[1] for x in selected],'o-',label=f'assumed f{focal:g}',color=color)
    axes[2].set_xlabel('Synthetic plane tilt (degrees)');axes[2].set_ylabel('Author-selected primary center error (px)');axes[2].legend(fontsize=8)
    fig.suptitle('Geometry prototype only: known synthetic truth; real current ellipses have no center/pose labels',fontsize=11);fig.savefig(OUT/'projected_center_controls.png',dpi=160);plt.close(fig)
    def tab(rr,cols):return '<table><tr>'+''.join('<th>'+x+'</th>' for x in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(round(row[x],5) if isinstance(row[x],float) else row[x]))+'</td>' for x in cols)+'</tr>' for row in rr)+'</table>'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>58：投影圆心与共同桨盘几何</title><style>body{max-width:1450px;margin:28px auto;font:16px/1.8 Arial;color:#233e4c}img{width:100%}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #ccd8df}</style><h1>58：复用作者圆心校正，验证共同四桨圆锥约束</h1><p>固定TIM2026作者库提交168b85c1，原23个源/协议/许可文件逐一Git blob与SHA校验，无依赖安装。作者原圆心两候选及第二共面等半径圆消歧仅用于已知K、无畸变的受控投影；没有用真值挑选分支。另用一般单应性H、未知共同半径及未知矩形机架比例联合拟合四个完整圆锥，增加不同于任意四点单应性的约束。该联合拟合是标准射影模型的独立诊断实现，不称新算法。</p><img src="projected_center_controls.png">'''+tab(summary,list(summary[0]))+'''<h2>真实自采当前观测：内部一致性，不是真值准确率</h2>'''+tab(real,list(real[0]))+'''<p>三个57实际RAW结果每10个原生时刻固定取1个，共130个当前时刻；缺四点/缺完整椭圆均计入分母，不填桨心。实采未找到可信K和畸变参数，作者已标定圆心函数没有用于RAW。当前联合拟合把检测椭圆当作共面等半径圆的投影假设，畸变、错误轮廓、非矩形机架都会破坏它；小残差不证明真实身份或姿态。stress是人为改变一个椭圆形状而保持四中心的诊断，不是实际误报真值。</p><p>所有失败、未收敛和近侧视条件数保存；没有改变54/57部署、频率/RLS/15模型，尚无新未来位置成绩。合成精度不能当实飞精度。</p><p><a href="analysis.json">统计</a> · <a href="synthetic.json">所有合成试验与失败</a> · <a href="real.json">所有当前样本</a> · <a href="protocol.json">冻结协议</a></p><p>原始研究与实际作者代码：<a href="https://arxiv.org/html/2511.06611v2">圆心投影校正TIM2026</a>、<a href="https://github.com/xiahaa/circular-center-calibration/tree/168b85c1cb241937a0c3e563ef382c20ad0be456">固定提交Apache2作者库</a>、<a href="https://link.springer.com/article/10.1186/s41074-018-0050-y">共面圆锥与单应性歧义</a>。LICENSE/NOTICE保持，未使用GPL的AAMED子模块。</p></html>'''
    (OUT/'第五十八阶段投影圆心与共同桨盘报告.html').write_text(page,encoding='utf-8')
    save(OUT/'completion.json',dict(source_hashes_unchanged=load(OUT/'protocol.json')['source_hashes']==sources(),synthetic_trials=28,real_current_ticks=130,production_changed=False,new_future_scoring=False))
    print(json.dumps(dict(synthetic=summary,real=real),ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','synthetic','real','report']);globals()[p.parse_args().action]()
