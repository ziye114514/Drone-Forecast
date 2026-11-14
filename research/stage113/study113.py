"""Frozen observed-contour versus inferred-perimeter geometry/forecast pilot."""
from pathlib import Path
import argparse,collections,copy,datetime,hashlib,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage113'),str(ROOT/'research/stage112'),str(ROOT/'research/stage110'),str(ROOT/'research/stage107')]
import support113 as geom
from capture112 import OUT as SUPPORT
from study110 import OUT as OLD,CASES
from audit109 import rows,save,sha
from center107 import CenterBank,NAMES
OUT=ROOT.parent/'output/stage113_joint_observed_arcs_20261022'
PARAMS=geom.PARAMS

def sources():
    paths=[Path(__file__),ROOT/'research/stage113/support113.py',ROOT/'research/stage112/capture112.py',
        ROOT/'research/stage58/geometry58.py',ROOT/'research/stage104/rigid104.py',ROOT/'research/stage107/center107.py',
        ROOT/'src/event_drone_forecast/geometry.py',ROOT/'research/stage110/delivery110.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}

def freeze():
    assert not OUT.exists()
    save(OUT/'protocol113.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources(),parameters=PARAMS,
        inputs={case:dict(support_sha256=sha(SUPPORT/case/'supports.jsonl'),reference_sha256=sha(OLD/case/'records.jsonl')) for case in CASES},
        derived_sha256=hashlib.sha256(geom.DERIVED_SOURCE.encode()).hexdigest(),
        design='Five matched branches: frozen110 inferred full-perimeterFD; same objective analyticJac; actual contours linear; curved actual support soft_l1; same arc fit with full-center uncertainty/bound/coverage admission. All original104 geometry gates/5-frame state and107 center banks unchanged. No parameter search or default promotion.',
        source_choice='Same exact112 selected measurements and current NN body/axis. Only what geometry objective receives and extra guarded admission change. Current contours are processed event support, not independently verified physical blade edges.',
        controls='Analytic Jacobian versus central numerical differences; transformed FD objective exact original58; known projective-circle full/partial/chord/edge-on contour-layer cases, not sensor simulation. Independent RAW112 prefixes plus independent113 cached prefix including all states.',
        evaluation='Same development videos/overlapping side and continuous windows; forecast before reading sparse BOX targets. No COM/RPM/pose GT, no blind or realtime claim.',
        related=dict(ellipse_covariance='https://github.com/zygmuntszpak/guaranteed-ellipse-fitting-with-a-confidence-region-and-an-uncertainty-measure',
            arc_selection='https://link.springer.com/article/10.1186/s41074-016-0007-y',solver='https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html'),
        related_scope='Prior-method review; author MATLAB covariance code was not run/ported. Actual reuse is existing project geometry helpers, compiled58 objective, original104/107 and installed SciPy/OpenCV.',default_promoted=False))
    (OUT/'derived_support_fit113.py').write_text(geom.DERIVED_SOURCE,encoding='utf-8')

def verify():
    p=json.loads((OUT/'protocol113.json').read_text('utf-8'));assert p['sources']==sources()
    assert p['derived_sha256']==hashlib.sha256(geom.DERIVED_SOURCE.encode()).hexdigest();return p

def stable(x):
    if isinstance(x,dict):return {k:stable(v) for k,v in x.items() if k not in ['elapsed_ms','processing_ms']}
    if isinstance(x,list):return [stable(v) for v in x]
    return x

def controls():
    verify();rng=np.random.default_rng(113);summary=[];jac_errors=[];records=[]
    for case in ['full','partial','occlusion_chord','edge_on']:
        H=np.array([[48.,4.,320.],[-2.,32. if case!='edge_on' else 3.,240.],[.045,-.025,1.]])
        aspect=1.15;radius=.32;phase=np.linspace(0,2*np.pi,96,endpoint=False);supports=[];ellipses=[]
        for xy in np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])*[1.,aspect]:
            th=phase if case in ['full','edge_on'] else phase[(phase>.25*np.pi)&(phase<1.75*np.pi)]
            uv=xy+radius*np.c_[np.cos(th),np.sin(th)]
            p=np.c_[uv,np.ones(len(uv))]@H.T;p=p[:,:2]/p[:,2,None]
            if case=='occlusion_chord':p=np.r_[p,np.linspace(p[-1],p[0],24)[1:-1]]
            p=p+rng.normal(0,.3,p.shape);supports.append(p)
            ellipses.append(cv2.fitEllipseDirect(np.float32(p)))
        # Original objective with numerical differences must remain unchanged.
        a=geom.geometry58.joint_fit(ellipses);b=geom.support_fit(ellipses,use_analytic=False)
        assert {k:stable(v) for k,v in a.items() if k!='elapsed_ms'}=={k:stable(b[k]) for k in a if k!='elapsed_ms'}
        for mode in ['perimeter_analytic','contour_linear','arc_robust']:
            points=None if mode=='perimeter_analytic' else geom.supports_for(supports,mode=='arc_robust')
            f=geom.support_fit(ellipses,points,'soft_l1' if mode=='arc_robust' else 'linear')
            error=float(np.linalg.norm(np.asarray(f['geometric_center'])-H[:2,2]));u=f['uncertainty']
            summary.append(dict(case=case,mode=mode,center_error_px=error,optimizer_success=f['optimizer_success'],
                sigma_px=u['center_sigma_px'],shape_on_bound=u['shape_on_bound'],rms_px=f['rms_px']))
            records.append(dict(case=case,mode=mode,fit=f))
            if case=='full':
                actual=geom.supports_for(supports) if points is None else points
                origin=np.asarray(f['origin_px']);scale=f['scale_px'];order=f['rotor_order']
                ep=[np.c_[(np.asarray(actual[i])-origin)/scale,np.ones(len(actual[i]))] for i in order]
                template=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
                x=np.asarray(f['parameter_vector'])
                def residual(p):
                    A=np.linalg.inv(np.r_[p[:8],1.].reshape(3,3));r=np.exp(p[8]);asp=np.exp(p[9]);out=[]
                    for xy,edge in zip(template,ep):
                        xx,yy=xy*[1.,asp];C=np.array([[1.,0.,-xx],[0.,1.,-yy],[-xx,-yy,xx*xx+yy*yy-r*r]])
                        cp=edge@(A.T@C@A);out.extend(np.sum(cp*edge,axis=1)/np.maximum(2*np.linalg.norm(cp[:,:2],axis=1),1e-12)*scale)
                    return np.array(out)
                numerical=[]
                for j in range(10):
                    step=1e-6*max(1.,abs(x[j]));xp=x.copy();xm=x.copy();xp[j]+=step;xm[j]-=step
                    numerical.append((residual(xp)-residual(xm))/(2*step))
                numerical=np.column_stack(numerical);analytical=geom.analytic_jacobian(x,ep,template,scale)
                difference=float(np.linalg.norm(numerical-analytical)/max(np.linalg.norm(numerical),1e-12));assert difference<1e-6,difference;jac_errors.append(difference)
    save(OUT/'controls113.json',dict(summary=summary,records=records,original_FD_exact=True,analytic_jacobian_relative_errors=jac_errors,
        scope='Known noisy projected circle contour inputs, including partial/chord/near-edge-on. Not native sensor or neural detection accuracy. Recorded failures not hidden.'))
    print(json.dumps(summary),flush=True)

def run(case,prefix=False):
    protocol=verify();folder=OUT/(case+('_prefix' if prefix else ''));folder.mkdir(exist_ok=False)
    assert protocol['inputs'][case]==dict(support_sha256=sha(SUPPORT/case/'supports.jsonl'),reference_sha256=sha(OLD/case/'records.jsonl'))
    source=SUPPORT/(case+('_prefix' if prefix else ''))/'supports.jsonl';rr=rows(source);reference={round(r['t'],8):r for r in rows(OLD/case/'records.jsonl')}
    models={mode:(geom.MeasuredCenter(),CenterBank()) for mode in PARAMS['modes']};result=[];cost=[]
    with (folder/'records.jsonl').open('x',encoding='utf-8') as stream:
        for r in rr:
            tick=time.perf_counter();rotors=r['geometry'].get('rotors',[]);old=reference[round(r['t'],8)]['outputs']['merge_safe'];oldm=old['center']['measurement']
            measures={};oldfit=oldm.get('fit')
            baseline=geom.measure_fitted(rotors,r['body'],oldfit) if oldfit is not None else copy.deepcopy(oldm)
            measures['perimeter_fd']=baseline
            if oldfit is not None and len(r['support_points'])==4 and all(len(p)>=6 for p in r['support_points']):
                ellipses=[geom.geometry58.from_rotor(z) for z in rotors]
                for mode in ['perimeter_analytic','contour_linear','arc_robust']:
                    points=None if mode=='perimeter_analytic' else geom.supports_for(r['support_points'],mode=='arc_robust')
                    fit=geom.support_fit(ellipses,points,'soft_l1' if mode=='arc_robust' else 'linear')
                    if fit.get('valid'):
                        fit['support_coverage']=geom.coverage(fit,r['support_points'] if points is None else points)
                        fit['points_per_rotor']=[len(p) for p in (r['support_points'] if points is None else points)]
                    measures[mode]=geom.measure_fitted(rotors,r['body'],fit)
                measures['arc_guarded']=geom.guarded(measures['arc_robust'])
            else:
                for mode in PARAMS['modes'][1:]:measures[mode]=dict(baseline,valid=False,reason='missing_observed_support') if oldfit is not None else copy.deepcopy(baseline)
            outputs={}
            for mode,(center,bank) in models.items():
                center.input_measurement=measures[mode];obs,d=center.step(r['t'],r['body'],rotors,r['axis']);q=bank.step(r['t'],r['body'],d,r['axis'])
                outputs[mode]=dict(center=d,forecast=q)
            assert outputs['perimeter_fd']['forecast']==old['forecast'],(case,r['t'],'baseline bank changed')
            assert stable(outputs['perimeter_fd']['center'])==stable(old['center']),(case,r['t'],'baseline state changed')
            body=[q['forecast']['coefficients'].get(NAMES[0]) for q in outputs.values()];assert all(v==body[0] for v in body)
            q=dict(t=r['t'],body=r['body'],axis=r['axis'],valid=r['valid'],outputs=outputs)
            stream.write(json.dumps(q,allow_nan=False)+'\n');result.append(q);cost.append((time.perf_counter()-tick)*1000)
            if len(result)%100==0:print(f'{case} fitted {len(result)}',flush=True)
    save(folder/'checks113.json',dict(frames=len(rr),original110_center_and_forecasts_exact=True,body_KF_all_modes_exact=True,
        current_support_cache=True,new_RAW=False,labels_read=False,source_unchanged=verify()['sources']==sources(),
        five_banks_and_three_fit_median_ms=float(np.median(cost))))
    if prefix:
        full=rows(OUT/case/'records.jsonl')[:len(result)];assert stable(full)==stable(result)
        save(folder/'prefix113.json',dict(independent_current_support_prefix=True,frames=len(rr),all_states_and_forecasts_exact=True,actual_RAW_prefix_in_stage112=True))
    print(json.dumps(dict(case=case,prefix=prefix,frames=len(rr))),flush=True)

def summarize():
    verify();summary=[]
    for case in CASES:
        rr=rows(OUT/case/'records.jsonl')
        for mode in PARAMS['modes']:
            states=[r['outputs'][mode]['center'] for r in rr]
            summary.append(dict(case=case,mode=mode,frames=len(rr),qualified=sum(r['measurement']['valid'] for r in states),
                current_centers=sum(r['current'] for r in states),initializations=sum(r['initialized_now'] for r in states),
                reasons=dict(collections.Counter(r['measurement']['reason'] for r in states)),
                shape_bound_fits=sum(r['measurement'].get('fit',{}).get('uncertainty',{}).get('shape_on_bound',False) for r in states)))
    save(OUT/'availability113.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','controls','run','prefix','summarize']);p.add_argument('--case',choices=CASES)
    a=p.parse_args()
    if a.action in ['freeze','controls','summarize']:globals()[a.action]()
    else:run(a.case,a.action=='prefix')
