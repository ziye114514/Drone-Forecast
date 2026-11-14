"""Fixed center-bias controls, causal replay, actual RAW and sparse scoring."""
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'research/stage107'),str(ROOT/'research/stage93')]
from center107 import CenterBank,PARAMS,NAMES,VENDOR,SimpleKF
from online93 import Kinematic
import run_center_bias_forecast as entry
OUT=ROOT.parent/'output/stage107_center_bias_filter_20261022'
OLD=ROOT.parent/'output/stage104_joint_center_20261022'
SCREEN=ROOT.parent/'output/stage106_continuous_screen_20261022'
BASE=ROOT.parent/'output/stage93_RAW_complementary_20261022'
INPUTS=dict(clear6=OLD/'raw/clear6/observations_forecasts.jsonl',side6=OLD/'raw/side6/observations_forecasts.jsonl',
    continuous6=SCREEN/'raw/observations_forecasts.jsonl')
RAW_CASES=dict(clear6=dict(raw='E:/dataset/train/6/media/event.raw',start=22.,duration=5.),
    continuous6=dict(raw='E:/dataset/train/6/media/event.raw',start=110.,duration=9.5),
    prefix_clear6=dict(raw='E:/dataset/train/6/media/event.raw',start=22.,duration=1.2),
    negative5=dict(raw='E:/dataset/train/5/media/event.raw',start=0.,duration=.6))
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def load(p):return json.loads(Path(p).read_text('utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8') as f:return [json.loads(s) for s in f if s.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sources():
    ns=entry.build();r=ns['fingerprints']();ns.clear();r['research/stage107/study107.py']=sha(__file__);return r
def freeze():
    assert not OUT.exists();save(OUT/'protocol107.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),
        parameters=PARAMS,input_hashes={c:sha(p) for c,p in INPUTS.items()},raw_cases=RAW_CASES,
        design='Four explicit causal center controls: matched body-onlyKF,100ms correction EMA,6D center/velocity/appearance-bias KF fixed noise, same6D state with geometry/view covariance. Pinned FilterPy author predict/update actually executed, no arithmetic rewriting. No new force parameters.',
        selection='All constants fixed before new filter outputs or scores. Sigma6400 from original90 training-only selection, position6px from74. EMA.1s, bias diffusion30px/sqrt(s), side sigma5x and correlation.5 are hypotheses, not tuned/learned from labels.',
        synthetic=dict(seed=107,frames=150,dt=.02,body_noise_px=2.,geometry_noise_px=.5,
            cases=['stationary_wobble','translation_wobble','abrupt_turn','long_side_gap','no_geometry','accepted_bad_geometry'],
            scope='Known measurement-layer center/bias/noise, not event sensor rendering or an actual detected UAV. Relative image offset aligned once; controls include true motion and harmful accepted geometry.'),
        validation='Author4D functions match previous kinematic within numerical tolerance; deterministic independent prefixes exact. Original60 retained in realRAW. Models never read future targets.',
        evaluation='Existing sparse manual box-center endpoints for clear6/side6; continuous window targets loaded only after forecasts, not independent body COM truth. Keep degraded results and matched body/smoothing controls.',
        promotion='No automatic default replacement; a geometry/side branch may worsen physical turn prediction. Success requires independent evidence beyond this development pilot.',
        default_promoted=False));print('107 frozen',flush=True)
def verify():
    p=load(OUT/'protocol107.json');assert p['source_hashes']==sources()
    for c,v in p['input_hashes'].items():assert sha(INPUTS[c])==v
    return p


def synthetic():
    p=verify();rng=np.random.default_rng(107);n=p['synthetic']['frames'];dt=.02;records=[];summary=[]
    # Verify actual author mathematics against the old filter before changing
    # the state dimension. No appearance geometry in this regression.
    a=SimpleKF();b=Kinematic();maximum=0.
    for j in range(100):
        z=None if j in [20,21,22] else rng.normal(size=2)*20+j
        x=a.step(j*dt,z);y=b.step(j*dt,z)
        maximum=max(maximum,float(np.max(np.abs(np.r_[x['position'],x['velocity']]-np.r_[y['position'],y['velocity']]))))
    assert maximum<1e-8,maximum
    for case in p['synthetic']['cases']:
        bank=CenterBank();data=[];targets=[]
        for j in range(n):
            t=j*dt;c=np.array([320.,240.]);v=np.zeros(2)
            if case=='translation_wobble':c+=np.array([80.*t,-20.*t]);v=np.array([80.,-20.])
            if case=='abrupt_turn':c+=np.array([80.*min(t,1.2)-100.*max(t-1.2,0.),0.]);v=np.array([80. if t<1.2 else -100.,0.])
            bias=np.array([16.*np.sin(3*t),8.*(np.cos(2.5*t)-1.)]);z=c+bias+rng.normal(0,2.,2)
            available=j>=4 and case!='no_geometry' and not(case=='long_side_gap' and .8<=t<2.)
            measured=c+rng.normal(0,.5,2)
            if case=='accepted_bad_geometry' and j==65:measured+=np.array([40.,-30.])
            d=dict(current=available,model_geometric_center=measured.tolist() if available else None,
                correction_px=[0.,0.],measurement=dict(view_weight=.7,quality=.85,center_noise_proxy_px=.5,fit=dict(rms_px=.5)))
            body=dict(detected=True,center=z.tolist(),confidence=.9)
            # Initialize104-style alignment for the plain smoothing control;
            # missing measurements keep a short correction, not fictitious g.
            if available:
                if not targets:targets.append((z-measured).copy())
                d['correction_px']=(measured+targets[0]-z).tolist()
            axis=dict(valid=case=='long_side_gap' and not available and j>=4,angle_rad=0.,confidence=.9)
            q=bank.step(t,body,d,axis)
            data.append(dict(t=t,truth=c.tolist(),true_velocity=v.tolist(),body=z.tolist(),geometry=measured.tolist() if available else None,
                output=q,body_input=body,geometry_input=d,axis_input=axis))
        offset=targets[0] if targets else np.zeros(2)
        for method in NAMES:
            pos=[];speed=[];fut=[]
            for j,r in enumerate(data):
                if j<20:continue
                c=r['output']['coefficients'][method];pos.append(np.linalg.norm(np.asarray(c['position'])-r['truth']-offset))
                speed.append(np.linalg.norm(np.asarray(c['velocity'])-r['true_velocity']))
                if j+20<len(data):fut.append(np.linalg.norm(np.asarray(c['position'])+.4*np.asarray(c['velocity'])-data[j+20]['truth']-offset))
            summary.append(dict(case=case,method=method,position_rmse_px=float(np.sqrt(np.mean(np.square(pos)))),
                velocity_rmse_px_s=float(np.sqrt(np.mean(np.square(speed)))),fde04_mean_px=float(np.mean(fut))))
        prefix=CenterBank()
        for r in data[:80]:assert prefix.step(r['t'],r['body_input'],r['geometry_input'],r['axis_input'])==r['output']
        if case=='no_geometry':
            for r in data:assert r['output']['coefficients'][NAMES[0]]==r['output']['coefficients'][NAMES[2]]
        records.append(dict(case=case,rows=data))
    save(OUT/'synthetic107.json',dict(summary=summary,records=records,author_math_maximum_difference=maximum,
        independent_prefixes_exact=True,no_geometry_fixed_control_exact=True,scope=p['synthetic']['scope']))
    print(json.dumps(summary),flush=True)


def cached():
    verify();checks=[]
    for case,path in INPUTS.items():
        rr=rows(path);bank=CenterBank();result=[];cost=[]
        for r in rr:
            t=time.perf_counter();q=r['frontends']['delayed_feedback'];v=bank.step(r['t'],r['body_observation'],q['rigid_center_diagnostic'],r['body_axis']);cost.append((time.perf_counter()-t)*1000)
            result.append(dict(t=r['t'],valid=q['valid'],**v))
        save(OUT/'cached'/case/'forecasts.json',result)
        prefix=CenterBank()
        for r,q in zip(rr[:100],result):
            v=prefix.step(r['t'],r['body_observation'],r['frontends']['delayed_feedback']['rigid_center_diagnostic'],r['body_axis'])
            assert v['coefficients']==q['coefficients'] and v['diagnostics']==q['diagnostics']
        checks.append(dict(case=case,ticks=len(rr),prefix_exact=True,filter_median_ms=float(np.median(cost)),
            models_ever_geometric=sum(q['diagnostics']['observable'].get('has_geometry_reference',False) for q in result),
            current_geometric=sum(q['diagnostics']['observable'].get('current_geometry',False) for q in result),
            side_covariance=sum(q['diagnostics']['observable'].get('side_anisotropic_noise',False) for q in result)))
        print(json.dumps(checks[-1]),flush=True)
    save(OUT/'cached_checks107.json',checks)


def raw(case):
    cfg=verify()['raw_cases'][case];args=[sys.executable,str(ROOT/'run_center_bias_forecast.py'),'--raw',cfg['raw'],
        '--output',str(OUT/'raw'/case),'--start',str(cfg['start']),'--duration',str(cfg['duration']),'--verify-frames']
    if not case.startswith('prefix'):args.append('--preview')
    with (OUT/(case+'_parent.log')).open('x',encoding='utf-8') as f:
        result=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert result.returncode==0,(case,result.returncode);check_raw(case)
def check_raw(case):
    cfg=verify()['raw_cases'][case];rr=rows(OUT/'raw'/case/'observations_forecasts.jsonl');assert len(rr)==round(cfg['duration']*50)
    base=case.removeprefix('prefix_');n=len(rr)
    if base in INPUTS:
        old=rows(INPUTS[base])[:n];cached=load(OUT/'cached'/base/'forecasts.json')[:n]
        for r,o,c in zip(rr,old,cached):
            for k in ['observation','sensing','body_axis','event_activity','event_fields_sha256']:assert r[k]==o[k],(case,r['t'],k)
            q=r['frontends']['delayed_feedback'];ref=o['frontends']['delayed_feedback']
            for k,v in ref['coefficients'].items():assert q['coefficients'][k]==v,(case,r['t'],k)
            for k,v in c['coefficients'].items():
                if q['valid']:assert q['coefficients'][k]==v,(case,r['t'],k)
            assert q['center_bias_diagnostic']==c['diagnostics']
    if case.startswith('prefix'):
        full=rows(OUT/'raw'/base/'observations_forecasts.jsonl')[:n]
        for a,b in zip(rr,full):
            qa=a['frontends']['delayed_feedback'];qb=b['frontends']['delayed_feedback']
            for name in NAMES:
                if name in qa['coefficients']:assert qa['coefficients'][name]==qb['coefficients'][name]
            assert qa['center_bias_diagnostic']==qb['center_bias_diagnostic']
    if case=='negative5':assert all(not r['frontends']['delayed_feedback']['valid'] for r in rr)
    save(OUT/'raw'/case/'checks107.json',dict(case=case,parent=n,native10=n*2,actual_RAW=True,
        original60_exact=base in INPUTS,new4_exact_cached=base in INPUTS,actual_prefix=case.startswith('prefix'),no_labels_read=True))
    print(case,'actualRAW107 verified',flush=True)


def score():
    verify();reference=load(ROOT.parent/'output/stage78_RAW_separated_forecast_20261021/scoring/analysis.json')['details'];details=[];table=[]
    for case in ['clear6','side6']:
        pred={round(r['t'],8):r for r in load(OUT/'cached'/case/'forecasts.json')}
        old={round(r['t'],8):r['frontends']['delayed_feedback'] for r in rows(INPUTS[case])}
        for r in reference:
            if r['case']!=case or r['frontend']!='event_force' or r['model']!='hold' or r['target_state']!='present':continue
            q=pred.get(round(r['anchor_t'],8)) if r.get('anchor_available') else None
            for name in [*NAMES,'rigid_center_kf','complement_fallback']:
                coef=None
                if q is not None and q['valid']:
                    coef=q['coefficients'].get(name) if name in NAMES else old[round(r['anchor_t'],8)]['coefficients'].get(name)
                error=None
                if coef is not None:
                    from hybrid74 import prior
                    error=float(np.linalg.norm(np.asarray(prior.evaluate(coef,r['actual_h']))-r['target_center']))
                details.append(dict(case=case,method=name,h=r['nominal_h'],error_px=error,valid=coef is not None,target_t=r['target_t']))
    for case in ['clear6','side6']:
        for h in [.1,.2,.4]:
            for method in [*NAMES,'rigid_center_kf','complement_fallback']:
                group=[r for r in details if r['case']==case and r['h']==h and r['method']==method];values=[r['error_px'] for r in group if r['valid']]
                table.append(dict(case=case,h=h,method=method,n=len(values),targets=len(group),mean_px=None if not values else float(np.mean(values))))
    save(OUT/'scores107.json',dict(table=table,details=details,scope='Same sparse future BOX centers, not independent COM/pose ground truth. Continuous screening has no new manual target scoring.'))
    print(json.dumps([r for r in table if r['h']==.4]),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','synthetic','cached','raw','check_raw','score']);p.add_argument('--case');a=p.parse_args()
    globals()[a.action](a.case) if a.action in ['raw','check_raw'] else globals()[a.action]()
