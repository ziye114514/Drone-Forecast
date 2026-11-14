"""Frozen geometry controls, causal cached replay, and actual RAW integration."""
from pathlib import Path
import argparse,collections,datetime,hashlib,html,json,sys,time,subprocess
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'research/stage104'),str(ROOT/'research/stage49')]
from rigid104 import PlanarCenter,PARAMS
from online104 import OnlineExtras,NAMES
import rigid49 as geometry
import run_planar_center_forecast as entry
OUT=ROOT.parent/'output/stage104_joint_center_20261022'
BASE=ROOT.parent/'output/stage93_RAW_complementary_20261022'
EXTRA=ROOT.parent/'output/stage78_RAW_separated_forecast_20261021'
CASES=['clear6','side6','train3']
REFERENCES=[
 dict(title='Relative State Estimation using Event-Based Propeller Sensing',url='https://arxiv.org/html/2604.18289v1',use='Ellipse direction and propeller frequency; this stage retains prior event inputs, does not reproduce metric state estimation.'),
 dict(title='IPPE author repository',url='https://github.com/tobycollins/IPPE',use='Planar pose ambiguities/calibration requirements researched; IPPE is not executed because real camera K and metric dimensions are unknown.'),
 dict(title='Accurate Measurement of 3D and 2D Circular Centers',url='https://eku127.github.io/circular-center-calibration/',use='Rechecked ellipse center vs projected circle center. Existing58 joint circle-conic fitting actually reused; calibrated author solver is not applied with guessed K.'),
 dict(title='HVC-Net',url='https://arxiv.org/abs/2209.08924',use='Joint homography/visibility/confidence principle researched; no claim to run pretrained HVC-Net on event data.')]


def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def load(p):return json.loads(Path(p).read_text('utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8') as f:return [json.loads(line) for line in f if line.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def deterministic(value):
    if isinstance(value,dict):return {k:deterministic(v) for k,v in value.items() if k not in ['processing_ms','elapsed_ms']}
    if isinstance(value,list):return [deterministic(v) for v in value]
    return value
def sources():
    ns=entry.build();result=ns['fingerprints']();ns.clear()
    result['research/stage104/study104.py']=sha(__file__);return result


def freeze():
    assert not OUT.exists()
    old=load(BASE/'protocol93.json')
    cases={c:old['cases'][c] for c in CASES}
    save(OUT/'protocol104.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),
        parameters=PARAMS,references=REFERENCES,cases=cases,
        input_hashes={c:sha(BASE/c/'observations_forecasts.jsonl') for c in CASES},
        synthetic=dict(seed=104,frames=100,dt=.02,noise_sigma_px=.3,
            cases=['stationary_wobble','translating_wobble','abrupt_turn_wobble','edge_on_missing','non_drone_inconsistent_conics'],
            evaluation='Known pinhole projected equal circles on symmetric rectangular frame. Compare displacement relative to first five-frame lock; the alignment offset is not absolute COM truth.'),
        raw_cases=dict(clear6=cases['clear6'],side6=cases['side6'],prefix_clear6=dict(cases['clear6'],duration=1.2),negative5=dict(raw='E:/dataset/train/5/media/event.raw',start=0.,duration=.6)),
        design='Original55 unchanged. Fit current four conics, five consecutive locks, joint geometric center and layout, initialize image offset once, view/conditioning weighted correction; missing/edge-on expires to body center. Additional independent force gains use corrected centers and blended normal, with constant and zero matched controls.',
        cached='Only current body/sensing/body_axis and already computed causal features; cached replay is NOT a new RAW test. Compute predictions before reading frozen sparse future references.',
        pass_checks=['No stationary prior; translating and turning controls retain motion','Edge-on and missing support never insert virtual hubs as measurements','Independent prefixes reproduce all deterministic state','Actual RAW retains original55 numerical outputs'],
        evaluation='Same existing sparse box-center endpoints and all failure cases; not dense true body-COM or independent pose/RPM evaluation. Fixed development clips, no threshold tuning to their outcomes.',
        default_promoted=False))
    print('104 protocol frozen',flush=True)


def verify():
    p=load(OUT/'protocol104.json');expected=dict(p['source_hashes'])
    amendment=OUT/'validation_amendment104.json'
    if amendment.exists():
        a=load(amendment);assert a['original_study_sha256']==p['source_hashes']['research/stage104/study104.py']
        assert sha(OUT/'study104_initial_prefix_check.py')==a['original_study_sha256']
        expected.update(a['updated_source_hashes'])
    assert expected==sources()
    for c,h in p['input_hashes'].items():assert sha(BASE/c/'observations_forecasts.jsonl')==h
    return p


def fake_measurement(H,rng,sigma=0.):
    phase=np.linspace(0,2*np.pi,128,endpoint=False);circle=np.c_[np.cos(phase),np.sin(phase)]*.16
    rotors=[];outlines=[]
    for i,center in enumerate(geometry.TEMPLATE):
        contour=geometry.project(H,center+circle);outlines.append(contour)
        ellipse=cv2.fitEllipse((contour+rng.normal(0,sigma,contour.shape)).astype(np.float32))
        (x,y),(a,b),angle=ellipse
        if a<b:a,b=b,a;angle+=90.
        rotors.append(dict(cx=x,cy=y,major=a,minor=b,angle=angle,ratio=b/a,mode='ellipse',observed_now=True,
            center_sigma_px=max(.1,sigma),geometry_quality=.95,rotor_id=i))
    pp=np.concatenate(outlines);bbox=np.r_[pp.min(0),pp.max(0)]
    body=dict(detected=True,center=((bbox[:2]+bbox[2:])/2).tolist(),bbox=bbox.tolist(),confidence=.9)
    return body,rotors


def synthetic():
    p=verify();rng=np.random.default_rng(p['synthetic']['seed']);records=[];summaries=[]
    for case in p['synthetic']['cases']:
        model=PlanarCenter();allrows=[];lock=None
        for j in range(p['synthetic']['frames']):
            t=j*.02;rx=15.+48.*np.sin(t*2.4);ry=14.*np.cos(t*3.1)
            translation=np.array([0.,0.,3.])
            if case in ['translating_wobble','abrupt_turn_wobble']:
                translation[0]=.35*t if case=='translating_wobble' else .35*min(t,1.)-.5*max(t-1.,0.)
            if case=='edge_on_missing' and j>=35:rx=89.9
            H=geometry.homography(geometry.rotation(rx,ry,8.*np.sin(t)),translation)
            truth=geometry.project(H,[[0.,0.]])[0];body,rs=fake_measurement(H,rng,p['synthetic']['noise_sigma_px'])
            if case=='edge_on_missing' and j>=55:rs=rs[:2]
            if case=='non_drone_inconsistent_conics':rs[-1]['major']*=2.;rs[-1]['angle']+=45.
            corrected,d=model.step(t,body,rs,dict(valid=False))
            if d['initialized_now'] and lock is None:lock=dict(body=np.asarray(body['center']),corrected=np.asarray(corrected['center']),truth=truth.copy(),t=t)
            item=dict(case=case,t=t,truth=truth.tolist(),body=body['center'],corrected=corrected['center'],
                mode=d['mode'],ready=d['ready'],current=d['current'],reason=d['measurement']['reason'],
                weight=d['weight'],hubs=d['virtual_hubs'],hubs_are_observations=False)
            if lock is not None:
                move=truth-lock['truth'];item.update(body_displacement_error_px=float(np.linalg.norm(np.asarray(body['center'])-lock['body']-move)),
                    corrected_displacement_error_px=float(np.linalg.norm(np.asarray(corrected['center'])-lock['corrected']-move)))
            allrows.append(item)
        scored=[r for r in allrows if 'body_displacement_error_px' in r]
        summaries.append(dict(case=case,initialized=lock is not None,lock_t=None if lock is None else lock['t'],
            current=sum(r['current'] for r in allrows),body_displacement_rmse_px=None if not scored else float(np.sqrt(np.mean([r['body_displacement_error_px']**2 for r in scored]))),
            corrected_displacement_rmse_px=None if not scored else float(np.sqrt(np.mean([r['corrected_displacement_error_px']**2 for r in scored]))),
            end_displacement_px=float(np.linalg.norm(np.asarray(allrows[-1]['truth'])-allrows[0]['truth'])),
            no_virtual_measurements=all(not r['hubs_are_observations'] for r in allrows)))
        records+=allrows
    save(OUT/'synthetic104.json',dict(summary=summaries,records=records))
    assert summaries[0]['initialized'] and summaries[1]['initialized'] and summaries[2]['initialized']
    assert summaries[0]['corrected_displacement_rmse_px']<summaries[0]['body_displacement_rmse_px']
    assert summaries[1]['end_displacement_px']>100.
    assert not summaries[-1]['initialized']
    edge=[r for r in records if r['case']=='edge_on_missing' and r['t']>1.5]
    assert all(not r['current'] and r['mode']=='body_fallback' for r in edge)
    print(json.dumps(summaries),flush=True)


def cached():
    verify();checks=[]
    for case in CASES:
        data=rows(BASE/case/'observations_forecasts.jsonl');context={};model=OnlineExtras(1280,720,context)
        def one(model,r):
            t=r['t'];body=r['body_observation'];model.obs104,model.diag104=model.center104.step(t,body,r['sensing']['rotors'],r['body_axis'])
            model.current=dict(t=t,body=r['body_axis']);return model.add_forecasts(r['frontends']['delayed_feedback'],r['observation'].get('geometry_quality',0.))
        output=[]
        for r in data:
            q=one(model,r);output.append(q)
            for name,c in r['frontends']['delayed_feedback']['coefficients'].items():assert q['coefficients'][name]==c
        save(OUT/'cached'/case/'forecasts.json',output)
        prefix=OnlineExtras(1280,720,{})
        for r,q in zip(data[:min(75,len(data))],output):
            other=one(prefix,r)
            for name in NAMES:
                if name in q['coefficients']:assert q['coefficients'][name]==other['coefficients'][name]
            assert deterministic(q['rigid_center_diagnostic'])==deterministic(other['rigid_center_diagnostic'])
        counts=collections.Counter(q['rigid_center_diagnostic']['mode'] for q in output)
        reasons=collections.Counter(q['rigid_center_diagnostic']['measurement']['reason'] for q in output)
        summary=dict(case=case,ticks=len(data),modes=dict(counts),reasons=dict(reasons),
            current=sum(q['rigid_center_diagnostic']['current'] for q in output),
            force_current=sum(q['rigid_center_diagnostic']['force_current'] for q in output),
            maximum_gain_updates=max([g['updates'] for q in output for g in q['rigid_center_diagnostic']['gains'].values()] or [0]),
            max_correction_px=max(np.linalg.norm(q['rigid_center_diagnostic']['correction_px']) for q in output),
            original55_preserved=True,independent_prefix_exact=True)
        checks.append(summary);print(json.dumps(summary),flush=True)
    save(OUT/'cached_checks104.json',checks)


def raw(case):
    cfg=verify()['raw_cases'][case];args=[sys.executable,str(ROOT/'run_planar_center_forecast.py'),'--raw',cfg['raw'],
        '--output',str(OUT/'raw'/case),'--start',str(cfg['start']),'--duration',str(cfg['duration']),'--verify-frames']
    if not case.startswith('prefix'):args.append('--preview')
    with (OUT/(case+'_parent.log')).open('x',encoding='utf-8') as f:
        result=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert result.returncode==0,(case,result.returncode)
    check_raw(case)


def check_raw(case):
    cfg=verify()['raw_cases'][case];data=rows(OUT/'raw'/case/'observations_forecasts.jsonl');assert len(data)==round(cfg['duration']*50)
    oldcase=case.removeprefix('prefix_');cache=None
    if oldcase in CASES:
        old=rows(BASE/oldcase/'observations_forecasts.jsonl')[:len(data)];cache=load(OUT/'cached'/oldcase/'forecasts.json')[:len(data)]
        for a,b,c in zip(data,old,cache):
            for key in ['observation','sensing','body_axis','event_activity','event_fields_sha256']:assert a[key]==b[key],(case,a['t'],key)
            q=a['frontends']['delayed_feedback'];ref=b['frontends']['delayed_feedback']
            for name,coef in ref['coefficients'].items():assert q['coefficients'][name]==coef,(case,a['t'],name)
            for name in NAMES:
                if name in c['coefficients']:assert q['coefficients'][name]==c['coefficients'][name],(case,a['t'],name)
    if case.startswith('prefix'):
        full=rows(OUT/'raw'/oldcase/'observations_forecasts.jsonl')[:len(data)]
        for a,b in zip(data,full):
            q=dict(a['frontends']['delayed_feedback']);r=dict(b['frontends']['delayed_feedback'])
            d=q.pop('rigid_center_diagnostic');e=r.pop('rigid_center_diagnostic')
            assert q==r and deterministic(d)==deterministic(e)
    if case=='negative5':
        assert all(not r['frontends']['delayed_feedback']['valid'] and not r['frontends']['delayed_feedback']['rigid_center_diagnostic']['ready'] for r in data)
    save(OUT/'raw'/case/'checks104.json',dict(case=case,parent=len(data),native10=len(data)*2,actual_RAW=True,
        original55_exact=oldcase in CASES,new5_exact_cached=cache is not None,actual_prefix_exact=case.startswith('prefix'),no_labels_read=True))
    print(case,'actual RAW104 checked',flush=True)


def score():
    verify();reference=load(EXTRA/'scoring/analysis.json')['details'];details=[]
    for case in CASES:
        pred={round(q['t'],8):q for q in load(OUT/'cached'/case/'forecasts.json')}
        for r in reference:
            if r['case']!=case or r['frontend']!='event_force' or r['model']!='hold' or r['target_state']!='present':continue
            q=pred.get(round(r['anchor_t'],8)) if r.get('anchor_available') else None
            for name in ['kinematic_train6400','trusted_acc','complement_fallback',*NAMES]:
                c=None if q is None or not q['valid'] else q['coefficients'].get(name)
                error=None
                if c is not None:
                    from hybrid74 import prior
                    xy=prior.evaluate(c,r['actual_h']);error=float(np.linalg.norm(np.asarray(xy)-r['target_center']))
                details.append(dict(case=case,method=name,h=r['nominal_h'],target_t=r['target_t'],valid=c is not None,error_px=error))
    table=[]
    for case in CASES:
        for h in [.1,.2,.4]:
            for method in ['kinematic_train6400','trusted_acc','complement_fallback',*NAMES]:
                group=[r for r in details if r['case']==case and r['h']==h and r['method']==method];values=[r['error_px'] for r in group if r['valid']]
                table.append(dict(case=case,h=h,method=method,n=len(values),targets=len(group),mean_px=None if not values else float(np.mean(values))))
    save(OUT/'scores104.json',dict(table=table,details=details,scope='Sparse annotated visible box centers, development clips; no true geometric center/RPM truth.'))
    print(json.dumps([r for r in table if r['h']==.4]),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','synthetic','cached','raw','check_raw','score']);p.add_argument('--case');a=p.parse_args()
    globals()[a.action](a.case) if a.action in ['raw','check_raw'] else globals()[a.action]()
