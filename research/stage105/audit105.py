"""Current-only transition audit. No new predictor or future-label evaluation."""
from pathlib import Path
import argparse,datetime,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT.parent/'output/stage104_joint_center_20261022'
OUT=ROOT.parent/'output/stage105_transition_audit_20261022'
CASES=['clear6','side6','train3']


def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def load(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sources():return {str(Path(__file__).relative_to(ROOT)):sha(__file__)}


def freeze():
    assert not OUT.exists()
    save(OUT/'protocol105.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources(),
        input_hashes={c:sha(BASE/'cached'/c/'forecasts.json') for c in CASES},
        design='Audit mode transitions and correction first/second finite differences in existing104 causal diagnostics only. No new RAW/no new predictor/no future-position truth read. Mode-switch components are not automatically true errors.',
        metrics=['mode changes','maximum and95th correction-induced velocity/acceleration components','switch/non-switch components','current-geometry gaps'],
        next_candidate='Based only on old native ON/OFF visual samples: train6 110-119.5s includes clear and line-like views; needs dense screening, no guarantee of usable four-center initialization.',
        prior_art=[
            dict(title='LOPET, TIP2024',paper='https://arxiv.org/abs/2408.03225',code='https://github.com/Zibin6/LOPET',
                 use='Read author abstract/repository: robust event-line association and model-based pose refinement. MATLAB demo, known object model; researched, not executed.'),
            dict(title='Direct 3D model-based object tracking, ICRA2024',code='https://github.com/BluecatLi/direct_event_object_tracker/',
                 use='Existing model-based tracking alternative, ROS/3D model/initial pose; researched, not executed.'),
            dict(title='Event6D, CVPR2026',paper='https://arxiv.org/abs/2603.28045',project='https://chohoonhee.github.io/Event6D/',
                 use='Event-depth and reconstructed intensity/depth; outside current RAW-only/noG requirements, not run.'),
            dict(title='EVPeriscope2026',paper='https://arxiv.org/html/2609.11920v1',
                 use='Known frame geometry, four visible centers in actual method; comparison, not new code.')],
        inference_parameters_changed=False,default_promoted=False))


def stats(values):
    return dict(n=len(values),median=None if not values else float(np.median(values)),
        p95=None if not values else float(np.quantile(values,.95)),maximum=None if not values else float(np.max(values)))


def audit():
    protocol=load(OUT/'protocol105.json');assert protocol['sources']==sources()
    result=[];events=[]
    for case in CASES:
        path=BASE/'cached'/case/'forecasts.json';assert sha(path)==protocol['input_hashes'][case]
        data=load(path);allv=[];alla=[];switchv=[];switcha=[];normalv=[];normala=[];last=None;lastv=None
        modes={};gaps=[];last_current=None;switches=0
        for row in data:
            d=row['rigid_center_diagnostic'];t=row['t'];c=np.asarray(d['correction_px'],float)
            modes[d['mode']]=modes.get(d['mode'],0)+1
            if d['current']:
                if last_current is not None:gaps.append(t-last_current)
                last_current=t
            if last is not None:
                dt=t-last['t'];v=(c-last['c'])/dt;vn=float(np.linalg.norm(v));changed=d['mode']!=last['mode']
                a=None if lastv is None else float(np.linalg.norm((v-lastv)/dt))
                allv.append(vn)
                if a is not None:alla.append(a)
                if changed:
                    switches+=1;switchv.append(vn)
                    if a is not None:switcha.append(a)
                    events.append(dict(case=case,t=t,before=last['mode'],after=d['mode'],
                        delta_correction_px=float(np.linalg.norm(c-last['c'])),
                        correction_velocity_component_px_s=vn,correction_acceleration_component_px_s2=a,
                        correction=c.tolist(),is_true_motion_error_known=False))
                else:
                    normalv.append(vn)
                    if a is not None:normala.append(a)
                lastv=v
            last=dict(t=t,c=c,mode=d['mode'])
        result.append(dict(case=case,ticks=len(data),modes=modes,mode_changes=switches,
            correction_velocity_px_s=stats(allv),correction_acceleration_px_s2=stats(alla),
            switch_velocity_px_s=stats(switchv),nonswitch_velocity_px_s=stats(normalv),
            switch_acceleration_px_s2=stats(switcha),nonswitch_acceleration_px_s2=stats(normala),
            geometry_update_gap_s=stats(gaps)))
    save(OUT/'audit105.json',dict(records=result,switches=events,
        interpretation='Finite-difference components contributed by observation correction, NOT actual velocity/acceleration error. No actual COM truth; association with104 history degradation remains a hypothesis.',
        no_new_forecast_accuracy_claim=True,no_inference_changes=True))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','audit']);a=parser.parse_args();globals()[a.action]()
