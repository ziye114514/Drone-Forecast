"""One fixed same-observation replay; labels never enter structural inference."""
from pathlib import Path
import sys,json,copy,time
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from continuous_five_circle import ContinuousFiveCircle
from five_circle_model import FiveCircleModel
from research.stage146.branches146 import BranchGeometry,plane_axis
from check_five_circle_v3 import sequence
from summarize_live_pilot import reference_at,stats


def geometry(rotors,t,body,observer):
    good=[r for r in rotors if r.get('mode')=='ellipse' or r.get('arc_center_qualified') or r.get('curvature_validated')]
    g=observer.step(t,body,rotors)
    if len(good)==len(rotors) and len(rotors) in (3,4):
        g=observer.step(t,body,[dict(r,mode='ellipse',inferred_shape=False) for r in rotors]);g['observed_rotors']=rotors
    return g


def controls():
    observer=ContinuousFiveCircle();rows=[]
    for t,b,g in sequence():rows.append(observer.step(t,b,g))
    assert all(r['mode']=='three' for r in rows[15:30])
    assert all(not r['joint_center_is_current_measurement'] for r in rows[30:45])
    assert all(not r['inferred_rotor_frequencies'] for r in rows)
    assert all(np.linalg.norm(np.asarray(r['current_geometric_center'])-[640,360])<.5 for r in rows if r['current_geometric_center'] is not None)
    assert rows[45]['mode']=='four'
    # Keep same object through a >0.4s partial-visibility interval.
    ob=ContinuousFiveCircle();stream=list(sequence());ob.step(*stream[0]);ob.step(*stream[1])
    t,b,g=stream[15]
    for j in range(200):
        z=ob.step(.04+j*.02,b,g)
    assert z['mode']=='three' and z['continuous']['measurement_rank']==6
    return dict(same_object_partial_continuity=True,partial_held_for_4_seconds=True,stationary_center_under_plane_wobble=True,
        hidden_frequencies_never_added=True,axis_never_fabricates_center=True,synthetic_only=True)


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);control=controls();results={}
    for name in ['clear_release_v2','side_release']:
        source=WORK/'output/five_circle_fast30_20261026'/name
        data=[json.loads(s) for s in (source/'predictions.jsonl').read_text('utf-8').splitlines()]
        branches={n:(FiveCircleModel(3) if n=='original' else ContinuousFiveCircle()) for n in ['original','continuous']}
        bg={n:BranchGeometry() for n in branches};rows=[];cost={n:[] for n in branches}
        for r in data:
            observations=r['geometry'].get('rotors',[]);result={}
            for n,model in branches.items():
                g=geometry(copy.deepcopy(observations),r['t'],r['body'],bg[n]);start=time.perf_counter()
                result[n]=model.step(r['t'],r['body'],g);cost[n].append((time.perf_counter()-start)*1000)
            rows.append(dict(t=r['t'],body=r['body'],results=result))
        (out/(name+'.jsonl')).write_text('\n'.join(json.dumps(r,ensure_ascii=False,default=lambda v:v.tolist() if hasattr(v,'tolist') else v) for r in rows),'utf-8')
        # Only now open existing sparse development labels for centre scoring.
        labels=WORK/'output/dense_labels_side_110_1195/labels.json'
        truth=sorted(json.loads(labels.read_text('utf-8'))['records'].values(),key=lambda z:z['t']) if name=='side_release' else []
        arms={}
        for n in branches:
            gs=[r['results'][n] for r in rows];errors=[];steps=[];last=None
            for r,g in zip(rows,gs):
                p=g['current_geometric_center'];ref=reference_at(truth,r['t']) if truth else None
                if p is not None:
                    if ref is not None:errors.append(float(np.linalg.norm(np.asarray(p)-ref)))
                    if last is not None and r['t']-last[0]<.07:
                        # Remove current body displacement to isolate a geometry jump.
                        steps.append(float(np.linalg.norm(np.asarray(p)-last[1]-(np.asarray(r['body']['center'])-last[2]))))
                    last=(r['t'],np.asarray(p),np.asarray(r['body']['center']))
            arms[n]=dict(modes=dict(Counter(g['mode'] for g in gs)),current_centers=sum(g['current_geometric_center'] is not None for g in gs),
                template_outputs=sum(g['model_centers'] is not None for g in gs),center_error=stats(errors),
                body_compensated_center_step=stats(steps),model_step_ms=stats(cost[n]),
                partial_updates=sum(g.get('continuous',{}).get('updated',False) and g.get('continuous',{}).get('measurement_rank',0)<6 for g in gs))
        results[name]=arms
    summary=dict(controls=control,results=results,new_RAW=False,cached_current_observations=True,
        no_traditional_runtime_forecast=True,parameters_swept=False,full_forecast_gain_proven=False)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8');print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':run(sys.argv[1])
