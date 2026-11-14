"""One matched-input diagnosis of what the rotor/unknown-force route adds."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from adaptive_response_observer import AdaptiveResponseObserver
from physical_publication import evaluate
from run_rotor_only import serial
from replay_joint236 import source
from score_rotor164 import score


if __name__=='__main__':
    OUT=ROOT.parent/'output/mainline239_20261108';OUT.mkdir(exist_ok=False);results={}
    for n in ['13','15','16','11']:
        out=OUT/f'recording{n}';out.mkdir();rows=[json.loads(s) for s in source(n).read_text().splitlines()]
        models={k:AdaptiveResponseObserver() for k in ['control','no_measured_period','no_plane_axis']};outputs={k:[] for k in models};maxdelta=0.
        for old in rows:
            for arm,m in models.items():
                r=copy.deepcopy(old);u=r['force'].get('input_u');u=None if u is None else np.asarray(u);pm=r['period_selection'];g=r['structure']
                if arm=='no_measured_period':u=None;pm=dict(observed_components={},input_sigma_proxy=0.,learning_input_current=False)
                if arm=='no_plane_axis':
                    g=copy.deepcopy(g);g['axis_xy']=None;g['axis_support']=dict(valid=False);g['pose_source']=None
                    g['continuous']=dict(g.get('continuous') or {},updated=False)
                if r['state'].get('two_point_initialization_selected_by_partial_entry'):m.two_point_init=True
                c,v,f,s=m.step(r['t'],r['body'],u,.7 if u is not None else 0.,allow_gain_learning=pm.get('learning_input_current',True),input_meta=pm,motion=r['motion'],structure=g)
                r.update(coefficient=c,valid=v,force=f,state=s,normal_turn_forecast=s.get('normal_turn_forecast'),period_selection=pm,structure=g)
                r['physical']=evaluate(r,[.1,.2,.4]) if c else {};outputs[arm].append(r)
                if arm=='control' and c:
                    for h,p in r['physical'].items():maxdelta=max(maxdelta,float(np.max(abs(np.asarray(p)-old['physical'][h]))))
        assert maxdelta<1e-7,maxdelta
        reference={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16','11':'regional212_20261108/reference11'}[n]
        result={}
        for arm,rs in outputs.items():
            path=out/f'{arm}.jsonl';path.write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rs),'utf-8')
            result[arm]=score(path,ROOT.parent/f'output/{reference}/references.json',np.round(np.arange(10.8,15.6001,.2),1))
        results[n]=dict(control_replay_max_px=maxdelta,scores=result)
        print(n,{k:v['error'].get('mean') for k,v in result.items()},flush=True)
    (OUT/'results.json').write_text(json.dumps(dict(results=results,weak_references=True,source_time_only=True,NN_used=False,
        new_accuracy_improvement_claimed=False,known_frequency_proxy_not_metric_thrust=True),indent=2),'utf-8')
