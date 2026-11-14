"""Single rigid-template plane intervention; no detection/force tuning."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from metric_template_plane import MetricTemplatePlane
from adaptive_response_observer import AdaptiveResponseObserver
from physical_publication import evaluate
from run_rotor_only import serial
from replay_joint236 import source
from score_rotor164 import score

if __name__=='__main__':
    OUT=ROOT.parent/'output/metric264_20261108';OUT.mkdir(exist_ok=False)
    files=['metric_template_plane.py','tools/replay_metric264.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    protocol=dict(source_sha256=hashes,development=['13','15','16'],reserved_raw='E:/dataset/experiment/1/media/event.raw',
        reserved_opened=False,change='only replace current rank6 plane with five-circle metric transport; side unchanged',
        known_period_input_and_native_motion_unchanged=True,no_parameter_search=True,no_recording_specific_rules=True,NN_used=False,GPU_used=False)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf-8')
    for f in files:
        d=OUT/'source_snapshot'/f;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes((ROOT/f).read_bytes())
    alloutputs={};checks={}
    for n in protocol['development']:
        rows=[json.loads(s) for s in source(n).read_text('utf-8').splitlines()];planes=MetricTemplatePlane();models={k:AdaptiveResponseObserver() for k in ['control','metric']};outputs={k:[] for k in models};maxdelta=0.;used=0
        for old in rows:
            current,info=planes.step(old['structure'],old['geometry']);used+=info['applied']
            for arm,m in models.items():
                r=copy.deepcopy(old);g=r['structure'] if arm=='control' else current;pm=r['period_selection'];u=r['force'].get('input_u');u=None if u is None else np.asarray(u)
                if r['state'].get('two_point_initialization_selected_by_partial_entry'):m.two_point_init=True
                c,v,f,s=m.step(r['t'],r['body'],u,.7 if u is not None else 0.,allow_gain_learning=pm.get('learning_input_current',True),input_meta=pm,motion=r['motion'],structure=g)
                r.update(coefficient=c,valid=v,force=f,state=s,normal_turn_forecast=s.get('normal_turn_forecast'),structure=g,metric_plane=info if arm=='metric' else None)
                r['physical']=evaluate(r,[.1,.2,.4]) if c else {};outputs[arm].append(r)
                if arm=='control' and c:
                    for h,p in r['physical'].items():maxdelta=max(maxdelta,float(np.max(abs(np.asarray(p)-old['physical'][h]))))
        assert maxdelta<1e-7,maxdelta;checks[n]=dict(control_max_px=maxdelta,metric_plane_frames=used);alloutputs[n]=outputs
        for arm,rs in outputs.items():(OUT/f'{n}_{arm}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rs),'utf-8')
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    result={}
    for n,outputs in alloutputs.items():
        folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16'}[n]
        result[n]={arm:score(OUT/f'{n}_{arm}.jsonl',ROOT.parent/f'output/{folder}/references.json',np.round(np.arange(10.8,15.6001,.2),1)) for arm in outputs}
        print(n,'used',checks[n]['metric_plane_frames'],{k:v['error']['mean'] for k,v in result[n].items()},flush=True)
    (OUT/'results.json').write_text(json.dumps(dict(scores=result,checks=checks,protocol=protocol),indent=2),'utf-8')
