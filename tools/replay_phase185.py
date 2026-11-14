"""Accuracy-only causal phase-bank trial. No new detector or NN observations."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,copy,argparse,time,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from observed_phase_bank import ObservedPhaseBank
from layout_reference_observer import LayoutReferenceObserver
from pure_visibility170 import window
from physical_publication import evaluate
from run_rotor_only import serial,SOURCE_FILES
BASE=ROOT.parent/'output/phase185_20261107'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['recording15','side','train2','recording18'],default='recording15');a=p.parse_args()
    OUT=BASE/a.case;OUT.mkdir(exist_ok=False)
    if a.case.startswith('recording'):
        n=a.case.removeprefix('recording');parent=ROOT.parent/'output/motion181_20261107' if n=='15' else BASE
        source=parent/f'holdout/{a.case}_control176.jsonl';cache=parent/f'cache_{a.case}'
    else:
        source=ROOT.parent/f'output/layout176_20261107/integrated/{a.case}_fixed.jsonl'
        cache=ROOT.parent/('output/side_temporal164_20261101/cache_side_warm' if a.case=='side' else 'output/native_plane165_20261104/cache_train2_inside')
    names=list(dict.fromkeys(SOURCE_FILES+['observed_phase_bank.py','observed_ring_phase.py','tools/replay_phase185.py']))
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    for n in names:
        d=OUT/'source_snapshot'/n;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes((ROOT/n).read_bytes())
    (OUT/'protocol.json').write_text(json.dumps(dict(case=a.case,source=str(source),native_cache=str(cache),
        NN_used=False,labels_during_inference=False,RAW_FPS_measured=False,force_model='176 unchanged',
        data_development=a.case!='recording18',source_sha256=hashes),indent=2),'utf-8')
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0]))
            if path.name=='labels.json' or 'evaluation_data' in path.parts:raise RuntimeError('No labels during inference')
    sys.addaudithook(guard)
    bank=ObservedPhaseBank();models={k:LayoutReferenceObserver() for k in ['control176','no_input','zero_phase','phase']};rows={k:[] for k in models};periods=[];timings=[];maxdelta=0.
    inputs=[json.loads(s) for s in source.read_text('utf-8').splitlines()]
    for i,old in enumerate(inputs):
        tick=time.perf_counter();t=old['t'];e=window(cache,t)
        u,q,meta=bank.step(t,e,old['geometry'].get('rotors',[]),old['motion'])
        timings.append((time.perf_counter()-tick)*1000)
        periods.append(dict(t=t,u=u,quality=q,meta=meta))
        for arm,m in models.items():
            r=copy.deepcopy(old)
            if arm=='control176':
                pm=r['period_selection'];pu=r['force']['input_u'];pu=None if pu is None else np.asarray(pu);quality=.7 if pu is not None else 0.
            elif arm=='no_input':pm=dict(observed_components={},input_sigma_proxy=0.,learning_input_current=False);pu=None;quality=0.
            else:
                pm=copy.deepcopy(meta);pu=None if u is None else u.copy();quality=q
                if arm=='zero_phase' and pu is not None:
                    pu[:]=0.;pm['observed_components']={k:[0.,0.] for k in pm['observed_components']}
            if r['state'].get('two_point_initialization_selected_by_partial_entry'):m.two_point_init=True
            c,v,f,s=m.step(t,r['body'],pu,quality,allow_gain_learning=pm.get('learning_input_current',True),input_meta=pm,motion=r['motion'])
            if arm=='control176' and c:
                maxdelta=max(maxdelta,max(float(np.max(abs(np.asarray(c[k])-old['coefficient'][k]))) for k in ['position','velocity','acceleration']))
            r.update(coefficient=c,valid=v,force=f,state=s,phase_bank_input=meta if arm!='control176' else None)
            r['physical']=evaluate(r,[.1,.2,.4]) if c else {};rows[arm].append(r)
        if i%30==0:print(a.case,i,'current rings',meta['qualified_current_rings'],flush=True)
    assert maxdelta<1e-8,maxdelta
    for arm,rs in rows.items():
        (OUT/f'{arm}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rs),'utf-8')
    (OUT/'periods.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in periods),'utf-8')
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    summary=dict(frames=len(inputs),current_input_frames=sum(r['u'] is not None for r in periods),
        known_ring_counts={str(k):sum(r['meta']['qualified_current_rings']==k for r in periods) for k in range(5)},
        phase_bank_ms_mean=float(np.mean(timings)),phase_bank_ms_p95=float(np.percentile(timings,95)),
        control_replay_delta=maxdelta,model_gain_updates={k:sum(r['force'].get('gain_learned',False) for r in rs) for k,rs in rows.items()},
        source_changed=False,NN_used=False,RAW_FPS_measured=False)
    (OUT/'completion.json').write_text(json.dumps(summary,indent=2),'utf-8');print(json.dumps(summary),flush=True)
