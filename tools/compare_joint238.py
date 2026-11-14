"""Matched fixed-observation forecast comparison, after phase inference ends."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,argparse,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from adaptive_response_observer import AdaptiveResponseObserver
from joint_force_adapter import adapt
from physical_publication import evaluate
from run_rotor_only import serial
from score_rotor164 import score
from replay_joint236 import source


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recording',default='15');p.add_argument('--phase-dir',required=True);a=p.parse_args()
    OUT=ROOT.parent/f'output/joint244_20261108/recording{a.recording}';OUT.mkdir(parents=True,exist_ok=False)
    phase_dir=Path(a.phase_dir);periods=[json.loads(s) for s in (phase_dir/'periods.jsonl').read_text().splitlines()]
    rows=[json.loads(s) for s in source(a.recording).read_text().splitlines()];assert len(periods)==len(rows)
    files=['joint_force_adapter.py','tools/compare_joint238.py','adaptive_response_observer.py']
    for n in files:
        d=OUT/'source_snapshot'/n;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes((ROOT/n).read_bytes())
    names=['control','joint_overlay','joint_geometry','no_measured_phase'];models={k:AdaptiveResponseObserver() for k in names};outputs={k:[] for k in names};maxdelta=0.;normal_used=0
    for old,period in zip(rows,periods):
        assert abs(old['t']-period['t'])<1e-9
        for arm,m in models.items():
            r=copy.deepcopy(old)
            if arm in ('control','no_measured_phase'):
                u=r['force'].get('input_u');u=None if u is None else np.asarray(u);meta=r['period_selection'];structure=r['structure']
                if arm=='no_measured_phase':u=None;meta=dict(observed_components={},learning_input_current=False,input_sigma_proxy=0.)
            else:
                u,meta,structure=adapt(r,period,refine_normal=arm=='joint_geometry')
                if arm=='joint_geometry' and structure.get('normal_refinement_source'):normal_used+=1
            if r['state'].get('two_point_initialization_selected_by_partial_entry'):m.two_point_init=True
            c,v,f,s=m.step(r['t'],r['body'],u,.7 if u is not None else 0.,allow_gain_learning=meta.get('learning_input_current',True),
                input_meta=meta,motion=r['motion'],structure=structure)
            r.update(coefficient=c,valid=v,force=f,state=s,normal_turn_forecast=s.get('normal_turn_forecast'),structure=structure,period_selection=meta)
            r['physical']=evaluate(r,[.1,.2,.4]) if c else {};outputs[arm].append(r)
            if arm=='control' and c:
                for h in r['physical']:maxdelta=max(maxdelta,float(np.max(abs(np.asarray(r['physical'][h])-old['physical'][h]))))
    assert maxdelta<1e-7,maxdelta
    # Annotation-free inference above has ended. These are fixed weak outline
    # references, not synchronized physical centres or RPM ground truth.
    ref={'15':'motion181_20261107/reference_review15','13':'separated201_20261107/reference13','16':'unseen191_20261107/reference16','11':'regional212_20261108/reference11'}[a.recording]
    result={}
    for arm,rs in outputs.items():
        path=OUT/f'{arm}.jsonl';path.write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rs),'utf-8')
        result[arm]=score(path,ROOT.parent/f'output/{ref}/references.json',np.round(np.arange(10.8,15.6001,.2),1))
    summary=dict(recording=a.recording,control_replay_max_px=maxdelta,scores=result,
        joint_input_frames=sum(p['u'] is not None for p in periods),joint_normal_frames=normal_used,
        references_are_weak_native_outline_not_COM_truth=True,source_time_only=True,GPU_used=False,
        inference_uses_annotation_coordinates=False,NN_used=False,RAW_FPS_measured=False)
    (OUT/'results.json').write_text(json.dumps(summary,indent=2),'utf-8')
    print(json.dumps({k:dict(available=v['available'],error=v['error']) for k,v in result.items()}),flush=True)
