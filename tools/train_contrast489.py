"""One fixed source-separated recalibration for the improved rotor front.

Same13 coefficients/objective/10..40s training protocol. No evaluation source,
manual target, reserved video, parameter sweep or favorable replacement window.
"""
from pathlib import Path
import os,sys,subprocess,json,hashlib,copy,numpy as np
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from run_rotor_only import SOURCE_FILES
from capture_training400 import TRAINING,EVALUATION
from replay_endpoint432 import run
from equilibrium_gain_prior import CovarianceOnlyObserver
from plane_conditioned_response import plane_samples,fit_model,PlaneResponse
from future_drive_response import drive_feature
from reference_qualification import QualifiedReferenceResponse
from bootstrap_native_reference import BootstrapNativeReference
from native_fov_observation import VisibleEnvelopeResponse
from window_position_observation import WindowPositionObservation
from physical_publication import evaluate
from extend_inverse321 import references
from score_rotor164 import score

def reapply(rows,model):
    response=QualifiedReferenceResponse(PlaneResponse(model));boot=BootstrapNativeReference()
    vis=VisibleEnvelopeResponse();win=WindowPositionObservation();result=[]
    for old in rows:
        r=copy.deepcopy(old)
        for k in ('conditional_unknown_force','visibility_response','plane_conditioned_response','future_drive_response',
                  'native_fov_observation','window_position_observation','native_reference_bootstrap','physical'):r.pop(k,None)
        r=win.step(vis.step(boot.step(response.step(r))))
        assert r['coefficient']==old['coefficient'] and r['force']==old['force'] and r['valid']==old['valid']
        result.append(r)
    return result

if __name__=='__main__':
    out=ROOT.parent/'output/contrast_training489_20261112';out.mkdir(exist_ok=False)
    files=sorted(set(SOURCE_FILES+['tools/train_contrast489.py','tools/replay_endpoint432.py']))
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    old=json.loads((ROOT/'models/native_response_plane_v1.json').read_text('utf8'))
    assert not set(TRAINING).intersection(EVALUATION)
    protocol=dict(training_source_ids=TRAINING,entire_evaluation_sources_excluded=sorted(EVALUATION),
        start_s=10.,duration_s=30.,source_sha256=hashes,model_dimensions_unchanged=13,
        fit_objective_and_prior_unchanged=True,one_fit_no_parameter_search=True,
        manual_references_not_used_for_training=True,teacher_is_native_outline_not_COM_truth=True,
        originals_and_sidecars_not_written=True,reserved_video_not_read=True)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    groups=[];manifest={};capture={}
    for n in TRAINING:
        folder=out/f'recording{n}'
        cmd=[sys.executable,str(ROOT/'run_rotor_local_contrast_candidate.py'),'--raw',f'E:/dataset/experiment/{n}/media/event.raw',
             '--start','10','--duration','30','--output',str(folder),'--gpu-python',
             'C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe','--rotor-native-geometry-dll',
             str(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll')]
        print(n,'training RAW 10..40 beginning',flush=True)
        with (out/f'run{n}.log').open('w',encoding='utf8') as log:rc=subprocess.run(cmd,cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT).returncode
        if rc:
            capture[n]=dict(status='failed_no_interval_replacement',returncode=rc);manifest[n]=0;groups.append([])
        else:
            rows=[json.loads(s) for s in (folder/'predictions.jsonl').read_text('utf8').splitlines()]
            rr=run(rows,CovarianceOnlyObserver,old,calibrated=False);g=plane_samples(rr);lookup={r['t']:r for r in rr}
            for s in g:s['drive_feature']=drive_feature(lookup[s['t']],s['h']).tolist()
            groups.append(g);manifest[n]=len(g);capture[n]=dict(status='complete',frames=len(rows),teachers=len(g))
            (out/f'teachers{n}.json').write_text(json.dumps(g),'utf8')
        (out/'progress.json').write_text(json.dumps(dict(capture=capture,training_manifest=manifest),indent=2),'utf8')
        print(n,capture[n],flush=True)
    assert all(c['status']=='complete' for c in capture.values()),'Do not silently train after capture failures'
    model=fit_model(groups,collective_drive=True)
    model.update(training_manifest=manifest,all6_evaluation_sources_excluded_from_fit=True,
        native_local_contrast_front_required=True,experimental=True,
        teacher_is_native_outline_not_COM_truth=True,metric_thrust_measured=False)
    (out/'model.json').write_text(json.dumps(model,indent=2),'utf8');print('MODEL FROZEN',manifest,flush=True)
    ns=('13','15','16','11','3','4');parity={}
    for n in ns:
        rows=[json.loads(s) for s in (ROOT.parent/f'output/raw_local_contrast487_20261112/{n}_local_contrast/predictions.jsonl').read_text('utf8').splitlines()]
        for arm,m in [('control',old),('matched',model)]:
            vv=reapply(rows,m)
            if arm=='control':
                delta=max(np.max(abs(np.asarray(evaluate(r,[.4])['0.4'])-evaluate(s,[.4])['0.4'])) for r,s in zip(vv,rows) if r['valid'])
                assert delta<1e-7,(n,delta);parity[n]=float(delta)
            (out/f'{n}_{arm}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8')
        print(n,'evaluation inference frozen',flush=True)
    original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'))
    cases={};pool={a:[] for a in ('control','matched')}
    for n in ns:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n)
        ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']};cases[n]={}
        for a in pool:
            s=score(out/f'{n}_{a}.jsonl',ref,np.round(np.arange(10.8,15.6001,.2),1))
            v=[q['error_px'] for q in s['records'] if q['valid'] and q['t'] in ids]
            assert len(v)==len(ids);pool[a].extend(v);s['original_available']=dict(n=len(v),mean=float(np.mean(v)));cases[n][a]=s
        print(n,{a:s['original_available'] for a,s in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within30p5=sum(x<=30.5 for x in v)) for a,v in pool.items()}
    assert hashes=={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    result=dict(cases=cases,pooled=pooled,control_parity=parity,training_manifest=manifest,
        all6_evaluation_sources_excluded_from_fit=True,all6_exposed_development=True,
        original102_preserved=True,new_video_validated=False,FPS_work=False,
        actual_RAW_front_already_executed=True,recalibrated_readout_is_frozen_observation_replay=True)
    (out/'results.json').write_text(json.dumps(result,indent=2),'utf8');print('POOLED',pooled,flush=True)
