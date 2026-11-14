"""Expanded data, frozen visibility/plane mechanisms, all six sources excluded."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from visibility_response import fit_model as fit_visibility,VisibilityResponse
from plane_conditioned_response import plane_samples,fit_model as fit_plane,PlaneResponse
from future_drive_response import drive_feature
from extend_inverse321 import references
from score_rotor164 import score

if __name__=='__main__':
    sources=[ROOT.parent/'output'/s for s in ('training400_20261110','training408_20261110')]
    assert all((s/'completion.json').exists() for s in sources),'Wait for frozen408 completion'
    out=ROOT.parent/'output/expanded411_412_20261110';out.mkdir(exist_ok=False)
    declaration=ROOT.parent/'output/checkpoints/training411_412_predeclared_20261110.json';(out/'protocol.json').write_bytes(declaration.read_bytes())
    groups={n:[] for n in ('1','2','5','6','7','8','9','10','12','14','17','18','19')};manifest={}
    files=['plane_conditioned_response.py','future_drive_response.py','conditional_unknown_force.py','physical_publication.py','visibility_response.py','tools/evaluate_expanded411.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    for source in sources:
        completion=json.loads((source/'completion.json').read_text('utf8'));manifest[source.name]={}
        for n,status in completion['records'].items():
            assert n in groups and n not in ('3','4','11','13','15','16')
            if status['status']!='complete':manifest[source.name][n]=dict(status=status['status'],teachers=0);continue
            p=source/f'recording{n}/predictions.jsonl';raw=p.read_bytes();rows=[json.loads(s) for s in raw.decode('utf8').splitlines()]
            # Never bridge a teacher/context across independently restarted runs.
            g=plane_samples(rows);lookup={r['t']:r for r in rows}
            if source==sources[0]:
                d=json.loads((ROOT.parent/f'output/drive406_20261110/teachers{n}.json').read_text('utf8'));assert len(g)==len(d)
                for sample,known in zip(g,d):assert sample['t']==known['t'];sample['drive_feature']=known['drive_feature']
            else:
                for sample in g:sample['drive_feature']=drive_feature(lookup[sample['t']],sample['h']).tolist()
            groups[n].extend(g);manifest[source.name][n]=dict(status='complete',teachers=len(g),sha256=hashlib.sha256(raw).hexdigest())
            print(source.name,n,'teachers',len(g),flush=True)
    for n,g in groups.items():(out/f'teachers{n}.json').write_text(json.dumps(g),'utf8')
    models=dict(visibility411=fit_visibility(list(groups.values())),joint412=fit_plane(list(groups.values()),collective_drive=True))
    for a,m in models.items():
        m.update(training_manifest=manifest,all6_evaluation_sources_excluded_from_fit=True)
        (out/f'{a}_model.json').write_text(json.dumps(m,indent=2),'utf8')
    print('MODELS FROZEN',[(a,m['training_samples']) for a,m in models.items()],flush=True)
    evaluation=('13','15','16','11','3','4');arms=('visibility411','joint412','no_plane412','gamma1_412')
    for n in evaluation:
        rr=[json.loads(s) for s in (ROOT.parent/f'output/force357_20261109/{n}.jsonl').read_text('utf8').splitlines()]
        for a in arms:
            pred=VisibilityResponse(models[a]) if a=='visibility411' else PlaneResponse(models['joint412'],disabled=a=='no_plane412',fixed_gamma=1. if a=='gamma1_412' else None)
            vv=[pred.step(r) for r in rr];assert all(r['valid']==old['valid'] and r['coefficient']==old['coefficient'] and r['force']==old['force'] for r,old in zip(vv,rr))
            (out/f'{n}_{a}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8')
        print(n,'inference frozen',flush=True)
    cases={};pool={a:[] for a in ('control','visibility402','joint410',*arms)}
    for n in evaluation:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1)
        cases[n]=dict(control=score(ROOT.parent/f'output/force357_20261109/{n}.jsonl',ref,origins),visibility402=score(ROOT.parent/f'output/visibility402_20261110/{n}_visibility.jsonl',ref,origins),joint410=score(ROOT.parent/f'output/combined410_20261110/{n}_joint.jsonl',ref,origins),**{a:score(out/f'{n}_{a}.jsonl',ref,origins) for a in arms})
        for a,r in cases[n].items():pool[a].extend(q['error_px'] for q in r['records'] if q['valid'])
        print(n,{a:r['error']['mean'] for a,r in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.percentile(v,95)),within35=sum(e<=35 for e in v),within30=sum(e<=30 for e in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,training_manifest=manifest,source_sha256=hashes,all6_evaluation_sources_excluded_from_fit=True,
        all_inference_before_reference_reads=True,one_global_model_no_recording_selection=True,no_hyperparameter_scan=True,independent_new_holdout_claimed=False,RAW_FPS_certificate=False),indent=2),'utf8');print('POOLED',pooled,flush=True)
