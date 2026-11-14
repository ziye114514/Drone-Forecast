"""Combine verified continuity with source-excluded future unknown-force learning.

Predeclared matched attribution: full, without extra source teachers, camera
axis, without future plane. All same current frames and availability. No new
parameter search or per-recording algorithm choice.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from conditional_unknown_force import teachers,fit,ConditionalUnknownForce
from extend_inverse321 import references
from score_rotor164 import score

if __name__=='__main__':
    out=ROOT.parent/'output/combined389_20261110';out.mkdir(exist_ok=False)
    source=ROOT.parent/'output/response388_20261110';assert (source/'completion.json').exists()
    files=['conditional_unknown_force.py','physical_publication.py','tools/combine_response389.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    evaluation=('13','15','16','11','3','4');data={p.stem:[json.loads(s) for s in p.read_text('utf8').splitlines()] for p in source.glob('*.jsonl')}
    samples={a:{n:teachers(rr,a) for n,rr in data.items()} for a in ('disk','camera','no_future_plane')};models={};arms=('full','no_extra_sources','camera','no_future_plane')
    for n in evaluation:
        models[n]={}
        for arm in arms:
            mode='disk' if arm in ('full','no_extra_sources') else arm
            groups=[g for k,g in samples[mode].items() if k!=n and (arm!='no_extra_sources' or k in evaluation)]
            model=fit(groups,mode);models[n][arm]=model;pred=ConditionalUnknownForce(model);outputs=[pred.step(r) for r in data[n]]
            assert all(r['valid']==old['valid'] and r['coefficient']==old['coefficient'] for r,old in zip(outputs,data[n]))
            (out/f'{n}_{arm}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in outputs),'utf8')
        print(n,'inference frozen',flush=True)
    (out/'models.json').write_text(json.dumps(dict(LOO=models,shared_for_future=fit(list(samples['disk'].values())),teacher_counts={n:len(s) for n,s in samples['disk'].items()}),indent=2),'utf8')
    cases={};aggregate={a:[] for a in ('control','continuation')+arms};extra={a:[] for a in aggregate}
    for n in evaluation:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1)
        cases[n]=dict(control=score(ROOT.parent/f'output/force357_20261109/{n}.jsonl',ref,origins),continuation=score(source/f'{n}.jsonl',ref,origins),**{a:score(out/f'{n}_{a}.jsonl',ref,origins) for a in arms})
        old={r['t'] for r in cases[n]['control']['records'] if r['valid']}
        for arm,result in cases[n].items():
            lookup={r['t']:r for r in result['records']};kept=[lookup[t] for t in sorted(old) if lookup[t]['valid']]
            common=dict(n=len(kept),mean=float(np.mean([r['error_px'] for r in kept])),p95=float(np.percentile([r['error_px'] for r in kept],95)))
            result['original_baseline_origin_subset']=common;aggregate[arm].extend(r['error_px'] for r in kept);extra[arm].extend(r['error_px'] for r in result['records'] if r['valid'] and r['t'] not in old)
            result['original_baseline_origins_lost']=len(old)-len(kept)
        print(n,{a:r['original_baseline_origin_subset'] for a,r in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.percentile(v,95)),within35=sum(e<=35 for e in v),within30=sum(e<=30 for e in v),extra_count=len(extra[a]),extra_mean=None if not extra[a] else float(np.mean(extra[a]))) for a,v in aggregate.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled_original_origins=pooled,source_sha256=hashes,no_manual_label_training=True,all_sources_exposed_development=True,all_inference_before_reference_reads=True,no_per_video_selection=True,no_RAW_FPS_measurement=True),indent=2),'utf8');print('POOLED',pooled,flush=True)
