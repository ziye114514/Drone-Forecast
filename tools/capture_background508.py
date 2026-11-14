"""One frozen candidate, actual six RAW, old102 intact, replay parity required."""
from pathlib import Path
import os,sys,json,subprocess,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from run_rotor_only import SOURCE_FILES
from physical_publication import evaluate
from extend_inverse321 import references
from score_rotor164 import score

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--shared-endpoints',action='store_true');p.add_argument('--selection',action='store_true');p.add_argument('--memory',action='store_true');args=p.parse_args()
    out=ROOT.parent/('output/raw_bg_memory517_20261113' if args.memory else 'output/raw_bg_selection512_20261113' if args.selection else 'output/raw_bg_endpoints509_20261112' if args.shared_endpoints else 'output/raw_background508_20261112');out.mkdir(exist_ok=False)
    expected='background_endpoints506d_20261112' if args.shared_endpoints else 'background_translation504_20261112'
    entry='run_rotor_background_endpoint_candidate.py' if args.shared_endpoints else 'run_rotor_background_candidate.py'
    if args.selection:
        expected='background_evidence511_20261112';entry='run_rotor_background_selection_candidate.py'
    if args.memory:
        expected='background_memory516_20261113';entry='run_rotor_background_memory_candidate.py'
    files=sorted(set(SOURCE_FILES+['tools/capture_background508.py']));hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    (out/'protocol.json').write_text(json.dumps(dict(source_sha256=hashes,entry=entry,expected_replay=expected,
        selected_one_global_candidate=True,no_new_model_or_parameter_fit=True,original102_fixed=True,
        all_inference_before_references=True,all6_sources_exposed_development=True,
        reserved_video_unread=True,FPS_work=False,background_not_a_new_target_detector=True),indent=2),'utf8')
    ns=('13','15','16','11','3','4');progress={};parity={}
    for n in ns:
        folder=out/f'recording{n}';cmd=[sys.executable,str(ROOT/entry),'--raw',f'E:/dataset/experiment/{n}/media/event.raw',
            '--start','10','--duration','6','--output',str(folder),'--gpu-python',
            'C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe','--rotor-native-geometry-dll',
            str(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll')]
        print(n,'RAW beginning',flush=True)
        with (out/f'run{n}.log').open('w',encoding='utf8') as log:rc=subprocess.run(cmd,cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT).returncode
        assert rc==0,(n,rc)
        actual=[json.loads(s) for s in (folder/'predictions.jsonl').read_text('utf8').splitlines()]
        filename=f'{n}.jsonl' if args.selection or args.memory else f'{n}_reference.jsonl'
        target=[json.loads(s) for s in (ROOT.parent/f'output/{expected}/{filename}').read_text('utf8').splitlines()]
        assert len(actual)==len(target)==181
        assert all(r['t']==s['t'] and r['counts']['events']==s['counts']['events'] and r['valid']==s['valid'] and r['body']==s['body'] and r['motion']==s['motion'] for r,s in zip(actual,target))
        d=max(np.max(abs(np.asarray(evaluate(r,[.4])['0.4'])-evaluate(s,[.4])['0.4'])) for r,s in zip(actual,target) if r['valid'])
        assert d<1e-7,(n,float(d));parity[n]=float(d)
        summary=json.loads((folder/'summary.json').read_text('utf8'))
        assert summary['frames']==summary['native_packets_read']==181 and summary['skipped_observations']==0
        progress[n]=summary;(out/'progress.json').write_text(json.dumps(dict(progress=progress,parity=parity),indent=2),'utf8')
        print(n,'RAW frozen;replay maxdelta',d,flush=True)
    assert hashes=={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'));cases={};pool={a:[] for a in ('baseline487','background')}
    for n in ns:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n)
        ids={r['t'] for r in original['cases'][n]['control410']['records'] if r['valid']};cases[n]={}
        for a,path in [('baseline487',ROOT.parent/f'output/raw_local_contrast487_20261112/{n}_local_contrast/predictions.jsonl'),('background',out/f'recording{n}/predictions.jsonl')]:
            s=score(path,ref,np.round(np.arange(10.8,15.6001,.2),1));values=[r['error_px'] for r in s['records'] if r['valid'] and r['t'] in ids]
            assert len(values)==len(ids);pool[a].extend(values);s['original_available']=dict(n=len(values),mean=float(np.mean(values)));cases[n][a]=s
        print(n,{a:s['original_available'] for a,s in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within30p5=sum(x<=30.5 for x in v)) for a,v in pool.items()}
    result=dict(cases=cases,pooled=pooled,replay_parity=parity,all6_actual_RAW=True,original102_preserved=True,
        source_unchanged_during_batch=True,new_video_validated=False,FPS_work=False,entry=entry)
    (out/'results.json').write_text(json.dumps(result,indent=2),'utf8');print('POOLED',pooled,flush=True)
