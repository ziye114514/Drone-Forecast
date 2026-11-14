"""Current appearance uncertainty on six actual RAWs; no FPS benchmark."""
from pathlib import Path
import os,subprocess,sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from run_rotor_only import SOURCE_FILES
from extend_inverse321 import references
from score_rotor164 import score

if __name__=='__main__':
    out=ROOT.parent/'output/raw_polarity470_20261111';out.mkdir(exist_ok=False)
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in SOURCE_FILES}
    (out/'protocol.json').write_text(json.dumps(dict(source_sha256=hashes,entry='run_rotor_polarity_candidate.py',
        old_joint_native_motion_preserved=True,only_ONOFF_disagreement_covariance_added=True,
        one_policy_all6=True,no_fit_or_noise_search=True,all_inference_before_references=True,
        source_time_accuracy_not_realtime_certificate=True,new_video_unread=True,FPS_work=False),indent=2),'utf8')
    ns=('13','15','16','11','3','4');progress={}
    for n in ns:
        command=[sys.executable,str(ROOT/'core/run_rotor_polarity_candidate.py'),'--raw',f'E:/dataset/experiment/{n}/media/event.raw','--start','10','--duration','6','--output',str(out/f'recording{n}'),
            '--gpu-python','C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe','--rotor-native-geometry-dll',str(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll')]
        print(n,'RAW beginning',flush=True)
        with (out/f'run{n}.log').open('w',encoding='utf8') as log:result=subprocess.run(command,cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,(n,result.returncode)
        progress[n]=json.loads((out/f'recording{n}/summary.json').read_text('utf8'));(out/'progress.json').write_text(json.dumps(progress,indent=2),'utf8');print(n,'inference frozen',flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==v for f,v in hashes.items())
    original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'));cases={};pool={a:[] for a in ('partial468','polarity470')};evidence={}
    for n in ns:
        path=out/f'recording{n}/predictions.jsonl';base=ROOT.parent/f'output/covariance468_20261111/{n}_covariance_partial.jsonl'
        rr=[json.loads(s) for s in path.read_text('utf8').splitlines()];old=[json.loads(s) for s in base.read_text('utf8').splitlines()]
        assert len(rr)==len(old)==181
        for a,b in zip(rr,old):
            assert a['t']==b['t'] and a['counts']==b['counts'] and a['valid']==b['valid'] and a['body']==b['body'],(n,a['t'],'front or source changed')
            assert a['period_selection']['observed_components']==b['period_selection']['observed_components'],(n,a['t'],'force input changed')
            for k in ['dx','dy']:
                assert (a['motion'].get('estimate') or {}).get(k)==(b['motion'].get('estimate') or {}).get(k),(n,a['t'],'joint motion changed')
        disagreement=[r['motion']['estimate']['polarity_motion_quality']['disagreement_px'] for r in rr if (r['motion'].get('estimate') or {}).get('polarity_motion_quality',{}).get('available')]
        evidence[n]=dict(polarity_checked_frames=len(disagreement),disagreement_median_px=float(np.median(disagreement)) if disagreement else None,disagreement_max_px=max(disagreement,default=None),original_joint_displacements_identical=True)
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']};cases[n]={}
        for a,p in [('partial468',base),('polarity470',path)]:
            s=score(p,ref,np.round(np.arange(10.8,15.6001,.2),1));v=[q['error_px'] for q in s['records'] if q['valid'] and q['t'] in ids];assert len(v)==len(ids);pool[a].extend(v);s['original_available']=dict(n=len(v),mean=float(np.mean(v)));cases[n][a]=s
        print(n,{a:s['original_available'] for a,s in cases[n].items()},evidence[n],flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within31=sum(x<=31 for x in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,evidence=evidence,original102_preserved=True,new_video_validated=False,FPS_work=False),indent=2),'utf8');print('POOLED',pooled,flush=True)
