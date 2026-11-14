"""One RAW accuracy run on all6, not an FPS benchmark or cached score claim."""
from pathlib import Path
import os,subprocess,sys,json,hashlib,time,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from run_rotor_only import SOURCE_FILES
from extend_inverse321 import references
from score_rotor164 import score
from physical_publication import evaluate

if __name__=='__main__':
    out=ROOT.parent/'output/raw_projection437_20261110';out.mkdir(exist_ok=False)
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in SOURCE_FILES}
    protocol=dict(purpose='Actual originalRAW accuracy on all6, fixed10-16s; not FPS measurement',
        SDK_python=sys.executable,GPU_python='C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',
        source_sha256=hashes,entry='run_rotor_projection_candidate.py',no_frame_skips=True,shared_model='native_response_plane_v1.json',
        cached_source_grid_parity_not_assumed=True,all_inference_before_reference_reads=True,
        all6_exposed_development=True,new_video_unread=True,FPS_work=False)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    progress={};ns=('13','15','16','11','3','4')
    for n in ns:
        raw=Path(f'E:/dataset/experiment/{n}/media/event.raw');assert raw.is_file()
        command=[sys.executable,str(ROOT/'core/run_rotor_projection_candidate.py'),'--raw',str(raw),'--start','10','--duration','6','--output',str(out/f'recording{n}'),
            '--gpu-python',protocol['GPU_python'],'--rotor-native-geometry-dll',str(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll')]
        print(n,'actual RAW beginning',flush=True)
        with (out/f'run{n}.log').open('w',encoding='utf8') as log:
            result=subprocess.run(command,cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,(n,result.returncode)
        progress[n]=dict(status='complete',summary=json.loads((out/f'recording{n}/summary.json').read_text('utf8')))
        (out/'progress.json').write_text(json.dumps(progress,indent=2),'utf8');print(n,'actual RAW inference frozen',flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==v for f,v in hashes.items())
    original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'));cases={};pool={a:[] for a in ('cached436','actualRAW437')};missing=[];common_control=[];coverage={};parity={}
    for n in ns:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1);ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']}
        cases[n]=dict(cached436=score(ROOT.parent/f'output/projection436_fixed_20261110/{n}_qualified.jsonl',ref,origins),actualRAW437=score(out/f'recording{n}/predictions.jsonl',ref,origins))
        old={q['t']:q for q in cases[n]['cached436']['records'] if q['valid']};new={q['t']:q for q in cases[n]['actualRAW437']['records'] if q['valid']};coverage[n]=dict(original=len(ids),original_available=sum(t in new for t in ids),extra_available=sum(t not in ids for t in new))
        for t in ids:
            pool['cached436'].append(old[t]['error_px'])
            if t in new:pool['actualRAW437'].append(new[t]['error_px']);common_control.append(old[t]['error_px'])
            else:missing.append(dict(recording=n,t=t))
        rows=[json.loads(s) for s in (out/f'recording{n}/predictions.jsonl').read_text('utf8').splitlines()];wanted=[json.loads(s) for s in (ROOT.parent/f'output/projection436_fixed_20261110/{n}_qualified.jsonl').read_text('utf8').splitlines()]
        stamps=bool(len(rows)==len(wanted) and all(r['t']==q['t'] and r['counts']['events']==q['counts']['events'] for r,q in zip(rows,wanted)))
        parity[n]=dict(exact_same_source_packets=stamps,RAW_first_source_t=rows[0]['t'],cached_first_source_t=wanted[0]['t'])
        if stamps:parity[n]['maximum_prediction_difference_px']=max([float(np.max(abs(np.asarray(evaluate(r,[.4])['0.4'])-evaluate(q,[.4])['0.4']))) for r,q in zip(rows,wanted) if r['valid']] or [0.])
        print(n,'coverage',coverage[n],'RAW mean all available',cases[n]['actualRAW437']['error']['mean'],flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within30=sum(x<=30 for x in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,original_origin_coverage=coverage,missing_original_origins=missing,
        pooled_original_origins=pooled,same102_comparison_valid=not missing,matched_cached_subset_mean=float(np.mean(common_control)),
        packet_parity=parity,all_inference_before_reference_reads=True,source_sha256=hashes,
        target30_achieved=not missing and pooled['actualRAW437']['mean']<=30,new_video_validated=False,FPS_work=False),indent=2),'utf8');print('POOLED',pooled,'missing',missing,flush=True)
