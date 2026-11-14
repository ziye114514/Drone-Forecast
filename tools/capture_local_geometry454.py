"""Six original RAWs, one geometry policy, all inference before references."""
from pathlib import Path
import os,subprocess,sys,json,hashlib,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from run_rotor_only import SOURCE_FILES
from local_conic_opportunity import nonflat_contour_present
from extend_inverse321 import references
from score_rotor164 import score

if __name__=='__main__':
    out=ROOT.parent/'output/local_geometry454_20261110';out.mkdir(exist_ok=False)
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in SOURCE_FILES}
    mask=np.zeros((70,600),np.uint8)
    for x in (30,210,390,570):cv2.circle(mask,(x,35),14,255,-1)
    mm=cv2.moments(mask,True);v=np.linalg.eigvalsh(np.array([[mm['mu20'],mm['mu11']],[mm['mu11'],mm['mu02']]])/mm['m00']);ratio=float(np.sqrt(v[0]/v[1]));cc,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    assert ratio<.16 and nonflat_contour_present(cc)
    line=np.zeros_like(mask);line[34:37,20:580]=255;ll,_=cv2.findContours(line,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);assert not nonflat_contour_present(ll)
    protocol=dict(entry='run_rotor_local_geometry_candidate.py',source_sha256=hashes,
        one_policy_all6=True,geometry_search_not_detection_permission=True,old_fit_quality_unchanged=True,
        synthetic_global_ratio=ratio,synthetic_four_discs_get_search=True,synthetic_line_does_not_get_search=True,
        model_unchanged=True,new_video_unread=True,FPS_work=False,all_inference_before_reference_reads=True)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    progress={};ns=('13','15','16','11','3','4')
    for n in ns:
        raw=Path(f'E:/dataset/experiment/{n}/media/event.raw');assert raw.is_file()
        command=[sys.executable,str(ROOT/protocol['entry']),'--raw',str(raw),'--start','10','--duration','6','--output',str(out/f'recording{n}'),
            '--gpu-python','C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe','--rotor-native-geometry-dll',str(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll')]
        print(n,'actual RAW beginning',flush=True)
        with (out/f'run{n}.log').open('w',encoding='utf8') as log:result=subprocess.run(command,cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,(n,result.returncode)
        progress[n]=json.loads((out/f'recording{n}/summary.json').read_text('utf8'));(out/'progress.json').write_text(json.dumps(progress,indent=2),'utf8');print(n,'inference frozen',flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==v for f,v in hashes.items())
    original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'));cases={};pool={a:[] for a in ('accuracy451','local454')};missing=[];evidence={}
    for n in ns:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1);ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']};cases[n]={}
        for a,p in [('accuracy451',ROOT.parent/f'output/accuracy451_20261110/{n}.jsonl'),('local454',out/f'recording{n}/predictions.jsonl')]:
            s=score(p,ref,origins);v=[q['error_px'] for q in s['records'] if q['valid'] and q['t'] in ids];pool[a].extend(v);s['original_available']=dict(n=len(v),mean=float(np.mean(v)));cases[n][a]=s
            if a=='local454':missing.extend(dict(recording=n,t=t) for t in ids if not any(q['t']==t and q['valid'] for q in s['records']))
        rr=[json.loads(s) for s in (out/f'recording{n}/predictions.jsonl').read_text('utf8').splitlines()];old=[json.loads(s) for s in (ROOT.parent/f'output/accuracy451_20261110/{n}.jsonl').read_text('utf8').splitlines()]
        assert len(rr)==len(old)==181 and all(r['t']==s['t'] and r['counts']['events']==s['counts']['events'] for r,s in zip(rr,old))
        evidence[n]={a:dict(current_slot_sum=sum(r['period_selection']['source']=='current_visible_slot_f2_sum' for r in rows),gain_updates=sum(r['force'].get('gain_learned',False) for r in rows),four=sum(r['structure']['mode']=='four' for r in rows)) for a,rows in [('old',old),('new',rr)]}
        print(n,{a:s['original_available'] for a,s in cases[n].items()},evidence[n],flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within30=sum(x<=30 for x in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,missing_original=missing,evidence=evidence,source_sha256=hashes,all_original102=not missing,new_video_validated=False,FPS_work=False),indent=2),'utf8');print('POOLED',pooled,'missing',missing,flush=True)
