"""Score this batch at the actual publication time with fixed missing count."""
from pathlib import Path
import json,sys,numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent/'output';sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from physical_publication import evaluate
from compare_final_pipeline import reference_at,stats

if __name__=='__main__':
    truth=sorted(json.loads((BASE/'separated201_20261107/reference13/references.json').read_text())['records'].values(),key=lambda r:r['t'])
    output={}
    for folder in ['available_gpu246_20261108/raw13','available_gpu248_20261108/raw13_iterator','available_gpu252_20261108/raw13_cooperative']:
        p=BASE/folder;rows=[json.loads(s) for s in (p/'predictions.jsonl').read_text().splitlines()];pub=[json.loads(s) for s in (p/'publication_trace.jsonl').read_text().splitlines()];records=[]
        for t in np.round(np.arange(10.8,15.6001,.2),1):
            recent=[r for r in pub if r['t']<=t+1e-9];r=recent[-1] if recent else None;target=reference_at(truth,t+.4)
            available=bool(r and r['valid'] and t-r['t']<=.066667 and t-r['source_t']<=.16 and
                r.get('last_image_t') is not None and t-r['last_image_t']<=.16 and target is not None)
            result=dict(t=float(t),valid=available)
            if available:
                source=rows[r['index']];h=t-source['t']+.4;prediction=evaluate(source,[h])[str(float(h))]
                result.update(error_px=float(np.linalg.norm(np.asarray(prediction)-target)),prediction=prediction,target=target.tolist(),
                    source_age_s=float(t-source['t']),publication_age_s=float(t-r['t']))
            records.append(result)
        output[folder]=dict(available=sum(r['valid'] for r in records),eligible=25,error=stats([r['error_px'] for r in records if r['valid']]),records=records,
            actual_publication_time_used=True,source_time_only=False,reference_is_weak_event_outline=True)
        (p/'fixed_publication_scores.json').write_text(json.dumps(output[folder],indent=2),'utf-8')
    aa=output['available_gpu248_20261108/raw13_iterator'];bb=output['available_gpu252_20261108/raw13_cooperative']
    ta={r['t']:r for r in aa['records'] if r['valid']};tb={r['t']:r for r in bb['records'] if r['valid']};common=sorted(set(ta)&set(tb))
    output['common_iterator_cooperative']=dict(n=len(common),origins=common,
        iterator_mean_px=float(np.mean([ta[t]['error_px'] for t in common])) if common else None,
        cooperative_mean_px=float(np.mean([tb[t]['error_px'] for t in common])) if common else None)
    dest=BASE/'rotor_accuracy243_20261108';dest.mkdir(exist_ok=True);(dest/'publication_scores.json').write_text(json.dumps(output,indent=2),'utf-8')
    print(json.dumps({k:{a:b for a,b in r.items() if a!='records'} for k,r in output.items()}),flush=True)
