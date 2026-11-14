"""Post-inference development scoring; never imported by the runtime."""
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from physical_publication import evaluate
from compare_final_pipeline import reference_at,stats


def score(path,labels,origins):
    rows=[json.loads(s) for s in Path(path).read_text('utf-8').splitlines()]
    truth=sorted(json.loads(Path(labels).read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    records=[]
    for t in origins:
        available=[r for r in rows if r['t']<=t+1e-9]
        if not available:
            records.append(dict(t=float(t),valid=False,reason='no_source'));continue
        r=available[-1];age=t-r['t'];now=reference_at(truth,t);future=reference_at(truth,t+.4)
        good=r['valid'] and age<=.04 and future is not None
        result=dict(t=float(t),source_t=r['t'],source_age_s=float(age),valid=bool(good),
            reason='scored' if good else 'unavailable_forecast_or_reference')
        if good:
            prediction=np.asarray(evaluate(r,[age+.4])[str(float(age+.4))])
            result.update(error_px=float(np.linalg.norm(prediction-future)),prediction=prediction.tolist(),target=future.tolist(),
                current_position_error_px=None if now is None else float(np.linalg.norm(np.asarray(r['coefficient']['position'])-now)))
        records.append(result)
    return dict(eligible=len(origins),available=sum(r['valid'] for r in records),
        error=stats([r['error_px'] for r in records if r['valid']]),records=records,
        source_time_only=True,manual_boxes_are_development_references=True,
        new_temporal_extension_is_not_a_new_recording=True)


if __name__=='__main__':
    path,labels,out=sys.argv[1:]
    result=dict(fixed=score(path,labels,np.arange(110.4,112.0001,.2)),
        temporal_extension=score(path,labels,np.arange(112.2,114.0001,.2)))
    Path(out).write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='records'} for k,v in result.items()}))
