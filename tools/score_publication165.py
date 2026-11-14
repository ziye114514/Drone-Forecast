"""Score only predictions actually available at predefined wall-clock queries.

Trajectory references are opened after inference. No runtime imports this tool.
Freshness expiration and physical retiming use the existing deployment rules.
"""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from physical_publication import publication,freeze
from compare_final_pipeline import reference_at,stats


def score(directory,labels,queries,predictions=None):
    directory=Path(directory)
    rows={r['t']:r for r in map(json.loads,Path(predictions or directory/'predictions.jsonl').read_text('utf-8').splitlines())}
    trace=list(map(json.loads,(directory/'publication_trace.jsonl').read_text('utf-8').splitlines()))
    truth=sorted(json.loads(Path(labels).read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    records=[]
    for query in queries:
        query=float(query);available=[p for p in trace if p['t']<=query+1e-9]
        target=reference_at(truth,query+.4)
        result=dict(t=query,reference_available=target is not None,forecast_available=False,scored=False)
        if available:
            actual=available[-1];row=rows[actual['source_t']]
            current=publication(freeze(row,actual.get('last_image_t')),query,None,0)
            result.update(forecast_available=current['valid'],source_t=actual['source_t'],
                original_emission_t=actual['t'],source_age_s=current['source_age_s'])
            if current['valid'] and target is not None:
                prediction=np.asarray(current['physical']['0.4'])
                result.update(scored=True,error_px=float(np.linalg.norm(prediction-target)),
                    prediction=prediction.tolist(),target=target.tolist())
        records.append(result)
    return dict(queries=len(records),recent_forecasts=sum(r['forecast_available'] for r in records),
        references=sum(r['reference_available'] for r in records),scored=sum(r['scored'] for r in records),
        unique_recent_observations=len({r['source_t'] for r in records if r['forecast_available']}),
        error=stats([r['error_px'] for r in records if r['scored']]),records=records,
        uses_actual_emission_time=True,no_repeated_projection_counted_as_new_observation=True,
        manual_box_centers_are_development_references=True,independent_test=False)


if __name__=='__main__':
    directory,labels,out=sys.argv[1:]
    result=dict(fixed=score(directory,labels,np.arange(110.4,112.0001,.2)),
        temporal_extension=score(directory,labels,np.arange(112.2,113.0001,.2)),
        clock30=score(directory,labels,np.arange(110.4,113.0001,1/30)))
    Path(out).write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='records'} for k,v in result.items()}))
