"""Same5Hz wall-clock queries and future targets; only completed states allowed.

This measures actual availability as well as forecast error. A source update
that finishes after the query cannot be used, even if its source stamp is old.
"""
from pathlib import Path
import sys,json,bisect
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from physical_publication import evaluate
from compare_final_pipeline import stats
from prepare_fred_transfer import parse
from score_fred_transfer import reference


def main(name,control,candidate,labels,start,duration,output=None):
    out=Path(output) if output is not None else ROOT.parent/'output/stability_audit_20261031'
    out.mkdir(parents=True,exist_ok=True)
    labels=parse(Path(labels).read_text('utf-8-sig'));cases={}
    for tag,folder in [('control',control),('candidate',candidate)]:
        p=Path(folder);pubs=[json.loads(s) for s in (p/'publications.jsonl').read_text().splitlines()]
        by_time={r['t']:r for r in (json.loads(s) for s in (p/'predictions.jsonl').read_text().splitlines())};times=[r['t'] for r in pubs]
        values=[]
        for index in range(1,int(round(duration*5))+1):
            q=start+index/5;truth=reference(labels,q+.4)
            if truth is None:continue
            item=dict(query_t=q,target_t=q+.4,valid=False)
            slot=bisect.bisect_right(times,q)-1
            if slot>=0:
                pub=pubs[slot];r=by_time.get(pub.get('source_t'))
                image_t=max((s['t'] for s in by_time.values() if s['t']<=pub.get('source_t',-1) and s['body'].get('detected')),default=None) if r else None
                age=None if r is None else q-r['t']
                valid=bool(pub['valid'] and r is not None and 0<=age<=.16 and image_t is not None and q-image_t<=.16)
                item.update(valid=valid,latest_completed_publication_t=pub['t'],source_t=pub.get('source_t'),source_age_s=age)
                if valid:
                    pred=evaluate(r,[q+.4-r['t']])[str(float(q+.4-r['t']))];center=(truth[:2]+truth[2:])/2
                    item.update(error_px=float(np.linalg.norm(np.asarray(pred)-center)),prediction=pred)
            values.append(item)
        cases[tag]=values
    common=[(a,b) for a,b in zip(cases['control'],cases['candidate']) if a['valid'] and b['valid']]
    result=dict(queries=cases,statistics={tag:dict(query_slots=len(rows),valid=sum(r['valid'] for r in rows),
        error_px=stats([r['error_px'] for r in rows if r['valid']])) for tag,rows in cases.items()},
        same_query_and_target_times=True,completed_publication_required=True,labels_not_used_by_predictor=True,
        matched=dict(n=len(common),control_error_px=stats([a['error_px'] for a,b in common]),candidate_error_px=stats([b['error_px'] for a,b in common])))
    (out/(name+'_common_publication.json')).write_text(json.dumps(result,indent=2,allow_nan=False),'utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='queries'},indent=2))


if __name__=='__main__':main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],float(sys.argv[5]),float(sys.argv[6]),sys.argv[7] if len(sys.argv)>7 else None)
