"""Post-hoc withheld-candidate consistency; NOT independent motor-center GT."""
from pathlib import Path
import json,sys,itertools
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage4'))
from study import OUT, save
from tracker import fit_affine,X,initialization

def main():
    dest=OUT/'withheld_candidate_protocol.json'
    if dest.exists():raise RuntimeError('Probe already run')
    save(dest,dict(type='Post-hoc diagnostic, not a blind test',
        selection='All current frames with four selected ellipse candidates; also report initialization-qualified subset.',
        operation='Known cyclic identities; hide each candidate and fit affine projection to other three.',
        reference='Withheld original ellipse-center candidate, NOT independently annotated motor center.',
        warning='Three-point affine fit has zero surplus residual; output consistency cannot establish true pose.'))
    rows=[]
    for case in ['clear','side']:
        for line in (OUT/case/'trace.jsonl').open():
            r=json.loads(line);rs=r['observation']['rotors']
            if len(rs)!=4 or any(q['mode']!='ellipse' for q in rs):continue
            pts=np.array([[q['cx'],q['cy']] for q in rs]);pts=pts[np.argsort(np.arctan2(*(pts-pts.mean(0))[:,::-1].T))]
            accepted=initialization(rs)[0] is not None
            for hidden in range(4):
                ids=np.array([i for i in range(4) if i!=hidden]);A=fit_affine(ids,pts[ids])
                error=float(np.linalg.norm(X[hidden]@A.T-pts[hidden]))
                visible=float(np.max(np.linalg.norm(X[ids]@A.T-pts[ids],axis=1)))
                rows.append(dict(case=case,t=r['t'],hidden=hidden,error_px=error,
                                 maximum_visible_fit_residual_px=visible,initialization_qualified=accepted))
    summary=[]
    for case,qualified in itertools.product(['clear','side'],[False,True]):
        r=[v for v in rows if v['case']==case and (not qualified or v['initialization_qualified'])]
        summary.append(dict(case=case,qualified_only=qualified,samples=len(r),frames=len(r)//4,
            median_error_px=float(np.median([v['error_px'] for v in r])) if r else None,
            p95_error_px=float(np.percentile([v['error_px'] for v in r],95)) if r else None))
    save(OUT/'withheld_candidates.json',dict(summary=summary,rows=rows,
         independent_truth=False,known_cyclic_correspondences=True))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
