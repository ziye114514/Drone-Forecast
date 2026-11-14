"""Past-provided-box diagnostic, explicitly NOT RAW-only inference.

Compare at exactly the existing RAW candidate origins. A causal reader only
hands over annotations stamped at/before each current cutoff; never a future
interpolation. These controls isolate position input from future dynamics.
"""
from pathlib import Path
import copy
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from prepare_fred_transfer import parse,OUT
from run_fast_structure import PhysicalPositionState
from unknown_force_feedback import UnknownForceFeedback
from score_fred_transfer import score


def main():
    out=OUT/'past_box_controls';out.mkdir(exist_ok=False)
    protocol=json.loads((OUT/'protocol.json').read_text());frozen={}
    for s in protocol['sequences']:
        labels=parse(Path(s['annotation']).read_text('utf-8-sig'))
        source=[json.loads(x) for x in (OUT/f"RAW_{s['sequence']}_candidate/predictions.jsonl").read_text().splitlines()]
        state=PhysicalPositionState();force=UnknownForceFeedback();index=0;past=None
        controls={k:[] for k in ('same_observed_pv_CV','past_provided_box_CV','past_provided_box_force_control')}
        for r in source:
            while index<len(labels) and labels[index]['t']<=r['t']+1e-9:
                past=labels[index];index+=1
            available=past is not None and 0<=r['t']-past['t']<=.05+1e-9
            body=dict(detected=False)
            if available:
                b=np.asarray(past['bbox'])
                body=dict(detected=True,bbox=b.tolist(),center=((b[:2]+b[2:])/2).tolist(),confidence=1.,
                    measurement_source='provided_past_FRED_box',label_source_t=past['t'])
                assert past['t']<=r['t']+1e-9
            c,valid,_=state.step(r['t'],body)
            p,f=force.step(r['t'],c,body.get('center'),None,0.)
            original=r['coefficient']
            cv={} if original is None else {str(h):(np.asarray(original['position'])+h*np.asarray(original['velocity'])).tolist() for h in (.1,.2,.4)}
            ideal={} if c is None else {str(h):(np.asarray(c['position'])+h*np.asarray(c['velocity'])).tolist() for h in (.1,.2,.4)}
            controls['same_observed_pv_CV'].append(dict(r,physical=cv))
            for name,value in [('past_provided_box_CV',ideal),('past_provided_box_force_control',p)]:
                controls[name].append(dict(r,body=body,coefficient=c,physical=value,valid=bool(r['valid'] and valid),
                    diagnostic_control_not_RAW_only=True,provided_position_t=None if not available else past['t']))
        for name,rows in controls.items():
            (out/f"{s['sequence']}_{name}.jsonl").write_text('\n'.join(json.dumps(r,allow_nan=False) for r in rows)+'\n','utf-8')
        frozen[str(s['sequence'])]=(s,controls)
    result={}
    for seq,(s,controls) in frozen.items():
        labels=parse(Path(s['annotation']).read_text('utf-8-sig'))
        result[seq]={name:score(rows,labels,s['start'],s['duration']) for name,rows in controls.items()}
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),'utf-8')
    print(json.dumps({seq:{k:dict(error=v['error_04s_px'],coverage=v['forecast_coverage']) for k,v in modes.items()} for seq,modes in result.items()},indent=2))


if __name__=='__main__':main()
