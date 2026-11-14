"""Score current short RAW attempts without hiding unavailable old origins."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from prepare_fred_transfer import parse
from score_fred_transfer import score,reference
from score_fred_adaptation import matched


def main(name,control,candidate,label,start):
    base=ROOT.parent/'output';out=base/'stability_audit_20261031'
    labels=parse((base/label).read_text('utf-8-sig'));runs={}
    for key,path in [('control',control),('candidate',candidate)]:
        runs[key]=[json.loads(s) for s in (base/path/'predictions.jsonl').read_text().splitlines()]
    assert [r['t'] for r in runs['control']]==[r['t'] for r in runs['candidate']]
    def at(t):
        box=reference(labels,t);return None if box is None else (box[:2]+box[2:])/2
    result={key:score(rows,labels,start,4.) for key,rows in runs.items()}
    result['matched']=matched(runs['control'],runs['candidate'],at)
    for key,rows in runs.items():
        result[key]['weak_recovery_attempts']=sum(bool((r.get('weak_body_continuation') or {}).get('attempted')) for r in rows)
        result[key]['weak_recoveries']=sum(bool((r.get('weak_body_continuation') or {}).get('accepted')) for r in rows)
        result[key]['large_motion_recoveries']=sum(bool((r['tracking'].get('wide_motion_search') or {}).get('old_size_displacement_gate_exceeded')) for r in rows)
    (out/(name+'_scores.json')).write_text(json.dumps(result,indent=2,allow_nan=False),'utf-8')
    print(json.dumps({key:{k:v for k,v in r.items() if k not in ('anchors','targets')} for key,r in result.items() if key in runs},indent=2))
    print(json.dumps({k:v for k,v in result['matched'].items() if k!='records'},indent=2))


if __name__=='__main__':main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],float(sys.argv[5]))
