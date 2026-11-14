"""Frozen current observations, unchanged force path, one shared state replaced."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from singer_position_state import SingerPositionState,transition
import probe_same_window_motion as engine
from score_fred_transfer import score
from prepare_fred_transfer import parse


def main(output):
    out=Path(output);options=dict(curved=True,evidence_gain=True,memory=True,endpoint=True,memory_gaps=True,weak_support=True,
        position_history_rebase=True,contained_reanchor_motion=True)
    F,Q=transition(33333);assert np.linalg.eigvalsh(Q).min()>-1e-8
    assert np.max(abs(F@np.array([1.,2.,3.,4.,0.,0.])-np.array([1+3*.033333,2+4*.033333,3,4,0,0])))<1e-10
    original=engine.PhysicalPositionState;results={}
    for name,folder,lab,start in [('FRED9','RAW_FRED9_adapt4s','stability_audit_20261031/holdout9/input/coordinates.txt',32),
        ('FRED7','RAW_FRED7_adapt4s','fred_transfer_20261029/unseen7/input/7/coordinates.txt',20)]:
        path=out/folder/'predictions.jsonl'
        if not path.exists() or not (path.parent/'summary.json').exists():continue
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        engine.PhysicalPositionState=original;control,_=engine.replay(rows,False,**options)
        delta=max(np.linalg.norm(np.array(a['physical'][h])-b['physical'][h]) for a,b in zip(rows,control) for h in a['physical'])
        assert delta<1e-7,(name,delta)
        engine.PhysicalPositionState=SingerPositionState;candidate,cost=engine.replay(rows,False,**options)
        labels=parse((ROOT.parent/'output'/lab).read_text('utf-8-sig'))
        result=dict(control=score(rows,labels,start,4),candidate=score(candidate,labels,start,4),baseline_replay_max_delta=delta,
            CPU_mean_ms=float(np.mean(cost)))
        results[name]=result
        (out/(name+'_singer_replay.jsonl')).write_text('\n'.join(json.dumps(x) for x in candidate)+'\n','utf-8')
        print(name,result['control']['error_04s_px'],result['candidate']['error_04s_px'],flush=True)
    engine.PhysicalPositionState=original
    (out/'singer_replay_scores.json').write_text(json.dumps(results,indent=2),'utf-8')


if __name__=='__main__':main(sys.argv[1])
