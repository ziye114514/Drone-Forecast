"""Causal same-input motion qualification pilot; no RAW speed claim."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from probe_same_window_motion import replay
from compare_final_pipeline import score, stats
from score_fred_transfer import score as score_fred
from prepare_fred_transfer import parse, OUT as FRED


def run():
    visible='--visible' in sys.argv
    shorter='--shorter' in sys.argv
    tangent='--tangent' in sys.argv
    commit='--commit-velocity' in sys.argv
    calibrated='--commit-calibrated' in sys.argv
    out=FRED/('calibrated_velocity_commit' if calibrated else 'shared_velocity_commit' if commit else 'tangential_force_response' if tangent else 'motion_visibility_shorter' if shorter else 'motion_visibility' if visible else 'motion_observability_v3')
    out.mkdir(exist_ok=False)
    jobs={f'fred{n}':FRED/f'RAW_{n}_candidate' for n in (0,1,2)}
    jobs.update(side=ROOT.parent/'output/motion_consistency_30px_20261028/RAW_side_release_candidate8s',
                other=ROOT.parent/'output/thrust_rate_review_20261029/experiment/other_baseline.jsonl')
    frozen={};details={}
    assert all((p if p.is_file() else p/'predictions.jsonl').is_file() for p in jobs.values())
    for name,path in jobs.items():
        data=(path if path.is_file() else path/'predictions.jsonl').read_bytes()
        rows=[json.loads(x) for x in data.splitlines()]
        control,cost=replay(rows,False,curved=True,evidence_gain=True,memory=True,endpoint=True,memory_gaps=True)
        max_diff=max((max(np.max(abs(np.asarray(r['physical'][h])-b['physical'][h]))
                         for h in r['physical']) for r,b in zip(rows,control) if r['physical']),default=0.)
        # Older train2 source precedes the default endpoint/memory extension;
        # those changes were previously demonstrated to leave it identical.
        if max_diff>1e-7:raise AssertionError((name,max_diff))
        modified,cost=replay(rows,False,curved=True,evidence_gain=True,memory=True,endpoint=True,memory_gaps=True,weak_support=True,visible_support=visible,shorter_memory=shorter,tangent_response=tangent,commit_motion_velocity=commit,calibrated_velocity_commit=calibrated)
        assert all(a['body']==b['body'] and a['valid']==b['valid'] for a,b in zip(rows,modified))
        for variant,output in [('control',control),('motion',modified)]:
            (out/f'{name}_{variant}.jsonl').write_text('\n'.join(json.dumps(r,allow_nan=False) for r in output)+'\n','utf-8')
        frozen[name]=(control,modified)
        details[name]=dict(input=str(path),input_sha256=hashlib.sha256(data).hexdigest(),
            reproduced_default_max_difference_px=float(max_diff),
            qualified_weak_ticks=sum(bool((r['image_motion'].get('weak_semantic_motion') or {}).get('qualified')) for r in modified),
            control_motion_ticks=sum(r['image_motion']['observation']['valid'] for r in control),
            motion_ticks=sum(r['image_motion']['observation']['valid'] for r in modified),
            component_CPU_ms=stats(cost),body_and_position_eligibility_exactly_preserved=True)
    # All predictions have been saved before scoring annotations are opened.
    protocol=json.loads((FRED/'protocol.json').read_text())
    for name,(a,b) in frozen.items():
        if name.startswith('fred'):
            seq=next(r for r in protocol['sequences'] if r['sequence']==int(name[4:]))
            labels=parse(Path(seq['annotation']).read_text('utf-8-sig'))
            scoring=lambda rows:score_fred(rows,labels,seq['start'],seq['duration'])
            details[name]['scores']={k:scoring(v) for k,v in [('control',a),('motion',b)]}
        else:
            label=ROOT.parent/('output/dense_labels_side_110_1195/labels.json' if name=='side' else 'output/dense_labels_train2_40_46/labels.json')
            truths=sorted(json.loads(label.read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
            details[name]['scores']={k:score(v,truths) for k,v in [('control',a),('motion',b)]}
    (out/'summary.json').write_text(json.dumps(details,indent=2,allow_nan=False),'utf-8')
    print(json.dumps({name:dict(weak=d['qualified_weak_ticks'],motion=[d['control_motion_ticks'],d['motion_ticks']],
        errors={k:(v['error_04s_px'] if name.startswith('fred') else v['physical_future_error_px']['0.4']) for k,v in d['scores'].items()}) for name,d in details.items()},indent=2),flush=True)


if __name__=='__main__':run()
