"""Same last-state semantics and exact final costs/counters; no speed claim."""
from pathlib import Path
import copy,json,pickle,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from forecast_output import LeanHistory


def main():
    out=ROOT.parent/'output/fred_transfer_20261029'
    run=out/'RAW_clear_integrated2s'
    rows=[json.loads(x) for x in (run/'predictions.jsonl').read_text().splitlines()]
    timing=[json.loads(x) for x in (run/'timings.jsonl').read_text().splitlines()]
    history=LeanHistory();assert not history and not history[1:]
    old=[]
    for r,t in zip(rows,timing):
        row=copy.deepcopy(r);row['cost']={k:v for k,v in t.items() if k!='t'}
        history.append(row);old.append(row)
        assert history[-1] is row and len(history)==len(old)
        assert history[-1]['structure']==old[-1]['structure']
        assert history[-1]['force']==old[-1]['force']
    columns=list(history)
    for r,t in zip(columns,timing):
        assert dict(t=r['t'],**r['cost'])==t
    for a,b in zip(old[1:],history[1:]):
        for key in ('cost','t','valid'):assert a[key]==b[key]
        assert bool(a['body'].get('detected'))==b['body']['detected']
        assert a['structure']['mode']==b['structure']['mode']
        assert a['force']['active']==b['force']['active']
        assert a['force']['gain_learned']==b['force']['gain_learned']
        assert bool((a.get('axis_amplitude_bridge') or {}).get('applied'))==b['axis_amplitude_bridge']['applied']
        assert bool((a.get('nn_structure_schedule') or {}).get('deferred_original_due'))==b['nn_structure_schedule']['deferred_original_due']
    # Later optional cost columns and final post-append write costs are exact.
    r=copy.deepcopy(old[-1]);r['cost']['new_optional_cost']=.125;history.append(r)
    r['cost']['output_ms']+=.01;history.finish_current_costs()
    assert 'new_optional_cost' not in list(history)[0]['cost']
    assert list(history)[-1]['cost']==r['cost']
    large=LeanHistory();full=[]
    for k in range(30):
        for row in old:
            row=copy.deepcopy(row);row['t']+=k*2
            full.append(row);large.append(row)
    result=dict(frozen_records=len(old),timings_and_summary_inputs_exact=True,current_full_state_identity_preserved=True,
        optional_late_cost_columns_exact=True,final_post_write_cost_update_exact=True,
        scaling_records=len(large),full_history_pickle_bytes=len(pickle.dumps(full)),
        lean_history_pickle_bytes=len(pickle.dumps(large)),numeric_storage_bytes=large.numeric_storage_bytes,
        full_research_records_in_memory=1,exact_timing_storage_still_linear=True,RAW_FPS_measured=False)
    (out/'lean_history_check.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
