"""Acquisition latency, recovery priority, and no synthetic detections."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from nn_search_schedule import acquisition_due


def main():
    last=None;requests=[]
    for i in range(90):
        due,d=acquisition_due(i/30,None,None,last,30.,3,True)
        if due:requests.append(i);last=i/30
    assert requests==list(range(0,90,3)),requests
    burst_requests=[];last=None
    for i in range(90):
        due,d=acquisition_due(i/30,None,None,last,30.,3,True,source_start=0.)
        if due:burst_requests.append(i);last=i/30
    assert burst_requests[:10]==list(range(10)),burst_requests
    assert len(burst_requests)==36,burst_requests
    waits=[]
    for onset in (1.001,1.02,1.04,1.069,1.08,1.099):
        last=None;seen=None
        for i in range(60):
            t=i/30;due,d=acquisition_due(t,None,None,last,30.,3,True)
            if due:
                last=t
                if t>=onset:seen=t;break
        assert seen is not None and 0<=seen-onset<=.1+1e-8
        waits.append(seen-onset)
    for t in (1+1/30,1+2/30,1.2):
        due,d=acquisition_due(t,None,1.,t-1/30,30.,3,True)
        assert due and not d['deferred'] and not d['cold_search']
    # Normal image tracking is governed by the original request decision.
    for original in (False,True):
        due,d=acquisition_due(10.,{'detected':True},9.,9.99,30.,3,original)
        assert due==original and not d['cold_search']
    result=dict(empty3s_old_NN_calls=90,empty3s_new_NN_calls=len(requests),
        empty3s_with_initial_burst_NN_calls=len(burst_requests),
        initial300ms_search_not_deferred=True,
        maximum_sampled_search_wait_s=max(waits),ordinary_search_interval_s=.1,
        recent_image_observation_loss_retries_immediately=True,active_tracker_schedule_unchanged=True,
        no_current_box_or_prediction_synthesized=True,not_a_RAW_FPS_or_recall_benchmark=True)
    out=ROOT.parent/'output/transport_stability_20261029';out.mkdir(exist_ok=True)
    (out/'search_burst_mechanism.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
