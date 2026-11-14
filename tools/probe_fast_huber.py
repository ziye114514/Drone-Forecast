"""Same histories and robust objective; compare full physical forecasts/cost."""
from pathlib import Path
import os,sys,json,time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import fast_huber
from probe_same_window_motion import replay
from probe_gap_segments import OPTIONS


def main(out):
    out.mkdir(exist_ok=False);base=ROOT.parent/'output';result={};total_native=total_fallback=0
    cases=[('train2','target25_continuity_20261031/RAW_other_center6s'),
        ('side','reacquisition_motion_20261029/RAW_side_default8s'),
        ('FRED0','target25_continuity_20261031/RAW_fred0_center4s'),
        ('FRED1','fred_domain_adaptation_20261029/RAW_1_verified'),
        ('FRED2','fred_transfer_20261029/RAW_2_candidate'),
        ('FRED7','fred_domain_adaptation_20261029/RAW_7_baseline'),
        ('clear','target25_continuity_20261031/RAW_clear_center4s')]
    warm=fast_huber.configure(True);fast_huber.configure(False)
    for name,folder in cases:
        rows=[json.loads(s) for s in (base/folder/'predictions.jsonl').read_text().splitlines()]
        options=dict(OPTIONS)
        if name=='FRED2':options.update(weak_support=False,position_history_rebase=False,contained_reanchor_motion=False)
        # Warm Python paths first. Alternate order for timing after warmup.
        fast_huber.configure(False);reference,_=replay(rows,False,**options)
        baseline_difference=max((float(np.max(abs(np.asarray(r['physical'][h])-b['physical'][h]))) for r,b in zip(rows,reference) for h in r['physical']),default=0.)
        assert baseline_difference<1e-7,(name,baseline_difference)
        times={False:[],True:[]};costs={};new=None
        for enabled in (True,False,False,True):
            fast_huber.configure(enabled);start=time.perf_counter();value,cost=replay(rows,False,**options)
            times[enabled].append((time.perf_counter()-start)*1000);costs[enabled]=cost
            if enabled:new=value
        differences=[float(np.linalg.norm(np.asarray(a['physical'][h])-b['physical'][h])) for a,b in zip(reference,new) for h in a['physical']]
        changed_gates=[a['t'] for a,b in zip(reference,new) if a['image_motion']['acceleration'].get('applied')!=b['image_motion']['acceleration'].get('applied')]
        result[name]=dict(source_ticks=len(rows),reference_reproduced_max_px=baseline_difference,
            max_prediction_difference_px=max(differences,default=0.),changed_acceleration_gates=changed_gates,
            old_ms=float(np.median(times[False])),new_ms=float(np.median(times[True])),
            old_frame_p95_ms=float(np.percentile(costs[False],95)),new_frame_p95_ms=float(np.percentile(costs[True],95)))
        (out/(name+'.jsonl')).write_text('\n'.join(json.dumps(r,allow_nan=False) for r in new)+'\n','utf-8')
    result={'warmup':warm,'counts':fast_huber.counters(),'clips':result,
        'RAW_FPS_measured':False,'history_and_objective_unchanged':True,'parameters_swept':False}
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),'utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main(Path(sys.argv[1]))
