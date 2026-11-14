"""Past frequency hypotheses avoid a dense rediscovery grid every current tick."""
from pathlib import Path
import sys,json,time,os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[variable]='1'
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'tools')]
from predictive_harmonics import predictive_harmonics
from native_event_alignment import past_warp


def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);summary=[]
    prior=out.parent/'native_pilot'
    for name,folder in [('clear','raw_release_clear'),('partial','raw_release_clear'),('side','raw_release_side')]:
        packet=np.load(prior/name/'native_window.npz');e=packet['events'];t=float(packet['t'])
        rows=[json.loads(s) for s in (ROOT.parent/'output/three_continuation_20261026'/folder/'predictions.jsonl').read_text('utf-8').splitlines()]
        i=min(range(len(rows)),key=lambda i:abs(rows[i]['t']-t));old=rows[i-1]
        hypotheses=[r['frequency_hz'] for r in old['rates'] if r.get('valid')]
        if old['period'].get('qualified'):hypotheses.append(old['period']['frequency_hz'])
        hypotheses=sorted(set(float(f) for f in hypotheses if f is not None and np.isfinite(f)))
        assert old['t']<=t-33333/1e6+1e-7 and hypotheses
        warped,diag=past_warp(e,t,old['coefficient']['velocity'],old['t'])
        start=time.perf_counter();r=predictive_harmonics(warped,t,base_frequencies=hypotheses);elapsed=(time.perf_counter()-start)*1000
        directory=out/name;directory.mkdir()
        cv2.imwrite(str(directory/'predictive_mask.png'),r['mask'])
        np.savez_compressed(directory/'evidence.npz',frequency_hz=r['frequency_hz'],evidence=r['evidence'],spatial_phase=r['spatial_phase'])
        record=dict(name=name,t=t,prior_source_t=old['t'],prior_frequencies_hz=hypotheses,
            accepted_pixels=r['accepted_pixels'],CPU_ms=elapsed,warp=diag,
            future_current_events_not_used_to_choose_candidate_band=True,motor_ID_of_prior_blobs_not_assumed=True,
            candidate_frequency_bank_size=len(hypotheses),dense_bank_size=57,not_end_to_end_FPS=True)
        (directory/'record.json').write_text(json.dumps(record,indent=2),'utf-8');summary.append(record);print(json.dumps(record),flush=True)
    (out/'summary.json').write_text(json.dumps(dict(records=summary,labels_read=False,only_cached_RAW_used=True,
        production_default_unchanged=True),indent=2),'utf-8')


if __name__=='__main__':main(sys.argv[1])
