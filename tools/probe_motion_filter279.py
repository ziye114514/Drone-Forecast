"""One upstream root: translation invariance of periodic rotor filtering.

Reuse project189's raw-activity proposal and early-half velocity, but evaluate
the SAME all-frame policy (including low velocity) before touching physics.
No reconstruction, NN, frequency threshold search or recording-specific code.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,argparse,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from moving_event_proposal import propose
from pure_visibility170 import window
from rotor_cuda_worker import CudaRotorFrontWorker


def cache_path(n):
    if n=='1':return ROOT.parent/'output/frozen_unseen278_20261108/cache'
    return ROOT.parent/f'output/{"motion181_20261107" if n=="15" else "phase185_20261107"}/cache_recording{n}'


def evidence(packet):
    on,off,meta=packet[0],packet[1],packet[2];native,diag=meta['native_precomputed']
    count=int(native.sum());accepted=int(packet[-1]['accepted_pixels'])
    collective=bool(diag.get('regional_seed_groups',0)>0 and diag.get('regional_seed_pixels',0)>=3)
    return dict(native_support=count,accepted_period_pixels=accepted,
        regional_seed_pixels=int(diag.get('regional_seed_pixels',0)),
        target_entry_qualified=bool(count>=12 and (accepted>=12 or collective)),
        period_median_hz=float(np.median(packet[-1]['frequency_hz'][packet[-1]['mask']>0])) if accepted else None)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recording',choices=['1','13','15','16'],required=True);args=p.parse_args();n=args.recording
    OUT=ROOT.parent/f'output/motion_filter279_20261108/recording{n}';OUT.mkdir(parents=True,exist_ok=False)
    files=['tools/probe_motion_filter279.py','moving_event_proposal.py','native_event_alignment.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    protocol=dict(change='same raw-activity ROI and early-half translation compensation on every packet; original period gates unchanged',
        source_sha256=hashes,recording=n,recording1_is_now_failure_analysis_development_not_heldout=True,
        NN_used=False,GPU_used=True,labels_read=False,physics_untouched=True,
        support_counts_are_not_prediction_accuracy=True,no_recording_specific_rules=True,no_threshold_search=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    cv2.setNumThreads(2);worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,remote_unpack=True,regional_graph=True)
    cache=cache_path(n);manifest=json.loads((cache/'manifest.json').read_text('utf8'));rows=[]
    try:
        for i,item in enumerate(manifest['frames']):
            t=item['t'];events=window(cache,t);raw=worker.prepare_uncached(events,t);base=evidence(raw)
            warped,proposal=propose(events,t,raw[0],raw[1],active=True)
            candidate=base if warped is None else evidence(worker.prepare_uncached(warped,t))
            rows.append(dict(t=t,raw=base,compensated=candidate,proposal=proposal,compensation_used=warped is not None))
            if i%45==0:print(n,i,'entry raw/aligned',sum(r['raw']['target_entry_qualified'] for r in rows),sum(r['compensated']['target_entry_qualified'] for r in rows),flush=True)
    finally:worker.cached=None;worker.close()
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    summary={k:dict(qualified_entry_frames=sum(r[k]['target_entry_qualified'] for r in rows),
        median_accepted_period_pixels=float(np.median([r[k]['accepted_period_pixels'] for r in rows])),
        mean_accepted_period_pixels=float(np.mean([r[k]['accepted_period_pixels'] for r in rows]))) for k in ['raw','compensated']}
    summary['compensation_frames']=sum(r['compensation_used'] for r in rows)
    (OUT/'results.json').write_text(json.dumps(dict(summary=summary,rows=rows,protocol=protocol),indent=2),'utf8')
    print(n,summary,flush=True)
