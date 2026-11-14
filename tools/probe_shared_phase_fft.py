"""Bounded RAW module probe; frozen upstream causal geometry, no labels/GPU."""
from pathlib import Path
import sys,json,time,hashlib,os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'src')]
from live_decode import native_batches
from shared_phase_fft import SharedPhaseFFT


def run(source,output):
    from metavision_sdk_core import RoiMaskAlgorithm
    source=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((source/'protocol.json').read_text('utf-8'))
    upstream=list(map(json.loads,(source/'predictions.jsonl').read_text('utf-8').splitlines()))
    mask=np.random.default_rng(141000).integers(0,4,size=(720,1280),dtype=np.uint8)==0
    pixel_filter=RoiMaskAlgorithm(mask.astype(float));buffer=pixel_filter.get_empty_output_buffer()
    model=SharedPhaseFFT();rows=[];native=selected=0
    snapshot=out/'source_snapshot';snapshot.mkdir()
    hashes={}
    for f in ['shared_phase_fft.py','tools/probe_shared_phase_fft.py']:
        b=(ROOT/f).read_bytes();p=snapshot/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);hashes[f]=hashlib.sha256(b).hexdigest()
    config=dict(source=str(source),saved_upstream_causal_geometry=True,source_hashes=hashes,
        fixed_native_pixel_fraction=.25,all_temporal_events_at_selected_pixels=True,
        current_bin_us=model.bin_us,rolling_window_us=model.window_us,labels_read=False,
        mechanical_RPM_truth=False,force_prediction_changed=False,end_to_end_FPS=False)
    (out/'protocol.json').write_text(json.dumps(config,indent=2),'utf-8')
    with (out/'observations.jsonl').open('w',encoding='utf-8') as stream:
        for r,(t,e) in zip(upstream,native_batches(protocol['raw'],protocol['actual_iterator_start'],protocol['actual_duration'],False,protocol['native_step_us'])):
            assert abs(t-r['t'])<1e-8;native+=len(e);start=time.perf_counter()
            pixel_filter.process_events(e,buffer);e=buffer.numpy();selected+=len(e)
            c=r['coefficient'];velocity=[0,0] if c is None else c['velocity']
            result=model.step(e,t,r['body'],r['structure'],r['geometry'].get('rotors',[]),velocity)
            result['processing_ms']=(time.perf_counter()-start)*1000;result['t']=t;rows.append(result)
            stream.write(json.dumps(result)+'\n')
    summary=dict(source_ticks=len(rows),native_events=native,phase_input_events=selected,
        qualified_counts=dict(Counter(r['qualified_count'] for r in rows)),
        current_qualified_rotor_estimates=sum(r['qualified_count'] for r in rows),
        reasons=dict(Counter(r['reason'] for r in rows)),
        component_ms=dict(mean=float(np.mean([r['processing_ms'] for r in rows])),
            median=float(np.median([r['processing_ms'] for r in rows])),p95=float(np.percentile([r['processing_ms'] for r in rows],95))),
        timing_scope='Native fixed-pixel selection plus shared geometry rectification, moments and FFT; excludes RAW seek/decoder and upstream detector',
        production_default_changed=False,mechanical_RPM_truth=False,prediction_accuracy_scored=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),'utf-8');print(json.dumps(summary))


if __name__=='__main__':run(sys.argv[1],sys.argv[2])
