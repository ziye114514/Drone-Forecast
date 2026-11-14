"""Read only the two predeclared original RAW intervals; no scene/model search."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib
import numpy as np
from metavision_core.event_io import EventsIterator
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent/'output/phase185_20261107'

if __name__=='__main__':
    protocol=json.loads((BASE/'protocol.json').read_text('utf-8'));out=[]
    for spec in [protocol['fresh_test']]:
        n=Path(spec['raw']).parts[-3];dest=BASE/f'cache_recording{n}';dest.mkdir(exist_ok=True)
        if (dest/'manifest.json').exists() or any(dest.glob('*.npy')):raise RuntimeError('Do not overwrite a completed/partial cache')
        raw=Path(spec['raw']);manifest=dict(storage='npy',spec=dict(raw=str(raw),start=spec['start'],duration=spec['duration']),
            input_bytes=raw.stat().st_size,frames=[],labels_or_model_used=False)
        begin=round(spec['start']*1e6);end=round((spec['start']+spec['duration'])*1e6)
        decoder_start=begin//33333*33333
        manifest.update(decoder_start_us=decoder_start,requested_start_us=begin,requested_end_us=end,
            SDK_grid_start_precedes_requested_by_us=begin-decoder_start)
        it=EventsIterator(str(raw),start_ts=decoder_start,max_duration=end-decoder_start,delta_t=33333,relative_timestamps=False)
        for i,e in enumerate(it):
            e=e[(e['t']>=begin)&(e['t']<end)]
            t=float(min(end,it.get_current_time())/1e6);p=dest/f'{i:04d}.npy';np.save(p,e)
            manifest['frames'].append(dict(index=i,t=t,events=len(e),file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        (dest/'manifest.json').write_text(json.dumps(manifest,indent=2),'utf-8');out.append(dict(recording=n,frames=len(manifest['frames']),native_events=sum(r['events'] for r in manifest['frames'])))
        print(out[-1],flush=True)
    (BASE/'holdout_cache_completion.json').write_text(json.dumps(out,indent=2),'utf-8')
