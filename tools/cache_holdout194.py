"""Read only the declared fresh recording14 interval, preserving original data."""
from pathlib import Path
import json,hashlib,numpy as np
from metavision_core.event_io import EventsIterator
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    spec=json.loads((ROOT.parent/'output/timing194_20261107/freeze.json').read_text('utf-8'))['recording14']
    raw=Path(spec['raw']);dest=ROOT.parent/'output/phase185_20261107/cache_recording14';dest.mkdir(exist_ok=False)
    begin=round(spec['start']*1e6);end=round((spec['start']+spec['duration'])*1e6);start=begin//33333*33333
    manifest=dict(storage='npy',spec=spec,frames=[],input_bytes=raw.stat().st_size,labels_or_model_used=False)
    it=EventsIterator(str(raw),start_ts=start,max_duration=end-start,delta_t=33333,relative_timestamps=False)
    for i,e in enumerate(it):
        e=e[(e['t']>=begin)&(e['t']<end)];p=dest/f'{i:04d}.npy';np.save(p,e)
        manifest['frames'].append(dict(index=i,t=min(end,it.get_current_time())/1e6,events=len(e),file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2),'utf-8');print(dict(frames=len(manifest['frames']),events=sum(r['events'] for r in manifest['frames'])))
