"""Past whole-body support for current rotor geometry; raw frequency history stays."""
from pathlib import Path
import hashlib,importlib.util,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage54'))
from packets54 import Publisher
spec=importlib.util.spec_from_file_location('frozen_inside17_for55',ROOT/'research/stage17/frontend.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
inside=old.inside
PARAMS=dict(margin_fraction=.25,expiry_us=160000,geometry_only=True,
    source='Reuse actual17 inside;25% from48 body association and160ms existing expiry. No threshold tuning. No box-derived fake rotor, no altered FFT/phase history sampling.',
    temporal='Hint must be known by packet start, not merely by the20ms endpoint. Each10ms geometry uses only its current events. Empty/stale/invalid hint falls back to original full-frame geometry.')

def named_fields_sha(events):
    h=hashlib.sha256();names=sorted(events.dtype.names)
    h.update(json.dumps(dict(count=len(events),fields=[(n,events.dtype.fields[n][0].str) for n in names]),sort_keys=True).encode('utf-8'))
    for name in names:h.update(np.ascontiguousarray(events[name]).tobytes())
    return h.hexdigest()

class SemanticPublisher(Publisher):
    def publish(self,events,request_id,cutoff,verify):
        header=super().publish(events,request_id,cutoff,verify);start=time.perf_counter()
        header['named_fields_sha256']=named_fields_sha(events) if verify else None;header['named_fields_hash_ms']=(time.perf_counter()-start)*1000
        return header

def validate_hint_request(request):
    hint=request.get('body_hint')
    if hint is not None:
        assert np.isfinite(hint['t']) and round(hint['t']*1e6)<=request['packet']['start_us'],'ROI hint from a future subframe'

class GeometryROI:
    def __init__(self,width,height,mode):assert mode in ['off','body'];self.width=width;self.height=height;self.mode=mode
    def apply(self,events,hint,t):
        d=dict(mode='global',reason='disabled' if self.mode=='off' else 'no_hint',raw_events=len(events),geometry_events=len(events),hint_t=None,roi=None)
        if self.mode=='off' or hint is None or not hint.get('detected'):return events,d
        assert np.isfinite(hint['t']) and hint['t']<=t+1e-8,'Future ROI hint'
        age=round((t-hint['t'])*1e6);d.update(hint_t=hint['t'],hint_age_us=age)
        if age>PARAMS['expiry_us']:d['reason']='hint_expired';return events,d
        box=np.asarray(hint.get('bbox',[]),float)
        if box.shape!=(4,) or not np.isfinite(box).all() or np.any(box[2:]<=box[:2]):d['reason']='invalid_box';return events,d
        pad=(box[2:]-box[:2])*PARAMS['margin_fraction'];lo=np.maximum(np.floor(box[:2]-pad),0).astype(int);hi=np.minimum(np.ceil(box[2:]+pad),[self.width,self.height]).astype(int)
        if np.any(hi<=lo):d['reason']='outside_fov';return events,d
        roi=[int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])];filtered=inside(events,roi)
        d.update(mode='body_roi',reason='past_body_box',geometry_events=len(filtered),roi=roi,pixel_area_fraction=float(np.prod(hi-lo)/(self.width*self.height)))
        return filtered,d
