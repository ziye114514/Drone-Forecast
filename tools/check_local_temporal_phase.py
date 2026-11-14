"""Small synthetic mechanisms only; no RAW, GPU, labels or timing claims."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from local_temporal_phase import local_temporal_phase


def stream(f,t,velocity=(0.,0.)):
    times=np.arange(t-.033333+.0001,t,1/(2*f))
    e=np.zeros(len(times)*12,dtype=[('x','u2'),('y','u2'),('p','i2'),('t','i8')])
    e['t']=np.repeat(np.rint(times*1e6).astype(np.int64),12)
    e['p']=np.repeat(np.arange(len(times))%2,12)
    age=t-e['t']/1e6
    e['x']=np.rint(305-age*velocity[0]+np.tile(np.arange(12)%4,len(times)))
    e['y']=np.rint(205-age*velocity[1]+np.tile(np.arange(12)//4,len(times)))
    return e


def main(output):
    kw=dict(center=[300.,200.],velocity=[0.,0.],candidates_hz=[150.,300.,600.],rotor_support_confirmed=True)
    results={}
    for f in [300.,600.]:
        r=local_temporal_phase(stream(f,1.),1.,**kw)
        qualified=[v['frequency_hz'] for v in r['regions'] if v['qualified']]
        assert qualified and all(v==f for v in qualified),r
        results[str(f)]=r
    e=stream(300.,1.);future=e.copy();future['t']+=1000000
    assert local_temporal_phase(np.concatenate([e,future]),1.,**kw)==results['300.0']
    moving=local_temporal_phase(stream(300.,1.,(600.,0.)),1.,**dict(kw,velocity=[600.,0.]))
    assert moving['qualified_regions']>0,moving
    assert local_temporal_phase(e,1.,center=[300,200],velocity=[0,0],candidates_hz=[300.])['reason']=='requires_upstream_rotor_support'
    rng=np.random.default_rng(164);noise=e.copy();noise['p']=rng.integers(0,2,len(noise))
    assert local_temporal_phase(noise,1.,**kw)['qualified_regions']==0
    over=local_temporal_phase(np.tile(e,20),1.,**dict(kw,max_events_per_cell=100))
    assert over['qualified_regions']==0 and over['regions'][0]['reason']=='capacity_exceeded_no_temporal_subsampling'
    summary=dict(synthetic_only=True,no_shared_plane_or_hub_required=True,
        current_300_to_600_candidates_followed=True,future_events_excluded=True,
        translation_compensation_checked=True,random_polarity_rejected=True,
        missing_rotor_support_refused=True,capacity_overflow_explicit=True,
        actual_RAW_accuracy_or_FPS_improvement_proven=False,
        periodic_light_not_distinguished_by_this_module=True,results=results)
    Path(output).write_text(json.dumps(summary,indent=2),'utf-8');print(json.dumps({k:v for k,v in summary.items() if k!='results'}))


if __name__=='__main__':main(sys.argv[1])
