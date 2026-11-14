"""Independent rotating and static negative controls, then one saved native ring."""
from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from observed_ring_phase import ring_phase_prior
from pilot_author_phase180 import transformed
from pure_visibility170 import window
OUT=ROOT.parent/'output/helix_bridge180_20261107'

if __name__=='__main__':
    ellipse=dict(cx=100.,cy=80.,major=40.,minor=20.,angle=15.,ratio=.5)
    n=10000;t=np.arange(n)*.033333/n;theta=np.pi*195*t+(np.arange(n)%2)*np.pi
    # For phase frequency195Hz and two blades, theta'=pi*195rad/s.
    angle=np.deg2rad(ellipse['angle']);R=np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
    xy=np.c_[20*np.cos(theta),10*np.sin(theta)]@R+[100,80]
    e=np.zeros(n,dtype=[('x','u2'),('y','u2'),('p','i2'),('t','i8')]);e['x']=np.rint(xy[:,0]);e['y']=np.rint(xy[:,1]);e['t']=1000000+np.rint(t*1e6).astype(np.int64)
    positive=ring_phase_prior(e,ellipse,1.033333);assert positive['qualified'] and abs(positive['phase_hz']-195.)<1
    future=e[:100].copy();future['t']+=1000000
    assert ring_phase_prior(np.r_[e,future],ellipse,1.033333)==positive
    static=e.copy();rng=np.random.default_rng(180);order=rng.permutation(n);static['x']=e['x'][order];static['y']=e['y'][order]
    negative=ring_phase_prior(static,ellipse,1.033333);assert not negative['qualified']
    changed=e.copy();later=changed['t']>=1016666
    changed['x'][later]=static['x'][later];changed['y'][later]=static['y'][later]
    changed_prior=ring_phase_prior(changed,ellipse,1.033333)
    assert changed_prior['candidate_signed_phase_hz']==positive['candidate_signed_phase_hz']
    probe=json.loads((OUT/'native_pilot.json').read_text('utf-8'));ellipse=probe['ellipse'];seed_t=probe['seed_t']
    rows=[json.loads(s) for s in (ROOT.parent/'output/layout176_20261107/integrated/train2_fixed.jsonl').read_text('utf-8').splitlines()]
    old=max((r for r in rows if r['t']<=seed_t-.033333),key=lambda r:r['t']);m=old['motion'];s=m['estimate'];v=np.array([s['dx'],s['dy']])/m['dt_s']
    events=window(ROOT.parent/'output/native_plane165_20261104/cache_train2_inside',seed_t)
    native=transformed(events,ellipse,np.zeros(2),v,seed_t);result=ring_phase_prior(native,ellipse,seed_t)
    output=dict(synthetic_rotating=positive,synthetic_temporally_shuffled=negative,
        future_events_do_not_change_result=True,validation_does_not_select_frequency=True,
        native=result,labels_used=False,geometry_source='177 current native isolated refit',
        independent_RPM_truth=False,production_enabled=False)
    (OUT/'ring_prior_checks.json').write_text(json.dumps(output,indent=2),'utf-8');print(json.dumps(output))
