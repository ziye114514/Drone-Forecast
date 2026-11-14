"""Known synthetic rotor only: bridge geometry, frequency convention and guards."""
from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from helixtrack_adapter import HelixCore
from event_drone_forecast.geometry import _distance
OUT=ROOT.parent/'output/helix_bridge180_20261107'

if __name__=='__main__':
    ellipse=dict(cx=1135.9,cy=39.4,major=25.,minor=12.,angle=16.3)
    p=np.array([ellipse['cx'],ellipse['cy'],ellipse['major']/2,ellipse['minor']/2,np.deg2rad(ellipse['angle'])])
    with HelixCore(ellipse,200.) as tracker:
        ring=tracker.ring();error=float(np.max(abs(_distance(ring,p))))
        assert error<1e-3
        # Independent physical generator: constant angular rotation in an
        # orthographically projected disk, not points from the adapter's H.
        n=24000;times=np.arange(n)*.12/n;phase=2*np.pi*200*times
        theta=phase/2+(np.arange(n)%2)*np.pi
        angle=np.deg2rad(ellipse['angle']);co,si=np.cos(angle),np.sin(angle)
        xy=np.c_[ellipse['major']/2*np.cos(theta),ellipse['minor']/2*np.sin(theta)]@np.array([[co,si],[-si,co]])+[ellipse['cx'],ellipse['cy']]
        e=np.zeros(n,dtype=[('x','u2'),('y','u2'),('p','i2'),('t','i8')])
        e['x']=np.rint(xy[:,0]);e['y']=np.rint(xy[:,1]);e['p']=np.arange(n)%2;e['t']=1000000+np.rint(times*1e6).astype(np.int64)
        state=tracker.process(e,cutoff_s=1.12);frequency_error=abs(state['phase_hz']-200)
        assert frequency_error<5
        rejected=False
        try:tracker.process(e[:1],cutoff_s=1.12)
        except ValueError:rejected=True
        assert rejected
    bad=False
    try:HelixCore(ellipse,0.)
    except ValueError:bad=True
    assert bad
    result=dict(author_core_compiled=True,conic_initialization_max_residual_px=error,
        synthetic_event_phase_frequency_hz=200.,synthetic_estimated_phase_hz=state['phase_hz'],
        synthetic_error_hz=frequency_error,independent_affine_rotation_generator=True,
        zero_prior_rejected=True,overlap_interval_rejected=True,
        native_RPM_truth=False,forecast_improvement_tested=False,production_enabled=False)
    (OUT/'checks.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))
