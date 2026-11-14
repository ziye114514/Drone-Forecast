"""Geometric phase-coordinate controls, not a photometric event simulator."""
from pathlib import Path
import json,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from reference_api import AuthorTracker,OUT

N=90000;duration=.6;physical_hz=50.;blades=2
t=np.arange(N)/N*duration
blade=np.arange(N)%2;pol=(np.arange(N)//2)%2
radius=.82+.25*((np.arange(N)//4)%16)/15
theta=2*np.pi*physical_hz*t+blade*np.pi+np.where(pol>0,-.125,.125)
x=np.rint(128+80*radius*np.cos(theta));y=np.rint(128+80*radius*np.sin(theta))
tt=np.rint(t*1e6).astype(np.int64)+1000000
controls={}
for name,seed,xx,yy in [('rotor_seed_10pct_high',6600.,x,y),('rotor_half_phase_seed',3000.,x,y),
                       ('global_flicker',6600.,np.rint(128+80*radius*np.cos(np.arange(N)*2*np.pi/97)),np.rint(128+80*radius*np.sin(np.arange(N)*2*np.pi/97)))]:
    stamps=tt if name!='global_flicker' else (1000000+np.floor(t*100)*10000).astype(np.int64)
    trace=[];began=time.perf_counter()
    with AuthorTracker(seed,blades) as tracker:
        for start in range(0,N,3000):trace.append(tracker.feed(xx[start:start+3000],yy[start:start+3000],stamps[start:start+3000],pol[start:start+3000]))
    controls[name]=dict(seed_phase_equivalent_rpm=seed,trace=trace,wall_s=time.perf_counter()-began,
                        steady_phase_hz=float(np.median([s['phase_hz'] for s in trace[-10:]])),
                        steady_mechanical_rpm_under_known_B=float(np.median([s['mechanical_rpm_under_known_blade_count'] for s in trace[-10:]])),
                        steady_predictive_phase_concentration=float(np.median([s['predictive_phase_concentration'] for s in trace[-10:]])))
report=dict(scope='Unmodified author phase tracker in a fixed known circle, generated phase coordinates, no photometric event simulation, no real-flight RPM truth.',
            true_mechanical_hz=physical_hz,true_mechanical_rpm=physical_hz*60,true_blades=blades,
            expected_phase_hz=physical_hz*blades,pose_refinement_enabled=False,controls=controls)
(OUT/'synthetic_phase_controls.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:{n:v for n,v in q.items() if n!='trace'} for k,q in controls.items()}),flush=True)
