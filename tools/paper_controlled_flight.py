"""Controlled observation-level dynamics experiment; explicitly not camera data.

Verifies the prediction link from observed tilt/RPM to future position. Known
pixel gravity matches the nominal prior; it cannot validate real calibration.
"""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'src'))
from event_drone_forecast.paper_config import PaperConfig
from event_drone_forecast.paper_bank import PaperForecastBank
from event_drone_forecast.paper_physics import weak_normal
from event_drone_forecast.motion import extrapolate
out=ROOT.parent/'output/paper_reproduction_v030/controlled_flight';out.mkdir(exist_ok=True,parents=True)
dt=.01;times=np.arange(0,8+dt/2,dt);rng=np.random.default_rng(16);cfg=PaperConfig();bank=PaperForecastBank(cfg)
position=np.zeros((len(times),2));velocity=np.zeros_like(position);position[0]=[300,300];velocity[0]=[70,10]
truth_forces=[];outputs=[];ratio=.4;k=200/np.sqrt(1-ratio*ratio)
for i,t in enumerate(times):
    segment=min(int(t//2),3);angle=(0.,.22,-.12,0.)[segment];s=(1.,1.05,.95,1.)[segment]
    r=dict(cx=0.,cy=0.,major=80.,minor=80*ratio,angle=float(np.rad2deg(angle)),ratio=ratio,
           geometry_quality=.95,activity=5000.,mode='ellipse',axis_only=False,observed_now=True)
    n=weak_normal(r);a=k*s*n[:2]-200*np.asarray(cfg.image_up);truth_forces.append(a)
    if i:
        # Propagate with previous command, no command is visible in advance.
        position[i]=position[i-1]+velocity[i-1]*dt+.5*np.asarray(truth_forces[i-1])*dt**2
        velocity[i]=velocity[i-1]+np.asarray(truth_forces[i-1])*dt
    z=position[i]+rng.normal(0,.6,2)
    rotors=[dict(r,cx=float(z[0]+x),cy=float(z[1]+y),blade_pass_hz=300*np.sqrt(s),frequency_valid=True) for x,y in [(-60,-25),(60,-25),(-60,25),(60,25)]]
    observation=dict(t=float(t),detected=True,center=z.tolist(),confidence=.95,geometry_quality=.95,center_sigma_px=1.,
        center_covariance_px2=[[1,0],[0,1]],normal_xy=n[:2].tolist(),frequency_quality=.95,frequency_squared_relative=s)
    sensing=dict(rotors=rotors,four_visible=True,four_frequencies=True,thrust_ratio=s,frequency_quality=.95,differential_proxy=0.)
    outputs.append(bank.step(observation,sensing))
rows=[]
for horizon in (.1,.2,.4):
    lag=round(horizon/dt)
    for name in ('cv','ca','paper_nominal','paper_adaptive','paper_fused'):
        errors=[]
        for i in range(80,len(times)-lag):
            p=extrapolate(outputs[i]['states'][name],horizon);errors.append(np.linalg.norm(p-position[i+lag]))
        rows.append(dict(method=name,horizon_s=horizon,n=len(errors),mean_error_px=float(np.mean(errors)),median_error_px=float(np.median(errors))))
result=dict(seed=16,dt_s=dt,duration_s=8,metrics=rows,
    protocol='All anchors after 0.8s; deterministic piecewise tilt and RPM; 0.6px Gaussian current-center noise; no future command available.',
    interpretation='Observation-level integration test. Rotor ellipse/RPM observations supplied, not extracted from rendered RAW; nominal pixel gravity matches truth.',
    limitations='Does not validate detection, real attitude/RPM, physical scale, wind, held-out performance or camera photometry.')
(out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
np.savez(out/'truth.npz',times=times,position=position,velocity=velocity,acceleration=truth_forces)
with (out/'forecasts.jsonl').open('w',encoding='utf-8') as f:
    for row in outputs:f.write(json.dumps(row)+'\n')
print(json.dumps(rows),flush=True)
