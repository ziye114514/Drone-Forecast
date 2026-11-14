"""Known-speed synthetic check followed by cached raw-window diagnostics."""
from pathlib import Path
import json
import os
import sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rotor_icp_speed import estimate,PARAMS


def synthetic(rpm):
    rng=np.random.default_rng(220902205)
    t=np.sort(rng.uniform(0,33333,22000))
    r=rng.uniform(.15,1.,len(t));theta=t*1e-6*rpm*2*np.pi/60+rng.integers(0,2,len(t))*np.pi
    angle=np.deg2rad(20);c,s=np.cos(angle),np.sin(angle)
    xy=np.column_stack([40*r*np.cos(theta),25*r*np.sin(theta)])@np.array([[c,s],[-s,c]])
    xy+=[500,400]
    e=np.zeros(len(t),dtype=[('x','u2'),('y','u2'),('p','i2'),('t','i8')])
    e['x']=np.rint(xy[:,0]);e['y']=np.rint(xy[:,1]);e['t']=t;e['p']=rng.integers(0,2,len(t))
    return e,dict(cx=500.,cy=400.,major=80.,minor=50.,angle=20.,ratio=.625)


def main(output):
    out=Path(output);results=dict(parameters=PARAMS,synthetic=[],real={},live_input_changed=False)
    for rpm in [3000,6000,9000]:
        e,r=synthetic(rpm);d=estimate(e,r)
        d['truth_RPM']=rpm;d['relative_error']=abs(d['mechanical_rpm_estimate']-rpm)/rpm if d.get('mechanical_rpm_estimate') else None
        results['synthetic'].append(d)
    for name in ['clear','partial','side']:
        packet=np.load(out/'rpm'/f'{name}.npz');reference=json.loads((out/'rpm'/f'{name}_reference.json').read_text('utf-8'))
        rotors=reference['geometry'].get('rotors',[])
        # Geometry is an existing algorithm hypothesis, not labeled rotor truth.
        estimates=[estimate(packet['events'],r,reference['coefficient']['velocity'],float(packet['t'])) for r in rotors]
        results['real'][name]=dict(rotors=estimates,current_mode=reference['structure']['mode'],
            reason='current_ellipses_available' if rotors else 'no_rectifiable_rotor_ellipse',
            existing_frequency_proxies=reference['rates'],comparison_is_not_accuracy_truth=True)
    (out/'rpm/icp_results.json').write_text(json.dumps(results,indent=2,allow_nan=False),'utf-8')
    print(json.dumps(results,ensure_ascii=False),flush=True)


if __name__=='__main__':main(sys.argv[1])
