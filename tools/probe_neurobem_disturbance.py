"""Bounded public-telemetry module test; does NOT test event perception.

The lexically first public CSV was selected before scores. The initial0.8s
calibrates only effective thrust acceleration gain from reference motor/IMU
data; subsequent inference uses current noisy position, attitude and motors.
No future commands/states enter forecasts. This is an input-oracle subsystem
test, not evidence that target-only event footage recovers these inputs.
"""
from pathlib import Path
import json,sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from lumped_disturbance import LumpedDisturbance,forecasts


def run(folder,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    x=np.loadtxt(Path(folder)/'sample.csv',delimiter=',',skiprows=1)
    times=x[:,0];times-=times[0];R=Rotation.from_quat(x[:,7:11]).as_matrix()
    position=x[:,17:20][:,[0,2]]
    speed2=np.sum(x[:,20:24]**2,axis=1);cal=times<=.8
    reference_speed2=float(np.mean(speed2[cal]));feature=speed2/reference_speed2
    gain=float(np.dot(feature[cal],x[cal,13])/np.dot(feature[cal],feature[cal]))
    inputs=gain*feature[:,None]*R[:,:,[2]].reshape(-1,3)[:,[0,2]]
    models={n:LumpedDisturbance(.005,.5) for n in ['no_thrust_total_acceleration','thrust_plus_lumped_disturbance']}
    rng=np.random.default_rng(42);records=[];last_published=-np.inf
    #40ms samples/200ms forecast publication, matching the intended low rate.
    indexes=np.arange(0,len(x),16)
    for i in indexes:
        t=float(times[i]);z=position[i]+rng.normal(0,.005,2);states={}
        for name,m in models.items():
            u=inputs[i] if name=='thrust_plus_lumped_disturbance' else None
            c,a=m.step(t,z,u)
            states[name]=dict(coefficient=c,audit=a,predictions=forecasts(c))
        if t>=.8 and t-last_published>=.2-1e-8 and t+.4<=times[-1]:
            last_published=t
            # Only scoring below reads future reference positions.
            truth=np.array([np.interp(t+.4,times,position[:,j]) for j in range(2)])
            errors={n:float(np.linalg.norm(np.asarray(v['predictions']['0.4'])-truth)) for n,v in states.items()}
            c=states['no_thrust_total_acceleration']['coefficient']
            cv=np.asarray(c['position'])+.4*np.asarray(c['velocity'])
            errors['same_state_constant_velocity']=float(np.linalg.norm(cv-truth))
            records.append(dict(t=t,target_t=t+.4,states=states,truth=truth.tolist(),errors_m=errors))
    names=list(records[0]['errors_m'])
    scores={n:dict(n=len(records),mean_m=float(np.mean([r['errors_m'][n] for r in records])),
        p95_m=float(np.percentile([r['errors_m'][n] for r in records],95))) for n in names}
    result=dict(source_provenance=json.loads((Path(folder)/'provenance.json').read_text('utf-8')),
        rows=len(x),duration_s=float(times[-1]),calibration_prefix_s=.8,effective_gain_m_s2=gain,
        individual_mass_or_thrust_coefficient_identified=False,scores=scores,
        event_images_used=False,independent_external_force_truth=False,parameters_swept=False,
        assumptions=['R maps body to world','acc z is body-frame specific acceleration as provided',
            'reference attitude/motor speed supplied for module test','5mm synthetic position noise with fixed seed42',
            '0.8s gain calibration uses reference IMU; not available from event-only target observation'],
        scope='One short takeoff fragment of real public telemetry; no generalization/SOTA or event-only force reconstruction claim.')
    (out/'comparison.json').write_text(json.dumps(result,indent=2),'utf-8')
    (out/'predictions.json').write_text(json.dumps(records,indent=2),'utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(position[:,0],position[:,1],label='reference')
    axes[0].set(xlabel='world x [m]',ylabel='world z [m]',title='Public telemetry: selected first CSV')
    for n in names:axes[1].plot([r['t'] for r in records],[r['errors_m'][n] for r in records],'.-',label=n)
    axes[1].set(xlabel='forecast source time [s]',ylabel='0.4s error [m]',title='Same origin, subsystem inputs')
    for a in axes:a.grid(alpha=.25);a.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=130);plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k!='source_provenance'}),flush=True)


if __name__=='__main__':run(sys.argv[1],sys.argv[2])
