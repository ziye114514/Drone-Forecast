"""CPU-only focused acceleration hypothesis; no detector thresholds fitted."""
from pathlib import Path
import os,sys,json,hashlib,argparse,time,numpy as np
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from accelerating_ring_phase import accelerating_ring_phase
from angular_phase_candidates import ring_phase_candidates
OUT=ROOT.parent/'output/chirp231_20261108'
DTYPE=np.dtype({'names':['x','y','p','t'],'formats':['<u2','<u2','<i2','<i8'],'offsets':[0,2,4,8],'itemsize':16})
ELLIPSE=dict(cx=300.,cy=250.,major=80.,minor=50.,angle=0.)

def fixture(blades,acceleration):
    e=np.zeros(3000*blades,DTYPE);e['t']=np.repeat(np.arange(3000)*33332//2999,blades)
    tt=e['t']/1e6;theta=2*np.pi*(250*tt+.5*acceleration*tt**2)+np.tile(np.arange(blades)*2*np.pi/blades,3000)
    e['x']=np.rint(300+40*np.cos(theta));e['y']=np.rint(250+25*np.sin(theta));e['p']=np.arange(len(e))%2
    return e

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('group',choices=['controls','real']);args=p.parse_args()
    OUT.mkdir(exist_ok=True);target=OUT/f'{args.group}.json'
    if target.exists():raise FileExistsError(target)
    files=['angular_phase_candidates.py','accelerating_ring_phase.py','tools/probe_chirp231.py']
    sha={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files}
    protocol=OUT/'protocol.json'
    if protocol.exists():assert json.loads(protocol.read_text('utf-8'))['source_sha256']==sha
    else:
        protocol.write_text(json.dumps(dict(source_sha256=sha,GPU_used=False,NN_used=False,
            future_position_labels_used=False,changed='Discovery-only constant angular acceleration, unchanged confirmation gates',
            full_RAW_FPS_measured=False,prediction_gain_claimed=False,
            real_geometry_frozen_in='225/226 snapshots, early-half only; inspect all saved hypotheses, no late-score selection'),indent=2),'utf-8')
        for n in files:
            dst=OUT/'source_snapshot'/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/n).read_bytes())
    records=[]
    if args.group=='controls':
        for blades in [2,3]:
            for acceleration in [0.,2500.,-2500.]:
                e=fixture(blades,acceleration);result=accelerating_ring_phase(e,ELLIPSE,.033333)
                expected=blades*(250+acceleration*.033333);expected_rate=blades*acceleration
                assert result['qualified'],(blades,acceleration,result)
                # The selected harmonic may be an equivalent multiple. Compare
                # spatial-order-normalized rate; no blade-count GT is supplied
                # to the estimator, only to this synthetic verification.
                order=result['angular_harmonic'];rotation=result['phase_hz']/order
                slope=result['phase_frequency_rate_hz_s']/order
                assert abs(rotation-(250+acceleration*.033333))<1.
                assert abs(slope-acceleration)<15.
                changed=e.copy();late=changed['t']>=16666;rng=np.random.default_rng(231)
                xy=np.c_[changed['x'][late],changed['y'][late]];rng.shuffle(xy)
                changed['x'][late]=xy[:,0];changed['y'][late]=xy[:,1]
                other=accelerating_ring_phase(changed,ELLIPSE,.033333)
                for key in ['angular_harmonic','discovery_frequency_at_reference_hz','phase_frequency_rate_hz_s']:
                    assert result[key]==other[key],(key,result[key],other[key])
                records.append(dict(kind='rotating',blades=blades,rotation_acceleration_hz_s=acceleration,result=result,
                    heldout_permutation_changes_discovery_parameters=False))
        for seed in range(8):
            rng=np.random.default_rng(seed);e=np.zeros(3000,DTYPE);theta=rng.uniform(0,2*np.pi,len(e))
            e['x']=np.rint(300+40*np.cos(theta));e['y']=np.rint(250+25*np.sin(theta))
            e['t']=rng.integers(0,33333,len(e));e['p']=rng.integers(0,2,len(e));e.sort(order='t')
            result=accelerating_ring_phase(e,ELLIPSE,.033333);assert not result['qualified']
            records.append(dict(kind='nonperiodic',seed=seed,result=result))
        # Existing constant-rate controls retain their original values after
        # extracting the shared sample preparation helper.
        old=json.loads((ROOT.parent/'output/geometry_first226_20261108/results.json').read_text('utf-8'))
        for record in old['controls']:
            actual=ring_phase_candidates(fixture(record['blades'],0.),ELLIPSE,.033333)
            assert actual['qualified'] and actual['angular_harmonic']==record['candidate']['angular_harmonic']
        print('Six rotating controls, eight nonperiodic negatives and causal suffix checks passed',flush=True)
    else:
        from pure_visibility170 import window
        for parent in ['geometry_first225_20261108','geometry_first226_20261108']:
            source=json.loads((ROOT.parent/f'output/{parent}/results.json').read_text('utf-8'))
            for item in source['records']:
                n=item['recording'];t=item['t'];cacheparent='motion181_20261107' if n=='15' else 'phase185_20261107'
                e=window(ROOT.parent/f'output/{cacheparent}/cache_recording{n}',t)
                for arm,a in item['arms'].items():
                    velocity=(a.get('motion_proposal') or {}).get('velocity_px_s') if arm=='early_motion' else None
                    for j,h in enumerate(a.get('hypotheses',[])):
                        tick=time.perf_counter();result=accelerating_ring_phase(e,h['ellipse'],t,velocity)
                        records.append(dict(recording=n,t=t,arm=arm,hypothesis=j,ellipse=h['ellipse'],result=result,
                            CPU_ms=1000*(time.perf_counter()-tick)))
                (OUT/'real_partial.json').write_text(json.dumps(records,indent=2),'utf-8')
                subset=[r for r in records if r['recording']==n and r['t']==t]
                print(n,t,'constant',sum(r['result']['constant_rate_control']['qualified'] for r in subset),
                    'accelerating',sum(r['result']['qualified'] for r in subset),flush=True)
    assert 'torch' not in sys.modules,'This probe must not initialize a GPU framework'
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in sha.items())
    target.write_text(json.dumps(dict(records=records,GPU_used=False,source_changed=False,
        trajectory_FDE_measured=False,mechanical_RPM_GT_in_real_recordings=False),indent=2),'utf-8')
