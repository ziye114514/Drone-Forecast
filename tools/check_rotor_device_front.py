"""Small real-window equivalence and observer mechanism checks, not a sweep."""
from pathlib import Path
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'research/stage79')]
from rotor_cuda_worker import CudaRotorFrontWorker
from rotor_support_front import accumulate,density_mask
from rotor_only_front import native_reconstruction
from rotor_force_observer import RotorForceObserver


def main(output):
    out=Path(output);out.mkdir(exist_ok=True)
    base=ROOT.parent/'output/rotor_pipeline_20261101'
    python=Path('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe')
    results=[];engine=CudaRotorFrontWorker(python)
    try:
        for clip,indices in [('clear',[0,8,14,23]),('side',[0,8,14,23]),('empty',[0])]:
            for i in indices:
                with np.load(base/f'cache/{clip}/{i:04d}.npz') as z:
                    e=z['events'];t=float(z['t'])
                on,off,meta,dense,local,period=engine.prepare_rotor_support(e,t)
                reference_on,reference_off,_=accumulate(e,t)
                reference_density,_=density_mask(reference_on+reference_off,fast_spatial=True)
                expected_native,expected_diag=native_reconstruction(dict(on=reference_on,off=reference_off,period_mask=period['mask']))
                native,diag=meta['native_precomputed']
                record=dict(clip=clip,index=i,t=t,on_equal=np.array_equal(on,reference_on),
                    off_equal=np.array_equal(off,reference_off),density_difference=int(np.count_nonzero(dense!=reference_density)),
                    native_difference=int(np.count_nonzero(native!=expected_native)),
                    strict_pixels=meta['accepted_pixels'],service_ms=meta['cuda_service_ms'],
                    roundtrip_ms=meta['cuda_roundtrip_ms'],sparse_bytes=meta['sparse_transfer_bytes'])
                assert record['on_equal'] and record['off_equal']
                assert record['density_difference']==0,record
                assert record['native_difference']==0,record
                frozen=base/f'{clip}_bank/images/{i:04d}_evidence.npz'
                if frozen.exists():
                    with np.load(frozen) as z:
                        record['frozen_period_difference']=int(np.count_nonzero((period['mask']>0)!=z['period']))
                        record['frozen_native_difference']=int(np.count_nonzero(native!=z['native_arcs']))
                results.append(record);print(json.dumps(record),flush=True)
        # A future duplicate must have zero influence on the current GPU output.
        with np.load(base/'cache/clear/0008.npz') as z:e=z['events'];t=float(z['t'])
        a=engine.prepare_rotor_support(e,t)
        future=e.copy();future['t']+=100000
        b=engine.prepare_rotor_support(np.concatenate([e,future]),t)
        for key in ('mask','frequency_hz'):assert np.array_equal(a[5][key],b[5][key])
        assert np.array_equal(a[0],b[0])
    finally:engine.close()
    # Preserve the physical-input interpretation on acquisition/expiration,
    # while making no claim that missing observations are new detections.
    model=RotorForceObserver();warm=[]
    for i in range(12):
        t=i/30.;body=dict(detected=True,center=[200.+20*t,150.])
        c,valid,f,s=model.step(t,body,[0.,-1.],.8)
        warm.append(dict(valid=valid,active=f['active']))
        assert not f['active'] or valid
        assert np.linalg.eigvalsh(model.P).min()>-1e-7
    assert all(not r['active'] for r in warm[:4])
    c,valid,f,s=model.step(.6,dict(detected=False),None,0.)
    assert not valid and not f['active']
    summary=dict(real_window_checks=results,future_exclusion=True,warmup_does_not_activate_force=True,
        covariance_positive_semidefinite=True,stale_observation_invalid=True,
        note='New GPU frequency arithmetic is checked against frozen masks, not declared universally bitwise identical.')
    (out/'mechanism_checks.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')


if __name__=='__main__':main(sys.argv[1])
