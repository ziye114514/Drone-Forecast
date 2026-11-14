"""One fixed negative control, not a threshold search or a paper refutation."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from harmonic_pixels import harmonic_pixels


def main():
    rng=np.random.default_rng(260308386);n=256;events_per_pixel=40
    dtype=np.dtype([('x','u2'),('y','u2'),('p','i2'),('t','i8')])
    e=np.zeros(n*events_per_pixel,dtype=dtype)
    for i in range(n):
        q=e[i*events_per_pixel:(i+1)*events_per_pixel];q['x']=i%32;q['y']=i//32
        q['p']=rng.integers(0,2,events_per_pixel);q['t']=rng.integers(1,33332,events_per_pixel)
    e=e[np.argsort(e['t'])];r=harmonic_pixels(e,.033333,width=32,height=8)
    calibrated=float(np.quantile(r['comb_score'],.99))
    result=dict(synthetic_only=True,nonperiodic_independent_timestamps=True,pixels=n,events_per_pixel=events_per_pixel,
        accepted_null_pixels=r['retained_pixels'],accepted_null_fraction=r['retained_pixels']/n,
        empirical_null_comb_quantile99=calibrated,
        empirical_null_gate_is_not_a_calibrated_real_world_false_alarm_probability=True,
        own_incomplete_reference_not_author_code=True,
        interpretation='Published scalar thresholds plus our declared K/base-band choices are not selective enough here; cannot claim frequency discrimination merely from an attractive mask.')
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
    (out/'null_control.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))
    np.savez_compressed(out/'null_evidence.npz',score=r['comb_score'],flatness=r['spectral_flatness'])


if __name__=='__main__':main()
