"""Fixed positives and hard negatives before looking at real-frame outcomes."""
from pathlib import Path
import sys,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from predictive_harmonics import predictive_harmonics
from harmonic_pixels import harmonic_pixels


def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(270101);dtype=np.dtype([('x','u2'),('y','u2'),('p','i2'),('t','i8')]);packets=[];metadata=[]
    for kind in ['random','single_edge','periodic','common_flicker']:
        for count in [12,24,48,96]:
            for repeat in range(16):
                k=len(metadata);e=np.zeros(count,dtype);e['x']=k%32;e['y']=k//32
                if kind=='random':e['t']=rng.integers(1,33332,count);e['p']=rng.integers(0,2,count)
                elif kind=='single_edge':
                    e['t']=np.clip(rng.normal(17000,400,count),1,33332).astype(int);e['p']=e['t']>=17000
                else:
                    f=300. if kind=='common_flicker' else [180.,300.,450.][repeat%3]
                    phase=0. if kind=='common_flicker' else rng.uniform(0,1/f)
                    crossings=[]
                    for cycle in range(-1,17):
                        for p,offset in [(1,0.),(0,.27/f)]:
                            t=phase+cycle/f+offset
                            if 0<t<.033333:crossings.append((t,p))
                    chosen=np.linspace(0,len(crossings)-1,count).astype(int)
                    e['t']=[np.clip(round(crossings[i][0]*1e6+rng.normal(0,20)),1,33332) for i in chosen]
                    e['p']=[crossings[i][1] for i in chosen]
                metadata.append(dict(kind=kind,count=count,pixel=k));packets.append(e)
    e=np.concatenate(packets);e=e[np.argsort(e['t'])];start=time.perf_counter()
    result=predictive_harmonics(e,.033333,width=32,height=8)
    accepted=result['mask'].ravel()>0;scores={}
    for kind in ['random','single_edge','periodic','common_flicker']:
        scores[kind]={}
        for count in [12,24,48,96]:
            ids=[m['pixel'] for m in metadata if m['kind']==kind and m['count']==count]
            scores[kind][str(count)]=dict(accepted=int(accepted[ids].sum()),total=len(ids))
    print(json.dumps(scores),flush=True)
    assert sum(v['accepted'] for v in scores['random'].values())<=2
    assert sum(v['accepted'] for v in scores['single_edge'].values())==0
    assert scores['periodic']['96']['accepted']>=12
    assert scores['common_flicker']['96']['accepted']>=12 # periodicity is not rotor identity
    legacy=harmonic_pixels(e,.033333,width=32,height=8)
    legacy_accepted=legacy['mask'].ravel()>0
    same_input_legacy={kind:int(sum(legacy_accepted[m['pixel']] for m in metadata if m['kind']==kind))
        for kind in ['random','single_edge','periodic','common_flicker']}
    report=dict(synthetic_only=True,fixed_seed=270101,scores=scores,parameters=result['parameters'],
        same_input_old_reference_accepted=same_input_legacy,
        processing_ms=(time.perf_counter()-start)*1000,
        common_flicker_expected_to_pass=True,qualified_periodicity_is_not_a_rotor_class_or_true_RPM=True)
    (out/'controls.json').write_text(json.dumps(report,indent=2),'utf-8')
    np.savez_compressed(out/'events.npz',events=e)
    print(json.dumps(dict(same_input_old_reference_accepted=same_input_legacy)),flush=True)


if __name__=='__main__':main(sys.argv[1])
