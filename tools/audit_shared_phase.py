"""Exact output audit, current-pixel corroboration and a small primitive timing pair."""
from pathlib import Path
import json,time,sys
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/shared_angular_phase_20261027'


def lines(p):return list(map(json.loads,p.read_text('utf-8').splitlines()))
def old(a):
    order=np.argsort(a,axis=0)
    return order[0],np.take_along_axis(a,order[1:2],axis=0)[0]-a.min(0)
def new(a):
    lo=np.minimum(a[0],a[1]);hi=np.maximum(a[0],a[1]);L=np.minimum(a[2],a[3]);H=np.maximum(a[2],a[3])
    return a.argmin(0),np.minimum(np.minimum(hi,H),np.maximum(lo,L))-np.minimum(lo,L)


def main():
    a=lines(OUT/'raw_clear_threads1/observations.jsonl');b=lines(OUT/'raw_clear_pairmin/observations.jsonl')
    for x,y in zip(a,b):
        x.pop('processing_ms');y.pop('processing_ms');assert x==y
    rng=np.random.default_rng(20261027);matrix=rng.random((4,48000));matrix[:,:5]=.5
    expected=old(matrix);got=new(matrix)
    assert np.array_equal(expected[1],got[1])
    assert np.array_equal(expected[0][expected[1]>=.1],got[0][got[1]>=.1])
    timing={n:[] for n in ['sort','pair_minimum']}
    for k in range(15):
        for name,fn in ([('sort',old),('pair_minimum',new)] if k%2==0 else [('pair_minimum',new),('sort',old)]):
            start=time.perf_counter();fn(matrix);timing[name].append((time.perf_counter()-start)*1000)
    upstream={z['t']:z for z in lines(ROOT/'output/three_continuation_20261026/raw_release_clear/predictions.jsonl')}
    comparisons=[]
    for z in b:
        if not z['rotors']:continue
        r=upstream[z['t']];h=np.array(r['structure']['model_centers']);rotors=r['geometry'].get('rotors',[])
        if not rotors:continue
        centers=np.array([[q['cx'],q['cy']] for q in rotors]);cost=np.linalg.norm(h[:,None]-centers[None],axis=2)
        ii,jj=linear_sum_assignment(cost);mapping=dict(zip(ii.tolist(),jj.tolist()))
        for phase in z['rotors']:
            i=phase['model_id'];j=mapping.get(i)
            if not phase['qualified'] or j is None or j>=len(r['rates']):continue
            pixel=r['rates'][j]
            if not pixel.get('valid') or cost[i,j]>max(4.,.15*rotors[j]['major']):continue
            f=phase['frequency_hz'];p=pixel['frequency_hz'];ratio=f/p
            harmonic=min([.5,1.,2.],key=lambda q:abs(np.log(ratio/q)))
            comparisons.append(dict(t=z['t'],model_id=i,phase_hz=f,pixel_hz=p,ratio=ratio,
                nearest_harmonic=harmonic,within_ten_percent=bool(abs(np.log(ratio/harmonic))<=np.log(1.1))))
    fig,axes=plt.subplots(2,1,figsize=(10,5),sharex=True,layout='constrained')
    for i in range(4):
        t=[];f=[];c=[]
        for z in b:
            q=next((x for x in z['rotors'] if x['model_id']==i),None)
            t.append(z['t']);f.append(q['frequency_hz'] if q and q['qualified'] else np.nan)
            c.append(q.get('phase_concentration',np.nan) if q else np.nan)
        axes[0].plot(t,f,'.-',label=f'model slot {i}');axes[1].plot(t,c,'.-')
    axes[0].set(ylabel='Blade-phase frequency (Hz)',title='Current native events + frozen causal five-circle geometry');axes[0].legend(ncol=4,fontsize=8)
    axes[1].axhline(.5,color='gray',ls='--');axes[1].set(ylabel='Spatial-phase concentration',xlabel='Source cutoff (s)',ylim=(0,1))
    fig.savefig(OUT/'phase_tracks.png',dpi=150);plt.close(fig)
    result=dict(actual_source_ticks_exact=60,phase_outputs_exact_after_sort_removal=True,
        primitive_pair_calls_per_method=15,primitive_input_events=48000,
        primitive_median_ms={n:float(np.median(v)) for n,v in timing.items()},
        matched_current_pixel_comparisons=len(comparisons),
        matches_within_ten_percent=sum(q['within_ten_percent'] for q in comparisons),
        unit_ratio_matches=sum(q['within_ten_percent'] and q['nearest_harmonic']==1 for q in comparisons),
        mean_relative_difference=None if not comparisons else float(np.mean([abs(q['ratio']-1) for q in comparisons])),
        corroboration_is_not_independent_RPM_truth=True,comparisons=comparisons)
    (OUT/'audit.json').write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='comparisons'}))


if __name__=='__main__':main()
