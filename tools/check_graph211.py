"""Corrected regional statistic: fixed graph parity, negative and real inputs."""
from pathlib import Path
import os,sys,json,time,numpy as np
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from cuda_regional_graph import RegionalGraph,DTYPE
import regional_timing_period as reference
from native_timing_histogram import NativeTimingHistogram
def window(cache,t):
    m=json.loads((cache/'manifest.json').read_text('utf-8'));end=round(t*1e6);begin=end-33333;parts=[]
    for item in m['frames']:
        e=np.load(cache/item['file']);e=e[(e['t']>=begin)&(e['t']<end)]
        if len(e):parts.append(e)
    return np.concatenate(parts)
OUT=ROOT.parent/'output/graph211_20261108'

if __name__=='__main__':
    OUT.mkdir(exist_ok=True);result=[]
    small=RegionalGraph(64,64,100000)
    for seed in range(9):
        rng=np.random.default_rng(seed);e=np.zeros(20000,DTYPE);pixel=rng.integers(0,64,len(e))
        e['x']=16+pixel%8;e['y']=16+pixel//8;e['p']=1;e['t']=rng.integers(0,33333,len(e));e.sort(order='t')
        if seed==8:
            e=np.zeros(512,DTYPE);e['t']=np.repeat(np.arange(8)*1e6/240+300,64)+rng.normal(0,20,512)
            pixel=np.tile(np.arange(64),8);e['x']=16+pixel%8;e['y']=16+pixel//8;e['p']=1;e.sort(order='t')
        _,d,g=reference.measure(e,.033333,width=64,height=64);arrays,s,diag=small.measure(e,.033333)
        expected={(int(v['cell_xy'][0]/16),int(v['cell_xy'][1]/16),v['polarity']) for v in g}
        got={(int(i//2%4),int(i//2//4),int(i%2)) for i in np.flatnonzero(s[:,4])}
        assert expected==got,(seed,expected,got)
        assert int(arrays[:2].sum())==len(e)
        result.append(dict(seed=seed,positive=seed==8,**diag))
    assert not sum(r['qualified_groups'] for r in result[:8]) and result[-1]['qualified_groups']>0
    del small
    import gc,torch;gc.collect();torch.cuda.empty_cache()
    graph=RegionalGraph();engine=NativeTimingHistogram(ROOT.parent/'output/timing208_20261108/timing_histogram.dll')
    reference.HISTOGRAM_ENGINE=engine;real=[]
    for n,parent,t in [(15,'motion181_20261107',13.5),(12,'phase185_20261107',12.),(18,'phase185_20261107',13.5)]:
        e=window(ROOT.parent/f'output/{parent}/cache_recording{n}',t)
        _,d,groups=reference.measure(e,t);timings=[]
        for _ in range(4):
            arrays,s,diag=graph.measure(e,t);timings.append(diag['graph_ms'])
        expected={(int(g['cell_xy'][0]//16),int(g['cell_xy'][1]//16),g['polarity']) for g in groups}
        got={(int(i//2%80),int(i//2//80),int(i%2)) for i in np.flatnonzero(s[:,4])}
        assert expected==got,(n,expected,got)
        e=e[(e['t']>=round(t*1e6)-33333)&(e['t']<round(t*1e6))]
        assert arrays[:2].sum()==len(e)
        for g in groups:
            x,y=g['cell_xy'];i=((y//16)*80+x//16)*2+g['polarity']
            assert abs(s[i,0]-g['score'])<.01
        real.append(dict(recording=n,t=t,groups=groups,graph_ms=timings,**{k:v for k,v in diag.items() if k!='graph_ms'}))
        print(n,diag,flush=True)
    engine.close();reference.HISTOGRAM_ENGINE=None
    (OUT/'checks.json').write_text(json.dumps(dict(negatives_and_positive=result,real_frames=real,regional_groups_equal=True,
        graph_fps_is_not_RAW_FPS=True),indent=2),'utf-8')
    print('regional graph parity passed',flush=True)
