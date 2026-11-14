"""Same RAW events/current saved body boxes, guided-arc switch only."""
from pathlib import Path
import sys,json,time
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fast_structure_front import FastStructure
from live_decode import native_batches
from research.stage79.native79 import engine,copy_frequency


def summarize(out,results):
    summary={}
    for guided in [False,True]:
        data=[r['arms'][str(guided)] for r in results]
        summary[str(guided)]=dict(modes=dict(Counter(r['structure']['mode'] for r in data)),
            recovered_current_arcs=sum(r['geometry'].get('shared_arc_recovery',{}).get('accepted',0) for r in data),
            geometry_ms_mean=float(np.mean([r['component_ms'] for r in data])),
            recovery_details=[dict(t=z['t'],audit=z['arms'][str(guided)]['geometry']['shared_arc_recovery']) for z in results if z['arms'][str(guided)]['geometry'].get('shared_arc_recovery',{}).get('accepted')])
    result=dict(arms=summary,frames=len(results),same_events_and_current_body_observations=True,
        RAW_redecoded=True,NN_not_rerun=True,periodicity_and_forecast_not_evaluated=True,
        not_end_to_end_fps_benchmark=True,no_labels_read=True)
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8');print(json.dumps(result,ensure_ascii=False))


def run(source,output):
    source=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((source/'protocol.json').read_text('utf-8'));rows=[json.loads(s) for s in (source/'predictions.jsonl').read_text('utf-8').splitlines()]
    models={False:FastStructure(continuous=True,shared_arc_recovery=False),True:FastStructure(continuous=True,shared_arc_recovery=True)}
    e=engine(1280,720,'strict7');q=copy_frequency(e.get_empty_output_buffer())
    results=[]
    for row,(t,events) in zip(rows,native_batches(protocol['raw'],protocol['actual_iterator_start'],protocol['actual_duration'],False,protocol['native_step_us'])):
        assert abs(t-row['t'])<1e-8
        arms={}
        for guided in ([False,True] if len(results)%2==0 else [True,False]):
            start=time.perf_counter();s,r,d,_=models[guided].step(events,t,row['body'],q,[0.,0.])
            arms[str(guided)]=dict(structure=s,geometry=d,component_ms=(time.perf_counter()-start)*1000)
        results.append(dict(t=t,body=row['body'],arms=arms))
    (out/'observations.jsonl').write_text('\n'.join(json.dumps(r,default=lambda v:v.tolist() if hasattr(v,'tolist') else v) for r in results),'utf-8')
    summarize(out,results)


if __name__=='__main__':
    if sys.argv[1]=='--summarize':
        out=Path(sys.argv[2]);summarize(out,[json.loads(z) for z in (out/'observations.jsonl').read_text('utf-8').splitlines()])
    else:run(sys.argv[1],sys.argv[2])
