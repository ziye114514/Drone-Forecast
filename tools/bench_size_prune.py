"""Isolate size-prune savings from previously optimized geometry; no labels."""
from pathlib import Path
import sys,json,time,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'src'),str(ROOT/'tools')]
import fast_geometry_compute as compute
from fast_structure_front import FastStructure
from check_final_compute import compare


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(2);result=[]
    source=ROOT.parent/'output/predictive_harmonics_20261027/native_pilot'
    uncertainty=compute._uncertainty
    dtype=np.dtype([('x','u2'),('y','u2'),('t','i8'),('frequency','f4')])
    for name,folder in [('clear','raw_release_clear'),('partial','raw_release_clear'),('side','raw_release_side')]:
        native=np.load(source/name/'native_window.npz');t=float(native['t']);events=native['events']
        rows=[json.loads(s) for s in (ROOT.parent/'output/three_continuation_20261026'/folder/'predictions.jsonl').read_text('utf-8').splitlines()]
        row=min(rows,key=lambda r:abs(r['t']-t));timing={k:[] for k in ['control','size_prune']};calls={k:0 for k in timing};values={}
        for repeat in range(3):
            order=['control','size_prune'] if repeat%2==0 else ['size_prune','control']
            for label in order:
                model=FastStructure(optimized_compute=True)
                if label=='control':
                    def no_size_gate(contour,**options):
                        options['minimum_major']=0.
                        return compute.fit_rotor_geometry_fast(contour,**options)
                    model.fit_current=no_size_gate
                def count_uncertainty(*args):
                    calls[label]+=1
                    return uncertainty(*args)
                compute._uncertainty=count_uncertainty
                try:
                    start=time.perf_counter();values[label]=model.step(events,t,row['body'],np.zeros(0,dtype),row['coefficient']['velocity'],True)
                    timing[label].append((time.perf_counter()-start)*1000)
                finally:compute._uncertainty=uncertainty
        delta=compare(values['control'],values['size_prune'])
        result.append(dict(name=name,all_outputs_max_difference=delta,
            uncertainty_calls_over3_repeats=calls,median_component_ms={k:float(np.median(v)) for k,v in timing.items()}))
    summary=dict(cases=result,component_only_not_full_pipeline_FPS=True,
        same_native_events_and_size_threshold=True,annotation_read=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),'utf-8');print(json.dumps(summary))


if __name__=='__main__':run(sys.argv[1])
