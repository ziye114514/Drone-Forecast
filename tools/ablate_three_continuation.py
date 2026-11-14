"""One paired RAW/current-body ablation of guidance freshness, not a full FPS test."""
from pathlib import Path
import sys,json,time,hashlib
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fast_structure_front import FastStructure
from live_decode import native_batches
from research.stage79.native79 import engine,copy_frequency


def chains(stamps,max_gap=.120001):
    groups=[]
    for t in stamps:
        if not groups or t-groups[-1][-1]>max_gap:groups.append([t])
        else:groups[-1].append(t)
    return max((dict(events=len(g),span_s=g[-1]-g[0],start=g[0],end=g[-1]) for g in groups),
        key=lambda z:(z['span_s'],z['events']),default=dict(events=0,span_s=0.))


def summarize(out,rows):
    arms={}
    for name in rows[0]['arms']:
        data=[r['arms'][name] for r in rows];times=[r['t'] for r in rows if r['arms'][name]['structure']['mode']=='three']
        audits=[dict(t=r['t'],**r['arms'][name]['geometry'].get('shared_arc_recovery',{})) for r in rows]
        arms[name]=dict(modes=dict(Counter(z['structure']['mode'] for z in data)),
            partial_arc_updates=sum(z['geometry'].get('partial_arc_information',{}).get('updated',False) for z in data),
            three_chain_with_gaps_at_most_120ms=chains(times),
            recovered_arcs=sum(z.get('accepted',0) for z in audits),
            recoveries=[z for z in audits if z.get('accepted')],
            guidance_rejections=dict(Counter(z.get('guidance_reference',{}).get('reason','not_checked') for z in audits)),
            geometry_mean_ms=float(np.mean([z['component_ms'] for z in data])),
            current_centers=sum(z['structure']['current_geometric_center'] is not None for z in data))
    result=dict(arms=arms,frames=len(rows),same_RAW_events_and_saved_current_body=True,
        extra_three_chain_may_have_rank4_frames_between=True,source_fps=30,
        no_NN_or_future_prediction_or_label_scoring=True,not_end_to_end_speed_benchmark=True)
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(result,ensure_ascii=False))


def run(source,output,normal=False):
    source=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((source/'protocol.json').read_text('utf-8'))
    data=[json.loads(s) for s in (source/'predictions.jsonl').read_text('utf-8').splitlines()]
    snapshot=out/'source_snapshot';snapshot.mkdir();hashes={}
    for f in ['continuous_five_circle.py','shared_arc_recovery.py','partial_arc_information.py','fast_structure_front.py','tools/ablate_three_continuation.py']:
        content=(ROOT/f).read_bytes();p=snapshot/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content);hashes[f]=hashlib.sha256(content).hexdigest()
    (out/'protocol.json').write_text(json.dumps(dict(source=str(source),sources=hashes,labels_read=False),indent=2),'utf-8')
    names=['contrast_area','normal_constraints'] if normal else ['full_only','contrast_area']
    models={name:FastStructure(continuous=True,guide_full_only=name=='full_only',
        bipolar_guide_area=False,contrast_guide_area=name=='contrast_area',
        current_shape_feedback=name!='full_only',partial_arc_information=name=='normal_constraints') for name in names}
    if normal:
        for model in models.values():model.contrast_guide_area=True
    algo=engine(1280,720,'strict7');q=copy_frequency(algo.get_empty_output_buffer());rows=[]
    for row,(t,events) in zip(data,native_batches(protocol['raw'],protocol['actual_iterator_start'],protocol['actual_duration'],False,protocol['native_step_us'])):
        assert abs(t-row['t'])<1e-8
        result={}
        order=list(models) if len(rows)%2==0 else list(models)[::-1]
        for name in order:
            start=time.perf_counter();s,_,d,_=models[name].step(events,t,row['body'],q,[0.,0.])
            result[name]=dict(structure=s,geometry=d,component_ms=(time.perf_counter()-start)*1000)
        rows.append(dict(t=t,body=row['body'],arms=result))
    (out/'observations.jsonl').write_text('\n'.join(json.dumps(r,default=lambda z:z.tolist() if hasattr(z,'tolist') else z) for r in rows),'utf-8')
    assert len(rows)==len(data);summarize(out,rows)


if __name__=='__main__':
    if sys.argv[1]=='--summarize':
        p=Path(sys.argv[2]);summarize(p,[json.loads(s) for s in (p/'observations.jsonl').read_text('utf-8').splitlines()])
    else:run(sys.argv[1],sys.argv[2],normal='--normal' in sys.argv[3:])
