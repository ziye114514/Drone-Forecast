"""Small native-event probe; no detector/forecaster modification or labels."""
from pathlib import Path
from collections import deque
import json,sys,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from live_decode import native_batches
from side_moment_period import observe


def checks():
    dtype=[('x','u2'),('y','u2'),('p','i2'),('t','i8')]
    ticks=np.arange(640);time=(ticks+.5)*.00025
    # Matched locations repeated equally in a bin. Motion and multiplicity
    # cannot create a frequency in the normalized variance of a fixed shape.
    def data(rotating,translation=False,multiplicity=False):
        rows=[]
        for i,t in enumerate(time):
            radius=40*np.cos(2*np.pi*200*t) if rotating else 35.
            base=400+200*t if translation else 400.
            points=np.array([base-radius,base+radius])
            repeats=int(8+6*np.sin(2*np.pi*350*t)) if multiplicity else 8
            for x in np.tile(points,max(4,repeats)):rows.append((round(x),300,1,round(t*1e6)))
        return np.array(rows,dtype=dtype)
    rotating,_=observe(data(True,True,True),.16,[400,300],[1,0],[200,0])
    static,_=observe(data(False,False,True),.16,[400,300],[1,0],[0,0])
    assert rotating['qualified'] and abs(rotating['full']['frequency_hz']-400)<4.,rotating
    assert not static['qualified'],static
    return dict(controlled_rotating_line_component_400Hz=rotating,common_multiplicity_no_false_spatial_period=not static['qualified'],real_RAW_RPM_proved=False)


def main():
    out=ROOT.parent/'output/fred_transfer_20261029/side_spatial_moment';out.mkdir(exist_ok=False)
    results={'checks':checks(),'samples':[],'default_enabled':False,'annotations_used':False,'RAW_FPS_measured':False}
    for name,raw,start,targets,path in [
        ('side','E:/dataset/train/6/media/event.raw',110.,[110.5,111.5,113.,116.],ROOT.parent/'output/fred_transfer_20261029/motion_observability_v3/side_motion.jsonl'),
        ('clear','E:/dataset/train/6/media/event.raw',26.,[26.5,27.5],ROOT.parent/'output/fred_transfer_20261029/RAW_clear_integrated2s/predictions.jsonl')]:
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        chosen=[min(rows,key=lambda r:abs(r['t']-t)) for t in targets];cutoffs={round(r['t']*1e6):r for r in chosen}
        begin=(round(start*1e6)//33333)*33333/1e6;queue=deque()
        for t,e in native_batches(Path(raw),begin,max(targets)+.1-begin,False,33333,reader_backend='rolling'):
            queue.append(e.copy())
            while queue and (not len(queue[0]) or queue[0]['t'][-1]<(t-.17)*1e6):queue.popleft()
            row=cutoffs.get(round(t*1e6))
            if row is None:continue
            if not row['body'].get('detected') or row['structure'].get('axis_xy') is None:
                results['samples'].append(dict(clip=name,t=t,qualified=False,reason='no_current_body_or_plane'));continue
            all_events=np.concatenate(queue);box=np.asarray(row['body']['bbox']);c=np.asarray(row['body']['center']);v=row['coefficient']['velocity']
            normal=np.asarray(row['structure']['axis_xy']);axis=np.array([-normal[1],normal[0]])
            def sample(box):
                keep=(all_events['x']>=box[0])&(all_events['x']<box[2])&(all_events['y']>=box[1])&(all_events['y']<box[3])
                return observe(all_events[keep],t,c,axis,v)
            value,series=sample(box)
            shift=260 if box[2]+260<1280 else -260
            background,unused=sample(box+np.array([shift,0,shift,0]))
            result=dict(clip=name,t=t,observation=value,background_control=background,native_current_period=row['period'],series=series)
            results['samples'].append(result)
            print(json.dumps({k:v for k,v in result.items() if k!='series'}),flush=True)
    (out/'summary.json').write_text(json.dumps(results,indent=2,allow_nan=False),'utf-8')


if __name__=='__main__':main()
