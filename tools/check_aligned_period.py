"""Controlled moving periodic source, moving edge and shared illumination."""
from pathlib import Path
import os,sys,json
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research/stage79')]
import numpy as np
from native79 import synthetic,events
from aligned_period_recovery import AlignedPeriodRecovery
from body_period_stream import BodyPeriodStream

def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);result={}
    width,height=160,120;dt=1/30;velocity=np.array([40.,0.]);bbox=np.array([35.,25.,93.,71.]);duration=.4
    def run(kind,streaming=False,configuration='strict7'):
        model=(BodyPeriodStream if streaming else AlignedPeriodRecovery)(width,height,configuration);values=[]
        x,y=np.meshgrid(np.arange(43,86,3),np.arange(33,64,3));pixels=np.column_stack((x.ravel(),y.ravel()))
        if kind=='common':
            x,y=np.meshgrid(np.arange(3,width-3,3),np.arange(3,height-3,3));pixels=np.column_stack((x.ravel(),y.ravel()))
        times=[];xy=[];polarities=[]
        for j,p in enumerate(pixels):
            if kind=='edge':
                stamps=np.array([.06+.001*j]);pol=np.ones(len(stamps),int)
            else:stamps=np.arange(.002,duration,1/(2*287.));pol=np.arange(len(stamps))%2
            q=p+stamps[:,None]*velocity if kind!='common' else np.tile(p,(len(stamps),1))
            times.extend(stamps*1e6);xy.extend(np.rint(q).astype(int));polarities.extend(pol)
        e=events(np.asarray(times,np.int64),np.asarray(xy),np.asarray(polarities))
        for i in range(1,13):
            t=i*dt;b=bbox+np.tile(velocity*t,2);packet=e[(e['t']>=round((t-dt)*1e6))&(e['t']<round(t*1e6))]
            body=dict(detected=True,confidence=.8,bbox=b.tolist(),center=((b[:2]+b[2:])/2).tolist())
            tracking=dict(reason='current_image_translation',points=30,delta=(dt*velocity).tolist())
            p,d=model.step(t,packet,body,tracking,dict(qualified=False),True);values.append(d)
        return values
    tone=run('tone');edge=run('edge');common=run('common')
    assert any(r['applied'] for r in tone),tone
    assert not any(r['applied'] for r in edge),edge
    assert not any(r['applied'] for r in common),common
    stream=run('tone',True,'fast3');stream_edge=run('edge',True,'fast3');stream_common=run('common',True,'fast3')
    assert any(r['applied'] for r in stream),stream
    assert not any(r['applied'] for r in stream_edge),stream_edge
    assert not any(r['applied'] for r in stream_common),stream_common
    result=dict(moving_periodic_source_qualified_ticks=sum(r['applied'] for r in tone),
        moving_single_edge_qualified_ticks=0,global_illumination_qualified_ticks=0,
        timing_scope='Synthetic mechanism only; not measured real-data false positive rate or FPS',
        event_timestamps_polarities_unchanged=True,source_frequency_hz=287.)
    result['persistent_stream_periodic_ticks']=sum(r['applied'] for r in stream)
    result['persistent_stream_edge_or_global_flicker_ticks']=0
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))
if __name__=='__main__':main(sys.argv[1])
