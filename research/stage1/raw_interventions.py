"""Paired real RAW deletion experiment, same historical warm start for all arms."""
from pathlib import Path
import copy,hashlib,json,sys,time
import cv2,numpy as np
from mechanism import ComparisonBank,METHODS,ROOT
from run_study import OUT,save,verify
from event_drone_forecast.raw_io import open_raw,raw_chunks
from event_drone_forecast.rotor_sensing import PaperPerception,RotorSensing
from event_drone_forecast.paper_physics import ThrustAxisFilter

CONDITIONS=('clean','one_rotor_rectangle','left_pair_rectangle','drop_half')
def raw_sample(obs,sensing,axis,t,bank):
    current=[r for r in sensing['rotors'] if r['observed_now']]
    n=axis.step(t,current,sensing['differential_proxy'],True)
    if n is None:n=bank.last_n
    valid=[r['blade_pass_hz'] for r in sensing['rotors'] if r['frequency_valid']]
    return dict(t=t,position=np.asarray(obs.get('center',[0.,0.])),observed=obs['detected'],normal=n,
        frequencies=np.asarray(valid),four_visible=sensing['four_visible'],four_frequencies=sensing['four_frequencies'],
        quality=float(sensing['frequency_quality']*obs.get('geometry_quality',0)),
        position_confidence=obs.get('confidence',0),position_covariance=obs.get('center_covariance_px2'))

def ellipse_bounds(r,width,height):
    angle=np.deg2rad(r['angle']);a,b=r['major']/2,r['minor']/2
    half=np.sqrt([(a*np.cos(angle))**2+(b*np.sin(angle))**2,(a*np.sin(angle))**2+(b*np.cos(angle))**2])+3
    return np.r_[np.maximum(0,np.floor(np.array([r['cx'],r['cy']])-half)),
                 np.minimum([width,height],np.ceil(np.array([r['cx'],r['cy']])+half))].astype(int)

def delete_events(events,condition,rectangles,t):
    if not 24<t<=25+1e-8 or condition=='clean':return events
    if condition=='drop_half':
        rng=np.random.default_rng(42+round(t*1e6));return events[rng.random(len(events))>=.5]
    x0,y0,x1,y1=rectangles[condition]
    return events[~((events['x']>=x0)&(events['x']<x1)&(events['y']>=y0)&(events['y']<y1))]

def run():
    protocol=verify();destination=OUT/'raw_interventions'
    if (destination/'completion.json').exists():raise RuntimeError('Completed raw run already exists')
    destination.mkdir(exist_ok=True,parents=True)
    path=Path(__file__);save(destination/'supplemental_code_freeze.json',dict(before_raw_run=True,
        raw_script_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),controlled_protocol_sha256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
        snapshot=str(destination/'raw_interventions.py')))
    (destination/'raw_interventions.py').write_bytes(path.read_bytes())
    raw=protocol['raw_experiment']['raw'];reader=open_raw(raw);height,width=reader.get_size();bank=ComparisonBank(.01)
    frontend=PaperPerception(width,height,bank.cfg);sensing=RotorSensing(bank.cfg);axis=ThrustAxisFilter(bank.cfg)
    warm_records=[];begin=time.perf_counter();warm_count=0;lastprint=begin
    for t,events in raw_chunks(reader,0,24.,10000):
        obs,total=frontend.detect(events,t);sense=sensing.step(obs,frontend.history,t);sample=raw_sample(obs,sense,axis,t,bank)
        result=bank.step(sample)
        if t>=23 and result is not None:warm_records.append((t,obs,sense,result))
        warm_count+=1
        if time.perf_counter()-lastprint>12:
            print(json.dumps(dict(phase='shared_warmup',sensor_time=t)),flush=True);lastprint=time.perf_counter()
    visible=[r for r in sense['rotors'] if r['observed_now']]
    if len(visible)<4:raise RuntimeError('Predeclared24s warm start lacks four observed rotors; do not choose a new interval based on future errors.')
    visible.sort(key=lambda r:r['cx']);one=ellipse_bounds(visible[0],width,height)
    bounds=np.array([ellipse_bounds(r,width,height) for r in visible[:2]])
    pair=np.r_[bounds[:,:2].min(axis=0),bounds[:,2:].max(axis=0)]
    rectangles={'one_rotor_rectangle':one.tolist(),'left_pair_rectangle':pair.tolist()}
    save(destination/'intervention_manifest.json',dict(raw=raw,warm_start_s=24,stop_s=30,fault_start_s=24,fault_end_s=25,
        rectangles=rectangles,chosen_from_observations_at_or_before_s=24,uses_future_labels=False,conditions=list(CONDITIONS),
        shared_position_filter=True,model_methods=list(METHODS),
        interpretation='Fixed-coordinate deletion, not physical occluder simulation. All methods and conditions share clean historical warmup. Oracle-reset arm gets clean learner state only as a diagnostic.'))
    pipelines={name:copy.deepcopy((frontend,sensing,axis,bank)) for name in CONDITIONS}
    handles={};arrays={name:[] for name in CONDITIONS};counts={name:dict(input_events=0,deleted_events=0,detected=0,four_ellipses=0,physics_active=0) for name in CONDITIONS}
    for name in CONDITIONS:
        folder=destination/name;folder.mkdir(exist_ok=True)
        handles[name]=(folder/'observations.jsonl').open('w',encoding='utf-8')
    reset_done=False;frames=0;writer=None;gifs=[];lastprint=time.perf_counter()
    try:
        # Reader is already at24s. Do not seek/reset the shared event clock.
        while not reader.is_done() and reader.current_time<30000000:
            events=reader.load_delta_t(min(10000,30000000-reader.current_time));t=reader.current_time/1e6
            clean_reset=copy.deepcopy(pipelines['clean'][3].models['ordinary_rls']) if t>25+1e-8 and not reset_done else None
            if clean_reset is not None:reset_done=True
            panels=[]
            for name in CONDITIONS:
                front,sensor,normal_filter,predictor=pipelines[name]
                modified=delete_events(events,name,rectangles,t)
                o,density=front.detect(modified,t);s=sensor.step(o,front.history,t);sample=raw_sample(o,s,normal_filter,t,predictor)
                result=predictor.step(sample,reset_model=clean_reset if name!='clean' else None)
                result['t']=t;arrays[name].append(result)
                o.update(four_ellipses=s['four_ellipses'],four_visible=s['four_visible'],rotor_measurements=s['rotors'],
                         relative_force=result['relative'],physics_active=result['active'],input_events=len(modified),removed_events=len(events)-len(modified))
                handles[name].write(json.dumps(o,allow_nan=False)+'\n')
                c=counts[name];c['input_events']+=len(modified);c['deleted_events']+=len(events)-len(modified)
                c['detected']+=int(o['detected']);c['four_ellipses']+=int(s['four_ellipses']);c['physics_active']+=int(result['active'])
                if frames%5==0:
                    canvas=cv2.cvtColor(np.uint8(np.clip(np.log1p(density)/4,0,1)*255),cv2.COLOR_GRAY2BGR)
                    if name in rectangles and t<=25+1e-8:
                        x0,y0,x1,y1=rectangles[name];cv2.rectangle(canvas,(x0,y0),(x1,y1),(40,60,255),2)
                    for r in s['rotors']:
                        if r['observed_now']:cv2.ellipse(canvas,((r['cx'],r['cy']),(r['major'],r['minor']),r['angle']),(40,220,70),1)
                    if o['detected']:
                        cv2.drawMarker(canvas,tuple(np.int32(result['prediction'][2,-1])),(30,180,250),cv2.MARKER_CROSS,18,2)
                    # Fixed crop from warm-start only, identical for all panels.
                    cx,cy=warm_records[-1][1]['center'];x=int(np.clip(cx-400,0,width-800));y=int(np.clip(cy-225,0,height-450))
                    crop=canvas[y:y+450,x:x+800];panel=np.zeros((510,800,3),np.uint8);panel[60:]=crop
                    cv2.putText(panel,f'{name} t={t:.2f}s',(12,24),0,.65,(240,240,240),1)
                    cv2.putText(panel,f"four={s['four_ellipses']} physics={result['active']} | orange: ordinary RLS +0.4s",(12,49),0,.48,(220,220,220),1)
                    panels.append(panel)
            if panels:
                canvas=np.vstack((np.hstack(panels[:2]),np.hstack(panels[2:])))
                if writer is None:writer=cv2.VideoWriter(str(destination/'paired_raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1600,1020))
                writer.write(canvas)
                if frames in (0,100,200,400):cv2.imwrite(str(destination/f'comparison_{frames}.png'),canvas)
            frames+=1
            if time.perf_counter()-lastprint>12:
                print(json.dumps(dict(phase='paired_raw',sensor_time=t,frames=frames)),flush=True);lastprint=time.perf_counter()
    finally:
        for h in handles.values():h.close()
        if writer:writer.release()
    for name,records in arrays.items():
        data={key:np.asarray([r[key] for r in records]) for key in records[0]}
        np.savez_compressed(destination/name/'forecasts.npz',**data)
    # Paired inference before deletion must be identical, including all learners.
    times=np.array([r['t'] for r in arrays['clean']]);assert len(times)==600
    normal_after=times>25+1e-8
    for name in CONDITIONS[1:]:
        assert sum(r['removed_events'] for r in map(json.loads,(destination/name/'observations.jsonl').read_text().splitlines()) if r['t']>25+1e-8)==0
    save(destination/'completion.json',dict(frames_per_condition=frames,conditions=counts,warmup_frames=warm_count,
        wall_s=time.perf_counter()-begin,inference_inputs=[raw],future_labels_used=False,
        notes='No real RPM/pose truth; paired clean predictions are a counterfactual reference, not ground truth.'))
    print(json.dumps(counts),flush=True)

if __name__=='__main__':run()
