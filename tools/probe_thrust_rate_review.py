"""Explain clip differences and compare a causal source-aware thrust-rate prior.

All forecast variants are frozen before development labels are read. This is
saved-input replay, not another RAW/FPS measurement or mechanical thrust test.
"""
from pathlib import Path
import os,sys,json,time,collections
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
import numpy as np
from thrust_trend_forecast import ThrustTrendForecast,SourceAwareThrustTrendForecast
from axis_amplitude_bridge import current_axis_qualified
from compare_final_pipeline import score,reference_at

def checks():
    result={};old=ThrustTrendForecast();new=SourceAwareThrustTrendForecast()
    for i in range(10):
        t=i/30;f=300.*np.exp(.08*t)
        selected='visible_frequency_squared_mean_hidden_equal_mean_hypothesis' if i%2 else 'pooled_flicker_representative_speed_hypothesis'
        if i==9:selected='pooled_flicker_representative_speed_hypothesis'
        selection=dict(source=selected,relative_f2=(f/300.)**2,coherent=True)
        pooled=dict(qualified=True,frequency_hz=f,mode_fraction=.8)
        a=old.observe(t,selection,pooled);b=new.observe(t,selection,pooled,geometry_epoch=0)
    assert not a['qualified'] and b['qualified'] and b['log_thrust_rate_per_s']>0.,(a,b)
    result['valid_pooled_history_survives_unselected_partial_frames']=True
    b=new.observe(10/30,dict(selection,source='visible_frequency_squared_mean_hidden_equal_mean_hypothesis'),pooled,geometry_epoch=0)
    assert not b['qualified'];result['pooled_rate_not_applied_to_partial_motor_mean']=True
    b=new.observe(11/30,selection,pooled,dict(learning_input_current=False),geometry_epoch=0)
    assert not b['qualified'];result['held_amplitude_not_a_current_rate_measurement']=True
    b=new.observe(12/30,selection,pooled,geometry_epoch=1)
    assert not b['qualified'];result['geometry_epoch_cannot_create_a_thrust_rate_jump']=True
    model=SourceAwareThrustTrendForecast()
    for i in range(10):
        p=dict(qualified=True,frequency_hz=300.,mode_fraction=.8)
        s=dict(source='pooled_flicker_representative_speed_hypothesis',relative_f2=1.,coherent=True)
        b=model.observe(i/30,s,p)
    assert b['qualified'] and b['log_thrust_rate_per_s']==0.
    for gain in [200.,2000.,4500.]:
        base={'0.4':[10.,20.]};f=dict(active=True,input_u=[.1,-1.],effective_gain_px_s2=gain,acceleration_correlation_s=.35)
        unchanged,_=model.correct(base,f,None,b)
        assert unchanged==base
    result['constant_frequency_zero_thrust_rate_despite_gain_gauge_changes']=True
    separated=model.observe(10/30,s,dict(p,frequency_hz=306.,source='persistent_body_translation_frequency'))
    assert not separated['qualified'];result['native_and_stabilized_period_histories_not_differenced_together']=True
    # Analytic amplitude-and-direction integral, independently quadratured;
    # the unknown-force term must never acquire a thrust amplitude multiplier.
    from adaptive_force_memory import kernel
    known=np.array([40.,-500.]);d=np.array([30.,400.]);omega=.4;rate=.18;tau=.35;weight=.6;h=.4
    lam=-1/tau+1j*omega;z=complex(*known)
    baseline=((1-weight)*kernel(lam,h)+weight*kernel(1j*omega,h))*z
    baseline=np.array([baseline.real,baseline.imag])+((1-weight)*kernel(-1/tau,h)+weight*kernel(0.,h))*d
    f=dict(active=True,input_u=known.tolist(),effective_gain_px_s2=1.,acceleration_correlation_s=tau)
    normal=dict(applied=True,trend=dict(shrunk_angular_rate_rad_s=omega));trend=dict(qualified=True,log_thrust_rate_per_s=rate)
    pred,info=model.correct({'0.4':baseline.tolist()},f,normal,trend,dict(applied=True,weight=weight))
    x=np.linspace(0,h,20001);memory=(1-weight)*np.exp(-x/tau)+weight
    thrust=np.exp((rate+1j*omega)*x)*z
    integrand=(h-x)[:,None]*memory[:,None]*(np.column_stack((thrust.real,thrust.imag))+d)
    expected=np.trapezoid(integrand,x,axis=0);error=float(np.linalg.norm(np.asarray(pred['0.4'])-expected))
    assert error<1e-6,error;result['force_rate_direction_and_fixed_disturbance_integral_error_px']=error
    return result

def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);base=ROOT.parent/'output';controls=checks();frozen={}
    specs=[('side','motion_consistency_30px_20261028/RAW_side_release_candidate8s','dense_labels_side_110_1195'),
        ('other','motion_consistency_30px_20261028/calibrated_memory_gaps','dense_labels_train2_40_46')]
    for name,folder,label in specs:
        source=base/folder/('train2.jsonl' if name=='other' else 'predictions.jsonl')
        rows=[json.loads(s) for s in source.read_text('utf-8').splitlines()];variants={'baseline':rows};cost={}
        for key,cls in [('old_rate',ThrustTrendForecast),('source_aware_rate',SourceAwareThrustTrendForecast)]:
            model=cls();values=[];times=[]
            for r in rows:
                start=time.perf_counter();args=(r['t'],r['period_selection'],r['period'],r.get('axis_amplitude_bridge'))
                trend=model.observe(*args,geometry_epoch=r['structure'].get('geometry_epoch')) if key=='source_aware_rate' else model.observe(*args)
                p,d=model.correct(r['physical'],r['force'],r.get('normal_turn_forecast'),trend,r.get('adaptive_force_memory'))
                times.append((time.perf_counter()-start)*1000);values.append(dict(r,physical=p,thrust_trend_forecast=d))
            variants[key]=values;cost[key]=float(np.mean(times))
        for key,values in variants.items():(out/(name+'_'+key+'.jsonl')).write_text('\n'.join(json.dumps(r) for r in values)+'\n','utf-8')
        frozen[name]=(variants,cost,label)
    # The following is diagnosis/scoring only, after all causal outputs exist.
    result=dict(controlled=controls,clips={},RAW_retimed=False,parameters_swept=False,default_enabled=False)
    for name,(variants,cost,label) in frozen.items():
        truth=sorted(json.loads((base/label/'labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
        rows=variants['baseline'];details=[];slot=None
        for r in rows:
            new=int(np.floor(r['t']*5+1e-8))
            if not r['valid'] or slot==new:continue
            slot=new;target=reference_at(truth,r['t']+.4);now=reference_at(truth,r['t'])
            if target is None:continue
            details.append(dict(t=r['t'],error_px=float(np.linalg.norm(np.asarray(r['physical']['0.4'])-target)),
                current_position_error_px=None if now is None else float(np.linalg.norm(np.asarray(r['coefficient']['position'])-now)),
                predicted_displacement=(np.asarray(r['physical']['0.4'])-r['coefficient']['position']).tolist(),
                actual_displacement=None if now is None else (target-now).tolist(),
                flow_reason=(r.get('image_motion') or {}).get('acceleration',{}).get('reason'),force_active=r['force']['active']))
        result['clips'][name]=dict(frames=len(rows),current_body=sum(r['body'].get('detected',False) for r in rows),
            qualified_axis=sum(current_axis_qualified(r['structure']) for r in rows),
            qualified_period=sum(r['period'].get('qualified',False) for r in rows),
            current_force=sum(r['force']['active'] and (r.get('axis_amplitude_bridge') or {}).get('learning_input_current',True) for r in rows),
            held_amplitude=sum((r.get('axis_amplitude_bridge') or {}).get('applied',False) for r in rows),
            gain_updates=sum(r['force']['gain_learned'] for r in rows),
            scores={k:score(v,truth)['physical_future_error_px'] for k,v in variants.items()},
            rate_applied={k:sum((r.get('thrust_trend_forecast') or {}).get('applied',False) for r in v) for k,v in variants.items() if k!='baseline'},
            rate_reasons={k:dict(collections.Counter(r['thrust_trend_forecast']['reason'] for r in v)) for k,v in variants.items() if k!='baseline'},
            worst_forecasts=sorted(details,key=lambda r:-r['error_px'])[:4],component_CPU_mean_ms=cost,
            same_inputs_and_valid_forecast_slots=True)
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main(sys.argv[1])
