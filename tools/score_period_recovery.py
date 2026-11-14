"""Freeze same-input native-period control for an optional aligned recovery run."""
from pathlib import Path
import os,sys,json,collections
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
import numpy as np
from visibility_period import LegacyPeriodProxy
from axis_amplitude_bridge import AxisAmplitudeBridge,current_axis_qualified
from projected_unknown_force import ProjectedResidualForce
from normal_turn_forecast import NormalTurnForecast
from adaptive_force_memory import AdaptiveForceMemory
from compare_final_pipeline import score
def main(run,label):
    out=Path(run);rows=[json.loads(s) for s in (out/'predictions.jsonl').read_text('utf-8').splitlines()]
    protocol=json.loads((out/'protocol.json').read_text('utf-8'));result={};versions={}
    for enabled in [True,False]:
        force_class=ProjectedResidualForce
        if protocol.get('endpoint_force_balance'):
            from endpoint_force_balance import EndpointForceBalance
            force_class=EndpointForceBalance
        selector=LegacyPeriodProxy();bridge=AxisAmplitudeBridge();force=force_class(gain_signal_to_noise=2.)
        force.evidence_gain=True;normal=NormalTurnForecast();memory=AdaptiveForceMemory(extend_calibrated=protocol.get('calibrated_force_memory',False));values=[]
        for r in rows:
            pooled=r['period']
            if not enabled:pooled=pooled.get('original_native_period',pooled)
            s=r['structure'];m=r.get('image_motion') or {}
            u,q,selection=selector.step(r['t'],s,r['geometry'].get('rotors',[]),r['rates'],pooled)
            u,q,ad=bridge.step(r['t'],s,u,q,selection);force.input_learning_current=ad['learning_input_current']
            force.observe_flow(m.get('observation',dict(valid=False)),m.get('acceleration',{}).get('applied',False));force.observe_axis_motion(s,m)
            p,f=force.step(r['t'],r['coefficient'],r.get('force_observed_position'),u,q)
            trend=normal.observe(r['t'],s,r['coefficient'],m);p,nd=normal.correct(p,f,trend)
            p,md=memory.step(r['t'],r['coefficient'],r.get('force_observed_position'),p,f,nd,r['valid'],
                m.get('acceleration',{}).get('applied',False),current_axis_qualified(s))
            values.append(dict(r,physical=p,force=f,period_selection=selection,normal_turn_forecast=nd,adaptive_force_memory=md))
        key='aligned' if enabled else 'same_input_native_control';versions[key]=values
        (out/(key+'.jsonl')).write_text('\n'.join(json.dumps(r) for r in values)+'\n','utf-8')
    difference=max(float(np.max(abs(np.asarray(a['physical'][h])-b['physical'][h]))) for a,b in zip(rows,versions['aligned']) for h in a['physical'])
    assert difference<1e-7,difference
    truth=sorted(json.loads(Path(label).read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    diag=[r['aligned_period_recovery'] for r in rows];tried=[d for d in diag if 'aligned_period' in d]
    result=dict(scores={k:score(v,truth)['physical_future_error_px'] for k,v in versions.items()},actual_replay_max_difference=difference,
        added_qualified_period_ticks=sum(d['applied'] for d in diag),attempts=len(tried),
        maximum_mode_pixels=max((d['aligned_period'].get('mode_pixels',0) for d in tried),default=0),
        reasons=dict(collections.Counter(d['reason'] for d in diag)),
        current_force_inputs=sum(r['force']['active'] and r['force']['last_current_force_age_s']<1e-9 for r in rows),
        native_configuration=protocol.get('aligned_period_filter','strict7'),
        recovery_CPU_mean_ms=float(np.mean([d['CPU_ms'] for d in diag])),
        same_inputs_prediction_slots=True,minimum_pixel_count_not_lowered=True)
    (out/'recovery_score.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))
if __name__=='__main__':main(*sys.argv[1:])
