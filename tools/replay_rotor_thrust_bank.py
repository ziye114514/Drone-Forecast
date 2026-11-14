"""Freeze same-input controls before labels; do not re-time RAW or alter geometry."""
from pathlib import Path
import sys,json,time,collections,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from rotor_thrust_bank import RotorThrustBankProxy
from visibility_period import LegacyPeriodProxy,force_input
from axis_amplitude_bridge import AxisAmplitudeBridge
from projected_unknown_force import ProjectedResidualForce
from flow_force_balance import FlowBalancedForce
from compare_final_pipeline import score

def encoded(x):
    return json.dumps(x,allow_nan=False,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item())

def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    base=ROOT.parent/'output/shared_template_budget_20261027';reports={}
    specs=[('clear','clear_fused_line_goal',None),('side','side_visibility_final8s','dense_labels_side_110_1195'),
           ('train2','train2_visibility_final','dense_labels_train2_40_46')]
    for name,folder,label in specs:
        rows=[json.loads(s) for s in (base/folder/'predictions.jsonl').read_text('utf-8').splitlines()]
        bank=RotorThrustBankProxy();bridge=AxisAmplitudeBridge()
        protocol=json.loads((base/folder/'protocol.json').read_text('utf-8'))
        cls=ProjectedResidualForce if protocol.get('partial_tangent_unknown_force') else FlowBalancedForce
        old_force=cls(gain_signal_to_noise=2.);new_force=cls(gain_signal_to_noise=2.)
        arms=dict(control=[],candidate=[]);sources=collections.Counter();mapping=collections.Counter();costs=[];baseline_diff=0.
        for r in rows:
            start=time.perf_counter();u,q,period=bank.step(r['t'],r['structure'],r['geometry'].get('rotors',[]),r['rates'],r['period'])
            if period.get('rotor_bank_reason')=='no_current_identified_geometry':
                # JSON changes NumPy scalar dtypes. Keep the original frozen
                # fallback input rather than call a float32/64 replay difference a gain.
                period=dict(r['period_selection'],rotor_bank_used=False,rotor_bank_reason='no_current_identified_geometry')
                u,q=force_input(r['structure'],period['relative_f2'],period['coherent'])
            u,q,amp=bridge.step(r['t'],r['structure'],u,q,period);costs.append((time.perf_counter()-start)*1000)
            sources[period['source']]+=1;mapping[str(len(period.get('current_model_ids',[])))]+=1
            motion=r.get('image_motion') or {}
            for arm,force,inp,quality,learn in [
                ('control',old_force,r['force']['input_u'],.35,(r.get('axis_amplitude_bridge') or {}).get('learning_input_current',True)),
                ('candidate',new_force,u,q,amp['learning_input_current'])]:
                force.input_learning_current=learn
                force.observe_flow(motion.get('observation',dict(valid=False)),motion.get('acceleration',{}).get('applied',False))
                if protocol.get('partial_tangent_unknown_force'):force.observe_axis_motion(r['structure'],motion)
                p,f=force.step(r['t'],r['coefficient'],r.get('force_observed_position'),inp,quality)
                arms[arm].append(dict(r,physical=p,force=f,period_selection=period if arm=='candidate' else r['period_selection']))
                if arm=='control' and p:
                    baseline_diff=max(baseline_diff,float(np.max(np.abs(np.array(list(p.values()))-np.array(list(r['physical'].values()))))))
        assert baseline_diff<1e-8,(name,baseline_diff)
        # Save every causal output before opening any manual reference.
        for arm,v in arms.items():(out/(name+'_'+arm+'.jsonl')).write_text('\n'.join(encoded(r) for r in v)+'\n','utf-8')
        report=dict(source_frames=len(rows),mode_counts=dict(collections.Counter(r['structure']['mode'] for r in rows)),
            sources=dict(sources),mapped_current_counts=dict(mapping),control_max_difference_from_original=baseline_diff,
            mechanism_CPU_ms=dict(mean=float(np.mean(costs)),p95=float(np.percentile(costs,95))),
            changed_future04_predictions=sum(not np.allclose(a['physical'].get('0.4',[]),b['physical'].get('0.4',[]),atol=1e-8,rtol=0) for a,b in zip(*arms.values())),
            scores=None,RAW_retimed=False)
        if label:
            truth=sorted(json.loads((ROOT.parent/'output'/label/'labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
            report['scores']={arm:score(v,truth) for arm,v in arms.items()}
            for v in report['scores'].values():v.pop('anchors',None)
        reports[name]=report
    (out/'summary.json').write_text(json.dumps(reports,indent=2),'utf-8')
    print(json.dumps(reports))
if __name__=='__main__':main(sys.argv[1])
