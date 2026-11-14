"""Independent response ODE, headless invariance, publication and identity reset."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,numpy as np
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from conditional_unknown_force import ConditionalUnknownForce
from learned_rotor_predictor import load_model
from physical_publication import evaluate,freeze,publication

if __name__=='__main__':
    out=ROOT.parent/'output/controls391_20261110';out.mkdir(exist_ok=False)
    rr=[json.loads(s) for s in (ROOT.parent/'output/force357_20261109/15.jsonl').read_text('utf8').splitlines()];r=next(r for r in rr if r['t']>11.2)
    m,meta=load_model(ROOT/'models/native_response_v1.json');predictor=ConditionalUnknownForce(m);q=predictor.step(r);info=q['conditional_unknown_force'];j=np.asarray(info['future_unknown_rate_px_s3']);offset=np.asarray(info['position_offset'])
    def derivative(t,x):return np.r_[x[2:4],x[4:6],-x[4:6]/.35+j]
    horizons=[0.,.01,.1,.2,.4,.533];sol=solve_ivp(derivative,[0,max(horizons)],np.zeros(6),rtol=1e-10,atol=1e-11,dense_output=True)
    a=evaluate(q,horizons);b=evaluate(r,horizons);error=max(float(np.max(abs(np.asarray(a[str(h)])-np.asarray(b[str(h)])-offset-sol.sol(h)[:2]))) for h in horizons);assert error<1e-6,error
    flipped=copy.deepcopy(r);flipped['structure']['axis_xy']=(-np.asarray(r['structure']['axis_xy'])).tolist();f=ConditionalUnknownForce(m).step(flipped)
    headless=max(float(np.max(abs(np.asarray(evaluate(f,[h])[str(h)])-a[str(h)]))) for h in horizons);assert headless<1e-9,headless
    assert q['coefficient']==r['coefficient'] and q['force']==r['force']
    state=freeze(q,q['t']);age=.019;pub=publication(state,q['t']+age,0.,1);direct=evaluate(q,[age+.4])[str(age+.4)]
    pub_error=float(np.max(abs(np.asarray(pub['physical']['0.4'])-direct)));assert pub_error<1e-7,pub_error
    predictor.offset=np.array([100.,200.]);predictor.observations=50;new=copy.deepcopy(r);new['state']['observations']=1;new['body']['center_provisional']=True
    reset=predictor.step(new);assert reset['conditional_unknown_force']['position_offset']==[0.,0.]
    try:load_model(out/'missing_model.json')
    except FileNotFoundError:missing_rejected=True
    else:raise AssertionError('Missing model silently accepted')
    result=dict(independent_unknown_response_ODE_max_px=error,headless_normal_flip_max_px=headless,actual_publication_age_max_px=pub_error,
        internal_current_p_v_a_and_force_unchanged=True,reference_offset_reset_on_new_state=True,missing_model_rejected=True,
        main_accuracy_not_proven_by_unit_checks=True,RAW_FPS_measured=False)
    (out/'results.json').write_text(json.dumps(result,indent=2),'utf8');print(result)
