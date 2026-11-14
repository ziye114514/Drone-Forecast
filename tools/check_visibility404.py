"""User invariants: changing visibility alone cannot create current force/motion."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,argparse,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from learned_rotor_predictor import load_model
from visibility_response import VisibilityResponse
from plane_conditioned_response import PlaneResponse
from conic_observability import usable
from physical_publication import evaluate,freeze,publication

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',default='controls404_20261110');parser.add_argument('--response-model',type=Path,default=ROOT/'models/native_response_visibility_v1.json');args=parser.parse_args()
    out=ROOT.parent/'output'/args.batch;out.mkdir(exist_ok=False);model,info=load_model(args.response_model)
    predictor=PlaneResponse if model.get('plane_conditioned') else VisibilityResponse
    rr=[json.loads(s) for s in (ROOT.parent/'output/training400_20261110/recording1/predictions.jsonl').read_text('utf8').splitlines()]
    r=next(r for r in rr if r['valid'] and len([g for g in r['geometry'].get('rotors',[]) if usable(g)])>=4 and r['coefficient'])
    seed=copy.deepcopy(r);seed['coefficient'].update(velocity=[0.,0.],acceleration=[0.,0.]);seed['force']['acceleration_px_s2']=[0.,0.]
    for k in ('response_model_mixture','external_unknown_rate_px_s3'):seed['force'].pop(k,None)
    seed['normal_turn_forecast']=dict(applied=False);g=[g for g in r['geometry']['rotors'] if usable(g)][:4];m=predictor(model);values=[];weights=[]
    for i in range(60):
        q=copy.deepcopy(seed);q['t']=100.+i/30;q['state']['observations']=i+1;q['geometry']['rotors']=copy.deepcopy(g[:2] if 20<=i<35 else g)
        for j,e in enumerate(q['geometry']['rotors']):
            e['minor']=e['major']*(.35+.12*np.sin(i/7));e['ratio']=e['minor']/e['major'];e['cx']+=5*np.sin(i/6+j*np.pi/2);e['cy']+=5*np.cos(i/6+j*np.pi/2)
        a=m.step(q);assert a['coefficient']==q['coefficient'] and a['force']==q['force'];assert np.linalg.norm(a['conditional_unknown_force']['future_unknown_rate_px_s3'])==0.
        values.append(evaluate(a,[.4])['0.4']);weights.append(a['visibility_response']['weight'])
    stationary=float(np.max(abs(np.asarray(values)-values[0])));assert stationary<1e-8,stationary
    moving=predictor(model).step(r);state=freeze(moving,moving['t']);age=.019;pub=publication(state,moving['t']+age,0.,1)
    direct=evaluate(moving,[age+.4])[str(age+.4)];publication_error=float(np.max(abs(np.asarray(pub['physical']['0.4'])-direct)));assert publication_error<1e-7
    flipped=copy.deepcopy(r);flipped['structure']['axis_xy']=(-np.asarray(r['structure']['axis_xy'])).tolist();other=predictor(model).step(flipped)
    flip_error=float(np.max(abs(np.asarray(evaluate(other,[.4])['0.4'])-evaluate(moving,[.4])['0.4'])));assert flip_error<1e-8
    result=dict(fixed_body_uniform_plane_wobble_and4_to2_to4_context_no_new_motion_max_px=stationary,headless_normal_flip_error_px=flip_error,
        geometry_or_count_change_does_not_change_current_force=True,current_p_v_a_unchanged=True,
        actual_publication_model_age_error_px=publication_error,context_weight_range=[min(weights),max(weights)],
        optional_author_joint_phase_required=False,these_are_mechanism_checks_not_real35px_proof=True)
    (out/'results.json').write_text(json.dumps(result,indent=2),'utf8');print(result)
