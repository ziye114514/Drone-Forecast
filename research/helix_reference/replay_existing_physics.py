"""Feed NEW phase-verified features into the EXISTING146 physical expert.

The author's rotor core is not imported here. This is a frozen-observation
source-time replay, not a new RAW run or deployment. The project RLS arithmetic,
delay, clipping, forgetting and per-horizon maturity gates are reused verbatim.
"""
from pathlib import Path
from collections import deque
import ast,hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT.parent/'output/helix_author_probe_20261025'
OUT=BASE/'existing146_phase_replay';OUT.mkdir(exist_ok=False)
source=ROOT/'research/stage146/branches146.py';tree=ast.parse(source.read_text(encoding='utf-8'))
node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CausalPhysicsExpert')
namespace=dict(np=np,deque=deque)
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
Expert=namespace['CausalPhysicsExpert']
input_file=BASE/'RAW_shadow_recovery/predictions.jsonl'
rows=[json.loads(s) for s in input_file.read_text(encoding='utf-8').splitlines()]

def features(obs):
    proxy=obs.get('conditional_rotation_proxy') or {};rho=proxy.get('relative_to_first_supported')
    if not obs.get('valid') or rho is None:return None
    vectors=[];confidence=[]
    for v in obs['rotors'].values():
        p=v.get('current_projection')
        if p is None:return None
        n=np.asarray(p['normal_axis_unoriented'],float)
        if n[1]>0:n=-n
        vectors.append(n*np.sqrt(max(0.,1-p['minor_to_major']**2)))
        confidence.append(v['selected']['weighted_phase_concentration'])
    return np.median(vectors,axis=0),rho,float(min(confidence))

physical=bias=None;epoch=None;issued=[]
for row in rows:
    obs=row['auxiliary_observation'];feature=features(obs);current_epoch=obs.get('rotation_proxy_epoch')
    if current_epoch!=epoch:
        epoch=current_epoch;physical=bias=None
    if physical is None and feature is not None:
        physical=Expert();bias=Expert();normal,rho,quality=feature
        X0=np.array([[rho*normal[0],1.,0.],[rho*normal[1],0.,1.]])
        # Match the initial acceleration and its covariance. A default hover
        # cancellation must not masquerade as a benefit from event features.
        initial_a=X0@physical.theta;initial_P=X0@physical.P@X0.T
        bias.theta=np.r_[0.,initial_a];bias.P=np.zeros((3,3));bias.P[1:,1:]=initial_P
    if physical is None:continue
    proxy=dict(valid=feature is not None,relative_f2_proxy=feature[1] if feature else 1.,hidden_speed_assumption_used=False)
    geom=dict(axis_xy=feature[0].tolist() if feature else None,confidence=feature[2] if feature else 0.)
    bg=dict(axis_xy=[0.,0.] if feature else None,confidence=geom['confidence'])
    c=row['coefficients'].get('optimized_geometry');base=row['predictions'].get(row['preferred_forecast'],{})
    a=physical.step(row['t'],row['body'],geom,proxy,c,base)
    b=bias.step(row['t'],row['body'],bg,proxy,c,base)
    if feature is None or not row['valid'] or a['physical'] is None:continue
    for h in [.1,.2,.4]:
        key=str(h);weight=a['weights_by_horizon'][key]
        matched=(1-weight)*np.asarray(base[key])+weight*np.asarray(b['physical'][key])
        issued.append(dict(t=row['t'],h=h,target_t=round(row['t']+h,8),epoch=epoch,
            predictions=dict(baseline=base[key],physical=a['physical'][key],bias_only=b['physical'][key],
                physical_fused=a['fused'][key],bias_fused=b['fused'][key],same_event_weight_bias=matched.tolist()),
            event_weight=weight,bias_weight=b['weights_by_horizon'][key],updates=a['updates'],
            parameters_effective_pixel_units=a['parameters_effective_pixel_units']))
future={round(row['t'],6):row for row in rows};scored=[]
for item in issued:
    reference=future.get(round(item['target_t'],6))
    if reference is None or not reference['body']['detected']:continue
    target=np.asarray(reference['body']['center'])
    scored.append(dict(item,errors={k:float(np.linalg.norm(np.asarray(v)-target)) for k,v in item['predictions'].items()}))
summary={}
for h in [.1,.2,.4]:
    seq=[r for r in scored if r['h']==h]
    summary[str(h)]=dict(scored=len(seq),event_active=sum(r['event_weight']>0 for r in seq),
        bias_active=sum(r['bias_weight']>0 for r in seq),
        mean_error_px={k:float(np.mean([r['errors'][k] for r in seq])) if seq else None
                       for k in ['baseline','physical','bias_only','physical_fused','bias_fused','same_event_weight_bias']})
result=dict(scope='Frozen MAIN RAW outputs replayed at source time; same detector future center only, no independent ground truth',
    reused_class='Exact146 CausalPhysicsExpert AST; no parameter changes',
    counterfactual='Zero-feature expert starts with same instantaneous acceleration and covariance',
    normal_sign_assumption='image-up; not camera gravity calibration',
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),input_sha256=hashlib.sha256(input_file.read_bytes()).hexdigest(),
    summary=summary,scored=scored,integrated_into_main=False,physical_parameters_identified=False)
(OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(summary),flush=True)
