"""Known global transform versus localized/deforming point support."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from live_flow_spatial import support,observation,DESIGN
OUT=ROOT.parent/'output/spatial_flow_20261026';OUT.mkdir(exist_ok=False)
center=np.array([100.,100.]);rng=np.random.default_rng(261026)
p=np.concatenate([rng.uniform([15,15],[40,40],(12,2))*[sx,sy]+center for sx,sy in [(-1,-1),(-1,1),(1,-1),(1,1)]])
theta=.03;rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
q=(p-center)@rotation.T+center+[4.,-3.]
s=support(p,q,center,.04)
assert s['center_in_hull'] and s['jackknife_groups']==4
assert np.linalg.norm(np.asarray(s['leave_group_displacements'])-[4.,-3.])<1e-8
assert np.linalg.norm(s['extra_velocity_covariance'])<1e-10
one=p[(p[:,0]>100)&(p[:,1]>100)];localized=support(one,one+[4.,-3.],center,.04)
assert not localized['center_in_hull']
original=dict(valid=True,velocity=[100.,-75.],velocity_covariance=[[625.,0.],[0.,625.]],spatial_support=localized)
assert not observation(original,'center-hull')['valid'] and original['valid']
perturbed=q.copy();perturbed[(p[:,0]>100)&(p[:,1]>100),0]+=4.
deformed=support(p,perturbed,center,.04);assert np.trace(deformed['extra_velocity_covariance'])>0
adjusted=observation(dict(original,spatial_support=deformed),'spatial-covariance')
assert adjusted['valid'] and np.trace(adjusted['velocity_covariance'])>1250.
checks=dict(global_rotation_center_displacement_exact=True,localized_cluster_hull_rejected=True,
    spatial_deformation_increases_uncertainty=True,input_not_mutated=True,parameters_swept=False)
(OUT/'mechanism_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
(OUT/'design.json').write_text(json.dumps(dict(**DESIGN,
    sources=['https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html'],
    scope='Two fixed spatial mechanisms, no label-selected thresholds. Frozen current v4 observation/publication cohort.'),indent=2),encoding='utf-8')
print(json.dumps(checks),flush=True)
