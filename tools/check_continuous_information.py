"""Check partial-information rank, angle-only update, and identity reset."""
from pathlib import Path
import sys,json,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from continuous_five_circle import ContinuousFiveCircle
from check_five_circle_v3 import sequence


def run(output):
    stream=list(sequence());m=ContinuousFiveCircle()
    m.step(*stream[0]);m.step(*stream[1])
    _,b,full=stream[1];b=dict(b,confidence=.8)
    g=copy.deepcopy(full);g.update(state='insufficient',confidence=0.)
    g['observed_rotors']=g['observed_rotors'][:2];g['two_axis']={'valid':False}
    before=m.P.copy()+np.diag([36.,225.,225.,36.,225.,225.])*.02
    r=m.step(.04,b,g)
    assert r['mode']=='partial_two' and r['continuous']['measurement_rank']==4
    information=np.linalg.inv(m.P)-np.linalg.inv(before)
    assert np.linalg.matrix_rank(information,tol=1e-6)==4
    assert r['current_geometric_center'] is None and not r['joint_center_is_current_measurement']
    prior=m.P.copy();angle=np.deg2rad(8.);normal=[-np.sin(angle),np.cos(angle)]
    a=m.observe_axis(r,normal,.8,.04);assert a['continuous_axis']['accepted']
    line_information=np.linalg.inv(m.P)-np.linalg.inv(prior)
    assert np.linalg.matrix_rank(line_information,tol=1e-6)==1
    assert np.linalg.eigvalsh(m.P).min()>0
    assert abs(a['continuous_axis']['posterior_axis_residual_deg'])<abs(a['continuous_axis']['innovation_deg'])
    assert a['current_geometric_center'] is None
    # An unrelated target must not inherit the previous model.
    new_body=dict(b,body_track_id=2)
    q=m.step(.06,new_body,dict(observed_rotors=[],state='insufficient',confidence=0.))
    assert m.beta is None and q['model_centers'] is None
    cold=ContinuousFiveCircle();z=cold.step(.1,b,dict(observed_rotors=[],state='insufficient',confidence=0.))
    assert cold.observe_axis(z,normal,.8,.1)['model_centers'] is None
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    summary=dict(two_current_centres_information_rank=4,line_information_rank=1,
        covariance_positive_definite=True,axis_does_not_fabricate_center=True,
        target_identity_resets_template=True,cold_line_does_not_initialize_quad=True,
        controlled_checks_only=True)
    (out/'information_controls.json').write_text(json.dumps(summary,indent=2),'utf-8');print(json.dumps(summary))


if __name__=='__main__':run(sys.argv[1])
