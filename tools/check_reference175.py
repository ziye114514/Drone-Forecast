"""A body-reference correction must not move any current image geometry."""
from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from fast_structure_front import FastStructure
from coherent_gap_observer import ShortGapObserver
OUT=ROOT.parent/'output/reference175_20261107'

if __name__=='__main__':
    OUT.mkdir(exist_ok=True);f=FastStructure();hubs=np.array([[170.,90.],[230.,90.],[230.,110.],[170.,110.]])
    original_body=np.array([210.,130.]);delta=np.array([-10.,-30.]);body=dict(center=original_body.tolist())
    core=f.model.core;core.hubs=hubs.copy();core.anchor=original_body.copy();core.shapes=np.tile(np.diag([100.,25.]),(4,1,1))
    core.pending=(hubs.copy(),original_body.copy());f.previous_body=original_body.copy()
    f.model._seed(1.,body,dict(model_centers=hubs.tolist()))
    P=f.model.P.copy();tensors=f.model.shapes.copy();before=f.model._hubs().copy()
    old_phantom=float(np.linalg.norm(core._expected(original_body+delta)-hubs,axis=1).mean())
    audit=f.rebase_body_reference(delta);new_body=original_body+delta
    np.testing.assert_allclose(core._expected(new_body),hubs,atol=1e-12)
    np.testing.assert_allclose(f.model._hubs(),before,atol=1e-12)
    np.testing.assert_array_equal(f.model.P,P);np.testing.assert_array_equal(f.model.shapes,tensors)
    np.testing.assert_allclose(new_body-f.previous_body,np.zeros(2),atol=1e-12)
    np.testing.assert_allclose(new_body-f.model.body,np.zeros(2),atol=1e-12)
    np.testing.assert_allclose(core.pending[0],hubs);np.testing.assert_allclose(core.pending[1],new_body)
    motion=np.array([5.,-2.]);np.testing.assert_allclose(core._expected(new_body+motion),hubs+motion,atol=1e-12)
    m=ShortGapObserver();m._initialize(original_body);m.x[2:6]=[100.,-20.,30.,-10.];force=m.x[2:7].copy();m._rebase_observation_reference(delta)
    np.testing.assert_array_equal(m.x[2:7],force)
    try:f.rebase_body_reference([np.nan,0.])
    except ValueError:pass
    else:raise AssertionError('Invalid reference shift accepted')
    result=dict(audit=audit,old_phantom_motion_px=old_phantom,new_phantom_motion_px=0.,
        image_hubs_shapes_covariance_unchanged=True,true_translation_still_observed=True,
        pending_confirmation_references_transported=True,force_and_velocity_unchanged=True,
        synthetic_mechanism_not_forecast_accuracy=True)
    (OUT/'mechanism.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))
