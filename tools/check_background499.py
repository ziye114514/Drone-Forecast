"""Coordinate, shared-uncertainty control and publication-order checks."""
from pathlib import Path
import sys,json,copy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from background_relative_force import BackgroundReference
from physical_publication import evaluate,freeze
from native_fov_observation import project

if __name__=='__main__':
    ref=BackgroundReference();noise=BackgroundReference(True);dt=1/30.;maximum=0.
    for i in range(12):
        t=1+i*dt
        ev=dict(t=t,proof=dict(qualified=i>=3,velocity_xy=[400.,0.],acceleration_xy=[0.,0.]),
            pair=None if i==0 else dict(source_t=t-dt,displacement_xy=[400*dt,0.],covariance_px2=(np.eye(2)*4).tolist()))
        ref.advance(ev);noise.advance(ev)
        body=dict(center=[500+300*i*dt,200.]);motion=dict(dt_s=dt if i else None,
            estimate=None if i==0 else dict(dx=300*dt,dy=0.,score=1.,displacement_covariance_px2=(np.eye(2)*4).tolist()))
        b,m=ref.inputs(t,body,motion);bn,mn=noise.inputs(t,body,motion)
        maximum=max(maximum,float(np.linalg.norm(np.asarray(b['center'])-[500-100*i*dt,200.])))
        assert bn==body
        if i:
            assert abs(m['estimate']['dx']+100*dt)<1e-8
            assert m['estimate']['displacement_covariance_px2']==mn['estimate']['displacement_covariance_px2']
    assert maximum<1e-8
    c=dict(position=[1220.,300.],velocity=[0.,0.],acceleration=[0.,0.])
    row=dict(t=1.,coefficient=copy.deepcopy(c),force=dict(acceleration_correlation_s=.35,acceleration_px_s2=[0.,0.]),
        valid=True,background_frame_motion=dict(relative_coefficient=c,velocity_xy=[400.,0.],acceleration_xy=[0.,0.],correlation_s=.35),
        native_fov_observation=dict(full_extent_px=[100.,40.]))
    measured=np.asarray(evaluate(row,[.1])['0.1']);expected=project([1260.,300.],[100.,40.])[0]
    assert np.linalg.norm(measured-expected)<1e-8
    assert np.linalg.norm(np.asarray(evaluate(freeze(row,1.),[.1])['0.1'])-expected)<1e-8
    #500's trained readout lives in image coordinates. Its residual offset
    #does not already contain the200px camera gauge, which must be added once.
    c2=dict(position=[1000.,300.],velocity=[0.,0.],acceleration=[0.,0.])
    r2=dict(coefficient=dict(c2,position=[1200.,300.]),force=row['force'],
        conditional_unknown_force=dict(mode='camera',position_offset=[0.,0.],future_unknown_rate_px_s3=[0.,0.]),
        background_frame_motion=dict(relative_coefficient=c2,velocity_xy=[400.,0.],acceleration_xy=[0.,0.],
            correlation_s=.35,offset_xy=[200.,0.],readout_in_image_coordinates=True))
    assert np.linalg.norm(np.asarray(evaluate(r2,[.1])['0.1'])-[1240.,300.])<1e-8
    print(json.dumps(dict(known_background_subtraction_max_error=maximum,
        uncertainty_only_and_mean_arm_covariances_equal=True,
        camera_added_before_FOV_projection=True,publication_retains_camera_component=True,
        fixed_image_readout_current_camera_gauge_added_once=True,
        hidden_rotor_measurement_created=False,real_accuracy_claim=False)))
