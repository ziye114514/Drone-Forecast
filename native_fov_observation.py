"""Project a predicted full rotor envelope to the sensor's visible envelope.

Native training teachers require a complete FOV at both endpoints; evaluation
includes clipped visible-outline centres. These are different observation
points at image borders. Clip the projected envelope, not the point itself.
Only a past/current complete native support supplies the extent; hidden rotor
RPM/force is never created. An entirely outside prediction is retained and
flagged, never dropped to improve scoring coverage.
"""
import copy
import numpy as np

WIDTH=1280;HEIGHT=720


class EnvelopeMemory:
    def __init__(self):self.extent=None;self.stamp=None;self.observations=None;self.last=None
    def step(self,row):
        obs=row.get('state',{}).get('observations');t=row['t']
        if obs is not None and self.observations is not None and obs<self.observations or self.last is not None and t-self.last>.5:
            self.extent=None;self.stamp=None
        self.observations=obs;self.last=t
        b=row['body'];box=b.get('bbox')
        if b.get('detected') and not b.get('center_provisional') and box and b.get('native_reference_calibration_allowed') is not False:
            x,y,X,Y=map(float,box)
            if x>0 and y>0 and X<WIDTH-1 and Y<HEIGHT-1:
                # Existing front adds6px each side; remove exactly that known
                # padding rather than learn a test-video size correction.
                full=np.array([X-x-12+1,Y-y-12+1])
                if np.all(full>0):self.extent=full;self.stamp=t
        return dict(full_extent_px=None if self.extent is None else self.extent.tolist(),
            last_complete_source_t=self.stamp,extent_is_past_native_projection_prior=True,
            future_shape_change_unobserved=True,body_heading_or_motor_identity_used=False)


class VisibleEnvelopeResponse:
    def __init__(self):self.memory=EnvelopeMemory()
    def step(self,row):
        info=self.memory.step(row);r=copy.deepcopy(row)
        if r.get('coefficient') and info['full_extent_px'] is not None:
            r['native_fov_observation']=info
            from physical_publication import evaluate
            r['physical']=evaluate(r,[.1,.2,.4])
            r['output_reference']='predicted_visible_native_rotor_envelope_center'
        return r


def project(point,extent):
    p=np.asarray(point,float);half=np.asarray(extent,float)/2
    low=np.maximum(p-half,[0.,0.]);high=np.minimum(p+half,[float(WIDTH),float(HEIGHT)])
    if np.any(high<=low):return p,dict(predicted_visible=False,unclipped_point_retained=True)
    return (low+high)/2,dict(predicted_visible=True,projected_visible_bbox=[*low.tolist(),*high.tolist()],
        geometric_extent_clipped=bool(np.any(low!=p-half) or np.any(high!=p+half)))


def predict(row,horizons):
    from physical_publication import evaluate
    r=copy.deepcopy(row);info=r.pop('native_fov_observation');values=evaluate(r,horizons)
    if info.get('rotor_template'):
        from rotor_template_projection import projected
        return {str(float(h)):projected(values[str(float(h))],info,float(h))[0].tolist() for h in horizons}
    return {str(float(h)):project(values[str(float(h))],info['full_extent_px'])[0].tolist() for h in horizons}
