"""An older CURRENTLY reobserved rotor image constrains cumulative motion.

Only filtered native ON/OFF pixels enter the existing LK/RANSAC transform.
No four-rotor template is required. The key image is an owned past observation,
not a prediction or a current detection. Direct factors replace the correlated
adjacent-frame factor; they are not stacked as independent observations.
"""
import cv2
import numpy as np
from native_history_buffer import NativeHistoryBuffer
from native_plane_motion import estimate_plane_motion


class KeyframeRotorMotion:
    def __init__(self):
        self.history=NativeHistoryBuffer();self.image=None;self.mask=None
        self.center=None;self.t=None;self.active=False

    def _anchor(self,frame,t):
        mask=frame['native_mask'];x,y,w,h=cv2.boundingRect(mask.view(np.uint8))
        self.image,self.mask=self.history.update((frame['on'],frame['off']),mask,(x,y,w,h))
        self.center=np.asarray(frame['body']['center'],float).copy();self.t=float(t)
        return dict(create=True,source_t=float(t),reference_xy=self.center.tolist(),
            reference_is_observed_rotor_point_not_COM_truth=True)

    def observe(self,frame,t,previous_t):
        body=frame['body'];motion=frame['motion']
        if (frame.get('periodic_track_continuation') or {}).get('used'):self.active=True
        good=bool(body.get('detected') and not body.get('center_provisional') and np.any(frame['native_mask']))
        if not good:
            if self.t is not None and t-self.t>.4:
                self.image=None;self.t=None;self.active=False
            return
        if not self.active:return
        packet=None
        if self.t is None or body.get('measurement_reference_shift_xy') is not None or t-self.t>.4:
            packet=self._anchor(frame,t)
        elif previous_t is not None and self.t<previous_t-1e-7:
            shift,info=estimate_plane_motion(self.image,self.mask,(frame['on'],frame['off']),frame['native_mask'],self.center)
            if shift is not None and info.get('qualified'):
                packet=dict(create=False,source_t=self.t,cutoff_t=float(t),dt_s=t-self.t,
                    displacement_xy=[shift['dx'],shift['dy']],covariance_proxy=shift['displacement_covariance_px2'],
                    reference_xy=self.center.tolist(),transform=info,
                    replaces_adjacent_factor=True,independent_absolute_fix=False,
                    input_is_current_and_past_filtered_rotor_ONOFF=True)
            else:
                # Keep the adjacent observation, then refresh from CURRENT
                # evidence; a failed old template is never advanced by inertia.
                packet=self._anchor(frame,t);packet['previous_keyframe_rejected']=info
        if packet is not None:motion['keyframe_rotor_observation']=packet


def install(front):
    """Research hook: preserve the production front and old default path."""
    original=front.step;helper=KeyframeRotorMotion()
    def step(events,t):
        previous_t=front.last_t
        frame=original(events,t)
        helper.observe(frame,t,previous_t)
        return frame
    front.step=step
    return helper
