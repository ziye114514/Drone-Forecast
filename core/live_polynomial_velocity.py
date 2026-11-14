"""Experimental velocity from the same existing robust quadratic fit.

Local polynomial differentiation is established (e.g. Savitzky-Golay). Here
the existing23 Huber solve uses actual irregular timestamps. No new fit model,
window, threshold or learned weight; the KF state remains untouched.
"""
import copy
import numpy as np
from scipy.optimize import least_squares
from live_motion import LiveMotion
from event_drone_forecast.position.robust import PARAMS


class PolynomialVelocityMotion(LiveMotion):
    def step(self,t,body):
        c,valid,diagnostic=super().step(t,body)
        active=False
        if c is not None and diagnostic['acceleration_eligible']:
            times=np.asarray([s for s,p in self.history]);points=np.asarray([p for s,p in self.history])
            span=np.ptp(times);tau=(times-times.mean())/span
            design=np.column_stack([np.ones(len(times)),tau,.5*tau*tau])
            initial=np.linalg.lstsq(design,points,rcond=None)[0];jacobian=np.kron(design,np.eye(2))
            fit=least_squares(lambda x:(design@x.reshape(3,2)-points).ravel(),initial.ravel(),
                jac=lambda x:jacobian.copy(),loss='huber',f_scale=PARAMS['huber_delta_px'],max_nfev=PARAMS['max_nfev'])
            if fit.success:
                coefficients=fit.x.reshape(3,2)
                np.testing.assert_allclose(coefficients[2]/span**2,diagnostic['fit']['acceleration'],rtol=1e-10,atol=1e-8)
                # Retain the SAME uncertainty-shrunk current acceleration,
                # rather than injecting a noisy unshrunk quadratic derivative.
                velocity=coefficients[1]/span+np.asarray(c['acceleration'])*(t-times.mean())
                c=copy.deepcopy(c);c['velocity']=velocity.tolist();active=True
        return c,valid,dict(diagnostic,polynomial_velocity_used=active,
                            velocity_is_current_robust_slope=True,KF_state_modified=False)
