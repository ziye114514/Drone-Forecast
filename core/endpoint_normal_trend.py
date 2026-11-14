"""Causal endpoint normal trend, with uncertainty-qualified curvature.

The old straight-line slope is a window-average angular velocity. Under
angular acceleration it lags the current rate. Estimate curvature using the
SAME current/past planes, never position labels or a flight-specific model.
Unresolved curvature exactly preserves the original angular-rate forecast.
"""
import numpy as np
from normal_turn_forecast import NormalTurnForecast


class EndpointNormalTrend(NormalTurnForecast):
    def observe(self,t,structure,coefficient,motion=None):
        info=super().observe(t,structure,coefficient,motion)
        if not info.get('qualified'):return info
        a=np.asarray(self.rows);tt=a[:,0]-t;weights=1/a[:,2]**2;weights/=weights.sum()
        X=np.c_[np.ones(len(tt)),tt,.5*tt*tt]
        # Keep the existing correlated-error floor; do not manufacture
        # information by treating consecutive silhouettes as independent.
        noise=info['angle_sigma_proxy_rad'];maximum=np.deg2rad(30.)
        prior_sigma=maximum/.4**2
        precision=X.T@(weights[:,None]*X)/(noise*noise)+np.diag([0.,0.,1/prior_sigma**2])
        covariance=np.linalg.inv(precision);beta=covariance@(X.T@(weights*a[:,1]))/(noise*noise)
        alpha=float(beta[2]);sigma=float(np.sqrt(covariance[2,2]))
        trust=alpha*alpha/(alpha*alpha+sigma*sigma)
        # Exact nesting: when alpha=0, both rate and rollout are unchanged.
        rate=info['angular_rate_rad_s']
        endpoint=rate+trust*(float(beta[1])-rate)
        rate_noise=info['angular_rate_sigma_proxy']
        reliability=endpoint*endpoint/(endpoint*endpoint+rate_noise*rate_noise)
        omega=float(np.clip(endpoint*reliability,-maximum/.4,maximum/.4))
        alpha*=trust
        # One bounded polynomial over the whole .4s horizon. The same
        # coefficients are used for shorter horizons and publication ages.
        times=[0.,.4]
        if abs(alpha)>1e-12 and 0<-omega/alpha<.4:times.append(-omega/alpha)
        sweep=max(abs(omega*s+.5*alpha*s*s) for s in times)
        scale=min(1.,maximum/max(sweep,1e-12));omega*=scale;alpha*=scale
        return dict(info,shrunk_angular_rate_rad_s=omega,angular_acceleration_rad_s2=alpha,
            angular_acceleration_sigma_proxy_rad_s2=sigma,curvature_credit=trust,
            original_average_rate_rad_s=info['shrunk_angular_rate_rad_s'],
            reason='causal_endpoint_plane_trend',past_window_only=True,
            maximum_future_sweep_deg=30.,curvature_noise_calibrated=False)


NODES,WEIGHTS=np.polynomial.legendre.leggauss(16)


def rotation_kernel(lam,h,omega,alpha):
    """Integral of (h-s) exp(lam*s + i*(omega*s+.5*alpha*s²))."""
    s=(NODES+1)*h/2
    return complex(np.sum(WEIGHTS*(h-s)*np.exp(lam*s+1j*(omega*s+.5*alpha*s*s)))*h/2)
