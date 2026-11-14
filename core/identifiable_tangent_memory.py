"""Eliminate unknown constant force BEFORE claiming velocity response learned.

A finite zero-mean prior on gravity/wind can falsely identify c from constant
velocity. Weighted nuisance projection gives zero data information in that
case and makes c invariant to any constant projected external-force offset.
This correction is specified before inspecting new recording3 outcomes.
"""
import numpy as np
from scipy.special import log_ndtr
from tangential_response_memory import TangentialResponseMemory,adapt


def fit_samples(samples):
    prior=1/.35;A=np.array([s[1] for s in samples]);y=np.array([s[2] for s in samples]);variance=np.array([s[3] for s in samples])
    weight=1/np.sqrt(variance);B=A[:,:2]*weight[:,None];x=A[:,2]*weight;z=y*weight
    inverse=np.linalg.pinv(B);q=x-B@(inverse@x);target=z-B@(inverse@z)
    information=float(q@q);covariance=1/(1/prior**2+information)
    mu=float(covariance*(prior/prior**2+q@target));sigma=float(np.sqrt(covariance))
    standard=mu/sigma;positive=mu+sigma*np.exp(-.5*standard**2-.5*np.log(2*np.pi)-log_ndtr(standard))
    fraction=float(np.clip(1-covariance/prior**2,0,1));effective=(1-fraction)*prior+fraction*positive
    b=inverse@(z-x*mu)
    return dict(effective_response_s_inv=float(effective),unconstrained_response_mean_s_inv=mu,
        response_sigma_proxy_s_inv=sigma,positive_response_mean_s_inv=float(positive),information_fraction=fraction,
        unknown_constant_bias_px_s2=b.tolist(),data_only_response_information=information,
        nuisance_bias_rank=int(np.linalg.matrix_rank(B)),finite_zero_mean_gravity_wind_prior_used=False,
        unknown_force_offset_eliminated_before_response_identification=True)


class IdentifiableTangentMemory(TangentialResponseMemory):
    def observe(self,row):
        info=super().observe(row)
        if info.get('qualified'):info.update(fit_samples(self.samples))
        return info
