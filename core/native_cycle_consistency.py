"""Forward/backward qualification for the existing rotor-only matcher.

Reuse the original pairwise matcher on swapped frames. The current2px
displacement noise floor gives8px^2 cycle noise;99%2D chi-square is9.21034.
This is an engineering uncertainty check, not calibrated probability. No scene,
time, velocity direction, rotor count or trajectory-label condition is used.
"""
import numpy as np


def qualify(forward,backward):
    result=dict(forward)
    if backward is None or backward.get('peak_at_search_boundary'):
        return dict(result,current_displacement_qualified=False,
            cycle_consistency=dict(qualified=False,reason='reverse_match_unavailable_or_boundary'))
    a=np.array([forward['dx'],forward['dy']],float);b=np.array([backward['dx'],backward['dy']],float)
    residual=a+b;chi=float(residual@residual/8.)
    good=bool(np.isfinite(chi) and chi<=9.21034037197618 and not forward.get('peak_at_search_boundary'))
    covariance=np.eye(2)*4.+np.outer(residual,residual)/4.
    return dict(result,current_displacement_qualified=good,
        displacement_covariance_px2=covariance.tolist(),
        cycle_consistency=dict(qualified=good,reason='bidirectional_match_consistent' if good else 'forward_reverse_disagree',
            reverse_displacement_xy=b.tolist(),closure_residual_px=residual.tolist(),chi2_proxy=chi,
            threshold=9.21034037197618,probability_calibrated=False,
            covariance_divided_by_number_of_pixels=False))
