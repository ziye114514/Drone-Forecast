"""Remove pair-order bias using two current native rotor matches.

The two matches share images and are NOT independent measurements. Retain
the forward covariance and its qualification. Only an existing consistent
reverse match may correct the mean; the old 2px floor is never divided.
This is a translation-specific projection, not a reproduction of nonlinear
inverse-consistent registration.
"""
import numpy as np


def combine(forward,reverse):
    out=dict(forward)
    info=dict(applied=False,independent_measurements=False,covariance_reduced=False)
    out['symmetric_native_translation']=info
    if reverse is None or forward.get('peak_at_search_boundary') or reverse.get('peak_at_search_boundary'):
        info['reason']='no_two_interior_matches';return out
    a=np.array([forward['dx'],forward['dy']],float)
    b=np.array([reverse['dx'],reverse['dy']],float)
    closure=a+b;chi=float(closure@closure/8.)
    if not np.isfinite(chi) or chi>9.21034037197618:
        info.update(reason='existing_cycle_threshold_failed',chi2_proxy=chi);return out
    corrected=(a-b)/2.
    out.update(dx=float(corrected[0]),dy=float(corrected[1]))
    info.update(applied=True,reason='inverse_consistent_translation_projection',
        original_forward_xy=a.tolist(),reverse_xy=b.tolist(),closure_xy=closure.tolist(),
        correction_xy=(corrected-a).tolist(),chi2_proxy=chi,
        existing_cycle_threshold=9.21034037197618)
    return out
