"""Optional CPU solution of the existing convex Huber regression objective.

No history, loss, whitening, fitted degree or acceptance threshold changes.
A bounded active-set Newton solve falls back to the original SciPy solver if
it cannot certify a small gradient. Compilation is warmed before RAW pacing.
"""
import numpy as np
from scipy.optimize import least_squares

_solver=None
_counts={'native':0,'fallback':0}


def configure(enabled=True):
    global _solver
    if not enabled:_solver=None;return dict(enabled=False)
    import time
    from native_huber import solve
    start=time.perf_counter()
    J=np.column_stack((np.ones(12),np.linspace(-1.,0.,12)))
    target=np.arange(12,dtype=float);target[-1]+=20.
    solve(np.ascontiguousarray(J),target,np.linalg.lstsq(J,target,rcond=None)[0],8.07/6.)
    _solver=solve
    return dict(enabled=True,warm_compile_load_ms=(time.perf_counter()-start)*1000,
        objective='same whitened Huber8.07/6',fallback='original scipy least_squares',
        no_fastmath=True,gradient_tolerance=1e-8)


def fit(J,target,initial):
    if _solver is not None:
        p,ok,nfev,gradient=_solver(np.ascontiguousarray(J),np.ascontiguousarray(target),
            np.ascontiguousarray(initial),8.07/6.)
        if ok:
            _counts['native']+=1
            return p,True,int(nfev)
        _counts['fallback']+=1
    result=least_squares(lambda p:J@p-target,initial,jac=lambda p:J.copy(),loss='huber',f_scale=8.07/6.,max_nfev=30)
    return result.x,bool(result.success),int(result.nfev)


def counters():return dict(_counts)
