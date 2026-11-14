"""Optional local author-core interface. Does not load files or infer force.

Inputs are a current measured ellipse and a positive angular-phase prior. This
interface NEVER falls back to author ground-truth RPM/position initialization.
No binary or author source is included in the release tree. Not a default route.
"""
from pathlib import Path
import ctypes as C
import os
import numpy as np


class HelixCore:
    def __init__(self,ellipse,phase_hz,phase_rad=0.,direction=1,dll=None,rectification='affine'):
        self.center=np.array([ellipse['cx'],ellipse['cy']],float)
        if (not np.isfinite([*self.center,ellipse['major'],ellipse['minor'],ellipse['angle'],phase_rad]).all()
                or min(ellipse['major'],ellipse['minor'])<=0 or ellipse.get('fov_clipped')
                or not np.isfinite(phase_hz) or phase_hz<=0):
            raise ValueError('Finite ellipse centre and positive event-phase frequency required; no GT fallback')
        ratio=ellipse['minor']/ellipse['major']
        if not(.16<ratio<=1) or direction not in (-1,1):raise ValueError('Nonflat ellipse and explicit phase direction required')
        if rectification not in ('affine','author-conic'):raise ValueError('Unknown ellipse rectification')
        self.rectification=rectification
        self.rectifier=np.eye(3)
        angle=np.deg2rad(ellipse['angle'])
        if rectification=='affine':
            # A contour determines a conic, not its material-point phase
            # parameterization. Under the explicit weak-perspective assumption,
            # whitening the measured axes preserves constant angular rotation.
            co,si=np.cos(angle),np.sin(angle)
            self.rectifier[:2,:2]=np.diag([16/(ellipse['major']/2),16/(ellipse['minor']/2)])@np.array([[co,si],[-si,co]])
            radius,ratio,angle=16.,1.,np.pi/2
        else:radius=ellipse['minor']/2
        self._dll_dir=os.add_dll_directory('D:/Prophesee/third_party/bin')
        path=Path(dll) if dll else Path(__file__).resolve().parent.parent/'output/helix_bridge180_20261107/helixtrack_bridge.dll'
        self.lib=C.CDLL(str(path))
        self.lib.ht_create.argtypes=[C.c_double]*5+[C.c_int];self.lib.ht_create.restype=C.c_void_p
        self.lib.ht_destroy.argtypes=[C.c_void_p]
        self.lib.ht_process.argtypes=[C.c_void_p]+[C.c_void_p]*4+[C.c_int64,C.c_int]
        self.lib.ht_state.argtypes=[C.c_void_p,C.c_void_p]
        self.lib.ht_ring.argtypes=[C.c_void_p,C.c_void_p,C.c_int]
        self.handle=self.lib.ht_create(radius,ratio,angle,phase_hz,phase_rad,direction)
        if not self.handle:raise RuntimeError('Author core initialization failed')
        self.last_us=None

    def process(self,events,velocity=None,cutoff_s=None,solve_pose=False):
        if not len(events):return self.state()
        t=np.ascontiguousarray(events['t'],dtype=np.int64)
        if np.any(np.diff(t)<0) or self.last_us is not None and t[0]<=self.last_us:
            raise ValueError('Nonoverlapping time-ordered event intervals required')
        if cutoff_s is not None and t[-1]>=round(cutoff_s*1e6):raise ValueError('Future event supplied')
        xy=np.c_[events['x'],events['y']].astype(float)-self.center
        if velocity is not None:
            if cutoff_s is None:raise ValueError('Alignment requires a cutoff')
            xy+=(cutoff_s-t/1e6)[:,None]*np.asarray(velocity,float)
        xy=xy@self.rectifier[:2,:2].T
        x=np.ascontiguousarray(np.rint(xy[:,0]),dtype=np.int32)
        y=np.ascontiguousarray(np.rint(xy[:,1]),dtype=np.int32)
        p=np.ascontiguousarray(events['p'],dtype=np.int32)
        code=self.lib.ht_process(self.handle,*[C.c_void_p(a.ctypes.data) for a in [x,y,t,p]],len(t),int(solve_pose))
        if code:raise RuntimeError('Author core processing failed: '+str(code))
        self.last_us=int(t[-1]);return self.state()

    def state(self):
        out=np.empty(19,np.float64);code=self.lib.ht_state(self.handle,C.c_void_p(out.ctypes.data))
        if code:raise RuntimeError('Author core read failed')
        image_H=np.linalg.inv(self.rectifier)@out[10:19].reshape(3,3)
        return dict(phase_hz=float(out[0]),phase_acceleration_hz_s=float(out[1]),phase0_rad=float(out[2]),
            local_time_s=float(out[3]),pose_q=out[4:10].tolist(),H=image_H.tolist(),
            ellipse_rectification=self.rectification,weak_perspective_assumed=self.rectification=='affine',
            phase_candidate_in_native_band=bool(np.isfinite(out[0]) and 60<=abs(out[0])<=1200),
            qualified_for_force=False,phase_output_requires_external_observability_check=True,
            mechanical_RPM_truth=False,thrust_measured=False,author_core_used=True)

    def ring(self,n=180):
        points=np.empty((n,2),np.float64)
        code=self.lib.ht_ring(self.handle,C.c_void_p(points.ctypes.data),n)
        if code:raise RuntimeError('Author ring projection failed')
        q=np.c_[points,np.ones(n)]@np.linalg.inv(self.rectifier).T
        return q[:,:2]/q[:,2:]+self.center

    def close(self):
        if getattr(self,'handle',None):self.lib.ht_destroy(self.handle);self.handle=None
        if getattr(self,'_dll_dir',None):self._dll_dir.close();self._dll_dir=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
