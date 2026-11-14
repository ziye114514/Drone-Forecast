"""Local ctypes adapter to the unmodified author tracker core."""
from pathlib import Path
import ctypes as C
import os
import numpy as np
from functools import lru_cache

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT.parent/'output/helix_author_probe_20261025'
_dll_dirs=[os.add_dll_directory(p) for p in ['D:/Prophesee/third_party/bin','D:/Prophesee/bin'] if Path(p).is_dir()]
_feed_types=[C.c_void_p,C.POINTER(C.c_uint16),C.POINTER(C.c_uint16),C.POINTER(C.c_int64),C.POINTER(C.c_int16),C.c_int,C.c_int,C.POINTER(C.c_double)]

@lru_cache(maxsize=8)
def _load_library(path):
    lib=C.CDLL(str(path))
    lib.ht_create.argtypes=[C.c_int,C.c_double];lib.ht_create.restype=C.c_void_p
    lib.ht_destroy.argtypes=[C.c_void_p];lib.ht_destroy.restype=None
    lib.ht_feed_v2.argtypes=_feed_types;lib.ht_feed_v2.restype=C.c_int
    return lib


class AuthorTracker:
    def __init__(self,phase_equivalent_rpm,blades=2,dll_path=None,canvas=None,signed_coordinates=False,fixed_direction=False,rotation_sign=None,initial_phase_rad=None):
        self.lib=_load_library(OUT/'build/Release/helix_reference.dll' if dll_path is None else dll_path)
        self.signed_coordinates=signed_coordinates
        if canvas is not None:
            self.lib.ht_create_canvas.argtypes=[C.c_int,C.c_double,C.c_int,C.c_int,C.c_double,C.c_double,C.c_double]
            self.lib.ht_create_canvas.restype=C.c_void_p
            self.handle=self.lib.ht_create_canvas(blades,phase_equivalent_rpm,*canvas)
        else:
            self.lib.ht_create.argtypes=[C.c_int,C.c_double];self.lib.ht_create.restype=C.c_void_p
            self.handle=self.lib.ht_create(blades,phase_equivalent_rpm)
        self.blades=blades
        if fixed_direction or rotation_sign is not None:
            self.lib.ht_fixed_direction.argtypes=[C.c_void_p,C.c_int]
            self.lib.ht_fixed_direction.restype=None
            self.lib.ht_fixed_direction(self.handle,1 if rotation_sign is None else int(rotation_sign))
        if initial_phase_rad is not None:
            self.lib.ht_initialize_phase.argtypes=[C.c_void_p,C.c_double,C.c_double]
            self.lib.ht_initialize_phase.restype=None
            self.lib.ht_initialize_phase(self.handle,float(phase_equivalent_rpm),float(initial_phase_rad))
        if signed_coordinates:
            self.lib.ht_feed_signed_v2.argtypes=[C.c_void_p,C.POINTER(C.c_int32),C.POINTER(C.c_int32),*_feed_types[3:]]
            self.lib.ht_feed_signed_v2.restype=C.c_int
        if not self.handle:raise RuntimeError('Author tracker allocation failed')

    def feed(self,x,y,t,p,solve_pose=False):
        coordinate=np.int32 if self.signed_coordinates else np.uint16
        xx=np.ascontiguousarray(x,dtype=coordinate);yy=np.ascontiguousarray(y,dtype=coordinate)
        tt=np.ascontiguousarray(t,dtype=np.int64);pp=np.ascontiguousarray(p,dtype=np.int16)
        assert len(xx)==len(yy)==len(tt)==len(pp)
        output=np.zeros(32,np.float64)
        method=self.lib.ht_feed_signed_v2 if self.signed_coordinates else self.lib.ht_feed_v2
        pointer=C.POINTER(C.c_int32 if self.signed_coordinates else C.c_uint16)
        status=method(self.handle,xx.ctypes.data_as(pointer),yy.ctypes.data_as(pointer),
                            tt.ctypes.data_as(C.POINTER(C.c_int64)),pp.ctypes.data_as(C.POINTER(C.c_int16)),len(xx),int(solve_pose),
                            output.ctypes.data_as(C.POINTER(C.c_double)))
        if status!=0 or not np.isfinite(output).all():raise RuntimeError('Author tracker returned invalid state')
        phase_hz=abs(float(output[0]))/(2*np.pi)
        return dict(phase_hz=phase_hz,phase_equivalent_rpm=60*phase_hz,angular_velocity_rad_s=float(output[0]),
                    mechanical_rpm_under_known_blade_count=60*phase_hz/self.blades,
                    alpha=float(output[1]),phi0=float(output[2]),predictive_phase_concentration=float(output[3]),
                    accepted_events=int(output[4]),azimuth_sectors=int(output[5]),rotation_sign=int(output[6]),
                    q=output[7:13].tolist(),H=output[13:22].reshape(3,3,order='F').tolist(),
                    weighted_phase_concentration=float(output[22]),effective_radial_weight=float(output[23]),
                    informative_ring_sectors=int(output[24]),polarity_concentrations=output[25:27].tolist(),
                    polarity_event_counts=output[27:29].astype(int).tolist(),polarity_phase_difference=float(output[29]),
                    polarity_split_weighted_concentration=float(output[30]))

    def close(self):
        if self.handle:self.lib.ht_destroy(self.handle);self.handle=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
