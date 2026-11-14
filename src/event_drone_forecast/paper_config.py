"""Paper-oriented settings; no unprovided physical calibration is fabricated."""
from dataclasses import dataclass,asdict
from pathlib import Path
import json


@dataclass
class PaperConfig:
    step_us:int=10000
    spatial_bin:int=1
    density_quantile:float=.72
    min_repeated_count:int=2
    min_area:float=10.
    max_candidates:int=20
    enable_partial_arcs:bool=True
    frequency_window_us:int=160000
    frequency_update_steps:int=4
    frequency_min_hz:float=60.
    frequency_max_hz:float=1200.
    blade_count:int=2
    max_buffer_events:int=450000
    max_frequency_samples:int=5000
    track_expiry_s:float=.3
    max_coast_s:float=.16
    normal_process_sigma:float=.18
    angular_rate_decay:float=3.
    nominal_gravity_px_s2:float=200.
    max_acceleration_px_s2:float=2500.
    adaptation_delay_s:float=.20
    adaptation_forgetting:float=.999
    image_up:tuple=(0.,-1.)
    opencv_threads:int=1
    horizons:tuple=(.1,.2,.4)
    # Optional real camera calibration. Absence selects weak-perspective cues.
    camera_matrix:list|None=None
    distortion:list|None=None
    metric_mass_kg:float|None=None
    metric_thrust_coefficient:float|None=None
    metric_diagonal_m:float|None=None

    def __post_init__(self):
        if self.step_us<=0 or self.blade_count<1 or self.spatial_bin<1:
            raise ValueError('Invalid time step, blade count or spatial bin')
        if self.distortion is not None and self.camera_matrix is None:
            raise ValueError('Distortion correction requires the real camera matrix')

    @classmethod
    def load(cls,path=None):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8'))) if path else cls()

    def save(self,path):
        Path(path).write_text(json.dumps(asdict(self),indent=2),encoding='utf-8')
