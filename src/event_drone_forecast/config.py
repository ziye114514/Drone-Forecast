from dataclasses import dataclass, asdict
import json
from pathlib import Path

@dataclass
class Config:
    step_us: int = 20000
    spatial_bin: int = 1
    min_pixel_count: int = 3
    density_quantile: float = .78
    min_blob_area: float = 12.
    max_blob_fraction: float = .08
    max_candidates: int = 24
    enable_partial_geometry: bool = True
    enable_line_geometry: bool = True
    opencv_threads: int = 1
    single_rotor_max_age_s: float = .3
    geometry_learning_min_quality: float = .35
    frequency_learning_min_quality: float = .35
    geometry_prior_max_age_s: float = .2
    frequency_window_us: int = 100000
    frequency_update_steps: int = 5
    frequency_min_hz: float = 60.
    frequency_max_hz: float = 1600.
    max_frequency_events: int = 350000
    position_sigma: float = 7.
    acceleration_sigma: float = 1800.
    learning_delay_s: float = .16
    forgetting_factor: float = .997
    disturbance_time_s: float = .35
    max_acceleration_px_s2: float = 3000.
    max_coast_s: float = .16
    reset_gap_s: float = .5
    horizons: tuple = (.1,.2,.4)

    @classmethod
    def load(cls, path=None):
        data=json.loads(Path(path).read_text(encoding='utf-8')) if path else {}
        unknown=set(data)-set(cls.__dataclass_fields__)
        if unknown: raise ValueError(f'Unknown configuration keys: {unknown}')
        return cls(**data)

    def save(self,path):
        Path(path).write_text(json.dumps(asdict(self),indent=2),encoding='utf-8')
