"""Single-process hot-path timing, separated from RAW open/seek and batch runs."""
from pathlib import Path
import json,os,platform,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from event_drone_forecast.raw_io import open_raw
from event_drone_forecast.paper_config import PaperConfig
from event_drone_forecast.rotor_sensing import PaperPerception,RotorSensing
from event_drone_forecast.paper_bank import PaperForecastBank
OUT=ROOT.parent/'output/paper_reproduction_v030'
cfg=PaperConfig.load(OUT/'frozen_config.json');raw=r'E:\dataset\train\6\media\event.raw'
opened=time.perf_counter();reader=open_raw(raw);reader.seek_time(109300000);open_seek=time.perf_counter()-opened
height,width=reader.get_size();front=PaperPerception(width,height,cfg);sensing=RotorSensing(cfg);bank=PaperForecastBank(cfg)
latencies=[];events_count=0;physics=0;started=time.perf_counter()
while reader.current_time<112300000 and not reader.is_done():
    clock=time.perf_counter();events=reader.load_delta_t(min(10000,112300000-reader.current_time));t=reader.current_time/1e6
    obs,total=front.detect(events,t);rotors=sensing.step(obs,front.history,t)
    obs.update({k:v for k,v in rotors.items() if k!='rotors'});obs['frequency_quality']=rotors['frequency_quality']
    obs['frequency_squared_relative']=rotors['thrust_ratio'] or 1.
    result=bank.step(obs,rotors);latencies.append(time.perf_counter()-clock);events_count+=len(events)
    physics+=int(bool(result and result['physics_active']))
elapsed=time.perf_counter()-started
result=dict(raw=raw,start_s=109.3,duration_s=3.,frames=len(latencies),event_count=events_count,physical_active_frames=physics,
    raw_open_seek_s=open_seek,hot_path_wall_s=elapsed,sensor_seconds_per_wall_second=3/elapsed,
    processing_median_ms=float(np.median(latencies)*1000),processing_p95_ms=float(np.percentile(latencies,95)*1000),
    fraction_within_10ms=float(np.mean(np.array(latencies)<=.01)),python=sys.version,platform=platform.platform(),
    logical_cpu_count=os.cpu_count(),opencv_threads=cfg.opencv_threads,
    includes='RAW decoding + detection + per-rotor periodicity + all eight motion branches',
    excludes='RAW open/seek (reported separately), preview, JSON/CSV writes, sensor hardware delay',
    interpretation='One workstation, one fixed difficult interval; not a broad latency guarantee. Other experiment workers have finished.')
(OUT/'single_process_timing.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
