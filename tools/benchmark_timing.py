"""Single-process steady-state timing; initialization/RAW seeking separately timed."""
from pathlib import Path
import sys,time,json
import numpy as np,cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from event_drone_forecast.raw_io import open_raw
from event_drone_forecast.perception import RotorPerception
from event_drone_forecast.motion import ForecastBank
from event_drone_forecast.config import Config

root=Path(__file__).resolve().parents[2]/'output/raw_only_project'
cfg=Config.load(root/'frozen_config.json')
start=time.perf_counter();reader=open_raw('E:/dataset/train/6/media/event.raw');reader.seek_time(108800000)
initialization=time.perf_counter()-start
h,w=reader.get_size();perception=RotorPerception(w,h,cfg);bank=ForecastBank(cfg)
latency=[];events_total=0
for i in range(175):
    before=time.perf_counter();events=reader.load_delta_t(20000)
    obs,total=perception.detect(events,reader.current_time/1e6);result=bank.step(obs)
    elapsed=time.perf_counter()-before
    if i>=25:latency.append(elapsed);events_total+=len(events)
summary=dict(single_process=True,raw='E:/dataset/train/6/media/event.raw',sensor_interval_s=[109.3,112.3],
    warmup_s=.5,measured_sensor_s=3.,measured_frames=len(latency),event_count=events_total,
    raw_open_and_seek_s=initialization,wall_processing_s=sum(latency),realtime_factor=3/sum(latency),
    median_ms=float(np.median(latency)*1000),p95_ms=float(np.percentile(latency,95)*1000),max_ms=float(max(latency)*1000),
    fraction_within_20ms=float(np.mean(np.array(latency)<=.02)),opencv_threads=cv2.getNumThreads(),
    timing_scope='SDK RAW decode + perception + all 5 motion models, no video/JSON disk output. Seeking and warmup excluded.',
    limitation='One short high-event-rate scene; not a general live-device timing guarantee.')
(root/'single_process_timing.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary),flush=True)
