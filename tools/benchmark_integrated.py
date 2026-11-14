"""Single-process final-version timing in the exact v0.1 high-rate scene."""
from pathlib import Path
import argparse,json,sys,time
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from event_drone_forecast.config import Config
from event_drone_forecast.raw_io import open_raw
from event_drone_forecast.perception import RotorPerception
from event_drone_forecast.motion import ForecastBank


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();cfg=Config.load(out/'frozen_config.json')
    begin=time.perf_counter();reader=open_raw('E:/dataset/train/6/media/event.raw');reader.seek_time(108800000)
    initialization=time.perf_counter()-begin;h,w=reader.get_size()
    perception=RotorPerception(w,h,cfg);bank=ForecastBank(cfg);latency=[];events_total=0
    for i in range(175):
        before=time.perf_counter();events=reader.load_delta_t(20000)
        observation,_=perception.detect(events,reader.current_time/1e6);bank.step(observation)
        elapsed=time.perf_counter()-before
        if i>=25:latency.append(elapsed);events_total+=len(events)
    result=dict(single_process=True,sensor_interval_s=[109.3,112.3],warmup_s=.5,measured_sensor_s=3.,
                measured_frames=len(latency),event_count=events_total,raw_open_and_seek_s=initialization,
                wall_processing_s=sum(latency),realtime_factor=3/sum(latency),
                median_ms=float(np.median(latency)*1000),p95_ms=float(np.percentile(latency,95)*1000),
                max_ms=float(max(latency)*1000),fraction_within_20ms=float(np.mean(np.array(latency)<=.02)),
                opencv_threads=cv2.getNumThreads(),
                timing_scope='SDK decode + perception + all 5 models; initialization, warmup and output writing excluded.',
                comparison_caveat='v0.1 used 16 OpenCV threads; final version uses config.opencv_threads. Whole-system configuration comparison, not isolated algorithm speedup.')
    (out/'single_process_timing.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
