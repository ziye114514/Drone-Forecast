"""Run unchanged EEPPR core/utils with a recorded-SDK compatibility bridge.

Python 3.13 has CUDA PyTorch; the installed SDK is Python 3.11 only. Event
decoding is performed separately with that SDK. This bridge only substitutes
SDK RAW iteration and rectangular filtering using decoded original events.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time,types
import numpy as np
root=Path(__file__).resolve().parents[2];out=root/'output/paper_reproduction_v030/eeppr_benchmark'
sys.path.insert(0,str(root/'tmp/eeppr_dependencies'));sys.path.insert(0,str(root/'EEPPR-main/method'))
parser=argparse.ArgumentParser();parser.add_argument('--name',choices=['velcro_front','velcro_side'],required=True)
args=parser.parse_args();meta=json.loads((out/f'{args.name}_input.json').read_text());events=np.load(out/f'{args.name}.npy',mmap_mode='r')

class RecordedReader:
    width=meta['width'];height=meta['height']
    def load_delta_t(self,dt):return np.array(events[events['t']<dt],copy=True)
    def reset(self):pass

class RecordedIterator:
    def __init__(self,path,delta_t,max_duration):self.dt=delta_t;self.end=max_duration
    def __iter__(self):
        ts=events['t'];bounds=np.searchsorted(ts,np.arange(0,self.end+self.dt,self.dt))
        for a,b in zip(bounds[:-1],bounds[1:]):yield np.array(events[a:b],copy=True)

class Buffer:
    def numpy(self):return self.events

class ROI:
    def __init__(self,x0,y0,x1,y1,output_relative_coordinates=False):self.coords=x0,y0,x1,y1
    def get_empty_output_buffer(self):return Buffer()
    def process_events(self,e,buffer):
        x0,y0,x1,y1=self.coords
        buffer.events=e[(e['x']>=x0)&(e['x']<=x1)&(e['y']>=y0)&(e['y']<=y1)]

sdk=types.ModuleType('metavision_sdk_core');sdk.RoiFilterAlgorithm=ROI
sdk.BaseFrameGenerationAlgorithm=type('UnusedDisplay',(),{})
core=types.ModuleType('metavision_core');io=types.ModuleType('metavision_core.event_io');io.EventsIterator=RecordedIterator
sys.modules.update(metavision_sdk_core=sdk,metavision_core=core);sys.modules['metavision_core.event_io']=io
import torch
from eeppr import EEPPR
torch.set_num_threads(1)
author_args=argparse.Namespace(file=meta['raw'],output_dir=str(out),aggreg_t=100,read_t=meta['read_t'],win_size=45,
    event_count=1800,roi_coords=meta['roi'],aggreg_fn='median',decimals=1,viz_corr_resp=False,
    device='cuda:0' if torch.cuda.is_available() else 'cpu',log='WARNING',verbose=False)
started=time.perf_counter();result,freq,corr=EEPPR(author_args,str(out),RecordedReader()).run();elapsed=time.perf_counter()-started
np.save(out/f'{args.name}_window_estimates.npy',freq)
result=dict(sequence=args.name,gt_freq_hz=meta['gt_freq'],estimated_hz=float(result),
    absolute_error_hz=abs(float(result)-meta['gt_freq']),relative_error_percent=100*abs(float(result)/meta['gt_freq']-1),
    valid_windows=int((freq>0).sum()),total_windows=freq.size,wall_s=elapsed,device=author_args.device,
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'EEPPR-main/method').glob('*.py')},
    core_modified=False,sdk_io_replaced_by_exact_decoded_events=True,
    protocol='Author default 1s public sequence, 100us bins, 45px windows, 1800 template events, provided full ROI, median.',
    limitation='Stationary public periodic target, not flight, not rotor pose or future drone position truth.')
(out/f'{args.name}_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
