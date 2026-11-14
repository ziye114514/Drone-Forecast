"""Read-only native SDK stream diagnostics; no SDK files are changed."""
from pathlib import Path
import hashlib,json,time,sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'output/fred_transfer_20261029/raw_stream_diagnostic'


class Stream:
    def __init__(self,source,blocking=False):
        self.source=source;self.blocking=blocking;self.calls=0;self.empty=0;self.buffers=0;self.wall=0.
    def __getattr__(self,name):return getattr(self.source,name)
    def poll_buffer(self):
        before=time.perf_counter()
        answer=self.source.wait_next_buffer() if self.blocking else self.source.poll_buffer()
        self.wall+=time.perf_counter()-before;self.calls+=1;self.empty+=int(answer==0)
        return answer
    def get_latest_raw_data(self):self.buffers+=1;return self.source.get_latest_raw_data()


class Decoder:
    def __init__(self,source):self.source=source;self.calls=0;self.wall=0.
    def __getattr__(self,name):return getattr(self.source,name)
    def decode(self,data):
        start=time.perf_counter();result=self.source.decode(data);self.calls+=1;self.wall+=time.perf_counter()-start
        return result


def sample(blocking=False,read_buffers=3,paced=False):
    from metavision_hal import DeviceDiscovery,RawFileConfig
    from metavision_core.event_io import EventsIterator
    raw=Path('E:/dataset/train/2/aligned/event.raw');config=RawFileConfig()
    config.do_time_shifting=True;config.n_events_to_read=100000;config.build_index=False
    config.n_read_buffers=read_buffers
    device=DeviceDiscovery.open_raw_file(str(raw),config)
    begin=(40000000//33333)*33333;duration=180*33333
    iterator=EventsIterator.from_device(device,start_ts=begin,delta_t=33333,max_duration=duration,mode='delta_t',relative_timestamps=False)
    stream=Stream(iterator.reader.i_events_stream,blocking);decoder=Decoder(iterator.reader.i_events_stream_decoder)
    iterator.reader.i_events_stream=stream;iterator.reader.i_events_stream_decoder=decoder
    answer=[];before=time.perf_counter();cpu=time.process_time();previous=(0,0,0,0.,0.)
    for events in iterator:
        now=time.perf_counter();current=(stream.calls,stream.empty,stream.buffers,stream.wall,decoder.wall)
        delta=[a-b for a,b in zip(current,previous)]
        answer.append(dict(t=iterator.get_current_time()/1e6,events=len(events),
            sha256=hashlib.sha256(events.tobytes()).hexdigest(),wall_ms=(now-before)*1000,
            process_CPU_ms=(time.process_time()-cpu)*1000,poll_calls=delta[0],empty_polls=delta[1],
            buffers=delta[2],poll_wall_ms=delta[3]*1000,decode_wall_ms=delta[4]*1000))
        previous=current
        if paced:time.sleep(.025)
        before=time.perf_counter();cpu=time.process_time()
    return answer


def main():
    buffers='--buffers' in sys.argv
    folder=OUT.parent/'raw_buffer_diagnostic' if buffers else OUT
    folder.mkdir(exist_ok=False)
    modes=('buffers3','buffers8') if buffers else ('poll','blocking')
    for mode in modes:
        rows=sample(mode=='blocking',read_buffers=8 if mode=='buffers8' else 3,paced=buffers)
        (folder/f'{mode}.json').write_text(json.dumps(rows,indent=2),'utf-8')
        print(json.dumps(dict(mode=mode,frames=len(rows),slow=sorted(rows,key=lambda r:r['wall_ms'],reverse=True)[:3])),flush=True)
    a=json.loads((folder/f'{modes[0]}.json').read_text());b=json.loads((folder/f'{modes[1]}.json').read_text())
    exact=len(a)==len(b) and all((x['t'],x['events'],x['sha256'])==(y['t'],y['events'],y['sha256']) for x,y in zip(a,b))
    result=dict(native_event_bytes_and_cutoffs_exact=exact,SDK_files_unchanged=True,
        modes={k:dict(total_wall_ms=sum(r['wall_ms'] for r in rows),max_steady_wall_ms=max(r['wall_ms'] for r in rows[1:]),
            empty_polls=sum(r['empty_polls'] for r in rows),poll_calls=sum(r['poll_calls'] for r in rows),
            decode_wall_ms=sum(r['decode_wall_ms'] for r in rows)) for k,rows in zip(modes,(a,b))})
    (folder/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
