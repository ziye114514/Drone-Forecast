"""Frozen42 RAW pipeline with independent per-component clocks only."""
from pathlib import Path
import argparse,datetime,hashlib,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT.parent/'output/stage43_latency_profile_20261019'
CASES={'train3':dict(raw='E:/dataset/train/3/aligned/event.raw',start=25.,duration=2.),'side6':dict(raw='E:/dataset/train/6/media/event.raw',start=108.,duration=2.),'negative5':dict(raw='E:/dataset/train/5/media/event.raw',start=0.,duration=1.)}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sources():
    result=load(ROOT.parent/'output/stage42_raw_body_bridge_20261018/protocol.json')['source_hashes'].copy()
    for name,value in result.items():assert sha(ROOT/name)==value,name
    for name in ['profile43.py','worker_profile43.py']:result['research/stage43/'+name]=sha(Path(__file__).with_name(name))
    return result
def freeze():
    assert not (OUT/'protocol.json').exists();save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),cases=CASES,
        design='Original42 source held unchanged. Instrument copies in memory at explicitly checked unique string anchors, preserving operation order/model/PNG/current state; separate clocks for RAW next, geometry, SDK filter, render, PNG encode, IPC/model, history, JSON and preview. Includes real child GPU worker batch1 and source50Hz. No GT read and no speed claim before measurement.',
        references=['https://docs.ultralytics.com/modes/predict/','https://docs.ultralytics.com/modes/export/'],
        checks='Compare train3 first100 full outputs to42, keep float differences if any. Old/default/protected paths unchanged; same fixed best40 SHA. Timing is current local workstation, not hardware-general SOTA benchmark.'))
    print('Stage43 profile frozen',flush=True)
def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'];return p

def instrument():
    path=ROOT/'run_body_forecast.py';source=path.read_text(encoding='utf-8')
    changes=[
      ("str(ROOT/'research/stage42/worker42.py')","str(ROOT/'research/stage43/worker_profile43.py')"),
      ("else dict(t=t,detected=False,rotors=[])\n                noise.process_events", "else dict(t=t,detected=False,rotors=[]);mark_geometry=time.perf_counter()\n                noise.process_events"),
      ("noise.process_events(e,buf);clean=buf.numpy().copy()", "noise.process_events(e,buf);clean=buf.numpy().copy();mark_noise=time.perf_counter()"),
      ("BaseFrameGenerationAlgorithm.generate_frame(clean,frame);ok,png=", "BaseFrameGenerationAlgorithm.generate_frame(clean,frame);mark_render=time.perf_counter();ok,png="),
      ("cv2.imencode('.png',frame);assert ok", "cv2.imencode('.png',frame);assert ok;mark_encode=time.perf_counter()"),
      ("result=json.loads(line);assert result['id']==n", "result=json.loads(line);assert result['id']==n;mark_receive=time.perf_counter()"),
      ("f.write(json_text(row)+'\\n');n+=1", "write_started=time.perf_counter();f.write(json_text(row)+'\\n');n+=1;profile_samples.append(dict(t=t,raw_next_ms=decode_stats['last_ms'],geometry_ms=(mark_geometry-tick)*1000,noise_ms=(mark_noise-mark_geometry)*1000,render_ms=(mark_render-mark_noise)*1000,png_encode_ms=(mark_encode-mark_render)*1000,ipc_model_ms=(mark_receive-mark_encode)*1000,history_ms=(write_started-mark_receive)*1000,json_write_ms=(time.perf_counter()-write_started)*1000,parent_processing_ms=elapsed,worker_inference_ms=result['inference_ms'],worker_profile=result['profile']));preview_started=time.perf_counter()"),
      ("video.write(frame)","video.write(frame);profile_samples[-1]['preview_ms']=(time.perf_counter()-preview_started)*1000")]
    for old,new in changes:assert source.count(old)==1,old;source=source.replace(old,new)
    namespace={'__name__':'profiled42','__file__':str(path),'profile_samples':[],'decode_stats':{'last_ms':0}}
    exec(compile(source,str(path),'exec'),namespace);original=namespace['raw_chunks']
    def chunks(*args,**kwargs):
        iterator=iter(original(*args,**kwargs))
        while True:
            began=time.perf_counter()
            try:value=next(iterator)
            except StopIteration:return
            namespace['decode_stats']['last_ms']=(time.perf_counter()-began)*1000;yield value
    namespace['raw_chunks']=chunks;return namespace

def run():
    verify()
    for case,cfg in CASES.items():
        pipeline=instrument();pipeline['run'](cfg['raw'],OUT/case,cfg['start'],cfg['duration'],preview=True)
        samples=pipeline['profile_samples'];save(OUT/case/'component_times.json',samples)
        summary={key:dict(median_ms=float(np.median([x[key] for x in samples])),p95_ms=float(np.percentile([x[key] for x in samples],95)),sum_s=float(sum(x[key] for x in samples)/1000)) for key in samples[0] if key.endswith('_ms')}
        save(OUT/case/'component_summary.json',summary);print(case,json.dumps(summary),flush=True)
    verify()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);globals()[p.parse_args().action]()
