"""Diagnose native20 baseline read-window regression before continuous testing."""
import json
from common41 import *
from windows21 import Observer
from event_drone_forecast.raw_io import open_raw,raw_chunks
sys.path.insert(0,str(ROOT/'research/stage39'))
from audit39 import difference
def main():
    old={round(r['t'],8):r['methods']['native20']['observation'] for r in map(json.loads,(ROOT.parent/'output/stage21_native_windows_20261018/clear6/observations.jsonl').read_text(encoding='utf-8').splitlines())};results=[]
    for warm in [False,True]:
        reader=open_raw(CASES['clear6']['raw']);h,w=reader.get_size();front=Observer(w,h,'native20')
        if warm:reader.seek_time(21990000);reader.load_delta_t(10000)
        for t,e in raw_chunks(reader,22.,.1,20000):
            o=front.step(e,t)['observation'];delta=difference(o,old[round(t,8)])
            results.append(dict(warm=warm,t=t,event_count=len(e),differences=delta))
    save(OUT/'baseline_failure_audit.json',results);print(json.dumps(results),flush=True)
if __name__=='__main__':main()
