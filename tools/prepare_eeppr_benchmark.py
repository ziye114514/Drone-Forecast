"""Decode public benchmark with the real SDK, without changing author sources."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from event_drone_forecast.raw_io import open_raw
root=Path(__file__).resolve().parents[2];out=root/'output/paper_reproduction_v030/eeppr_benchmark';out.mkdir(exist_ok=True,parents=True)
config=json.loads((root/'EEPPR-main/dataset/config.json').read_text())['sequence_info']
for name in ('velcro_front','velcro_side'):
    info=config[name];path=root/'EEPPR-main/dataset'/info['raw_filepath'];reader=open_raw(path)
    e=np.concatenate([reader.load_delta_t(10000).copy() for _ in range(100)])
    np.save(out/f'{name}.npy',e)
    record=dict(name=name,raw=str(path),raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                width=reader.width,height=reader.height,event_count=len(e),roi=info['roi'],gt_freq=info['gt_freq'],read_t=1000000)
    (out/f'{name}_input.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(record,flush=True)
