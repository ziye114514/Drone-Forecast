"""One short three-window RAW pilot; serial CUDA, no parameter sweep."""
from pathlib import Path
import json
import subprocess
import sys
import argparse

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--side-periodic',action='store_true');a=p.parse_args()
OUT=a.output.resolve()
OUT.mkdir(exist_ok=False)
cases=[('clear','E:/dataset/train/6/media/event.raw',22.,2.),
       ('side','E:/dataset/train/6/media/event.raw',110.,4.),
       ('train2','E:/dataset/train/2/aligned/event.raw',40.,4.)]
for name,raw,start,duration in cases:
    cmd=[sys.executable,'-X','utf8',str(ROOT/'core/forecast.py'),'run','--config',str(ROOT/'configs/local.json'),
         '--profile','live','--raw',raw,'--output',str(OUT/name),'--start',str(start),
         '--duration',str(duration),'--preview','--verify-frames']
    # Demonstrate continuous native-frequency feeding in just the side window.
    if name=='side' and a.side_periodic:cmd.append('--periodic-axis')
    with (OUT/(name+'.log')).open('x',encoding='utf-8') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,
                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    report=json.loads((OUT/name/'completion.json').read_text(encoding='utf-8'))
    print(json.dumps(dict(case=name,summary=report)),flush=True)
