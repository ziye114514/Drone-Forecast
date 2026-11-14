"""Single installed-ByteTrack mechanism; retain all current raw candidates."""
from pathlib import Path
import argparse,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
OUT=a.output.resolve();OUT.mkdir(exist_ok=False)
cases=[('train2','E:/dataset/train/2/aligned/event.raw',40.,6.),
       ('side','E:/dataset/train/6/media/event.raw',110.,4.),
       ('clear','E:/dataset/train/6/media/event.raw',22.,2.),
       ('negative','E:/dataset/train/5/media/event.raw',0.,1.)]
for name,raw,start,duration in cases:
    cmd=[sys.executable,'-X','utf8',str(ROOT/'forecast.py'),'run','--config',str(ROOT/'configs/local.json'),
         '--profile','live','--body-association','bytetrack','--publication-clock','independent','--raw',raw,'--output',str(OUT/name),
         '--start',str(start),'--duration',str(duration),'--preview','--verify-frames']
    with (OUT/(name+'.log')).open('x',encoding='utf-8') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    q=json.loads((OUT/name/'completion.json').read_text(encoding='utf-8'))
    print(json.dumps(dict(case=name,completion=q)),flush=True)
