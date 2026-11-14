"""Small serial deployment comparisons; no heavy inference parent."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT.parent/'output/stage138_shared_objective_20261024'
def main():
    for name,start,duration,backend,raw,preview in [('fast100',22,2,'fast','E:/dataset/train/6/media/event.raw',False),('natural6',22,8,'shared','E:/dataset/train/6/media/event.raw',True),('side50',110,1,'shared','E:/dataset/train/6/media/event.raw',False),('negative5',0,.6,'shared','E:/dataset/train/5/media/event.raw',False)]:
        args=[sys.executable,str(ROOT/'run_optimized_forecast.py'),'--raw',raw,'--output',str(OUT/name),'--start',str(start),'--duration',str(duration),'--geometry-backend',backend,'--verify-frames']
        if preview:args.append('--preview')
        with (OUT/(name+'.log')).open('x',encoding='utf-8') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        print(name+' complete',flush=True)
if __name__=='__main__':main()
