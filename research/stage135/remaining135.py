"""Lightweight two-check deployment run, no full old control suite."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT.parent/'output/stage135_optimized_forecast_20261024'
def main():
    for name,raw,start,duration in [('prefix_natural6','E:/dataset/train/6/media/event.raw',22.,1.2),('negative5','E:/dataset/train/5/media/event.raw',0.,.6)]:
        args=[sys.executable,str(ROOT/'run_optimized_forecast.py'),'--raw',raw,'--output',str(OUT/'raw'/name),'--start',str(start),'--duration',str(duration),'--verify-frames']
        with (OUT/(name+'_parent.log')).open('x',encoding='utf-8') as log:subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        subprocess.run([sys.executable,str(ROOT/'research/stage135/check135.py'),'--case',name],cwd=ROOT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
if __name__=='__main__':main()
