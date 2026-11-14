"""Only a standard-library scheduler during NN/RAW inference."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT.parent/'output/stage137_fast_geometry_20261024'
def main():
    for name,raw,start,duration,preview in [('natural6','E:/dataset/train/6/media/event.raw',22,8,True),('side50','E:/dataset/train/6/media/event.raw',110,1,False),('negative5','E:/dataset/train/5/media/event.raw',0,.6,False)]:
        args=[sys.executable,str(ROOT/'core/run_optimized_forecast.py'),'--raw',raw,'--output',str(OUT/name),'--start',str(start),'--duration',str(duration),'--verify-frames']
        if preview:args.append('--preview')
        with (OUT/(name+'.log')).open('x',encoding='utf-8') as stream:subprocess.run(args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        print(name+' inference complete',flush=True)
    # Checks/plotting load numerical libraries only after inference workers exit.
    for name in ['legacy100','natural6','side50','negative5']:subprocess.run([sys.executable,str(ROOT/'research/stage137/check137.py'),name],cwd=ROOT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
if __name__=='__main__':main()
