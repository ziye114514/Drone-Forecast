"""Serial actual RAW runs with the confidence-conditioned motion fallback."""
from pathlib import Path
import hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT.parent/'output/stage140_visible_fallback_20261024'
def main():
    sources=[ROOT/'research/stage140/visible140.py',ROOT/'research/stage139/dense139.py',ROOT/'research/stage140/study140.py',ROOT/'research/stage139/study139.py',ROOT/'core/run_optimized_forecast.py']
    (OUT/'protocol140_RAW.json').write_text(json.dumps(dict(scope='140 cached development outcomes known; freeze before RAW deployment, not independent validation.',sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},inherited_cached_protocol='protocol139.json captures139 gate SHA and140 parameters through adapter; this record supplies explicit140 source provenance.'),ensure_ascii=False,indent=2),'utf-8')
    for name,raw,start,duration,preview in [('continuous6','E:/dataset/train/6/media/event.raw',110,9.5,True),('extension2','E:/dataset/train/2/aligned/event.raw',40,6,True),('prefix_natural6','E:/dataset/train/6/media/event.raw',22,1.2,False),('negative5_raw','E:/dataset/train/5/media/event.raw',0,.6,False)]:
        args=[sys.executable,str(ROOT/'core/run_optimized_forecast.py'),'--raw',raw,'--output',str(OUT/name),'--start',str(start),'--duration',str(duration),'--verify-frames']
        if preview:args.append('--preview')
        with (OUT/(name+'.log')).open('x',encoding='utf-8') as stream:subprocess.run(args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        print(name+' complete',flush=True)
if __name__=='__main__':main()
