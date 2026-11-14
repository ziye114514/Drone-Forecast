"""Standard-library-only serial inference/check orchestration."""
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT.parent/'output/stage129_FOV_guard_20261024'
def main():
    protocol=json.loads((OUT/'protocol_raw129.json').read_text('utf-8'))
    for case,c in protocol['cases'].items():
        args=[sys.executable,str(ROOT/'core/run_FOV_guard_forecast.py'),'--raw',c['raw'],'--output',str(OUT/'raw'/case),'--start',str(c['start']),'--duration',str(c['duration']),'--verify-frames']
        with (OUT/(case+'_parent.log')).open('x',encoding='utf-8') as log:subprocess.run(args,check=True,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        subprocess.run([sys.executable,str(ROOT/'research/stage129/raw129.py'),'check','--case',case],check=True,cwd=ROOT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
if __name__=='__main__':main()
