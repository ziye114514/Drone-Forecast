"""Low-memory orchestration: no NumPy/build namespaces waiting beside RAW.

Inference and verification use the same frozen entry/protocol. The heavyweight
checker is a fresh process after each prediction process and workers exit.
"""
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT.parent/'output/stage128_RAW_independent_support_20261023'
def main():
    protocol=json.loads((OUT/'protocol128.json').read_text('utf-8'))
    for case in ['extension2','negative5','prefix_natural6']:
        c=protocol['cases'][case];args=[sys.executable,str(ROOT/'core/run_independent_support_forecast.py'),'--raw',c['raw'],'--output',str(OUT/'raw'/case),'--start',str(c['start']),'--duration',str(c['duration']),'--verify-frames']
        if not case.startswith('prefix'):args.append('--preview')
        print('PREDICT',case,flush=True)
        with (OUT/(case+'_parent.log')).open('x',encoding='utf-8') as log:
            subprocess.run(args,check=True,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        print('VERIFY',case,flush=True)
        subprocess.run([sys.executable,str(ROOT/'research/stage128/study128.py'),'check','--case',case],check=True,cwd=ROOT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
if __name__=='__main__':main()
