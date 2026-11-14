"""Install own built wheel into a fresh target and test outside the checkout."""
from pathlib import Path
import argparse,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT.parent/'output/portable_position_core_20261025'
manifest=json.loads((OUT/'wheel_manifest.json').read_text(encoding='utf-8'))
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--target-name',default='installed')
args=parser.parse_args();assert args.target_name.replace('_','').isalnum()
target=OUT/args.target_name;assert not target.exists()
env=os.environ.copy();env.update(TEMP=str(OUT/'build_temp'),TMP=str(OUT/'build_temp'),PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
command=[sys.executable,'-X','utf8','-m','pip','--disable-pip-version-check','install','--no-index','--no-deps','--no-compile',
         '--target',str(target),manifest['wheel']]
log_path=OUT/('pip_install_'+args.target_name+'.log')
with log_path.open('w',encoding='utf-8') as log:
    result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
if result.returncode:
    print(log_path.read_text(encoding='utf-8')[-2000:]);raise SystemExit(result.returncode)
check=[sys.executable,'-I','-X','utf8',str(ROOT/'tools/check_position_core.py'),
       '--package-root',str(target),'--output',str(OUT/'pip_target_equivalence.json')]
subprocess.run(check,env=env,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
print('Own wheel installed in a fresh target; existing NumPy/SciPy reused, not a clean-machine RAW test.',flush=True)
