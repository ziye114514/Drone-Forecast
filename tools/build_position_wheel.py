"""Offline build with workspace-local temp files and bounded console output."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT.parent/'output/portable_position_core_20261025'
temp=OUT/'build_temp';temp.mkdir(exist_ok=True);wheels=OUT/'wheels';wheels.mkdir(exist_ok=True)
env=os.environ.copy();env.update(TEMP=str(temp),TMP=str(temp),PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
command=[sys.executable,'-X','utf8','-m','pip','--disable-pip-version-check','wheel','--no-index','--no-deps',
         '--no-build-isolation','--wheel-dir',str(wheels),str(ROOT)]
log=OUT/'wheel_build.log'
with log.open('w',encoding='utf-8') as stream:
    run=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
if run.returncode:
    print(log.read_text(encoding='utf-8',errors='replace')[-2500:]);raise SystemExit(run.returncode)
files=list(wheels.glob('event_drone_forecast-0.3.1-*.whl'));assert len(files)==1,files
wheel=files[0];relocated=OUT/'relocated_wheel';relocated.mkdir(exist_ok=False)
with zipfile.ZipFile(wheel) as z:
    assert z.testzip() is None
    names=z.namelist()
    assert not any('/research/' in n or '__pycache__' in n or n.endswith(('.raw','.onnx','.dll','.pyd','.pyc')) for n in names)
    for vendor in ['filterpy_linear_upstream','filterpy_process_upstream']:
        assert f'event_drone_forecast/_vendor/{vendor}/LICENSE' in names
        assert f'event_drone_forecast/_vendor/{vendor}/UPSTREAM_SNAPSHOT.json' in names
    z.extractall(relocated)
report=dict(wheel=str(wheel),sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),bytes=wheel.stat().st_size,
    files=len(names),relocated_root=str(relocated),licenses_included=True,
    research_or_private_config_or_RAW_or_weights_included=False,build_requires_network=False,
    limitation='This packages the existing package and current position core. Full RAW source launcher/CUDA worker still require the checkout, SDK and weights.')
(OUT/'wheel_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
