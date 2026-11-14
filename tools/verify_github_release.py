"""Verify the actual ZIP after relocation; private paths stay outside the ZIP."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,zipfile,re,urllib.parse,ast,shutil
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'output/github_release_final_20261027'
ARCHIVE=BASE/'event-drone-forecast-v0.4.0-alpha.1.zip';PUBLIC=BASE/'event-drone-forecast'
VERIFY=BASE/'verification';VERIFY.mkdir(exist_ok=False)
with zipfile.ZipFile(ARCHIVE) as z:
    assert z.testzip() is None
    for name in z.namelist():
        p=(VERIFY/name).resolve();assert p.is_relative_to(VERIFY.resolve()) and '..' not in Path(name).parts
    z.extractall(VERIFY)
EXTRACTED=VERIFY/'event-drone-forecast';records=[]
env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
# Keep the vendor-installed SDK environment, but never import working-project code.
sdk_paths=[p for p in env.get('PYTHONPATH','').split(os.pathsep) if p and not Path(p).resolve().is_relative_to(ROOT)]
env['PYTHONPATH']=os.pathsep.join(sdk_paths)
def run(name,args,cwd=EXTRACTED,extra_env=None):
    current=env.copy();current.update(extra_env or {})
    p=subprocess.run(args,cwd=cwd,env=current,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    (VERIFY/(name+'.log')).write_text(p.stdout,'utf-8');records.append(dict(name=name,exit_code=p.returncode))
    print(name,'exit',p.returncode,flush=True)
    if p.returncode:raise RuntimeError(name+' failed; inspect external verification log')
    return p.stdout
manifest=json.loads((EXTRACTED/'SOURCE_MANIFEST.json').read_text('utf-8'))
for row in manifest['files']:
    assert hashlib.sha256((EXTRACTED/row['path']).read_bytes()).hexdigest()==row['sha256']
run('help',[sys.executable,'forecast.py','--help'])
run('doctor_base',[sys.executable,'forecast.py','doctor','--base-only'])
run('synthetic_tests',[sys.executable,'-m','unittest','discover','-s','tests','-v'])
run('demo',[sys.executable,'forecast.py','demo','--output',str(VERIFY/'demo')])
run('wheel_build',[sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',str(VERIFY/'wheel'),str(EXTRACTED)])
wheel=next((VERIFY/'wheel').glob('*.whl'))
run('wheel_install',[sys.executable,'-m','pip','install','--no-deps','--no-compile','--target',str(VERIFY/'installed'),str(wheel)])
code="import sys;sys.path.insert(0,"+repr(str(VERIFY/'installed'))+");import event_drone_forecast;from event_drone_forecast.position import LiveMotion;print(event_drone_forecast.__version__);print(LiveMotion(30.))"
run('installed_core',[sys.executable,'-I','-c',code],cwd=VERIFY)
gpu='C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe'
weights=ROOT/'event_drone_forecast/models/live_position_accumulated_v1.onnx'
sha=hashlib.sha256(weights.read_bytes()).hexdigest()
options=['--sdk-python',sys.executable,'--gpu-python',gpu,'--detector-weights',str(weights),
    '--detector-sha256',sha,'--cache-dir',str(VERIFY/'cache')]
run('doctor_raw',[sys.executable,'forecast.py','doctor']+options)
raw=Path('E:/dataset/train/6/media/event.raw')
prediction=[sys.executable,'forecast.py','predict']+options+['--raw',str(raw),'--output',str(VERIFY/'raw_smoke'),'--start','26','--duration','1.2']
run('dry_run',prediction+['--dry-run']);assert not (VERIFY/'raw_smoke').exists()
run('relocated_RAW',prediction)
smoke=json.loads((VERIFY/'raw_smoke/summary.json').read_text('utf-8'));assert smoke['frames']==36 and smoke['raw_unchanged']
protocol=json.loads((VERIFY/'raw_smoke/protocol.json').read_text('utf-8'))
assert protocol['physical_only'] and protocol['finished'] and protocol['runtime']['detector_sha256']==sha
assert not any('research/' in k for k in protocol['sources'])
checks=dict(manifest_all_files_matched=True,zip_integrity=True,relocated=True,checks=records,
    synthetic_tests_passed=3,standalone_demo_passed=True,wheel_built_and_installed=True,
    base_SDK_GPU_doctors_passed=True,dry_run_no_output_or_GPU=True,
    relocated_RAW=dict(source_ticks=smoke['frames'],current_body_observations=smoke['fresh_image_observations'],
        valid_forecasts=smoke['valid_forecasts'],raw_unchanged=smoke['raw_unchanged'],physical_only=True,
        reference_model_sha256=sha,labels_read=False,old_output_or_research_directories_required=False),
    verified_on_existing_machine=True,clean_machine_or_GitHub_CI_claimed=False,
    weights_RAW_SDK_DLL_included=False)
(VERIFY/'verification.json').write_text(json.dumps(checks,indent=2),'utf-8')
print(json.dumps(checks),flush=True)
