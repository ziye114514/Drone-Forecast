"""Verify the new, relocated preview; do not touch the frozen alpha.1 release."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,zipfile,re,ast,urllib.parse
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'output/github_release_update_20261028'
PUBLIC=BASE/'event-drone-forecast';ARCHIVE=BASE/'event-drone-forecast-v0.4.0-alpha.2.zip'
VERIFY=BASE/'verification'
if not VERIFY.exists():
    VERIFY.mkdir()
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        for name in z.namelist():
            assert (VERIFY/name).resolve().is_relative_to(VERIFY.resolve()) and '..' not in Path(name).parts
        z.extractall(VERIFY)
EXTRACTED=VERIFY/'event-drone-forecast'
records=json.loads((VERIFY/'state.json').read_text('utf-8')) if (VERIFY/'state.json').exists() else []
env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
temporary=VERIFY/'tmp';temporary.mkdir(exist_ok=True)
env['TMP']=env['TEMP']=str(temporary)
env['EDF_NUMBA_PATH']=str(ROOT/'.cache/numba_cpu_064')
env['EDF_SDK_ACTIVITY_PATH']=str(VERIFY/'adapter')
env['PYTHONPATH']=os.pathsep.join(p for p in env.get('PYTHONPATH','').split(os.pathsep) if p and not Path(p).resolve().is_relative_to(ROOT))
def run(name,args,cwd=EXTRACTED):
    if any(r['name']==name and r['exit_code']==0 for r in records):
        print(name,'previously passed',flush=True);return (VERIFY/(name+'.log')).read_text('utf-8')
    result=subprocess.run(args,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    (VERIFY/(name+'.log')).write_text(result.stdout,'utf-8')
    records[:]=[r for r in records if r['name']!=name]
    records.append(dict(name=name,exit_code=result.returncode))
    (VERIFY/'state.json').write_text(json.dumps(records,indent=2),'utf-8')
    print(name,result.returncode,flush=True)
    if result.returncode:raise RuntimeError(name+' failed; inspect verification log')
    return result.stdout
manifest=json.loads((EXTRACTED/'SOURCE_MANIFEST.json').read_text('utf-8'))
for row in manifest['files']:
    assert hashlib.sha256((EXTRACTED/row['path']).read_bytes()).hexdigest()==row['sha256']
# Relative doc links and extracted arithmetic definitions.
links=0
for p in EXTRACTED.rglob('*.md'):
    for target in re.findall(r'!?\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)',p.read_text('utf-8')):
        if '://' in target or target.startswith('#'):continue
        assert (p.parent/urllib.parse.unquote(target.split('#')[0])).exists(),(p.name,target)
        links+=1
provenance=json.loads((EXTRACTED/'SOURCE_PROVENANCE.json').read_text('utf-8'));definitions=0
for item in provenance['files']:
    if item['mode']!='AST_verified_definition_extraction':continue
    tree=ast.parse((EXTRACTED/item['destination']).read_text('utf-8'))
    for expected in item['definitions']:
        nodes=[]
        for n in tree.body:
            name=getattr(n,'name',None)
            if isinstance(n,ast.Assign):name=next((t.id for t in n.targets if isinstance(t,ast.Name)),None)
            if name==expected['name']:nodes.append(n)
        assert len(nodes)==1
        assert hashlib.sha256(ast.dump(nodes[0],include_attributes=False).encode()).hexdigest()==expected['ast_sha256']
        definitions+=1
run('help',[sys.executable,'forecast.py','--help'])
run('candidate_help',[sys.executable,'forecast.py','predict','--help'])
run('adapter_help',[sys.executable,'tools/build_sdk_activity.py','--help'])
run('base',[sys.executable,'forecast.py','doctor','--base-only'])
run('synthetic',[sys.executable,'-m','unittest','discover','-s','tests','-v'])
run('demo',[sys.executable,'forecast.py','demo','--output',str(VERIFY/'demo')])
run('wheel',[sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',str(VERIFY/'wheel'),str(EXTRACTED)])
wheel=next((VERIFY/'wheel').glob('*.whl'))
run('wheel_install',[sys.executable,'-m','pip','install','--no-deps','--no-compile','--target',str(VERIFY/'installed'),str(wheel)])
snippet="import sys;sys.path.insert(0,"+repr(str(VERIFY/'installed'))+");import event_drone_forecast;from event_drone_forecast.position import LiveMotion;assert event_drone_forecast.__version__=='0.4.0a2';print(LiveMotion(30.))"
run('installed_core',[sys.executable,'-I','-c',snippet],cwd=VERIFY)
run('adapter_build',[sys.executable,'tools/build_sdk_activity.py','--sdk-root','D:/Prophesee',
    '--pybind-include',str(ROOT/'.cache/native_build_pybind/pybind11/include'),'--output',env['EDF_SDK_ACTIVITY_PATH']])
gpu='C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe'
weights=ROOT/'event_drone_forecast/models/live_position_accumulated_v1.onnx';sha=hashlib.sha256(weights.read_bytes()).hexdigest()
options=['--sdk-python',sys.executable,'--gpu-python',gpu,'--detector-weights',str(weights),'--detector-sha256',sha,'--cache-dir',str(VERIFY/'cache')]
run('SDK_GPU_contract',[sys.executable,'forecast.py','doctor']+options)
prediction=[sys.executable,'forecast.py','predict']+options+['--raw','E:/dataset/train/6/media/event.raw',
    '--output',str(VERIFY/'raw_smoke'),'--start','26','--duration','1.2','--profile','candidate','--sdk-avx2','--no-frame-verification']
dry=json.loads(run('candidate_dry_run',prediction+['--dry-run']))
assert not (VERIFY/'raw_smoke').exists()
for flag in ['--target-candidate','--rigid-motion','--tangent-disturbance','--structure-guided-nn','--sdk-avx2']:assert flag in dry['command']
assert '--with-kinematic-control' not in dry['command']
baseline=list(prediction);baseline[baseline.index('candidate')]='baseline'
base=json.loads(run('baseline_dry_run',baseline+['--dry-run']))
assert '--target-candidate' not in base['command']
run('relocated_candidate_RAW',prediction)
smoke=json.loads((VERIFY/'raw_smoke/summary.json').read_text('utf-8'))
protocol=json.loads((VERIFY/'raw_smoke/protocol.json').read_text('utf-8'))
assert smoke['frames']==36 and smoke['raw_unchanged'] and protocol['physical_only'] and protocol['finished']
assert protocol['runtime']['detector_sha256']==sha and not any('research/' in k for k in protocol['sources'])
checks=dict(version='0.4.0-alpha.2',checks=records,manifest_all_files_matched=True,zip_integrity=True,
    source_definitions_verified=definitions,relative_document_links_verified=links,synthetic_tests_passed=4,
    relocated=True,wheel_core_built_installed=True,adapter_built_from_exported_source=True,
    candidate_and_baseline_commands_checked=True,RAW_smoke=dict(source_ticks=smoke['frames'],
        current_body_observations=smoke['fresh_image_observations'],valid_forecasts=smoke['valid_forecasts'],
        raw_unchanged=smoke['raw_unchanged'],physical_only=True,labels_read=False,
        reference_model_sha256=sha),verified_on_existing_machine=True,independent_accuracy_or_clean_machine_claim=False,
    GitHub_CI_executed=False,published=False)
(VERIFY/'verification.json').write_text(json.dumps(checks,indent=2),'utf-8')
print(json.dumps(checks),flush=True)
