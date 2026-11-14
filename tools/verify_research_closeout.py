"""Bounded packaging checks only; no RAW/GPU experiments or performance claims."""
from pathlib import Path
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parent
DELIVERY=WORK/'output/github_research_closeout_20261105'
BASE=WORK/'output/github_release_update_20261028/event-drone-forecast'
OUT=DELIVERY/'event-drone-forecast'
VERIFY=DELIVERY/'verification'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def nested_readme(s):
    return re.sub(r'(?P<pre>!?\[[^\]]*\]\()(?P<url>[^)\s]+)(?P<end>\))',
        lambda m:m.group(0) if '://' in m['url'] or m['url'].startswith('#') else m['pre']+'event-drone-forecast/'+m['url']+m['end'],s)


if __name__=='__main__':
    VERIFY.mkdir(exist_ok=True)
    legacy=OUT/'docs/LEGACY_RAW_README_20261028.md'
    original=(BASE/'README.md').read_text('utf-8')
    legacy.write_text(re.sub(r'(?P<pre>!?\[[^\]]*\]\()(?P<url>[^)\s]+)(?P<end>\))',
        lambda m:m.group(0) if '://' in m['url'] or m['url'].startswith('#') else m['pre']+'../'+m['url']+m['end'],original),'utf-8')
    # No math/runtime edits to the old RAW base are allowed.
    changed=[]
    for p in BASE.rglob('*'):
        if p.is_file() and sha(p)!=sha(OUT/p.relative_to(BASE)):
            changed.append(p.relative_to(BASE).as_posix())
    assert set(changed)<= {'README.md','NOTICE.md','SOURCE_PROVENANCE.json','SOURCE_MANIFEST.json','.github/workflows/synthetic.yml'},changed
    q=OUT/'src/event_drone_forecast/rotor_observation'
    assert sha(q/'five_circle_model.py')==sha(ROOT/'core/five_circle_model.py')
    assert (q/'evidence.py').read_text('utf-8').replace('from .five_circle_model import','from five_circle_model import')==(ROOT/'core/rotor_evidence.py').read_text('utf-8')
    missing=[];links=0;private=[];compiled=0
    forbidden={'.raw','.onnx','.pt','.pth','.dll','.pyd','.pdf','.pyc'}
    for p in OUT.rglob('*'):
        if not p.is_file():continue
        assert p.suffix.lower() not in forbidden,p.name
        if p.suffix=='.py':ast.parse(p.read_text('utf-8-sig'));compiled+=1
        if p.suffix.lower() in ('.py','.md','.json','.jsonl','.toml','.txt','.yml','.yaml'):
            s=p.read_text('utf-8-sig')
            if re.search(r'(?:C:[/\\]Users[/\\]子夜|D:[/\\]drone forecast|E:[/\\]dataset)',s,re.I):private.append(str(p.relative_to(OUT)))
        if p.suffix=='.md':
            for target in re.findall(r'!?\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)',p.read_text('utf-8')):
                if '://' in target or target.startswith('#'):continue
                links+=1
                if not (p.parent/target.split('#')[0]).resolve().exists():missing.append([p.relative_to(OUT).as_posix(),target])
    assert not missing,missing
    assert not private,private
    # Seal a fresh manifest, sync web batches, and rebuild the ZIP after the link fix.
    manifest=dict(snapshot='research-snapshot-20261105',files=[dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='SOURCE_MANIFEST.json'])
    (OUT/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2),'utf-8')
    readme=nested_readme((OUT/'README.md').read_text('utf-8'))
    (DELIVERY/'ROOT_README_FOR_NESTED.md').write_text(readme,'utf-8')
    (DELIVERY/'root_files_for_nested_layout/README.md').write_text(readme,'utf-8')
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file()]
    for i,p in enumerate(files):
        dest=DELIVERY/f'网页上传第{i//90+1}批/event-drone-forecast'/p.relative_to(OUT)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    archive=DELIVERY/'event-drone-forecast-research-snapshot-20261105.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,'event-drone-forecast/'+p.relative_to(OUT).as_posix())
    (DELIVERY/(archive.name+'.sha256')).write_text(sha(archive)+'  '+archive.name+'\n','utf-8')
    extracted=VERIFY/'extracted'
    if extracted.exists():raise FileExistsError('Fresh-extraction check already attempted; inspect its results, do not loop silently.')
    extracted.mkdir()
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for n in z.namelist():assert (extracted/n).resolve().is_relative_to(extracted.resolve())
        z.extractall(extracted)
    source=extracted/'event-drone-forecast';env=os.environ.copy();env.pop('PYTHONPATH',None)
    results=[]
    cases=[('rotor_mechanisms',['-I','tools/check_rotor_observation.py']),
           ('rotor_demo',['-I','rotor_demo.py','--output',str(VERIFY/'rotor_demo.jsonl')]),
           ('legacy_help',['-I','forecast.py','--help'])]
    for name,args in cases:
        run=subprocess.run([sys.executable,*args],cwd=source,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=45)
        (VERIFY/(name+'.log')).write_text(run.stdout+run.stderr,'utf-8')
        results.append(dict(name=name,exit_code=run.returncode))
        if run.returncode:
            (VERIFY/'failed_checks.json').write_text(json.dumps(results,indent=2),'utf-8')
            raise RuntimeError(name+' failed; fix the concrete packaging defect, no experiments')
    records=[json.loads(s) for s in (VERIFY/'rotor_demo.jsonl').read_text('utf-8').splitlines()]
    counts={}
    for row in records:counts[row['layout']['model']]=counts.get(row['layout']['model'],0)+1
    assert len(records)==120 and counts.get('current_four_quad')==20 and counts.get('recent_layout_shared_translation')==4,counts
    assert all(row['layout']['current_center'] is None for row in records if row['layout']['model']=='side_axis_without_full_center')
    details=dict(base_runtime_files_unchanged=True,changed_base_document_or_manifest_files=changed,
                 source_import_adaptation_only=True,python_syntax_files=compiled,local_markdown_links=links,
                 missing_links=missing,private_workspace_paths=private,forbidden_binary_or_dataset_files=False,
                 isolated_extracted_checks=results,example_frames=120,example_models=counts,
                 actual_RAW_or_GPU_started=False,clean_machine_deployment_proven=False,
                 current_force_prediction_gain_proven=False,archive_sha256=sha(archive))
    (VERIFY/'checks.json').write_text(json.dumps(details,ensure_ascii=False,indent=2),'utf-8')
    delivery=json.loads((DELIVERY/'delivery.json').read_text('utf-8'))
    delivery.update(smoke_verification_pending=False,verification=str(VERIFY/'checks.json'),archive_sha256=sha(archive))
    (DELIVERY/'delivery.json').write_text(json.dumps(delivery,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(details,ensure_ascii=False))
