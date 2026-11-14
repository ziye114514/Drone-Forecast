"""Relocated NN-release checks and reproduction parity, not new optimization."""
from pathlib import Path
import ast,hashlib,json,os,re,shutil,subprocess,sys,zipfile

ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
DELIVERY=WORK/'output/github_nn_main_20261105';OUT=DELIVERY/'event-drone-forecast'
VERIFY=DELIVERY/'verification_2';STUDY=WORK/'output/nn_rotor166_20261105'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def nested(s):
    return re.sub(r'(?P<a>!?\[[^\]]*\]\()(?P<b>[^)\s]+)(?P<c>\))',lambda m:m.group(0) if '://' in m['b'] or m['b'].startswith('#') else m['a']+'event-drone-forecast/'+m['b']+m['c'],s)


if __name__=='__main__':
    VERIFY.mkdir(exist_ok=True)
    missing=[];private=[];syntax=0;links=0
    for p in OUT.rglob('*'):
        if not p.is_file():continue
        assert p.suffix.lower() not in ('.raw','.onnx','.pt','.pth','.dll','.pyd','.pdf','.pyc'),p
        if p.suffix=='.py':ast.parse(p.read_text('utf-8-sig'));syntax+=1
        if p.suffix.lower() in ('.py','.md','.json','.jsonl','.toml','.txt','.yml','.cff','.bib'):
            t=p.read_text('utf-8-sig')
            if re.search(r'(?:C:[/\\]Users[/\\]子夜|D:[/\\]drone forecast|E:[/\\]dataset)',t,re.I):private.append(p.relative_to(OUT).as_posix())
        if p.suffix=='.md':
            for target in re.findall(r'!?\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)',p.read_text('utf-8')):
                if '://' in target or target.startswith('#'):continue
                links+=1
                if not (p.parent/target.split('#')[0]).resolve().exists():missing.append([p.relative_to(OUT).as_posix(),target])
    assert not missing,missing
    assert not private,private
    # No inherited manifest is allowed to describe the updated sources.
    manifest=dict(version='0.5.0-research.1',files=[dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='SOURCE_MANIFEST.json'])
    (OUT/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2),'utf-8')
    archive=DELIVERY/'event-drone-forecast-v0.5.0-research.1-NN-main.zip'
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file()]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,'event-drone-forecast/'+p.relative_to(OUT).as_posix())
    extracted=VERIFY/'extracted'
    if extracted.exists():raise FileExistsError('Inspect existing verification; use a named retry if a packaging defect is fixed')
    extracted.mkdir()
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for n in z.namelist():assert (extracted/n).resolve().is_relative_to(extracted.resolve())
        z.extractall(extracted)
    source=extracted/'event-drone-forecast';env=os.environ.copy();env.pop('PYTHONPATH',None);env['PYTHONIOENCODING']='utf-8'
    results=[]
    def run(name,args):
        r=subprocess.run([sys.executable,*args],cwd=source,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=60)
        (VERIFY/(name+'.log')).write_text(r.stdout+r.stderr,'utf-8');results.append(dict(name=name,exit_code=r.returncode))
        (VERIFY/'progress.json').write_text(json.dumps(results,indent=2),'utf-8')
        assert r.returncode==0,name+' failed; inspect log'
    run('nn_help',['-I','nn_forecast.py','--help'])
    run('nn_predict_dry_run',['-I','nn_forecast.py','predict','--raw','not_read.raw','--output','outputs/not_created','--dry-run'])
    dry=json.loads((VERIFY/'nn_predict_dry_run.log').read_text('utf-8'))
    assert '--realtime30' in dry['command'] and '--rotor-only' not in dry['command'] and not dry['additional166_rotor_assist']
    assert not (source/'outputs/not_created').exists()
    run('aux_mechanisms',['-I','tools/check_nn_aux.py'])
    parity={}
    for case in ['clear','side','train2']:
        for mode in ['baseline','rotor-assist','matched-zero']:
            result=VERIFY/f'{case}_{mode}.jsonl'
            run(f'{case}_{mode}',['-I','nn_assist_replay.py','--nn',str(source/f'examples/nn_aux/{case}_nn.jsonl'),
                '--rotor',str(source/f'examples/nn_aux/{case}_rotor.jsonl'),'--mode',mode,'--output',str(result)])
            actual=[json.loads(s) for s in result.read_text('utf-8').splitlines()]
            if mode=='baseline':expected=[json.loads(s) for s in (source/f'examples/nn_aux/{case}_nn.jsonl').read_text('utf-8').splitlines()]
            else:
                label='native_visible' if mode=='rotor-assist' else 'native_evidence_only'
                expected=[json.loads(s) for s in (STUDY/f'{case}_gated_{label}.jsonl').read_text('utf-8').splitlines()]
            assert len(actual)==len(expected)
            delta=0.
            for a,b in zip(actual,expected):
                assert a['t']==b['t'] and a['valid']==b['valid'] and a['coefficient']==b['coefficient']
                for h,p in a['physical'].items():delta=max(delta,max(abs(x-y) for x,y in zip(p,b['physical'][h])))
            assert delta<1e-6,(case,mode,delta)
            parity[case+'_'+mode]=dict(frames=len(actual),max_prediction_delta_px=delta)
    # Block all SDK/NN-framework imports to substantiate the lightweight replay claim.
    snippet='''import sys,runpy,importlib.abc
class NoInference(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0].startswith('metavision') or fullname.split('.')[0] in ('torch','ultralytics','onnxruntime'):raise RuntimeError('Forbidden inference import: '+fullname)
sys.meta_path.insert(0,NoInference());sys.path.insert(0,'.')
sys.argv=['nn_forecast.py','demo','--case','clear','--rotor-assist','--output',sys.argv[1]]
runpy.run_path('nn_forecast.py',run_name='__main__')
'''
    run('SDK_NN_blocked_demo',['-I','-c',snippet,str(VERIFY/'no_inference_demo.jsonl')])
    readme=nested((OUT/'README.md').read_text('utf-8'))
    (DELIVERY/'ROOT_README_FOR_NESTED.md').write_text(readme,'utf-8')
    root_files=DELIVERY/'root_files_for_nested_layout';root_files.mkdir(exist_ok=True)
    (root_files/'README.md').write_text(readme,'utf-8');shutil.copy2(OUT/'LICENSE',root_files/'LICENSE')
    (root_files/'NOTICE.md').write_text('NN主体研究版。完整版权/来源见[event-drone-forecast/NOTICE.md](event-drone-forecast/NOTICE.md)与[第三方许可](event-drone-forecast/THIRD_PARTY_NOTICES.md)。额外辅助未有稳定增益，默认关闭。根LICENSE与源码目录一致。\n','utf-8')
    batches=[]
    for i in range(0,len(files),90):
        name=f'网页上传第{i//90+1}批';batch=DELIVERY/name/'event-drone-forecast'
        for p in files[i:i+90]:
            q=batch/p.relative_to(OUT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
        batches.append(dict(folder=name,files=len(files[i:i+90])))
    (DELIVERY/(archive.name+'.sha256')).write_text(sha(archive)+'  '+archive.name+'\n','utf-8')
    (DELIVERY/'上传说明.txt').write_text('''推荐发布目录是本次event-drone-forecast，替代前一份几何主体快照；旧版本保留作历史，不需要先删除整个仓库。
标准根布局：用Git/GitHub Desktop提交event-drone-forecast里面的内容。
既有外部README+同名子目录布局：按网页上传第1批、第2批（如有第3批则继续）的同名文件夹分批传到仓库根，再上传root_files_for_nested_layout里的根README、LICENSE、NOTICE。ROOT_README_FOR_NESTED.md可复制全文替换外部README。
ZIP放GitHub Releases附件，标签v0.5.0-research.1；仅上传ZIP不会自动变成可浏览源码。
默认入口nn_forecast.py是NN定位/跟踪/状态外推，不是神经轨迹网络。额外辅助默认关闭，仅有固定输入因果回放，尚未串联实时RAW。收益小且不稳定，创新尝试未验证成功。简介与Release文字另附。
SDK、权重、RAW不附带。本地只核验打包、历史回放复原与帮助，不声明新包RAW实时成绩。没有自动上传GitHub。
''','utf-8')
    checks=dict(version='0.5.0-research.1',python_syntax_files=syntax,markdown_local_links=links,missing_links=missing,private_paths=private,
                checks=results,historical_replay_parity=parity,SDK_and_NN_imports_blocked_demo_passed=True,
                RAW_or_GPU_run=False,new_accuracy_or_FPS_measured=False,remote_uploaded=False)
    (VERIFY/'checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),'utf-8')
    delivery=dict(source=str(OUT),archive=str(archive),archive_MB=round(archive.stat().st_size/1048576,2),sha256=sha(archive),files=len(files),batches=batches,
                  verification=str(VERIFY/'checks.json'),positioning='NN-led main; default-off experimental rotor residual; unsuccessful innovation attempt under current evaluation',remote_uploaded=False)
    (DELIVERY/'delivery.json').write_text(json.dumps(delivery,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(delivery,ensure_ascii=False))
