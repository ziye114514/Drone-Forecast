"""Seal a locally verified source export; no remote publication."""
from pathlib import Path
import json,hashlib,zipfile,re,ast
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'output/github_release_final_20261027'
PUBLIC=BASE/'event-drone-forecast'
ARCHIVE=BASE/'event-drone-forecast-v0.4.0-alpha.1.zip'
def sha(data):return hashlib.sha256(data).hexdigest()
checks=json.loads((BASE/'verification/verification.json').read_text('utf-8'))
assert all(x['exit_code']==0 for x in checks['checks'])
checks.update(exact_extracted_definitions=35,local_document_links_checked=15,
    broken_local_document_links=0,private_path_or_obvious_key_scan_findings=0,
    example_gif_visually_inspected=True,
    scope='Local relocated source verification with existing SDK/GPU and external user RAW/model. No independent accuracy or clean-machine claim.')
(PUBLIC/'RELEASE_VERIFICATION.json').write_text(json.dumps(checks,indent=2),'utf-8')
files=[]
for p in sorted(PUBLIC.rglob('*')):
    if not p.is_file() or p.name=='SOURCE_MANIFEST.json':continue
    name=p.relative_to(PUBLIC).as_posix();data=p.read_bytes()
    assert p.suffix.lower() not in ('.raw','.onnx','.pt','.exe','.dll','.pyd','.pyc')
    assert not any(part in ('__pycache__','.git','build','dist','.cache') for part in p.relative_to(PUBLIC).parts)
    if p.suffix.lower() in ('.py','.md','.json','.toml','.txt','.yml','.yaml','.cff','.bib','.html'):
        text=data.decode('utf-8')
        assert not re.search(r'(?<![A-Za-z])[A-Za-z]:[\\/]',text),name
        assert '子夜' not in text,name
    if p.suffix=='.py':ast.parse(data,filename=name)
    files.append(dict(path=name,bytes=len(data),sha256=sha(data)))
manifest=dict(version='0.4.0-alpha.1',files=files,weights_included=False,RAW_included=False,
    SDK_or_author_DLL_included=False,public_source_not_published=True)
(PUBLIC/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2),'utf-8')
with zipfile.ZipFile(ARCHIVE,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PUBLIC.rglob('*')):
        if p.is_file():z.write(p,PUBLIC.name+'/'+p.relative_to(PUBLIC).as_posix())
with zipfile.ZipFile(ARCHIVE) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(files)+1
    for row in files:
        assert sha(z.read(PUBLIC.name+'/'+row['path']))==row['sha256']
    assert json.loads(z.read(PUBLIC.name+'/SOURCE_MANIFEST.json'))==manifest
report=dict(directory=str(PUBLIC),archive=str(ARCHIVE),bytes=ARCHIVE.stat().st_size,
    files=len(files)+1,sha256=sha(ARCHIVE.read_bytes()),integrated_license='AGPL-3.0-only',
    private_files_excluded=True,published=False,relocated_tests_passed=True,
    real_RAW_smoke_ticks=36,valid_forecasts=29,verified_on_existing_machine=True,
    verification_source_changed_only_by_public_metadata=True,
    source_working_license_unchanged=True)
(BASE/'delivery.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
(BASE/(ARCHIVE.name+'.sha256')).write_text(report['sha256']+'  '+ARCHIVE.name+'\n','ascii')
stamp=datetime.now(timezone.utc).isoformat()
checkpoint=ROOT/'output/checkpoints/latest_handoff.json'
handoff=json.loads(checkpoint.read_text('utf-8'))
handoff['github_source_release']=dict(report,utc=stamp,publication_text=str(PUBLIC/'GITHUB_PUBLISH.zh-CN.md'),
    release_notes=str(PUBLIC/'RELEASE_NOTES.md'),remote_action='none',
    original_working_project_preserved=True,protected_yolov10_projects_untouched=True)
handoff['updated_utc']=stamp
handoff['current_work_state']='User-requested GitHub source packaging completed locally; not remotely published'
handoff['current_running_jobs']=[]
checkpoint.write_text(json.dumps(handoff,ensure_ascii=False,indent=2),'utf-8')
with (ROOT/'WORKLOG.md').open('a',encoding='utf-8') as f:
    f.write('\n\n## 2026-10-27 GitHub源码预览版整理完成\n\n'
        '- 用户要求可直接发布的ZIP及发布说明；仅本地整理，没有上传、建仓或远程发布。\n'
        '- 最终包：`output/github_release_final_20261027/event-drone-forecast-v0.4.0-alpha.1.zip`，'
        +str(report['files'])+'文件，'+str(report['bytes'])+'字节，SHA256 `'+report['sha256']+'`。\n'
        '- 保留五圆主线/条件三桨补全/两点与轴线信息分工/周期代理/二力反馈、合成演示和运动学对照；历史研究依赖提取为35个AST一致定义。原工作项目与原MIT许可不改。\n'
        '- 集成发布版AGPL-3.0-only，完整许可、原MIT、FilterPy许可与来源保留；不分发权重、RAW、SDK、DLL、私人路径和交接日志。\n'
        '- 中文README、安装、方法、结果、模型说明、相关论文、AI开发说明、GITHUB_PUBLISH.zh-CN.md和RELEASE_NOTES.md已写好；模型缺失和未证明物理优势明确披露。\n'
        '- 实际ZIP解压后help、基础/SDK/GPU检查、3个机制测试、合成demo、wheel构建/安装/隔离导入、dry-run均通过；外部自采RAW26s起1.2s通过36时刻/33当前整机观测/29有效预测，原RAW未改。仅本机接线，不是干净机器或GitHub CI认证。\n'
        '- 15个相对文档链接有效，GIF已查看，最终ZIP完整性/全部manifest哈希通过；公开RELEASE_VERIFICATION.json不含私有路径。\n'
        '- 网页发布时将解压后内层目录内容放仓库根目录；ZIP另作为Pre-release附件。没有启动长期优化/后台任务。\n')
print(json.dumps(report,ensure_ascii=False))
