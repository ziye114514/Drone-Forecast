"""Seal the verified alpha.2 package; preserve frozen alpha.1 and position core."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,re,shutil,zipfile
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'output/github_release_update_20261028'
PUBLIC=BASE/'event-drone-forecast';ARCHIVE=BASE/'event-drone-forecast-v0.4.0-alpha.2.zip'
OLD=ROOT/'output/github_release_final_20261027'
def sha(data):return hashlib.sha256(data).hexdigest()
assert sha((OLD/'event-drone-forecast-v0.4.0-alpha.1.zip').read_bytes())=='ff1172b3f2d2e1752d8ebc6d299c6d972adfaaa1776f8773240e44aa3eb58abb'
core=list((ROOT/'event_drone_forecast/src/event_drone_forecast/position').glob('*.py'))
assert len(core)==7
for p in core:
    rel=Path('src/event_drone_forecast/position')/p.name
    assert p.read_bytes()==(OLD/'event-drone-forecast'/rel).read_bytes()==(PUBLIC/rel).read_bytes()
assert (ROOT/'event_drone_forecast/LICENSE').read_bytes()==(PUBLIC/'LICENSES/MIT-original.txt').read_bytes()
checks=json.loads((BASE/'verification/verification.json').read_text('utf-8'))
assert all(r['exit_code']==0 for r in checks['checks'])
assert checks['RAW_smoke']['source_ticks']==36
checks.update(frozen_alpha1_SHA256_unchanged=True,position_core_seven_files_unchanged=True,
    working_MIT_unchanged=True,latest_plot_visually_inspected=True,
    optional_adapter_machine_encoding_fixed=True,
    note='Validation is local on the existing SDK/GPU machine, not independent benchmark accuracy or a clean-machine/GitHub CI certificate.')
(PUBLIC/'RELEASE_VERIFICATION.json').write_text(json.dumps(checks,indent=2),'utf-8')
# Existing nested-layout root file alternatives; no redundant source archive.
rootfiles=BASE/'root_files_for_nested_layout'
shutil.copytree(PUBLIC/'LICENSES',rootfiles/'LICENSES',dirs_exist_ok=True)
rows=[]
for p in sorted(PUBLIC.rglob('*')):
    if not p.is_file() or p.name=='SOURCE_MANIFEST.json':continue
    name=p.relative_to(PUBLIC).as_posix();data=p.read_bytes()
    assert not any(q in ('__pycache__','.git','.cache','build','dist') for q in p.relative_to(PUBLIC).parts)
    assert p.suffix.lower() not in ('.raw','.onnx','.pt','.exe','.dll','.pyd','.pyc','.obj','.lib','.exp')
    if p.suffix.lower() in ('.py','.md','.json','.toml','.txt','.yml','.yaml','.cff','.bib','.cpp','.html'):
        text=data.decode('utf-8')
        assert not re.search(r'(?<![A-Za-z])[A-Za-z]:[\\/]',text),(name,'private absolute path')
        assert '子夜' not in text,(name,'personal path component')
        assert not re.search(r'gh[pousr]_[A-Za-z0-9]{30,}|sk-proj-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16}',text),(name,'obvious credential pattern')
    if p.suffix=='.py':ast.parse(data,filename=name)
    rows.append(dict(path=name,bytes=len(data),sha256=sha(data)))
manifest=dict(version='0.4.0-alpha.2',files=rows,weights_included=False,RAW_included=False,
    SDK_or_author_DLL_included=False,public_source_not_published=True)
(PUBLIC/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2),'utf-8')
temp=BASE/'event-drone-forecast-v0.4.0-alpha.2.zip.tmp'
with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PUBLIC.rglob('*')):
        if p.is_file():z.write(p,'event-drone-forecast/'+p.relative_to(PUBLIC).as_posix())
with zipfile.ZipFile(temp) as z:
    assert z.testzip() is None and len(z.namelist())==len(rows)+1
    for row in rows:assert sha(z.read('event-drone-forecast/'+row['path']))==row['sha256']
temp.replace(ARCHIVE)
delivery=dict(version='0.4.0-alpha.2',directory=str(PUBLIC),archive=str(ARCHIVE),
    files=len(rows)+1,bytes=ARCHIVE.stat().st_size,sha256=sha(ARCHIVE.read_bytes()),
    published=False,integrated_license='AGPL-3.0-only',verified_locally=True,
    source_only=True,RAW_weights_SDK_binaries_excluded=True,
    baseline_release_and_position_core_unchanged=True,
    root_README_for_existing_nested_layout=str(BASE/'ROOT_README_FOR_NESTED.md'),
    publication_instructions=str(PUBLIC/'GITHUB_PUBLISH.zh-CN.md'))
(BASE/'delivery.json').write_text(json.dumps(delivery,ensure_ascii=False,indent=2),'utf-8')
(BASE/(ARCHIVE.name+'.sha256')).write_text(delivery['sha256']+'  '+ARCHIVE.name+'\n','ascii')
instructions='''本次只更新本地发布材料，没有推送GitHub。
推荐：将event-drone-forecast内层目录的内容更新到仓库根目录。
保留现有嵌套布局：更新event-drone-forecast子目录，再用root_files_for_nested_layout中的文件更新仓库根目录。
ROOT_README_FOR_NESTED.md可全文复制到外部README。不要只改README而不更新源码。
网页上传单批不超过100文件，本版需分批；也可GitHub Desktop/Git提交。
ZIP单独作为v0.4.0-alpha.2 Pre-release附件。
完整发布文案和许可说明见event-drone-forecast/GITHUB_PUBLISH.zh-CN.md。
verification、delivery.json、交接日志和本说明是本地交付记录，不必上传源码仓库。
'''
(BASE/'更新操作说明.txt').write_text(instructions,'utf-8')
handoff_path=ROOT/'output/checkpoints/latest_handoff.json'
handoff=json.loads(handoff_path.read_text('utf-8'));stamp=datetime.now(timezone.utc).isoformat()
handoff.update(updated_utc=stamp,current_work_state='User-requested current pipeline explanation and alpha.2 GitHub update prepared locally; not published',
    current_running_jobs=[],github_update=delivery)
handoff['github_update']['verification']=checks
handoff['github_update']['remote_action']='none'
handoff['github_update']['prior_optimization_goal_complete']=False
handoff_path.write_text(json.dumps(handoff,ensure_ascii=False,indent=2),'utf-8')
with (ROOT/'WORKLOG.md').open('a',encoding='utf-8') as f:
    f.write('\n\n## 2026-10-28 当前流程核对与GitHub增量包\n'
        '- 用户要求完整展示检测逻辑并准备更新GitHub，本轮切换到源码/文档发布整理；没有远程上传、建仓或推送。\n'
        '- 新包output/github_release_update_20261028/event-drone-forecast-v0.4.0-alpha.2.zip；'+str(delivery['files'])+'文件，SHA256 '+delivery['sha256']+'。旧alpha.1 ZIP与position七文件逐字节保持，原工作MIT不变。\n'
        '- README完整Mermaid；docs/CURRENT_PIPELINE.md逐层说明RAW/ROI/弧度桨心/四三两点轴线/单状态/周期代理/二力/反馈与时序。baseline/candidate入口明确，默认不恢复传统并行预测。\n'
        '- 候选参数公开；可选SDK构建移除硬编码，修复中文Windows编译环境输出编码，实际从解压源码编译通过。无SDK头文件/DLL/pyd/RAW/权重/私有路径。\n'
        '- 四机制测试、合成demo、wheel安装隔离导入、SDK/GPU契约、baseline/candidate dry-run与1.2秒36时刻RAW接线通过；细节见RELEASE_VERIFICATION。不是新精度实验或干净机器认证。\n'
        '- 最新结果43.27px/P9580.60、train2 62.65px/周期0、初期610ms未隐瞒；目标仍未全面完成。仅修复离线误差图跨无标注空档连线，不改冻结预测。\n'
        '- 原有嵌套仓库提供ROOT_README_FOR_NESTED.md和root_files_for_nested_layout；GitHub网页单批100文件，118+文件需分批，已写说明。不要把verification目录/本地日志上传。\n')
print(json.dumps(delivery,ensure_ascii=False))
