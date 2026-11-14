"""Finish stage49 navigation and verify/archive sources, without running experiments."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,re,zipfile

ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;OUT=PROJECT/'output'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

log='''

## 57. 2026-10-20继续；完成49五圆刚体中心诊断

- 最新继续任务与get_goal为active，19日收尾暂停保留为历史。用户提出“中心不动、四桨平面摆动，单桨却在运动”，本轮先验证平移/姿态分离，而未直接启动原49参数参考转移方案。长期目标完整保留。
- 已查作者IPPE、ICRA2024 direct_event_object_tracker、CVPR2024 Unbiased Estimator for Distorted Conics及2604.18289。标准平面刚体/中心投影已有基础，圆的像椭圆中心也不等于圆心投影；不宣称该观察本身创新，不声称执行了上述完整开源系统。
- 新rigid49/study49冻结后运行，受控固定中心转动四点均值最大假移10.282px，真实轴心对角线中心误差<1e-8；真实平移控制及弱透视均值控制均通过。椭圆中心当轴心的单桨偏差最大2.004px/机架中心1.296px。这是人为相机/尺寸的几何示范，不是实采误差。0.75px噪声在接近共线时明显放大。
- 真实缓存650当前cutoff：四当前候选clear76/100、side22/250、train3 3/300；train3仅1凸四边形。当前NN框扩大25%关联后side19/250。没有未来/人工标签读取；框中心差和自报sigma扰动仅一致性诊断，不是真值精度。四点任意单应性零残差不证明身份；还缺可靠轴心对应、独立中心或跨时刻约束。
- 所有计算正常结束；图已查看。状态PROJECT_STATUS_20261020.md、总导航与交接更新。下一步共同机架/姿态约束关联，侧视按条件数降权；48频率参考重置导致学习饥饿的后续仍保留。未升级预测默认，不把49说成未来精度改善；47在线入口和30完整物理路径范围不变。
- 本轮受限venv无法启动用户目录系统Python，读配置后通过正常require_escalated执行原环境；没有重建环境/装包。所有旧数据/失败/模型保持，不用G，不访问yolov10-main，无额度重置或外部消息。
'''
worklog=PROJECT/'WORKLOG.md';old=worklog.read_text(encoding='utf-8')
if '## 57. 2026-10-20继续；完成49五圆刚体中心诊断' not in old:worklog.write_text(old+log,encoding='utf-8')
readme=ROOT/'README.md';old=readme.read_text(encoding='utf-8')
banner='2026-10-20已按最新继续任务恢复，完成49五圆刚体中心诊断。最新状态见[20日交接](</D:/drone forecast/PROJECT_STATUS_20261020.md>)，图与实采可用率见[49](</D:/drone forecast/output/stage49_rigid_center_20261020/第四十九阶段刚体中心分离诊断.html>)。49验证了姿态变化可造成中心代理假移，尚未证明未来预测改善。以下19日暂停记录保留为历史。\n\n'
if '2026-10-20已按最新继续任务恢复' not in old:
    header,body=old.split('\n',1);readme.write_text(header+'\n\n'+banner+body.lstrip(),encoding='utf-8')
overview=OUT/'research_overview_20261018/index.html';page=overview.read_text(encoding='utf-8')
lead='''<div class="lead"><b>2026-10-20已继续，完成49：把整机平移与机架姿态变化分开。</b><p>受控固定中心仿真中，四桨投影均值最大假移10.28px，真实轴心对角线中心保持；这不是实采精度。现有自采四当前候选clear76/100、side22/250、train3 3/300，仍缺可靠轴心和身份。49只做诊断，尚无新的未来预测提升。</p><p>47完整几何单步74.52ms/仅位置FP16 31.05ms仍未实时；48物理信息尚无稳定额外收益，侧视20次重置导致学习饥饿问题仍待解决。19日暂停及快照保留为历史，当前长期目标active。</p></div>'''
page,n=re.subn(r'<div class="lead">.*?</div>(?=<div class="grid">)',lambda _:lead,page,count=1,flags=re.S);assert n==1
if '../stage49_rigid_center_20261020/' not in page:page=page.replace('<div class="grid">','<div class="grid"><article><h2><a href="../stage49_rigid_center_20261020/第四十九阶段刚体中心分离诊断.html">49 · 五圆机架：平移与姿态分离</a></h2><p>固定中心/真实平移控制通过；650实采时刻检查四桨候选与侧视条件数。</p></article>',1)
page=page.replace('../../PROJECT_STATUS_20261019.md','../../PROJECT_STATUS_20261020.md').replace('最新收尾交接</a>','最新继续状态</a>')
overview.write_text(page,encoding='utf-8')

# Reuse prior source/protocol/supplement verifier; route its current summary to20.
spec=importlib.util.spec_from_file_location('checkpoint19_reused20',ROOT/'tools/checkpoint_20261019.py');checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
original_save=checker.save
def route(p,x):
    if p.name=='current_delivery_checks_20261019.json':p=p.with_name('current_delivery_checks_20261020.json')
    original_save(p,x)
checker.save=route;checker.run(False,'active')
summary_path=OUT/'research_overview_20261018/current_delivery_checks_20261020.json';summary=load(summary_path)
summary.update(next_action='Use49 rigid center/attitude diagnosis to improve true hub association and uncertainty gating. Preserve48 baseline; separately verify frequency-reference coordinate transfer before any rollout. No forecast accuracy improvement claimed by49.',
    latest_status_document=str(PROJECT/'PROJECT_STATUS_20261020.md'),experiments_running=[],last_goal_turn_classification='progress: new frozen49 synthetic controls and650 actual cached current diagnostics completed')
assert summary['last_completed_stage']==49 and summary['in_progress']==[]
diagnostics=OUT/'stage49_rigid_center_20261020';protocol=load(diagnostics/'protocol.json')
for name,value in protocol['input_hashes'].items():assert sha(PROJECT/name)==value
records=[json.loads(x) for x in (diagnostics/'current_diagnostics.jsonl').read_text(encoding='utf-8').splitlines()];assert len(records)==650
for page_path in [overview,*diagnostics.glob('*.html')]:
    for ref in re.findall(r'(?:href|src)="([^"]+)"',page_path.read_text(encoding='utf-8')):
        if '://' not in ref and not ref.startswith('#'):assert (page_path.parent/ref).exists(),(page_path,ref)
summary.update(stage49_input_hashes_unchanged=True,stage49_actual_saved_records=len(records),overview_and_stage49_links_valid=True)
save(summary_path,summary)
files=set(ROOT.glob('*.py'))|set(ROOT.glob('*.md'))|set(PROJECT.glob('PROJECT_STATUS*.md'))|{worklog}
for folder in [ROOT/'src',ROOT/'research',ROOT/'tools']:
    for pattern in ['*.py','*.md']:files.update(p for p in folder.rglob(pattern) if '__pycache__' not in p.parts)
for n in summary['completed']:
    folder=next(p for p in OUT.glob(f'stage{n}_*') if (p/'protocol.json').exists())
    for pattern in ['*.json','*.md','*.html']:files.update(p for p in folder.rglob(pattern) if p.stat().st_size<2_000_000 and 'failed_border_preparation_v1' not in p.parts)
files.update(overview.parent.glob('*.html'));files.add(summary_path)
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');archive=OUT/'checkpoints'/f'research_through_stage49_{stamp}.zip'
manifest={p.relative_to(PROJECT).as_posix():sha(p) for p in sorted(files)}
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(files):z.write(p,p.relative_to(PROJECT).as_posix())
    z.writestr('SNAPSHOT_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2));z.writestr('HANDOFF_CHECKPOINT.json',json.dumps(summary,ensure_ascii=False,indent=2))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name,value in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==value
summary.update(archive=str(archive),archive_sha256=sha(archive),source_summary_files=len(files),zip_verified=True,archive_scope='Source/protocol/small summaries only. No RAW, weights, PNG/MP4, full JSONL or large analysis; retain originals.')
save(OUT/'checkpoints/latest_handoff.json',summary);save(OUT/'checkpoints/latest_progress_20261020.json',summary)
print(json.dumps(dict(stage=49,goal_status='active',files=len(files),archive=str(archive),verified=True),ensure_ascii=False))
