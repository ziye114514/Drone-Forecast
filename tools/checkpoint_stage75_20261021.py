"""Audit68–75, publish exact latest state, archive sources/summary with hashes."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
import datetime,hashlib,json,re,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;BASE=PROJECT/'output'
sys.path[:0]=[str(ROOT/'research/stage75'),str(ROOT/'research/stage74'),str(ROOT/'research/stage73'),str(ROOT/'research/stage71'),str(ROOT/'research/stage68')]
import study75 as actual
import study74 as forecast
import study73 as extended
import study71 as raw_force
import study68 as observation
load=actual.load;save=actual.save;sha=actual.sha;OUT=actual.OUT
class Links(HTMLParser):
    def __init__(self):super().__init__();self.local=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ['href','src'] and v and not urlsplit(v).scheme and not v.startswith('#'):self.local.append(unquote(v.split('#')[0]))
def run():
    actual.verify();forecast.verify();extended.verify();raw_force.verify();observation.verify()
    raw=load(OUT/'checks75.json');assert raw['parent']==1435 and raw['native10']==2870 and raw['video_frames_decoded']==1100
    score=load(forecast.OUT/'score_checks.json');assert score==dict(previous33_rows_exact=396,total_rows=468,same_output_coverage=True)
    delivery=load(OUT/'delivery75.json');assert delivery['helper_sha256']==sha(ROOT/'tools/deliver_stages74_75.py')
    assert load(extended.OUT/'delivery73.json')['helper_sha256']==sha(ROOT/'tools/deliver_stages68_73.py')
    inventory=load(OUT/'process_inventory_20261021.json');assert inventory['matching_project_python_count']==0
    reports=[Path(delivery['report']),Path(load(extended.OUT/'delivery73.json')['report'])];link_count=0
    for report in reports:
        parser=Links();parser.feed(report.read_text(encoding='utf-8'))
        for name in parser.local:assert (report.parent/name).resolve().is_file(),name
        link_count+=len(parser.local)
    data=load(OUT/'delivery_data75.json');prior=load(BASE/'checkpoints/latest_handoff.json');assert sha(prior['archive'])==prior['archive_sha256']
    status=PROJECT/'PROJECT_STATUS_20261021.md';text=status.read_text(encoding='utf-8');heading='# 当前继续状态：2026-10-21\n'
    update='''
**最新68–75已完成（用户插回E盘后恢复）。** 事件活动→有效推力→延迟反馈→位置预测已经真正接通，随后按用户追加要求接入原始框CV与物理分支并行的作者FilterPy完整IMM。最新[双分支报告/视频](</D:/drone forecast/output/stage75_RAW_parallel_forecast_20261021/第七十四至七十五阶段双分支融合报告.html>)；前一轮[推力接线/时间扩展](</D:/drone forecast/output/stage73_temporal_extension_20261021/第六十八至七十三阶段事件推力预测报告.html>)。

75实际RAW1435父/2870原生窗口（1150主段+285重复前缀）、1100视频帧全解码，所有当前输入精确73、全部39路预测/学习/协方差/模式概率精确74；两个真实RAW截断前缀通过。74评分468行，396旧行精确。最新完整RAW入口 `event_drone_forecast/run_hybrid_forecast.py`（75）；默认仍 `run_deploy_forecast.py`（54）。

23秒主段0.4s稀疏框误差，原CV/强历史/双分支事件增量：清晰24.292/17.404/14.746px（7有效目标），侧视123.169/80.243/85.767（7），3号127.708/95.288/143.039（17）。清晰受益，侧视/3号未稳胜历史。机动分组只是因果CV代理，清晰机动仅1目标。六银行融合约10–12ms，但整条观测链仍非实时；没有真实RPM/质量/风速识别。**稳定高精度实时目标未达成，不升级默认。**

下一步先定位错误周期/方向/框跳变及陈旧输入，验证桨叶信息是否领先运动变化，并补更密的变向标签；保持无事件高Q、恒定输入与强历史对照。SDK原生FrequencyAlgorithm已只读核查但尚未运行实验。后面“暂停/未开始”均为保留的历史状态，不代表当前进度。
'''
    if '**最新68–75已完成' not in text:text=text.replace(heading,heading+update+'\n',1)
    status.write_text(text,encoding='utf-8')
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8');heading=text.splitlines()[0]
    lead='''
2026-10-21最新完成68–75：[框CV与事件物理双分支报告/视频](</D:/drone forecast/output/stage75_RAW_parallel_forecast_20261021/第七十四至七十五阶段双分支融合报告.html>)、[RAW入口与复用说明](research/stage75/README.md)、[推力接线/时间扩展报告](</D:/drone forecast/output/stage73_temporal_extension_20261021/第六十八至七十三阶段事件推力预测报告.html>)。真实RAW验证39路预测及两独立前缀，1435父/2870原生窗口、1100视频帧；作者FilterPy IMM固定提交/MIT保留。清晰0.4s原CV24.292→融合14.746px，但侧视/3号仍未稳胜强历史，端到端不实时。最新实验入口 `run_hybrid_forecast.py`，默认54保持。以下旧“未接推力/最新62/暂停”为历史记录。
'''
    if '2026-10-21最新完成68–75' not in text:text=text.replace(heading,heading+'\n'+lead,1)
    readme.write_text(text,encoding='utf-8')
    worklog=PROJECT/'WORKLOG.md';text=worklog.read_text(encoding='utf-8')
    if '## 100. 75完整结束与本轮交接' not in text:
        text+='''
## 100. 75完整结束与本轮交接

- 89385退出0：23秒主对照1150父/2300原生窗口，独立side4.5/train31.2前缀另285/570，共1435/2870，1100主视频帧完整解码。全部当前输入精确73、39路预测/参数/协方差/模式概率精确74，两实际RAW前缀逐项相同。不是1435独立样本；与73重叠。
- tools/deliver_stages74_75.py退出0；双分支报告包含原CV、强历史、高Q无事件、恒定输入、物理/增量的完整对照、平稳/机动代理分组及实际视频。清晰整体0.4s融合14.746px对CV24.292和历史17.404，但侧视85.767仍弱于历史80.243，3号143.039比历史95.288更差。不能选择性只报清晰或只与CV比较。
- 已核对报告图和视频固定帧。融合六银行约10–12ms，完整链仍非实时。默认未升级，无G/yolov10-main访问/新依赖安装；作者IMM源码与MIT许可固定保留。原SDK像素周期 API仅调查未实验。科学目标未达成，不把连通或历史回退称成功。
- 当前最新RAW入口75 run_hybrid_forecast.py；绝对力入口71 run_event_force_forecast.py、增量入口73 run_force_innovation_forecast.py均保留。最新缓存未来74，所有源/协议/报告/摘要交接校验由checkpoint_stage75_20261021.py负责，RAW/完整JSONL/视频留原路径不打入小快照。
'''
    worklog.write_text(text,encoding='utf-8')
    overview=BASE/'research_overview_20261018/index.html';page=overview.read_text(encoding='utf-8')
    lead='<div class="lead"><b>68–75完成：事件推力链已接通，框CV与物理IMM已并行；稳定高精度实时仍未达到。</b><p>最新RAW入口run_hybrid_forecast.py。39路/当前观测/学习状态与缓存精确，1435父/2870原生窗口、1100视频帧和实际前缀通过。清晰受益，侧视/3号未稳胜强历史；默认54保持。</p></div>'
    page,n=re.subn(r'<div class="lead">.*?</div>(?=<div class="grid">)',lambda _:lead,page,count=1,flags=re.S);assert n==1
    for report,title in zip(reports,['74–75 · 原始CV与事件物理双分支','68–73 · 事件推力接线与时间扩展']):
        rel='../'+report.relative_to(BASE).as_posix()
        if rel not in page:page=page.replace('<div class="grid">','<div class="grid"><article><h2><a href="'+rel+'">'+title+'</a></h2><p>完整对照、负结果、真实视频与时延。</p></article>',1)
    overview.write_text(page,encoding='utf-8')
    summary=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution_status='checkpoint_ready',research_objective_status='incomplete',
        last_completed_stage=75,completed=list(range(19,76)),latest_full_forecast_stage=75,latest_cached_forecast_stage=74,in_progress=[],experiments_running=[],
        completed_scope='68 force activity;69 full force/RLS;70 guard;71 actual RAW;72 increments;73 temporal extension;74 author IMM dual branch;75 actual RAW full39 and prefixes.',
        new_thrust_input_enabled=True,new_thrust_input_scope='Experimental branches only, default unchanged.',default_promoted=False,
        latest_report=str(reports[0]),previous_report=str(reports[1]),local_report_links_checked=link_count,
        latest_verified_baseline_entry=str(ROOT/'run_deploy_forecast.py'),experimental_RAW_entry=str(ROOT/'run_hybrid_forecast.py'),experimental_RAW_stage=75,
        absolute_force_RAW_entry=str(ROOT/'run_event_force_forecast.py'),increment_RAW_entry=str(ROOT/'run_force_innovation_forecast.py'),
        experimental_cache_predictor=str(ROOT/'research/stage74/hybrid74.py'),
        stage68_counts=dict(parent=700,native10=1400),stage71_counts=dict(parent=985,native10=1970,video_frames_decoded=650),
        stage71_pilot_counts=dict(parent=40,native10=80,video_frames_decoded=40),stage73_counts=dict(parent=1150,native10=2300,video_frames_decoded=1100),
        stage75_counts=dict(parent=1435,native10=2870,video_frames_decoded=1100),
        native10_requested_windows_all_new_passes=8620,distinct_native10_windows_this_round=2300,
        count_scope='23s unique intervals; repeated68/71/73/75 contexts/pilots/prefixes are not independent samples. Warm seek/decode overhead excluded.',
        author_IMM=forecast.snapshot,stage74_score_checks=score,
        sparse04_comparison=[r for r in data['main'] if r['model'] in ['cv','trusted_acc','dual_cv_event_increment']],
        regime_annotations_independent=False,stable_high_accuracy_realtime_achieved=False,
        terminal_sessions=[81742,89228,21708,13118,9002,80162,27674,98051,89385],inventory_file=str(OUT/'process_inventory_20261021.json'),
        retained_failed_attempts=prior.get('retained_failed_attempts',[])+[
            '68 missing E: freeze failed before creating protocol; user explicitly paused then resumed',
            '69 score import path failed; separate scoring-only adapter fixed it, frozen inference unchanged',
            '69 direct force/residual worse than strong history for side/train3',
            '70 event guard returned history everywhere; constant/all guards worsened side',
            '72/73 event increments no stable side/train3 advantage',
            '74/75 CV/physics IMM train3 degrades; only clear improves vs strong history'],
        no_G_or_yolov10_main_access=True,new_dependencies_installed=False,
        current_status_document=str(status),next_action='Keep current matched/no-event/strong-history controls. '
            'Diagnose side/train3 force direction, harmonic/area confounds, detector bias and aged inputs. '
            'Need dense maneuver annotations to establish cue lead, and a frontend latency plan. '
            'Native SDK FrequencyAlgorithm API inspected but not yet tested. No further sparse-target tuning or default promotion without stronger evidence.',
        archive_scope='Prior through67 manifest plus all current source/docs and68–75 small protocols/checks/figures/reports, upstream IMM/MIT preserved. RAW,weights,full JSONL/video stay at original paths.',
        prior_archive=prior['archive'],prior_archive_sha256=prior['archive_sha256'])
    files=set()
    with zipfile.ZipFile(prior['archive']) as zipped:
        manifest=json.loads(zipped.read('SNAPSHOT_MANIFEST.json'))
        for name,value in manifest.items():
            assert hashlib.sha256(zipped.read(name)).hexdigest()==value
            p=PROJECT/name
            if p.exists():files.add(p)
    for parent in [ROOT/'src',ROOT/'research',ROOT/'tools']:
        for pattern in ['*.py','*.md']:files.update(p for p in parent.rglob(pattern) if '__pycache__' not in p.parts)
    files.update(ROOT.glob('*.py'));files.update(ROOT.glob('*.md'));files.update(PROJECT.glob('PROJECT_STATUS*.md'));files.add(worklog)
    for n in range(68,76):
        for folder in BASE.glob('stage'+str(n)+'_*'):
            for pattern in ['*.json','*.md','*.html','*.png']:
                files.update(p for p in folder.rglob(pattern) if p.stat().st_size<2_000_000 and p.name!='analysis.json' and not p.name.startswith('regression_failure'))
    files.update(p for p in forecast.VENDOR.iterdir() if p.is_file());files.update(overview.parent.glob('*.html'))
    archive=BASE/'checkpoints'/('research_through_stage75_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.zip')
    manifest={p.relative_to(PROJECT).as_posix():sha(p) for p in sorted(files)}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as zipped:
        for p in sorted(files):zipped.write(p,p.relative_to(PROJECT).as_posix())
        zipped.writestr('SNAPSHOT_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2));zipped.writestr('HANDOFF_CHECKPOINT.json',json.dumps(summary,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        for name,value in manifest.items():assert hashlib.sha256(zipped.read(name)).hexdigest()==value
    summary.update(archive=str(archive),archive_sha256=sha(archive),source_summary_files=len(files),zip_verified=True)
    save(BASE/'checkpoints/latest_handoff.json',summary);save(BASE/'checkpoints/latest_progress_20261021.json',summary)
    print(json.dumps(dict(last_stage=75,files=len(files),archive=str(archive),verified=True),ensure_ascii=False),flush=True)
if __name__=='__main__':run()
