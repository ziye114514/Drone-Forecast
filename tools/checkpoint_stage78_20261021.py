"""Verify76–78 failure audit, independent state controls and RAW portability."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
import datetime,hashlib,json,re,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;BASE=PROJECT/'output'
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage78'),str(ROOT/'research/stage77'),str(ROOT/'research/stage76')]
import study78 as actual
import study77 as cache
import diagnose76 as audit
load=actual.load;save=actual.save;sha=actual.sha;OUT=actual.OUT
class Links(HTMLParser):
    def __init__(self):super().__init__();self.local=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ['href','src'] and v and not urlsplit(v).scheme and not v.startswith('#'):self.local.append(unquote(v.split('#')[0]))
def run():
    actual.verify();cache.verify()
    c=load(OUT/'checks78.json');assert c['parent']==2185 and c['native10']==4370 and c['video_frames_decoded']==1850
    score=load(OUT/'score_checks.json');assert score==dict(rows=1029,original77_rows_exact=588,additional_rules_unchanged=True)
    cached_score=load(cache.OUT/'score_checks.json');assert cached_score==dict(previous39_rows_exact=468,total_rows=588,same_output_coverage=True)
    protocol=load(cache.OUT/'protocol.json');assert protocol['controls']['independent_cv_protected_from_wrong_other_control'] and protocol['controls']['wrapper_mixture_covariance_matches_hand_calculation']
    assert load(audit.OUT/'audit76.json')['component_reconstruction_exact']
    d=load(OUT/'delivery78.json');assert d['helper_sha256']==sha(ROOT/'tools/deliver_stages76_78.py')
    inventory=load(OUT/'process_inventory_20261021.json');assert inventory['matching_project_python_count']==0
    report=Path(d['report']);parser=Links();parser.feed(report.read_text(encoding='utf-8'))
    for name in parser.local:assert (report.parent/name).resolve().is_file(),name
    prior=load(BASE/'checkpoints/latest_handoff.json');assert sha(prior['archive'])==prior['archive_sha256']
    data=load(OUT/'delivery_data78.json')
    for vendor in [ROOT/'external/filterpy_imm_upstream',cache.VENDOR]:
        snap=load(vendor/'UPSTREAM_SNAPSHOT.json')
        for item in snap['files']:
            raw=(vendor/item['path']).read_bytes();assert sha(vendor/item['path'])==item['sha256']
            if item.get('git_blob_sha1'):assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==item['git_blob_sha1']
    status=PROJECT/'PROJECT_STATUS_20261021.md';text=status.read_text(encoding='utf-8');heading='# 当前继续状态：2026-10-21\n'
    update='''
**最新76–78已完成：失败拆解、状态/未来输入分离与MMAE独立模式保护。** [新报告/六段预测视频](</D:/drone forecast/output/stage78_RAW_separated_forecast_20261021/第七十六至七十八阶段状态隔离与预测修正报告.html>)。新RAW入口 `event_drone_forecast/run_separated_forecast.py`，默认54保持。

原23秒0.4s旧融合→新mmae_state_fresh：clear14.746→13.752、side85.767→63.383、train3143.039→97.443px；强历史为17.404/80.243/95.288，3号仍略差。匹配常量13.932/62.918/97.405，主要收益来自结构修正，不能当RPM信息成功。额外1/2号评分点无当前物理输入，输出等于历史；2号新60.498差于CV46.966。额外4号新55.862差于历史49.912和常量48.565，未证实普遍迁移增益。物理缺失时回强历史，保持输出覆盖；当前支持/恒定输入资格共用同一时钟。

77旧39路保持、refresh-only滤波状态/概率/协方差精确74，两缓存前缀通过。78实际RAW原23秒+两个真实前缀+另外三段固定录像15秒，共2185父/4370原生窗口、1850视频帧全解码；原区域49预测/状态精确77、当前输入精确75。1029评分行、588原区域行精确，额外录像参数未改；具体新录像的正/负结果见新报告，不能按旧开发段单独判定泛化。

原作者MMAE独立预测/概率复用，固定提交/MIT保留；其向量混合协方差问题由控制暴露并在外部适配器修正。CV子状态在另一模式错误控制下与单独CV一致。端到端仍不实时，无机械RPM/质量/风速识别或新物体盲测主张。下一步优先有用周期/姿态覆盖、密集变向标注与观测链延迟；不要继续靠相同稀疏目标挑模型。以下“进行中/暂停”均为保留的历史记录。
'''
    if '**最新76–78已完成' not in text:text=text.replace(heading,heading+update+'\n',1)
    status.write_text(text,encoding='utf-8')
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8');heading=text.splitlines()[0]
    lead='''
2026-10-21最新完成76–78：[状态隔离/未来修正与六段RAW视频](</D:/drone forecast/output/stage78_RAW_separated_forecast_20261021/第七十六至七十八阶段状态隔离与预测修正报告.html>)、[新入口说明](research/stage78/README.md)。旧融合误差拆解后，固定十路结构对照，全部49路和实际RAW前缀验证；2185父/4370原生窗口、1850视频帧。原侧视0.4s85.767→63.383px、3号143.039→97.443，但恒定输入多数近似或更好；额外三个固定录像未证实普遍收益，2号回历史反而差于CV、4号差于历史。新实验 `run_separated_forecast.py`，默认54仍保持，端到端仍非实时。下面旧状态均为历史。
'''
    if '2026-10-21最新完成76–78' not in text:text=text.replace(heading,heading+'\n'+lead,1)
    readme.write_text(text,encoding='utf-8')
    worklog=PROJECT/'WORKLOG.md';text=worklog.read_text(encoding='utf-8')
    if '## 105. 78完整结束与交接' not in text:
        text+='''
## 105. 78完整结束与交接

- 40854退出0：原23秒主段1150父、两真实截断前缀285父、另外train1/train2/train4各5秒共750父，总2185父/4370原生10ms窗口；1850主/额外视频帧全解码。原区域所有当前输入精确75、49路预测/模式/学习/协方差精确77，两实际RAW前缀通过。重复上下文不能算独立样本。
- 1029评分行、588原区域行逐项精确77；额外三个片段规则/参数不变，完整成功/失败/缺席和输出覆盖保留。tools/deliver_stages76_78.py退出0，已查看主结果、额外结果图与固定画面；结果详见78报告及delivery_data78.json，不只引用开发收益。
- 额外1号4/6有效目标，新113.431px等于历史，1个缺席标签仍预测在画内；额外2号9/10，新60.498等于历史，差于CV46.966；两段有效评分锚点均0个当前物理输入。额外4号3/3，新55.862差于强历史49.912/匹配常量48.565。额外未调参数据不支持普遍收益，不能只拿原开发段进步宣布成功。
- 原开发段结构收益明显，匹配常量也几乎同样好；未证明RPM独立收益、稳定高精度实时或新物体泛化。3号主审阅分支仍略弱强历史，保留更好的其他结构路作对照，不按录像挑专家。原39路、坏旧IMM、默认54不删。
- 新RAW入口78 run_separated_forecast.py，最新缓存77 separated77.py，失败拆解76。FilterPy两套固定上游源/许可、MMAE协方差显式外部修正与独立CV保护控制均保留。SDK原生FrequencyAlgorithm仍仅调查未测试。交接快照由checkpoint_stage78_20261021.py逐文件校验；无本项目实验后台。
'''
    worklog.write_text(text,encoding='utf-8')
    overview=BASE/'research_overview_20261018/index.html';page=overview.read_text(encoding='utf-8')
    lead='<div class="lead"><b>76–78完成：状态隔离和未来输入修正改善旧融合，事件独立增益仍待验证。</b><p>新RAW入口run_separated_forecast.py。49路/原观测/真实前缀通过，另三固定录像无调参；2185父/4370原生窗口、1850视频帧。默认54保持，所有对照和额外失败保留。</p></div>'
    page,n=re.subn(r'<div class="lead">.*?</div>(?=<div class="grid">)',lambda _:lead,page,count=1,flags=re.S);assert n==1
    rel='../'+report.relative_to(BASE).as_posix()
    if rel not in page:page=page.replace('<div class="grid">','<div class="grid"><article><h2><a href="'+rel+'">76–78 · 融合失败、状态隔离与额外录像</a></h2><p>完整结构对照、固定新片段、真实视频与时延。</p></article>',1)
    overview.write_text(page,encoding='utf-8')
    summary=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution_status='checkpoint_ready',research_objective_status='incomplete',
        last_completed_stage=78,completed=list(range(19,79)),latest_full_forecast_stage=78,latest_cached_forecast_stage=77,in_progress=[],experiments_running=[],
        completed_scope='76 saved forecast failure decomposition;77 fixed structural controls preserving39;78 actual RAW full49 and additional fixed recordings.',
        default_promoted=False,new_thrust_input_enabled=True,new_thrust_input_scope='Experimental branches only.',
        latest_report=str(report),previous_report=prior['latest_report'],local_report_links_checked=len(parser.local),
        latest_verified_baseline_entry=str(ROOT/'core/run_deploy_forecast.py'),experimental_RAW_entry=str(ROOT/'core/run_separated_forecast.py'),experimental_RAW_stage=78,
        previous_IMM_RAW_entry=str(ROOT/'core/run_hybrid_forecast.py'),experimental_cache_predictor=str(ROOT/'research/stage77/separated77.py'),
        stage78_counts=dict(parent=2185,native10=4370,video_frames_decoded=1850),
        unique_current_context_seconds=38.,unique_native10_current_windows_this_round=3800,
        requested_native10_this_round=4370,count_scope='Old23s + fixed additional15s unique current contexts;5.7s prefix repeats. Warm seek/decode excluded.',
        stage77_score_checks=cached_score,stage78_score_checks=score,
        structural_controls=protocol['controls'],author_MMAE=cache.SNAPSHOT,
        sparse04_comparison=data['headline'],physical_target_usage=data['target_usage'],
        stable_high_accuracy_realtime_achieved=False,event_RPM_advantage_proven=False,
        no_G_or_yolov10_main_access=True,new_dependencies_installed=False,
        terminal_sessions=[98634,40854],inventory_file=str(OUT/'process_inventory_20261021.json'),
        current_status_document=str(status),
        retained_failed_attempts=prior.get('retained_failed_attempts',[])+[
            'Old75 missing physical input loses strong history; state controls can spread errors through IMM',
            'Pinned upstream MMAE covariance scalar zip defect detected; explicit wrapper correction tested',
            'New event variants do not consistently beat matched constant inputs',
            'Primary new3 segment still slightly worse than strong history'],
        next_action='Assess the fixed additional-segment table before further model selection. '
                    'Structure gains must be separated from event input gains. Prioritise useful '
                    'period/pose coverage, dense maneuver truth, detector bias and frontend latency. '
                    'Native SDK FrequencyAlgorithm remains inspected-only. Keep constant/no-event/history '
                    'controls and avoid continued tuning on these same sparse labels.',
        archive_scope='Prior75 source manifest plus current sources/docs and76–78 small protocols/checks/reports/figures. '
                      'Both original FilterPy sources and MIT retained; RAW/weights/full JSONL/video stay at original paths.',
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
    for n in range(76,79):
        for folder in BASE.glob('stage'+str(n)+'_*'):
            for pattern in ['*.json','*.md','*.html','*.png']:
                files.update(p for p in folder.rglob(pattern) if p.stat().st_size<2_000_000 and p.name!='analysis.json' and not p.name.startswith('regression_failure'))
    files.update(p for p in cache.VENDOR.iterdir() if p.is_file());files.update(overview.parent.glob('*.html'))
    archive=BASE/'checkpoints'/('research_through_stage78_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.zip')
    manifest={p.relative_to(PROJECT).as_posix():sha(p) for p in sorted(files)}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as zipped:
        for p in sorted(files):zipped.write(p,p.relative_to(PROJECT).as_posix())
        zipped.writestr('SNAPSHOT_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2));zipped.writestr('HANDOFF_CHECKPOINT.json',json.dumps(summary,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        for name,value in manifest.items():assert hashlib.sha256(zipped.read(name)).hexdigest()==value
    summary.update(archive=str(archive),archive_sha256=sha(archive),source_summary_files=len(files),zip_verified=True)
    save(BASE/'checkpoints/latest_handoff.json',summary);save(BASE/'checkpoints/latest_progress_20261021.json',summary)
    print(json.dumps(dict(last_stage=78,files=len(files),archive=str(archive),verified=True),ensure_ascii=False),flush=True)
if __name__=='__main__':run()
