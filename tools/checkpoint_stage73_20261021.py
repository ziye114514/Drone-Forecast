"""Verify completed68–73 artifacts and preserve a recoverable source checkpoint."""
from pathlib import Path
from html.parser import HTMLParser
import datetime,hashlib,json,re,sys,zipfile
from urllib.parse import unquote,urlsplit
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;BASE=PROJECT/'output'
sys.path[:0]=[str(ROOT/'research/stage73'),str(ROOT/'research/stage71'),str(ROOT/'research/stage68')]
import study73 as latest
import study71 as actual
import study68 as observation
load=latest.load;save=latest.save;sha=latest.sha;OUT=latest.OUT

class Links(HTMLParser):
    def __init__(self):super().__init__();self.local=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ['href','src'] and v and not urlsplit(v).scheme and not v.startswith('#'):self.local.append(unquote(v.split('#')[0]))

def run():
    latest.verify();actual.verify();observation.verify()
    for folder in [BASE/'stage69_event_force_forecast_20261021',BASE/'stage70_guarded_event_force_20261021',BASE/'stage72_force_innovation_20261021']:
        assert (folder/'score_checks.json').is_file()
    data=load(OUT/'delivery_data73.json');delivery=load(OUT/'delivery73.json')
    assert delivery['helper_sha256']==sha(ROOT/'tools/deliver_stages68_73.py')
    inventory=load(OUT/'process_inventory_20261021.json');assert inventory['matching_project_python_count']==0
    report=Path(delivery['report']);parser=Links();parser.feed(report.read_text(encoding='utf-8'))
    for name in parser.local:assert (report.parent/name).resolve().is_file(),name
    prior=load(BASE/'checkpoints/latest_handoff.json');assert sha(prior['archive'])==prior['archive_sha256']
    raw71=load(actual.OUT/'checks71.json')
    assert raw71['parent']==985 and raw71['native10']==1970 and raw71['video_frames_decoded']==650
    assert sum(r['parent'] for r in data['raw_table'])==1150
    overview=BASE/'research_overview_20261018/index.html';page=overview.read_text(encoding='utf-8')
    lead='<div class="lead"><b>68–73：事件活动→有效推力→在线反馈→位置预测已真正接通；稳定预测收益尚未证明。</b><p>两个RAW独立入口完成验证。73冻结参数后延长到清晰27/侧视116/3号34秒，33路预测和原区域回归通过。旧片段清晰0.4s14.31→12.74px但仅2个有效目标，侧视/3号没有稳定事件收益。后续时间段结果、恒定输入对照、完整视频均在新报告。默认仍54，未称实时/RPM或风速识别成功。</p></div>'
    page,n=re.subn(r'<div class="lead">.*?</div>(?=<div class="grid">)',lambda _:lead,page,count=1,flags=re.S);assert n==1
    relative='../'+report.relative_to(BASE).as_posix()
    if relative not in page:page=page.replace('<div class="grid">','<div class="grid"><article><h2><a href="'+relative+'">68–73 · 事件推力与未来位置预测</a></h2><p>完整RAW链、失败对照、推力增量和后续时间段测试。</p></article>',1)
    overview.write_text(page,encoding='utf-8')
    summary=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution_status='checkpoint_ready',
        research_objective_status='incomplete',last_completed_stage=73,completed=list(range(19,74)),
        latest_full_forecast_stage=73,latest_cached_forecast_stage=72,in_progress=[],experiments_running=[],
        completed_scope='68 current RAW event-force proxies;69 full-chain force/RLS cache ablation;70 causal guard; '
                        '71 actual RAW full26 and prefixes;72 increments;73 full33 actual RAW temporal extension. '
                        'Engineering chain complete, stable predictive advantage not established.',
        stage71_counts=dict(parent=985,native10=1970,video_frames_decoded=650),
        stage71_pilot_counts=dict(parent=40,native10=80,video_frames_decoded=40),
        stage73_counts=dict(parent=1150,native10=2300,video_frames_decoded=sum(r['video_frames'] for r in data['raw_table'])),
        stage68_counts=dict(parent=700,native10=1400),
        raw_unique_native_windows_this_round=2300,raw_reads_native_windows_this_round=5750,
        raw_count_scope='All68/71/pilot73 old intervals overlap73 main; do not add repeated contexts as independent samples.',
        new_thrust_input_enabled=True,new_thrust_input_scope='Experimental69/71/72/73 branches only, not default.',
        default_promoted=False,latest_report=str(report),local_report_links_checked=len(parser.local),
        latest_verified_baseline_entry=str(ROOT/'run_deploy_forecast.py'),
        experimental_RAW_entry=str(ROOT/'run_force_innovation_forecast.py'),experimental_RAW_stage=73,
        absolute_force_RAW_entry=str(ROOT/'run_event_force_forecast.py'),
        experimental_cache_predictor=str(ROOT/'research/stage72/increment72.py'),
        temporal_extension_scores=str(OUT/'temporal_scores73.json'),
        inventory_file=str(OUT/'process_inventory_20261021.json'),
        terminal_batches=[81742,89228,21708,13118,9002,80162],
        retained_failed_attempts=prior.get('retained_failed_attempts',[])+[
            '68 initial freeze before disk reconnected: E: missing, no protocol written',
            '69 initial score import path lacked predict22; separate scoring-only score69.py adapter, inference unchanged',
            '69 force and residual experimental predictions worse than strong history in side/train3',
            '70 constant/all guards worsened side future; strict event guard chose history everywhere'],
        no_G_or_yolov10_main_access=True,new_dependencies_installed=False,
        current_status_document=str(PROJECT/'PROJECT_STATUS_20261021.md'),
        next_action='Do not present connected chain, history fallback or two-target clear improvement as research success. '
                    'Use fixed temporal-extension results and matched constant controls to locate missing event information; '
                    'prioritise certifying local tone/rotor origin and handling changing apparent area/detector bias. '
                    'No further tuning on the same sparse targets and no default promotion without stronger evidence.',
        archive_scope='Prior through67 archive files plus new sources/docs, protocols, small summary checks and figures. '
                      'RAW/weights/full JSONL/video remain in original locations; snapshot is source/summary recovery.',
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
    files.update(ROOT.glob('*.py'));files.update(ROOT.glob('*.md'));files.update(PROJECT.glob('PROJECT_STATUS*.md'));files.add(PROJECT/'WORKLOG.md')
    for n in range(68,74):
        folders=list(BASE.glob('stage'+str(n)+'_*'))
        for folder in folders:
            for pattern in ['*.json','*.md','*.html','*.png']:
                files.update(p for p in folder.rglob(pattern) if p.stat().st_size<2_000_000 and p.name!='analysis.json' and not p.name.startswith('regression_failure'))
    files.update(overview.parent.glob('*.html'))
    archive=BASE/'checkpoints'/('research_through_stage73_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.zip')
    manifest={p.relative_to(PROJECT).as_posix():sha(p) for p in sorted(files)}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as zipped:
        for p in sorted(files):zipped.write(p,p.relative_to(PROJECT).as_posix())
        zipped.writestr('SNAPSHOT_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        zipped.writestr('HANDOFF_CHECKPOINT.json',json.dumps(summary,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        for name,value in manifest.items():assert hashlib.sha256(zipped.read(name)).hexdigest()==value
    summary.update(archive=str(archive),archive_sha256=sha(archive),source_summary_files=len(files),zip_verified=True)
    save(BASE/'checkpoints/latest_handoff.json',summary);save(BASE/'checkpoints/latest_progress_20261021.json',summary)
    print(json.dumps(dict(last_stage=73,files=len(files),archive=str(archive),verified=True),ensure_ascii=False),flush=True)

if __name__=='__main__':run()
