"""Verify completed61/62 and observation-only63; save a recoverable source handoff."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
import datetime,hashlib,json,re,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;BASE=PROJECT/'output'
sys.path[:0]=[str(ROOT/'research/stage63'),str(ROOT/'research/stage62'),str(ROOT/'research/stage61')]
import study63 as obs
import raw62 as live
import forecast61 as full61
load=obs.load;save=obs.save;sha=obs.sha


class Links(HTMLParser):
    def __init__(self):super().__init__();self.local=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ['href','src'] and value:
                parsed=urlsplit(value)
                if not parsed.scheme and not parsed.netloc and parsed.path:self.local.append(unquote(parsed.path))


def run():
    obs.verify();live.verify();full61.verify()
    d62=load(obs.previous.OUT/'delivery62.json');d63=load(obs.OUT/'delivery63.json')
    assert d62['helper_sha256']==sha(ROOT/'research/stage62/delivery62.py')
    assert d63['helper_sha256']==sha(ROOT/'research/stage63/delivery63.py')
    assert load(obs.previous.OUT/'score_adapter.json')['helper_sha256']==sha(ROOT/'research/stage62/score62.py')
    assert load(full61.current.OUT/'conclusion_full61.json')['helper_sha256']==sha(ROOT/'research/stage61/conclusion_full61.py')
    assert d62['raw_checks']['totals']==dict(parent=1025,native10=2050,video_frames_decoded=740)
    assert d62['score_checks']['rows']==540 and d62['score_checks']['baseline_rows_exact61']==180
    assert d63['native_RAW_ticks']==1400 and d63['current_parent_ticks']==700 and not d63['new_future_scores']
    inventory=load(obs.OUT/'process_inventory_20261021.json');assert inventory['matching_project_python_count']==0
    prior=load(BASE/'checkpoints/latest_handoff.json')
    assert sha(prior['archive'])==prior['archive_sha256']
    reports=[full61.current.OUT/'稳定线段与配对完整预测报告.html',
             obs.previous.OUT/'第六十二阶段方向与转速独立使用报告.html',
             obs.OUT/'第六十三阶段不依赖四桨的整机轴观测报告.html']
    report_checks=[]
    for path in reports:
        parser=Links();parser.feed(path.read_text(encoding='utf-8'))
        for ref in parser.local:assert (path.parent/ref).resolve().exists(),(path,ref)
        report_checks.append(dict(report=str(path),local_links_checked=len(parser.local),sha256=sha(path)))
    vendor=ROOT/'external/circular_center_calibration_168b85c1'
    snapshot=load(vendor/'UPSTREAM_SNAPSHOT.json');assert len(snapshot['files'])==23
    for item in snapshot['files']:
        p=vendor/item['path'];raw=p.read_bytes()
        assert sha(p)==item['sha256']
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==item['git_blob_sha1']
    overview=BASE/'research_overview_20261018/index.html';page=overview.read_text(encoding='utf-8')
    lead='''<div class="lead"><b>62完整方向分支与63整机方向观测已完成。</b><p>62实际RAW1025父/2050子、740视频帧和独立前缀通过。侧视0.4秒轴向122.89→113.92px，仍差于历史81.91；3号变差，不升级默认。63无四桨初始化即可提整机方向，但候选不等于平面法线，未接新预测。</p><p>61完整两路已评测、配对无实质未来收益；60旧失败保留。长期目标active。下一步先区分运动拖尾与刚体方向，再验证局部周期，无新64开始。</p></div>'''
    page,n=re.subn(r'<div class="lead">.*?</div>(?=<div class="grid">)',lambda _:lead,page,count=1,flags=re.S);assert n==1
    for path,title,text in [(reports[0],'61 · 稳定线段完整预测对照','两路正式预测完成，配对未获实质未来收益。'),
                            (reports[1],'62 · 方向与转速独立使用','完整RAW及预测，侧视局部改善、3号变差。'),
                            (reports[2],'63 · 不依赖四桨的整机轴观测','只有方向候选诊断，没有新未来评分。')]:
        ref='../'+path.relative_to(BASE).as_posix()
        if ref not in page:page=page.replace('<div class="grid">','<div class="grid"><article><h2><a href="'+ref+'">'+title+'</a></h2><p>'+text+'</p></article>',1)
    overview.write_text(page,encoding='utf-8')
    summary=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),goal_status='active',
        last_completed_stage=63,latest_full_forecast_stage=62,
        completed_scope='61 complete RAW paired controls;62 complete RAW independent-axis forecasting; '
                        '63 current whole-body direction observations only, no new forecast/RPM/pose accuracy.',
        completed=list(range(19,64)),in_progress=[],experiments_running=[],
        stage60_full_future_complete=False,stage60_future_scored=False,stage61_full_future_complete=True,
        stage61_counts=load(full61.current.OUT/'full_future_delivery61.json')['totals'],
        stage62_counts=d62['raw_checks']['totals'],stage62_scoring_rows=540,
        stage63_counts=dict(native10=1400,parent=700),stage63_future_scored=False,
        latest_report=str(reports[1]),latest_observation_report=str(reports[2]),
        latest_verified_baseline_entry=str(ROOT/'run_deploy_forecast.py'),
        experimental_entry=str(ROOT/'run_axis_forecast.py'),experimental_axis_mode='axis_learn',
        default_promoted=False,report_links=report_checks,
        terminal_batches=[71092,97719,5237,61122],
        retained_failures=['stage60 formal strict geometry regression, unscored',
            'stage62 first generated-step indentation guard failed before freeze, corrected',
            'stage62 original score entry case-list mismatch, superseded by separate score62 adapter',
            'stage62 first score62 import-path omission, corrected before writing scores'],
        next_action='Reuse6/21 existing causal motion compensation with past NN-CV velocity '
                    'to distinguish rigid side-axis from accumulated body/motion trails. '
                    'Validate known-projection controls and partial undirected measurement '
                    'uncertainty before using63 candidates in prediction/learning. Preserve62 '
                    'shared/gate/learn controls, original history and RPM experts. Do not '
                    'promote body PCA to full3D normal or strongest frequency to total thrust. '
                    'Local polarity-periodicity/overlap multi-frequency remains separate. '
                    'No64 started. No need to collect true mass/wind; effective combinations only.',
        current_status_document=str(PROJECT/'PROJECT_STATUS_20261021.md'),
        inventory_file=str(obs.OUT/'process_inventory_20261021.json'),
        no_G_or_yolov10_main_access=True,new_dependencies_installed=False,
        archive_scope='Prior verified source/protocol/report checkpoint plus new sources and '
                      'small61/62/63 diagnostics/reports, all23 pinned58 vendor files and licenses. '
                      'RAW/weights/full JSONL/media/large score JSON remain at original paths.',
        last_goal_turn_classification='progress: independent geometry admission and learning '
                    'validated in full RAW and actual prefixes; limited side gain, negative '
                    'train3 result retained. No-four-lock whole-body axis diagnostic exposes '
                    'remaining identity/support/motion-trail and projection limitations.')
    save(BASE/'research_overview_20261018/current_delivery_checks_stage63_20261021.json',summary)
    files=set()
    with zipfile.ZipFile(prior['archive']) as zipped:
        old_manifest=json.loads(zipped.read('SNAPSHOT_MANIFEST.json'))
        for name,value in old_manifest.items():
            assert hashlib.sha256(zipped.read(name)).hexdigest()==value
            path=PROJECT/name
            if path.exists():files.add(path)
    for parent in [ROOT/'src',ROOT/'research',ROOT/'tools']:
        for pattern in ['*.py','*.md']:
            files.update(p for p in parent.rglob(pattern) if '__pycache__' not in p.parts)
    files.update(ROOT.glob('*.py'));files.update(ROOT.glob('*.md'));files.update(PROJECT.glob('PROJECT_STATUS*.md'));files.add(PROJECT/'WORKLOG.md')
    for folder in [full61.current.OUT,obs.previous.OUT,obs.OUT]:
        for pattern in ['*.json','*.html','*.md','derived_*.py']:
            files.update(p for p in folder.rglob(pattern) if p.stat().st_size<2_000_000)
    files.update(vendor/item['path'] for item in snapshot['files']);files.add(vendor/'UPSTREAM_SNAPSHOT.json')
    files.update((BASE/'research_overview_20261018').glob('*.html'))
    files.add(BASE/'research_overview_20261018/current_delivery_checks_stage63_20261021.json')
    archive=BASE/'checkpoints'/('research_through_stage63_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.zip')
    manifest={p.relative_to(PROJECT).as_posix():sha(p) for p in sorted(files)}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as zipped:
        for p in sorted(files):zipped.write(p,p.relative_to(PROJECT).as_posix())
        zipped.writestr('SNAPSHOT_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        zipped.writestr('HANDOFF_CHECKPOINT.json',json.dumps(summary,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        for name,value in manifest.items():assert hashlib.sha256(zipped.read(name)).hexdigest()==value
    summary.update(archive=str(archive),archive_sha256=sha(archive),zip_verified=True,source_summary_files=len(files))
    save(BASE/'checkpoints/latest_handoff.json',summary);save(BASE/'checkpoints/latest_progress_20261021.json',summary)
    print(json.dumps(dict(last_stage=63,latest_full_forecast=62,files=len(files),archive=str(archive),verified=True),ensure_ascii=False),flush=True)


if __name__=='__main__':run()
