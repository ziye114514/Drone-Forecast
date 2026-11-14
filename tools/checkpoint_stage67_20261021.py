"""Verify64–67 frozen studies, local report links and a recoverable handoff."""
from pathlib import Path
import datetime,hashlib,json,re,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;BASE=PROJECT/'output'
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'tools'),*[str(ROOT/'research'/f'stage{n}') for n in [64,65,66,67]]]
import checkpoint_stage63_20261021 as old
import study64 as motion
import study65 as forecast
import study66 as original_periodic
import study67 as periodic
load=motion.load;save=motion.save;sha=motion.sha


def run():
    motion.verify();forecast.verify();original_periodic.verify();periodic.verify()
    old.obs.verify();old.live.verify();old.full61.verify()
    helper=ROOT/'tools/deliver_stages64_67.py'
    for folder,name in [(motion.OUT,'delivery64.json'),(forecast.OUT,'delivery65.json'),
                        (original_periodic.OUT,'delivery66.json'),(periodic.OUT,'delivery67.json')]:
        assert load(folder/name)['helper_sha256']==sha(helper)
    assert load(periodic.OUT/'holdout_rate_controls67.json')['helper_sha256']==sha(ROOT/'research/stage67/holdout67.py')
    raw=load(motion.OUT/'raw_prefix_checks.json');assert raw['totals']==dict(parent=985,native10=1970)
    scores=load(forecast.OUT/'score_checks.json');assert scores==dict(rows=540,baseline_rows_exact62=180,all_coverage_exact62=True)
    for c in periodic.CASES:
        assert load(periodic.OUT/c/'regression67.json')['original66_current_spectra_velocity_and_sampling_exact']
        assert not load(periodic.OUT/c/'completion.json')['labels_read']
    inventory=load(periodic.OUT/'process_inventory_20261021.json');assert inventory['matching_project_python_count']==0
    prior=load(BASE/'checkpoints/latest_handoff.json');assert sha(prior['archive'])==prior['archive_sha256']
    report=periodic.OUT/'第六十七阶段多峰周期与本轮预测总览.html'
    parser=old.Links();parser.feed(report.read_text(encoding='utf-8'))
    for name in parser.local:assert (report.parent/name).resolve().exists(),name
    vendor=ROOT/'external/circular_center_calibration_168b85c1';snapshot=load(vendor/'UPSTREAM_SNAPSHOT.json')
    assert len(snapshot['files'])==23
    for item in snapshot['files']:
        path=vendor/item['path'];rawbytes=path.read_bytes()
        assert sha(path)==item['sha256']
        assert hashlib.sha1(b'blob '+str(len(rawbytes)).encode()+b'\0'+rawbytes).hexdigest()==item['git_blob_sha1']
    overview=BASE/'research_overview_20261018/index.html';page=overview.read_text(encoding='utf-8')
    lead='''<div class="lead"><b>64–67完成：方向补偿没有明显未来提升，局部周期仍不能直接当转速。</b><p>64实际RAW985父/1970窗口及前缀通过；65缓存540评分，3号138.262→138.107px、侧视不变。66/67无需桨心即可提局部周期，但额外有色非周期31/40、共同照明40/40被接受，未接推力学习。</p><p>最新完整自动RAW入口仍62，最新缓存未来消融65。68未开始：复用已有相位一致性、空间背景和谐波约束。默认不升级，所有负结果保留。</p></div>'''
    page,n=re.subn(r'<div class="lead">.*?</div>(?=<div class="grid">)',lambda _:lead,page,count=1,flags=re.S);assert n==1
    relative='../'+report.relative_to(BASE).as_posix()
    if relative not in page:page=page.replace('<div class="grid">','<div class="grid"><article><h2><a href="'+relative+'">64–67 · 补偿、方向预测与多峰周期总览</a></h2><p>全部控制、微小未来变化、有色噪声失效与下一步。</p></article>',1)
    overview.write_text(page,encoding='utf-8')
    summary=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),research_objective_status='incomplete',
        last_completed_stage=67,latest_full_forecast_stage=62,latest_cached_forecast_stage=65,
        completed=list(range(19,68)),in_progress=[],experiments_running=[],
        completed_scope='64 actual RAW motion-compensated observations;65 cache-only '
                        'projected-direction future ablation;66/67 actual RAW local '
                        'periodicity shadow tests and count-process controls. No new '
                        'joint RAW predictor/deployment/RPM or pose ground-truth accuracy.',
        stage60_full_future_complete=False,stage60_future_scored=False,stage61_full_future_complete=True,
        stage62_counts=prior['stage62_counts'],stage64_counts=raw['totals'],stage65_score_rows=540,
        stage66_counts=dict(native10=320,parent=160),stage67_counts=dict(native10=320,parent=160),
        stage66_67_same_native_windows=True,stage66_67_distinct_native_windows=320,
        stage67_extra_aperiodic_false_acceptances=31,stage67_extra_aperiodic_controls=40,
        default_promoted=False,new_thrust_input_enabled=False,
        latest_report=str(report),local_report_links_checked=len(parser.local),
        latest_verified_baseline_entry=str(ROOT/'core/run_deploy_forecast.py'),
        experimental_RAW_entry=str(ROOT/'core/run_axis_forecast.py'),experimental_RAW_stage=62,
        experimental_cache_predictor=str(ROOT/'research/stage65/partial65.py'),
        terminal_batches=[6771,48939,32697,32032],inventory_file=str(periodic.OUT/'process_inventory_20261021.json'),
        retained_failed_attempts=prior.get('retained_failures',[]),
        current_status_document=str(PROJECT/'PROJECT_STATUS_20261021.md'),
        next_action='68 not started. First reuse existing perception._frequency '
                    'two-half temporal phase coherence and rotor_sensing harmonic/rotation '
                    'phase hypotheses, then spatial common-mode lighting/background and '
                    'rotor association. Do not push local spectral candidates into thrust/RLS '
                    'until colored-noise, flicker, mixtures and frequency-switch controls pass '
                    'with useful coverage. Preserve67 multi-vs-single control; do not tune '
                    'on the3.2s known RAW. Direction chain must still handle NN identity '
                    'loss and aging unknown tilt, not simply extend expiry.',
        no_G_or_yolov10_main_access=True,new_dependencies_installed=False,
        archive_scope='Prior source/protocol/report snapshot plus new64–67 sources, '
                      'small diagnostics and figures. All23 pinned58 vendor files/LICENSE/NOTICE. '
                      'RAW,weights,full JSONL,videos and large score JSON retained at original paths.',
        research_turn_progress='Compensation and isolated projection correction verified; '
                    'no substantial future gain. Local center-free periodicity works as a '
                    'candidate extractor but additional aperiodic/flicker controls reveal '
                    'unsafe force interpretation, retained as explicit negative evidence.')
    summary_path=overview.parent/'current_delivery_checks_stage67_20261021.json';save(summary_path,summary)
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
    for folder in [motion.OUT,forecast.OUT,original_periodic.OUT,periodic.OUT]:
        for pattern in ['*.json','*.md','*.html','*.png']:
            files.update(p for p in folder.rglob(pattern) if p.stat().st_size<2_000_000)
    files.update(vendor/r['path'] for r in snapshot['files']);files.add(vendor/'UPSTREAM_SNAPSHOT.json')
    files.update(overview.parent.glob('*.html'));files.add(summary_path)
    archive=BASE/'checkpoints'/('research_through_stage67_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.zip')
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
    print(json.dumps(dict(last_stage=67,latest_full_RAW_forecast=62,latest_cache_forecast=65,
                         files=len(files),archive=str(archive),verified=True),ensure_ascii=False),flush=True)


if __name__=='__main__':run()
