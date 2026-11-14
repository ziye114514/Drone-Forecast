"""Verify the measured RNG mechanism, stable line pilot and explicitly failed60 attempt."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,re,sys,zipfile
import cv2
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;BASE=PROJECT/'output'
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage61'),str(ROOT/'research/stage60')]
import study61 as current
import delivery61
import forecast60
import checks60
load=current.load;save=current.save;sha=current.sha

def run():
    current.verify();checks60.verify();delivery61.run();folder=current.OUT;prior=current.previous.OUT
    result=load(folder/'results.json');assert result['unique_native10']==300 and result['total_RAW_window_reads']==600 and not result['labels_read'] and not result['new_future_scoring']
    for case in current.PARAMS['cases']:
        assert load(folder/f'normal_{case}.json')==load(folder/f'consume_rng_{case}.json')
    assert load(folder/'process_inventory_20261021.json')['matching_project_python_count']==0
    cause=load(prior/'rng_cause60.json');assert all(cause[n] for n in ['fixed_seed_all_equal','consumption_intervention_changed_fit','sequential_fit_changes'])
    for file,helper in [('rng_cause60.json','rng60.py'),('degenerate_contour_probe.json','degenerate60.py')]:assert load(prior/file)['helper_sha256']==sha(ROOT/'research/stage60'/helper)
    for kind in ['original','off','interleaved']:
        d=load(prior/f'diagnostic_train3_{kind}.json');assert d['all_complete'] and d['exact_ticks_before_divergence']==600
    raw=load(prior/'isolated_raw/raw_summary.json');assert raw['off_exact57'] and sum(r['native10'] for r in raw['table'])==1400
    # The failed formal attempt is retained as failed, not reclassified as completion.
    written=dict(parent=0,native10=0,video=0)
    for case in ['pilot_off_clear6','clear6','side6','train3']:
        f=forecast60.OUT/case;a=load(f/'completion.json');b=load(f/'rotor_completion.json')
        assert a['source_hashes_unchanged'] and b['source_hashes_unchanged'] and not a['labels_read'] and not b['labels_read']
        written['parent']+=a['frames'];written['native10']+=b['native10_ticks']
        video=cv2.VideoCapture(str(f/'forecasts.mp4'));count=0
        while True:
            ok,frame=video.read()
            if not ok:break
            assert frame.shape[:2]==(720,1280);count+=1
        video.release();assert count==a['frames'];written['video']+=count
    assert written==dict(parent=690,native10=1380,video=690)
    for case in ['pilot_off_clear6','clear6','side6']:assert load(forecast60.OUT/case/'regression60.json')['live_geometry_exact_standalone_RAW60']
    assert not (forecast60.OUT/'train3/regression60.json').exists()
    assert not (forecast60.OUT/'analysis.json').exists() and not (forecast60.OUT/'raw_prefix_checks.json').exists()
    assert not (forecast60.OUT/'negative5').exists() and not (forecast60.OUT/'prefix_side6').exists() and not (prior/'future_delivery.json').exists()
    save(prior/'stopped_future_attempt_inventory.json',dict(status='Stopped on strict train3 current-geometry regression at26.17; no future scoring, no full-prefix check.',written_and_video_decoded=written,
        no_future_truth_scoring=True,do_not_run_unchanged60=True,retained_frozen_entry=str(ROOT/'core/run_paired_forecast.py'),next='Use61 StablePairedFrontend for both baseline/off and paired modes, then fresh full future protocol.'))
    # Pin the actual binary/runtime, in addition to Python source.
    cvfolder=Path(cv2.__file__).parent;binary={p.relative_to(cvfolder).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in cvfolder.rglob('*') if p.suffix.lower() in ['.pyd','.dll']}
    assert binary and cv2.__version__==cause['opencv_version']=='5.0.0'
    probe=load(prior/'degenerate_contour_probe.json');assert cv2.getBuildInformation()==probe['opencv_build']
    save(folder/'runtime61.json',dict(opencv_version=cv2.__version__,opencv_package_path=str(cvfolder),binary_hashes=binary,
        build_information_sha256=hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),opencv_threads_during_checkpoint=cv2.getNumThreads(),binaries_archived=False))
    overview=BASE/'research_overview_20261018/index.html';page=overview.read_text(encoding='utf-8')
    lead='''<div class="lead"><b>61当前测量修复完成：先判线，避免共线轮廓随机薄椭圆。</b><p>实际OpenCV5同一11px点集fitEllipse长度约9.86–13.23px，可控RNG干预证实随机状态影响。新增先判线分支，300不同RAW时刻、两历史共600窗口，所有当前观测/审计精确一致。宽椭圆原路径保持，默认仍54。</p><p>60完整未来尝试已停在3号26.17严格回归失败，未执行前缀/评分；不能当成已改善。下一步正式评测61稳定基线与稳定+配对两路完整15及实际RAW前缀，预测保存后才评分。长期目标active，未完成。</p></div>'''
    page,n=re.subn(r'<div class="lead">.*?</div>(?=<div class="grid">)',lambda _:lead,page,count=1,flags=re.S);assert n==1
    for ref,title in [('stage60_paired_lines_20261021/第六十阶段双边中心线与尺度确认报告.html','60 · 双边诊断（完整未来尝试失败保留）'),('stage61_preellipse_lines_20261021/第六十一阶段先判线再拟合椭圆报告.html','61 · 薄轮廓先判线与真实RNG控制')]:
        if '../'+ref not in page:page=page.replace('<div class="grid">','<div class="grid"><article><h2><a href="../'+ref+'">'+title+'</a></h2><p>当前控制与局限，未来收益尚待正式评测。</p></article>',1)
    overview.write_text(page,encoding='utf-8')
    spec=importlib.util.spec_from_file_location('checker19_for61',ROOT/'tools/checkpoint_20261019.py');checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker);old_save=checker.save
    def routed(path,value):
        if path.name=='current_delivery_checks_20261019.json':path=path.with_name('current_delivery_checks_stage61_20261021.json')
        old_save(path,value)
    checker.save=routed;checker.run(False,'active');summary_path=overview.parent/'current_delivery_checks_stage61_20261021.json';summary=load(summary_path)
    assert summary['last_completed_stage']==61
    summary.update(completed_scope='Stage numbers60/61 refer to delivered current-module diagnostics only;60 formal future attempt failed before scoring,61 formal future not started.',
        incomplete_attempts=[dict(stage=60,part='full_RAW15_forecast',status='stopped_strict_geometry_regression_failure',future_scored=False,prefix_checked=False,original_outputs_preserved=True)],
        next_action='Start new61 formal protocol/worker/entry using61 StablePairedFrontend injected into existing60 shared-worker namespace. line_mode=off keeps preline active as stable baseline; paired adds60 pairing/fusion. Same clear22-24/side108-113/train3 25-31/empty0-1 RAW, full original frequency/15/RLS, same NN body/six-history/coverage. Wide pilot with no thin route may exactly57; corrected rotor geometry need not exactly old random ellipses. Independent actual RAW prefix must exactly all new states and semantic event hashes. Save all predictions before original22 sparse scoring. Pin runtime61 binary/build. Do not resume unchanged60 or report its unexecuted delivery draft as complete. No new62 started.',
        latest_report=str(folder/'第六十一阶段先判线再拟合椭圆报告.html'),latest_verified_baseline_entry=str(ROOT/'core/run_deploy_forecast.py'),
        experimental_frontend=str(ROOT/'research/stage61/preline61.py'),default_promoted=False,new_future_scoring=False,
        stage61_unique_native10=300,stage61_total_RAW_window_reads=600,all_RNG_controls_exact=True,
        stage60_actual_written_partial_future_counts=written,stage60_full_future_complete=False,
        experiments_running=[],terminal_batches=[89399,31005,18571,1565,26481,1447,57994,37342,75737],
        current_status_document=str(PROJECT/'PROJECT_STATUS_20261021.md'),
        last_goal_turn_classification='progress: standard edge pairing and live full forecast attempt exposed real OpenCV5 RNG-dependent degenerate ellipse sizing; controlled fixed-array interventions verified mechanism; pre-ellipse line repair passed300 unique RAW ticks across RNG histories. Complete future benefit remains unproven.')
    save(summary_path,summary)
    files=set(ROOT.glob('*.py'))|set(ROOT.glob('*.md'))|set(PROJECT.glob('PROJECT_STATUS*.md'))|{PROJECT/'WORKLOG.md'}
    for parent in [ROOT/'src',ROOT/'research',ROOT/'tools']:
        for pattern in ['*.py','*.md']:files.update(p for p in parent.rglob(pattern) if '__pycache__' not in p.parts)
    vendor=current.previous.previous.previous.geom.VENDOR;snapshot=load(vendor/'UPSTREAM_SNAPSHOT.json');files.add(vendor/'UPSTREAM_SNAPSHOT.json')
    for r in snapshot['files']:files.add(vendor/r['path'])
    assert vendor/'LICENSE' in files and vendor/'NOTICE' in files
    for number in summary['completed']:
        parent=next(p for p in BASE.glob(f'stage{number}_*') if (p/'protocol.json').exists())
        for pattern in ['*.json','*.md','*.html','derived_*.py']:
            files.update(p for p in parent.rglob(pattern) if p.stat().st_size<2_000_000 and 'failed_border_preparation_v1' not in p.parts)
    files.update(overview.parent.glob('*.html'));files.add(summary_path)
    archive=BASE/'checkpoints'/('research_through_stage61_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.zip');manifest={p.relative_to(PROJECT).as_posix():sha(p) for p in sorted(files)}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as zipped:
        for p in sorted(files):zipped.write(p,p.relative_to(PROJECT).as_posix())
        zipped.writestr('SNAPSHOT_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2));zipped.writestr('HANDOFF_CHECKPOINT.json',json.dumps(summary,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        for name,value in manifest.items():assert hashlib.sha256(zipped.read(name)).hexdigest()==value
    summary.update(archive=str(archive),archive_sha256=sha(archive),source_summary_files=len(files),zip_verified=True,
        archive_scope='Source/protocol/small reports/derived source/diagnostics and all23 pinned58 vendor files+manifest/LICENSE/NOTICE. Actual61 runtime binaries hashed, not archived. RAW/weights/media/full JSONL/large derived JSON retained in original directories.')
    save(BASE/'checkpoints/latest_handoff.json',summary);save(BASE/'checkpoints/latest_progress_20261021.json',summary)
    print(json.dumps(dict(last_current_module=61,stage60_future='failed_unscored',stage61_current_unique=300,RAW_window_reads=600,files=len(files),archive=str(archive),verified=True),ensure_ascii=False),flush=True)

if __name__=='__main__':run()
