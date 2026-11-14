"""Verify58/59 and retain vendor licenses in a recoverable handoff snapshot."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,re,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;BASE=PROJECT/'output'
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage59'))
import study59 as current
import lsd59
import delivery59
load=current.load;save=current.save;sha=current.sha

def run():
    current.verify();lsd59.verify();delivery59.run();folder=current.OUT
    a=load(current.previous.OUT/'completion.json');b=load(folder/'completion.json');d=load(folder/'delivery59.json')
    assert a['source_hashes_unchanged'] and a['synthetic_trials']==28 and a['real_current_ticks']==130
    assert b['source_hashes_unchanged'] and b['native10']==450 and b['geometry_exact57'] and not b['new_forecasts']
    assert load(folder/'process_inventory_20261021.json')['matching_project_python_count']==0
    assert d['unique_native10']==450 and d['total_RAW_window_reads']==900 and not d['new_future_scoring']
    for case,cfg in current.CASES.items():
        r=load(folder/f'{case}.json');s=load(folder/f'lsd_{case}.json');assert len(r['records'])==len(s['records'])==round(cfg['duration']*100)
        assert r['current_geometry_exact57'] and not r['labels_used']
        assert all(v['old57_geometry_exact'] and v['cap_binary_support_exact'] for v in r['records'])
    overview=BASE/'research_overview_20261018/index.html';page=overview.read_text(encoding='utf-8')
    lead='''<div class="lead"><b>已完成59：侧视线段观测可行，真实桨心身份与端点仍待验证。</b><p>450个不同RAW时刻、主试验/LSD各一次，共900窗口；阈值增强未稳定改善。侧视LSD70/200帧有线，但不能把双边/机臂/碎片中点都当桨心。正视椭圆与侧视线段应更新同一状态，未来预测尚无新成绩，默认仍54。</p><p>58实际运行固定版本作者圆心算法，28合成/130当前样本：中等倾角有效、极端侧视退化。下一步优先验证线段端点、双边中心线与过去桨槽/周期身份，再接质量权重融合。长期目标未完成，goal active。</p></div>'''
    page,n=re.subn(r'<div class="lead">.*?</div>(?=<div class="grid">)',lambda _:lead,page,count=1,flags=re.S);assert n==1
    for number,ref,title in [(58,'stage58_projected_centers_20261021/第五十八阶段投影圆心与共同桨盘报告.html','58 · 作者圆心与共同桨盘圆锥'),(59,'stage59_side_lines_20261021/结论与分支融合方案.html','59 · 侧视线段中点与融合方案')]:
        if '../'+ref not in page:page=page.replace('<div class="grid">','<div class="grid"><article><h2><a href="../'+ref+'">'+title+'</a></h2><p>全部受控失败、实际当前数据和局限。</p></article>',1)
    overview.write_text(page,encoding='utf-8')
    spec=importlib.util.spec_from_file_location('checker19_for59',ROOT/'tools/checkpoint_20261019.py');checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker);old_save=checker.save
    def routed(path,value):
        if path.name=='current_delivery_checks_20261019.json':path=path.with_name('current_delivery_checks_stage59_20261021.json')
        old_save(path,value)
    checker.save=routed;checker.run(False,'active');summary_path=overview.parent/'current_delivery_checks_stage59_20261021.json';summary=load(summary_path)
    assert summary['last_completed_stage']==59 and not summary['in_progress']
    for p in [overview,*folder.glob('*.html'),*current.previous.OUT.glob('*.html')]:
        for ref in re.findall(r'(?:href|src)="([^"]+)"',p.read_text(encoding='utf-8')):
            if '://' not in ref and not ref.startswith('#'):assert (p.parent/ref).exists(),(p,ref)
    summary.update(next_action='No new experiment60 started. Validate single-rotor line identity and full endpoints: pair opposite rim edges into a centerline, cross-threshold/time consistency and past slot/period evidence. Missing endpoints => axis-only update with large along-line center uncertainty. Then fuse ellipse/line observations into shared existing state with quality weighting/hysteresis and compare same-body-position full future controls. No raw event count as thrust truth or forced four points.',
        latest_verified_baseline_entry=str(ROOT/'core/run_deploy_forecast.py'),experimental_entry=str(ROOT/'core/run_candidate_forecast.py'),latest_report=str(folder/'结论与分支融合方案.html'),
        verified59_unique_native10=450,verified59_total_RAW_window_reads=900,stage58_known_truth_trials=28,stage58_uniform_current_samples=130,
        original_geometry_exact57=True,lsd_density_threshold_exact59=True,default_promoted=False,new_future_scoring=False,
        stage58_upstream_executed=True,stage59_RT_EvLDT_executed=False,stage59_actual_reuse='unchanged local geometry._line,57/9/8 and installed OpenCV LSD',
        experiments_running=[],terminal_batches=[56969,97466,38541,76310],current_status_document=str(PROJECT/'PROJECT_STATUS_20261021.md'),
        last_goal_turn_classification='progress: pinned/executed author circle-center control and jointly constrained conics; user-directed450-window line/threshold pilot plus450-window actual LSD diagnostic, explicit failures and no forecast-gain claim')
    save(summary_path,summary)
    files=set(ROOT.glob('*.py'))|set(ROOT.glob('*.md'))|set(PROJECT.glob('PROJECT_STATUS*.md'))|{PROJECT/'WORKLOG.md'}
    for parent in [ROOT/'src',ROOT/'research',ROOT/'tools']:
        for pattern in ['*.py','*.md']:files.update(p for p in parent.rglob(pattern) if '__pycache__' not in p.parts)
    # Explicitly include the pinned vendor source, reference protocols and legal notices.
    vendor=current.previous.geom.VENDOR;snapshot=load(vendor/'UPSTREAM_SNAPSHOT.json');files.add(vendor/'UPSTREAM_SNAPSHOT.json')
    for r in snapshot['files']:files.add(vendor/r['path'])
    assert vendor/'LICENSE' in files and vendor/'NOTICE' in files
    for number in summary['completed']:
        parent=next(p for p in BASE.glob(f'stage{number}_*') if (p/'protocol.json').exists())
        for pattern in ['*.json','*.md','*.html','derived_*.py']:
            files.update(p for p in parent.rglob(pattern) if p.stat().st_size<2_000_000 and 'failed_border_preparation_v1' not in p.parts)
    files.update(overview.parent.glob('*.html'));files.add(summary_path)
    archive=BASE/'checkpoints'/('research_through_stage59_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.zip');manifest={p.relative_to(PROJECT).as_posix():sha(p) for p in sorted(files)}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as zipped:
        for p in sorted(files):zipped.write(p,p.relative_to(PROJECT).as_posix())
        zipped.writestr('SNAPSHOT_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2));zipped.writestr('HANDOFF_CHECKPOINT.json',json.dumps(summary,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        for name,value in manifest.items():assert hashlib.sha256(zipped.read(name)).hexdigest()==value
    summary.update(archive=str(archive),archive_sha256=sha(archive),source_summary_files=len(files),zip_verified=True,
        archive_scope='Source/protocol/small reports, derived code and all23 pinned vendor files plus upstream manifest/LICENSE/NOTICE. RAW/weights/media/full JSONL remain in original directories.')
    save(BASE/'checkpoints/latest_handoff.json',summary);save(BASE/'checkpoints/latest_progress_20261021.json',summary)
    print(json.dumps(dict(stage=59,unique_native10=450,total_RAW_window_reads=900,files=len(files),archive=str(archive),verified=True),ensure_ascii=False),flush=True)

if __name__=='__main__':run()
