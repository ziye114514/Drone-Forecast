"""Verify complete57, retain exact threshold-probe evidence and active handoff."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,re,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;BASE=PROJECT/'output'
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage57'))
import study57 as current
import delivery57 as delivery
import density_probe57 as density

def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def run():
    current.verify();delivery.verify();density.verify();folder=current.OUT;protocol=load(folder/'protocol.json')
    for name,value in protocol['derived_source_hashes'].items():assert hashlib.sha256((folder/f'derived_{name}.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value
    for name,value in [('numeric_render',load(folder/'delivery_supplement.json')['derived_numeric_render_sha256']),('density_probe',load(folder/'density_probe_supplement.json')['instrumented_source_sha256'])]:
        assert hashlib.sha256((folder/f'derived_{name}.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value
    counts=dict(parent=0,native10=0,video=0)
    for case in current.CASES:
        r=load(folder/case/'regression.json');c=load(folder/case/'completion.json')
        assert c['source_hashes_unchanged'] and c['history_all_six_exact'] and not c['labels_read']
        assert r['native20_body_pixels_six_history_and_validity_exact54'] and r['hint_causality_checked'] and r['all_current_events_preserved']
        if current.CASES[case]['geometry_roi']=='off':assert r['off_complete15_exact54']
        counts['parent']+=r['parent_frames'];counts['native10']+=r['native10_ticks'];counts['video']+=r['video_decoded_frames']
    assert counts==dict(parent=1715,native10=3430,video=1480)
    prefix=load(folder/'raw_prefix_checks.json');assert prefix['parent_ticks']==225 and prefix['native10_ticks']==450 and prefix['actual_RAW_prefix'] and prefix['canonical_all_named_event_fields_hash_exact'] and prefix['complete_forecasts_lifecycle_ROI_hint_exact'] and prefix['per_transfer_opaque_byte_checksum_preserved']
    scores=load(folder/'score_checks.json');assert scores['scoring_rows']==360 and scores['old_baseline_or_six_history_exact_rows']==252 and scores['all_coverage_identical54']
    assert not list(folder.rglob('ipc_failure.json'))
    for mode in ['full','clipped']:
        r=load(folder/f'density_probe_{mode}.json');assert r['geometry_exact55'] and r['native10']==200 and not r['labels_read']
    probe=load(folder/'density_probe_analysis.json')['summary'];assert probe['active_roi_ticks']==probe['changed_threshold_ticks']==192
    # Where all current rotor observations were unchanged, prove full15 output,
    # not just the six historical branches or sparse endpoint averages.
    exact_cases={}
    for case in ['clear6','train3']:
        new=current.rows(folder/case/'observations_forecasts.jsonl');old=current.rows(folder/('off_'+case)/'observations_forecasts.jsonl')
        assert len(new)==len(old)
        assert all(a['frontends']==b['frontends'] and a['lifecycle']==b['lifecycle'] for a,b in zip(new,old))
        exact_cases[case]=len(new)
    assert load(folder/'process_inventory_20261021.json')['matching_project_python_count']==0
    overview=BASE/'research_overview_20261018/index.html';page=overview.read_text(encoding='utf-8')
    lead='''<div class="lead"><b>已完成57：修复硬ROI造成的清晰几何退化，侧视预测仍未突破。</b><p>保留完整轮廓/全图阈值，只筛候选；清晰0.4s物理从硬裁17.20恢复原14.22px，侧视122.45px、3号不变，联合成功数未提升。清晰与3号全部15模型逐帧精确原全图。默认仍54。</p><p>1715父/3430子时刻、1480视频帧、真实RAW前缀和360行评分通过。额外400子时刻阈值探针精确旧几何，确认硬裁每帧阈值比值中位1.315，但切边与阈值尚未独立归因。下一步58核查共面桨盘圆锥、真实轴心投影及消歧作者代码；长期目标active。</p></div>'''
    page,n=re.subn(r'<div class="lead">.*?</div>(?=<div class="grid">)',lambda _:lead,page,count=1,flags=re.S);assert n==1
    if '../stage57_candidate_roi_20261021/' not in page:
        page=page.replace('<div class="grid">','<div class="grid"><article><h2><a href="../stage57_candidate_roi_20261021/实验结论与密度阈值诊断.html">57 · 完整轮廓候选筛选与实际阈值诊断</a></h2><p>恢复清晰几何，侧视仍无新预测收益；三路图、视频、真实RAW和精确阈值记录。</p></article>',1)
    overview.write_text(page,encoding='utf-8')
    spec=importlib.util.spec_from_file_location('checker19_for57',ROOT/'tools/checkpoint_20261019.py');checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker);old_save=checker.save
    def routed(path,value):
        if path.name=='current_delivery_checks_20261019.json':path=path.with_name('current_delivery_checks_stage57_20261021.json')
        old_save(path,value)
    checker.save=routed;checker.run(False,'active');summary_path=overview.parent/'current_delivery_checks_stage57_20261021.json';summary=load(summary_path)
    assert summary['last_completed_stage']==57 and not summary['in_progress']
    for path in [overview,*folder.glob('*.html')]:
        for ref in re.findall(r'(?:href|src)="([^"]+)"',path.read_text(encoding='utf-8')):
            if '://' not in ref and not ref.startswith('#'):assert (path.parent/ref).exists(),(path,ref)
    summary.update(next_action='Stage58 not started. Verify/pin reusable author code for projected-circle centers and coplanar conic ambiguity; check real RAW intrinsic metadata. Start with49 known projection/noise/edge-on controls, then current RAW observability/sensitivity without claiming true pose/RPM. Extra conic constraints must be independent of arbitrary-four-point homography fit. No virtual measured rotors or lowered gates.',
        latest_verified_baseline_entry=str(ROOT/'core/run_deploy_forecast.py'),experimental_entry=str(ROOT/'core/run_candidate_forecast.py'),latest_report=str(folder/'实验结论与密度阈值诊断.html'),
        verified_counts=counts,separate_density_probe_native10=400,density_probe=probe,exact_full15_and_lifecycle_cases=exact_cases,
        score_rows=360,exact_baseline_or_history_rows=252,future_three_way=load(folder/'three_way_results.json')['table'],default_promoted=False,
        all_current_events_preserved=True,experiments_running=[],terminal_batches=[35676,20880,27526],current_status_document=str(PROJECT/'PROJECT_STATUS_20261021.md'),
        prior_art_for58=['https://github.com/xiahaa/circular-center-calibration','https://link.springer.com/article/10.1186/s41074-018-0050-y','https://github.com/BluecatLi/direct_event_object_tracker'],external58_code_executed=False,
        last_goal_turn_classification='progress: full RAW candidate-gating regression repair and same-target future scoring completed; exact old geometry threshold probe identifies hard-crop statistical change, no stable added prediction advantage')
    save(summary_path,summary)
    files=set(ROOT.glob('*.py'))|set(ROOT.glob('*.md'))|set(PROJECT.glob('PROJECT_STATUS*.md'))|{PROJECT/'WORKLOG.md'}
    for parent in [ROOT/'src',ROOT/'research',ROOT/'tools']:
        for pattern in ['*.py','*.md']:files.update(p for p in parent.rglob(pattern) if '__pycache__' not in p.parts)
    for number in summary['completed']:
        parent=next(p for p in BASE.glob(f'stage{number}_*') if (p/'protocol.json').exists())
        for pattern in ['*.json','*.md','*.html','derived_*.py']:
            files.update(p for p in parent.rglob(pattern) if p.stat().st_size<2_000_000 and 'failed_border_preparation_v1' not in p.parts)
    files.update(overview.parent.glob('*.html'));files.add(summary_path)
    archive=BASE/'checkpoints'/('research_through_stage57_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.zip');manifest={p.relative_to(PROJECT).as_posix():sha(p) for p in sorted(files)}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as zipped:
        for p in sorted(files):zipped.write(p,p.relative_to(PROJECT).as_posix())
        zipped.writestr('SNAPSHOT_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2));zipped.writestr('HANDOFF_CHECKPOINT.json',json.dumps(summary,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        for name,value in manifest.items():assert hashlib.sha256(zipped.read(name)).hexdigest()==value
    summary.update(archive=str(archive),archive_sha256=sha(archive),source_summary_files=len(files),zip_verified=True,archive_scope='Source/protocol/small reports and derived source; RAW, weights, media and full JSONL retained in original directories.')
    save(BASE/'checkpoints/latest_handoff.json',summary);save(BASE/'checkpoints/latest_progress_20261021.json',summary)
    print(json.dumps(dict(stage=57,goal_status='active',counts=counts,density_probe_native10=400,full15_exact_cases=exact_cases,files=len(files),archive=str(archive),verified=True),ensure_ascii=False))

if __name__=='__main__':run()
