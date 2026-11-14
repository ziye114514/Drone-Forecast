"""Verify stage55 evidence, update navigation and save an active handoff."""
from pathlib import Path
import datetime, hashlib, importlib.util, json, re, sys, zipfile
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;BASE=PROJECT/'output'
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage55'))
import study55 as current
import recorded_run55 as recorded
import delivery55 as delivery

def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def run():
    recorded.verify();delivery.verify();folder=current.OUT
    protocol=load(folder/'protocol.json')
    # freeze() hashes source text, then Windows write_text stores CRLF bytes.
    # Match that declared LF text exactly; archive manifests still hash raw bytes.
    for name,value in protocol['derived_source_hashes'].items():
        stored=(folder/f'derived_{name}.py').read_bytes()
        assert hashlib.sha256(stored.replace(b'\r\n',b'\n')).hexdigest()==value
    g=load(folder/'geometry_diagnostic_supplement.json')
    assert g['helper_sha256']==sha(ROOT/'research/stage55/geometry_diagnostic55.py') and g['actual_reused49_sha256']==sha(ROOT/'research/stage49/rigid49.py')
    for name,value in g['input_hashes'].items():assert sha(folder/name)==value
    counts=dict(parent=0,native10=0,video=0)
    for case in current.CASES:
        r=load(folder/case/'regression.json');c=load(folder/case/'completion.json')
        assert c['source_hashes_unchanged'] and c['history_all_six_exact'] and not c['labels_read']
        assert r['native20_body_pixels_six_history_and_validity_exact54'] and r['hint_causality_checked']
        if current.CASES[case]['geometry_roi']=='off':assert r['off_complete15_exact54']
        counts['parent']+=r['parent_frames'];counts['native10']+=r['native10_ticks'];counts['video']+=r['video_decoded_frames']
    assert counts==dict(parent=1715,native10=3430,video=1480),counts
    prefix=load(folder/'raw_prefix_checks.json');assert prefix['parent_ticks']==225 and prefix['native10_ticks']==450
    assert prefix['actual_RAW_prefix'] and prefix['canonical_all_named_event_fields_hash_exact'] and prefix['complete_forecasts_lifecycle_ROI_hint_exact'] and prefix['per_transfer_opaque_byte_checksum_preserved']
    score=load(folder/'score_checks.json');assert score['all_coverage_identical54'] and score['scoring_rows']==360 and score['old_baseline_or_six_history_exact_rows']==252
    assert not (folder/'ipc_observation_failures.jsonl').exists()
    processes=load(folder/'process_inventory_20261021.json');assert processes['matching_project_python_count']==0
    overview=BASE/'research_overview_20261018/index.html';page=overview.read_text(encoding='utf-8')
    lead='''<div class="lead"><b>已完成55：整机框引导桨盘，尚无稳定预测提升。</b><p>同位置0.4秒轴向频率：侧视122.46→120.94px，清晰14.22→17.20px，3号不变；三段联合成功均未变，侧视仍差于历史81.91px。没有升级默认。</p><p>1715父/3430桨叶时刻、1480视频帧和真实225/450 RAW前缀通过；关闭完整15模型精确54，全部360行未来评分同覆盖。原首步JSON失败保留、原因未确认，后续严格规则批次完成。下一步复用50机架关联，先检查四点初始化和身份连续性。长期目标active。</p></div>'''
    page,n=re.subn(r'<div class="lead">.*?</div>(?=<div class="grid">)',lambda _:lead,page,count=1,flags=re.S);assert n==1
    if '../stage55_body_guided_rotors_20261021/' not in page:
        page=page.replace('<div class="grid">','<div class="grid"><article><h2><a href="../stage55_body_guided_rotors_20261021/桨盘提取对照结果与固定时刻画面.html">55 · 整机框引导桨盘与实际未来误差</a></h2><p>完整对照/前缀/负例通过；局部改善和退化均公开，候选四点仍卡关联和频率质量。</p></article>',1)
    overview.write_text(page,encoding='utf-8')
    spec=importlib.util.spec_from_file_location('checker19_for55',ROOT/'tools/checkpoint_20261019.py');checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    old_save=checker.save
    def routed(path,value):
        if path.name=='current_delivery_checks_20261019.json':path=path.with_name('current_delivery_checks_stage55_20261021.json')
        old_save(path,value)
    checker.save=routed;checker.run(False,'active')
    summary_path=overview.parent/'current_delivery_checks_stage55_20261021.json';summary=load(summary_path)
    assert summary['last_completed_stage']==55 and not summary['in_progress']
    for path in [overview,*folder.glob('*.html')]:
        for ref in re.findall(r'(?:href|src)="([^"]+)"',path.read_text(encoding='utf-8')):
            if '://' not in ref and not ref.startswith('#'):assert (path.parent/ref).exists(),(path,ref)
    selected=[r for r in load(folder/'paired_results.json')['table'] if r['horizon']==.4 and r['case']!='negative5' and r['model'] in ['trusted_acc','adaptive_axis_freq','adaptive_all_mix']]
    summary.update(next_action='Stage56 not started. Reuse unchanged50 LayoutSlots on same55 current geometry with causally known body hints; compare33 association, initialization streak and independent prefix before deciding full RAW frequency/15-model evaluation. Do not relax shape/five-frame gates or treat inferred points as observations.',
        latest_verified_baseline_entry=str(ROOT/'core/run_deploy_forecast.py'),experimental_entry=str(ROOT/'core/run_roi_forecast.py'),
        latest_report=str(folder/'桨盘提取对照结果与固定时刻画面.html'),verified_counts=counts,score_rows=360,exact_old_baseline_or_history_rows=252,
        true_RAW_prefix_parent=225,true_RAW_prefix_native10=450,separate_one_step_diagnostic=True,
        zero_new_ipc_parse_failures=True,original_first_step_failure_preserved=True,original_ipc_cause='Unresolved; raw offending line was not retained. No lenient parsing fix claimed.',
        derived_text_hash_check='Normalize only Windows CRLF to the LF source text originally hashed by freeze; original failed raw-byte comparison preserved. ZIP manifest hashes actual file bytes.',
        future_04s_selected=selected,default_promoted=False,experiments_running=[],current_status_document=str(PROJECT/'PROJECT_STATUS_20261021.md'),
        last_goal_turn_classification='progress: completed frozen paired actual RAW geometry intervention, full prefix/control/negative and true future scoring; evidence shows mixed/no stable gain, identifies geometry-to-track and quality bottlenecks')
    save(summary_path,summary)
    files=set(ROOT.glob('*.py'))|set(ROOT.glob('*.md'))|set(PROJECT.glob('PROJECT_STATUS*.md'))|{PROJECT/'WORKLOG.md'}
    for parent in [ROOT/'src',ROOT/'research',ROOT/'tools']:
        for pattern in ['*.py','*.md']:files.update(p for p in parent.rglob(pattern) if '__pycache__' not in p.parts)
    for number in summary['completed']:
        parent=next(p for p in BASE.glob(f'stage{number}_*') if (p/'protocol.json').exists())
        for pattern in ['*.json','*.md','*.html','derived_*.py']:
            files.update(p for p in parent.rglob(pattern) if p.stat().st_size<2_000_000 and 'failed_border_preparation_v1' not in p.parts)
    files.update(overview.parent.glob('*.html'));files.add(summary_path);files.add(folder/'failed_checkpoint_derived_bytes_v1.py')
    archive=BASE/'checkpoints'/('research_through_stage55_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.zip')
    manifest={p.relative_to(PROJECT).as_posix():sha(p) for p in sorted(files)}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as zipped:
        for p in sorted(files):zipped.write(p,p.relative_to(PROJECT).as_posix())
        zipped.writestr('SNAPSHOT_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2));zipped.writestr('HANDOFF_CHECKPOINT.json',json.dumps(summary,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        for name,value in manifest.items():assert hashlib.sha256(zipped.read(name)).hexdigest()==value
    summary.update(archive=str(archive),archive_sha256=sha(archive),source_summary_files=len(files),zip_verified=True,archive_scope='Sources/protocols/small reports; RAW, weights, PNG/MP4 and full JSONL retained in original directories. Original partial failed output retained in workspace.')
    save(BASE/'checkpoints/latest_handoff.json',summary);save(BASE/'checkpoints/latest_progress_20261021.json',summary)
    print(json.dumps(dict(stage=55,goal_status='active',counts=counts,files=len(files),archive=str(archive),verified=True),ensure_ascii=False))

if __name__=='__main__':run()
