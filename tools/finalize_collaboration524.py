"""Finalize a verified collaboration artifact and update local handoff only."""
from pathlib import Path
from datetime import datetime
import hashlib,json,re,shutil,zipfile
BASE=Path(__file__).resolve().parents[2];ROOT=BASE/'event_drone_forecast'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if __name__=='__main__':
    parent=BASE/'output/collaboration_20261113';out=parent/'event-drone-forecast-collab'
    summary=json.loads((parent/'verification/raw_smoke/summary.json').read_text('utf8'))
    assert summary['frames']==summary['native_packets_read']==4 and summary['skipped_observations']==0
    old=json.loads((BASE/'output/raw_bg_memory517_20261113/protocol.json').read_text('utf8'))['source_sha256']
    assert all(sha(ROOT/f)==h for f,h in old.items()),'Original517 source changed during packaging'
    provenance=json.loads((out/'SOURCE_PROVENANCE.json').read_text('utf8'))
    provenance.pop('original_workspace_untouched',None);provenance['original_runtime_sources_untouched']=True
    (out/'SOURCE_PROVENANCE.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),'utf8')
    changes=out/'CHANGES_FROM_RESEARCH.md'
    changes.write_text(changes.read_text('utf8').replace('原工程目录和既有发布包保留原状。','原运行源码和既有发布包保持原状；工作日志另行更新交接。'),'utf8')
    (out/'docs/VALIDATION.md').write_text('''# 本次打包检查

- 逐文件SHA256清单与全部随附Python源码语法检查通过。
- CPU已有状态示例能运行，0.1/0.2/0.4秒传播结果与保存值一致。
- 配置SDK路径后，当前研究入口的`--help`在独立包目录成功加载。
- 统一启动器dry-run与PowerShell几何编译脚本语法检查通过。
- 独立包实际启动SDK读流、CUDA工作进程与自有几何DLL，完成0.1秒RAW冒烟检查：4包、零跳包、正常退出。
- 本次冒烟使用FRED11已有本地副本；该区间无有效桨叶输出，因此只证明依赖和运行链可启动，不是精度、召回或FPS验收。
- 原工程517源码和参数SHA256逐份保持不变。包内唯一数值文件差异为OpenCV DLL路径的环境变量适配。
- 未在协作者电脑运行，不承诺未经环境配置即可跨机运行。

测试输出与本机配置留在包外；源码ZIP不附RAW、个人环境路径配置或字节码缓存。
''','utf8')
    files={str(p.relative_to(out)).replace('\\','/'):sha(p) for p in sorted(out.rglob('*'))
        if p.is_file() and p.name!='MANIFEST.sha256.json' and '__pycache__' not in p.parts}
    forbidden={'.raw','.onnx','.pt','.tmp_index','.env'}
    secrets=re.compile(r'sk-[A-Za-z0-9]{32,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----')
    for name in files:
        p=out/name;assert p.suffix.lower() not in forbidden,name
        assert name!='configs/local.json' and not name.endswith('.pyc'),name
        if p.suffix.lower() in ('.py','.md','.json','.txt','.ps1','.cpp','.cu','.yaml','.yml','.toml'):
            assert not secrets.search(p.read_text('utf8',errors='replace')),f'Potential credential in{name}'
    manifest=dict(files=files,manifest_excludes_itself=True,cache_and_local_configuration_excluded=True)
    (out/'MANIFEST.sha256.json').write_text(json.dumps(manifest,indent=2),'utf8')
    zip_path=parent/'event-drone-forecast-collab-20261113.zip'
    assert not zip_path.exists()
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name in sorted(list(files)+['MANIFEST.sha256.json']):z.write(out/name,f'{out.name}/{name}')
    with zipfile.ZipFile(zip_path) as z:
        assert z.testzip() is None and len(z.namelist())==len(files)+1
        for name,h in files.items():assert hashlib.sha256(z.read(f'{out.name}/{name}')).hexdigest()==h
    digest=sha(zip_path);(parent/(zip_path.name+'.sha256')).write_text(f'{digest}  {zip_path.name}\n','ascii')
    # Authoritative local handoff supersedes historical banners, not source.
    handoff=ROOT/'docs/HANDOFF_20261113_COLLABORATION.md'
    handoff.write_text('# 当前协作交接（2026-11-13）\n\n'+
        f'源码包：`{zip_path}`。独立源码目录：`{out}`。先读包内README、docs/HANDOFF.md、docs/RESULTS.md。\n\n'+
        (out/'docs/HANDOFF.md').read_text('utf8'),'utf8')
    work=BASE/'WORKLOG.md';marker='<!-- collaboration523-current:start -->'
    assert marker not in work.read_text('utf8')
    banner=f'''{marker}
当前权威交接：[协作交接](event_drone_forecast/docs/HANDOFF_20261113_COLLABORATION.md)；[协作源码包](output/collaboration_20261113/event-drone-forecast-collab-20261113.zip)。517实际六段RAW、原102点均29.997218px，中位24.904973；487基线31.471473。开发30px门槛余量仅0.002782px，不能称稳定泛化达标。原预留录像518为标定板阴性，两入口25查询0误报；按用户最新要求转FRED11固定10–16秒522，两入口0/25有效输出、FDE=null，原生画面与官方坐标对齐仍待复核，新录像验收未过。当前主线纯ON/OFF桨叶＋五圆/可见性＋可见贡献/不可测合力，未改NN发布包。523/524整理源码与依赖、保留基线、补环境与协作文档；独立包通过CPU示例、SDK入口加载、4包RAW启动检查。所有RAW/GPU作业已退出，无后台任务。以下为历史。
<!-- collaboration523-current:end -->

'''
    work.write_text(banner+work.read_text('utf8'),'utf8')
    readme=ROOT/'README.md'
    readme.write_text('<!-- collaboration523-current:start -->\n'+
        '> 当前协作版本（2026-11-13）：517纯桨叶研究线，原102开发点实际RAW均29.997218px；FRED11新录像0/25有效输出，泛化未通过，持续30FPS未通过。'+
        '[最新交接](docs/HANDOFF_20261113_COLLABORATION.md)及[独立源码包](../output/collaboration_20261113/event-drone-forecast-collab-20261113.zip)。下方为历史。\n'+
        '<!-- collaboration523-current:end -->\n\n'+readme.read_text('utf8'),'utf8')
    p=BASE/'output/checkpoints/latest_handoff.json';state=json.loads(p.read_text('utf8'))
    backup=p.parent/'history/latest_handoff_before_collaboration_20261113.json';backup.parent.mkdir(exist_ok=True)
    assert not backup.exists();shutil.copy2(p,backup)
    actual=json.loads((BASE/'output/raw_bg_memory517_20261113/results.json').read_text('utf8'))
    state.update(timestamp=datetime.now().astimezone().isoformat(),research_objective_status='development_gate_met_generalization_unresolved',
        objective_completed=False,in_progress=False,background_jobs=[],experiments_running=[],current_running_jobs=[],
        status='Collaboration source snapshot523/524 finalized.517 actual6 n102 mean29.997218;FRED11 fixed10–16s both0/25 valid forecasts, FDE=null. No new-video accuracy pass or30FPS. All jobs exited.',
        current_status_document=str(handoff),latest_report=str(out/'docs/RESULTS.md'),
        candidate_entry=str(ROOT/'core/run_rotor_background_memory_candidate.py'),best_experimental_entry=str(ROOT/'core/run_rotor_background_memory_candidate.py'),
        current_precision_mean_px=actual['pooled']['background']['mean'],best_experimental_mean_px=actual['pooled']['background']['mean'],
        aggregate_same102=actual['pooled']['background'],candidate_scores={n:c['background']['original_available'] for n,c in actual['cases'].items()},
        milestone31_achieved=True,milestone30p5_achieved=True,development30_gate_achieved=True,
        independent_new_drone_video_validated=False,FPS30_proven=False,current_target_FDE_px=30.,next_target_FDE_px=30.,
        collaboration_zip=str(zip_path),collaboration_folder=str(out),collaboration_sha256=digest,
        external518_execution='Completed after actual<=30 gate; calibration board, no drone;25 reviewed queries0 false positives; no FDE.',
        external522_execution='FRED11 user-directed source, fixed10–16s, both algorithms0/25 valid; full native/official alignment review pending; now exposed.',
        next_action='First inspect native FRED11 target/official-coordinate alignment and locate which rotor-evidence/startup stage loses all outputs; no per-video threshold tuning.')
    state['single_root_focus'].update(actual517_mean_px=29.997218068634453,
        current_source_freeze='No active batch. Original517 source hashes unchanged; packaged copy has one documented DLL-path adaptation.',
        background_coordinate_fix_has_development_gain=True,new_video_generalization_not_passed=True)
    p.write_text(json.dumps(state,ensure_ascii=False,indent=2),'utf8')
    result=dict(zip=str(zip_path),folder=str(out),files=len(files)+1,zip_bytes=zip_path.stat().st_size,sha256=digest,
        source_tree_preserved=True,all_jobs_exited=True,SDK_GPU_RAW_smoke_packets=summary['frames'])
    (parent/'DELIVERY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf8');print(json.dumps(result,ensure_ascii=False,indent=2))
