"""Verify the new local artifacts and save an ongoing handoff checkpoint."""
from pathlib import Path
import datetime,hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;OUT=WORK/'output/optimization_run_20261025'
portable=WORK/'output/portable_position_core_20261025'
source=OUT/'source_candidate';archive=OUT/'source_candidate.zip'
manifest=json.loads((source/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert all('helix_reference' not in n and not n.endswith(('.raw','.onnx','.dll','.pyd'))
               and 'configs/local.json' not in n for n in z.namelist())
for row in manifest['files']:
    p=source/row['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],p
wheel=json.loads((portable/'wheel_manifest.json').read_text(encoding='utf-8'))
installed=json.loads((portable/'pip_target_equivalence.json').read_text(encoding='utf-8'))
smoke=json.loads((portable/'RAW_live_smoke/completion.json').read_text(encoding='utf-8'))
assert installed['rows']==632 and installed['all_coefficients_and_forecasts_exact']
assert smoke['sources_unchanged'] and smoke['publication_slots']==6
delivery=dict(source_archive=str(archive),source_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    source_manifest_files=len(manifest['files'])+1,source_archive_bytes=archive.stat().st_size,
    CRC_and_file_hashes_verified=True,raw_weights_private_config_or_author_DLL_included=False,
    wheel=wheel,pip_installed_core=installed,RAW_smoke=smoke,
    no_full_clean_machine_RAW_claim=True,goal_status='active')
(OUT/'delivery_checkpoint.json').write_text(json.dumps(delivery,ensure_ascii=False,indent=2),encoding='utf-8')
h_path=WORK/'output/checkpoints/latest_handoff.json';h=json.loads(h_path.read_text(encoding='utf-8'))
h['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
h['execution_status']='active optimization; portable position core completed, no current subprocesses'
h['in_progress']='Investigate reducing native event payload IPC to improve observation freshness without changing frames'
h['next_action']='Profile ProcessReplay transport; reuse shared memory for identical native frames if appropriate. Preserve frame hashes/time cutoff and independent publication clock.'
h['experiments_running']=False;h['running_jobs']=[]
h['latest_portable_position_core_20261025']=delivery
h['latest_optimization_20261025']['latest_source_candidate']=str(archive)
h['latest_optimization_20261025']['source_archive_sha256']=delivery['source_archive_sha256']
h['latest_optimization_20261025']['minimal_position_core_extracted']=True
h_path.write_text(json.dumps(h,ensure_ascii=False,indent=2),encoding='utf-8')
marker='### 位置核心迁移与本地候选包（2026-10-25）'
log=WORK/'WORKLOG.md'
if marker not in log.read_text(encoding='utf-8'):
    with log.open('a',encoding='utf-8') as f:
        f.write('\n\n'+marker+'''\n\n0.3.1已提取可安装event_drone_forecast.position。对迁移前632个实际输入，两噪声模型在源码、移出原目录wheel及实际pip离线安装新目录后，全部系数、预测、有效性与诊断严格保持；测试禁止读原research/external且不导入SDK。根目录live数学文件现为兼容导入。固定FilterPy作者源/元数据/许可证随包保留，原research链未搬动。

wheel为output/portable_position_core_20261025/wheels/event_drone_forecast-0.3.1-py3-none-any.whl，33文件、69,943字节，SHA256=2747f1d79a94452d96646eafa0530a04712fa0f2da3452199bc5088ac2b3a2cb。包只提取了位置核心，旧控制台仍历史流程，完整RAW/CUDA链仍需源码、SDK和兼容权重，未做新机器全链复现。仅安装了固定哈希wheel0.45.1构建工具，没有改SDK/系统Python。

实际live RAW侧视1.2秒，65原生片含预热，25次跟踪，6次独立发布、3次有效。源/共享帧校验通过；年龄P95约231ms、计算中位40.61ms，不能称连续有效5Hz或速度/精度升级。当前仍有解码/传输/计算延迟问题，下一步针对进程间事件大载荷复制，争取用最新原生画面减少额外等待，不继续调物理权重。

新的本地源码候选output/optimization_run_20261025/source_candidate.zip：841文件、36,978,546字节，SHA256=a821efd74ebe2c2b405f53674f06331ac38ef62087b94b6e4a007e95325a8347，CRC与全部文件哈希已核对。不含RAW/权重/SDK/私有配置/Helix作者代码与DLL，未上传远端。它是此刻快照，后续修改不能自动称已入包；完整清单在delivery_checkpoint.json。报告index.html已包含相位/侧视视频与负结果。goal保持active，当前无本轮运行中的子进程。
''')
print(json.dumps(dict(source_files=delivery['source_manifest_files'],archive_sha256=delivery['source_archive_sha256'],wheel_rows=installed['rows'],goal='active')),flush=True)
