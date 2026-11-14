"""Checked handoff, notes and recoverable archive after complete experiments."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlparse,unquote
import datetime,hashlib,json,sys,zipfile
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT.parent
sys.path[:0]=[str(ROOT/'research/stage128')]
from study128 import OUT,verify,rows,save,sha
def load(p):return json.loads(Path(p).read_text('utf-8-sig'))
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ['src','href','poster'] and v:self.links.append(v)
def main():
    verify();d=load(OUT/'delivery_checks128.json');report128=Path(d['report']);final129=WORK/'output/stage129_FOV_guard_20261024';d129=load(final129/'delivery129.json');report=Path(d129['report']);checked=0
    for case in ['prefix_natural6','negative5']:assert load(final129/'raw'/case/'checks129.json')['new_guard_exact129']
    for page in [report128,report]:
        parser=Links();parser.feed(page.read_text('utf-8'))
        for link in parser.links:
            u=urlparse(link)
            if u.scheme or link.startswith('#'):continue
            p=(page.parent/unquote(u.path)).resolve();assert p.is_file(),p;checked+=1
    # Characterize preserved partial output without treating it as a run.
    partial=OUT/'interrupted_natural6_20261024/observations_forecasts.jsonl';n=0;bad=0
    for s in partial.read_text('utf-8').splitlines():
        try:json.loads(s);n+=1
        except json.JSONDecodeError:bad+=1
    save(OUT/'interruption128.json',dict(cause='Intentional session interruption; no Python processes remained when checked2026-10-24. Complete frozen rerun, no parameter change or state splice.',
        preserved_directory=str(partial.parent),complete_json_records=n,incomplete_tail_lines=bad,completion_present=(partial.parent/'completion.json').exists(),not_used_in_scores=True))
    failed=OUT/'failed_memory_extension2_20261024/observations_forecasts.jsonl';memory_n=0
    for line in failed.read_text('utf-8').splitlines():
        try:json.loads(line);memory_n+=1
        except json.JSONDecodeError:pass
    save(OUT/'execution_repair128.json',dict(preserved_directory=str(failed.parent),complete_failed_records=memory_n,
        error='Original73 FastSupports.step NumPy3.52MiB float32(720,1280) allocation failure. No completion; not scored.',
        change='Heavy checker/build parent and overlapping analysis avoided during prediction. Standard-library-only serial_runner128.py waits for all inference workers to exit before checker; plots after all jobs.',
        frozen_inference_unchanged=True,thread_settings_unchanged=True,OS_settings_unchanged=True,other_apps_untouched=True,
        snapshots_not_at_exact_failure=[dict(phase='After failed process exit',FreePhysicalMemory_KiB=1893216,FreeVirtualMemory_KiB=10375284),dict(phase='During lightweight retry',FreePhysicalMemory_KiB=1028584,FreeVirtualMemory_KiB=1636636)],
        cause_not_isolated_experiment=True,full_retry_checked=True))
    intro=f'''## 124—129：占据支持与完整性修复，预测收益未成立

2026-10-24收尾。[修正后的最终报告与RAW小试视频](<{report.as_posix()}>)；[129复现与入口](<{(ROOT/'research/stage129/README.md').as_posix()}>)。完整RAW实验仍128；129追加90时刻实际RAW守门小试，入口run_FOV_guard_forecast.py，原71路保留。默认54/API93不改，129预测退步，不推荐替换主路径。

- 修复对象：原native20归一计数≥2把原生双极黑色支持压成小核。新增同像素ON/OFF二值占据、3×3 Gaussian≥.5及原开闭/防合并/先判线；四路全局/同ROI对照，原104资格与连续五帧锁定均不改。
- 124：实际RAW几何1505＋两个60前缀＝1625时刻。2号四候选0→72、完整合格0→7，最长仅2，仍无首次模型。占据独用清晰当前196/400低于计数236/400，未替换旧前端；连续侧视仍无首次模型。空场0/30。候选数量不是可见性真值。
- 125/126负结果：把不同支持的椭圆混进同一已学结构，当前仍236。126清晰实际RAW400逐项精确123/124/125，原69路保留；其他126单独计划由128保留控制，不称已单独执行。
- 127/128：两套支持各自学结构，再把当前合格观测送进同一107位置/速度/偏移KF。原表面236→244、22.80→22.68不能保留为可靠收益：视频QA发现新增8次中6次拟合椭圆超出FOV却标记完整。仅count坐标适配严格等价，短前缀通过；机制不称创新/COM标定。
- 128完整RAW790时刻（400清晰/300扩展/30负例/60活动前缀）；原69/当前NN精确123、清晰原70精确126；新占据和模型精确124、新简单混合精确125、新共享滤波精确127。禁止缓存/标注输入，cutoff、独立活动前缀和全视频解码通过。
- 所有128预测完毕再读稀疏人工未来框。{d['outcome']}是框中心而非COM。129为视频QA后开发诊断，规则非按端点调参，但不是独立验证。
- 129修复：旋转椭圆解析包围边界超出相机则保留为arc/inferred_shape，禁止完整四/三桨资格。缓存1505时刻＋短前缀：清晰235/400（count233/占据2），首次22.86，其他正例0；追加真实RAW正例60/空场30，原71精确128、守门精确129、cutoff/空场通过。{d129['outcome']}129没有新8秒全链路或实时性/RPM/COM证明。
- 中断输出保留：128第一次约{n}条完整记录、无completion，保存在interrupted_natural6_20261024，未拼接/计分；确认进程退出后同冻结源码重新运行。开发两帧先于冻结，不称独立验证。
- 扩展2运行{memory_n}条记录后原73支持数组分配失败，原目录/日志保留failed_memory_extension2_20261024；改轻量标准库调度，预测退出后再核验/绘图，没有改推理、线程或OS设置，完整重跑通过后才评分。

下一步：用ROI内逐像素时间周期区分桨盘与机体边缘；拆分所学形状、初始化图像偏移和位置状态的误差，避免更严格观测改变参考系后误差放大。再验证重叠/截断与连续首次建模。事件数不直接等于转速/推力，不能靠保留越界观测恢复成绩。保留全部负结果；本轮不宣称卡点全解或创新成立。E盘只读、yolov10-main未访问/修改、无G/新依赖。

'''
    worklog=WORK/'WORKLOG.md';text=worklog.read_text('utf-8')
    marker='## 136. 124—'
    if marker in text:text=text[:text.index(marker)].rstrip()+'\n'
    worklog.write_text(text+'\n## 136. 124—129：占据支持与完整性修复收尾\n\n'+intro,'utf-8')
    for path in [WORK/'PROJECT_STATUS_20261022.md',ROOT/'README.md']:
        old=path.read_text('utf-8')
        if not old.startswith('## 124—129：'):path.write_text(intro+'以下123及更早为历史。\n\n'+old,'utf-8')
    hpath=WORK/'output/checkpoints/latest_handoff.json';handoff=load(hpath);previous=Path(handoff.get('archive') or handoff['prior_archive']);previous_sha=handoff.get('archive_sha256') or handoff['prior_archive_sha256'];assert sha(previous)==previous_sha
    handoff.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution_status='checkpoint_ready',research_objective_status='incomplete',last_completed_stage=129,
        algorithm_stage=129,latest_full_forecast_stage=128,experimental_RAW_stage=128,latest_RAW_geometry_stage=129,latest_diagnostic_stage=129,latest_RAW_pilot_stage=129,
        latest_full_forecast_report=str(report128),latest_report=str(report),latest_checks=str(final129/'final_checks129.json'),experimental_RAW_entry=str(ROOT/'run_independent_support_forecast.py'),experimental_RAW_pilot_entry=str(ROOT/'run_FOV_guard_forecast.py'),recommended_forecast_research_stage=122,
        next_action='ROI pixel temporal periodicity to reject body edges; separate shape-learning and first-frame image gauge bias from position dynamics.129 FOV guard withdraws invalid early lock,235 current and37.642px futureBOX vs33.679 old. Keep diagnostic, do not promote. Side/overlap startup still0; no fake hubs/gate relaxation. CLI/process-safe API; lightweight scheduler then checker/plots, not concurrent heavy parents.',
        completion_scope='Full RAW128790 with original69 exact123/old70 clear exact126;129 additional RAW90 pilot preserves original71. Apparent8-update early-lock gain retracted after6 outside-FOV fits; guard corrects completeness but cached futureBOX regresses. Full1298s/side/COM/RPM/realtime and independent innovation benefit incomplete.',
        source_summary_files=[str(WORK/'PROJECT_STATUS_20261022.md'),str(worklog),str(ROOT/'README.md'),str(ROOT/'research/stage129/README.md')],
        prior_archive=str(previous),prior_archive_sha256=previous_sha,latest_completed_stages=list(range(124,130)),original69_numerically_preserved=True,old70_overlap400_numerically_preserved=True,old71_active_prefix60_preserved=True,
        report_local_links_checked=checked,running_jobs=[],archive_scope='Verified123 base plus124..129 source/protocols/small results/plots/new handoff; RAW, weights, JSONL and videos retained at original paths. Embedded handoff omits self archive SHA.')
    handoff['completed']=sorted(set(handoff.get('completed',[]))|set(range(124,130)))
    old_negatives=[v for v in handoff.get('preserved_negative_results',[]) if v!='128 interrupted approximately79-frame and NumPy memory failure75 kept; lightweight serial retries complete']
    handoff['preserved_negative_results']=list(dict.fromkeys(old_negatives+['124 occupancy-alone clear196 vs count236; side/extension startup still0','125 mixed-template fallback no coverage benefit; preserved126 control','127/128 apparent clear236->244 early-lock retracted:6/8 outside-FOV fits incorrectly full','129 conservative full-boundary guard235 current;futureBOX37.642 worse than33.679; actualRAW only90 pilot',f'128 interrupted{n} complete records and memory-failed{memory_n} complete records kept; coarse logs printed79/75, lightweight serial retries complete']))
    handoff.pop('archive',None);handoff.pop('archive_sha256',None);handoff['zip_verified']=False;save(hpath,handoff)
    # Recheck frozen RAW byte sizes/mtimes against124 protocol, no expensive
    # multi-GB checksum and no claim that mtime proves byte-for-byte equality.
    protocol124=load(WORK/'output/stage124_occupancy_support_20261023/protocol124.json')
    for raw,s in protocol124['raw_metadata'].items():
        stat=Path(raw).stat();assert s==dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
    save(OUT/'final_checks128.json',dict(report_local_links_checked=checked,interrupted_records_preserved=n,all_checks_complete=True,default_promoted=False,RAW_input_files_size_mtime_unchanged=True,bytewise_RAW_checksum_not_performed=True))
    save(final129/'final_checks129.json',dict(actual128_RAW790_complete=True,actual129_RAW90_pilot_complete=True,original71_prefix60_exact=True,guard_current=235,guard_first22_86=True,apparent_early_gain_retracted=True,guarded_future_error_px=d129['guarded_score'],full129_end_to_end8s=False,default_promoted=False,report_local_links_checked=checked))
    current={}
    for path in [worklog,WORK/'PROJECT_STATUS_20261022.md',ROOT/'README.md',hpath,ROOT/'run_support_forecast.py',ROOT/'run_independent_support_forecast.py',ROOT/'run_FOV_guard_forecast.py']:
        current[path.relative_to(WORK).as_posix()]=path.read_bytes()
    for stage in range(124,130):
        for path in (ROOT/f'research/stage{stage}').rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:current[path.relative_to(WORK).as_posix()]=path.read_bytes()
    for folder in [WORK/'output/stage124_occupancy_support_20261023',WORK/'output/stage125_complementary_support_20261023',WORK/'output/stage126_RAW_complementary_support_20261023',WORK/'output/stage127_independent_support_20261023',OUT,final129]:
        for path in folder.rglob('*'):
            if path.is_file() and path.suffix.lower() in ['.json','.html','.png','.log','.md'] and not any(v.startswith('interrupted_') for v in path.relative_to(folder).parts):
                current[path.relative_to(WORK).as_posix()]=path.read_bytes()
    # Keep small interrupted protocol/log evidence, large RAW/JSONL/video stay.
    for path in [OUT/'interrupted_natural6_parent_20261024.log',OUT/'interrupted_natural6_20261024/protocol.json']:
        current[path.relative_to(WORK).as_posix()]=path.read_bytes()
    now=datetime.datetime.now(datetime.timezone.utc);archive=WORK/'output/checkpoints'/('research_through_stage129_20261024_'+now.strftime('%H%M%S_%f')+'.zip');hashes={}
    with zipfile.ZipFile(previous) as old,zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as new:
        assert old.testzip() is None
        for name in old.namelist():
            if name in current or name=='output/stage129_FOV_guard_20261024/recovery_manifest129.json':continue
            data=old.read(name);new.writestr(name,data);hashes[name]=hashlib.sha256(data).hexdigest()
        for name,data in current.items():new.writestr(name,data);hashes[name]=hashlib.sha256(data).hexdigest()
        manifest=dict(base_archive=str(previous),base_sha256=previous_sha,files=hashes,count=len(hashes),scope=handoff['archive_scope'])
        new.writestr('output/stage129_FOV_guard_20261024/recovery_manifest129.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,digest in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
    digest=sha(archive);handoff.update(archive=str(archive),archive_sha256=digest,zip_verified=True);save(hpath,handoff)
    save(final129/'archive_checks129.json',dict(path=str(archive),sha256=digest,files=len(hashes)+1,CRC_and_all_entry_SHA_verified=True,embedded_handoff_omits_own_archive_SHA=True))
    save(WORK/'output/checkpoints/in_progress_stage128_20261024.json',dict(state='complete',client_date='2026-10-24',latest_handoff=str(hpath),report=str(report),archive=str(archive),archive_sha256=digest,running_jobs=[]))
    print(json.dumps(dict(report=str(report),archive=str(archive),sha256=digest,files=len(hashes)+1,links=checked)),flush=True)
if __name__=='__main__':main()
