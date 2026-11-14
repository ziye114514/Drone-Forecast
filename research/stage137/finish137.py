"""Small checked handoff after one actual execution optimization."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlparse,unquote
import datetime,hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT.parent;OUT=WORK/'output/stage137_fast_geometry_20261024'
def load(p):return json.loads(Path(p).read_text('utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),'utf-8')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['src','href','poster'] and v)

def main():
    summary=load(OUT/'summary137.json');report=Path(summary['report']);parser=Links();parser.feed(report.read_text('utf-8'));checked=0
    for link in parser.links:
        u=urlparse(link)
        if u.scheme or link.startswith('#'):continue
        assert (report.parent/unquote(u.path)).resolve().is_file(),link;checked+=1
    for name in ['legacy100','natural6','side50','negative5']:
        p=load(OUT/name/'protocol.json');c=load(OUT/name/'completion.json');assert c['sources_unchanged'] and not c['labels_read'] and not c['output_caches_read']
        for rel,digest in p['sources'].items():assert sha(ROOT/rel)==digest,rel
        stat=Path(p['raw']).stat();assert (stat.st_size,stat.st_mtime_ns)==(p['raw_size'],p['raw_mtime_ns'])
    hpath=WORK/'output/checkpoints/latest_handoff.json';handoff=load(hpath);assert handoff['last_completed_stage']==136;previous=Path(handoff['archive']);previous_sha=handoff['archive_sha256'];assert sha(previous)==previous_sha
    intro=f'''## 137：几何实际加速，预测结果保持

2026-10-24。[本轮结果和视频](<{report.as_posix()}>)；[运行说明](<{(ROOT/'research/stage137/README.md').as_posix()}>)。当前入口仍`D:/drone forecast/event_drone_forecast/run_optimized_forecast.py`，默认fast；同入口可用`--geometry-backend legacy`运行旧136几何。默认54/API93不变。

- 先剖析2秒RAW：固定结构Jacobian与轮廓前端是热点。改为四桨/八参数批量解析导数；复用124计数掩膜缓存；占据0.5固定阈值去掉未用全图分位数。保留原目标函数/优化器/初始化/迭代数/全图阈值/完整轮廓/FOV及五帧门槛，不直接裁ROI。
- 同入口共同100步：整步233.279→189.984ms（18.56%），几何171.640→117.788ms（31.37%），结构拟合111.413→58.098ms（47.85%），前端72.399→63.758ms。顺序运行、系统负载未受控，不作跨硬件基准。完整新400步180.659ms，另序列化/视频12.983ms，总含冷启动读取视频97.906秒，仍非20ms实时。
- 新400步机体/前端观测/当前几何资格精确旧136，预测坐标最大差1.751e-11px；同11发布端点0.1/0.2/0.4秒8.578/11.312/30.834px，覆盖口径沿用136，仍是已暴露开发录像和人工未来框中心。本轮是速度收益，不是新增精度/侧视/RPM/创新证明。
- 26组实际输入导数与原实现最大差1.706e-12，独立方向数值差分通过；交替同输入导数1.814→0.629ms。实际RAW旧100＋新清晰400＋侧视50＋空场30＝580；另剖析100不作速度对照，总680。空场无有效预测；cutoff/无缓存标签输入/源码指纹/视频400帧解码通过。侧视仅机体及预测比较，未声称逐项几何核验或运动收益。
- 首次cProfile相对路径在NN启动前指纹失败，profile_raw空目录/日志保留；绝对路径剖析完成，入口Path(__file__).resolve修复兼容。136完整源快照保存research/stage136/run_optimized_forecast_v3.py，仅历史指纹。未改线程配置、依赖、OS或保护目录，无G。

下一项实际优化：保留全图阈值统计的ROI共享支持，对跨ROI轮廓回退全图；全图前端/几何/GPU仍有开销。再接侧视线法线与真实周期，不能用事件总数冒充转速/推力。小固定段比较耗时与真实预测，不扩张历史控制树。本轮全部作业已退出。

'''
    worklog=WORK/'WORKLOG.md';text=worklog.read_text('utf-8');assert '## 138. 137：' not in text;worklog.write_text(text.rstrip()+'\n\n## 138. 137：几何计算加速\n\n'+intro,'utf-8')
    for path in [WORK/'PROJECT_STATUS_20261022.md',ROOT/'README.md']:path.write_text(intro+'以下136及更早为历史。\n\n'+path.read_text('utf-8'),'utf-8')
    handoff.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),last_completed_stage=137,algorithm_stage=135,geometry_execution_stage=137,latest_full_forecast_stage=137,experimental_RAW_stage=137,latest_RAW_geometry_stage=137,latest_RAW_pilot_stage=137,latest_diagnostic_stage=137,
        latest_full_forecast_report=str(report),latest_report=str(report),latest_checks=str(OUT/'final_checks137.json'),experimental_RAW_entry=str(ROOT/'core/run_optimized_forecast.py'),experimental_RAW_pilot_entry=str(ROOT/'core/run_optimized_forecast.py'),latest_completed_stages=[137],
        next_action='Implement ROI shared support while retaining global threshold statistics; contours crossing ROI must fall back to global. Preserve full boundary and lock gates. Continue with side-line normals and genuine periodic evidence, not raw event count as thrust. One small fixed clip per actual change; no growth of historical control tree.',
        completion_scope='137 actual execution optimization only: same135 prediction/129 qualification; batched fixed-shape Jacobian +124 count mask reuse +unused occupancy quantile removal. Common100 step233.279->189.984ms;geometry171.640->117.788. Full400180.659ms; forecasts maxdiff1.75e-11px,0.4s30.834px. RAW580 plus profiled100; side50 only unchanged body/forecast, no side physics improvement. Still not realtime or independent innovation result.',
        source_summary_files=[str(worklog),str(WORK/'PROJECT_STATUS_20261022.md'),str(ROOT/'README.md'),str(ROOT/'research/stage137/README.md')],running_jobs=[],report_local_links_checked=checked,
        prior_archive=str(previous),prior_archive_sha256=previous_sha,optimization_current_full_frames=400,optimization_this_stage_actual_RAW_frames=580,optimization_this_stage_profile_RAW_frames=100,
        prediction_logic_unchanged_from_stage135=True,side_improved=False,realtime_proven=False,default_promoted=False,execution_status='checkpoint_ready',research_objective_status='incomplete',
        archive_scope='Verified136 base plus137 sources/protocols/small results/plots/profiles/report/new entry and136 original full-entry snapshot. RAW/weights/JSONL/videos remain in original paths. Embedded handoff omits own archive SHA.')
    handoff['completed']=sorted(set(handoff.get('completed',[]))|{137});handoff['preserved_negative_results']+=['137 first cProfile relative-entry fingerprint failure before NN; empty directory/log preserved; absolute profile completed and path handling fixed']
    handoff.pop('archive',None);handoff.pop('archive_sha256',None);handoff['zip_verified']=False;save(hpath,handoff)
    save(OUT/'final_checks137.json',dict(report_links_checked=checked,all_four_completed_runs_checked=True,actual_RAW580_plus_profile100=True,current_full400=True,source_fingerprints_unchanged=True,raw_size_mtime_unchanged=True,RAW_bytewise_checksum_not_done=True,plots_preview_visually_checked=True,video400_decoded=True,prediction_improvement_this_stage=False,realtime=False,running_jobs=[]))
    current={}
    for p in [worklog,WORK/'PROJECT_STATUS_20261022.md',ROOT/'README.md',hpath,ROOT/'core/run_optimized_forecast.py',ROOT/'research/stage136/run_optimized_forecast_v3.py']:current[p.relative_to(WORK).as_posix()]=p.read_bytes()
    for folder in [ROOT/'research/stage137',OUT]:
        for p in folder.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix.lower() in ['.py','.json','.html','.png','.log','.md','.pstats']:current[p.relative_to(WORK).as_posix()]=p.read_bytes()
    now=datetime.datetime.now(datetime.timezone.utc);archive=WORK/'output/checkpoints'/('research_through_stage137_20261024_'+now.strftime('%H%M%S_%f')+'.zip');hashes={};manifestname='output/stage137_fast_geometry_20261024/recovery_manifest137.json'
    with zipfile.ZipFile(previous) as old,zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as new:
        assert old.testzip() is None
        for name in old.namelist():
            if name in current or name==manifestname:continue
            data=old.read(name);new.writestr(name,data);hashes[name]=hashlib.sha256(data).hexdigest()
        for name,data in current.items():new.writestr(name,data);hashes[name]=hashlib.sha256(data).hexdigest()
        new.writestr(manifestname,json.dumps(dict(base_archive=str(previous),base_sha256=previous_sha,count=len(hashes),files=hashes,scope=handoff['archive_scope']),ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,digest in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
    digest=sha(archive);handoff.update(archive=str(archive),archive_sha256=digest,zip_verified=True);save(hpath,handoff);save(OUT/'archive_checks137.json',dict(path=str(archive),sha256=digest,entries=len(hashes)+1,CRC_and_all_entry_SHA_verified=True,embedded_handoff_omits_own_archive_SHA=True))
    print(json.dumps(dict(report=str(report),archive=str(archive),sha256=digest,files=len(hashes)+1)),flush=True)
if __name__=='__main__':main()
