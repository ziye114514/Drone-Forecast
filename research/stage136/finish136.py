"""Update the existing worklog/handoff and one recoverable source snapshot."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlparse
import datetime,hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT.parent
OUT=WORK/'output/stage135_optimized_forecast_20261024'
def load(p):return json.loads(Path(p).read_text('utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),'utf-8')
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        self.links.extend(v for k,v in attrs if k in ['src','href','poster'] and v)

def main():
    d=load(OUT/'delivery136.json');report=Path(d['report']);parser=Links();parser.feed(report.read_text('utf-8'));checked=0
    for link in parser.links:
        u=urlparse(link)
        if u.scheme or link.startswith('#'):continue
        p=(report.parent/unquote(u.path)).resolve();assert p.is_file(),p;checked+=1
    for name in ['natural6','natural6_fast','prefix_natural6','negative5']:
        c=load(OUT/'raw'/name/'checks135.json');assert c['predictor_exact135_replay'];assert c['maximum_coefficient_difference']==0
    latest=load(OUT/'raw/natural6_fast/protocol.json');completion=load(OUT/'raw/natural6_fast/completion.json')
    assert completion['sources_unchanged'] and not completion['labels_read'] and not completion['output_caches_read']
    for rel,digest in latest['sources'].items():assert sha(ROOT/rel)==digest,rel
    stat=Path(latest['raw']).stat();assert stat.st_size==latest['raw_size'] and stat.st_mtime_ns==latest['raw_mtime_ns']
    handoffpath=WORK/'output/checkpoints/latest_handoff.json';handoff=load(handoffpath);previous=Path(handoff['archive']);previous_sha=handoff['archive_sha256'];assert sha(previous)==previous_sha
    assert handoff['last_completed_stage']==129,'Avoid overwriting another completed handoff.'
    intro=f'''## 130—136：已改预测算法，交付精简RAW入口

2026-10-24。[优化结果与预测视频](<{report.as_posix()}>)；[运行说明](<{(ROOT/'research/stage135/README.md').as_posix()}>)。当前新增入口是`D:/drone forecast/event_drone_forecast/run_optimized_forecast.py`，默认54/API93保持原样。

- 实际算法优化：取消首次几何位置永久锚点，改为来源内当前几何中心短窗速度观测；机体维持绝对位置。复用23稳健加速度，按估计误差收缩，只用于未来外推。合格结构学成前保持机体匀速；模型记忆不充当当前观测。129边界完整性与五帧建模门槛不变。
- 清晰同11对未来框：0.2秒18.9482→11.3116px；0.4秒33.6792→30.8343px（8.45%）。同条件机体加速度对照37.3575px，新几何速度30.8343px。0.1秒原12对中启动1对未发布，对称比较共同11对15.8883→8.5781px；另记1次未发布，不能混用135潜在系数12对16.0727→9.3717px。全部为已暴露开发录像，稀疏人工框中心，不是COM/转速或独立泛化成绩。
- 135在原1505观测时刻重放；实际RAW新入口两次完整400帧、独立前缀60帧、空场30帧，890成功时刻。当前400帧与冻结135系数差0，实际输出重新计分；视频400帧解码，事件cutoff/无缓存标注推理/身份重置发布通过。
- 工程优化：计数/占据共享一遍当前ON/OFF计数，只运行优化/同条件机体/匀速三路，去掉旧研究入口几十路历史控制。结构学成后跳过随后会被丢弃的自由结构拟合。旧研究入口约883.5ms；精简两次212.2/235.6ms/步，主要来自控制路径裁剪，不是受控硬件基准，也未证明最后一项单独提速。当前几何约169.9ms、GPU约33.9ms，另有序列化预览13.2ms；远未20ms实时。
- 负结果130—134只留小源码/JSON：仅偏移/相对速度不足；六维加速度反馈退步；普遍启用加速度伤害断续片段；延迟误差守门和一致性变体未胜出。135最终依赖合格结构支持。未再增加几十路部署控制。
- 实际空场首轮只26/30，因热身错误跳过前4个发布时刻；保留negative_startup_count_failure及日志。修复成仅抑制几何、继续输出所有请求时刻后，空场30完整且无有效预测。v1/v2入口源快照保留，仅供指纹历史，不在research目录直接运行。
- 侧视continuous6仍61.095px；扩展1/2仍84.190/71.099px，均匀速回退、未改善。本轮这些片段只有固定观测重放，未重新全链路RAW。转速→推力/真实法线动力学尚未完成，不能据本轮称创新或物理预测成功。

下一步直接改运行代码：先做机体ROI内共享支持/几何计算，保留边界外延和完整轮廓，减少全图重复处理；固定一个短段比较实际预测和耗时。随后把侧视线法线与真实时间周期接到预测，不能用事件总数冒充转速/推力。禁止继续扩张历史控制树。E盘只读、yolov10-main未访问/修改、无G/新依赖。本轮任务已收尾，无本轮后台作业。

'''
    worklog=WORK/'WORKLOG.md';text=worklog.read_text('utf-8');assert '## 137. 130—136' not in text
    worklog.write_text(text.rstrip()+'\n\n## 137. 130—136：预测与运行入口优化\n\n'+intro,'utf-8')
    for path in [WORK/'PROJECT_STATUS_20261022.md',ROOT/'README.md']:
        path.write_text(intro+'以下129及更早为历史。\n\n'+path.read_text('utf-8'),'utf-8')
    handoff.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),last_completed_stage=136,algorithm_stage=135,latest_full_forecast_stage=136,experimental_RAW_stage=136,
        latest_RAW_geometry_stage=136,latest_diagnostic_stage=136,latest_full_forecast_report=str(report),latest_report=str(report),latest_checks=str(OUT/'final_checks136.json'),
        experimental_RAW_entry=str(ROOT/'run_optimized_forecast.py'),recommended_forecast_research_stage=135,latest_completed_stages=list(range(130,137)),execution_status='checkpoint_ready',research_objective_status='incomplete',
        next_action='Implement shared ROI support/geometry computation in the lean RAW runner; keep full-boundary guards and current measurement qualification. One fixed short clip for forecast/runtime comparison, no expansion of old control tree. Then side-line normals + temporal periodicity; density alone is not thrust/RPM.',
        completion_scope='Optimization135: relative geometry velocity + robust acceleration rollout after learned guarded structure; unmodeled CV. Actual RAW lean v1/v3 400+400, prefix60, negative30 complete; actual0.4s same11 futureBOX33.679->30.834px. Current235.6ms, not realtime. Other exposed clips replay-only and unchanged; side/RPM/physics/generalization incomplete.',
        source_summary_files=[str(worklog),str(WORK/'PROJECT_STATUS_20261022.md'),str(ROOT/'README.md'),str(ROOT/'research/stage135/README.md')],
        running_jobs=[],prior_archive=str(previous),prior_archive_sha256=previous_sha,report_local_links_checked=checked,
        optimization_actual_raw_frames=890,optimization_current_full_frames=400,optimization_score=dict(horizon_s=.4,pairs=11,before_px=33.679231576694406,after_px=30.834253764868958,metric='manual future bbox-center error; exposed development clip'),
        side_improved=False,realtime_proven=False,default_promoted=False,archive_scope='Verified129 base plus130..136 source/protocols/results/plots/report and new entry/handoff. RAW/weights/JSONL/videos stay in original paths. Embedded handoff omits self archive SHA.')
    handoff['completed']=sorted(set(handoff.get('completed',[]))|set(range(130,137)))
    handoff['preserved_negative_results']+=['131 acceleration-state feedback regresses;132 ungated robust acceleration harms intermittent detections;133/134 not adopted','135 negative RAW warmup count failure26/30 preserved, fixed30/30; current136 runtime235.6ms not faster than212.2 v1, no separate fit-skip speed claim']
    handoff.pop('archive',None);handoff.pop('archive_sha256',None);handoff['zip_verified']=False;save(handoffpath,handoff)
    save(OUT/'final_checks136.json',dict(report_local_links_checked=checked,actual_RAW890_complete=True,current_full_RAW400=True,predictor_difference=0,actual_future_scores=str(OUT/'actual_RAW_scores136.json'),video_preview_visually_checked=True,all_video_frames_decoded=True,source_fingerprints_current=True,raw_size_mtime_unchanged=True,RAW_bytewise_checksum_not_performed=True,side_improved=False,realtime=False,default_promoted=False,running_jobs=[]))
    current={}
    for p in [worklog,WORK/'PROJECT_STATUS_20261022.md',ROOT/'README.md',handoffpath,ROOT/'run_optimized_forecast.py']:
        current[p.relative_to(WORK).as_posix()]=p.read_bytes()
    for stage in range(130,137):
        for p in (ROOT/f'research/stage{stage}').rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:current[p.relative_to(WORK).as_posix()]=p.read_bytes()
        for folder in (WORK/'output').glob(f'stage{stage}_*'):
            for p in folder.rglob('*'):
                if p.is_file() and p.suffix.lower() in ['.json','.html','.png','.log','.md']:current[p.relative_to(WORK).as_posix()]=p.read_bytes()
    now=datetime.datetime.now(datetime.timezone.utc);archive=WORK/'output/checkpoints'/('research_through_stage136_20261024_'+now.strftime('%H%M%S_%f')+'.zip');hashes={};manifestname='output/stage135_optimized_forecast_20261024/recovery_manifest136.json'
    with zipfile.ZipFile(previous) as old,zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as new:
        assert old.testzip() is None
        for name in old.namelist():
            if name in current or name==manifestname:continue
            data=old.read(name);new.writestr(name,data);hashes[name]=hashlib.sha256(data).hexdigest()
        for name,data in current.items():new.writestr(name,data);hashes[name]=hashlib.sha256(data).hexdigest()
        new.writestr(manifestname,json.dumps(dict(base_archive=str(previous),base_sha256=previous_sha,files=hashes,count=len(hashes),scope=handoff['archive_scope']),ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,digest in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
    digest=sha(archive);handoff.update(archive=str(archive),archive_sha256=digest,zip_verified=True);save(handoffpath,handoff)
    save(OUT/'archive_checks136.json',dict(path=str(archive),sha256=digest,entries=len(hashes)+1,CRC_and_all_entry_SHA_verified=True,embedded_handoff_omits_own_archive_SHA=True))
    print(json.dumps(dict(report=str(report),archive=str(archive),sha256=digest,files=len(hashes)+1,links=checked)),flush=True)
if __name__=='__main__':main()
