"""Update worklog and verified handoff after the actual138..142 work."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlparse
import datetime,hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT.parent;OUT=WORK/'output/stage140_visible_fallback_20261024'
def load(p):return json.loads(Path(p).read_text('utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),'utf-8')
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src','poster'] and v)
def main():
    summary=load(OUT/'optimization_summary142.json');report=Path(summary['report']);links=0
    for page in [report,WORK/'output/latest_optimized_forecast.html']:
        parser=Links();parser.feed(page.read_text('utf-8'))
        for link in parser.links:
            u=urlparse(link)
            if u.scheme or link.startswith('#'):continue
            assert (page.parent/unquote(u.path)).resolve().is_file(),link;links+=1
    # Historical NN/RAW entry snapshots restore the exact main-source hash.
    runs=[]
    for stage,folder,names,snapshot in [(138,WORK/'output/stage138_shared_objective_20261024',['fast100','natural6','side50','negative5'],ROOT/'research/stage138/run_optimized_forecast_v5.py'),
        (140,OUT,['continuous6','extension2','prefix_natural6','negative5_raw'],ROOT/'research/stage140/run_optimized_forecast_v6.py'),
        (141,WORK/'output/stage141_periodic_axis_20261024',['continuous6'],ROOT/'run_optimized_forecast.py')]:
        for name in names:
            p=load(folder/name/'protocol.json');c=load(folder/name/'completion.json');assert c['sources_unchanged'] and not c['labels_read'] and not c['output_caches_read']
            for rel,digest in p['sources'].items():assert sha(snapshot if rel=='run_optimized_forecast.py' else ROOT/rel)==digest,(stage,name,rel)
            stat=Path(p['raw']).stat();assert (stat.st_size,stat.st_mtime_ns)==(p['raw_size'],p['raw_mtime_ns']);runs.append(dict(stage=stage,case=name,frames=c['frames'],entry_source_snapshot=str(snapshot)))
    assert sum(r['frames'] for r in runs)==1920
    assert load(OUT/'checks140.json')['actual_RAW865_complete'];assert load(WORK/'output/stage141_periodic_axis_20261024/checks141.json')['forecast_exact140']
    hpath=WORK/'output/checkpoints/latest_handoff.json';handoff=load(hpath);assert handoff['last_completed_stage']==137;previous=Path(handoff['archive']);previous_sha=handoff['archive_sha256'];assert sha(previous)==previous_sha
    intro=f'''## 138—142：继续加速，侧视运动预测已改善

2026-10-24。[最新统一结果与视频](<{(WORK/'output/latest_optimized_forecast.html').as_posix()}>)；[本次固定报告](<{report.as_posix()}>)；[运行/交接](<{(ROOT/'research/stage140/README.md').as_posix()}>)。推荐入口仍`D:/drone forecast/event_drone_forecast/run_optimized_forecast.py`，默认shared138几何＋visible140运动；141周期轴可用`--periodic-axis`开启，默认关闭且不反馈预测。142动力学试验未采用。旧默认54/API93保持。

- 138已改代码：全图阈值后围住全部非零支持加4px余量，形态学/轮廓/父标签图局部处理、offset恢复全局，范围≥80%回全图，开运算复用。固定结构残差/Jacobian共享准备和同迭代中间量，CPU计数与同帧GPU检测等待重叠。20边缘/空/满/稀疏掩膜逐位及轮廓点序相同，24实际拟合导数/独立差分通过，实际400步前端/资格相同。
- 同入口共同100步188.557→145.860ms（22.64%），几何121.247→102.999、结构59.935→45.631ms；GPU32.769→28.672也变，系统负载非受控，不把总差全部归于代码。完整138清晰400步138.176ms，另序列化/视频12.670ms。仍非20ms实时。
- 139负结果：密集观测允许稳健加速度，侧视改善，但扩展2的0.4秒71.099→103.876px，未采用。140仅增加整个0.35秒拟合窗口的机体框边界守门，避免触边外观框直接资格化额外加速度。它是框边界代理，不保证整机/全部桨叶真实可见。
- 140真实RAW：连续侧视475＋扩展2300＋清晰活动前缀60＋空场30＝865。系数精确冻结缓存（清晰前缀仅浮点微差）；空场无有效预测；事件cutoff、无标签/缓存输入、视频全解码通过。完整清晰400使用138实际观测作140预测重放，系数保持，不称新增140完整清晰NN/RAW。活动回退80步预测前缀重放通过，不称额外RAW前缀。
- 同稀疏未来框端点：侧视0.4秒61.095→48.776px（10/11可选对，20.16%）；扩展2为71.099→63.626（6/8，10.51%）；清晰30.834保持（11/11）。0.2秒侧视26.854→24.396、扩展25.272→20.245；0.1秒侧视12.794→12.911、扩展11.651→12.370，短期略退步。清晰0.1沿用共同发布11/12为8.578，不混用潜在12对9.372。扩展1仅重放保持。全部为已暴露开发录像和人工未来框中心，不是COM/RPM/姿态或独立泛化成绩。
- 141新增可选周期线：直接复用79已装SDK严格7次/200us、当前20ms唯一像素、60—1200Hz、16像素/0.35模态/环带拒绝，以及原_line宽长比.14。当前CV速度对齐像素。真实RAW侧视475：332周期ROI资格、235无向线候选，预测逐项精确140。候选无独立姿态真值，不能称235次正确姿态或机械RPM，更不能当总推力。条纹77/80、移动条纹20/80；共同闪烁/Poisson/有色随机/突发各四种子无合格ROI，切频76/80周期合格但无直线；有限像素控制非实飞误报率。
- 142负结果：复用旧在线RLS像素模型，旧four_visible接口显式改为feature_supported，新141线/相对闪烁率平方作输入，不假造四桨。475实际观测重放，204支持/151更新/155启用，0.4秒48.776→50.891px，未采用、未新增RAW、未继续调参筛选胜者。频率/法线/深度/检测偏差等因素未隔离，不可用参数表报质量/风/推力系数。
- 本轮成功RAW1920时刻；全部视频全解码。137v4、138v5、140v6主入口源快照保留；新141完整RAW用当前主源码。历史缓存140协议来自139适配器只含139源SHA，实际RAW前protocol140_RAW.json补全明确140源；不冒称独立验证。保护目录未访问/修改，无G/新依赖，E盘RAW只读（核查大小/mtime，未做全盘字节哈希）。

下一步：先处理周期像素的多电机/谐波歧义与无向线可信度，用已有公开周期真值核查，142保持未采用；再让物理特征参与预测。计算继续减少全图支持和NN成本。先改代码、小固定段决定保留，不扩张历史控制树，也不要重复实现79已有SDK适配。所有本轮已启动作业已退出，最新入口可直接运行。

'''
    worklog=WORK/'WORKLOG.md';text=worklog.read_text('utf-8');assert '## 139. 138—142' not in text;worklog.write_text(text.rstrip()+'\n\n## 139. 138—142：加速与退化场景预测连续优化\n\n'+intro,'utf-8')
    for p in [WORK/'PROJECT_STATUS_20261022.md',ROOT/'README.md']:p.write_text(intro+'以下137及更早为历史。\n\n'+p.read_text('utf-8'),'utf-8')
    handoff.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),last_completed_stage=142,algorithm_stage=140,geometry_execution_stage=138,optional_periodic_axis_stage=141,
        latest_full_forecast_stage=141,experimental_RAW_stage=141,latest_RAW_geometry_stage=141,latest_RAW_pilot_stage=141,latest_diagnostic_stage=142,
        latest_report=str(report),latest_full_forecast_report=str(report),latest_checks=str(OUT/'final_checks142.json'),experimental_RAW_entry=str(ROOT/'run_optimized_forecast.py'),experimental_RAW_pilot_entry=str(ROOT/'run_optimized_forecast.py'),recommended_forecast_research_stage=140,latest_completed_stages=list(range(138,143)),
        next_action='Resolve multimotor/harmonic frequency ambiguity and validate unsigned periodic-line direction against available public module truth before trusting a force feature. 142 RLS feature model regressed48.776->50.891 and remains unadopted. Keep138 compute +140 dense/unclipped-body fallback, optional141 axis observer separate. Reduce global support/NN cost with one small fixed clip per actual change, no historical-control expansion.',
        completion_scope='138 actual RAW580 retains geometry/135 predictions and common100188.557->145.860ms;140 actual RAW865 dense-history+body-boundary fallback yields side0.4s61.095->48.776 (10/11),extension271.099->63.626(6/8),clear30.834 replay unchanged. Short0.1 slight regressions retained.141 actual RAW475 periodic observer332 qualified/235 unsigned lines, predictions exact140; not direction/RPM truth.142 replay force-feature pilot worse50.891, unadopted. Total RAW1920. Still not realtime or independent innovation/generalization proof.',
        source_summary_files=[str(worklog),str(WORK/'PROJECT_STATUS_20261022.md'),str(ROOT/'README.md'),str(ROOT/'research/stage140/README.md'),str(ROOT/'research/stage141/README.md'),str(ROOT/'research/stage142/README.md')],
        optimization_this_stage_actual_RAW_frames=1920,optimization_current_full_frames=475,optimization_current_full_case='continuous6',optimization_score=dict(metric='Previously exposed sparse manual future bbox-center endpoints, same common origins',side=dict(horizon_s=.4,paired=10,eligible=11,before_px=61.09496995848966,after_px=48.77615402846245),extension2=dict(horizon_s=.4,paired=6,eligible=8,before_px=71.09912348345942,after_px=63.625839226024134),clear=dict(horizon_s=.4,paired=11,eligible=11,before_px=30.834253764868958,after_px=30.834253764868926)),
        prediction_logic_unchanged_from_stage135=False,side_improved=True,side_improvement_type='Confidence-conditioned historical image acceleration, not measured thrust',side_physical_prediction_proven=False,realtime_proven=False,default_promoted=False,report_local_links_checked=links,
        running_jobs=[],execution_status='checkpoint_ready',research_objective_status='incomplete',prior_archive=str(previous),prior_archive_sha256=previous_sha,
        archive_scope='Verified137 base plus138..142 sources/protocols/results/plots/report and v4/v5/v6 historical main snapshots; current main141. RAW/weights/JSONL/video remain at original paths. Embedded handoff omits own archive SHA.')
    handoff['completed']=sorted(set(handoff.get('completed',[]))|set(range(138,143)));handoff['preserved_negative_results']+=['139 dense acceleration alone regresses extension2 to103.876; replaced by140 body-boundary guard','140 short0.1s regressions retained:side12.794->12.911,extension211.651->12.370','142 periodic-axis/rate feature RLS regresses side0.4s48.776->50.891, no adoption or extra RAW;141 candidate lines unvalidated as physical attitude']
    handoff.pop('archive',None);handoff.pop('archive_sha256',None);handoff['zip_verified']=False;save(hpath,handoff)
    save(OUT/'final_checks142.json',dict(report_local_links_checked=links,actual_RAW_runs=runs,total_successful_RAW_frames=1920,source_snapshots_match_all_run_protocols=True,raw_size_mtime_unchanged=True,RAW_bytewise_checksum_not_performed=True,plots_and_periodic_preview_visually_checked=True,all_new_videos_completely_decoded=True,periodic_observer_does_not_change_forecast=True,failed142_not_adopted=True,independent_test=False,realtime=False,running_jobs=[]))
    current={}
    for p in [worklog,WORK/'PROJECT_STATUS_20261022.md',ROOT/'README.md',hpath,ROOT/'run_optimized_forecast.py',ROOT/'research/stage137/run_optimized_forecast_v4.py',WORK/'output/latest_optimized_forecast.html']:current[p.relative_to(WORK).as_posix()]=p.read_bytes()
    for stage in range(138,143):
        for folder in [ROOT/f'research/stage{stage}',*list((WORK/'output').glob(f'stage{stage}_*'))]:
            for p in folder.rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts and p.suffix.lower() in ['.py','.json','.html','.png','.log','.md']:current[p.relative_to(WORK).as_posix()]=p.read_bytes()
    now=datetime.datetime.now(datetime.timezone.utc);archive=WORK/'output/checkpoints'/('research_through_stage142_20261024_'+now.strftime('%H%M%S_%f')+'.zip');hashes={};manifestname='output/stage140_visible_fallback_20261024/recovery_manifest142.json'
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
    digest=sha(archive);handoff.update(archive=str(archive),archive_sha256=digest,zip_verified=True);save(hpath,handoff);save(OUT/'archive_checks142.json',dict(path=str(archive),sha256=digest,entries=len(hashes)+1,CRC_and_all_entry_SHA_verified=True,embedded_handoff_omits_own_archive_SHA=True))
    print(json.dumps(dict(report=str(report),archive=str(archive),sha256=digest,files=len(hashes)+1)),flush=True)
if __name__=='__main__':main()
