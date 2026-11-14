"""Finish frozen natural extension and verified handoff, preserving negative data."""
from pathlib import Path
import datetime,hashlib,json,sys,zipfile
from html.parser import HTMLParser
from urllib.parse import unquote,urlparse
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage123')]
from study123 import OUT,verify,save,sha
def load(p):return json.loads(Path(p).read_text('utf-8-sig'))
def main():
    verify();c=load(OUT/'delivery_checks123.json');assert c['actual_RAW_parent']==1190 and c['source_preview_decoded']==1030 and c['overlay_decoded']==1000
    assert load(OUT/'extraction_trace2/trace123.json')['all_rotors_exact122'];positive=load(OUT/'raw/prefix_natural6/checks123.json');assert positive['initialized_branch_union']==69
    report=Path(c['report'])
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.urls=[]
        def handle_starttag(self,t,attrs):self.urls.extend(v for k,v in attrs if k in ['href','src'] and v)
    parser=Links();parser.feed(report.read_text('utf-8'));local=0
    for url in parser.urls:
        if urlparse(url).scheme or url.startswith('#'):continue
        assert (OUT/unquote(url.split('#')[0])).resolve().is_file(),url;local+=1
    lead='''## 123：冻结122的自然退化扩展与真实未来框评分

2026-10-23。[新报告、三个RAW预测视频和提取漏斗](<D:/drone forecast/output/stage123_frozen_extension_20261023/第一百二十三阶段自然退化扩展与未来框预测报告.html>)；[123复现与研究边界](<D:/drone forecast/event_drone_forecast/research/stage123/README.md>)。算法仍122，入口`run_validated_structure_forecast.py`未改；默认54/API93不改。

- 原122所有参数/源码固定，不人为删桨，实际RAW：1号40—46秒300、2号40—46秒300、6号22—30秒400、1号40—42秒独立前缀100、负例30，补正例22—23.2秒前缀60，总1190父时刻；另外同2号300时刻只做提取诊断，不生成新预测。源1030帧和叠加1000帧全解码，源码/事件界/负例通过。
- 历史清点明确1号30—35秒之外仍有未知曝光、2/3号0—60秒、4/6号0—120秒早期已扫描；称冻结扩展，非盲测/未见主体/训练独立。区间按时间元数据固定，不按未来误差选取。
- 两扩展首次结构学习0/2。1号300时刻NN43、四候选25、几何测量合格仅1、不够5次确认；2号NN183、25次双候选，其余0，无四候选。2号8个稀疏present时刻NN8/8、目标中位188.85×78.69px、中位IoU约.8415，定位到桨盘提取/视角前提。
- 6号22.8秒学到结构；400时刻候选四279/三40/二78/一1/零2。自检当前几何236，其中自然三桨6，自检更新224；三桨时刻仿射6、bodyhold34。没有连续>=5时刻的三当前观测有效跟踪后恢复四桨，三正例完整自然恢复0。选择器类别文字不代表几何实际有效。
- 主预测/截断/负例完毕后读既有人工事件框；同组发布起点与目标，名义.1/.2/.4按实际target−origin、最多20ms差进行CV外推，无GT插值。0.4秒：1号84.190234px仅1/2端点，2号71.099123px为6/8，均结构未激活；6号bodyKF40.799416、原结构hold34.396218、affine3/自检33.679232、conic3为34.568705px，共11/11，实际h .4038—.4190s。自检对hold约.717px，与始终仿射相同；不称独立选择收益。人工框中心非COM，先前2.26px一致性与此FDE不能直接比较。
- 1号10、2号4个稀疏absent时刻均无有效预测；不等于整段无误检。present NN1/2、8/8、12/12；当前几何0/2、0/8、7/12。所有未发布/未建模进入覆盖分母，未以零误差隐藏。
- 2号提取链路300时刻逐项精确原122。43.5秒只有12.17×6.63和11.39×4.41px两个候选，分组0；45秒三个小候选只选2。43.5同NN+50px区域双极占据3215、阈值165、opening157、closing161；45秒2259/357/352/355。占据和计数阈值定义不同、含机体/背景，不报物理桨盘损失率。证据支持计数门槛只留下小内核，后续候选/分组不足；原生黑不等于重复计数高。未按两帧降阈值宣称改善。
- 1号实际100前缀全部状态严格，但0检测/预测；补60正例活动前缀69路初始化、19当前几何、17自检，所有已发布预测/内部状态/语义事件SHA严格匹配。补充在主评分后仅作因果回归，不新增评分/调参目标。6号前250时刻原64路预测/系数和NN精确118。
- 保留执行失败：三并行Torch导入Win1455分页/提交资源限制，三预测文件均0字节；归档failed_parallel_startup后串行成功，无OS设置变化。补前缀直接旧entry.run留下进程级audit-hook，生成60后读文件NameError；保留v1辅助脚本，改子进程CLI，在新进程验证已有60帧，无重跑/禁用审计/修改冻结入口。后续应使用CLI或现有进程安全API，不在需继续I/O父进程直接调旧run。

下一步优先改善首次建模的事件支持覆盖：复用109/115审计和弱边界、57NN ROI，先查相关现成方法，再以纯ON/OFF占据支持对照当前计数阈值。保留本轮失败区间/负例作回归，评估轮廓完整性、误检和首次结构确认率，另留独立连续片段。二值占据/阈值本身不称创新，122的模型适配性自检还未证明自然退化泛化优势。E盘只读、yolov10-main未访问/修改，无G或新依赖；本轮进程均已结束。

以下122及更早为历史。

'''
    status=ROOT.parent/'PROJECT_STATUS_20261022.md';text=status.read_text('utf-8')
    if not text.startswith('## 123：'):status.write_text(lead+text,encoding='utf-8')
    readme=ROOT/'README.md';text=readme.read_text('utf-8');intro='''2026-10-23最新实验123：[自然退化扩展/人工未来框误差/RAW视频](<D:/drone forecast/output/stage123_frozen_extension_20261023/第一百二十三阶段自然退化扩展与未来框预测报告.html>)、[123说明](research/stage123/README.md)。冻结算法122，实际RAW1190时刻；两个扩展均未建模，正例自然三桨仅6次，完整自然恢复0。正例.4秒bodyKF40.80/原结构hold34.40/自检33.68px，11端点且自检等于固定仿射。核心瓶颈为首次可靠提取/建模，未声称创新泛化或实时。以下122及更早为历史。

'''
    if not text.startswith('2026-10-23最新实验123'):readme.write_text(intro+text,encoding='utf-8')
    log=ROOT.parent/'WORKLOG.md';text=log.read_text('utf-8')
    if '## 135. 123：' not in text:log.write_text(text+'\n\n## 135. 123：冻结自然退化扩展、真实未来框评分与支持碎化诊断\n\n'+lead.split('\n\n',1)[1].split('以下122及更早为历史。')[0],encoding='utf-8')
    cp=ROOT.parent/'output/checkpoints';h=load(cp/'latest_handoff.json');prior=Path(h['archive']);assert sha(prior)==h['archive_sha256'];archive=cp/f'research_through_stage123_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.zip'
    h.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),last_completed_stage=123,completed=sorted(set(h['completed']+[123])),latest_full_forecast_stage=123,
        latest_diagnostic_stage=123,latest_RAW_geometry_stage=123,algorithm_stage=122,latest_report=str(report),latest_full_forecast_report=str(report),latest_checks=str(OUT/'delivery_checks123.json'),
        experimental_RAW_stage=122,experimental_RAW_entry=str(ROOT/'core/run_validated_structure_forecast.py'),in_progress=[],experiments_running=[],default_promoted=False,
        completed_scope='123 frozen natural RAW1190, two extension recordings/no model, natural positive6 partial updates/no100ms recovery, sparse manual BOX scoring, exact300 geometry extraction trace.',
        completion_scope='Frozen expansion exposed initialization/support bottleneck; no blind efficacy/COM/extreme-side/realtime or independent innovation gain proven.',
        next_action='Review reusable occupancy/bipolar support and NNROI approaches before changing extraction. Fixed occupancy-vs-rate controls on123 failed clips and negative, score first-lock/contour completeness/false detections. Retain123 as development regression and reserve other continuous data. Do not lower gates or claim pixel-loss ratio from different support definitions. Use CLI/process-safe API, not old in-process entry.run.',
        archive=str(archive),archive_sha256=None,prior_archive=str(prior),prior_archive_sha256=h['archive_sha256'],zip_verified=False,execution_status='checkpoint_pending',
        source_summary_files=[str(status),str(log),str(readme),str(ROOT/'research/stage123/README.md')],report_local_links_checked=local,
        archive_scope='Verified122 snapshot plus123 source/protocols/small results/native figures and two count NPZs/current handoff. RAW/weights/JSONL/videos remain at original paths. Embedded handoff omits own archive SHA.')
    h['counts_by_stage']['123']=dict(RAW_forecast_parent=1190,additional_RAW_geometry_trace=300,original_preview_frames=1030,overlay_frames=1000,extension_recordings=2,first_models_extensions=0,natural_valid_partial_updates=6,complete_natural_transitions=0)
    h['retained_failed_attempts']=list(dict.fromkeys(h['retained_failed_attempts']+['123 two frozen extension recordings:0 initial structural models; natural positive only6 partial updates/0 sustained100ms recovery',
        '123 three parallel GPU startup Win1455 shm.dll import, zero forecasts; all files retained, serial retries succeeded','123 supplemental direct entry.run completed RAW60 then old audit-hook NameError; preserved helperv1, subprocess CLI repair, fresh checker passes without rerun',
        '123 sparse futureBOX: selector equals fixed affine; no independent selection gain, realCOM/attitude/RPM still absent']))
    save(cp/'latest_handoff.json',h)
    files=[status,log,readme,cp/'latest_handoff.json']
    for folder in [ROOT/'research/stage123',OUT]:files.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.jsonl','.mp4'] and p.name not in ['recovery_manifest123.json','final_checks123.json'])
    contents={}
    with zipfile.ZipFile(prior) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if not name.endswith('/'):contents[name]=z.read(name)
    contents.update({p.relative_to(ROOT.parent).as_posix():p.read_bytes() for p in files});manifest={n:hashlib.sha256(d).hexdigest() for n,d in contents.items()};save(OUT/'recovery_manifest123.json',dict(archive=str(archive),files=manifest));contents[(OUT/'recovery_manifest123.json').relative_to(ROOT.parent).as_posix()]=(OUT/'recovery_manifest123.json').read_bytes()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in contents.items():z.writestr(name,data)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for n,digest in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==digest,n
    digest=sha(archive);h.update(archive_sha256=digest,zip_verified=True,execution_status='checkpoint_ready');save(cp/'latest_handoff.json',h)
    save(OUT/'final_checks123.json',dict(archive=str(archive),archive_sha256=digest,verified_files=len(contents),local_links_checked=local,RAW_forecast_parent=1190,RAW_geometry_trace300_exact=True,
        source_preview_frames=1030,overlay_frames=1000,original64_overlap250_exact=True,positive_prefix60_initialized69_exact=True,
        no_parameter_changes=True,no_default_promotion=True,complete_natural_transitions=c['complete_natural_transitions'],real_COM_or_realtime_proven=False))
    print(json.dumps(dict(report=str(report),archive=str(archive),verified_files=len(contents))),flush=True)
if __name__=='__main__':main()
