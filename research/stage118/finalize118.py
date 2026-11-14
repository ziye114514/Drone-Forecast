"""Finalize112-118, retain failed hypotheses, verify archive and handoff."""
from pathlib import Path
import datetime,hashlib,json,sys,zipfile
from html.parser import HTMLParser
from urllib.parse import unquote,urlparse
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage118'),str(ROOT/'research/stage109')]
from study118 import OUT,verify,save,sha
def load(p):return json.loads(Path(p).read_text('utf-8-sig'))

def main():
    verify();checks=load(OUT/'delivery_checks118.json');assert checks['actual_RAW_parent']==550 and checks['new_preview_decoded']==430
    report=Path(checks['report']);page=report.read_text('utf-8')
    # Explicitly keep geometric assumptions visible in the report.
    note='<p>结构仍采用四桨等半径、共面、近似矩形机架假设；未知尺寸由图像中学习，不等于支持任意非对称机架。</p>'
    if note not in page:page=page.replace('<h2>预测效果和实际RAW验证</h2>',note+'<h2>预测效果和实际RAW验证</h2>');report.write_text(page,encoding='utf-8')
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.urls=[]
        def handle_starttag(self,tag,attrs):self.urls.extend(v for k,v in attrs if k in ['src','href'] and v)
    parser=Links();parser.feed(page);local=0
    for url in parser.urls:
        if urlparse(url).scheme or url.startswith('#'):continue
        assert (OUT/unquote(url.split('#')[0])).resolve().is_file(),url;local+=1
    base=ROOT.parent/'output'
    for stage,prefix in [(113,'stage113_joint_observed_arcs_20261022'),(115,'stage115_RAW_weak_support_20261022'),(117,'stage117_persistent_structure_20261022')]:
        for case in ['clear6','continuous6']:
            f=base/prefix/(case+'_prefix')/f'prefix{stage}.json';assert f.is_file(),f
    lead='''## 112—118：真实轮廓对照、固定结构与完整RAW118完成

[新报告、结构曲线与实际RAW预测视频](</D:/drone forecast/output/stage118_RAW_learned_structure_20261022/第一百一十二至一百一十八阶段观测与固定结构预测报告.html>)。最新完整实验118，入口`event_drone_forecast/run_learned_structure_forecast.py`；原107全部64路精确保留，新增3路，默认54/API93不改。

- 112真实RAW轮廓记录四段1155＋两个实际截断各60＝1275个20ms时刻，geometry严格110。20个支持未匹配均为原arc推断候选，未当完整当前轮廓；112不产生预测。
- 113原58目标AST适配：原数值导数路径精确，解析Sampson导数中央差分通过。比较合成完整周长、实际阈值轮廓、曲率支持弧段/soft_l1及全参数局部协方差门槛。1155时刻/两60前缀严格、bodyKF同输入一致。清晰固定偏移0.4s原16.842569、实际轮廓21.316432、弧段21.642390，连续0初始化。实际轮廓不等于物理桨盘边缘，负结果未接入RAW默认。
- 114八个固定原生计数快照：已有种子在半阈值双极计数中OpenCV分水岭恢复弱边界；C80预算内合格。115完整RAW几何＋同期缓存机体/轴再次1275时刻，原110基线/前缀精确；清晰几何更新仅14、0.4s22.862705，连续0初始化。单C帧改善不代表整段效果，保留失败、不继续按端点调阈值。
- 116独立交错16对/4个固定非四椭圆输入跳过，FD中位28.288ms、解析含局部协方差25.694ms、配对比1.093727；中心差最大0.00002498px，终止状态无差异。不是端到端速度或逐位数学相等主张。
- 117原104在锁定后仍每帧自由估半径/长宽比，已学值只作事后筛选。本轮在原五帧资格后实际学习并固定无量纲结构，只求H8当前投影，结构和新鲜姿态分开；身份无当前机体支持超过.5s重置，旧五帧重新确认/姿态过期/位置滤波不改。各段独立，不将22s模型跨80s注入110s。局部位姿协方差以固定已学结构为条件，没有包含结构学习误差。合成静止/平移/变向/缺桨恢复各100带噪投影轮廓，缺桨仍约10.23px相对位置RMSE、0.4s19.36px。
- 同7/8个既有稀疏框端点，117清晰自由16.842569→固定16.780019px，改善很小；原完整融合13.708083更好。侧视各固定81.722934、原融合66.925684，0首次建模，不能声称侧视模型收益。RAW真实重心/姿态/RPM真值仍无。
- 118独立RAW几何进程防止额外椭圆运算影响旧OpenCV随机状态。执行输入仅RAW和旧NN权重，worker禁止读取output缓存/标注，按当前cutoff推进。真实RAW清晰250、连续150、负例30、两个真正截断各60＝550父时刻；原64系数/预测精确107，新3/模型/滤波状态精确117，bodyKF/时间界/实际前缀/负例通过。430原视频和430新增叠加帧全解码。运行仍非实时；并发记录不当跨版本速度保证。
- 查阅Szpak2015椭圆协方差作者仓库、Masuzaki/Sugaya2016支持弧段论文、SciPy/OpenCV；未运行作者MATLAB系统，实际复用现有58/104/107/几何辅助与本机库。无新依赖/G，不访问yolov10-main，E盘只读。基础拟合/记忆机制不称已验证原创方法。

下一步优先同一连续录像的“已学结构→部分/侧视→恢复”，结合结构关联剩余真实支持、估计结构不确定性和处理机体/桨盘事件混合；保持缺失就是缺失，不用隐藏点生成频率/姿态真值。侧视首次锁定、独立RPM收益、实时调度及论文创新仍未完成。本轮所有实验结束，无后台。

以下111及更早为历史。

'''
    status=ROOT.parent/'PROJECT_STATUS_20261022.md';text=status.read_text('utf-8')
    if not text.startswith('## 112—118：'):status.write_text(lead+text,encoding='utf-8')
    readme=ROOT/'README.md';text=readme.read_text('utf-8')
    intro='''2026-10-22最新完整RAW实验118：[112—118报告/预测视频](</D:/drone forecast/output/stage118_RAW_learned_structure_20261022/第一百一十二至一百一十八阶段观测与固定结构预测报告.html>)、[118入口与复现说明](research/stage118/README.md)。`run_learned_structure_forecast.py`保留原64，加3路已学结构预测；真实RAW550时刻/430视频帧全部核验。清晰局部16.843→16.780px很小，侧视仍0首次建模，未替换默认54/API93。以下117及更早为历史。

'''
    if not text.startswith('2026-10-22最新完整RAW实验118'):readme.write_text(intro+text,encoding='utf-8')
    log=ROOT.parent/'WORKLOG.md';text=log.read_text('utf-8')
    section='\n\n## 133. 112—118：原生轮廓负结果、结构固定与完整RAW交付\n\n'+lead.split('\n\n',2)[2].split('以下111及更早为历史。')[0]
    if '## 133. 112—118：' not in text:log.write_text(text+section,encoding='utf-8')
    env={}
    import cv2,numpy,scipy
    env.update(python=sys.version,opencv=cv2.__version__,numpy=numpy.__version__,scipy=scipy.__version__)
    save(OUT/'environment118.json',env)
    checkpoint=base/'checkpoints';h=load(checkpoint/'latest_handoff.json');prior=Path(h['archive']);assert sha(prior)==h['archive_sha256']
    archive=checkpoint/f'research_through_stage118_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.zip'
    h.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),last_completed_stage=118,completed=sorted(set(h['completed']+list(range(112,119)))),
        latest_full_forecast_stage=118,latest_cached_forecast_stage=117,latest_diagnostic_stage=116,latest_RAW_geometry_stage=118,
        latest_report=str(report),latest_full_forecast_report=str(report),experimental_RAW_entry=str(ROOT/'core/run_learned_structure_forecast.py'),experimental_RAW_stage=118,
        latest_checks=str(OUT/'delivery_checks118.json'),in_progress=[],experiments_running=[],default_promoted=False,
        completed_scope='112 native supports;113 robust support/uncertainty controls;114/115 weak-boundary failed pilot;116 analytic benchmark;117 learned fixed structure;118 actualRAW67 forecasts.',
        completion_scope='Faithful structural memory implemented and tested; clear local effect tiny, no first side model, no independent RPM benefit/realtime/original-method proof.',
        original64_numerically_preserved=True,stable_high_accuracy_realtime_achieved=False,independent_RPM_gain_proven=False,
        next_action='Use a single continuous recording that learns structure then partially/side degrades and recovers. Associate remaining observed supports with retained structure; model learned-structure uncertainty and body/propeller event mixture. Do not import22s state into110s or fabricate missing hubs/frequencies.',
        archive=str(archive),archive_sha256=None,prior_archive=str(prior),prior_archive_sha256=h['archive_sha256'],zip_verified=False,
        source_summary_files=[str(status),str(log),str(readme),str(ROOT/'research/stage118/README.md')],report_local_links_checked=local,
        archive_scope='Prior111 verified snapshot plus112-118 sources/small results/figures/native counts and current handoff. RAW/weights/fullJSONL/videos remain original paths; embedded handoff excludes self archive SHA.')
    h['counts_by_stage'].update({'112':dict(RAW_current20=1275,geometry_exact110=True),'113':dict(cached_current=1155,independent_prefixes=120),'114':dict(fixed_snapshots=8,new_RAW=False),
        '115':dict(RAW_geometry_current20=1275,cached_current_body=True),'116':dict(interleaved_pairs=16,skipped_nonfour=4),
        '117':dict(cached_current=1155,independent_prefixes=120,synthetic_frames=400),'118':dict(RAW_parent=550,original_video_frames=430,new_overlay_frames=430,old_branches=64,new_branches=3)})
    h['retained_failed_attempts']=list(dict.fromkeys(h['retained_failed_attempts']+['113 actual contour/arc fitting worsened clear forecast21.64px; no promotion','115 weak-boundary C success failed full-window forecast22.86px; no promotion','117 local shape covariance excludes learned-structure error; side no first model']))
    save(checkpoint/'latest_handoff.json',h)
    files=[status,log,readme,ROOT/'core/run_learned_structure_forecast.py',checkpoint/'latest_handoff.json']
    directories=[ROOT/'research'/f'stage{s}' for s in range(112,119)]
    directories += [base/name for name in ['stage112_observed_support_20261022','stage113_joint_observed_arcs_20261022','stage114_weak_boundary_pilot_20261022','stage115_RAW_weak_support_20261022','stage116_analytic_solver_benchmark_20261022','stage117_persistent_structure_20261022','stage118_RAW_learned_structure_20261022']]
    for folder in directories:
        files += [p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.jsonl','.mp4'] and p.name not in ['recovery_manifest118.json','final_checks118.json']]
    contents={}
    with zipfile.ZipFile(prior) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if not name.endswith('/'):contents[name]=z.read(name)
    contents.update({p.relative_to(ROOT.parent).as_posix():p.read_bytes() for p in files})
    manifest={name:hashlib.sha256(data).hexdigest() for name,data in contents.items()};save(OUT/'recovery_manifest118.json',dict(archive=str(archive),files=manifest))
    contents[(OUT/'recovery_manifest118.json').relative_to(ROOT.parent).as_posix()]=(OUT/'recovery_manifest118.json').read_bytes()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in contents.items():z.writestr(name,data)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
    h.update(archive_sha256=sha(archive),zip_verified=True,execution_status='checkpoint_ready');save(checkpoint/'latest_handoff.json',h)
    save(OUT/'final_checks118.json',dict(archive=str(archive),archive_sha256=sha(archive),verified_files=len(contents),local_links_checked=local,
        latest_full_RAW_stage=118,actual_RAW_parent=550,original64_exact107=True,new3_exact117=True,
        video_frames_decoded=860,original_preview_frames=430,new_overlay_frames=430,no_default_promotion=True,side_recovery_achieved=False,realtime_achieved=False))
    print(json.dumps(dict(report=str(report),archive=str(archive),verified_files=len(contents))),flush=True)

if __name__=='__main__':main()
