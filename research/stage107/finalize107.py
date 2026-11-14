"""Finalize106-108 without promoting a failed view-dependent variant."""
from study107 import *
from html.parser import HTMLParser
from urllib.parse import unquote,urlparse
import zipfile


def main():
    verify();checks=load(OUT/'delivery_checks107.json');assert checks['actual_RAW_parent']==815
    report=Path(checks['report'])
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.urls=[]
        def handle_starttag(self,tag,attrs):
            self.urls += [v for k,v in attrs if k in ['href','src'] and v]
    parser=Links();parser.feed(report.read_text('utf-8'));count=0
    for url in parser.urls:
        if urlparse(url).scheme or url.startswith('#'):continue
        assert (OUT/unquote(url.split('#')[0])).resolve().is_file(),url;count+=1
    status=ROOT.parent/'PROJECT_STATUS_20261022.md';old=status.read_text('utf-8')
    lead='''## 106—108：连续筛查与新中心偏移滤波已完成

[报告、对照图与实际RAW视频](</D:/drone forecast/output/stage107_center_bias_filter_20261022/第一百零六至一百零八阶段连续观测与中心滤波报告.html>)。最新完整RAW实验107，入口`event_drone_forecast/run_center_bias_forecast.py`，保留104原60路并增4条候选；默认54、API93不改。108是当前几何诊断，不是新完整部署。

- 106按105事先选定的train6 110—119.5秒完整RAW475父/950原生，body当前370、线有效353，但四桨初始化0，不能声称已完成正视→侧视→恢复。107对同窗口重跑，二者不能算独立数据。
- 107实际复用固定提交FilterPy原predict/update/reshape_z，无新依赖。状态[cx,cy,vx,vy,bx,by]，将外观偏移和中心速度分开。首次图像对齐参考保留，机架过期/重建不直接改变运动坐标。简单平滑、固定噪声、视角噪声控制同测；协方差/相关系数仍未校准。
- 清晰0.4s相同稀疏框中心目标7/8：bodyKF22.862705、简单平滑22.236070、固定偏移18.537313、视角噪声19.064049、104中心KF19.106234、原融合13.708083。固定版有局部改善，但原完整融合仍更好。侧视没有模型初始化：固定版81.722934，视角降权94.086336反而更坏，原融合66.925684。姿态与位置可靠性不能简单共用退化权重。
- 107冻结后真实RAW815父/1630原生（清晰250、连续475、实际截断60、负例30）、原60和新4严格对照、实际前缀/负例通过；755叠加视频帧全解码。缓存1125当前时刻。受控测量层含真实平移/变向/长缺失/错误几何，不是光电仿真；无几何等价4D/6D有4.55e-13浮点差异，仅独立校验适配到1e-8容差，原冻结推理/参数不改。
- 108只复用当前20ms候选：四候选83（10ms42），合格拟合14（10ms2），最长连续3<原5帧，仍0初始化，未降门槛。没有新RAW/精度提升主张。
- 用户直接核验A3盘、C4盘、BDF成线、E重叠；记录visibility_review/human_visibility107.json。C112.02s的10/20ms均只3候选，是下一轮提取链路重点。核验不作为COM/RPM/姿态真值，不进预测器；这些选中帧属于开发数据。

下一步优先对C附近原生极性支持→阈值/连通域→轮廓/弧段→椭圆拒绝/关联逐级诊断，查明第四盘丢失位置；然后分别评价姿态、几何中心与NN框的位置可靠性。当前新滤波器已接入，不再只停留方案，但侧视恢复、独立RPM收益和实时性仍未实现。所有本轮实验已退出，无后台。

以下105及更早记录为历史。

'''
    if not old.startswith('## 106—108：'):status.write_text(lead+old,encoding='utf-8')
    readme=ROOT/'README.md';old=readme.read_text('utf-8')
    intro='''2026-10-22最新完整实验107：[106—108连续观测/中心滤波报告与视频](</D:/drone forecast/output/stage107_center_bias_filter_20261022/第一百零六至一百零八阶段连续观测与中心滤波报告.html>)。
入口[run_center_bias_forecast.py](run_center_bias_forecast.py)，[107复现说明](research/stage107/README.md)。保留104原60并加4路；中心/速度/外观偏移联合KF实际运行。固定版清晰0.4s18.537px优于bodyKF22.863，仍差于原融合13.708；侧视降权反而94.086px，默认54/API93不替换。连续窗口未初始化；用户核验C四盘可见、算法仅3候选，下一步查提取链路。以下104及更早为历史。

'''
    if not old.startswith('2026-10-22最新完整实验107'):readme.write_text(intro+old,encoding='utf-8')
    log=ROOT.parent/'WORKLOG.md';old=log.read_text('utf-8')
    section='''

## 131. 106—108连续RAW、统一中心偏移滤波与人工可见性核验完成

- 用户明确“去做吧，需要我干什么吗”。先读状态/105方案并复查LOPET、FilterPy作者源码、EVPeriscope；不运行依赖深度/G的Event6D。只读E盘，下载FilterPy固定3b51149提交3个小源码/许可证到本项目，SHA和Git blob记录；无依赖安装。
- 106在密集输出前冻结110—119.5窗口及原104全部参数。真实RAW475父/950原生，body当前370、线方向353，四桨模型0初始化。固定间隔截图和全部失败保存，不声称正侧视完整恢复。
- 107新入口run_center_bias_forecast.py，原60逐项保留、新4；实际执行作者predict/update/reshape_z AST无数值改写。6D中心/速度/外观偏移，首次几何对齐图像偏置在同对象保留，104重建不移运动状态。原bodyKF、100ms校正平滑、固定噪声、几何/侧视噪声控制；sigma6400/位置6沿旧训练参数，新增扩散/相关系数事先冻结而非已校准。
- 6种已知测量层合成：固定版静止位置RMSE0.536（body14.893），平移0.582、变向0.602；长侧视12.974仍差、接受错误几何会产生速度尖峰。不是硬件事件或真实姿态真值。初次4D/6D无几何控制逐位断言失败，复查最大4.55e-13；原冻结源码保留，synthetic_checked107仅将这一验证改为1e-8容差，所有同维前缀精确。
- 缓存clear250/side400/continuous475共1125当前时刻，前缀严格。实际RAW107清晰250+连续475+独立截断60+无机30=815父/1630原生，原60/新4/新状态精确缓存，真实前缀通过，负例无预测；755叠加视频帧全解码。106和107的连续窗口是同一数据，不算两份独立样本。
- 稀疏0.4s7/8框中心端点：clear body22.862705/smooth22.236070/fixed18.537313/adaptive19.064049/104center19.106234/originalfusion13.708083；side body/fixed81.722934/adaptive94.086336/originalfusion66.925684。side没有几何初始化，降权差异来自沿线框噪声，不能把姿态退化等同整机位置不可用。负结果保留、不升默认、不宣称真实COM或RPM收益。
- 额外按固定6时刻读取原生20ms ON/OFF，红ON/青OFF/同像素两类黑，非G、非推测叠加。用户直接反馈A3盘/C4盘/BDF成线/E重叠，另存human_visibility107.json，未进推理。C112.02当前10/20ms均3候选，确认需重点检查提取而非靠已知机架补四。后续使用这些帧须承认开发选择。
- 108在输出前冻结复用原104门槛，仅改当前20ms候选源做缓存诊断：四候选83 vs10ms42；合格拟合14 vs2；最长3帧，原要求5，0初始化。不放宽、不生成隐藏桨心或频率，无新RAW/未来预测提升声明。
- 清晰RAW107单步中位651.69ms、连续305.99ms，四新滤波器缓存总耗时中位约0.81/0.51ms，整链路仍不实时；分次并发录时不可当严格加速比。报告/完整视频/人审/协议/状态和交接归档完毕，默认54/API93保持。所有实验退出。
- 下一步：C附近原生极性支持、阈值连通域、轮廓/弧段、椭圆拒绝及四桨关联逐级保存，先定位第四盘何处丢失；分别估计姿态/几何中心/整机位置可靠性，再验证持续退化恢复。用户本轮核验已足够，不要求其补采飞控真值。
'''
    if '## 131. 106—108连续RAW' not in old:log.write_text(old+section,encoding='utf-8')
    checkpoint=ROOT.parent/'output/checkpoints';h=load(checkpoint/'latest_handoff.json')
    base_archive=Path(h['archive']);base_sha=h['archive_sha256'];assert sha(base_archive)==base_sha
    supplement=Path(h['supplement_archive105']);assert sha(supplement)==h['supplement_archive105_sha256']
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');archive=checkpoint/f'research_through_stage108_{stamp}.zip'
    h.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),last_completed_stage=108,
        completed=sorted(set(h['completed']+[106,107,108])),latest_full_forecast_stage=107,latest_cached_forecast_stage=107,
        latest_diagnostic_stage=108,latest_report=str(report),latest_full_forecast_report=str(report),
        experimental_RAW_entry=str(ROOT/'core/run_center_bias_forecast.py'),experimental_RAW_stage=107,
        completed_scope='106 continuousRAW screening,107 actual center/bias forecasts and all controls,108 current20ms geometry audit.',
        completion_scope='New filter implemented/tested. Side recovery, independentRPM gain and realtime general objective incomplete.',
        latest_checks=str(OUT/'delivery_checks107.json'),in_progress=[],experiments_running=[],default_promoted=False,
        original60_numerically_preserved=True,stable_high_accuracy_realtime_achieved=False,independent_RPM_gain_proven=False,
        next_action='Trace fourth visible disk loss near human-reviewed C112.02: native polarity -> thresholds/components -> contours/arcs -> ellipse rejection -> association. Do not lower5-frame gate to3 just to fit this pilot. Separate pose vs body-position reliability.',
        archive=str(archive),archive_sha256=None,prior_archive=str(base_archive),prior_archive_sha256=base_sha,
        archive_scope='Complete104 base source snapshot plus105 supplement and106-108 current sources/small results/review images. RAW/weights/large forecast JSON/video remain at original paths. Embedded handoff omits archive self-SHA.',
        supplement105_included_in_archive=True,source_summary_files=[str(status),str(log),str(ROOT/'README.md'),str(ROOT/'research/stage107/README.md')],
        report_local_links_checked=count,zip_verified=False)
    h['counts_by_stage'].update({'106':dict(parent=475,native10=950,video_frames_decoded=475),
        '107':dict(parent=815,native10=1630,video_frames_decoded=755,cached_ticks=1125),'108':dict(cached_current_ticks=475,new_RAW=False)})
    h['retained_failed_attempts'] += ['107 cross-dimensional exact no-geometry validator differed4.55e-13; validation-only tolerance adapter retained',
        '106 continuous window no initialization;108 longer current geometry max3 consecutive vs required5, no gate reduction',
        '107 side covariance worsened0.4s81.723→94.086; not promoted']
    save(checkpoint/'latest_handoff.json',h)
    files=[status,log,ROOT/'README.md',ROOT/'core/run_center_bias_forecast.py',checkpoint/'latest_handoff.json']
    for directory in [ROOT/'research/stage106',ROOT/'research/stage107',ROOT/'research/stage108',VENDOR]:
        files += [p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for directory in [SCREEN,OUT,ROOT.parent/'output/stage108_current20_geometry_20261022']:
        files += [p for p in directory.rglob('*') if p.is_file() and p.suffix in ['.json','.html','.png'] and 'cached' not in p.relative_to(directory).parts and p.name not in ['recovery_manifest107.json','final_checks107.json']]
    contents={}
    for source in [base_archive,supplement]:
        with zipfile.ZipFile(source) as z:
            assert z.testzip() is None
            for name in z.namelist():
                if not name.endswith('/'):contents[name]=z.read(name)
    contents.update({p.relative_to(ROOT.parent).as_posix():p.read_bytes() for p in files})
    manifest={name:hashlib.sha256(data).hexdigest() for name,data in contents.items()}
    save(OUT/'recovery_manifest107.json',dict(archive=str(archive),files=manifest))
    contents[(OUT/'recovery_manifest107.json').relative_to(ROOT.parent).as_posix()]=(OUT/'recovery_manifest107.json').read_bytes()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in contents.items():z.writestr(name,data)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
    h.update(archive_sha256=sha(archive),zip_verified=True,execution_status='checkpoint_ready');save(checkpoint/'latest_handoff.json',h)
    save(OUT/'final_checks107.json',dict(archive=str(archive),archive_sha256=sha(archive),verified_files=len(contents),
        local_links_checked=count,source_protocol_verified=True,actual_RAW107_parent=815,native10=1630,
        video_frames_decoded=755,no_default_promotion=True,no_realtime_claim=True,human_visibility_review_received=True))
    print(json.dumps(dict(report=str(report),archive=str(archive),verified_files=len(contents)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
