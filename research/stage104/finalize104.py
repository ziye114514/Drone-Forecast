"""Write handoff and a verified source/small-result recovery snapshot."""
from study104 import *
from html.parser import HTMLParser
from urllib.parse import unquote,urlparse
import zipfile


def main():
    verify();checks=load(OUT/'delivery_checks104.json');assert len(checks['raw'])==4
    report=Path(checks['report']);matched=load(OUT/'matched_controls104.json')
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ['src','href'] and v:self.links.append(v)
    parser=Links();parser.feed(report.read_text('utf-8'));local=[]
    for url in parser.links:
        if urlparse(url).scheme or url.startswith('#'):continue
        path=(OUT/unquote(url.split('#')[0])).resolve();assert path.is_file(),str(path);local.append(str(path))
    save(OUT/'report_link_checks104.json',dict(local_links=len(local),all_exist=True))
    status=ROOT.parent/'PROJECT_STATUS_20261022.md';old=status.read_text('utf-8')
    lead='''## 104：机体中心与四桨投影分离已接入实验RAW（2026-10-22）

最新实验入口 `event_drone_forecast/run_planar_center_forecast.py`，保留93全部55路并加5路；默认54不变，Python接口仍93。报告：
[104报告、模型与预测视频](</D:/drone forecast/output/stage104_joint_center_20261022/第一百零四阶段机体中心与四桨姿态分离报告.html>)。

- 四桨共圆拟合/几何中心与投影布局分开，连续5帧初始化；按几何条件降权。缺桨校正短时保留160ms、300ms归零、500ms过期；虚拟桨心不进入测量。视角法向混合、独立有效增益与恒定/零控制已连接。
- 固定中心摆动受控位移RMSE21.117→0.045px；真实平移22.398→0.073、变向22.427→0.044。人为几何、初始对齐和0.3px轮廓噪声，不是实拍重心精度。侧视缺桨后退回框中心，仍退化。
- 清晰250时刻模型更新210次/保持8次；侧视400和3号450未初始化。清晰/侧视完整RAW及真实1.2秒前缀、0.6秒负例共740父/1480原生；旧55精确93，新5精确当前特征缓存，实际前缀通过。3号只做缓存复算。原650视频+30负例帧完整解码。
- 同body-only位置和KF参数，清晰0.4s误差22.863→19.106px；相同body-only历史17.210→26.165变差，原完整融合13.708仍更好。仅7/8稀疏未来框中心锚点，非真实机体重心验证。侧视/3号未初始化，新中心控制与body-only逐项相同，不能算机架收益。
- 新有效力clear23.701/constant23.848/zero23.850，侧视74.728/74.950/74.930，3号94.496/94.486/94.496；不足以证明RPM独立增益。RAW单步中位清晰730.59ms/侧视381.33ms，仍不实时，分次/并发运行耗时不可当严格加速比。
- 缓存前缀校验曾误比较嵌套拟合耗时；仅修校验器、原冻结/源码和修订保留。首次RAW主循环成功，外层旧校验器因源码修订未过，随后对原结果核验通过。没有改推理、参数或评分去迎合结果。

下一步：同一连续近景中的正视建模→侧视短时缺桨→恢复；先做中心不确定度与模式交接，防止校正变化污染速度/加速度。不能把不同时间段的成功建模强行延续到几十秒后的侧视。所有104实验已结束，无后台。

以下103及更早状态为历史，恢复时以本段与最新handoff为准。

'''
    if '## 104：机体中心与四桨投影分离已接入实验RAW' not in old:
        status.write_text(lead+old,encoding='utf-8')
    readme=ROOT/'README.md';old=readme.read_text('utf-8')
    intro='''2026-10-22 新实验104：[机体中心/四桨姿态分离报告与视频](</D:/drone forecast/output/stage104_joint_center_20261022/第一百零四阶段机体中心与四桨姿态分离报告.html>)。
入口 [run_planar_center_forecast.py](run_planar_center_forecast.py)，[实现/复现说明](research/stage104/README.md)。旧55路保留并增5路；默认54及93 Python接口不改。仿真通过，清晰段四桨中心更新210/250；侧视未初始化。相同body-only KF清晰0.4s22.863→19.106px，但原完整融合13.708更好，历史外推反而退步。仍无真实重心/RPM验证或实时性保证。下列93及更早说明保留为历史。

'''
    if not old.startswith('2026-10-22 新实验104'):readme.write_text(intro+old,encoding='utf-8')
    worklog=ROOT.parent/'WORKLOG.md';old=worklog.read_text('utf-8')
    log='''

## 129. 104机体中心/四桨投影分离，真实RAW和失败对照完成

- 用户核对五圆模型/侧视分支/固定中心摆动后明确“去做”。先复查IPPE作者仓库、圆心投影修正作者项目、2604.18289和HVC-Net；实际复用本地58 joint_fit与49 intersection、93 Kinematic、23历史、72增益、77 MMAE，不使用未知相机K跑IPPE或声称复现这些完整系统。
- 新rigid104连续5帧拟合四个当前共面圆，中心与布局分开；初始化图像偏置对齐框中心一次、形状尺度比冻结。平滑视角权重/有限差分噪声条件门槛，160ms校正保持/300ms归零/500ms重新初始化。仅几何中心与相对对齐位置，非真实重心；虚拟点不作为测量。视角法向混合及独立校正中心有效增益、恒定/零控制进入online104，新RAW入口原93全部55精确保留，加5路，不改默认54/API93。
- 冻结protocol104后500帧合成：固定中心、平移、变向误差21.117→0.045、22.398→0.073、22.427→0.044px（位移相对锁定；噪声0.3px）；真实平移207.9px未被压成静止。侧视缺桨仍19.36px，退回整机位置；非共圆干扰未初始化。无实际重心/姿态真值。
- 1100时刻缓存：clear联合210/250、短时保持8；side0/400、train3 0/450。清晰/侧视完整RAW650父+60真实截断+30负例=740父/1480原生；旧55严格93，新5严格缓存，实际截断前缀/空场通过。清晰/侧视及负例视频680帧全解码，输出另加96px信息栏以免遮挡原生画面。3号只复算缓存，不冒充新RAW。
- 增补匹配body-only输入与时间戳控制：清晰0.4s KF22.862705→19.106234、历史17.210441→26.164766；原完整融合13.708083更好。有效7/8稀疏框中心目标，不是独立COM精度。side/train3中心分支与body-only逐项相等，原kinematic_train6400的位置来源差异不归功于模型。有效力/恒定/零差别小，未建立RPM独立增益。
- 初始cached前缀只因nested fit.elapsed_ms不确定耗时比较失败，所有预测已精确；保留study104_initial_prefix_check.py，validation_amendment仅更新校验器SHA并递归去掉两种耗时字段。原协议不改，参数/推理/评分不调。启动较早的clearRAW外层仍用旧校验器而报SHA失败，原RAW完成/source unchanged后用修复校验器查同份结果通过；没有因此重跑挑结果。
- 当前完整RAW清晰730.59ms/侧视381.33ms中位，仍不实时。模型本身缓存额外中位44.31ms，非整链路配对开销。重心定义、错误桨盘/畸变、侧视缺初始化、状态交接/导数污染仍是瓶颈。
- 新报告、全视频、失败/协议、README/状态与交接归档完成。不访问yolov10-main、不用G、不改E盘、不装新依赖/发外部消息。所有实验已退出。下一步同一连续近景正视→侧视→恢复，再解决中心不确定度/交接；默认不推广。
'''
    if '## 129. 104机体中心/四桨投影分离' not in old:worklog.write_text(old+log,encoding='utf-8')
    checkpoint=ROOT.parent/'output/checkpoints';handoff=load(checkpoint/'latest_handoff.json')
    oldarchive=Path(handoff['archive']);prior_sha=sha(oldarchive)
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');archive=checkpoint/f'research_through_stage104_{stamp}.zip'
    handoff.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),last_completed_stage=104,
        completed=sorted(set(handoff['completed']+[104])),latest_full_forecast_stage=104,latest_cached_forecast_stage=104,
        latest_report=str(report),latest_full_forecast_report=str(report),previous_report=str(BASE/'第八十八至一百零三阶段实验总览.html'),
        latest_checks=str(OUT/'delivery_checks104.json'),experimental_RAW_entry=str(ROOT/'run_planar_center_forecast.py'),experimental_RAW_stage=104,
        completed_scope='104 center/layout separation with known-motion controls, matched body-only controls, four actual RAW checks and native videos.',
        completion_scope='104 research branch implemented and audited; general high-accuracy realtime objective incomplete.',
        in_progress=[],experiments_running=[],default_promoted=False,accuracy_changed=True,
        original55_numerically_preserved=True,stable_high_accuracy_realtime_achieved=False,event_RPM_advantage_proven=False,
        next_action='Continuous near-range clear→edge-on→recovery test, center uncertainty and state-transition handling; do not promote104 or repeat tuning on sparse development targets.',
        prior_archive=str(oldarchive),prior_archive_sha256=prior_sha,archive=str(archive),archive_sha256=None,
        archive_scope='Prior source/small-results snapshot plus104 sources/report/protocol/plots/checks. RAW, weights, full cached forecasts and video remain at original paths. Inside-archive handoff omits self-referential archive SHA.',
        report_local_links_checked=len(local),zip_verified=False)
    handoff['counts_by_stage']['104']=dict(parent=740,native10=1480,video_frames_decoded=680,cached_other_recording_ticks=450)
    handoff['retained_failed_attempts'].append('104 initial validator compared nested elapsed time; original source retained, deterministic fields/predictions checked after validation-only amendment; first RAW outer old-SHA check subsequently rechecked.')
    handoff['source_summary_files']=[str(status),str(worklog),str(ROOT/'README.md'),str(ROOT/'research/stage104/README.md')]
    save(checkpoint/'latest_handoff.json',handoff)
    additions=[status,worklog,ROOT/'README.md',ROOT/'run_planar_center_forecast.py',checkpoint/'latest_handoff.json']
    additions+=list((ROOT/'research/stage104').glob('*.py'))+list((ROOT/'research/stage104').glob('*.md'))
    additions+=[p for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.json','.html','.png','.py'] and 'cached' not in p.relative_to(OUT).parts]
    overlay={p.relative_to(ROOT.parent).as_posix():p.read_bytes() for p in additions}
    contents={}
    with zipfile.ZipFile(oldarchive) as oldzip:
        assert oldzip.testzip() is None
        for name in oldzip.namelist():
            if not name.endswith('/'):contents[name]=oldzip.read(name)
    contents.update(overlay)
    # A recovery manifest cannot contain its own previous digest when a
    # snapshot is refreshed; exclude it from the hashed payload first.
    contents.pop((OUT/'recovery_manifest104.json').relative_to(ROOT.parent).as_posix(),None)
    manifest={name:hashlib.sha256(data).hexdigest() for name,data in contents.items()}
    save(OUT/'recovery_manifest104.json',dict(archive=str(archive),files=manifest))
    contents[(OUT/'recovery_manifest104.json').relative_to(ROOT.parent).as_posix()]=(OUT/'recovery_manifest104.json').read_bytes()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in contents.items():z.writestr(name,data)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
    handoff.update(archive_sha256=sha(archive),zip_verified=True,execution_status='checkpoint_ready')
    save(checkpoint/'latest_handoff.json',handoff)
    save(OUT/'final_checks104.json',dict(source_protocol_verified=True,local_report_links=len(local),archive=str(archive),
        archive_sha256=handoff['archive_sha256'],verified_files=len(contents),actual_RAW_parent=740,actual_RAW_native10=1480,
        video_frames_decoded=680,default_promoted=False,full_general_objective_complete=False))
    print(json.dumps(dict(report=str(report),archive=str(archive),verified_files=len(contents)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
