"""Validate artifacts, explain111, update handoff and create verified snapshot."""
from pathlib import Path
import datetime,hashlib,json,re,sys,zipfile
from html.parser import HTMLParser
from urllib.parse import unquote,urlparse
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage110'),str(ROOT/'research/stage109')]
from study110 import OUT,verify,save,sha
AUDIT=ROOT.parent/'output/stage109_rotor_extraction_audit_20261022'
DIAG=ROOT.parent/'output/stage111_conic_budget_diagnostic_20261022'

def load(p):return json.loads(Path(p).read_text('utf-8-sig'))

def main():
    verify();r=load(DIAG/'diagnostic111.json');s=r['summary'];assert s['frames']==475 and s['refitted']==44
    c=next(x for x in r['records'] if abs(x['t']-112.02)<1e-8)
    import numpy as np
    shift=float(np.linalg.norm(np.asarray(c['new']['center'])-c['old']['fit']['geometric_center']))
    bounds=sum(x['new'].get('fit',{}).get('frame_aspect',1)>3.999 or x['new'].get('fit',{}).get('frame_aspect',1)<.2501 for x in r['records'] if x['new']['valid'])
    save(DIAG/'interpretation111.json',dict(C_center_shift_px=shift,C_old_rms_px=c['old']['fit']['rms_px'],C_new_rms_px=c['new']['fit']['rms_px'],
        C_new_frame_aspect=c['new']['fit']['frame_aspect'],newly_valid_on_aspect_bound=bounds,newly_valid_total=s['new_valid'],
        interpretation='Numerical convergence can coexist with weak model identification. Small residual change causes10px center shift and aspect hits4 upper bound. Not ground truth error, not proof wrong pose. Existing diagonal-center noise proxy omits full conic/shape uncertainty.',
        next='Use observed support arcs rather than full synthetic perimeters of threshold-fitted ellipses. Diagnose conic/shape parameter uncertainty and bound sensitivity; preserve usable body position separately from degraded pose. Ordinary smoothing/real translation controls required.'))
    report=OUT/'第一百零九至一百一十阶段桨盘合并与中心预测报告.html';page=report.read_text('utf-8')
    extra=f'''<section id="budget111"><h2>111补充：加大求解预算能否解决</h2><p>固定将44个当前求解耗尽的时刻从80次提高到320次，新增17个合格拟合，21个仍未收敛，6个转为共面圆不一致；475时刻中仍然0次模型初始化。未修改原80次代码，也未降低五帧门槛；这是缓存几何诊断，没有新预测。</p><p>C帧变为数值合格，但残差仅从{c['old']['fit']['rms_px']:.3f}降到{c['new']['fit']['rms_px']:.3f}px，几何中心移动{shift:.2f}px，机架长宽比触及4.0上界。新增17个合格拟合中{bounds}个触及长宽比边界。该变化不是已知重心误差，但说明“优化收敛”不能代替中心/模型可靠性验证。原先只传播四个中心位置噪声的代理没有覆盖完整椭圆与机架参数不确定性。</p><p>因此下一步优先把实际可见支持弧段进入联合拟合，并检查参数边界、残差对中心的敏感性；将姿态可靠性、中心可靠性、机体位置可靠性分别处理。只增加迭代和候选还不能完成侧视预测，也不能作为原创算法贡献。</p><p><a href="../stage111_conic_budget_diagnostic_20261022/diagnostic111.json">111全部固定诊断</a> · <a href="../stage111_conic_budget_diagnostic_20261022/interpretation111.json">中心敏感性解释</a> · <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html">SciPy求解/终止条件说明</a></p></section>'''
    if 'id="budget111"' not in page:page=page.replace('</body>','') if '</body>' in page else page;page=page.replace('</html>',extra+'</html>')
    report.write_text(page,encoding='utf-8')
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.urls=[]
        def handle_starttag(self,tag,attrs):self.urls.extend(v for k,v in attrs if k in ['src','href'] and v)
    parser=Links();parser.feed(page);local=0
    for url in parser.urls:
        if urlparse(url).scheme or url.startswith('#'):continue
        assert (OUT/unquote(url.split('#')[0])).resolve().is_file(),url;local+=1
    for case in ['continuous6','clear6','negative5']:assert load(AUDIT/case/'audit109.json')['exact_old_geometry']
    check=load(OUT/'delivery_checks110.json');assert all(v.get('sources_unchanged',True) for v in check['checks'].values())
    save(OUT/'delivery_failure110.json',dict(component='post-forecast report scorer first attempt',error='KeyError anchor_t on an unavailable anchor',
        repair='Read anchor_t only when anchor_available; preserve invalid anchors in denominator as original107. Frozen prediction code/parameters/results unchanged.',numeric_forecast_changed=False))
    lead=f'''## 109—111：漏桨来源修复与共面求解诊断完成

[新报告与C帧对照](</D:/drone forecast/output/stage110_merge_safe_geometry_20261022/第一百零九至一百一十阶段桨盘合并与中心预测报告.html>)。最新完整RAW部署仍107，默认54/API93不变；110是新RAW几何＋同期既有机体/轴观测回放实验，111是缓存当前几何诊断。

- 109只记录原生ON/OFF到分组链路，555个20ms旧几何逐帧精确。C112.02左侧两个开运算分量（287/584像素）被3×3闭运算合为913像素，最终被拟合成一个大椭圆。
- 110固定闭运算（先判线）、全局不闭运算、条件保留合并前合格分量三路；现有五帧/共面门槛不改，不补隐藏点。四段1155＋两个实际截断各60＝1275个20ms读取，每个时刻三路；侧视/连续重叠，不当独立录像。前缀精确，body-only KF三路逐帧精确，负例无初始化。
- C已从3候选恢复4个有事件支持的轮廓。连续475时刻四候选85→128、合格共面10→17，仍0初始化。全局不闭运算四候选146/合格18，也0初始化，不能将候选增加称成功。
- 清晰0.4s固定偏移KF：闭运算/条件保护均16.842569，全局不闭运算18.003784，bodyKF22.862705，原融合13.708083；侧视各固定版81.722934、原融合66.925684。各7/8个既有稀疏框端点。条件保护没有独立预测收益，不能把本轮20ms输入相对107的变化归功于拆分。
- 111将44个80次求解耗尽时刻固定320次诊断：新增17合格、21仍耗尽、6不一致，仍0初始化。C残差2.890→2.854px却中心移{shift:.2f}px、长宽比触及4.0。不是已知真值误差；说明需检查完整椭圆/机架参数的不确定性，不能只看收敛或只传播桨心位置噪声。
- 核查AAMED作者仓库和OpenCV/SciPy文档，实际复用本机工具及已有前端/拟合/滤波；AAMED未运行、无新依赖/G、E盘只读、不接触yolov10-main。形态学修复本身不称原创。

下一步：真实支持弧段联合拟合与参数边界/中心敏感性诊断；分别管理姿态、几何中心与整机位置可靠性，再验证真实平移/变向和静止姿态摆动。不要降低初始化门槛或直接用单线补四桨。完整侧视恢复、独立RPM收益、实时性和原创方法仍未完成。本轮实验已结束，无后台。

以下108及更早为历史。

'''
    status=ROOT.parent/'PROJECT_STATUS_20261022.md';text=status.read_text('utf-8')
    if not text.startswith('## 109—111：'):status.write_text(lead+text,encoding='utf-8')
    readme=ROOT/'README.md';text=readme.read_text('utf-8')
    intro='2026-10-22最新模块实验109—111：[桨盘合并与中心预测报告](</D:/drone forecast/output/stage110_merge_safe_geometry_20261022/第一百零九至一百一十阶段桨盘合并与中心预测报告.html>)、[复现/边界](research/stage110/README.md)。C四桨从3恢复4候选，但连续模型仍0初始化；80→320求解诊断暴露中心/结构参数敏感性。原完整RAW107与默认54/API93保持。以下108及更早为历史。\n\n'
    if not text.startswith('2026-10-22最新模块实验109—111'):readme.write_text(intro+text,encoding='utf-8')
    log=ROOT.parent/'WORKLOG.md';text=log.read_text('utf-8')
    section='\n\n## 132. 109—111：创新目标收窄后的提取修复与求解诊断\n\n'+lead.split('\n\n',2)[2].split('以下108及更早为历史。')[0]+'\n- 初次报告评分器对不可用锚点读取anchor_t失败，后仅修复报告器，所有无效锚点保留；冻结推理/参数/预测不变，delivery_failure110.json记录。\n'
    if '## 132. 109—111：' not in text:log.write_text(text+section,encoding='utf-8')
    checkpoint=ROOT.parent/'output/checkpoints';h=load(checkpoint/'latest_handoff.json');prior=Path(h['archive']);assert sha(prior)==h['archive_sha256']
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');archive=checkpoint/f'research_through_stage111_{stamp}.zip'
    h.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),last_completed_stage=111,completed=sorted(set(h['completed']+[109,110,111])),
        latest_full_forecast_stage=107,latest_cached_forecast_stage=110,latest_diagnostic_stage=111,latest_RAW_geometry_stage=110,
        latest_report=str(report),latest_checks=str(OUT/'delivery_checks110.json'),in_progress=[],experiments_running=[],default_promoted=False,
        completed_scope='109 exact extraction trace;110 RAW geometry with cached contemporaneous body/axis and matched center controls;111 current fit-budget diagnostic.',
        completion_scope='C four disks restored as candidates; continuous model still not initialized. Full side recovery, independent RPM benefit, realtime and originality remain incomplete.',
        next_action='Fit actual observed support arcs, quantify full conic/shape uncertainty and boundary sensitivity; keep pose/geometry-center/body-position confidence separate. Evaluate genuine translation/turn and stationary-wobble controls; do not lower five-frame admission.',
        prior_archive=str(prior),prior_archive_sha256=h['archive_sha256'],archive=str(archive),archive_sha256=None,zip_verified=False,
        source_summary_files=[str(status),str(log),str(readme),str(ROOT/'research/stage110/README.md')],report_local_links_checked=local,
        archive_scope='Verified prior108 snapshot plus109-111 sources/small diagnostic results/images/counts and current handoff. RAW/weights/large JSONL/videos remain original paths; embedded handoff omits self SHA.')
    h['counts_by_stage'].update({'109':dict(RAW_current20=555,old_geometry_exact=True),'110':dict(RAW_current20=1275,three_geometry_streams=True,cached_current_body=True,complete_RAW_deployment=False),'111':dict(cached_frames=475,current_fits_refitted=44,new_RAW=False)})
    h['retained_failed_attempts']=list(dict.fromkeys(h['retained_failed_attempts']+['110 report scorer unavailable anchor field fixed only in delivery','110 C disks restored but conic budget exhausted;111 more budget still zero continuous initialization']))
    save(checkpoint/'latest_handoff.json',h)
    files=[status,log,readme,checkpoint/'latest_handoff.json']
    for folder in [ROOT/'research/stage109',ROOT/'research/stage110',ROOT/'research/stage111',AUDIT,OUT,DIAG]:
        files += [p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.jsonl' and p.name not in ['recovery_manifest110.json','final_checks110.json']]
    contents={}
    with zipfile.ZipFile(prior) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if not name.endswith('/'):contents[name]=z.read(name)
    contents.update({p.relative_to(ROOT.parent).as_posix():p.read_bytes() for p in files})
    manifest={name:hashlib.sha256(data).hexdigest() for name,data in contents.items()}
    save(OUT/'recovery_manifest110.json',dict(archive=str(archive),files=manifest))
    contents[(OUT/'recovery_manifest110.json').relative_to(ROOT.parent).as_posix()]=(OUT/'recovery_manifest110.json').read_bytes()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in contents.items():z.writestr(name,data)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
    h.update(archive_sha256=sha(archive),zip_verified=True,execution_status='checkpoint_ready');save(checkpoint/'latest_handoff.json',h)
    save(OUT/'final_checks110.json',dict(archive=str(archive),archive_sha256=sha(archive),verified_files=len(contents),
        local_links_checked=local,source_protocol_verified=True,latest_full_RAW_stage=107,default_promoted=False,
        C_four_candidates_restored=True,continuous_initialized=False,COM_truth_available=False,new_forecast_gain_proven=False))
    print(json.dumps(dict(report=str(report),archive=str(archive),verified_files=len(contents),C_center_shift_px=shift)),flush=True)

if __name__=='__main__':main()
