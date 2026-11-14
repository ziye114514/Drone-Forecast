"""Deliver a bounded next experiment plan and preserve the audit handoff."""
from audit105 import *
import html,zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=load(OUT/'protocol105.json');assert p['sources']==sources()
    audit=load(OUT/'audit105.json');data=load(BASE/'cached/clear6/forecasts.json')
    tt=np.array([r['t'] for r in data]);cc=np.array([r['rigid_center_diagnostic']['correction_px'] for r in data]);dt=np.diff(tt)
    vel=np.linalg.norm(np.diff(cc,axis=0)/dt[:,None],axis=1)
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,constrained_layout=True)
    axes[0].plot(tt,cc[:,0],label='x correction');axes[0].plot(tt,cc[:,1],label='y correction');axes[0].set_ylabel('Correction (px)');axes[0].legend()
    axes[1].plot(tt[1:],vel,label='Correction first-difference norm');axes[1].set(xlabel='Source time (s)',ylabel='Component (px/s)')
    for e in audit['switches']:
        if e['case']=='clear6':
            for ax in axes:ax.axvline(e['t'],color='#bc3150',alpha=.6,ls='--')
    for ax in axes:ax.grid(alpha=.25)
    fig.suptitle('Existing104 clear clip: dashed lines are mode changes; components are NOT motion ground-truth errors')
    fig.savefig(OUT/'transition_components105.png',dpi=160);plt.close(fig)
    plan='''# 下一轮研究方案：可观测性变化下的中心稳定与二维预测

截至2026-10-22，104是最新完整RAW实验；105只完成现有输出的切换审计、文献核查和下一轮方案，不是新预测算法或新的准确率成绩。默认54不变。

## 先明确要验证的研究命题

在仅输入原生事件、未知具体机型尺寸、不依赖飞控/IMU日志的条件下，能否利用四桨共面结构和整机线状侧影，把姿态投影引起的外观中心变化与真正平移分开；当可用约束减少/恢复时，维持统一的中心运动状态，减少虚假速度和加速度，改善未来0.1/0.2/0.4秒预测？

这是待验证命题，不能预先宣称首次。四桨模型、线特征、滤波、按置信度加权本身都有先例。

## 已做的第一步：105切换审计

只读104三个片段的当前因果诊断，共1100时刻，没有读未来标签、重新读取RAW或改预测器。清晰250时刻只有4次模式变化；校正项的逐帧差分速度分量最大583.33px/s、95分位287.29px/s，而切换时刻最大55.81px/s。最大二阶差分分量52851.41px/s²。侧视和3号没有建模，校正恒为零。

这些是校正项进入差分的数学分量，不是真实速度/加速度误差，也不能据此证明104历史外推变差的全部原因。它们否定了“主要只有模式切换才有大波动”的简单解释：优先研究连续拟合噪声、位置偏置和滤波，再处理模式交接。

## 顺序1：先建立真正连续的验证片段

1. 复用现有原生ON/OFF预览，train6的110/112秒能见多个桨盘，114/116/118秒逐渐呈线状，候选窗口为110—119.5秒。它只是目测筛选候选，尚未密集验证，也不保证算法能成功初始化。104的22—27秒清晰段不能隔八十多秒直接接到108秒侧视。
2. 先对该候选做固定间隔密集筛查，记录四桨当前可见数、共圆拟合是否通过、线状条件、遮挡、目标尺度和相机运动，不读预测误差来挑片段。保留不成功的窗口。
3. 只有出现连续“已建模→退化→恢复”才进入恢复测试；否则扩大到其他既有录像。不得靠延长陈旧状态寿命伪造跟踪成功。
4. 开发窗口与评测窗口按录像/录制段冻结。已有反复开发的片段必须继续称开发数据；未用片段也不能自动称不同机型盲测。

交付：连续小视频、固定筛选规则和可用性表。先证明数据包含问题，再比较方法。

## 顺序2：分开位置、外观偏置与噪声

先做诊断模型：`框中心观测 = 机体图像中心 + 随姿态变化的外观偏置 + 噪声`。完整桨盘联合拟合提供几何中心约束；框中心提供整机定位，但二者来自相同事件，不能假设为独立真值并重复增加信息。

第一轮只比较三个版本：

- 104原始中心校正。
- 相同中心观测加普通因果平滑/带测量噪声的KF，作为必要基线。
- 统一中心运动状态、显式外观偏置/观测不确定度的候选方法。

所有版本输入时间戳、可见性和运动滤波参数尽量匹配；不借助未来窗口平滑。先测中心稳定性和真实运动保留，再看预测收益。若普通平滑就解决问题，就据实报告，不把简单降噪包装成新算法。

## 顺序3：侧视只更新仍能测到的量

四桨清楚时，使用联合圆锥曲线/机架中心和方向约束；四点扁平、近共线时，根据雅可比/拟合敏感性减小不可靠约束的信息。

单条不完整线段通常能约束线方向、以及中心垂直于线的坐标；沿线中心依赖完整端点或其他证据。不能默认取中点就是真实桨心，也不能从一条线恢复全部姿态。

侧视候选观测使用不同方向的不确定度：沿线不确定度增大，垂线方向仅在当前支持可靠时更新。保持同一个中心/速度状态，改变观测方程与信息权重；不要靠更换一个中心坐标再对它求差分实现切换。恢复时用当前残差、几何新鲜度和持续支持重新约束，保留旧状态和协方差，不凭虚拟桨心自我确认。

这套设计的通用滤波思想已有研究。可能的贡献应落在无人机桨盘退化的具体观测模型、误差传播/可靠性设计和实验证据上。

## 顺序4：冻结对照与成功/失败标准

必做控制：中心固定但姿态摆动、真实匀速平移、突然转向、正视→侧视→恢复、遮挡100/300/1000ms、错桨/合并轮廓、无机背景。静止不能作为运行先验。长时间退化允许不确定度增加和回退，不能要求算法凭不存在的信息始终正确。

至少对比：原完整融合93、104、普通因果平滑、固定噪声的统一中心状态、退化感知的统一中心状态。切换模块、几何降权和外观偏置逐项消融；保持相同预测有效覆盖，不能删除失败样本再算平均。

指标分开报告：

- 当前图像中心误差/抖动与真实平移保留；重心没有独立真值时不得声称真实COM精度。
- 侧视期间有效覆盖、丢失时长、错误持续跟踪和恢复延迟。
- 切换前后0.1/0.2/0.4秒ADE/FDE与尾部误差；特别单列突然转向。
- 因果前缀一致性、实际端到端延迟；视频50fps不等于实时。

在看新评测结果前设定实用收益门槛：相对强基线的侧视过渡误差有稳定下降，同时清晰段和真实转向没有明显退步。若样本少，先报告区间与每段结果，不凭一个平均值宣布成功。

真实数据参考可以是独立人工标注的可见机体图像中心/稳定特征点，并记录定义和不确定度；它不是已知真实三维重心。标注只用于评测，不进入RAW推理。没有IMU/ESC不妨碍先做二维中心研究，但不能验证真实RPM、风速和质量。

## 顺序5：中心稳定后再验证推力的独立帮助

保留原运动分支。先冻结几何/中心估计，再比较真实事件周期、恒定强度、零修正、过去打乱周期和仅运动历史；资格时钟相同。新频率分支应在变向片段稳定优于这些控制，才能声称从桨叶得到额外预测信息。

未知质量、推力系数、尺度等先用可辨识的图像域等效增益表示，扰动保留残差项；不能用一条二维轨迹分别宣称识别出质量、风和全部推力参数。

## 可以复用什么，不能直接套什么

- [LOPET / TIP2024论文](https://arxiv.org/abs/2408.03225)，[作者MATLAB代码](https://github.com/Zibin6/LOPET)：可参考事件—线匹配、稳健残差与权重。依赖对象模型，不是无需模型的无人机侧视方案；本轮只核查，没有运行代码。
- [ICRA2024直接模型跟踪作者代码](https://github.com/BluecatLi/direct_event_object_tracker/)：模型投影/位姿持续更新的参考，依赖3D模型和初始化，ROS环境；本轮没有执行。
- [EVPeriscope](https://arxiv.org/html/2609.11920v1)：四桨几何与EKF的直接相关对照，实际方法要求四桨心可见，不能将其地面机器人传感器遮挡实验理解为已解决桨叶遮挡。
- [Event6D / CVPR2026](https://chohoonhee.github.io/Event6D/)：事件与深度联合跟踪，并重建强度/深度；不符合当前只有RAW、无额外深度和不用G的输入范围。只作为相关工作，不直接接入。

以上相关工作说明本方向有基础，不证明无人做过我们的具体组合。新颖性和实验有效性需要分别核实。

## 当前交付边界

已完成105切换/差分审计和本方案；还没有实现上述新滤波器、密集读取110—119.5秒新RAW、生成新的精度成绩或完成论文级验证。最新完整实验仍104；无后台实验，默认54及API93不变。
'''
    note=OUT/'下一轮研究方案与切换诊断.md';note.write_text(plan,encoding='utf-8')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>105：下一轮研究与切换审计</title><style>body{max-width:1100px;margin:30px auto;padding:0 18px;font:17px/1.8 "Microsoft YaHei",Arial;color:#24394b}img{width:100%}pre{white-space:pre-wrap;font:inherit;padding:20px;background:#f3f7f9}a{color:#175da6}</style><h1>105：先研究中心波动，再验证侧视交接</h1><p>本轮完成现有输出审计与研究方案；没有新预测器或新的RAW精度成绩。图中的差分分量不是真实运动误差。</p><img src="transition_components105.png"><p><a href="下一轮研究方案与切换诊断.md">下载完整方案</a> · <a href="audit105.json">全部审计数值</a> · <a href="protocol105.json">冻结协议</a></p><pre>'''+html.escape(plan)+'''</pre></html>'''
    report=OUT/'第一百零五阶段切换诊断与研究方案.html';report.write_text(page,encoding='utf-8')
    status=ROOT.parent/'PROJECT_STATUS_20261022.md';old=status.read_text('utf-8')
    lead='''## 105：下一轮方案与切换审计完成（无新预测器）

[105方案/诊断图](</D:/drone forecast/output/stage105_transition_audit_20261022/第一百零五阶段切换诊断与研究方案.html>)；原104仍是最新完整RAW，默认54/API93不改。105只读既有1100时刻诊断，无RAW重读或未来真值输入。

清晰段只有4次模式变化；校正项差分速度最大583.33px/s，而切换点最大55.81px/s。大差分主要发生在连续拟合期，优先查中心观测噪声/外观偏置，不能都归因于切换；这些分量不是真实运动误差。下一轮按方案：密集筛查train6 110—119.5秒连续候选→匹配简单平滑基线/统一中心偏置模型→侧视只更新可观测分量→强基线及恢复/真实转向对照→再验证周期推力收益。105未实现新滤波器、未密集扫描新RAW、未运行LOPET/Event6D，无后台。

'''
    if not old.startswith('## 105：'):status.write_text(lead+old,encoding='utf-8')
    worklog=ROOT.parent/'WORKLOG.md';old=worklog.read_text('utf-8')
    if '## 130. 105切换审计与下一轮研究方案' not in old:
        worklog.write_text(old+'''

## 130. 105切换审计与下一轮研究方案

- 用户要求继续深挖、询问怎么做。复查LOPET/TIP2024作者论文和MATLAB代码、ICRA2024直接模型跟踪，以及新查Event6D/CVPR2026。LOPET可参考稳健事件线残差但需对象模型；Event6D需要深度和强度/深度重建，不接当前RAW-only/noG。本轮均仅文献核查，未下载或运行这些系统。
- 105冻结协议后只读104三段1100当前诊断，不读未来真值、不改参数/预测。clear4次模式变化，校正差分速度最大583.327px/s，切换点最大55.806；非切换点95分位287.492。差分分量不是实际速度误差，不能把104历史退步的因果责任全归模式切换。由此将下一轮优先级改为中心噪声/外观偏置→可观测性与模式衔接。
- 查看旧原生ON/OFF接触表：train6 110/112与114/116/118视角变化，110—119.5候选需密集筛查，未声称已找到成功四桨初始化或完整恢复序列。22秒模型不得跨八十多秒转移。
- 完整研究方案/HTML/图/协议保存stage105_transition_audit_20261022：先连续数据、匹配普通因果平滑、统一中心/偏置模型、侧视方向和单线约束的可观测分量、消融与独立评测，再验证事件周期相对恒定/零/打乱的收益。新增算法尚未实现，无新RAW精度或创新已证实主张。最新完整RAW仍104，默认54/API93不改，无后台。
''',encoding='utf-8')
    checks=ROOT.parent/'output/checkpoints';h=load(checks/'latest_handoff.json');base_archive=h['archive'];base_sha=h['archive_sha256']
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');archive=checks/f'research_stage105_delta_{stamp}.zip'
    h.update(last_completed_stage=105,completed=sorted(set(h['completed']+[105])),latest_report=str(report),
        latest_diagnostic_stage=105,latest_full_forecast_stage=104,latest_cached_forecast_stage=104,
        completed_scope='105 current-only transition audit and next research plan; no new forecast algorithm.',
        latest_checks=str(OUT/'audit105.json'),next_action='Read105 plan. First dense current-only screen of train6 110-119.5s, then matched smoothing/bias-state controls and observable-only side constraints; no claim that105 is implemented.',
        source_summary_files=[str(status),str(worklog),str(note)],in_progress=[],experiments_running=[],default_promoted=False,
        supplement_archive105=str(archive),supplement_archive105_sha256=None,
        supplement_scope105='Apply this small delta after the verified104 base source snapshot. RAW/video unchanged. Inside-delta handoff omits self archive SHA.',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(checks/'latest_handoff.json',h)
    files=[status,worklog,note,checks/'latest_handoff.json',report,*OUT.glob('*.json'),*OUT.glob('*.png'),Path(__file__),Path(__file__).with_name('audit105.py')]
    contents={v.relative_to(ROOT.parent).as_posix():v.read_bytes() for v in files}
    manifest={name:hashlib.sha256(data).hexdigest() for name,data in contents.items()}
    info=dict(base_archive=base_archive,base_sha256=base_sha,files=manifest)
    contents['stage105_delta_manifest.json']=json.dumps(info,ensure_ascii=False,indent=2).encode('utf-8')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in contents.items():z.writestr(name,data)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
    h['supplement_archive105_sha256']=sha(archive);save(checks/'latest_handoff.json',h)
    save(OUT/'checks105.json',dict(current_only_ticks=1100,no_future_truth_read=True,no_new_RAW_run=True,
        sources_unchanged=p['sources']==sources(),default_unchanged=True,report=str(report),
        delta_archive=str(archive),delta_sha256=sha(archive),verified_delta_files=len(contents),new_predictor_implemented=False))
    print(str(report),flush=True)


if __name__=='__main__':main()
