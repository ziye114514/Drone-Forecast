"""Document/verify completed actual RAW integration53 and save active handoff."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,re,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;OUT=PROJECT/'output'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

log='''

## 62. 完成53：真正RAW-only整机+桨叶+物理联合入口

- 新run_joint_forecast.py和stage53/rotor_worker53.py，复用47已验证共享内存GPU32、21 native20观测、33 ReplaceStale、51固定参考/原控制、52延迟反馈、28/27轴向RLS与29延迟选择器。实际NN整机位置、逐桨频率/轴向、有效推力进入同一入口，不读旧缓存/manifest/标签；固定本地权重是模型输入，无人工质量/大小/风速/RPM。原入口/默认不动。
- 20ms和10ms两独立RAW读器隔离OpenCV数值状态；父/子严格当前id/cutoff，任一事件t<截止。GPU只读原生BGR共享内存，双SHA核验，worker只处理当前请求；自己的子进程/共享内存结束回收。两reader增加IO/冷seek开销，属于已知工程成本。全长RAW尾部至多一个空10ms补桶逻辑已写入，但本轮仅验证显式有限片段，不称覆盖全部EOF边界。
- 先pilot_clear .8s40/80时刻和pilot_empty .2s10/20通过，再clear2s100/200、side5s250/500、train3 6s300/600、empty1s50/100、独立side4.5s225/450。工具90656、13606、36477全部正常结束，无待跑任务。
- 合计975整机/1950桨叶时刻；975 native20几何/像素、975 NN框（bbox最大差0）、三路各975完整15专家预测、六历史都精确旧45/51/52。正例1830 native10几何/逐桨频率/换槽审计精确旧34；负例没有34参考，零当前/有效预报。视频740帧全解码，9固定25/50/75%画面已看，包含无检测时刻，不挑成功画面。
- 真正独立RAW前缀225整机/450桨叶的完整观测、像素、频率、全部预测与学习状态精确。所有正式预测完成后按旧22实际未来标签评分，540表行（全部时距/覆盖/联合成功/无机统计）逐项精确旧51/52。整合保留修复效果，没有新增准确率提升。
- side实际58? 更正：clear58、side10、train3 0参数更新。side同5/6目标0.4秒物理122.4597px、trusted81.9130，clear14.2240、train3物理136.6933/trusted110.3190，仍缺稳定物理优势。current active=False时物理模型回退CV，视频紫线不能当持续可靠姿态/推力。
- 实际处理中位clear225.479ms、side182.459、train3135.714、empty85.373，全部>20ms，P95/双worker分项及wall见runtime_summary。含三路完整研究对照；排除父下一RAW块/JSON/视频，wall含双RAW/冷启动/视频。不与47仅位置31ms混算加速，也不称实时。代码/协议/报告/实际预测视频完整保留。

## 63. active交接至53；下一步保持完整功能再降开销/改善桨盘

- 最新完整RAW入口改为run_joint_forecast.py；47为只位置/历史，30为旧完整物理基线，全部保留。状态、README、导航同步53；尚未完成持续改进目标，不标complete/paused，不调用额度重置或外部消息/自动任务，不用G、不访问yolov10-main。
- 下一步54尚未启动：先让部署只运行最终delayed_feedback完整15模型（保留六历史/全部物理特征，省去重复的两组实验控制），验证与53最终分支精确；再考虑仅一次RAW解码，通过有界共享事件包传给独立10ms worker，保留OpenCV隔离/截止时钟和原生dtype/顺序，减少重复IO/seek。不能以删去桨叶物理换取“实时”。
- 后续NN引导桨盘提取优先：将过去已到达的body-only框（不能用该20ms截止未来帧给较早10ms子帧）随下一请求传给桨叶worker，用既有25%框余量、160ms过期限制当前原生ON/OFF事件区域，历史频率输入仍原始事件。先ROI关对照精确53，再ROI开同位置未来评分/空场/前缀，继续解决50四点初始化不足。此为待办，不是已实现或已证明增益。
'''
# Fix the wording before persisting; counts are sourced from completion files.
log=log.replace('side实际58? 更正：clear58、side10、train3 0参数更新。','真实参数更新clear58、side10、train3 0。')
worklog=PROJECT/'WORKLOG.md';text=worklog.read_text(encoding='utf-8')
if '## 63. active交接至53；' not in text:worklog.write_text(text+log,encoding='utf-8')
status=PROJECT/'PROJECT_STATUS_20261021.md';previous_status=PROJECT/'PROJECT_STATUS_20261020.md';text=(status if status.exists() else previous_status).read_text(encoding='utf-8').replace('# 当前继续状态：2026-10-20','# 当前继续状态：2026-10-21').replace('当前完成阶段52','当前完成阶段53')
section='''## 最新53：实际RAW联合入口已验证

- [run_joint_forecast.py](</D:/drone forecast/event_drone_forecast/run_joint_forecast.py>)接通原生20ms整机NN、独立10ms桨叶频率、姿态轴、有效推力/RLS与延迟选择。RAW加固定本地模型，无旧检测/频率缓存或人工标签输入；两个解码器用于保持数值隔离，当前仍有重复IO成本。
- 正式14秒700整机/1400桨叶，另有两个短试跑与独立前缀。总975整机像素/几何/框、三路各975完整预测精确旧控制，正例1830原生10ms几何/频率精确34；视频740帧全解码。
- 独立真实RAW前缀225整机/450桨叶完整状态精确；540行全部未来评分精确51/52，保留了修复但没有额外准确率提升。clear58/side10/train3 0次更新；空场零当前/预报。
- 单步处理clear225.5ms、side182.5、train3135.7、empty85.4，仍未实时；含三路完整研究控制，wall还含冷启动/双解码/视频。不能以视频50fps冒充处理吞吐。紫色物理分支在active=False时回退CV。

[53结果图/固定时刻画面](</D:/drone forecast/output/stage53_raw_joint_physics_20261020/联合RAW预测结果总览.html>) · [完整预测视频与命令](</D:/drone forecast/output/stage53_raw_joint_physics_20261020/第五十三阶段真实RAW联合预测报告.html>) · [使用与局限](</D:/drone forecast/event_drone_forecast/research/stage53/README.md>)

'''
if '## 最新53：' not in text:text=text.replace('## 最新50–52：',section+'## 最新50–52：',1)
start=text.index('## 下一步\n');end=text.index('当前49只是诊断',start)
text=text[:start]+'''## 下一步

54尚未启动。部署保持完整物理与全部15模型，只省去重复的两组实验控制；再评估单次RAW解码+有界共享原生事件包，保留10/20ms独立几何数值状态/当前截止/原生事件顺序。先严格回归53最终分支与RAW前缀，再谈降耗时。

接着用过去已到达的NN整机框限定当前桨盘检索区域，既有25%余量/160ms过期，不把20ms未来帧送给更早10ms子帧，禁止把预测补点当观测。ROI关要精确旧控制，ROI开才测同位置真实未来及空场，继续解决可靠四点不足。

'''+text[end:]
text=text.replace('当前49只是诊断，不是新预测器或已证明未来精度提升。47最新RAW入口仍`run_shared_body_forecast_v2.py`，完整几何74.52ms、仅位置FP16 31.05ms，未达到20ms预算；完整物理RAW路径仍30，48新位置物理消融未显示稳定额外收益。','49–52为历史诊断/缓存消融。47位置入口与30旧物理入口保留；53是最新已验证的实际RAW联合入口，尚未实时或显示稳定物理预测优势。')
status.write_text(text,encoding='utf-8')
readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8')
text=re.sub(r'2026-10-20继续，已完成52[^\n]+','2026-10-21继续，已完成53实际RAW联合预测。最新完整入口[run_joint_forecast.py](</D:/drone forecast/event_drone_forecast/run_joint_forecast.py>)；[预测视频与报告](</D:/drone forecast/output/stage53_raw_joint_physics_20261020/第五十三阶段真实RAW联合预测报告.html>)、[结果图](</D:/drone forecast/output/stage53_raw_joint_physics_20261020/联合RAW预测结果总览.html>)和[最新交接](</D:/drone forecast/PROJECT_STATUS_20261021.md>)。整机/逐桨频率/轴向推力/延迟物理学习已在RAW入口执行，独立RAW前缀通过；精度保持52、耗时仍高，未达实时。',text,count=1)
text=text.replace('最新可运行位置入口是','保留的位置/历史专用入口是')
text=text.replace('完整桨叶物理RAW路径仍为','旧完整桨叶物理RAW基线为')
text=text.replace('48为缓存同状态消融，新整机位置尚未与完整物理路径合并为已验证更好的在线系统。','48为缓存消融；53已把新整机位置和修复后的完整物理路径接到实际RAW并通过回归，但尚未显示稳定优于纯历史的精度。')
readme.write_text(text,encoding='utf-8')
overview=OUT/'research_overview_20261018/index.html';page=overview.read_text(encoding='utf-8')
lead='''<div class="lead"><b>已完成53：真实RAW整机、逐桨频率与物理学习已联合运行。</b><p>正式700整机/1400桨叶时刻，另有试跑及独立前缀。实际推理不读旧缓存/标签；完整观测/预测回归、225整机/450桨叶真实RAW前缀、540行未来评分回归均通过。侧视学习10次，精度保持52，没有新增准确率收益。</p><p>处理单步中位清晰225.5ms、侧视182.5ms、3号135.7ms，含三路完整实验对照，尚未实时；紫线在active=False时回退CV。下一步保留完整功能减少重复计算/解码，再做NN引导桨盘提取。长期目标active。</p></div>'''
page,n=re.subn(r'<div class="lead">.*?</div>(?=<div class="grid">)',lambda _:lead,page,count=1,flags=re.S);assert n==1
if '../stage53_raw_joint_physics_20261020/' not in page:page=page.replace('<div class="grid">','<div class="grid"><article><h2><a href="../stage53_raw_joint_physics_20261020/联合RAW预测结果总览.html">53 · 真实RAW联合入口、结果图与视频</a></h2><p>实际整机/逐桨频率/物理学习，700正式时刻和独立RAW前缀通过；精度保持，尚未实时。</p></article>',1)
page=page.replace('../../PROJECT_STATUS_20261020.md','../../PROJECT_STATUS_20261021.md')
overview.write_text(page,encoding='utf-8')

sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage53'));import study53 as current
current.verify();current.previous.verify()  # Includes immutable old sensing/position/control inputs.
folder=current.OUT;summary_counts=dict(parent=0,native10=0,videos=0,positive_native10_exact=0)
for case,cfg in current.CASES.items():
    r=load(folder/case/'regression.json');c=load(folder/case/'completion.json');assert c['source_hashes_unchanged'] and c['history_all_six_exact']
    assert r['body_observation_exact_to45']==r['frames'] and r['maximum_body_bbox_difference_px']==0.
    assert all(v==r['frames'] for v in r['full_forecast_exact_counts'].values())
    summary_counts['parent']+=r['frames'];summary_counts['native10']+=r['native10_ticks'];summary_counts['videos']+=r['video_decoded_frames']
    if cfg['geometry']:assert r['native10_geometry_sensing_exact_to34'];summary_counts['positive_native10_exact']+=r['native10_ticks']
assert summary_counts==dict(parent=975,native10=1950,videos=740,positive_native10_exact=1830),summary_counts
assert load(folder/'raw_prefix_checks.json')['complete_forecasts_lifecycle_observations_pixels_exact']
assert load(folder/'score_regression.json')['all_future_table_rows_exact_to51_52']==540
assert load(folder/'delivery_supplement.json')['helper_sha256']==sha(ROOT/'research/stage53/supplement53.py')
spec=importlib.util.spec_from_file_location('checker19_for53',ROOT/'tools/checkpoint_20261019.py');checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker);original_save=checker.save
def route(p,x):
    if p.name=='current_delivery_checks_20261019.json':p=p.with_name('current_delivery_checks_stage53_20261021.json')
    original_save(p,x)
checker.save=route;checker.run(False,'active')
summary_path=overview.parent/'current_delivery_checks_stage53_20261021.json';summary=load(summary_path);assert summary['last_completed_stage']==53 and not summary['in_progress']
for p in [overview,*folder.glob('*.html')]:
    for ref in re.findall(r'(?:href|src)="([^"]+)"',p.read_text(encoding='utf-8')):
        if '://' not in ref and not ref.startswith('#'):assert (p.parent/ref).exists(),(p,ref)
summary.update(next_action='Stage54 not started. Preserve complete final physical branch and15 models; remove duplicate research control groups then evaluate one RAW decode/shared bounded event packets with independent10ms geometry. Require exact53 final-branch and actual RAW prefix. Afterwards NN-past-box guided rotor extraction, not future20ms hints for earlier10ms ticks.',
    actual_raw_integration53=True,verified_counts=summary_counts,current_status_document=str(status),experiments_running=[],
    last_goal_turn_classification='progress: actual RAW-only body/rotor/physical integration, independent true RAW prefix and future scoring completed; no stable physical accuracy advantage yet')
save(summary_path,summary)
files=set(ROOT.glob('*.py'))|set(ROOT.glob('*.md'))|set(PROJECT.glob('PROJECT_STATUS*.md'))|{worklog}
for p in [ROOT/'src',ROOT/'research',ROOT/'tools']:
    for pattern in ['*.py','*.md']:files.update(q for q in p.rglob(pattern) if '__pycache__' not in q.parts)
for number in summary['completed']:
    p=next(q for q in OUT.glob(f'stage{number}_*') if (q/'protocol.json').exists())
    for pattern in ['*.json','*.md','*.html','derived_*.py']:files.update(q for q in p.rglob(pattern) if q.stat().st_size<2_000_000 and 'failed_border_preparation_v1' not in q.parts)
files.update(overview.parent.glob('*.html'));files.add(summary_path)
archive=OUT/'checkpoints'/('research_through_stage53_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.zip');manifest={p.relative_to(PROJECT).as_posix():sha(p) for p in sorted(files)}
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(files):z.write(p,p.relative_to(PROJECT).as_posix())
    z.writestr('SNAPSHOT_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2));z.writestr('HANDOFF_CHECKPOINT.json',json.dumps(summary,ensure_ascii=False,indent=2))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name,value in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==value
summary.update(archive=str(archive),archive_sha256=sha(archive),source_summary_files=len(files),zip_verified=True,archive_scope='Source/protocol/small reports only; RAW, weights, video/images and full JSONL retained in original workspace.')
save(OUT/'checkpoints/latest_handoff.json',summary);save(OUT/'checkpoints/latest_progress_20261021.json',summary)
print(json.dumps(dict(stage=53,goal_status='active',counts=summary_counts,archive=str(archive),files=len(files),verified=True),ensure_ascii=False))
