"""Verify completed54, retain failed byte-prefix check and record active handoff."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,re,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;OUT=PROJECT/'output'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
log='''

## 64. 完成54：保留完整15模型，单分支与单次RAW解码部署

- 实际复用53原run代码和10ms几何/频率代码，在独立命名空间编译有限锚点变换。run_deploy_forecast.py保留delayed_feedback完整15模型及独立六历史回归，去掉重复的reset/stable两组研究控制。dual仍两RAW读器；shared由父一次解码，将完整结构原生字节、dtype names/formats/offset/itemsize、当前截止和id/ack交给独立10ms进程。原53/52等源码完全不改。
- packets54有界32MiB、不采样不重排、超限明确失败；子进程只读原始共享事件、按10ms截止切分，保留所有原几何/FFT/相位/RLS/组合算术。单次解码worker无RAW路径和RawReader实例，仍独立OpenCV状态。控制验证原生布局/极性/顺序、空包、边界、背压、CRC/未来事件/陈旧id拒绝、超容量无丢包和自有内存回收。
- 冻结后先dual清晰.8s40、shared清晰.8s40、shared空场.2s10，再shared四正式段700、dual完整侧视250、shared独立4.5s前缀225，共1265父/2530原生10ms时刻，1030视频帧全解码。全部当前观测/图像/频率、最终15预测和生命周期精确53。工具74536/81971均正常完成，未重启或覆盖结果。
- 正式shared中位clear175.419ms、side153.746、train3120.714、empty83.130；仍全部>20ms。保留单分支dual侧视142.249ms，shared未稳定降低单步；wall dual82.717→shared75.992s（含冷启动/seek/视频）。不能把53三组182.459与54一组直接算成同范围算法加速，也不能由跨次未控负载宣称固定倍数。
- side完整共享包峰值21,151,920bytes，低于32MiB；发布/校验中位0.765ms。parent20几何38.284ms，wait/transport93.320含worker真实计算，physics10.164ms；瓶颈主要仍是几何/频率，不应只继续微优化IPC。原生频率历史仍450000事件上限的旧抽取策略，传输本身全量无丢弃；二者范围区分。

## 65. 54前缀字节SHA失败与校验指标更正（预测器未修改）

- 原study54.py prefix在108.04s因event_packet_sha256跨运行不同失败，原源/失败记录保留，未改任何数值容差或预测结果。每次传输内父/子完整字节SHA始终通过；初步独立小段clear40前缀SHA相同，但不能据此推断所有填充字节稳定。
- 冻结prefix_padding54补充后，用两个独立RAW Reader同截止读取真实108.02/108.04片段。EventCD原生itemsize16、x/y/p/t offsets0/2/4/8；未使用字节6/7。108.04所有字段与顺序精确，相异字节仅6/7；显式仅改padding可改变整体SHA而不改任何字段。这是跨解码校验指标问题，不是传输或预测修复。原始全程payload未存档，不声称重建当时每个padding字节。
- 最终prefix_padding54.py prefix核对225父/450子完整科学数据、图像、频率、15模型预测、学习状态与事件字节数精确，排除且仅排除跨解码无语义padding SHA；运行内完整SHA要求不放宽。原failed_prefix_byte_sha_v1.json、补充协议、结构诊断、最终raw_prefix_checks全部保存。
- 注意：原study54.py prefix保留了不合适的旧字节要求，不能把该命令重新包装成通过；最终检查入口为prefix_padding54.py prefix。下一轮跨运行用命名字段规范化哈希或语义字段，不以opaque padding为不变量。双端传输仍用完整字节SHA。
- 所有正式预测完成后score，含dual侧视对照225表行全时距/覆盖/联合成功精确53，无新增精度提升。报告/结果图已看，负例零输出，旧默认不升级成精度改进。

## 66. active交接至54；下一步55回到NN引导桨盘提取

- 最新可选部署入口run_deploy_forecast.py，53完整三路对照保留。当前54全计算与诊断已结束，无待跑任务；不标完成长期目标或暂停。没有新依赖、额度重置、外部消息、自动定时任务；不用G，不访问yolov10-main。
- 55尚未启动。优先用已到达的body-only框约束下一请求的当前10ms桨盘几何。复用stage17.frontend.inside，沿用48的25%框余量与160ms期限。父在当前GPU结果之前给worker发送上一有效body框：对TT-.01/TT两个子帧不使用TT的未来NN结果。ROI关闭须精确54；开启时几何输入裁剪、频率历史仍原始事件，所有六历史位置预测与54固定一致，实际未来只评估物理贡献/覆盖/负例。
- 建议先做全局/ROI两个模式真实RAW短对照，冻结参数及独立前缀，不能强行凑四桨或把推测点当频率测量。传输发布本身已<1ms，不应优先再花轮次微优化它。后续可评估将完整20ms几何与worker计算重叠，保持所有更新/特征，不通过删去桨叶物理称实时。
'''
worklog=PROJECT/'WORKLOG.md';text=worklog.read_text(encoding='utf-8')
if '## 66. active交接至54；' not in text:worklog.write_text(text+log,encoding='utf-8')
status=PROJECT/'PROJECT_STATUS_20261021.md';text=status.read_text(encoding='utf-8').replace('当前完成阶段53','当前完成阶段54')
section='''## 最新54：完整部署与单次解码，尚无稳定单步加速

- [run_deploy_forecast.py](</D:/drone forecast/event_drone_forecast/run_deploy_forecast.py>)保留最终完整15模型、逐桨频率/姿态/推力/在线学习及独立历史回归，省去重复研究控制；shared一次解码，将全量原生事件交给独立10ms进程，不采样不丢包，默认32MiB有界缓冲。
- 1265父/2530子时刻及1030视频帧验证完成，当前数据、15模型预测及生命周期精确53。225/450真实RAW语义前缀通过，225行未来评分精确53。无准确率变化。
- 正式shared中位clear175.4ms、side153.7、train3120.7、empty83.1，仍不实时。同范围单分支dual侧视142.2ms，shared单步未胜；wall82.7→76.0s。共享研究模式额外事件SHA、跨次负载不同，不能宣传稳定倍数。
- 前缀原全结构字节SHA在108.04失败；独立读取证明只变化未使用padding6/7，x/y/p/t和顺序相同。原失败/源保持，最终用prefix_padding54.py prefix核对完整语义与模型，传输内字节校验未放宽。原study54.py prefix不作为最终通过入口。
- 发布/校验中位0.765ms，主要瓶颈仍几何/频率。最大包21,151,920bytes，未溢出。下一步55应回到NN框引导当前桨盘提取，暂未启动。

[54结果图与校验说明](</D:/drone forecast/output/stage54_single_decode_deploy_20261021/部署结果图与前缀校验说明.html>) · [完整报告与视频](</D:/drone forecast/output/stage54_single_decode_deploy_20261021/第五十四阶段单次解码完整物理部署报告.html>) · [使用/复核入口](</D:/drone forecast/event_drone_forecast/research/stage54/README.md>)

'''
if '## 最新54：' not in text:text=text.replace('## 最新53：',section+'## 最新53：',1)
start=text.index('## 下一步\n');end=text.index('49–52为历史诊断',start)
text=text[:start]+'''## 下一步

55尚未启动。用过去已到达的NN body-only框约束当前10ms桨盘几何：复用17的inside，既有25%余量/160ms过期，不使用当前20ms未来帧给更早10ms子帧。频率历史仍原始事件，不把预测补点当观测。ROI关闭精确54，再ROI开启测同位置未来误差/空场/前缀，继续解决可靠四点不足。

共享传输中位<1ms，不是主要瓶颈；先改善几何输入，再考虑把完整20ms几何与两个worker重叠。保持全部物理特征和更新，不以删去功能换实时。跨运行不比较结构未使用padding SHA，运行内完整字节校验保留。

'''+text[end:]
text=text.replace('53是最新已验证的实际RAW联合入口','53是完整三路研究对照，54是最新已验证的完整部署入口')
status.write_text(text,encoding='utf-8')
readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8')
text=re.sub(r'2026-10-21继续，已完成53[^\n]+','2026-10-21继续，已完成54完整物理部署。可用[run_deploy_forecast.py](</D:/drone forecast/event_drone_forecast/run_deploy_forecast.py>)保留全部15模型、单次RAW解码与独立桨叶进程；[结果图/校验说明](</D:/drone forecast/output/stage54_single_decode_deploy_20261021/部署结果图与前缀校验说明.html>)、[报告视频](</D:/drone forecast/output/stage54_single_decode_deploy_20261021/第五十四阶段单次解码完整物理部署报告.html>)及[最新交接](</D:/drone forecast/PROJECT_STATUS_20261021.md>)。预测精确53，尚未实时；单解码单步未稳定胜双解码，不能宣传固定加速。53完整研究对照与旧入口均保留。',text,count=1)
readme.write_text(text,encoding='utf-8')
overview=OUT/'research_overview_20261018/index.html';page=overview.read_text(encoding='utf-8')
lead='''<div class="lead"><b>已完成54：完整15模型部署与单次解码，预测保持，仍未实时。</b><p>1265父/2530桨叶时刻当前数据/预测精确53，1030视频帧及独立225/450 RAW语义前缀通过。原前缀字节SHA含无语义padding6/7，失败保留，补充诊断与最终校验公开；传输内字节校验未放宽。</p><p>shared清晰175.4ms、侧视153.7ms、3号120.7ms；同范围dual侧视142.2ms，shared单步未胜，wall82.7→76.0s。传输中位<1ms，主要瓶颈仍几何/频率，下一步55回到NN引导桨盘区域。长期目标active。</p></div>'''
page,n=re.subn(r'<div class="lead">.*?</div>(?=<div class="grid">)',lambda _:lead,page,count=1,flags=re.S);assert n==1
if '../stage54_single_decode_deploy_20261021/' not in page:page=page.replace('<div class="grid">','<div class="grid"><article><h2><a href="../stage54_single_decode_deploy_20261021/部署结果图与前缀校验说明.html">54 · 完整部署、单次解码与实际耗时</a></h2><p>全部物理功能保持，语义前缀/未来评分回归；单步未稳定加速，原padding校验失败与修正保留。</p></article>',1)
overview.write_text(page,encoding='utf-8')

sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage54'));import study54 as current
current.verify();current.previous.verify();current.previous.previous.verify()
folder=current.OUT;counts=dict(parent=0,native10=0,video=0)
for case in current.CASES:
    r=load(folder/case/'regression.json');c=load(folder/case/'completion.json');assert c['source_hashes_unchanged'] and c['history_all_six_exact']
    assert r['complete_final15_forecast_and_lifecycle_exact_to53'] and r['native10_geometry_and_sensing_exact_to53'] and r['event_count_conserved']
    counts['parent']+=r['frames'];counts['native10']+=r['native10_ticks'];counts['video']+=r['video_decoded_frames']
assert counts==dict(parent=1265,native10=2530,video=1030),counts
for name,script in [('prefix_validation_supplement.json','prefix_padding54.py'),('delivery_supplement.json','supplement54.py')]:assert load(folder/name)['helper_sha256']==sha(ROOT/'research/stage54'/script)
prefix=load(folder/'raw_prefix_checks.json');assert prefix['complete_data_forecasts_lifecycle_exact'] and prefix['per_transfer_byte_sha_verified'] and prefix['parent_ticks']==225
padding=load(folder/'structured_padding_diagnostic.json');assert all(r['actual_named_fields_and_order_exact'] for r in padding['records']) and padding['records'][1]['actual_changed_byte_offsets']==[6,7]
assert load(folder/'score_regression.json')['full_table_rows_exact_to53']==225
spec=importlib.util.spec_from_file_location('checker19_for54',ROOT/'tools/checkpoint_20261019.py');checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker);old_save=checker.save
def route(path,x):
    if path.name=='current_delivery_checks_20261019.json':path=path.with_name('current_delivery_checks_stage54_20261021.json')
    old_save(path,x)
checker.save=route;checker.run(False,'active');summary_path=overview.parent/'current_delivery_checks_stage54_20261021.json';summary=load(summary_path)
assert summary['last_completed_stage']==54 and not summary['in_progress']
for p in [overview,*folder.glob('*.html')]:
    for ref in re.findall(r'(?:href|src)="([^"]+)"',p.read_text(encoding='utf-8')):
        if '://' not in ref and not ref.startswith('#'):assert (p.parent/ref).exists(),(p,ref)
summary.update(next_action='Stage55 not started. Prioritize past-body-only-box guided current10ms rotor geometry, reuse17 inside,25% margin/160ms expiry, original raw frequency history. ROI-off exact54; ROI-on same-position true future/empty/prefix. Do not compare opaque cross-decode padding SHA; retain per-transfer byte checksum. Do not claim shared stable per-step speedup.',
    latest_entry=str(ROOT/'core/run_deploy_forecast.py'),verified_counts=counts,prefix_byte_failure_preserved=True,semantic_prefix_verified=True,
    experiments_running=[],current_status_document=str(status),last_goal_turn_classification='progress: complete15-model single-branch/single-decode RAW deployment,1265 actual ticks, true future regression and verified padding-only validation repair completed')
save(summary_path,summary)
files=set(ROOT.glob('*.py'))|set(ROOT.glob('*.md'))|set(PROJECT.glob('PROJECT_STATUS*.md'))|{worklog}
for parent in [ROOT/'src',ROOT/'research',ROOT/'tools']:
    for pattern in ['*.py','*.md']:files.update(p for p in parent.rglob(pattern) if '__pycache__' not in p.parts)
for number in summary['completed']:
    parent=next(p for p in OUT.glob(f'stage{number}_*') if (p/'protocol.json').exists())
    for pattern in ['*.json','*.md','*.html','derived_*.py']:files.update(p for p in parent.rglob(pattern) if p.stat().st_size<2_000_000 and 'failed_border_preparation_v1' not in p.parts)
files.update(overview.parent.glob('*.html'));files.add(summary_path)
archive=OUT/'checkpoints'/('research_through_stage54_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.zip');manifest={p.relative_to(PROJECT).as_posix():sha(p) for p in sorted(files)}
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(files):z.write(p,p.relative_to(PROJECT).as_posix())
    z.writestr('SNAPSHOT_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2));z.writestr('HANDOFF_CHECKPOINT.json',json.dumps(summary,ensure_ascii=False,indent=2))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name,value in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==value
summary.update(archive=str(archive),archive_sha256=sha(archive),source_summary_files=len(files),zip_verified=True,archive_scope='Sources/protocols/small reports only; RAW, weights, media and full JSONL remain in original directories.')
save(OUT/'checkpoints/latest_handoff.json',summary);save(OUT/'checkpoints/latest_progress_20261021.json',summary)
print(json.dumps(dict(stage=54,goal_status='active',counts=counts,files=len(files),archive=str(archive),verified=True),ensure_ascii=False))
