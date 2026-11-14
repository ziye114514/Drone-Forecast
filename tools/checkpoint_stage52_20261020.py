"""Document completed50-52 and verify/archive current research state."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,re,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;OUT=PROJECT/'output'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

log='''

## 58. 完成50：实际复用旧机架，真实关联没有改善

- 先复核作者EVPeriscope2609.11920、IPPE、事件特征概率关联。EVPeriscope已知不对称机架/PnP+EKF且实际要求四桨可见；项目页访问失败，不能确认现成作者部署代码。项目已有4仿射骨架、5共圆联合拟合，故50实际复用4 initialization/fit_affine/X和33 ReplaceStale，不另造同类模块。
- 新layout50仅当前四观测、当前/过去NN框及形状/变形门槛合格时联合关联，不生成桨心/频率。已知身份仿真突发共同位移：旧33身份错误172次、新0；这有合成独立中心输入，不是实采身份准确率。无body严格回归33，缺输入过期。
- 原34同1300 native10当前输入：clear200、side500、train3 600。原33几何/ID逐项精确34，独立前缀120时刻各case完整结果精确。清晰52时刻联合关联、1次初始化；侧视/3号均0初始化，全部实际输出与原33相同，所以没有新预测收益，也不为相同关联重复跑RAW频率。
- posthoc initialization_diagnostic：clear候选60、非平行四边形92；side缺四452、大小不一致9、闭合失败25、近共线10，仅4候选、最长streak2<原5帧；train3缺四593、无候选。严格机架依然起不来，不靠放松门槛声称解决。50报告保留负结果，未升级默认。

## 59. 完成51：固定参考与状态保留，不足以恢复真实学习

- reference51/checks51/study51冻结。复用原28/27物理+29选择器，在48同700时刻比较reset_all、preserve_axis、stable_reference。原reset_all完整15专家精确回归48 associated_slotfix；三路六历史系数/有效性精确45。
- sum f²置于固定首个S0坐标，等价于k/协方差/队列参考变换，但不改变原数值约束及激励判据。实际原RLS/遗忘/队列/rollout不变性控制通过，500ms整机失锁仍全部重置，越出原0.2–5域弃权。实际700无越域拒绝。
- 保留状态后side20 epoch、train3 10 epoch仍0更新；clear58。side0.4s频率物理124.36→123.76px，仍远差trusted81.91；train3无收益。故19日把清零/预热当主要解释不充分，还需看反馈门控。原共享频率准入、四可见、0.6s预热均不变，未调阈值。

## 60. 完成52：缺桨时消费历史合格特征，侧视恢复10次更新

- 直接检查51：side只有110.06/110.10及112.20–112.38少数特征合格，0.2s后位置反馈到达时又因当前缺四/质量低被禁止更新，历史样本过0.26s失效。train3合格launch时刻为空，保留参考本身不能救。
- feedback52从原ProjectedThrustModel.update导出可审计函数，仅改两个调度判断：历史样本反馈要求当前有位置，不再要求当前也有四桨；新样本入队仍要求原四可见/质量/轴向门槛。矩阵、损失、边界、遗忘、激励、warmup和过期完全保留。当前预测点不充当独立观测；反馈仍为NN真实检测支持的CV状态，有噪声/滤波误差。
- 全观测控制参数/协方差/扰动逐项精确旧模型；受控缺桨但有位置原1→新11更新，位置也缺失仍1，队列空后不重复更新。独立派生函数哈希、源文件冻结；没有修改旧paper_physics源码。
- 同700时刻预测先存，再按真实稀疏未来框评分。baseline完整15专家精确51stable，六历史系数/有效性精确45。真实side更新0→10，clear58不变、train3仍0；空场50无有效预报。
- side同5/6目标，0.4s物理123.7619→122.4597px、joint1不变；纯历史trusted81.9130/joint3仍更好。0.2s99.4022→98.9490，all_mix0.4s93.6265→93.5946，收益小。clear和train3不变，不宣称稳定物理优势或创新。
- supplement52独立4.5s当前缓存前缀225完整forecast/lifecycle精确，包含实际学习发生时刻；不是新RAW解码。图已查看，补充图与前缀验证.html可看；原报告和所有0.1/.2/.4评分保留。工具68578/20592都正常结束，当前无运行实验。

## 61. 当前active，完成52后交接

- 新状态文档、README、实验导航和源码/协议快照更新至52。未完成长期目标，不标complete/paused；没有自动定时任务、新线程、新依赖、外部消息或额度重置。最近只读额度：5h15%、周2%、ordinaryAllowed true，重置credit仍1未使用。
- 当前最新真实RAW整机入口仍47 v2，完整物理RAW仍30；50–52是冻结缓存消融，没有新的完整RAW部署/实时性结论。下一步应把已验证的固定参考+反馈调度接入真实RAW整机/桨叶联合入口，再核对旧分支、真实前缀、空场和同目标未来误差；随后继续NN框内桨盘提取/共同机架关联，不把更松四桨门槛当解决。
- 49几何启发有价值，但50说明可靠四桨/初始化仍是瓶颈。53尚未启动。旧模型/结果/数据、失败保留，不用G，不访问yolov10-main。
'''
worklog=PROJECT/'WORKLOG.md';text=worklog.read_text(encoding='utf-8')
if '## 61. 当前active，完成52后交接' not in text:worklog.write_text(text+log,encoding='utf-8')
status=PROJECT/'PROJECT_STATUS_20261020.md';text=status.read_text(encoding='utf-8').replace('当前完成阶段49','当前完成阶段52')
section='''## 最新50–52：关联仍卡四点，修复了延迟学习反馈

- 50实际复用4仿射机架和33转子关联。1300时刻基线几何/ID回归、各case120前缀通过；真实新旧关联完全相同。清晰可启用，侧视只有4个初始化合格时刻、连续最多2帧，达不到原5帧确认；3号无合格机架。没有提高默认，也未为相同输出重复计算RAW频率。
- 51固定频率参考坐标并保留轴向/物理状态，原RLS代数/协方差/队列/rollout不变性通过，原48完整15专家700时刻回归。side/train3参数更新仍0，说明预热/清零不是唯一问题。
- 52将历史特征入队和位置反馈更新解耦：新入队仍须原四桨与质量门槛，已有历史样本可在当前只看到整机位置时更新。只改原update两个调度判断，原矩阵/约束/遗忘/预热/过期不变；原51完整15专家700时刻回归，六历史与45精确。
- side学习0→10次，clear58、train3 0；side同5/6目标0.4s频率物理123.76→122.46px，仍差于纯历史81.91px，联合成功未变。清晰及3号无改善。不能把学习恢复称为预测明显提高。
- 新模型独立当前缓存4.5秒前缀225完整预测/学习状态精确，全观测及缺位置控制通过，负例50无有效预报。52为缓存因果消融，尚未合并成新RAW入口或实时系统。

[52图与前缀验证](</D:/drone forecast/output/stage52_delayed_position_feedback_20261020/补充图与前缀验证.html>) · [51报告](</D:/drone forecast/output/stage51_preserved_physics_reference_20261020/第五十一阶段保留物理学习状态报告.html>) · [50报告](</D:/drone forecast/output/stage50_rigid_slot_association_20261020/第五十阶段机架辅助桨心关联.html>)

'''
if '## 最新50–52：' not in text:text=text.replace('用户新想法：',section+'用户新想法：',1)
start=text.index('## 下一步\n');end=text.index('当前49只是诊断',start)
text=text[:start]+'''## 下一步

53尚未启动。先把固定频率参考与已验证的延迟反馈接入真实RAW整机/桨叶联合入口，核对旧分支、独立真实RAW前缀、空场及同目标未来误差；不能只停在缓存。接着用已改进的NN位置限定桨盘提取区域，继续修复可靠四桨/共同机架初始化。真实轴心、身份和侧视条件数依然关键，禁止把预测补点当观测或真值。

'''+text[end:]
status.write_text(text,encoding='utf-8')
readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8')
text=re.sub(r'2026-10-20已按最新继续任务恢复[^\n]+', '2026-10-20继续，已完成52。见[最新交接](</D:/drone forecast/PROJECT_STATUS_20261020.md>)和[52结果图](</D:/drone forecast/output/stage52_delayed_position_feedback_20261020/补充图与前缀验证.html>)。机架关联在侧视仍难初始化；延迟反馈修复让侧视学习0→10次，0.4秒物理误差123.76→122.46px，仍差于纯历史81.91px。50–52为缓存消融，最新RAW入口仍47 v2/完整物理30，53尚未启动。',text,count=1)
readme.write_text(text,encoding='utf-8')
overview=OUT/'research_overview_20261018/index.html';page=overview.read_text(encoding='utf-8')
lead='''<div class="lead"><b>已完成52：侧视在线学习恢复，预测收益仍小，长期目标active。</b><p>50复用机架关联在真实数据未改变输出，侧视仅4个合格机架时刻、最长连续2帧。51保留学习状态仍0更新；52分开历史桨叶特征入队和当前位置反馈后，侧视恢复10次更新。</p><p>同5/6目标0.4秒物理误差123.76→122.46px，纯历史收缩81.91px仍更好。全观测回归、700时刻旧专家回归、独立225时刻当前缓存前缀通过；不宣称稳定物理优势。新RAW整合53尚未启动，最新RAW入口仍47 v2/完整物理30。</p></div>'''
page,n=re.subn(r'<div class="lead">.*?</div>(?=<div class="grid">)',lambda _:lead,page,count=1,flags=re.S);assert n==1
cards='''<article><h2><a href="../stage52_delayed_position_feedback_20261020/补充图与前缀验证.html">52 · 历史桨叶与当前整机位置反馈</a></h2><p>侧视更新0→10；物理0.4s误差123.76→122.46px，仍未胜历史；结果图与前缀验证。</p></article><article><h2><a href="../stage51_preserved_physics_reference_20261020/第五十一阶段保留物理学习状态报告.html">51 · 固定参考与物理状态保留</a></h2><p>等价坐标/RLS检查通过；保留状态本身仍不足以恢复侧视学习。</p></article><article><h2><a href="../stage50_rigid_slot_association_20261020/第五十阶段机架辅助桨心关联.html">50 · 共同机架辅助关联</a></h2><p>1300时刻同输入，真实输出未改善；侧视四点不足，原阈值保留。</p></article>'''
if '../stage52_delayed_position_feedback_20261020/' not in page:page=page.replace('<div class="grid">','<div class="grid">'+cards,1)
overview.write_text(page,encoding='utf-8')

# Verify current experiment inputs as well as all old frozen source protocols.
for number in [50,51,52]:
    folder=ROOT/f'research/stage{number}';sys.path.insert(0,str(folder));spec=importlib.util.spec_from_file_location(f'checkpoint_verify{number}',folder/f'study{number}.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.verify()
prefix=OUT/'stage52_delayed_position_feedback_20261020';p=load(prefix/'prefix_supplement.json');assert p['helper_sha256']==sha(ROOT/'research/stage52/supplement52.py')
assert load(prefix/'prefix_checks.json')['independent_cached_current_prefix_records']==225
for c in ['negative5']:
    for stage in ['stage51_preserved_physics_reference_20261020','stage52_delayed_position_feedback_20261020']:
        rr=[json.loads(x) for x in (OUT/stage/c/'forecasts.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(rr)==50 and not any(q['valid'] for r in rr for q in r['frontends'].values())
spec=importlib.util.spec_from_file_location('checker19_for52',ROOT/'tools/checkpoint_20261019.py');checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
original_save=checker.save
def route(path,x):
    if path.name=='current_delivery_checks_20261019.json':path=path.with_name('current_delivery_checks_stage52_20261020.json')
    original_save(path,x)
checker.save=route;checker.run(False,'active')
summary_path=overview.parent/'current_delivery_checks_stage52_20261020.json';summary=load(summary_path);assert summary['last_completed_stage']==52 and not summary['in_progress']
for html_path in [overview,*prefix.glob('*.html')]:
    for ref in re.findall(r'(?:href|src)="([^"]+)"',html_path.read_text(encoding='utf-8')):
        if '://' not in ref and not ref.startswith('#'):assert (html_path.parent/ref).exists(),(html_path,ref)
summary.update(next_action='Integrate fixed-reference/delayed-feedback into actual RAW joint body+rotor entry; verify original control, RAW prefix, empty scene and real sparse future targets. Then continue NN-guided rotor extraction and rigid association. Stage53 not started.',
    current_status_document=str(status),experiments_running=[],last_goal_turn_classification='progress: completed50 association negative control,51 coordinate retention controls and52 actual delayed-feedback fix with700 real cached current forecasts/true future scoring',
    stage50_51_52_input_hashes_unchanged=True,stage52_prefix_225_exact=True,negative_cases_no_valid_forecasts=True)
save(summary_path,summary)
files=set(ROOT.glob('*.py'))|set(ROOT.glob('*.md'))|set(PROJECT.glob('PROJECT_STATUS*.md'))|{worklog}
for folder in [ROOT/'src',ROOT/'research',ROOT/'tools']:
    for pattern in ['*.py','*.md']:files.update(p for p in folder.rglob(pattern) if '__pycache__' not in p.parts)
for number in summary['completed']:
    folder=next(p for p in OUT.glob(f'stage{number}_*') if (p/'protocol.json').exists())
    for pattern in ['*.json','*.md','*.html','derived_*.py']:
        files.update(p for p in folder.rglob(pattern) if p.stat().st_size<2_000_000 and 'failed_border_preparation_v1' not in p.parts)
files.update(overview.parent.glob('*.html'));files.add(summary_path)
archive=OUT/'checkpoints'/('research_through_stage52_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.zip')
manifest={p.relative_to(PROJECT).as_posix():sha(p) for p in sorted(files)}
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(files):z.write(p,p.relative_to(PROJECT).as_posix())
    z.writestr('SNAPSHOT_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2));z.writestr('HANDOFF_CHECKPOINT.json',json.dumps(summary,ensure_ascii=False,indent=2))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name,value in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==value
summary.update(archive=str(archive),archive_sha256=sha(archive),source_summary_files=len(files),zip_verified=True,archive_scope='Sources, protocols and small summaries only; RAW, weights, media and full JSONL remain in original directories.')
save(OUT/'checkpoints/latest_handoff.json',summary);save(OUT/'checkpoints/latest_progress_20261020.json',summary)
print(json.dumps(dict(stage=52,goal_status='active',archive=str(archive),files=len(files),verified=True),ensure_ascii=False))
