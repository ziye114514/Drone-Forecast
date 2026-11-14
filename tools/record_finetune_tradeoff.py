"""Retain a useful trade-off variant without manufacturing a passing gate."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;OUTPUT=WORK/'output';OUT=OUTPUT/'accumulated_v2_review_20261026'
def read(p):return json.loads(p.read_text('utf-8'))
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
d=read(OUT/'comparison.json');assert d['v1_numerical_replay_exact']
export=read(OUTPUT/'roi_augmented_detector_20261026/export.json')
source=Path(export['model']);target=ROOT/'models/live_position_accumulated_v2_finetune.onnx'
assert hashlib.sha256(source.read_bytes()).hexdigest()==export['sha256']
if not target.exists():shutil.copyfile(source,target)
assert hashlib.sha256(target.read_bytes()).hexdigest()==export['sha256']
manifest_path=ROOT/'configs/live_tracks.json';manifest=read(manifest_path)
recipe=dict(manifest['tracks']['accumulated_v1'])
recipe.update(model=target.relative_to(ROOT).as_posix(),sha256=export['sha256'],
    status='retained_tradeoff_variant_not_recommended_default')
manifest['tracks']['accumulated_v2_finetune']=recipe
assert manifest['recommended']=='accumulated_v1';write(manifest_path,manifest)
adoption=dict(utc=datetime.now(timezone.utc).isoformat(),variant='accumulated_v2_finetune',
    retained_as_selectable_variant=True,recommended_mainline='accumulated_v1',
    automatic_policy_pass=d['accepted_for_next_mainline'],guard_failures_unchanged=d['guard_failures'],
    reason='Development fixed0.4s macro nearly tied; test3 improves. Preserve useful tradeoff without calling it no benefit or claiming overall superiority.',
    source_model_sha256=export['sha256'],new_training_runs=0,new_weight_searches=0)
write(OUT/'retention_decision.json',adoption)
stub=OUT/'dry_run_only.raw';stub.write_bytes(b'');checks={}
try:
    for name,extra in [('default',[]),('v1',['--live-revision','accumulated_v1']),('v2',['--live-revision','accumulated_v2_finetune']),('original',['--profile','live-baseline'])]:
        args=[sys.executable,'-X','utf8',str(ROOT/'core/forecast_mainline.py'),'run','--config',str(ROOT/'configs/local.json'),
            '--raw',str(stub),'--output',str(OUT/(name+'_not_created')),'--dry-run']+extra
        r=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,encoding='utf-8');assert r.returncode==0,r.stderr
        j=json.loads(r.stdout);argv=j['argv'];assert not j['raw_events_read'] and not (OUT/(name+'_not_created')).exists()
        checks[name]=dict(recipe=argv[argv.index('--recipe-id')+1],sha256=argv[argv.index('--detector-sha256')+1])
    assert checks['default']==checks['v1']
    assert checks['v2']['sha256']==export['sha256'] and checks['original']['sha256'].startswith('e2531556')
finally:stub.unlink()
write(OUT/'revision_entry_checks.json',checks)
table=''
for case in ['side','train2']:
    s=d['results'][case]['scores']['0.4'];table+=f'<tr><td>{case}</td><td>{s["v1"]["n"]}</td><td>{s["v1"]["mean"]:.2f}</td><td>{s["v2"]["mean"]:.2f}</td></tr>'
s=d['results']['test3']['scores']['variable_0317_0451'];table+=f'<tr><td>test3，变时域</td><td>{s["v1"]["n"]}</td><td>{s["v1"]["mean"]:.2f}</td><td>{s["v2"]["mean"]:.2f}</td></tr>'
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>累计版的小收益复审</title><style>body{{max-width:1000px;margin:35px auto;padding:0 20px;font:17px/1.8 system-ui;color:#203047}}table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;border:1px solid #c9d3e0}}aside{{padding:18px;background:#edf3fa}}code{{background:#eef2f6}}</style><h1>有用的场景取舍，保留为v2变体</h1><aside>当前推荐主线仍是已经累计的v1。完整微调模型不再被简单归类为“没有收益”，现已保留成可选择的v2。原版基线也保留独立入口。</aside><p>没有重新训练或搜索权重。复用此前相同画面的当前检测，只补了侧视缓存缺失的110.44秒一帧；另在录像3的201个相同原生画面上运行既有微调模型。比较保持v1实际发布时间和目标时刻，属于离线候选运动对照，不是新的实时测速。</p><table><tr><th>片段</th><th>共同目标</th><th>累计v1均值px</th><th>v2变体均值px</th></tr>{table}</table><p>固定0.4秒的两个开发片段宏平均：{d['development_macro04s']['v1']:.3f}→{d['development_macro04s']['v2']:.3f}像素，差约0.01，基本持平。留出录像3为0.317—0.451秒的13个旧弱标注目标，改善约2.10像素；不与固定0.4秒混池。v2没有丢失v1的有效输出，侧视额外多一次有效候选，尚不当作召回提升证明。</p><p>侧视增加2.77像素，另一段减少2.75像素，说明不能把每段各自最好成绩直接累加。原有自动判据未全部通过，记录保留为false；没有为了让候选通过而改变门槛。保留v2是为了让这个有取舍的方案可使用，后续仍从当前累计主线继续改。</p><h2>切换方式</h2><p>默认：<code>forecast_mainline.py run ...</code>。显式v2：增加 <code>--live-revision accumulated_v2_finetune</code>。回到v1：增加 <code>--live-revision accumulated_v1</code>。原版：增加 <code>--profile live-baseline</code>。选中版本和实际模型SHA会记录进输出。</p><p><a href="comparison.json">完整比较与原判据</a> · <a href="retention_decision.json">保留决定</a> · <a href="revision_entry_checks.json">版本选择检查</a> · <a href="../accumulated_mainline_20261026/index.html">当前主线与此前累计增量</a></p></html>'''
(OUT/'index.html').write_text(page,encoding='utf-8')
doc=ROOT/'docs/ACCUMULATED_MAINLINE.md';text=doc.read_text('utf-8')
if 'accumulated_v2_finetune' not in text:
    text+='\n\n## 可保留的场景取舍版本\n\n默认仍为累计v1。增加 --live-revision accumulated_v2_finetune 可选完整微调+尺度版本；--live-revision accumulated_v1 可固定v1。v2在同一目标上side .4s51.368→54.134、train2 85.181→82.434，两段宏平均68.274→68.284基本持平；test3变时域61.770→59.669。保留为可使用变体，不把它叫没有收益，也不宣称总分已更好。原判据失败记录不改，不重新训练或扫alpha。\n'
    doc.write_text(text,encoding='utf-8')
handoff=OUTPUT/'checkpoints/latest_handoff.json';h=read(handoff)
h.update(utc=adoption['utc'],execution_status='active goal; cumulative v1 mainline with selectable v2 tradeoff',
    latest_report=str(OUT/'index.html'),experiments_running=False,current_running_jobs=[],
    latest_live_revision='accumulated_v1',available_tradeoff_revision='accumulated_v2_finetune',
    next_action='Continue from accumulated_v1; v2 full fine-tune now selectable, not discarded. No retraining/alpha sweep. Next research/reuse temporal localization consistency, accounting for real motion rather than smoothing it away. Preserve baseline_v0 and small-gain ledger. Check quota each batch, pause at10%.')
write(handoff,h)
with (WORK/'WORKLOG.md').open('a',encoding='utf-8') as f:
    f.write('\n\n### 复审既有完整微调：保留有用取舍（2026-10-26）\n\n'+
        '前一目标turn为progress（累计主线入口、模型注册、RAW接线落地）。本轮从实际handoff和91%短窗口余量恢复，复审既有完整微调+同尺度组合，无新训练、无权重搜索。录像3补算201个精确当前原生画面，侧视缓存仅缺110.44秒补1次检测；缺帧不插值。所有画面SHA匹配，v1数值回放完全一致，标签仅最后评分。\n\n'+
        '固定0.4s side v1 51.368→v2 54.134；train2 85.181→82.434；均等宏平均68.274→68.284，几乎持平。test3既有13个.317—.451s目标61.770→59.669，改善2.101px。无丢失有效输出，侧视另多1次候选，不称已证明召回。原自动判据false保留（side均值增2.766略超2.568工程线，开发宏平均未降），不改门槛凑通过；保留为有取舍的v2可选版本，默认仍是已累计v1而不是原版v0。\n\n'+
        '新增--live-revision选版本，模型按SHA复制models/live_position_accumulated_v2_finetune.onnx。默认、v1、v2、原版四入口dry-run通过，不启动RAW/GPU。报告output/accumulated_v2_review_20261026/index.html。两次离线RAW探针会话77289/93692均exit0；当前无本轮后台任务，goal保持active。下一步时序定位一致性，不重复alpha/阈值搜索。\n')
print(json.dumps(dict(recommended=manifest['recommended'],retained_variant=adoption['variant'],automatic_gate=d['accepted_for_next_mainline'],entries=checks),ensure_ascii=False))
