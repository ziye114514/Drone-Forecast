"""Retain a small aggregate gain as an explicit version, with scene tradeoffs."""
from pathlib import Path
import copy,hashlib,html,json,shutil,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
OUT=WORK/'output/flip_consistency_20261026'
assert not (OUT/'retention_decision.json').exists()
report=json.loads((OUT/'comparison.json').read_text('utf-8'))
policy=json.loads((ROOT/'configs/accumulation_policy.json').read_text('utf-8'))
failures=[]
for case in ['side','train2']:
    r=report['results'][case]
    if r['lost_valid_publications']['consistency']:failures.append(case+':lost_valid_publications')
    for horizon,s in r['scores'].items():
        for field,rule in [('mean','mean_regression_allowance'),('p95','p95_regression_allowance')]:
            limit=policy['accuracy'][rule];base=s['mainline_v1'][field]
            allowance=max(limit['absolute_px'],limit['relative_fraction']*base)
            if s['consistency'][field]>base+allowance:
                failures.append(f'{case}:{horizon}:{field}:regression={s["consistency"][field]-base:.6f}>allowance={allowance:.6f}')
if report['development_macro04s']['consistency']>=report['development_macro04s']['mainline_v1']:
    failures.append('development_macro04s_not_better')
export=json.loads((OUT/'consistency/export.json').read_text('utf-8'))
source=Path(export['model']);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(source)==export['sha256']
destination=ROOT/'models/live_position_accumulated_v3_consistency.onnx'
if destination.exists():assert sha(destination)==export['sha256']
else:shutil.copyfile(source,destination)
manifest_path=ROOT/'configs/live_tracks.json';manifest=json.loads(manifest_path.read_text('utf-8'))
assert manifest['recommended']=='accumulated_v1'
recipe=copy.deepcopy(manifest['tracks']['accumulated_v1'])
recipe.update(model='models/'+destination.name,sha256=export['sha256'],
    status='retained_small_aggregate_gain_with_side_regression',
    evidence=str(OUT/'comparison.json'),parent='accumulated_v1',
    note='Five-epoch same-frame flip consistency; single ordinary detector. Keep original and v1; automatic side guard failure disclosed.')
manifest['tracks']['accumulated_v3_consistency']=recipe
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
decision=dict(variant='accumulated_v3_consistency',retained_as_selectable_variant=True,
    recommended_mainline='accumulated_v1',automatic_policy_pass=not failures,guard_failures_unchanged=failures,
    aggregate04s_gain_px=report['development_macro04s']['mainline_v1']-report['development_macro04s']['consistency'],
    matched_control_gain_px=report['development_macro04s']['control']-report['development_macro04s']['consistency'],
    no_lost_scored_outputs=all(not report['results'][c]['lost_valid_publications']['consistency'] for c in ['side','train2']),
    reason='Retain a small positive aggregate gain and useful scene tradeoff; side guard fails, so keep current cumulative v1 as recommended.',
    deployment_extra_branches=0,raw_runtime_smoke='pending',not_a_statistically_proven_general_improvement=True)
stub=OUT/'dry_run_only.raw';stub.write_bytes(b'');checks={}
try:
    for name,extra in [('default',[]),('v3',['--live-revision','accumulated_v3_consistency']),('original',['--profile','live-baseline'])]:
        result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'forecast_mainline.py'),'run','--config',str(ROOT/'configs/local.json'),
            '--raw',str(stub),'--output',str(OUT/(name+'_not_created')),'--dry-run']+extra,cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        assert result.returncode==0,result.stderr
        data=json.loads(result.stdout);assert not data['raw_events_read'];argv=data['argv']
        def value(flag):return argv[argv.index(flag)+1] if flag in argv else None
        checks[name]=dict(recipe=value('--recipe-id'),sha256=value('--detector-sha256'))
        assert not (OUT/(name+'_not_created')).exists()
    assert checks['default']['recipe']=='accumulated_v1'
    assert checks['v3']['sha256']==export['sha256'] and checks['v3']['recipe']=='accumulated_v3_consistency'
    assert checks['original']['sha256']==manifest['tracks']['baseline_v0']['sha256']
finally:stub.unlink()
(OUT/'revision_entry_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
colors=['#377eb8','#888888','#d95f02'];names=['Cumulative v1','Paired control','Flip consistency']
fig,axes=plt.subplots(1,3,figsize=(11,3.7))
for ax,case,label in zip(axes,['side','train2','test3'],['Side: fixed 0.4 s','Train2: fixed 0.4 s','Test3: variable 0.317–0.451 s']):
    horizon='variable_0317_0451' if case=='test3' else '0.4';scores=report['results'][case]['scores'][horizon]
    values=[scores[n]['mean'] for n in ['mainline_v1','control','consistency']]
    ax.bar(range(3),values,color=colors);ax.set_xticks(range(3),['v1','Control','Consistency']);ax.set_title(label,fontsize=10)
    ax.set_ylabel('Mean position error (px)');ax.set_ylim(0,max(values)*1.18)
    for k,v in enumerate(values):ax.text(k,v+1,f'{v:.2f}',ha='center',fontsize=9)
    ax.text(.02,.96,'n = '+str(scores['mainline_v1']['n']),transform=ax.transAxes,va='top',fontsize=9)
fig.tight_layout();fig.savefig(OUT/'forecast_comparison.png',dpi=160);plt.close(fig)
overlay_links=[]
import cv2
runs=dict(side=WORK/'output/single_model_optimized_RAW_20261026/side',train2=WORK/'output/single_model_optimized_RAW_20261026/train2',test3=WORK/'output/heldout_motion_RAW_20261026')
for case,run in runs.items():
    original={round(r['t'],6):r for r in map(json.loads,(run/'tracking.jsonl').read_text('utf-8').splitlines())}
    probes={round(r['t'],6):r for r in json.loads((OUT/'native_probe'/(case+'.json')).read_text('utf-8'))}
    paths=sorted((OUT/'native_probe').glob(case+'_*.png'))
    for path in paths:
        t=round(float(path.stem.split('_',1)[1]),6);frame=cv2.imread(str(path))
        bodies=[original[t]['body'],probes[t]['control'],probes[t]['consistency']]
        for index,(body,color) in enumerate(zip(bodies,[(184,126,55),(150,150,150),(2,95,217)])):
            if body.get('detected'):
                b=np.rint(body['bbox']).astype(int);cv2.rectangle(frame,tuple(b[:2]),tuple(b[2:]),color,2)
                c=np.rint(body['center']).astype(int);cv2.drawMarker(frame,tuple(c),color,cv2.MARKER_CROSS,14,2)
        footer=np.full((90,1280,3),245,np.uint8)
        cv2.putText(footer,f'{case} t={t:.3f}s | current native 20ms ON/OFF; no future overlay',(14,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(15,15,15),2)
        for j,(name,color) in enumerate(zip(names,[(184,126,55),(150,150,150),(2,95,217)])):
            cv2.putText(footer,name,(14+420*j,65),cv2.FONT_HERSHEY_SIMPLEX,.7,color,2)
        dest=OUT/(path.stem+'_comparison.png');cv2.imwrite(str(dest),np.concatenate((frame,footer),axis=0))
        overlay_links.append(f'<a href="{dest.name}">{case} {t:.3f}s 当前检测对比</a>')

def table(case,horizon):
    s=report['results'][case]['scores'][horizon];rows=''
    for n,label in zip(['mainline_v1','control','consistency'],names):
        r=s[n];rows+=f'<tr><td>{label}</td><td>{r["n"]}</td><td>{r["mean"]:.3f}</td><td>{r["median"]:.3f}</td><td>{r["p95"]:.3f}</td></tr>'
    return '<table><tr><th>版本</th><th>目标数</th><th>平均 px</th><th>中位 px</th><th>P95 px</th></tr>'+rows+'</table>'
transfers={n:json.loads((OUT/n/'transfer_v2/scores.json').read_text('utf-8')) for n in ['control','consistency']}
det=transfers['consistency']['table'];base=det[0];candidate=det[1]
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>同帧一致性训练与累计预测版本</title>
<style>body{{font:17px/1.65 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;color:#243040}}h1,h2{{line-height:1.3}}table{{border-collapse:collapse;margin:18px 0;width:100%}}td,th{{padding:9px;border:1px solid #d2dce6;text-align:left}}th{{background:#eef3f8}}img{{max-width:100%}}code{{word-break:break-all}}.note{{background:#fff4d9;padding:15px}}</style>
<h1>同一时刻的翻转一致性：保留小收益和场景取舍</h1>
<p>从累计v1继续训练；相同五轮、相同配对增强、相同监督与BN，唯一区别为有无定位一致性约束。两个模型均完成训练和ONNX导出，没有新增部署分支。不是新网络，也不声称复现原论文或统计显著提升。</p>
<p class="note">一致性版固定0.4s开发段等权平均 {report['development_macro04s']['mainline_v1']:.3f} → {report['development_macro04s']['consistency']:.3f}px，省 {decision['aggregate04s_gain_px']:.3f}px；相对相同训练量对照省 {decision['matched_control_gain_px']:.3f}px。侧视均值增加3.296px，train2减少3.590px。保留为 <b>accumulated_v3_consistency</b>，当前默认仍累计v1。自动工程判据失败如实保留，未放宽门槛凑通过。</p>
<img src="forecast_comparison.png" alt="三版本未来位置误差；第三段的预测时域不同，单独展示">
<h2>原生RAW上的完整预测结果</h2><p>402张当前原生画面哈希全部匹配。固定既有v1接收时刻、发布时刻及未来目标，替换感知结果再跑同一个预测器；v1回放数值完全一致。未将缺检插值为检测。没有丢失既有有效发布；侧视多一次候选，尚不能称为已证明召回收益。这是离线完整预测回放，不是新一轮端到端测速结论。</p>
<h3>侧视：固定0.4s</h3>{table('side','0.4')}<h3>train2：固定0.4s</h3>{table('train2','0.4')}
<h3>录像3：13个0.317–0.451s目标，单独统计</h3>{table('test3','variable_0317_0451')}
<p>录像3没有进入训练。标签与片段曾在历史实验中使用，因此不能把这次称为全局盲测，也不能把这13个变时域目标混入固定0.4s均值。样本小，0.147px开发总收益可能受标注噪声影响；保留版本用于继续迭代，不承诺稳定精度提升。</p>
<h2>检测端变化与方法边界</h2><p>独立录像3的107个可见弱标注图像均检出；中心平均误差 {base['mean_center_error_px']:.3f} → {candidate['mean_center_error_px']:.3f}px，中位IoU {base['median_IoU']:.3f} → {candidate['median_IoU']:.3f}。同录像11张空图误报2 → 1，录像5另7张空图均无误报。这些小样本不代表完整检测召回率。</p>
<p>先查 <a href="https://github.com/tkyen1110/TEDNet">TEDNet</a>：其一致性关联位移和可见性头，并不等同减少框抖动。再参考 <a href="https://github.com/soo89/CSD-SSD">CSD官方项目</a> 和 <a href="https://papers.nips.cc/paper_files/paper/2019/hash/d0f4dae80c3d0277922f8371d5827292-Abstract.html">NeurIPS 2019论文</a>，自行适配已安装YOLO的翻转定位一致性。约束同一时刻的位置，不约束相邻时刻必须平滑，因此允许真实变向。背景与不匹配目标不参与一致性；没有使用灰度重建、未来图像、真实推力或虚构桨心。</p>
<h2>当前画面对比</h2><p>蓝=v1，灰=相同训练量对照，橙=一致性版。下方文字独立加边，不覆盖无人机。</p><p>{'<br>'.join(overlay_links)}</p>
<h2>使用与下一步</h2><p>默认入口继续选累计v1。可显式增加 <code>--live-revision accumulated_v3_consistency</code> 选择本轮版本，原版 <code>--profile live-baseline</code> 保留。接线短RAW状态见 <a href="runtime_status.json">runtime_status.json</a>。</p>
<p>目前更清楚的卡点是：单帧定位稍好仍不能保证未来轨迹更好。下一步优先引入有质量控制的局部位移观测，直接补充当前速度，而不是继续多轮调检测损失或随意缩小平滑窗口。先查现有SDK/开源实现，仍保留现有位置通道作为基线。</p>
<p>证据：<a href="design.json">固定实验设计</a> · <a href="mechanism_checks.json">几何与反向检查</a> · <a href="comparison.json">完整预测数据</a> · <a href="retention_decision.json">保留决定与失败项</a> · <a href="native_probe/summary.json">RAW画面核对</a> · <a href="revision_entry_checks.json">版本入口</a></p></html>'''
(OUT/'index.html').write_text(page,encoding='utf-8')
(OUT/'runtime_status.json').write_text(json.dumps(dict(status='pending',scope='A short entry smoke will check wiring, not establish new forecast accuracy.'),indent=2),encoding='utf-8')
(OUT/'retention_decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
with (ROOT/'docs/ACCUMULATED_MAINLINE.md').open('a',encoding='utf-8') as f:
    f.write('\n\n## 可选v3：同帧翻转一致性\n\n`--live-revision accumulated_v3_consistency` 选择单模型同帧一致性版本。开发固定0.4s宏平均68.274→68.127px；侧视51.368→54.664，train2 85.181→81.591；留出13个变时域目标61.770→59.461，分开统计。相同训练量无一致性对照宏平均69.613。保留小收益版本，默认仍累计v1，侧视工程判据未通过不隐藏。不是新网络或原论文复现。\n')
print(json.dumps(dict(decision=decision,entry_checks=checks),ensure_ascii=False),flush=True)
