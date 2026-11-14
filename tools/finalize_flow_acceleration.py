"""Close actual image-acceleration evidence and promote the cumulative stack."""
from pathlib import Path
import argparse,ast,datetime,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from live_scale_rollout import ScaleRolloutMotion,scale_predictions
from live_local_flow import velocity_coefficient
from live_flow_acceleration import FlowAcceleration
from summarize_live_pilot import read_rows
parser=argparse.ArgumentParser();parser.add_argument('--short-remaining',type=int,required=True);parser.add_argument('--weekly-remaining',type=int,required=True);args=parser.parse_args()
OUT=WORK/'output/flow_acceleration_20261026';assert not (OUT/'final_decision.json').exists()
policy=json.loads((ROOT/'configs/accumulation_policy.json').read_text('utf-8'));actual={};failures=[]
for name in ['actual_side','actual_train2','actual_side_headless']:
    run=OUT/name;completion=json.loads((run/'completion.json').read_text('utf-8'));protocol=json.loads((run/'protocol.json').read_text('utf-8'))
    assert completion['sources_unchanged'] and completion['frame_descriptor_mismatches']==0
    assert completion['flow_acceleration']=='position-consistent' and completion['recipe_id']=='accumulated_v7_flow_acceleration'
    assert not completion['labels_read'] and not completion['output_caches_read']
    motion=ScaleRolloutMotion(25.);acceleration=FlowAcceleration();states={};ticks=0
    for row in read_rows(run/'tracking.jsonl'):
        t=row['t'];c,valid,_=motion.step(t,row['body']);assert c==row['position_only_coefficient'] and valid==row['valid']
        flow=row['diagnostic']['local_flow']['observation'];velocity,_=velocity_coefficient(c,motion.kf.P[2:4,2:4] if c else None,flow,t)
        assert velocity==row['velocity_only_coefficient']
        revised,info=acceleration.step(t,velocity,row['body'],flow,'position-consistent');assert revised==row['coefficient']
        assert info==row['diagnostic']['flow_acceleration']
        if revised is not None:assert revised['position']==velocity['position'] and revised['velocity']==velocity['velocity']
        states[round(t,6)]=(velocity,revised);ticks+=1
    for p in read_rows(run/'predictions.jsonl'):
        if p['observation_t'] is None:continue
        velocity,revised=states[round(p['observation_t'],6)]
        for key,c in [('velocity_only',velocity),('live_motion',revised)]:
            assert scale_predictions(c,p['observation_age_ms']/1000)[0]==p['predictions'][key]
    scores=json.loads((OUT/(name+'_scores.json')).read_text('utf-8'))
    if name in ['actual_side_headless','actual_train2']:
        for h,table in scores['tables'].items():
            a,b=table['metrics']['velocity_only'],table['metrics']['live_motion']
            for field,key in [('mean','mean_regression_allowance'),('p95','p95_regression_allowance')]:
                limit=policy['accuracy'][key]
                if b[field]>a[field]+max(limit['absolute_px'],limit['relative_fraction']*a[field]):failures.append(f'{name}:{h}:{field}')
    actual[name]=dict(completion=completion,scores=scores,replay_exact_ticks=ticks,video_recorded=(run/'predictions.mp4').exists())
assert not failures,failures
assert actual['actual_side_headless']['completion']['valid_publications']>=18
assert actual['actual_train2']['completion']['valid_publications']>=6
replay=json.loads((OUT/'comparison.json').read_text('utf-8'));combo=json.loads((OUT/'overlap_combination.json').read_text('utf-8'));fast=json.loads((OUT/'fit_fastpath/comparison.json').read_text('utf-8'))
assert replay['development_macro04s']['position_consistent']<replay['development_macro04s']['v5']
path=ROOT/'configs/live_tracks.json';manifest=json.loads(path.read_text('utf-8'));assert manifest['recommended']=='accumulated_v5_fast_ci'
manifest['recommended']='accumulated_v7_flow_acceleration'
manifest['tracks']['accumulated_v7_flow_acceleration'].update(status='recommended_after_causal_replay_and_actual_RAW',evidence=str(OUT/'final_decision.json'))
for name in ['baseline_v0','accumulated_v5_fast_ci','accumulated_v7_flow_acceleration']:
    e=manifest['tracks'][name];assert hashlib.sha256((ROOT/e['model']).read_bytes()).hexdigest()==e['sha256']
path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
s=actual['actual_side_headless']['scores']['tables']['0.4']['metrics'];a,b=s['velocity_only'],s['live_motion']
decision=dict(recommended='accumulated_v7_flow_acceleration',parent='accumulated_v5_fast_ci',includes_support_overlap=True,
    default_promoted=True,actual_guard_failures=failures,current_position_velocity_and_validity_rules_preserved=True,
    actual_headless_side_04s=dict(baseline=a,new=b),actual_train2_no_unsupported_acceleration=True,
    headless_default_continuity_checked=True,first_video_run_with_extra_expiration_retained=True,
    no_wind_mass_thrust_or_metric_acceleration_claim=True,globally_blind=False)
(OUT/'final_decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
headless_rows=[r for r in read_rows(OUT/'actual_side_headless/tracking.jsonl') if r['coefficient'] is not None]
fig,axes=plt.subplots(2,1,figsize=(10,6.5))
t=[r['t'] for r in headless_rows];old_a=np.array([r['velocity_only_coefficient']['acceleration'] for r in headless_rows]);new_a=np.array([r['coefficient']['acceleration'] for r in headless_rows])
axes[0].plot(t,np.linalg.norm(old_a,axis=1),'--',label='Position-history acceleration');axes[0].plot(t,np.linalg.norm(new_a,axis=1),label='Qualified flow derivative')
axes[0].set_ylabel('Estimated |a| (px/s²)');axes[0].set_title('Image acceleration estimates; no independent acceleration ground truth');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
samples=[p for p in actual['actual_side_headless']['scores']['pairs'] if p['horizon_s']==.4]
for key,label in [('velocity_only','Existing velocity branch'),('live_motion','New acceleration branch')]:
    axes[1].plot([p['publication_t'] for p in samples],[p['errors_px'][key] for p in samples],'.-',label=label)
axes[1].set_xlabel('Actual publication time (s)');axes[1].set_ylabel('0.4s endpoint error (px)');axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'acceleration_and_actual_errors.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(11,3.6))
for ax,case,title in zip(axes,['side','train2','test3'],['Side: fixed0.4s','Train2: fixed0.4s','Test3: variable0.317–0.451s']):
    h='variable_0317_0451' if case=='test3' else '0.4';table=replay['results'][case]['scores'][h]
    values=[table[n]['mean'] for n in ['v5','position_consistent']];ax.bar([0,1],values,color=['#377eb8','#e17c05']);ax.set_xticks([0,1],['v5','New a']);ax.set_ylim(0,max(values)*1.2);ax.set_title(title,fontsize=10);ax.set_ylabel('Mean endpoint error (px)')
    for i,v in enumerate(values):ax.text(i,v+1,f'{v:.2f}',ha='center')
fig.tight_layout();fig.savefig(OUT/'fixed_replay_comparison.png',dpi=160);plt.close(fig)
train=actual['actual_train2'];tc=train['completion'];ts=train['scores']['tables']['0.4']['metrics'];head=actual['actual_side_headless']['completion'];video=actual['actual_side']['completion']
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>累计v7：速度变化估计加速度</title>
<style>body{{font:17px/1.65 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;color:#243040}}h1,h2{{line-height:1.3}}img,video{{max-width:100%}}table{{border-collapse:collapse;width:100%;margin:16px 0}}td,th{{border:1px solid #d4dee8;padding:9px}}.note{{background:#eaf4e9;padding:15px}}code{{word-break:break-all}}</style>
<h1>累计v7：从光流速度变化补充加速度</h1><p class="note">已切到默认 <b>accumulated_v7_flow_acceleration</b>，包括此前CI加速、支持图/GPU重叠和本轮加速度分支。默认无视频输出的真实侧视RAW中，同起点0.4s平均 {a['mean']:.2f} → {b['mean']:.2f}px，P95 {a['p95']:.2f} → {b['p95']:.2f}px。原版和所有累计版本继续可选。</p>
<h2>预测逻辑的实际变化</h2><p>保留原始光流在两帧中点的速度，使用过去0.35s内至少6个观测、至少0.16s跨度，按速度协方差白化后做Huber线性拟合，一阶导数得到图像加速度。沿用现有拟合常数、收缩方式与3000px/s²上限，没有扫窗口、阈值或权重。再用过去的整机框中心检查这一加速度是否与整体运动一致；信号不够或不一致时精确回退已有预测。</p>
<p>保持当前位置、当前速度和有效性规则，只更新有支持的预测加速度。复用<a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html">SciPy Huber最小二乘</a>；没有用居中滤波或未来速度。协方差只是工程代理，没有同步IMU/飞控真值，因此不能声称真实加速度、推力、质量或风已被识别。</p>
<h2>同一次实际运行、同一未来目标</h2><table><tr><th>片段/模式</th><th>目标数</th><th>平均 px</th><th>中位 px</th><th>P95 px</th><th>有效发布</th></tr>
<tr><td>侧视，默认无视频</td><td>{a['n']}</td><td>{a['mean']:.2f} → {b['mean']:.2f}</td><td>{a['median']:.2f} → {b['median']:.2f}</td><td>{a['p95']:.2f} → {b['p95']:.2f}</td><td>{head['valid_publications']}/{head['publication_slots']}</td></tr>
<tr><td>train2，带视频</td><td>{ts['live_motion']['n']}</td><td>{ts['velocity_only']['mean']:.2f} → {ts['live_motion']['mean']:.2f}</td><td>{ts['velocity_only']['median']:.2f} → {ts['live_motion']['median']:.2f}</td><td>{ts['velocity_only']['p95']:.2f} → {ts['live_motion']['p95']:.2f}</td><td>{tc['valid_publications']}/{tc['publication_slots']}</td></tr></table>
<p>基线是同次运行保存的“仅速度修正”分支。train2没有足够的导数信号，新增加速度0次，输出保持。侧视平均和尾误差改善，但中位误差增加 {b['median']-a['median']:.2f}px，部分时刻更差；不能称为全面提升。不同负载运行的绝对分数不可直接排名。样本小、标签曾在历史实验中使用，不是全局盲测或统计显著性证明。</p>
<img src="acceleration_and_actual_errors.png" alt="当前图像加速度估计及同起点0.4秒误差；无独立加速度真值">
<h2>冻结缓存与完整组合</h2><img src="fixed_replay_comparison.png" alt="固定观测的侧视、train2和独立变时域目标对比"><p>固定两段0.4s等权平均74.560 → 72.486px；录像3另13个0.317–0.451s目标57.224 → 49.899px，单独统计。只用速度导数的录像3尾误差反而增加到130.043px，加入位置一致性后降为115.721px；该资格检查有实际作用。</p>
<p>此前串行/重叠的已发布快照，在共同起点完整重放：原串行56.944px，新加速度串行53.045px，新加速度加重叠52.764px；不是把独立收益相加。这是旧记录回放，下面实际入口另作验证。</p>
<h2>计算与连续性：保留第一次不理想运行</h2><p>第一次带视频的侧视运行只有 {video['valid_publications']}/{video['publication_slots']}次有效，处理 {video['processing_median_ms']:.2f}ms、观测年龄P95 {video['observation_age_p95_ms']:.2f}ms，额外过期记录保留。不能把这次结果藏掉。</p>
<p>随后仅对Huber已在二次区域且满足原gtol的情形直接返回同一WLS解；386个时刻的系数/审计逐项一致，非线性调用214→30，观察器组件中位0.443→0.285ms。默认无视频模式再跑一次：{head['tracking_updates']}跟踪、{head['valid_publications']}/{head['publication_slots']}有效，处理 {head['processing_median_ms']:.2f}ms、加速度观察器 {head['flow_acceleration_median_ms']:.3f}ms、年龄P95 {head['observation_age_p95_ms']:.2f}ms。该次与视频模式同时改变输出负载，不能把全部耗时差归因于求解器加速。</p>
<p>所有实际状态与发布逐步精确回放，原生/共享/GPU画面检查通过，无标签和输出缓存进入推理。短回放不保证长期实时运行。</p>
<h2>预测预览</h2><p>视频来自第一次带视频运行，展示真实输出，测速和默认模式结论见上表。</p><video controls preload="metadata" src="actual_side/predictions.mp4"></video><video controls preload="metadata" src="actual_train2/predictions.mp4"></video>
<p>默认入口无需加版本参数；原版 <code>--profile live-baseline</code>，旧累计 <code>--live-revision accumulated_v5_fast_ci</code>。<code>--flow-acceleration none</code> 可关闭新增分支做对照，<code>--local-flow none</code> 同时关闭光流相关分支。</p>
<p>证据：<a href="final_decision.json">累计决定</a> · <a href="comparison.json">冻结完整结果</a> · <a href="overlap_combination.json">完整组合</a> · <a href="actual_side_headless_scores.json">默认模式评分</a> · <a href="actual_side_scores.json">第一次视频评分</a> · <a href="actual_train2_scores.json">train2评分</a> · <a href="fit_fastpath/comparison.json">等价加速</a> · <a href="mechanism_checks.json">已知运动与不一致检查</a></p></html>'''
(OUT/'index.html').write_text(page,encoding='utf-8')
files=['live_flow_acceleration.py','live_local_flow.py','run_live_forecast.py','forecast.py','configs/live_tracks.json','tools/check_flow_acceleration.py','tools/score_flow_acceleration.py','tools/score_acceleration_combinations.py','tools/register_flow_acceleration.py','tools/probe_acceleration_fastpath.py','tools/finalize_flow_acceleration.py']
for name in files:
    source=ROOT/name;data=source.read_bytes()
    if source.suffix=='.py':ast.parse(data)
    target=OUT/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
hpath=WORK/'output/checkpoints/latest_handoff.json';h=json.loads(hpath.read_text('utf-8'))
h.update(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution_status='active goal; v7 flow acceleration plus overlap recommended',latest_report=str(OUT/'index.html'),
    latest_live_revision='accumulated_v7_flow_acceleration',current_jobs=[],current_goal_turn_classification='progress',quota_remaining=dict(short=args.short_remaining,weekly=args.weekly_remaining),
    next_action='Continue from v7; accuracy gain now implemented, do not redo all validation. Raw flow midpoint velocity derivative with fixed Huber/position consistency is integrated. One potential next step: cache covariance whitening per accepted flow sample, exact values, to avoid repeated Cholesky/inverse across nine-history fit. Or assess joint current-velocity midpoint compensation only with proper uncertainty, no parameter sweeps. Current outputs preserve p/v. Keep video run17/20 failure and default headless18/20. All jobs terminal. Check quota before short batches; save and pause at10%.')
hpath.write_text(json.dumps(h,ensure_ascii=False,indent=2),encoding='utf-8')
(WORK/'output/latest_live_forecast.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=flow_acceleration_20261026/index.html"><a href="flow_acceleration_20261026/index.html">最新累计优化报告</a>',encoding='utf-8')
with (ROOT/'docs/ACCUMULATED_MAINLINE.md').open('a',encoding='utf-8') as f:
    f.write('\n\n## 当前默认v7：光流速度导数补充加速度\n\n包括v6重叠准备。固定.35s/6点/.16s，用原始midpoint v及协方差做Huber导数，再检验过去框中心轨迹。只改有支持的预测a，p/v/有效性不变。默认无视频侧视同起点.4s均60.144→55.229、P95111.698→92.478，但中位53.357→58.624。train2信号不足保持。视频初次17/20有效失败保留；默认无视频18/20。不是推力或真实加速度估计。\n')
with (WORK/'WORKLOG.md').open('a',encoding='utf-8') as f:
    f.write(f'\n\n### 累计v7：速度导数加速度落地（2026-10-26）\n\n前一turn为progress（支持图/GPU重叠可选v6）。本轮读现有robust.py及SciPy Huber/多项式导数，采用因果、不规则中点raw flow v一阶导数；沿用.35s/6点/.16s/.08gap、Huber8.07/6、cap3000及原收缩，协方差白化，位置历史6px/90%一致性。已知a1200/-800合成误差6.260px/s²，恒速不假造a，静止位置/局部运动不一致被拒绝，p/v不变。\n\n缓存395当前样本：side.4s60.873→56.725、train2 88.247保持，宏平均74.560→72.486；独立13变时域57.224→49.899，尾123.921→115.721。位置资格减少测试3更新140→121，避免未资格版本尾130.043；不冒充全局盲测或同步真实a真值。既有两实际模式在共同起点组合回放：v5串行56.944，新a串行53.045，新a+重叠52.764，故完整v7包括重叠而非独立增益相加。\n\n真实带视频side86tracking/17有效（额外过期），.4s同次velocity-only73.115→58.351；train2 99tracking/6有效、新a0次，96.740保持。保留首次连续性失败。Huber在二次区且原gtol已满足时直接用同一WLS解，386状态/审计完全一致，非线性214→30，CPU观察器中位.443→.285ms。默认无视频side再验：95tracking、20pub/18valid、53a更新，processing37.648ms、a观察器.705ms、ageP95121.422；同目标16个.4s60.143591→55.228641、P95111.698312→92.478189，但中位53.357341→58.624310变差，完整披露。\n\n三次实际状态/发布逐步精确，原p/v保留、有效性规则不改，源/图检查通过，无标签缓存进推理；所有真实物理未知量仍未知。会话4018(side视频)、45124(train2视频)、71652(side无视频)均exit0，无当前作业。默认升级accumulated_v7_flow_acceleration，旧版本保留。报告output/flow_acceleration_20261026/index.html；quota余短{args.short_remaining}%/周{args.weekly_remaining}%，goal active。下一批优先复用每条v观测的whitener作精确CPU优化；联合当前速度补偿需独立审慎实验，不扫参数。\n')
print(json.dumps(decision,ensure_ascii=False),flush=True)
