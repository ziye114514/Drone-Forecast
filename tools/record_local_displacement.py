"""Promote the measured cumulative flow stack and preserve all earlier trials."""
from pathlib import Path
import argparse,ast,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from live_scale_rollout import ScaleRolloutMotion
from live_local_flow import velocity_coefficient
from summarize_live_pilot import read_rows
parser=argparse.ArgumentParser();parser.add_argument('--short-remaining',type=int,required=True);parser.add_argument('--weekly-remaining',type=int,required=True);args=parser.parse_args()
OUT=WORK/'output/local_displacement_tight_20261026';assert not (OUT/'final_decision.json').exists()
replay=json.loads((OUT/'comparison.json').read_text('utf-8'));policy=json.loads((ROOT/'configs/accumulation_policy.json').read_text('utf-8'))
actual={};failures=[]
for case in ['side','train2']:
    run=OUT/('actual_'+case);completion=json.loads((run/'completion.json').read_text('utf-8'))
    assert completion['sources_unchanged'] and completion['frame_descriptor_mismatches']==0
    assert completion['recipe_id']=='accumulated_v4_local_flow' and completion['local_flow']=='klt'
    assert not completion['labels_read'] and not completion['output_caches_read'] and not completion['center_ensemble_enabled']
    model=ScaleRolloutMotion(25.)
    for row in read_rows(run/'tracking.jsonl'):
        c,valid,_=model.step(row['t'],row['body']);assert c==row['position_only_coefficient'] and valid==row['valid']
        if c is not None:
            observation=row['diagnostic']['local_flow']['observation']
            revised,_=velocity_coefficient(c,model.kf.P[2:4,2:4],observation,row['t']);assert revised==row['coefficient']
    scores=json.loads((OUT/('actual_'+case+'_scores.json')).read_text('utf-8'))
    old=json.loads((WORK/'output/single_model_optimized_RAW_20261026'/case/'completion.json').read_text('utf-8'))
    if completion['valid_publications']<old['valid_publications']:failures.append(case+':fewer_valid_publications_than_historical_v1')
    for h,table in scores['tables'].items():
        a,b=table['metrics']['position_only'],table['metrics']['live_motion']
        for field,key in [('mean','mean_regression_allowance'),('p95','p95_regression_allowance')]:
            limit=policy['accuracy'][key]
            if b[field]>a[field]+max(limit['absolute_px'],limit['relative_fraction']*a[field]):failures.append(f'{case}:{h}:{field}')
    actual[case]=dict(completion=completion,scores=scores,baseline_replay_exact=True)
macro={n:sum(actual[c]['scores']['tables']['0.4']['metrics'][n]['mean'] for c in ['side','train2'])/2 for n in ['position_only','live_motion']}
if macro['live_motion']>=macro['position_only']:failures.append('no_actual_paired_macro_gain')
if replay['development_macro04s']['klt']>=replay['development_macro04s']['mainline_v1']:failures.append('no_fixed_replay_macro_gain')
assert not failures,failures
path=ROOT/'configs/live_tracks.json';manifest=json.loads(path.read_text('utf-8'));assert manifest['recommended']=='accumulated_v1'
candidate=manifest['tracks']['accumulated_v4_local_flow'];candidate.update(status='recommended_after_fixed_replay_and_actual_paired_RAW',
    evidence=str(OUT/'final_decision.json'),note='112px aligned crop; velocity-only CI rollout correction. Original/v1 remain separately selectable. No extra neural detector.')
manifest['recommended']='accumulated_v4_local_flow'
for n in ['baseline_v0','accumulated_v1','accumulated_v4_local_flow']:
    e=manifest['tracks'][n];assert hashlib.sha256((ROOT/e['model']).read_bytes()).hexdigest()==e['sha256']
path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
decision=dict(recommended='accumulated_v4_local_flow',parent='accumulated_v1',default_promoted=True,
    actual_paired_macro04s=macro,actual_macro_gain_px=macro['position_only']-macro['live_motion'],
    fixed_replay_macro04s=replay['development_macro04s'],guard_failures=failures,
    historical_valid_publications_not_reduced=True,position_state_baseline_replay_exact=True,
    weights_unchanged=True,SDK_sparse_flow_not_deployed=True,not_a_globally_blind_or_statistical_significance_claim=True)
(OUT/'final_decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(11,3.6))
for ax,case,label in zip(axes,['side','train2','test3'],['Side: fixed 0.4s','Train2: fixed 0.4s','Test3: variable 0.317–0.451s']):
    horizon='variable_0317_0451' if case=='test3' else '0.4';s=replay['results'][case]['scores'][horizon]
    values=[s[n]['mean'] for n in ['mainline_v1','klt']];ax.bar([0,1],values,color=['#377eb8','#e17c05'])
    ax.set_xticks([0,1],['Cumulative v1','Local flow']);ax.set_title(label,fontsize=10);ax.set_ylabel('Mean endpoint error (px)');ax.set_ylim(0,max(values)*1.2)
    for x,v in enumerate(values):ax.text(x,v+1,f'{v:.2f}',ha='center')
fig.tight_layout();fig.savefig(OUT/'frozen_comparison.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(11,7.2))
for col,case in enumerate(['side','train2']):
    pubs={round(p['t'],9):p for p in read_rows(OUT/('actual_'+case)/'predictions.jsonl')}
    samples=[s for s in actual[case]['scores']['pairs'] if s['horizon_s']==.4]
    truth=np.array([s['truth'] for s in samples]);times=[s['publication_t'] for s in samples]
    xy=axes[0,col];errorplot=axes[1,col]
    xy.plot(truth[:,0],truth[:,1],'o-',color='#2a8b45',label='Future labeled center')
    for n,color,label in [('position_only','#377eb8','Position-only'),('live_motion','#e17c05','Local flow')]:
        points=np.array([pubs[round(s['publication_t'],9)]['predictions'][n]['0.4'] for s in samples])
        xy.plot(points[:,0],points[:,1],'.--',color=color,label=label)
        errorplot.plot(times,[s['errors_px'][n] for s in samples],'.-',color=color,label=label)
    xy.invert_yaxis();xy.set_title(case+' actual matched targets');xy.set_xlabel('x (px)');xy.set_ylabel('y (px)');xy.legend(fontsize=8)
    errorplot.set_xlabel('Actual publication time (s)');errorplot.set_ylabel('0.4s endpoint error (px)');errorplot.legend(fontsize=8);errorplot.grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'actual_predictions.png',dpi=160);plt.close(fig)

rows=''
for case,label in [('side','侧视'),('train2','train2')]:
    c=actual[case]['completion'];s=actual[case]['scores']['tables']['0.4']['metrics'];a,b=s['position_only'],s['live_motion']
    rows+=f'<tr><td>{label}</td><td>{a["n"]}</td><td>{a["mean"]:.2f} → {b["mean"]:.2f}</td><td>{a["p95"]:.2f} → {b["p95"]:.2f}</td><td>{c["valid_publications"]}/{c["publication_slots"]}</td><td>{c["processing_median_ms"]:.1f} ms</td></tr>'
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>局部位移并行速度与累计v4</title>
<style>body{{font:17px/1.65 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;color:#243040}}h1,h2{{line-height:1.3}}table{{border-collapse:collapse;margin:18px 0;width:100%}}td,th{{padding:9px;border:1px solid #d2dce6}}th{{background:#eef3f8}}img,video{{max-width:100%}}.note{{background:#eaf4e9;padding:15px}}code{{word-break:break-all}}</style>
<h1>局部位移补充速度：进入默认累计v4</h1><p class="note">默认入口已切到 <b>accumulated_v4_local_flow</b>。相同实际发布起点上，固定0.4s两段等权平均 {macro['position_only']:.2f} → {macro['live_motion']:.2f}px，减少 {decision['actual_macro_gain_px']:.2f}px。原版、累计v1以及v2/v3仍可单独选择。</p>
<h2>改了什么</h2><p>从纯原生ON/OFF支持图提取特征点，复用OpenCV金字塔LK及前后向检查，再用稳健相似变换求上一整机中心的位移。这样允许整体转动和尺度变化，避免直接把每个局部点的运动当作机体中心平移。通过质量检查后，使用协方差交集补充预测速度；当前KF位置、加速度、检测有效性和几何资格逻辑保持。推力、真实转速、质量与风没有在这里被求出。</p>
<p>仅增加CPU图像处理，不增加神经网络。使用二值事件支持图，不使用G通道灰度重建。同一目标缺检或特征匹配失败时，精确回退当前累计位置预测，没有补造无人机中心。</p>
<h2>实际4秒RAW：同一次运行中的对照</h2><table><tr><th>片段</th><th>0.4s目标数</th><th>平均误差 px</th><th>P95 px</th><th>有效/发布</th><th>总处理中位</th></tr>{rows}</table>
<p>“位置基线”和“局部位移版”使用同一次真实发布时刻、当前观测和未来标签目标；位置基线逐步数值回放完全一致。两段均由RAW重新解码和GPU检测，推理没有读预测缓存或标签。原始帧、GPU帧与共享帧核对通过。侧视P95有增加，未隐藏；工程容忍范围内通过。只测了短片段，不是长期实时、泛化或统计显著性保证。</p>
<img src="actual_predictions.png" alt="相同实际起点上的0.4秒预测端点与误差"><p>绿线是事后标注的未来框中心，只参与评分和这张结果图。不同负载运行的接收时刻会变化，因此不能把下方冻结回放的绝对误差直接当成这次实际运行的分数。</p>
<h2>冻结同一批原生画面：看机制本身</h2><img src="frozen_comparison.png" alt="固定原生画面与发布时刻比较"><p>固定开发两段0.4s宏平均68.274 → 67.402px；侧视51.368 → 51.440，train2 85.181 → 83.364。录像3另13个0.317–0.451s目标61.770 → 57.224px，时域不同，单独统计。旧标签和片段曾被使用，不能称全局盲测。</p>
<h2>计算开销确实做了优化</h2><p>全图LK约28–30ms。先用192px边界裁剪降至13ms，但实际侧视总处理42.99ms、85次跟踪、17/20有效，采样已受影响；该结果保留在<a href="../local_displacement_crop_20261026/actual_side_scores.json">第一次实际评分</a>。根据金字塔窗口支持改为112px边界并按8像素对齐，冻结画面合格时刻保持一致；侧视观察器约11ms，实际总处理33.70ms、94次跟踪、18/20有效。边界优化有微小数值差异，已用真实结果重新评分，没有宣称按位相同。</p>
<h2>没有采用的分支</h2><p>官方SDK稀疏光流虽能输出二维事件簇速度，本数据中冻结开发0.4s宏平均105.406px，显著差于位置基线68.274px；它容易混入桨叶局部运动。保留<a href="../local_displacement_20261026/comparison.json">失败结果</a>，没有接入默认预测。</p>
<h2>预测预览</h2><p>下方为实际入口生成的5Hz发布预览，紫线是当时输出的未来轨迹；输入保持原生时间戳。视频没有事后未来真值。</p><video controls preload="metadata" src="actual_side/predictions.mp4"></video><video controls preload="metadata" src="actual_train2/predictions.mp4"></video>
<h2>复用来源与下一步</h2><p>复用<a href="https://docs.opencv.org/4.x/dc/d6b/group__video__track.html">OpenCV LK</a>、<a href="https://github.com/opencv/opencv/blob/4.x/samples/python/lk_track.py">官方双向检查示例</a>及<a href="https://docs.prophesee.ai/stable/samples/modules/cv/sparse_flow_py.html">SDK稀疏光流</a>。CI是既有融合方法，这里的协方差只是工程不确定性代理，没有被标定成真实置信区间。本轮是已有算法的工程组合。</p>
<p>下一步检查匹配点是否包围整机中心，避免单片桨叶的局部匹配冒充整机平移；再考虑将图像准备或位移计算移出主检测关键路径。原位置通道继续单独保留，新收益以当前累计v4为起点迭代。</p>
<p>显式原版：<code>--profile live-baseline</code>；此前累计版：<code>--live-revision accumulated_v1</code>；当前默认无需附加版本选项。</p><p>证据：<a href="final_decision.json">累计决定</a> · <a href="comparison.json">冻结预测数据</a> · <a href="actual_side_scores.json">侧视实际评分</a> · <a href="actual_train2_scores.json">train2实际评分</a> · <a href="summary.json">裁剪/画面一致性</a> · <a href="mechanism_checks.json">已知平移与绕中心旋转测试</a></p></html>'''
(OUT/'index.html').write_text(page,encoding='utf-8')
snapshot=OUT/'source_snapshot';files=['live_local_flow.py','run_live_forecast.py','forecast.py','forecast_tracks.py','configs/live_tracks.json',
    'tools/check_local_flow.py','tools/probe_local_flow.py','tools/probe_crop_flow.py','tools/score_local_flow.py','tools/register_local_flow.py','tools/record_local_displacement.py']
for n in files:
    source=ROOT/n;data=source.read_bytes()
    if source.suffix=='.py':ast.parse(data)
    p=snapshot/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
hpath=WORK/'output/checkpoints/latest_handoff.json';h=json.loads(hpath.read_text('utf-8'))
h.update(execution_status='active goal; cumulative v4 local displacement recommended',latest_report=str(OUT/'index.html'),
    latest_live_revision='accumulated_v4_local_flow',current_jobs=[],current_goal_turn_classification='progress',
    quota_remaining=dict(short=args.short_remaining,weekly=args.weekly_remaining),
    next_action='Continue from recommended accumulated_v4_local_flow. All current jobs terminal. Prioritise spatial coverage/hull of LK inliers around body center to reject local rotor-only motion, and consider asynchronous image prep/flow to improve sampling continuity. v1/original frozen as selectable references; v2/v3 retained. No force or metric pose claim. Do not rerun completed training or alpha sweeps. Check quota each batch and pause at10%.')
hpath.write_text(json.dumps(h,ensure_ascii=False,indent=2),encoding='utf-8')
(WORK/'output/latest_live_forecast.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=local_displacement_tight_20261026/index.html"><a href="local_displacement_tight_20261026/index.html">最新累计优化报告</a>',encoding='utf-8')
with (ROOT/'docs/ACCUMULATED_MAINLINE.md').open('a',encoding='utf-8') as f:
    f.write('\n\n## 当前默认v4：局部位移补充速度\n\n`accumulated_v4_local_flow`在v1上增加纯ON/OFF支持图LK+双向检查+稳健相似变换，112px裁剪边界，CI只改rollout速度。模型权重保持v1，不增加神经网络。固定回放宏平均68.274→67.402；实际同发布起点side63.255→60.873、train2 94.817→88.247。侧视实际P95103.688→111.029，保留回退。原版/此前累计版均可选，长期实时与统计显著性未证明。\n')
with (WORK/'WORKLOG.md').open('a',encoding='utf-8') as f:
    f.write(f'\n\n### 局部位移进入默认累计v4（2026-10-26）\n\n本目标turn为progress：新增算法分支、CPU裁剪优化和真实RAW并行位置基线落地。112padding按窗口支持推导并8px对齐，未改匹配质量阈值；402当前帧合格时刻与full相同，最大位移差.085px；SDK稀疏失败保留不部署。最终固定0.4s macro68.274→67.402，留出变时域61.770→57.224，分别统计。\n\n最终真实4s两段：side94tracking、20pub/18valid、77flow，processing33.700ms/flow10.608ms、ageP95128.324；train2 100tracking、20pub/6valid、15flow，processing24.992ms/flow2.383ms、ageP9585.516。正确源/模型/图哈希，无缓存标签读取。当前同发布起点side.4均63.255→60.873、P95103.688→111.029；train2均94.817→88.247、P95132.372→132.869；等权平均{macro["position_only"]:.6f}→{macro["live_motion"]:.6f}，省{decision["actual_macro_gain_px"]:.6f}px。当前位置KF逐步精确回放，新增只改rollout速度；历史有效发布18/6未减。不是与旧低负载片段直接绝对排名，也不是长期实时或显著性证明。\n\n原版和v1/v2/v3保留，默认改为accumulated_v4_local_flow。192padding第一次实际负载/尾误差失败保留，未覆盖删除。会话15407(tight probe)、48498(tight side)、10202(tight train2)均exit0；前述13554/1131/73665/76112也均exit0。当前无本轮运行任务，goal保持active；quota余短{args.short_remaining}%/周{args.weekly_remaining}%。报告output/local_displacement_tight_20261026/index.html。下一步匹配点覆盖/凸包围绕中心资格，随后异步处理减少关键路径，不能再从旧原版反复重启。\n')
print(json.dumps(decision,ensure_ascii=False),flush=True)
