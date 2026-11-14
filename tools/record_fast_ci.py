"""Close spatial probes and accumulate an exact solver speedup."""
from pathlib import Path
import argparse,ast,datetime,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;OUT=WORK/'output/fast_ci_20261026';SPATIAL=WORK/'output/spatial_flow_20261026'
parser=argparse.ArgumentParser();parser.add_argument('--short-remaining',type=int,required=True);parser.add_argument('--weekly-remaining',type=int,required=True);args=parser.parse_args()
assert not (OUT/'final_decision.json').exists()
result=json.loads((OUT/'comparison.json').read_text('utf-8'));runtime=json.loads((OUT/'raw_smoke/completion.json').read_text('utf-8'))
assert runtime['sources_unchanged'] and runtime['frame_descriptor_mismatches']==0 and runtime['recipe_id']=='accumulated_v5_fast_ci'
assert not runtime['labels_read'] and not runtime['output_caches_read']
assert result['actual_coefficients_and_audits_exact'] and result['known_endpoint_and_complementary_axis_cases_exact']
path=ROOT/'configs/live_tracks.json';manifest=json.loads(path.read_text('utf-8'));assert manifest['recommended']=='accumulated_v4_local_flow'
manifest['recommended']='accumulated_v5_fast_ci';manifest['tracks']['accumulated_v5_fast_ci'].update(status='recommended_exact_solver_acceleration',evidence=str(OUT/'comparison.json'))
path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
spatial=json.loads((SPATIAL/'summary.json').read_text('utf-8'));weights=json.loads((WORK/'output/symmetric_velocity_20261026/comparison.json').read_text('utf-8'))
old=result['timing']['frozen_optimizer'];new=result['timing']['dominance_shortcut'];saved=old['median_ms']-new['median_ms']
decision=dict(recommended='accumulated_v5_fast_ci',parent='accumulated_v4_local_flow',default_promoted=True,
    point_predictions_changed=False,new_accuracy_gain_claimed=False,solver_median_saved_ms=saved,
    numerical_match_ticks=sum(r['exact_coefficient_and_audit_ticks'] for r in result['replay'].values()),
    component_only_speed_measurement=True,raw_entry_smoke_passed=True,
    spatial_hull_had_no_effect=True,symmetric_CI_retained_as_experiment=True)
(OUT/'final_decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(10,3.8))
ax=axes[0];values=[old['median_ms'],new['median_ms']];ax.bar([0,1],values,color=['#377eb8','#e17c05'])
ax.set_xticks([0,1],['Frozen scalar solver','PSD dominance shortcut']);ax.set_ylabel('Median CI solve cost (ms)');ax.set_ylim(0,max(values)*1.25)
for x,v in enumerate(values):ax.text(x,v+.006,f'{v:.4f}',ha='center')
ax.set_title('Solver component only; same predictions')
ax=axes[1];values=[spatial['results'][n]['v4_qualified'] for n in ['side','train2','test3']]
inside=[spatial['results'][n]['center_in_hull'] for n in ['side','train2','test3']]
ax.bar(np.arange(3)-.18,values,.36,label='Qualified v4 flow');ax.bar(np.arange(3)+.18,inside,.36,label='Center inside hull')
ax.set_xticks(range(3),['Side','Train2','Test3']);ax.set_ylabel('Accepted observation count');ax.set_title('Hull condition was already satisfied');ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/'mechanism_and_cost.png',dpi=160);plt.close(fig)

meta=json.loads((SPATIAL/'side_support.json').read_text('utf-8'));rows=json.loads((SPATIAL/'side.json').read_text('utf-8'))
index=next(i for i in range(len(rows)//2,len(rows)) if rows[i]['observation'].get('valid'))
cache=np.load(SPATIAL/'side_support.npz',allow_pickle=False)
mask=np.unpackbits(cache['packbits'][index-1]).reshape(720,1280)*255;s=rows[index]['observation']['spatial_support']
hull=np.asarray(s['hull']);center=np.asarray(s['center']);bbox=np.asarray(meta['records'][index-1]['body']['bbox'])
fig,ax=plt.subplots(figsize=(8,4));ax.imshow(mask,cmap='gray',vmin=0,vmax=255)
closed=np.r_[hull,hull[:1]];ax.plot(closed[:,0],closed[:,1],color='#32b5e9',lw=2,label='Matched feature hull');ax.plot(*center,'x',color='#ff9900',ms=12,mew=3,label='Previous detected center')
ax.set_xlim(bbox[0]-30,bbox[2]+30);ax.set_ylim(bbox[3]+30,bbox[1]-30)
ax.set_title(f'Binary native event support {meta["records"][index-1]["t"]:.3f}s; match to {rows[index]["t"]:.3f}s')
ax.legend(fontsize=8);ax.set_xlabel('x (px)');ax.set_ylabel('y (px)');fig.tight_layout();fig.savefig(OUT/'spatial_support_example.png',dpi=160);plt.close(fig);cache.close()

cache_bytes=sum(spatial['results'][n]['support_cache_bytes'] for n in spatial['results'])
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>空间支持试验与累计v5求解加速</title>
<style>body{{font:17px/1.65 system-ui,sans-serif;max-width:1050px;margin:40px auto;padding:0 20px;color:#243040}}h1,h2{{line-height:1.3}}img{{max-width:100%}}.note{{padding:15px;background:#eaf4e9}}table{{border-collapse:collapse;width:100%;margin:15px 0}}td,th{{border:1px solid #d4dee8;padding:9px}}code{{word-break:break-all}}</style>
<h1>空间支持试验与累计v5求解加速</h1><p class="note">默认入口已升为 <b>accumulated_v5_fast_ci</b>：保留v4预测数值，累积精确的CI求解加速。395个实际输入时刻的系数及求解审计逐项完全一致；本轮没有新增位置精度收益。</p>
<h2>真正累积的优化</h2><p>当一个信息矩阵在半正定意义上支配另一个时，CI最小体积解必在同一端点。新增一次2×2矩阵判断直接选出该端点；不满足支配关系仍使用原来的标量优化器。原v4输入中267次有效修正均命中这个可精确判断的情形。</p>
<p>交替五轮CPU对照，原求解器中位 {old['median_ms']:.4f}ms → {new['median_ms']:.4f}ms，每次有效融合节省约 {saved:.4f}ms。只测求解器组件，不能把该比例当作整条流水线的加速比。已比较流分支端点、位置端点、相等协方差与非支配互补轴情形，保留后者的原求解路径。</p>
<img src="mechanism_and_cost.png" alt="CI组件开销下降，空间凸包条件已全部满足">
<h2>空间检查没有带来精度增益的原因</h2><p>侧视77/77、train2 15/15、录像3 175/175的合格匹配都已包围整机框中心，因此增加凸包门槛没有筛除任何修正。按四个空间区域做留组重算并增加不确定性后，CI仍选择原光流端点，预测几乎完全不变。这些样本不支持“尾误差主要来自单片局部特征不覆盖中心”的判断，不能继续把该猜想当作既定原因。</p>
<img src="spatial_support_example.png" alt="原生二值事件支持图、匹配点凸包与整机框中心"><p>图中是整机框中心代理，不能据此认定真实重心或刚体运动已被测得。</p>
<h2>对称CI试验也完整保留</h2><p>查阅<a href="https://stonesoup.readthedocs.io/en/v1.7/_modules/stonesoup/updater/chernoff.html">Stone Soup官方ChernoffUpdater源码</a>，使用其默认omega=0.5的既有高斯CI公式，自行作小型NumPy适配，没有安装依赖或声称运行其完整框架。只试这一个固定设置，没有扫权重。当前实际v4同发布起点的0.4s宏平均 {weights['development_macro04s']['v4']:.6f} → {weights['development_macro04s']['symmetric']:.6f}px，变化很小且均值略差；空间协方差组合为 {weights['development_macro04s']['symmetric_spatial_cov']:.6f}px。侧视尾误差有小幅取舍，结果和实现保留在实验目录，默认使用数值不变的加速路径。</p>
<h2>减少以后重复解码</h2><p>395张精确原生二值支持图共压缩为 {cache_bytes/1e6:.3f}MB，每张都有支持图哈希、原生画面哈希、当前检测框和时间戳。已用侧视缓存重放94张并核对哈希，无需启动RAW或GPU。该缓存仅用于空间位移算法试验；帧内极性与微秒时间已丢失，转速/周期分析仍需原生事件。实际RAW入口继续独立解码并禁止读这些输出缓存。</p>
<p>离线复用工具：<code>tools/replay_local_support.py --cache-json &lt;case_support.json&gt; --output &lt;new.json&gt; --spatial-diagnostics</code>。</p>
<h2>实际入口与当前边界</h2><p>v5的1.2秒RAW接线完成：{runtime['tracking_updates']}次跟踪、{runtime['publication_slots']}次发布/{runtime['valid_publications']}次有效，源文件及共享画面核对通过，未读标签或输出缓存。它是短入口检查，不能与先前不同负载、长度的测速直接排名。此前v4的真实位置精度与视频继续见<a href="../local_displacement_tight_20261026/index.html">上一轮实际预测报告</a>，本轮不重新宣称精度提升。</p>
<h2>下一步</h2><p>优先将当前帧的事件支持图生成移到GPU请求发送后、等待GPU回复前，让这项CPU准备与当前推理重叠。输入画面保持不可变，先做这项不改变预测公式的并行化；随后再考虑局部位移和位置滤波的并行计算，以减少采样丢失。当前仍是二维位置预测，没有新推力或真实转速估计。</p>
<p>证据：<a href="comparison.json">逐条数值与CPU测量</a> · <a href="final_decision.json">累计决定</a> · <a href="../spatial_flow_20261026/comparison.json">空间方案完整结果</a> · <a href="../symmetric_velocity_20261026/comparison.json">对称CI完整结果</a> · <a href="../spatial_flow_20261026/summary.json">原生支持图缓存</a> · <a href="raw_smoke/completion.json">实际入口</a></p></html>'''
(OUT/'index.html').write_text(page,encoding='utf-8')
snapshot=OUT/'source_snapshot';files=['live_local_flow.py','live_flow_spatial.py','tools/check_spatial_flow.py','tools/probe_spatial_flow.py','tools/score_spatial_flow.py','tools/check_symmetric_velocity.py','tools/replay_local_support.py','tools/probe_fast_ci.py','tools/register_fast_ci.py','tools/record_fast_ci.py','configs/live_tracks.json']
for n in files:
    source=ROOT/n;data=source.read_bytes()
    if source.suffix=='.py':ast.parse(data)
    p=snapshot/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
hpath=WORK/'output/checkpoints/latest_handoff.json';h=json.loads(hpath.read_text('utf-8'))
h.update(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution_status='active goal; cumulative v5 exact CI acceleration recommended',
    latest_report=str(OUT/'index.html'),latest_live_revision='accumulated_v5_fast_ci',current_jobs=[],current_goal_turn_classification='progress',
    quota_remaining=dict(short=args.short_remaining,weekly=args.weekly_remaining),
    next_action='Continue from recommended accumulated_v5_fast_ci. First overlap frame_support(image,background) with current GPU inference: after gpu.stdin.flush(), compute support while GPU works, then receive and use same immutable support. No need delayed/asynchronous measurements. Preserve exact coefficients on packed native support cache. Then consider overlapping motion.step and LK. Spatial hull/leave-group and fixed symmetric CI experiments already completed, no aggregate accuracy gain; do not repeat sweeps. All jobs terminal. Check quota each batch; save/pause at10%.')
hpath.write_text(json.dumps(h,ensure_ascii=False,indent=2),encoding='utf-8')
(WORK/'output/latest_live_forecast.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=fast_ci_20261026/index.html"><a href="fast_ci_20261026/index.html">最新累计优化报告</a>',encoding='utf-8')
with (ROOT/'docs/ACCUMULATED_MAINLINE.md').open('a',encoding='utf-8') as f:
    f.write('\n\n## 当前默认v5：CI求解器精确加速\n\nv5保留v4点预测，PSD信息支配时直接选同一CI端点，否则走原优化器。395个输入的系数和审计逐项相同；CPU求解中位.2298→.0286ms，仅组件开销。空间凸包/留组协方差没有新增精度收益，对称CI宏平均略差，均保留实验。精确原生支持缓存395张共2.095MB供后续空间算法复用，不用于生产或RPM。\n')
with (WORK/'WORKLOG.md').open('a',encoding='utf-8') as f:
    f.write(f'\n\n### 空间支持诊断与累计v5精确CI加速（2026-10-26）\n\n前一turn为progress（v4实际RAW/默认落地）。本轮先读OpenCV凸包/LS及Stone Soup CI公开源码。两种固定空间机制：中心凸包包含与四象限留组模型分歧协方差，既有LK阈值不变。实际v4帧94+100及test3 201合计395张，所有原生哈希/支持图精确，v4匹配位移最大差0，267个合格匹配均含中心；空间两种方案宏平均74.559982不变。不能再假定尾误差主要由局部单桨覆盖不足造成。\n\n原CI267/267基线信息权重0。按Stone Soup默认固定omega.5试对称CI及空间协方差组合，不扫权重，宏平均74.605279/74.603692，略差但尾部有小取舍，实验/实现保留。\n\n转向数学等价的性能优化：PSD信息矩阵支配关系可直接决定CI端点，否则原优化器。395个系数及审计逐项完全一致，端点/相等/互补轴案例通过；交替5轮CPU微测，每方法1335有效调用，中位.2298→.0286ms，P95 .28517→.04470ms。只声称每次求解省.2012ms，不声称整条流水线8倍。默认升级accumulated_v5_fast_ci，原版/v1/v2/v3/v4继续可选；没有新增准确率收益。\n\n缓存395张纯原生二值空间支持图共{cache_bytes}bytes，逐张哈希；side离线94张重放.628s，无RAW/GPU/标签，不用于RPM或生产。v5实际1.2s RAW接线27tracking、6pub/4valid、10flow、processing32.086ms，源不变且帧核对0错、无缓存/标签读取，不与其他负载测速直接比较。会话18161(spatial native)、31959(v5 smoke)均exit0，其余同步工具exit0，无本轮运行任务。报告output/fast_ci_20261026/index.html。\n\n下一步在发送GPU请求后、等待回复前生成本帧支持图，重叠CPU与GPU，保留完全相同输入和公式，随后考虑位置滤波/LK并行。goal active；quota余短{args.short_remaining}%/周{args.weekly_remaining}%。\n')
print(json.dumps(decision,ensure_ascii=False),flush=True)
