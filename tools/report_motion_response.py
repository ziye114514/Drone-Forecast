"""Summarize completed optimization batches; preserve all failures and controls."""
from pathlib import Path
import sys,json,hashlib,html,re,shutil
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[2];CODE=ROOT/'event_drone_forecast';OUT=ROOT/'output/motion_consistency_30px_20261028'
sys.path.insert(0,str(CODE/'tools'))
from compare_final_pipeline import reference_at
def read(p):return json.loads(p.read_text('utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text('utf-8').splitlines()]
def timing(folder,stop=None):
    a=rows(OUT/folder/'timings.jsonl')[1:]
    if stop is not None:a=[r for r in a if r['t']<stop]
    service=np.array([r['total_service_ms'] for r in a]);lag=np.array([r['completion_lag_ms'] for r in a])
    return dict(n=len(a),capacity_FPS=float(1000/service.mean()),service_p95_ms=float(np.percentile(service,95)),
        completion_lag_p95_ms=float(np.percentile(lag,95)),last10_lag_ms=lag[-10:].tolist(),
        service_within_33ms=float(np.mean(service<=1000/30)),
        worst_decode=max(a,key=lambda r:r['producer_native_decode_ms']))
def errors(a,truth):
    times=[];values=[];last=None
    for r in a:
        slot=int(np.floor(r['t']*5+1e-8))
        if not r['valid'] or slot==last:continue
        last=slot;target=reference_at(truth,r['t']+.4)
        if target is not None:times.append(r['t']);values.append(float(np.linalg.norm(np.asarray(r['physical']['0.4'])-target)))
    return np.asarray(times),np.asarray(values)
def main():
    side=read(OUT/'scored_side_release/summary.json');other=read(OUT/'scored_train2_candidate/summary.json')
    final='calibrated_gap';side_score=side['scores'][final]['0.4'];train_score=other['scores']['candidate']['0.4']
    side_time=timing('RAW_side_release_candidate8s',114.)
    clear_time=timing('RAW_clear_release_candidate2s');clear_summary=read(OUT/'RAW_clear_release_candidate2s/summary.json')
    previous_stall=timing('RAW_clear_candidate2s');prior_audit=timing('RAW_clear_CPU_audit2s')
    stream=read(OUT/'RAW_train2_body_stream6s/recovery_score.json')
    window=read(OUT/'RAW_train2_aligned_fast3_6s/recovery_score.json')
    inertial=read(OUT/'net_acceleration_hold/summary.json')
    controls=read(OUT/'controls_complete/summary.json');geometry=read(OUT/'geometry_pruning_equivalence.json')
    source_names=['run_fast_structure.py','shared_image_motion.py','live_flow_acceleration.py','curved_flow_motion.py',
        'fast_force_response.py','flow_force_balance.py','projected_unknown_force.py','endpoint_force_balance.py',
        'normal_turn_forecast.py','adaptive_force_memory.py','flow_anchor_gauge.py','thrust_trend_forecast.py',
        'body_period_stream.py','aligned_period_recovery.py','fast_geometry_compute.py','fast_raw_reader.py',
        'tools/probe_same_window_motion.py','tools/score_motion_response.py','tools/score_period_recovery.py',
        'tools/check_motion_response.py','tools/check_aligned_period.py','tools/report_motion_response.py',
        'net_acceleration_hold.py','README.md','tools/prepare_github_release.py','docs/MOTION_RESPONSE_30PX.md']
    hashes={}
    for name in source_names:
        data=(CODE/name).read_bytes();p=OUT/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);hashes[name]=hashlib.sha256(data).hexdigest()
    archives=[('github_release_final_20261027','event-drone-forecast-v0.4.0-alpha.1.zip','ff1172b3f2d2e1752d8ebc6d299c6d972adfaaa1776f8773240e44aa3eb58abb'),
        ('github_release_update_20261028','event-drone-forecast-v0.4.0-alpha.2.zip','6260e0919a8dea574e8c502a0f982b3ed5791d5fe8d159c6b2e740f889b97e45')]
    for folder,name,digest in archives:assert hashlib.sha256((ROOT/'output'/folder/name).read_bytes()).hexdigest()==digest
    preserved=[]
    for p in (CODE/'src/event_drone_forecast/position').glob('*.py'):
        q=ROOT/'output/github_release_final_20261027/event-drone-forecast/src/event_drone_forecast/position'/p.name
        assert p.read_bytes()==q.read_bytes(),p;preserved.append(p.name)
    assert len(preserved)==7
    truth=sorted(read(ROOT/'output/dense_labels_side_110_1195/labels.json')['records'].values(),key=lambda r:r['t'])
    old=rows(OUT/'scored_side_release/pre_stage.jsonl');new=rows(OUT/'scored_side_release/calibrated_gap.jsonl')
    t,e0=errors(old,truth);_,e1=errors(new,truth)
    othertruth=sorted(read(ROOT/'output/dense_labels_train2_40_46/labels.json')['records'].values(),key=lambda r:r['t'])
    tt,ee0=errors(rows(OUT/'scored_train2_candidate/pre_stage.jsonl'),othertruth)
    _,ee1=errors(rows(OUT/'scored_train2_candidate/candidate.jsonl'),othertruth)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(3,1,figsize=(11,10),constrained_layout=True)
    for ax,stamps,before,after,title in [(axes[0],t,e0,e1,'侧视：相同26个0.4秒预测目标'),(axes[1],tt,ee0,ee1,'另一录像：相同15个0.4秒预测目标')]:
        for vals,label,color in [(before,'本轮之前','#8d98a7'),(after,'当前五圆物理候选','#167ba5')]:
            x=[];y=[]
            for i,(stamp,val) in enumerate(zip(stamps,vals)):
                if i and stamp-stamps[i-1]>.25:x.append(np.nan);y.append(np.nan)
                x.append(stamp);y.append(val)
            ax.plot(x,y,'.-',label=label,color=color)
        ax.axhline(30,color='#b94a43',ls='--',label='30px目标');ax.set(title=title,xlabel='源时间/s',ylabel='误差/px');ax.legend();ax.grid(alpha=.2)
    axes[2].plot([r['t'] for r in old],[r['force']['effective_gain_px_s2'] for r in old],label='原固定时间学习',color='#8d98a7')
    axes[2].plot([r['t'] for r in new],[r['force']['effective_gain_px_s2'] for r in new],label='新证据驱动增益',color='#167ba5')
    axes[2].set(title='仅用已发生的位置与当前周期学习有效图像增益；无机械推力真值',xlabel='源时间/s',ylabel='有效增益/(px/s²)')
    axes[2].grid(alpha=.2);axes[2].legend();fig.savefig(OUT/'optimization_results.png',dpi=150);plt.close(fig)
    result=dict(updated_local_date='2026-10-29',goal_complete=False,target_mean04_px=30.,target_processing_FPS=30.,
        actual_side=side,train2_frozen_candidate=other,side_fixed4s_timing=side_time,clear_final_timing=clear_time,
        clear_actual_summary=clear_summary,clear_first_stalled_trial=previous_stall,clear_previous_CPU_audit=prior_audit,
        side_below30_fraction=float(np.mean(e1<30)),train2_below30_fraction=float(np.mean(ee1<30)),
        body_period_experiment=stream,window_period_experiment=window,inertial_net_acceleration_experiment=inertial,
        controls=controls,geometry_equivalence=geometry,
        promoted=['curved current motion','novel precise gain evidence','endpoint unknown-force balance','matured force memory','calibrated period-gap persistence','equivalent geometry pruning'],
        not_default=['aligned period retry','persistent body period stream','direct balance EWMA removal','same-window intercept alternative','anchor gauge','thrust trend','net acceleration inertial hold'],
        published_archives_unchanged=True,position_core_files_unchanged=preserved,source_hashes=hashes,current_running_jobs=[])
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    labels={'pre_stage':'本轮之前','curved':'当前加速度曲线','curved_gain':'加证据增益','candidate':'加合力持续性','endpoint':'加端点未知力平衡','calibrated_gap':'加标定后缺周期延续'}
    table=''.join(f'<tr><td>{labels[k]}</td><td>{v["0.4"]["mean"]:.3f}</td><td>{v["0.4"]["p95"]:.3f}</td><td>{v["0.4"]["n"]}</td></tr>' for k,v in side['scores'].items())
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>五圆物理预测：向30px累计优化</title>
<style>body{{max-width:1100px;margin:36px auto;padding:0 24px;background:#f3f6fa;color:#1c3048;font:17px/1.8 system-ui}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{padding:10px;border:1px solid #cbd5e1;text-align:left}}.note{{border-left:5px solid #1682a5;padding:14px 20px;background:#e6f2f7}}img{{width:100%}}pre{{background:#e5ecf2;padding:16px;white-space:pre-wrap}}a{{color:#126e99}}</style>
<h1>五圆模型与二力反馈：向30px累计优化</h1><p>更新：2026年10月29日。二维、0.4秒平均欧氏误差；目标尚未完成。</p>
<p class="note">最终实际侧视RAW：同输入从{side['scores']['pre_stage']['0.4']['mean']:.2f}降到<strong>{side_score['mean']:.2f}px</strong>，P95为{side_score['p95']:.2f}px。
另一录像从{other['scores']['pre_stage']['0.4']['mean']:.2f}降到<strong>{train_score['mean']:.2f}px</strong>。没有移除原有评分时刻或修改标框。</p>
<h2>已经合并到候选版的改动</h2><ol><li>从近期速度曲线估计当前加速度，减少变向滞后；曲率依不确定度连续收缩，不把加加速度直接当未来命令。</li>
<li>有效增益按新的合格证据更新，相同成熟样本不重复学习；参数切换保持当前力坐标基准。</li>
<li>周期刚恢复但过去输入不完整时，让合格的当前总加速度约束未知合力，防止任意初值压过现有运动证据。</li>
<li>仅用已经成熟的自身位置学习合力持续性。增益已标定、当前轴线与净加速度合格时，周期缺失仍可延续这一净合力模型，不补转速。</li>
<li>去掉无用全椭圆遍历：60时刻全部几何/结构相同，距离计算722→444次；复用曲线已有拟合与取消无用协方差计算。</li></ol>
<pre>原生ON/OFF → 当前整机ROI → 原弧/椭圆与五圆约束
            → 当前机体平面/可见模式 + 周期代理
            → a_image = k·u_current + d_unknown
            → 当前加速度与未知合力衔接 → 未来法线/合力持续性 → 位置
机头和电机编号不作为合推力预测前提；没有新增并行运动学预测器或第二个位置滤波器。</pre>
<h2>相同侧视输入上的累计结果</h2><table><tr><th>版本</th><th>均值/px</th><th>P95/px</th><th>目标数</th></tr>{table}</table>
<p>侧视30px内比例：{100*np.mean(e1<30):.1f}%；另一录像：{100*np.mean(ee1<30):.1f}%。这些是已经多轮使用的开发标框，不是独立测试集或论文SOTA对比。</p>
<img src="optimization_results.png" alt="同输入误差和增益学习曲线">
<h2>实际速度与延迟</h2><table><tr><th>范围/版本</th><th>处理能力FPS</th><th>说明</th></tr>
<tr><td>最终侧视110–114秒</td><td>{side_time['capacity_FPS']:.2f}</td><td>原固定范围，含慢帧；尾10延迟{min(side_time['last10_lag_ms']):.1f}–{max(side_time['last10_lag_ms']):.1f}ms</td></tr>
<tr><td>最终清晰26–28秒</td><td>{clear_time['capacity_FPS']:.2f}</td><td>当前观测{clear_summary['fresh_image_observations']}/60；完成延迟P95 {clear_time['completion_lag_p95_ms']:.1f}ms</td></tr>
<tr><td>此前清晰段停顿试验</td><td>{previous_stall['capacity_FPS']:.2f}</td><td>保留3.109秒原生解码停顿，没有删除或当作启动排除</td></tr></table>
<p>处理能力、当前观测Hz、视频播放FPS和逐帧截止期限不同。侧视全8秒约49FPS含无目标时段，不能当检测50FPS。
即使短段处理能力超过30，初期积压与偶发停顿仍说明稳定低延迟30FPS尚未证明。额外记录了解码CPU与等待代理，停顿原因仍未定位。</p>
<h2>周期恢复做到了什么，卡在哪里</h2><p>复用SDK、原时间/极性、同16像素与环带规则：160ms重放严格7周期无新增，短3周期恢复8时刻；固定机体坐标持续处理恢复31个周期时刻，力输入3→26，速度48.13FPS。
但单独接入使0.4秒误差59.79→65.94px，端点平衡重放后60.95px，仍未优于原生通道。因此周期流保持可选，没有因候选数更多就宣称精度更高。</p>
<p>目前真正限制继续逼近30px的是：弱片段无法持续标定周期到图像加速度的关系，谐波/可见群体变化仍可能污染推力代理；0.4秒后命令未观测，合力持续性需要更可靠；整机缺席/遮挡与RAW停顿限制连续输出。</p>
<p>另做了120ms净加速度惯性续接：侧视评分不变，另一录像59.79→66.31px，未接默认。变向时当前速度仍可能暂时支持旧加速度，单靠惯性无法可靠区分失锁与真实制动；没有为压低开发误差而调长保持时间。</p>
<h2>复现与交接</h2><p>工作入口<code>run_fast_structure.py --target-candidate</code>已启用本轮有效改动。当前周期仍采用原生通道。
本轮同输入对照保留<code>--tangent-disturbance</code>，再用<code>--no-curved-motion --no-evidence-gain --no-endpoint-force-balance --no-adaptive-force-memory</code>；仅关闭最后缺周期延续用<code>--memory-requires-current-period</code>。</p>
<p>本轮没有上传GitHub、重做发布ZIP或修改保护目录；7个原位置核心文件和两个发布ZIP保持。
SDK事件流和频率是既有组件，复用方法见<a href="https://docs.prophesee.ai/stable/data/streaming_decoding/index.html">Prophesee官方说明</a>。</p>
<p><a href="summary.json">完整结果和源代码哈希</a> · <a href="scored_side_release/summary.json">最终RAW同输入对照</a> · <a href="controls_complete/summary.json">机制检查</a> · <a href="source_snapshot/">源代码快照</a></p></html>'''
    (OUT/'index.html').write_text(page,'utf-8')
    note=f'本轮收尾实际RAW：侧视{side_score["mean"]:.4f}px/P95 {side_score["p95"]:.4f}；同输入旧{side["scores"]["pre_stage"]["0.4"]["mean"]:.4f}。另一录像{train_score["mean"]:.4f}px。目标未完成。'
    log=ROOT/'WORKLOG.md';text=log.read_text('utf-8');marker='## 2026-10-28 当前加速度、证据增益与合力持续性（进行中）'
    block='## 2026-10-28—29 五圆物理预测累计优化\n\n- '+note+'\n- 已把曲线当前加速度、新证据增益、端点未知力平衡、成熟自身位置合力持续性、标定后缺周期延续接入target-candidate；原运动学对照和显式关闭开关保留，不依赖机头或电机编号。\n- 最终侧视前4秒处理'+f'{side_time["capacity_FPS"]:.4f}'+'FPS；最终清晰2秒'+f'{clear_time["capacity_FPS"]:.4f}'+'FPS。处理能力超过30不代表每帧准时；此前清晰3.109秒解码停顿保留，原因未解决。\n- 全椭圆去掉无用遍历，60冻结几何/结构逐项相同；曲线复用已有线性拟合、训练不计算无用协方差，实际重放输出不变。\n- 固定机体坐标频率流使用既有SDK fast3，事件只送入一次；新增31周期时刻、力输入3→26、48.13FPS，但59.79→65.94px，端点修正后60.95px，保持实验关闭。16像素规则未降低。\n- 报告output/motion_consistency_30px_20261028/index.html；全部失败、控制、RAW快照和分项时间保留。7位置核心/发布alpha.1和alpha.2 ZIP哈希保持，保护目录未动，未重新发布GitHub。\n- 下一步优先周期来源/谐波与推力关系、缺周期时可观测净力、变向长尾和解码停顿；不要只靠恢复更多周期或重复测已有收益。当前无后台作业。\n'
    if marker in text:
        begin=text.index(marker);end=text.find('\n\n## ',begin+len(marker));text=text[:begin]+block+(text[end:] if end>=0 else '')
    elif '## 2026-10-28—29 五圆物理预测累计优化' in text:
        begin=text.index('## 2026-10-28—29 五圆物理预测累计优化');end=text.find('\n\n## ',begin+10);text=text[:begin]+block+(text[end:] if end>=0 else '')
    else:text+='\n\n'+block
    log.write_text(text,'utf-8')
    checkpoint=ROOT/'output/checkpoints/latest_handoff.json';handoff=read(checkpoint);handoff.pop('motion_response_in_progress',None)
    handoff['motion_response_30px']=result
    handoff.update(utc=datetime.now(timezone.utc).isoformat(),updated_utc=datetime.now(timezone.utc).isoformat(),updated_local_date='2026-10-29',
        latest_report=str(OUT/'index.html'),latest_research_report=str(OUT/'index.html'),
        current_work_state=note,execution_status='Current optimization batch completed; 30px research goal incomplete; no running RAW/GPU jobs',
        research_objective_status='incomplete',last_completed_named_batch='motion_response_30px_20261029',
        in_progress='30px objective incomplete; current batch and report complete; no process running',
        current_running_jobs=[],running_jobs=[],current_jobs=[],pending_runs=[],experiments_running=False,
        current_mainline_entry=str(CODE/'run_fast_structure.py'),primary_research_entry=str(CODE/'run_fast_structure.py'),
        work_in_progress_output=str(OUT),next_action='Qualify periodicity-to-force mapping in the weak clip; preserve current endpoint acceleration and calibrated gap memory; reduce reversal tails and investigate native decode stalls',
        goal_state='30px objective incomplete; legacy goal tool status is separate; no user pause requested',quota_pause_requested=False,
        stable_high_accuracy_realtime_achieved=False,real_time_traditional_branch_enabled=False,
        latest_named_report_is_authoritative=True,older_numbered_stage_fields_are_historical=True)
    handoff['next_stage_target']=dict(handoff.get('next_stage_target',{}),report=str(OUT/'index.html'),mean04_px_target=30.,goal_complete=False,
        current_state=note,current_running_jobs=[],next_actions=['Qualify period source and harmonics before thrust use','Retain current endpoint force evidence during sparse periods','Reduce reversal tails and resolve decode stalls without repeated identical tests'])
    checkpoint.write_text(json.dumps(handoff,ensure_ascii=False,indent=2),'utf-8')
    entry=ROOT/'output/结果入口.md';content=entry.read_text('utf-8');line='- [2026-10-29 五圆物理预测累计优化](motion_consistency_30px_20261028/index.html) — '+note
    if line not in content:entry.write_text(line+'\n\n'+content,'utf-8')
    print(json.dumps(dict(report=str(OUT/'index.html'),side_mean=side_score['mean'],side_p95=side_score['p95'],train2_mean=train_score['mean'],side_fixed_FPS=side_time['capacity_FPS'],clear_FPS=clear_time['capacity_FPS'],goal_complete=False),ensure_ascii=False))
if __name__=='__main__':main()
