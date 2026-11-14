"""Current no-heading disk forecast report; preserve all fixed origins and failed trials."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib,html,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[2];CODE=ROOT/'event_drone_forecast';OUT=ROOT/'output/disk_motion_30px_20261028'
sys.path.insert(0,str(CODE/'tools'))
from compare_final_pipeline import reference_at
def read(p):return json.loads(p.read_text('utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text('utf-8').splitlines()]
def main():
    frozen=read(OUT/'normal_frozen_v2/summary.json')
    side=read(OUT/'RAW_side_normal8s/scored_summary.json')
    other=read(OUT/'RAW_train2_temporal6s_final/scored_summary.json')
    timing=rows(OUT/'RAW_side_normal8s/timings.jsonl')
    fixed=[r for r in timing[1:] if r['t']<114.]
    fixed_fps=1000/np.mean([r['total_service_ms'] for r in fixed])
    lag=[r['completion_lag_ms'] for r in fixed[-10:]]
    source_names=['normal_turn_forecast.py','temporal_period_support.py','axis_amplitude_bridge.py',
        'run_fast_structure.py','live_pixel_axis.py','tools/check_disk_prediction.py',
        'tools/probe_normal_forecast.py','tools/score_disk_prediction.py','docs/DISK_NORMAL_FORECAST.md']
    sources={}
    for name in source_names:
        b=(CODE/name).read_bytes();p=OUT/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
        sources[name]=hashlib.sha256(b).hexdigest()
    # Render already frozen source outputs, no further inference or model selection.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
    a=rows(OUT/'RAW_side_normal8s/predictions.jsonl');b=rows(OUT/'RAW_side_normal8s/normal_off.jsonl')
    truth=sorted(read(ROOT/'output/dense_labels_side_110_1195/labels.json')['records'].values(),key=lambda r:r['t'])
    times=[];old=[];new=[];last=None
    for left,right in zip(b,a):
        slot=int(np.floor(right['t']*5+1e-8))
        if not right['valid'] or slot==last:continue
        last=slot;target=reference_at(truth,right['t']+.4)
        if target is not None:
            times.append(right['t']);old.append(np.linalg.norm(np.asarray(left['physical']['0.4'])-target));new.append(np.linalg.norm(np.asarray(right['physical']['0.4'])-target))
    fig,axes=plt.subplots(2,1,figsize=(10,8),constrained_layout=True)
    for values,label,color in [(old,'同输入：方向保持','#858e9b'),(new,'当前圆饼法线演化','#1676a4')]:
        tt=[];vv=[]
        for i,(t,v) in enumerate(zip(times,values)):
            if i and t-times[i-1]>.25:tt.append(np.nan);vv.append(np.nan)
            tt.append(t);vv.append(v)
        axes[0].plot(tt,vv,'.-',label=label,color=color)
    axes[0].axhline(30,color='#c04b42',ls='--',label='下一目标30px')
    axes[0].set(title='实际侧视同26目标：42.805 → 42.404px；P95 77.999 → 77.078px',
        xlabel='预测起点源时间/s',ylabel='0.4秒误差/px');axes[0].legend(fontsize=9);axes[0].grid(alpha=.2)
    train=rows(OUT/'RAW_train2_temporal6s_final/predictions.jsonl');t=[r['t'] for r in train]
    axes[1].plot(t,[(r.get('temporal_period_support') or {}).get('current_mode_pixels',0) for r in train],label='当前同频像素',color='#6d7d98')
    axes[1].plot(t,[(r.get('temporal_period_support') or {}).get('distinct_mode_pixels',0) for r in train],label='短窗去重同频支持',color='#d28d24')
    yes=[r for r in train if (r.get('temporal_period_support') or {}).get('applied')]
    axes[1].scatter([r['t'] for r in yes],[(r['temporal_period_support'])['distinct_mode_pixels'] for r in yes],color='#268753',label='合格周期4次',zorder=5)
    axes[1].axhline(16,color='#9b4444',ls='--',label='保留16像素门槛')
    axes[1].set(title='另一录像恢复弱周期输入，当前位置误差未改善',xlabel='源时间/s',ylabel='不同原始像素数')
    axes[1].legend(fontsize=9);axes[1].grid(alpha=.2)
    fig.savefig(OUT/'disk_results.png',dpi=150);plt.close(fig)
    proof=read(OUT/'controls_wobble_final/summary.json')
    result=dict(target_mean_error_px=30.,goal_complete=False,heading_used=False,rotor_ids_used=False,
        normal_future_in_target_candidate=True,temporal_period_default=False,
        frozen_normal=frozen,actual_side=side,actual_train2=other,
        side_fixed4s_capacity_FPS=float(fixed_fps),side_fixed4s_last10_lag_ms=dict(min=min(lag),max=max(lag),mean=float(np.mean(lag))),
        mechanism_controls=proof,sources=sources,
        failed_first_RAW='Optional axis disagreement field None; fixed and counterexample added; partial outputs retained.',
        source_scope='Current exposed development clips; no independent physical force/RPM or true centroid labels')
    for folder,archive,digest in [('github_release_final_20261027','event-drone-forecast-v0.4.0-alpha.1.zip','ff1172b3f2d2e1752d8ebc6d299c6d972adfaaa1776f8773240e44aa3eb58abb'),
        ('github_release_update_20261028','event-drone-forecast-v0.4.0-alpha.2.zip','6260e0919a8dea574e8c502a0f982b3ed5791d5fe8d159c6b2e740f889b97e45')]:
        assert hashlib.sha256((ROOT/'output'/folder/archive).read_bytes()).hexdigest()==digest
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>圆饼机体：周期支持与未来法线</title>
<style>body{max-width:1100px;margin:36px auto;padding:0 24px;background:#f3f6fb;color:#19324e;font:17px/1.8 system-ui}table{border-collapse:collapse;width:100%;background:#fff}td,th{border:1px solid #cad5e2;padding:10px;text-align:left}pre{background:#e5ecf6;padding:18px;white-space:pre-wrap}.note{border-left:5px solid #2878a8;background:#e9f4fa;padding:12px 18px}a{color:#1265a4}img{width:100%}</style>
<h1>不分机头：圆饼平面、周期信号与未来法线</h1>
<p class="note">同型桨、四桨都计入时，交换编号不改变合推力。机头朝向不决定速度方向。当前新预测分支不使用机头/桨盘编号，保留机体平面倾斜与法线；漏计、重复和缺测集合改变仍需处理。</p>
<h2>已完成的优化</h2><p>从无向法线历史估计转动趋势，按当前观测噪声收缩，只把已知推力贡献随未来法线旋转；未知合力不旋转。
当前图像位移不支持平移时，不额外注入姿态摆动导致的位移。静止周期摆动91时刻、法线正负交换和桨编号置换的机制检查通过。</p>
<pre>未来位置 = 原有位置/速度/净加速度预测
        + ∫(h-s) exp(-s/τ) [R(ωs)-I] a_known ds
不需要机头或稳定桨盘编号；零转动/缺少当前资格时附加项为0。</pre>
<p>周期分支合并100ms内不同原生像素支持，保留16票、当前至少4票和跨窗一致性、环带抑制。
当前频率来自当前点，原生事件时间不改写，同一热像素不重复累计。</p>
<h2>固定评分结果</h2><table><tr><th>范围</th><th>控制均值 / P95</th><th>新预测均值 / P95</th><th>目标数</th></tr>
<tr><td>冻结侧视同输入</td><td>43.2745 / 80.6014</td><td>42.7192 / 77.5407</td><td>26</td></tr>
<tr><td>本次实际RAW侧视同输入</td><td>42.8050 / 77.9994</td><td>42.4040 / 77.0781</td><td>26</td></tr>
<tr><td>另一录像：无周期控制→联合</td><td>62.6464 / 124.0023</td><td>62.6796 / 124.0023</td><td>15</td></tr></table>
<p>新法线分支在侧视有小幅收益，已进入当前target-candidate；--no-normal-turn-forecast保留控制。
周期恢复4次合格输入、3次当前推力代理，但没有位置精度收益，保留显式实验选项--temporal-period-support。
所有原评分时刻保留，未换窗口/删除差预测。30px目标尚未完成。</p>
<img src="disk_results.png" alt="实际预测与周期支持">
<h2>实际RAW与速度</h2><p>侧视240源、175当前、171有效。全8秒49.97FPS含无目标轻负载；固定前4秒处理能力'''+f'{fixed_fps:.2f}'+'''FPS，尾10延迟'''+f'{min(lag):.1f}–{max(lag):.1f}'+'''ms，初期仍积压。
另一录像180源、111当前、107有效，实际28.95FPS；43.432899秒有一次约3019.93ms读取等待，生产端原生解码计时约3048.90ms。未删除该慢帧。
阻塞原因可能在SDK读取、存储或调度，当前未确认具体根因，不能声称稳定全程30FPS。</p>
<h2>限制和下一步</h2><p>机体圆饼省去机头身份，但推力方向仍受平面法线影响，速度还包含惯性和未知合力。当前仅外推已出现的法线趋势，不预知新控制指令。
弱周期只恢复少数时刻，下一步处理其连续性与噪声、利用当前事件信号支持；不以降低单帧门槛或假造RPM完成目标。
首轮RAW遇到可选字段None已修复，失败目录与反例保留。没有独立机械转速/推力真值。</p>
<p><a href="summary.json">完整记录</a> · <a href="source_snapshot/docs/DISK_NORMAL_FORECAST.md">方法与参数</a> · <a href="controls_wobble_final/summary.json">静止波动等机制检查</a> · <a href="RAW_side_normal8s/scored_summary.json">侧视</a> · <a href="RAW_train2_temporal6s_final/scored_summary.json">周期试验</a></p>
<p>参考：<a href="https://arxiv.org/abs/2603.08386">DDHF逐像素周期线索</a>、
<a href="https://stonesoup.readthedocs.io/en/latest/stonesoup.models.transition.html">Singer条件运动模型</a>。
无向角回归、方向积分和时序支持属于已有数学/工程方法，本实现不宣称通用原理首创。</p></html>'''
    (OUT/'index.html').write_text(page,'utf-8')
    cp=ROOT/'output/checkpoints/latest_handoff.json';h=read(cp);h.update(updated_utc=datetime.now(timezone.utc).isoformat(),current_running_jobs=[],
        current_work_state='Headless disk-normal forecast implemented and promoted in target candidate; weak-period support remains experimental; 30px target incomplete')
    h['disk_motion_30px']=result
    h['next_stage_target'].update(mean_error_px=30.,horizon_s=.4,report=str(OUT/'index.html'))
    cp.write_text(json.dumps(h,ensure_ascii=False,indent=2),'utf-8')
    log=ROOT/'WORKLOG.md';text=log.read_text('utf-8');heading='\n\n## 2026-10-28 圆饼无机头修正、周期支持与未来法线\n'
    entry=heading+'''- 用户指出编号与机头朝向不决定运动，已纠正：同型全桨合力对编号置换不变；漏/重计与缺测集合有区别。新分支不使用桨盘编号或机头，保留平面法线。
- normal_turn_forecast：模pi两倍角展开、加权角速趋势与不确定度收缩，解析积分只旋转已知输入，未知项不旋转；以当前图像速度/加速度确认平移，静止周期摆动91时刻不额外注入位移。
- 同冻结侧视26目标43.2745→42.7192/P9580.6014→77.5407；实际RAW同输入42.8050→42.4040/P9577.9994→77.0781。全240源175当前171有效，无四桨锁定仍能使用新方向预测。
- 已把新法线预测纳入工作区target-candidate，--no-normal-turn-forecast保留控制，原基线/发布alpha.1/alpha.2 ZIP未改。
- temporal_period_support：100ms不同native像素只一票，保留16票且当前>=4、至少2窗、环带抑制；当前频率取当前点，事件原时间不变。只平移已有频率点分配ROI，不重喂运动补偿事件制造假周期。
- 另一录像实际180源111当前107有效，恢复4合格周期/3当前输入/7方向校正；同输入62.6464→周期62.6780→联合62.6796，无精度收益，保持显式实验而非默认。
- 本次train2读取43.432899s阻塞3019.93ms，生产原生解码3048.90ms；完整计时28.9513FPS，不删除慢帧，不把阻塞确定归因硬件/新算法。
- 首次RAW可选line_pca_disagreement_deg=None崩溃已修，反例补充，失败目录保留；之后实测完成。新方向尚无独立姿态/推力真值。
- 输出集中output/disk_motion_30px_20261028，docs/DISK_NORMAL_FORECAST.md；下一步周期连续性/可靠性与读取阻塞，30px/稳定低延迟30FPS仍未完成。
- 导出器已列入新模块依赖；未来发布须另升版本并更新模板，不覆盖已上传ZIP。历史报告脚本WORKLOG更新不得截断后续阶段。
'''
    if heading in text:
        i=text.index(heading);j=text.find('\n\n## ',i+len(heading))
        text=text[:i]+entry+(text[j:] if j!=-1 else '')
    else:text+=entry
    log.write_text(text,'utf-8')
    print(json.dumps(dict(report=str(OUT/'index.html'),side_fixed4s_FPS=float(fixed_fps),side_mean_px=42.40399726165323,train2_mean_px=62.67962332391339,goal_complete=False),ensure_ascii=False))
if __name__=='__main__':main()
