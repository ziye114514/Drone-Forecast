"""Parallel CV/physics IMM report: sources, matched controls, modes and timing."""
from pathlib import Path
import html,json,sys
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;BASE=PROJECT/'output'
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage75'),str(ROOT/'research/stage74')]
import study75 as actual
import study74 as forecast
OUT=actual.OUT;load=actual.load;save=actual.save;sha=actual.sha;rows=actual.rows
def table(rr):
    if not rr:return '<p>无样本。</p>'
    cols=list(rr[0]);return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(round(r[k],3) if isinstance(r[k],float) else r[k]))+'</td>' for k in cols)+'</tr>' for r in rr)+'</table></div>'
def run():
    actual.verify();forecast.verify();checks=load(OUT/'checks75.json');analysis=load(forecast.OUT/'analysis.json');regimes=load(forecast.OUT/'regime_analysis74.json');protocol=load(forecast.OUT/'protocol.json')
    assert checks['parent']==1435 and checks['native10']==2870 and checks['video_frames_decoded']==1100
    selected=['cv','trusted_acc','increment_periodic','dual_cv_agile_zero','dual_cv_history','dual_cv_increment_constant','dual_cv_event_increment']
    main=[{k:r[k] for k in ['case','model','nominal_h','present_targets','valid_forecasts','mean_fde_px','joint_success']} for r in analysis['table'] if r['case']!='negative5' and r['model'] in selected and r['nominal_h']==.4]
    conditional=[r for r in regimes['sparse_box_errors'] if r['regime'] in ['steady_proxy','maneuver_proxy'] and r['nominal_h']==.4 and r['model'] in ['cv','trusted_acc','dual_cv_history','dual_cv_event_increment']]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    cases=['clear6','side6','train3'];fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    plot_models=['cv','trusted_acc','dual_cv_agile_zero','dual_cv_history','dual_cv_event_increment']
    for ax,case in zip(axes,cases):
        rr=[next(r for r in main if r['case']==case and r['model']==m) for m in plot_models];values=[r['mean_fde_px'] for r in rr]
        bars=ax.bar(np.arange(5),values,color=['#4492c6','#52677e','#aaaaaa','#7d8493','#d2a329']);ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=8)
        ax.set_xticks(np.arange(5),['Original CV','Strong history','CV/agile zero','CV/history','CV/event increment'],rotation=40,ha='right',fontsize=8)
        ax.set_title(f"{case} / n={rr[0]['valid_forecasts']}");ax.set_ylabel('0.4 s sparse-box endpoint error (px)');ax.set_ylim(0,max(values)*1.22);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Parallel hypotheses help some clips; event input must beat kinematic controls')
    fig.savefig(OUT/'parallel_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,7),layout='constrained');usage=[];timings=[];fixed=[]
    for ax,case in zip(axes,cases):
        rr=rows(OUT/case/'observations_forecasts.jsonl');tt=[r['t'] for r in rr]
        for m,color in [('dual_cv_event_force','#c36b6b'),('dual_cv_event_increment','#be922d'),('dual_cv_history','#627a99')]:
            weights=[r['frontends']['delayed_feedback']['dual_branch_diagnostic'][m]['mode_probability'] for r in rr]
            ax.plot(tt,[v[1] if v is not None else np.nan for v in weights],color=color,label=m,linewidth=1)
        ax.set_ylim(0,1);ax.set_ylabel(case+' / maneuver weight');ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper right')
        cfg=actual.CASES[case]
        # Fixed by the nearest future target to the clip midpoint, not by error.
        candidates=[d for d in analysis['details'] if d['case']==case and d['model']=='dual_cv_event_increment' and d['nominal_h']==.4 and d['forecast_valid'] and d['target_state']=='present']
        d=min(candidates,key=lambda d:abs(d['target_t']-(cfg['start']+cfg['duration']/2)))
        index=round((d['anchor_t']-cfg['start'])*50)-1;cap=cv2.VideoCapture(str(OUT/case/'forecasts.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,im=cap.read();cap.release();assert ok
        center=np.rint(d['target_center']).astype(int);cv2.drawMarker(im,tuple(center),(0,230,100),cv2.MARKER_CROSS,24,2)
        cv2.putText(im,f'Future sparse center at {d["target_t"]:.2f}s: evaluation only',(12,26),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1)
        name=f'{case}_parallel_fixed.png';cv2.imencode('.png',im)[1].tofile(str(OUT/name));fixed.append(dict(case=case,anchor_t=d['anchor_t'],target_t=d['target_t'],error_px=d['error_px'],file=name))
        c=load(OUT/case/'completion.json');f=load(forecast.OUT/case/'completion.json')
        timings.append(dict(case=case,full_pipeline_median_ms=c['processing_median_ms'],full_pipeline_p95_ms=c['processing_p95_ms'],fusion6_banks_median_ms=f['fusion6_banks_median_ms'],fusion6_banks_p95_ms=f['fusion6_banks_p95_ms']))
        for model in ['dual_cv_event_force','dual_cv_event_increment']:
            diag=[r['frontends']['delayed_feedback']['dual_branch_diagnostic'][model] for r in rr]
            usage.append(dict(case=case,model=model,current_physical_input_ticks=sum(d['physical_input_active'] for d in diag),
                mean_maneuver_weight=float(np.mean([d['mode_probability'][1] for d in diag if d['mode_probability'] is not None]))))
    axes[-1].set_xlabel('Source time (s)');fig.savefig(OUT/'mode_probabilities.png',dpi=160);plt.close(fig)
    videos=''.join('<article><h3>'+c+'</h3><video controls preload="metadata" src="'+c+'/forecasts.mp4"></video></article>' for c in cases)
    images=''.join('<figure><img src="'+r['file']+'"><figcaption>'+html.escape(str({k:v for k,v in r.items() if k!='file'}))+'</figcaption></figure>' for r in fixed)
    text='''<h1>框CV与事件物理分支：并行互补的第一次完整测试</h1><div class="lead"><b>双分支已接入真实RAW入口，原始CV始终保留；稳定高精度实时目标仍未达成。</b><p>清晰段融合比CV和强历史好；侧视比原CV好，但弱于强历史；3号录像融合变差。因此不能直接说“突然变向时一定更好”。默认入口保持，失败对照不删。</p></div>
<h2>相关论文与项目</h2><ul><li><a href="https://doi.org/10.1109/9.1299">Blom / Bar-Shalom 1988：IMM</a>——多个运动假设交互，使用新观测的似然更新模式概率。</li><li><a href="https://github.com/rlabbe/filterpy/tree/3b51149ebcff0401ff1e10bf08ffca7b6bbc4a33">FilterPy作者固定提交</a>——本轮实际运行其完整IMMEstimator；MIT许可和逐文件Git blob/SHA校验保留，未重新写IMM或安装依赖。</li><li><a href="https://arxiv.org/html/2603.01997v1">Event-Only Drone Trajectory Forecasting with RPM-Modulated Kalman Filtering</a>——转速调节CV过程噪声，与“运动学+桨叶信息互补”接近。论文位置输入来自真实框；本项目是自动检测框，不直接比较论文分数。未找到可确认的作者仓库链接。</li><li><a href="https://www.mathworks.com/help/fusion/ref/trackingimm.html">官方trackingIMM</a>——CV/CA/CT多模型参考；未运行MATLAB。本地Target_KF/旧14已有简化概率混合，当前升级为作者完整交互。</li></ul>
<h2>当前两条线怎样互补</h2><p>第一条是[x,y,vx,vy]匀速CV，只用框位置更新。第二条同样是4D状态，增加当前可信事件线索与上一时刻学到的推力系数，或“历史加速度+推力变化量”作为控制输入。两种假设各处理当前位置一次；IMM混合状态和协方差，并按预测与当前框的相符程度更新权重。两路不是独立传感器，不能把同源框当两份独立证据。</p><p>与之前0.2/0.4秒延迟评分门槛不同，当前IMM每20ms源时刻都能更新模式似然。但物理线索仍有160ms过去窗口、观测歧义及实际处理延迟；概率变化不等于真实姿态/机动识别。</p>
<h2>0.4秒未来位置结果：相同23秒录像、相同覆盖</h2><img src="parallel_comparison.png">'''+table(main)+'''
<p>同一目标比较，不因为变差就不输出。原33路保留，另加6路：高噪声无事件、历史加速度、恒定轴输入、变化推力、恒定增量与事件增量。高噪声无事件也能改善部分CV结果，必须把这部分与事件贡献分开。</p>
<h2>平稳与变向分开检查</h2><p>下面分组仅由过去0.16–0.24秒CV速度变化得到：转角≥25°或速度矢量变化≥150px/s为机动代理；转角≤10°且变化≤50px/s为平稳代理。它不是独立标注的突然变向真值。清晰机动分组0.4秒只有1个稀疏目标，不能据此证实一般规律。密集未来检测中心误差另存诊断文件，不当独立轨迹准确率。</p>'''+table(regimes['causal_regime_counts'])+table(conditional)+'''
<h2>已知机动的机制控制</h2><p>此控制使用已知合成轨迹、带噪位置和“正确/缺失/反号/延迟160ms”的加速度输入。正确输入是用于检验机制的理想已知线索，不是实采桨叶检测成绩。</p>'''+table(protocol['controls']['table'])+'''
<h2>模式权重、视频与固定画面</h2><img src="mode_probabilities.png">'''+table(usage)+'''<p>蓝色：原CV；紫色：原事件增量；黄色：CV/事件机动IMM融合。视频按源时间50fps播放；底部给出当前输入可用性与机动模式权重，不能由播放速度推断实时。</p><div class="grid">'''+videos+'''</div>'''+images+'''
<h2>延迟与真实性检查</h2>'''+table(timings)+'''<p>六组IMM的纯融合小试验约10–12ms，不代表端到端实时；当前检测、桨盘/方向和周期提取仍是主要耗时。要实现实时需要继续处理整条观测链及异步时间戳、过期控制，不能只把融合器运行快称实时。</p><p>75实际RAW共1435父/2870原生10ms窗口（1150主段+285重复前缀），1100视频帧全解码。所有当前输入精确73、39路预测/参数/协方差/模式概率精确74；两个实际RAW截断前缀通过。推理禁止旧特征缓存和未来标签；评分独立在预测保存后进行。两轮主段重叠，不重复算独立样本。</p>
<h2>接下来要解决什么</h2><ol><li>先解决3号和侧视物理输入的失效归因，区分错误方向、谐波/活动代理与框跳变；先不把IMM权重强推向物理。</li><li>补更密的变向段位置/机动标签，检验桨叶线索是否确实早于框速度变化；目前证据主要来自一个清晰机动目标与理想控制。</li><li>把观测链和融合输出的实际时间戳、陈旧程度、算力预算分开，之后才谈实时时延目标与默认升级。</li></ol>
<p><a href="../stage74_parallel_cv_physics_20261021/analysis.json">全部39模型未来评分</a> · <a href="../stage74_parallel_cv_physics_20261021/regime_analysis74.json">平稳/机动及密集检测一致性</a> · <a href="checks75.json">完整RAW与前缀检查</a> · <a href="protocol.json">冻结协议</a> · <a href="../stage73_temporal_extension_20261021/第六十八至七十三阶段事件推力预测报告.html">前一轮推力接线、失败与时间扩展</a></p>'''
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CV与物理双分支融合报告</title><style>body{font:16px/1.7 system-ui;background:#f2f5f8;color:#203044;max-width:1280px;margin:32px auto;padding:0 24px}h1{font-size:30px}h2{margin-top:36px}.lead,article,figure{background:white;padding:20px;border-radius:12px;margin:18px 0}.lead{border-left:6px solid #d2a329}img{max-width:100%;height:auto;background:white}video{width:100%;background:#111}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}.scroll{overflow:auto;margin:20px 0}table{border-collapse:collapse;background:white;font-size:13px;width:100%}th,td{border:1px solid #d4dde6;text-align:left;padding:7px;vertical-align:top}th{background:#e6edf5}a{color:#245e99}figcaption{font-size:13px;overflow-wrap:anywhere}</style>'+text+'</html>'
    report=OUT/'第七十四至七十五阶段双分支融合报告.html';report.write_text(page,encoding='utf-8')
    save(OUT/'delivery_data75.json',dict(main=main,conditional=conditional,timings=timings,usage=usage,fixed=fixed))
    save(OUT/'delivery75.json',dict(helper_sha256=sha(__file__),report=str(report),actual_RAW_39_completed=True,default_promoted=False,
        stable_high_accuracy_realtime_achieved=False,author_IMM_commit=forecast.snapshot['commit']))
    print('74–75 dual-branch report delivered',report,flush=True)
if __name__=='__main__':run()
