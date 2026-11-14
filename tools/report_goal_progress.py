"""Persist50px/30FPS progress, failures and exact continuation state."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,re,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;OUT=WORK/'output/goal50px30fps_20261027'


def main():
    data=json.loads((OUT/'goal_results.json').read_text('utf-8'));cases=data['cases'];side=cases['side']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    old=side['scores']['original_force_same_observations']['physical_future_error_px']['0.4']['mean']
    new=side['scores']['new']['physical_future_error_px']['0.4']['mean']
    size=None;size_note=''
    if (OUT/'clear_size_comparison.json').is_file():
        size=json.loads((OUT/'clear_size_comparison.json').read_text('utf-8'))
        size_note=f'''<section><h2>后续小优化：尺寸必要条件提前判断</h2><p>保持原“最大桨盘的1/2.5”规则，在单个conic假设的尺寸已确定时，提前结束必定被丢弃的小轮廓，不再计算其残差/覆盖率/不确定度。使用原同步NN、原力学配置的独立清晰段对照，60个时刻的框/几何/周期/预测逐项完全相同，完整处理能力{size['summaries']['reference']['steady_capacity_fps']:.2f}→{size['summaries']['optimized']['steady_capacity_fps']:.2f}FPS。仅2秒短测，可能包含系统抖动，不把它当稳定加速比例。</p><p>固定部分遮挡窗口重复3次，不确定度计算21→12次，组件中位14.24→13.30ms，所有输出精确相同；另两窗口调用数未减少，其时间波动不归功于尺寸剪枝。本项已保留到等价计算路径，未因收益小就删除。上表清晰19.17FPS使用目标异步预设，配置不同，不能与这组数字直接相减。</p><p><a href="clear_size_comparison.json">完整同步RAW同输出对照</a> · <a href="size_prune_isolated/summary.json">单独隔离尺寸剪枝</a></p></section>'''
    axes[0].bar([0,1],[old,new],color=['#8995a7','#167b91'])
    axes[0].axhline(50,color='#bd593b',ls='--',label='50px mean target')
    axes[0].set(xticks=[0,1],xticklabels=['Original force','Recent residual + damping'],ylabel='Mean0.4s error (px)',title='Same new side observations,18 targets',ylim=(0,85))
    for x,y in enumerate([old,new]):axes[0].text(x,y+1,f'{y:.2f}',ha='center')
    axes[0].legend(fontsize=9)
    names=['side','clear','train2'];x=np.arange(3);w=.35
    for shift,key,label,color in [(-w/2,'steady_capacity_fps','Full pipeline capacity','#167b91'),(w/2,'paced_fresh_observation_rate','Current observations per second','#d28a45')]:
        values=[cases[n]['summary'][key] for n in names]
        axes[1].bar(x+shift,values,w,label=label,color=color)
        for at,y in zip(x+shift,values):axes[1].text(at,y+.8,f'{y:.1f}',ha='center',fontsize=9)
    axes[1].axhline(30,color='#bd593b',ls='--');axes[1].set(xticks=x,xticklabels=['Side','Clear','Train2'],ylabel='FPS / observations per second',title='Capacity and availability are different',ylim=(0,85))
    axes[1].legend(fontsize=8,loc='upper left');fig.savefig(OUT/'progress.png',dpi=150);plt.close(fig)
    table=''
    for key,label in [('side','侧视110–114s'),('clear','清晰26–28s'),('train2','train2 40–46s'),('context','含过去0.5s突发，仍评分110–114s')]:
        r=cases[key];s=r['summary'];score=r.get('scores',{}).get('new',{}).get('physical_future_error_px',{}).get('0.4')
        error='无独立人工框' if score is None else f'{score["mean"]:.2f}px / {score["p95"]:.2f}px（{score["n"]}目标）'
        table+=f'<tr><td>{label}</td><td>{s["steady_capacity_fps"]:.2f}</td><td>{s["paced_fresh_observation_rate"]:.2f}</td><td>{error}</td></tr>'
    p95=side['scores']['new']['physical_future_error_px']['0.4']['p95'];within=100*side['within50_fraction']['new']
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>50px与30FPS目标优化</title><style>
body{{font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif;max-width:1140px;padding:0 20px;margin:32px auto;background:#f2f5f8;color:#233447}}section{{background:white;border-radius:10px;padding:22px 28px;margin:18px 0}}h1{{font-size:29px}}h2{{font-size:21px}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:9px;border-bottom:1px solid #dae2e9}}img{{max-width:100%}}a{{color:#176880}}.note{{background:#fff0d4;padding:12px}}code{{background:#edf1f6}}</style>
<h1>50px / 30FPS：已优化，尚未同时达标</h1><p>2026-10-27 · 用户指定目标 · 继续五圆、周期输入、未知合力反馈主线</p>
<section><h2>目标与当前结果</h2><p>先沿用此前口径：0.4秒预测平均二维欧氏误差&lt;50px，完整RAW流水线处理能力≥30FPS，同时看当前观测率、慢帧和积压。目标不是只有均值好看，更不是保证每个预测都低于50px。评分源时刻5Hz，推理源时刻30Hz，两个频率用途不同。</p><table><tr><th>实际RAW运行</th><th>完整处理能力FPS</th><th>当前观测Hz</th><th>0.4s平均 / P95误差</th></tr>{table}</table><p class="note">侧视计算能力超过30，但均值仍约52px；清晰段仍约19FPS；train2算得快却看不连续。不能把不同片段的最好误差与最好FPS拼成达标。目标保持进行中。</p><img src="progress.png" alt="同观测预测误差与完整流水线速度"></section>
<section><h2>这轮具体改了什么</h2><ol><li><b>侧视按当前轴线调度：</b>上一时刻轴线合格时，当前帧仍从原生20ms密度验证同一PCA/LSD规则；每3帧完整重搜桨盘，当前线不合格立即回完整路径。中间帧的电机数量明确“未重新判定”，不把没算的桨叶说成遮挡，不刷新四桨姿态资格。</li><li><b>未知合力跟近期偏差：</b>原增益仍慢学，d用已成熟0.2s积分残差的近期加权估计，更快修正；保持原p/v及当前事件输入。</li><li><b>未来净加速度不假定永远不变：</b>采用已有Singer模型条件均值形式，固定相关时间0.35s沿用项目原时间尺度。它是未观测未来控制量的预测先验，不是重力会衰减，更不是从RAW测到了风或空气阻力系数。</li><li><b>事件突发不再导致退出：</b>超过原200万事件共享槽时，通过仍有界的不可变消息传递整包，保留全部事件。真实过去0.5s触发6次溢出，最大2193664事件/窗；修复完整跑完，但确实拖慢，耗时如实保留。</li></ol><p>当前候选入口：<code>python run_fast_structure.py --raw &lt;event.raw&gt; --output &lt;新目录&gt; --start 110 --duration 4 --goal-profile --paced --verify-frames</code>。候选不是达标保证；旧默认力学模型、原计算对照和未通过实验都保留，未覆盖已发布ZIP。</p><p>衰减外推来源见<a href="https://stonesoup.readthedocs.io/en/latest/stonesoup.models.transition.html#stonesoup.models.transition.linear.Singer">Stone Soup官方Singer说明</a>，只参考数学形式，没有复制库源码，也不作为项目原创方法。</p></section>
<section><h2>精度收益拆开计算</h2><p>旧4秒RAW为73.22px；新整条RAW为52.01px，但异步回包与当前观测略有差别，不能全部归因给力学模型。将旧力学模型重放在<b>同一套新当前观测</b>上，平均65.89→52.01px，才是这套输入上的力学改动收益。</p><p>同18个目标，新P95为{p95:.2f}px，仅{within:.1f}%在50px内；若要求“几乎每次都&lt;50px”，距离还很明显。另一个既有3秒保存流，同13目标52.72→38.35px，属于缓存因果输入开发对照，不附会成新的独立RAW测速。</p><p>全部因果输出先保存，再打开既有人工框评分；相邻人工点只允许≤150ms插值。没有独立推力/RPM/姿态真值。本轮在已暴露开发数据上尝试了若干机制，不能宣称未见数据泛化或新测试集成绩。</p></section>
<section><h2>失败尝试与跨片段问题</h2><p>周期/几何线程重叠保留最终输出完全相同，但侧视短测17.53→14.77FPS，未启用；其生产者等待很短，线程/GIL竞争未带来收益。增益2σ门槛本批仍不改变输出。响应标量在线校准、位置状态力反馈、速度创新残差、局部力平衡拟合均保留；有中位或P95局部收益，但没有比当前候选更好的平均误差，因此不默认叠加。</p><p>train2原候选0个当前合格力输入，故新旧力学同67.97px。这是真实观测不足，不是把未知力学习快些就一定解决。接上项目已有Ultralytics ByteTrack，只返回当前原始NMS框、不把外推当检测；当前整机107→103/180，有效预测103→96，评分目标也不同，原始均值63.89不能直接当全面改善。该支保留可选，不纳入目标预设。</p><p>额外0.5s过去上下文没有删110–114s评分点，旧模型也获得同一历史。突发段拖慢至17.33FPS，新旧同观测80.95→63.04px，反而暴露了强闪烁/事件风暴问题。失败路径、溢出和完整记录保留。</p></section>
<section><h2>下一步优先级</h2><ol><li><b>清晰四桨共享模板的多速率更新：</b>重建/形状更新慢速运行，当前刚体/部分方向快速更新；区分“少算”与“少看见”，配合已有逐槽周期归一化，避免可见集合/计算集合改变制造假推力。</li><li><b>强事件突发：</b>优先原生C++/有界缓冲减少复制与阻塞，评估邻域/全局照明抑制；不能直接按最高密度删事件，因为桨叶也高密度。当前修的是完整性，30FPS突发承载还没解决。</li><li><b>变向段的未知力与力输入质量：</b>近期残差已有效，但剩余误差集中在快速加减速和周期缺失；要约束有偏的频率/轴线输入，保留少量无偏当前观测。停止为跨50px而继续微调时间常数。</li><li><b>整机连续观测：</b>train2低召回需要面向ON/OFF数据的检测器/弱目标接续改进，现有ByteTrack没有解决；不能用重复预测补足“30次新检测”。</li></ol></section>
{size_note}<section><h2>代码、评分与交接</h2><p><a href="goal_results.json">各运行完整指标和同目标误差</a> · <a href="local_force_balance/summary.json">全部固定机制开发对照</a> · <a href="force_state_feedback/summary.json">位置状态力反馈</a> · <a href="parallel_equivalence.json">并行路径逐项相同与速度失败</a> · <a href="mechanism_checks/summary.json">合力/滤波/当前轴资格检查</a> · <a href="../../event_drone_forecast/docs/GOAL_50PX_30FPS.md">入口与口径</a></p><p>各实际RAW源快照保留，原位置7文件与已发布ZIP哈希不变。目标及下一步已写入WORKLOG、latest_handoff；没有后台作业。未改yolov10保护目录。</p></section></html>'''
    (OUT/'index.html').write_text(report,'utf-8')
    local=[p for p in re.findall(r'(?:href|src)="([^"#]+)"',report) if '://' not in p]
    assert all((OUT/p).is_file() for p in local)
    frozen=WORK/'output/github_release_final_20261027';archive=frozen/'event-drone-forecast-v0.4.0-alpha.1.zip'
    digest=hashlib.sha256(archive.read_bytes()).hexdigest();assert digest=='ff1172b3f2d2e1752d8ebc6d299c6d972adfaaa1776f8773240e44aa3eb58abb'
    old=frozen/'event-drone-forecast/src/event_drone_forecast/position'
    same=[p.name for p in old.glob('*.py') if p.read_bytes()==(ROOT/'src/event_drone_forecast/position'/p.name).read_bytes()];assert len(same)==7
    snapshot=OUT/'final_source_snapshot';snapshot.mkdir(exist_ok=True);hashes={}
    paths=['run_fast_structure.py','fast_raw_reader.py','fast_structure_front.py','fast_geometry_compute.py','responsive_force.py','force_position_state.py',
        'tools/probe_goal_force.py','tools/probe_force_state_feedback.py','tools/check_goal_mechanisms.py','tools/score_goal_runs.py',
        'tools/report_goal_progress.py','tools/prepare_github_release.py','tools/bench_size_prune.py']
    for name in paths:
        content=(ROOT/name).read_bytes();ast.parse(content,filename=name);p=snapshot/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
        hashes[name]=hashlib.sha256(content).hexdigest()
    checks=dict(position_core_unchanged=same,frozen_public_ZIP_sha256=digest,local_links_checked=len(local),source_hashes=hashes,
        both_goals_achieved=False,all_RAW_runs_preserved=True,current_jobs=[])
    (OUT/'artifact_checks.json').write_text(json.dumps(checks,indent=2),'utf-8')
    checkpoint=WORK/'output/checkpoints/latest_handoff.json';handoff=json.loads(checkpoint.read_text('utf-8'))
    handoff.update(updated_utc=datetime.now(timezone.utc).isoformat(),current_work_state='Completed50px/30FPS goal optimization batch; goals not jointly achieved; no jobs',
        current_jobs=[],current_running_jobs=[],execution_status='Goal active; current batch complete, no jobs',latest_research_report=str(OUT/'index.html'))
    handoff['goal50px30fps']=dict(report=str(OUT/'index.html'),target=data['goal'],achieved=False,
        preset='--goal-profile',baseline_unchanged=True,side_current_fps=33.5972733247193,side_current_observation_Hz=28.13352789784628,
        side_mean_error_px=52.008933007956685,side_same_inputs_original_error_px=65.88670020134546,
        side_P95_px=p95,side_within50_fraction=side['within50_fraction']['new'],clear_capacity_FPS=19.166275463359913,
        train2_force_inputs=0,train2_error_px=67.9708829713479,burst_overflow_packets=6,
        burst_capacity_FPS=17.32640465078126,burst_error_px=63.041087444367314,
        unadopted=['parallel-period slowdown','scalar response calibration','force state feedback','velocity residual','window force balance','ByteTrack lower continuity'],
        next='Clear-view multi-rate shared5circle template/current constraints plus slot-normalized period, distinguish scheduling vs visibility; burst copy/decoder bottleneck; unbiased event-force input at maneuvers; train2 ON/OFF body recall. No further minor time-constant sweep on exposed labels.')
    if size:
        handoff['goal50px30fps']['size_prune']=dict(raw_report=str(OUT/'clear_size_comparison.json'),
            exact_same_sync_outputs=60,reference_optimized_FPS=[size['summaries'][k]['steady_capacity_fps'] for k in ['reference','optimized']],
            partial_component_uncertainty_calls_over3=[21,12],new_goal_profile_RAW_not_repeated_after_equivalent_change=True)
    checkpoint.write_text(json.dumps(handoff,ensure_ascii=False,indent=2),'utf-8')
    marker='## 2026-10-27 用户50px与30FPS目标优化'
    worklog=WORK/'WORKLOG.md'
    if marker not in worklog.read_text('utf-8'):
        with worklog.open('a',encoding='utf-8') as f:
            f.write('\n\n'+marker+'\n\n'
                '- 用户设定0.4s误差50px内、30FPS目标，已建立active goal；当前沿用平均欧氏误差口径，同时报告P95/50px内比例/当前观测率/积压，不保证每帧50px。目标未完成。\n'
                '- 新side-axis-budget：当前同PCA/LSD轴资格每tick，完整conic每3ticks，资格失效即回全路径；计数标为未重新判定，不伪造隐藏桨盘或刷新完整姿态。--responsive-force用成熟积分残差近期权重更新d，慢增益保持，并用Singer净加速度条件均值外推（相关时间.35沿用原模型；非重力衰减，不称原创）。\n'
                '- side110–114s RAW：120源/113当前/109有效；33.5973FPS，当前28.1335Hz，总服务中位23.20ms/P9573.36ms，末10延迟16–41ms，初始慢帧仍有积压。0.4s18目标新52.0089px/P9595.8271，50px内8/18；同一新观测原力学65.8867px→52.0089px。旧另一套输入73.216不可全部归因模型。缓存另一3s同13目标52.7185→38.3501，不是新独立RAW计时。\n'
                '- clear26–28s RAW保留four16/three2/partial_two4，但19.1663FPS、当前17.7371Hz，无独立人工框；未达30。train2实际路径E:/dataset/train/2/aligned/event.raw（先误用media路径失败保留），180源/107当前/103有效，70.0753处理能力但当前17.8026Hz，force输入0，误差67.9709px。\n'
                '- ByteTrack复用已有BodyAssociation，仅当前原NMS弱框，未用tracker外推当检测；同train2当前107→103、有效103→96，原始均值63.8942但目标集合变化，不能当全面改善，保留可选不入goal预设。\n'
                '- 过去.5s初始化没有改110–114评分范围，先触发共享槽溢出；修fast_raw_reader保留超200万事件整包、仍有界2消息，无截断。真实135源最大2193664事件/窗、6次overflow，17.3264FPS，当前14.1016Hz；同观测旧80.9522→新63.0411px，未达标且保留失败。\n'
                '- 并行周期/几何最终输出60时刻逐项同，但短测17.53→14.77FPS，不采用。额外标量校准、力反馈KF、速度创新残差、局部积分力平衡固定机制试验均保存，均未优于当前候选均值；不是扫时间常数。常加速度受控误差<1px/s²、恒定输入不认定增益、零控制KF精确一致、当前侧轴不虚构桨盘/中心检查通过。\n'
                '- 候选--goal-profile启用async/prefilter/side-axis/responsive，原默认力模型与--reference-compute保留。原位置7文件/已发布ZIP哈希不变，未改保护目录。报告output/goal50px30fps_20261027/index.html；源码快照、评分、WORKLOG/latest_handoff已保存，无后台作业。下一步清晰段共享五圆多速率+逐槽周期归一化、事件突发复制阻塞、变向输入质量和train2整机召回；不要为跨50px继续微调常数。\n')
    print(json.dumps(dict(report=str(OUT/'index.html'),goals_achieved=False,side_fps=side['summary']['steady_capacity_fps'],side_mean=new,source_files=len(hashes)),ensure_ascii=False))


if __name__=='__main__':main()
