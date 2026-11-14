"""One compact report for the 30px rotor-force stage; no data/code publication."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,html,shutil,collections
ROOT=Path(__file__).resolve().parents[2];CODE=ROOT/'event_drone_forecast';OUT=ROOT/'output/rotor_force_30px_20261028'
def read(p):return json.loads(p.read_text('utf-8'))
def main():
    replay=read(OUT/'frozen_input_exact_fallback/summary.json')
    controls=read(OUT/('controls_final' if (OUT/'controls_final/summary.json').exists() else 'controls')/'summary.json')
    raw=read(OUT/'RAW_clear2s/summary.json')
    extra=None
    if (OUT/'RAW_train2_allpixels6s/summary.json').exists():
        path=OUT/'RAW_train2_allpixels6s'
        trial=[json.loads(s) for s in (path/'predictions.jsonl').read_text('utf-8').splitlines()]
        sys_path=str(CODE/'tools')
        import sys
        if sys_path not in sys.path:sys.path.insert(0,sys_path)
        from compare_final_pipeline import score
        truth=sorted(read(ROOT/'output/dense_labels_train2_40_46/labels.json')['records'].values(),key=lambda r:r['t'])
        scored=score(trial,truth);scored.pop('anchors',None)
        times=[];last=None
        from compare_final_pipeline import reference_at
        import numpy as np
        for r in trial:
            slot=int(np.floor(r['t']*5+1e-8))
            if not r['valid'] or slot==last:continue
            last=slot;target=reference_at(truth,r['t']+.4)
            if target is not None:times.append(float(np.linalg.norm(np.asarray(r['physical']['0.4'])-target)))
        summary=read(path/'summary.json')
        extra=dict(scores=scored,within30_fraction=float(np.mean(np.asarray(times)<30)),
            qualified_period_ticks=sum(r['period'].get('qualified',False) for r in trial),
            maximum_frequency_pixels=max(r['period'].get('pixels',0) for r in trial),
            maximum_mode_pixels=max(r['period'].get('mode_pixels',0) for r in trial),
            source_frames=len(trial),current_observations=summary['fresh_image_observations'],valid_forecasts=summary['valid_forecasts'],
            capacity_FPS=summary['steady_capacity_fps'],current_Hz=summary['paced_fresh_observation_rate'],
            period_all_pixels=True,conclusion='All pixels recover up to15 frequency pixels but at most8 in one mode, below the unchanged16-vote gate. Sampling alone does not fix the missing qualified input; motion/contrast/observer assumptions remain unresolved.')
        (path/'scored_summary.json').write_text(json.dumps(extra,indent=2),'utf-8')
    rows=[json.loads(s) for s in (OUT/'RAW_clear2s/predictions.jsonl').read_text('utf-8').splitlines()]
    usage=collections.Counter(r['period_selection']['source'] for r in rows)
    accepted=collections.Counter(r['period_selection']['source'] for r in rows if r['period_selection'].get('rotor_bank_used'))
    trace=[]
    for r in rows:
        d=r['period_selection']
        if d.get('rotor_bank_used'):
            trace.append(dict(t=r['t'],mode=r['structure']['mode'],current_ids=d['current_model_ids'],
                inferred_ids=d['hidden_model_ids'],f2=d['individual_f2_estimates'],relative=d['relative_f2'],
                learning_current=d['learning_input_current']))
    (OUT/'RAW_clear2s/individual_trace.json').write_text(json.dumps(trace,indent=2),'utf-8')
    sources={}
    for name in ['propeller_physics.py','rotor_thrust_bank.py','axis_amplitude_bridge.py','live_pixel_axis.py','run_fast_structure.py',
        'tools/check_rotor_force_model.py','tools/replay_rotor_thrust_bank.py','docs/TARGET_30PX.md']:
        b=(CODE/name).read_bytes();p=OUT/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
        sources[name]=hashlib.sha256(b).hexdigest()
    for name,expected in [('github_release_final_20261027/event-drone-forecast-v0.4.0-alpha.1.zip','ff1172b3f2d2e1752d8ebc6d299c6d972adfaaa1776f8773240e44aa3eb58abb'),
        ('github_release_update_20261028/event-drone-forecast-v0.4.0-alpha.2.zip','6260e0919a8dea574e8c502a0f982b3ed5791d5fe8d159c6b2e740f889b97e45')]:
        assert hashlib.sha256((ROOT/'output'/name).read_bytes()).hexdigest()==expected
    result=dict(target_horizon_s=.4,target_mean_error_px=30.,target_processing_FPS=30.,goal_complete=False,
        scope='Existing exposed development clips; no independent RPM/thrust/metric pose truth',
        sources=sources,mechanism_controls={k:v for k,v in controls.items() if k!='records'},frozen_input=replay,
        all_pixel_period_trial=extra,
        current_RAW=dict(source_frames=raw['frames'],current_observations=raw['fresh_image_observations'],
            valid_forecasts=raw['valid_forecasts'],capacity_FPS=raw['steady_capacity_fps'],current_Hz=raw['paced_fresh_observation_rate'],
            median_service_ms=raw['timing_ms']['total_service_ms']['median'],p95_service_ms=raw['timing_ms']['total_service_ms']['p95'],
            first_result_after_s=raw['first_result_after_s'],selector_sources=dict(usage),bank_accepted=dict(accepted),
            independent_accuracy_scored=False,raw_unchanged=raw['raw_unchanged']),
        metric_force_inverse_connected_to_RAW=False,
        reason='RAW does not independently supply metric inertia, calibrated angular acceleration and rotor-arm geometry',
        next='Current overlap/period observation and stable rotor identity, then calibrated image-angular response; do not claim force recovery from instantaneous attitude.')
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    known=controls
    sections=[]
    for n,r in replay.items():
        scores=r.get('scores');e='未独立评分' if not scores else str(round(scores['candidate']['physical_future_error_px']['0.4']['mean'],4))+'px'
        sections.append('<tr><td>'+n+'</td><td>'+str(r['source_frames'])+'</td><td>'+html.escape(str(r['mapped_current_counts']))+'</td><td>'+e+'</td></tr>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>30px阶段：逐桨推力与缺桨约束</title>
<style>body{font:17px/1.8 system-ui;max-width:1100px;margin:35px auto;padding:0 24px;background:#f4f6fb;color:#19314b}table{border-collapse:collapse;background:white;width:100%}td,th{border:1px solid #cbd5e1;padding:10px;text-align:left}pre{white-space:pre-wrap;background:#e7edf5;padding:18px}.note{border-left:5px solid #d59d25;padding:12px 18px;background:#fff8e7}a{color:#145d9d}</style>
<h1>目标：0.4秒平均误差低于30px，保持30FPS方向</h1>
<p class="note">本轮完成逐桨独立频率平方状态、带数据秩的缺桨约束求解以及实际RAW接线。30px目标尚未达到。缺桨力矩反解为已知参数接口，未把二维姿态角冒充真实力矩。</p>
<h2>推力模型</h2><pre>T_i = ρ n_i² D_i⁴ C_Ti(J, Re, M; geometry)
T = B ∫ ½ ρ W² c(r) [C_L cosφ − C_D sinφ] dr
F = Σ T_i n_i
τ = Σ r_i × (T_i n_i) + 反扭矩</pre>
<p>平方转速是常用近似；变化的系数包含形状/来流影响。图像椭圆是扫掠桨盘，不能直接读出翼型与桨距。
标准平行桨轴合力沿共同法线，差异推力产生力矩。</p>
<h2>本轮实际改动</h2><p>保留四个不等频率平方状态，分别求和；三/两桨只更新对应可见槽位，缺失状态携带增长不确定性与120ms时限，不赋成可见均值。
部分推断不刷新完整幅值年龄、不进入增益或已知输入积分学习。
新选项<code>--rotor-thrust-bank</code>，默认和发布ZIP保持冻结。</p>
<h2>同一冻结输入</h2><table><tr><th>录像</th><th>源时刻</th><th>已映射当前槽位数分布</th><th>0.4秒均值</th></tr>'''+''.join(sections)+'''</table>
<p>侧视与另一录像无已识别逐桨结构，沿用相同回退，误差仍43.2745/62.6464px。
清晰段17次四桨输入、6次部分推断来源，预测改变，但缺独立位置参考，不能称精度提升。
首个回放出现约0.02px的float32/64序列化差异，已用冻结原回退输入消除，不把它当新机制收益。</p>
<h2>实际RAW小试</h2><pre>'''+html.escape(json.dumps(result['current_RAW'],ensure_ascii=False,indent=2))+'''</pre>
<p>本次吞吐未达30；旧清晰短测32.94FPS，异步输入与运行抖动不同，不能把差异全部归因逐桨机制。
新状态机制冻结回放约0.1ms/时刻；主耗时仍是几何、周期、NN及解码等待。
RAW小试确认接线，不代表独立转速/推力正确率。</p>
<h2>姿态能反推多少</h2><p>单帧姿态给方向。连续姿态的角加速度结合惯量、力臂和外力矩才可构成力矩约束。
三已知推力不加约束秩3；加一条有效独立力矩可秩4。
两个已知加roll/pitch约束，在本次对角布局示例仍秩3，相邻布局可秩4。二维单角度通常不足。
非负最小二乘显式报告先验依赖，秩不足不冒充唯一测量。</p>
<h2>当前卡点</h2><p>侧视粘连/短弧没有稳定电机身份；另一录像无周期输入；椭圆信息不足以识别桨型、惯量和真实推力。
下一步优先局部周期分层、运动补偿的真实弧与稳定槽位，再学习有资格的角响应。</p>
<p><a href="summary.json">完整机器记录</a> · <a href="controls/summary.json">机制反例</a> · <a href="RAW_clear2s/individual_trace.json">实际逐桨状态</a> · <a href="source_snapshot/docs/TARGET_30PX.md">完整推导与下一步</a></p>
<p>主要资料：<a href="https://m-selig.ae.illinois.edu/props/propDB.html">UIUC实测库</a>、
<a href="https://www.apcprop.com/technical-information/file-downloads/">APC几何/性能</a>、
<a href="https://pybemt.readthedocs.io/en/latest/theory.html">pyBEMT理论</a>、
<a href="https://github.com/kegiljarhus/pyBEMT">pyBEMT源码</a>、
<a href="https://ardupilot.org/copter/docs/systemid-model-development.html">ArduPilot逐桨模型</a>。</p></html>'''
    if extra is not None:
        page=page.replace('</html>','<h2>全像素周期试验</h2><p>另一录像固定40–46秒，改用100%原始像素、保留全部原始时间；仍无合格周期，排除仅25%采样造成零输入的猜测。</p><pre>'+html.escape(json.dumps(extra,ensure_ascii=False,indent=2))+'</pre><p>该次protocol的periodic_sampling沿用旧描述，frequency_observer_all_pixels=true及实际源码确认全像素；此为描述性元数据错误，后续源码已修正，原协议与数值保留。</p></html>')
    (OUT/'index.html').write_text(page,'utf-8')
    cp=ROOT/'output/checkpoints/latest_handoff.json';handoff=read(cp);stamp=datetime.now(timezone.utc).isoformat()
    handoff.update(updated_utc=stamp,current_work_state='30px stage: per-rotor state and rank-aware force completion implemented; RAW smoke complete; target remains incomplete',current_running_jobs=[])
    handoff['next_stage_target']=dict(mean_error_px=30.,horizon_s=.4,processing_FPS=30.,report=str(OUT/'index.html'),scope='same fixed development clips; coverage/P95 and lag retained')
    handoff['rotor_force_30px']=result
    cp.write_text(json.dumps(handoff,ensure_ascii=False,indent=2),'utf-8')
    log_path=ROOT/'WORKLOG.md'
    log_text=log_path.read_text('utf-8');header='\n\n## 2026-10-28 新30px目标、逐桨推力与缺桨约束\n'
    import io
    with io.StringIO() as f:
        f.write('\n\n## 2026-10-28 新30px目标、逐桨推力与缺桨约束\n'
            '- 用户设定0.4秒30px阶段，沿用30FPS和原固定评分；目标仍未完成，旧goal工具返回usageLimited/50px旧描述，未伪造完成或擅自重建覆盖。\n'
            '- 在线查UIUC/APC、NPTEL叶素、pyBEMT、ArduPilot分配矩阵与Faessler阻力模型。平方律有效但CT随几何/来流，不能由椭圆认出桨距翼型。姿态瞬时不等于力矩。\n'
            '- 新propeller_physics已知参数CT/曲线/合力力矩和非负缺桨求解，报告不含先验的数据秩；真实RAW无惯量/力矩标定，未启用物理反解。\n'
            '- rotor_thrust_bank四独立f²状态，短时缺失不等于可见均值；120ms不确定性/年龄门槛；AxisAmplitudeBridge部分推断不续期或学习。新显式--rotor-thrust-bank，旧默认/两发布ZIP未改。\n'
            '- 机制反例通过：三无力矩rank3、有有效力矩rank4；两对角+rollpitch仍rank3、两相邻rank4；未标定方程拒绝；隐藏不伪装观测/不续期。均为受控机制，不是真实推力验证。\n'
            '- 冻结控制预测逐值复原。清晰17全桨/6部分来源；侧视0槽位、train2无周期，新分支没有输入，保持43.2745/62.6464。回放0.02px数值类型伪差已剔除，不能宣称30px或物理提升。\n'
            '- 实际E盘26–28秒RAW60源58当前54有效，逐桨来源见summary；计算26.8553FPS、当前25.0757Hz、首次36.22s。本次未达30FPS，不能沿用旧速度称达标。没有无意义重复测速。\n'
            '- 新结果集中output/rotor_force_30px_20261028；docs/TARGET_30PX.md。下一步检测周期分层/真实弧稳定身份和有效角响应，不在无物理信息段靠复杂公式强补力。\n')
        if extra is not None:
            f.write('- 全像素40–46秒RAW实际完成180源111当前107有效；最多15周期像素/8同频像素，未过原16门槛，仍0合格输入，0.4秒62.6464px/30px内4/15。处理52.1554FPS包含缺测，当前18.4637Hz。采样损失并非唯一原因。\n'
                '- 下一关键环节为逐桨差速力矩→未来角运动→未来法线→平移；当前RAW尚未完成此闭环。四桨全见同系数均值与求和仅差常数，不能把求和改名当精度提升。新状态保留差分信息、缺桨约束有资格门槛。\n')
        entry=f.getvalue()
    if header in log_text:
        first=log_text.index(header);end=log_text.find('\n\n## ',first+len(header))
        log_text=log_text[:first]+entry+(log_text[end:] if end!=-1 else '')
    else:log_text+=entry
    log_path.write_text(log_text,'utf-8')
    print(json.dumps(dict(report=str(OUT/'index.html'),current_RAW=result['current_RAW'],goal_complete=False),ensure_ascii=False))
if __name__=='__main__':main()
