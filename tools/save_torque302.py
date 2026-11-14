from pathlib import Path
import json,datetime
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent

if __name__=='__main__':
    report=WORK/'output/torque302_20261109/index.html';doc=ROOT/'docs/HANDOFF_20261109_TORQUE302.md'
    assert report.exists() and doc.exists()
    log=WORK/'WORKLOG.md';marker='## 2026-11-09 至302：不等逐桨推力、力矩交接与未来法线'
    assert marker not in log.read_text('utf8')
    text=f'''

{marker}

用户要求攻最大剩余卡点、强调每桨推力不等。核实原版早已分别累计f_i²，缺失的是空间分布→力矩→未来姿态→未来合力方向。已新增匿名空间力矩代理、可见集合变化的未知角响应交接、圆饼惯量/现有图像力增益先验的角观察器，并以完全相同无桨输入的角观察器作差，接未来法线Singer积分。隐藏推力归未知合力，隐藏力矩另归角状态，不混单位；不需要机头或物理电机编号。

公共RAW显式新增--rotor-torque-innovation，默认关，复用当前原生逐桨周期，无新增NN/CPU作者相位/事件副本/几何搜索。三段各181包双臂实际GPU工厂已核验：旧预测最大差2.51e-12px、当前p/v/a差2.01e-11。0.4s固定25查询（16可评分22）：13/15/16基线37.516/48.476/44.851，原生候选37.516/48.470/44.851；CPU旧联合相位输入37.521/48.366/44.847。仅微小数值变化，参考弱、无独立RPM/姿态/COM真值，不能称实际显著增益。组合289＋原生为34.782/48.251/43.504，大头来自旧289，严禁归功本轮。目标30px/30Hz未达，未默认替换。

直接证据：原生有逐桨周期仅0/16/3包，同源差动变化仅0/1/0。角力矩源集合交接将15连续4→20帧，16为9→12。298过去频率只作当前早半窗拟合种子、晚半窗独立验证，15/16 ≥2环34→37、39→41，16连续23帧；增益信噪代理1.488/1.927未可靠辨识，CPU1.59/1.78s/包，接预测仍未胜旧相位输入，不能接实时默认。295分极性也无额外覆盖。没有改阈值/查询、没有按录像调γ。

301修正300零输入也报applied的元信息，预测与300差<1e-7；缓存只读小状态矩阵，没有再造精度收益。图文output/torque302_20261109/index.html，详细交接event_drone_forecast/docs/HANDOFF_20261109_TORQUE302.md。下一单项仍为持续空间归属明确的逐桨周期变化/相位状态，侧视不能硬套完整环9扇区；不得重复调框、固定tau、NN或盲目放大差动系数。

本批所有CPU/GPU作业已收尾，无后台任务。无新安装/训练/上传，保护目录、SDK/RAW/权重、旧NN发布包未改。研究目标未完成，没有额度耗尽的工具证据。
'''
    with log.open('a',encoding='utf8') as f:f.write(text)
    readme=ROOT/'README.md';s=readme.read_text('utf8');block='''
<!-- torque302-current:start -->
> **2026-11-09：不等逐桨推力已接力矩/未来法线，收益仍弱。** 新可选RAW参数 `--rotor-torque-innovation`，默认关；逐桨f²保持不同，缺桨交接未知项。实际原生15号48.476→48.470px，另两段不变；CPU联合相位15号约−0.110px，尚无显著普适收益。原生同源差动仅0/1/0包，角响应尚未可靠辨识。旧基线核验保持，30px/30Hz未达。先读[当前交接](docs/HANDOFF_20261109_TORQUE302.md)和[图文结果](../output/torque302_20261109/index.html)。下方“当前”为历史批次。
<!-- torque302-current:end -->
'''
    assert 'torque302-current:start' not in s
    title,rest=s.split('\n',1);readme.write_text(title+'\n'+block+rest,'utf8')
    checkpoint=WORK/'output/checkpoints/latest_handoff.json';c=json.loads(checkpoint.read_text('utf8'))
    old=c.get('latest_report')
    c.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),updated_local_date='2026-11-09',
        execution_status='Batch302 finished; all jobs closed; differential torque optional RAW entry verified. Objective incomplete.',
        research_objective_status='incomplete',last_completed_stage=302,latest_numbered_stage=302,
        latest_completed_stages=list(range(294,303)),in_progress=False,experiments_running=[],running_jobs=[],running_RAW_sessions=[],
        pending_runs=[],default_promoted=False,stable_high_accuracy_realtime_achieved=False,realtime_proven=False,
        latest_report=str(report),current_report=str(report),latest_research_report=str(report),previous_report=old,
        latest_handoff_document=str(doc),latest_handoff=str(doc),current_status_document=str(doc),current_findings_document=str(doc),
        latest_implementation_document=str(doc),handoff_priority_document=str(doc),latest_named_report_is_authoritative=True,
        older_numbered_stage_fields_are_historical=True,local_report_links_checked=True,
        active_batch='torque302_20261109',last_completed_named_batch='torque302_20261109',
        next_action='Continuous spatially identified per-source phase/frequency changes; visibility-partition torque transfer retained. Side line support must not require complete-ring sectors. No per-recording gamma tuning or more frame-centre/FPS/NN substitutes.',
        latest_stage302=dict(report=str(report),handoff=str(doc),RAW_option='--rotor-torque-innovation',default_enabled=False,
            baseline_same_current_state_verified=True,meaningful_accuracy_gain_proven=False,
            native_measured_drive_change_packets={'13':0,'15':1,'16':0},
            objective30px30Hz_met=False,source_time_only=True,no_new_raw_fps_claim=True,
            unequal_CT_inertia_or_true_RPM_identified=False,all_jobs_closed=True),
        GPU_resource_constraint='User authorized/free; actual native GPU verification completed; no running load now.')
    c['completed']=sorted(set(c.get('completed',[]))|set(range(294,303)))
    checkpoint.write_text(json.dumps(c,ensure_ascii=False,indent=2),'utf8')
    print(str(doc),flush=True)
