"""Final checkpoint before user-authorized quota pause; no experiment runs."""
from pathlib import Path
import datetime,json,re
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
p=WORK/'output/checkpoints/latest_handoff.json';h=json.loads(p.read_text('utf-8'))
assert h['quota_policy']['stop_at_remaining_percent']==10
h['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
h['execution_status']='Checkpoint saved; pausing at user quota reserve with1% handoff headroom; no running experiments'
h['goal_state']='user_quota_pause_requested; authoritative status is goal tool'
h['experiments_running']=False;h['running_jobs']=[]
h['quota_policy'].update(last_observed_primary_used_percent=89,last_observed_primary_remaining_percent=11,
    last_observed_weekly_used_percent=25,handoff_headroom_percent=1,
    reason_for_early_wind_down='Preserve requested10% and cover final checkpoint/accounting granularity',
    resume_requires_scheduled_check=True)
h['next_action']='At scheduled2026-10-26 01:08:30 Asia/Shanghai, verify usage has reset and resume per user authorization. Read docs/NEXT_PRECISION_EXPERIMENT.md. Do not repeat failed Q-adaptation sweeps or packaging work.'
h['next_precision_design']=str(ROOT/'docs/NEXT_PRECISION_EXPERIMENT.md')
h['known_terminal_exec_sessions']=[98684,16701,27059,78085,52003]
h['latest_velocity_probe_20261025']['current_center_error_side']=[5.162593651416255,6.1340075519998924]
h['latest_velocity_probe_20261025']['velocity_only_ablation0p4_side']=64.2477822648597
h['latest_velocity_probe_20261025']['candidate_is_not_in_default_or_source_bundle']=True
p.write_text(json.dumps(h,ensure_ascii=False,indent=2),encoding='utf-8')

main=WORK/'output/optimization_run_20261025';p=main/'summary.json';s=json.loads(p.read_text('utf-8'))
s['quota_policy']=h['quota_policy'];s['status']='user quota checkpoint; pausing with1% reserve headroom'
s['next_action']=h['next_action'];p.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
p=main/'index.html';s=p.read_text('utf-8')
banner='''<section id="gaming-pause-checkpoint"><h2>当前：按用户额度要求收尾暂停</h2>
<p>最后读数短周期剩余约11%，为保留10%并覆盖交接开销，不再启动新实验。已保存代码、正负结果与交接，无运行中的作业。一次恢复任务安排在北京时间10月26日01:08:30，先核对额度再继续。</p>
<p>最新自适应Q候选使两个片段预测变差，已拒绝，原默认与42文件运行候选保留。下一次从速度响应入手，待做方案不算已有改进。</p>
<p><a href="../live_innovation_covariance_20261025/index.html">最新机制负结果</a> · <a href="../live_startup_transport_20261025/index.html">实际RAW接线及预测视频</a> · <a href="../../WORKLOG.md">完整交接</a> · <a href="../../event_drone_forecast/docs/NEXT_PRECISION_EXPERIMENT.md">下一次具体方案</a></p></section>'''
s,n=re.subn(r'<section id="gaming-pause-checkpoint">.*?</section>',lambda _:banner,s,count=1,flags=re.S);assert n==1
p.write_text(s,encoding='utf-8')

marker='### 用户额度保留线：收尾暂停，01:08:30恢复（2026-10-25）'
log=WORK/'WORKLOG.md'
if marker not in log.read_text('utf-8'):
    with log.open('a',encoding='utf-8') as f:
        f.write('\n\n'+marker+'\n\n')
        f.write('用户要求剩余10%先暂停、到01:08补充后恢复。最后账户读数短周期已用89%（约余11%），周已用25%；为保留10%及交接开销不再开新实验，随后调用update_goal paused。最终目标工具状态才是暂停状态依据，不标完成，不使用重置券。\n\n')
        f.write('所有本轮实际RAW exec会话98684/16701/27059/78085/52003均已返回exit0；速度候选CPU机制与归因工具也已终止，当前无作业。游戏期间暂停已由用户“游戏关掉了”撤销，本次是额度暂停。\n\n')
        f.write('已创建唯一thread heartbeat：automation，ACTIVE，2026-10-26 01:08:30 Asia/Shanghai（预计账户01:08:17重置），一次COUNT1，届时先验额度、恢复本项目，执行后暂停自身。保持电脑/app/E盘可用。\n\n')
        f.write('保留：独立发布器、包内位置核心、可选共享帧/启动同步、42文件窄范围RAW跑通；不提升默认或宣称新精度。拒绝：这轮自适应Q，side0.4s51.089→66.677、train2 73.856→81.700；只换速度side64.248而只换位置52.764，说明本候选主要伤在速度响应（含非线性外推作用）。原加速度与输出资格相同。\n\n')
        f.write('恢复先读docs/NEXT_PRECISION_EXPERIMENT.md：只考虑一个固定的当前速度机制，可复用现有常加速度状态过滤器，尚未实现/试验，不是创新。不要再扫失败Q门槛、扩展打包、重复无意义测速或混比较不同评分口径。D:\\yolov10-main和E盘同名目录仍禁止触碰。\n')

print(json.dumps(dict(checkpoint_saved=True,experiments_running=False,remaining_percent_last_observed=11,
    resume_local=h['quota_policy']['resume_local'],next_action=h['next_action']),ensure_ascii=False,indent=2))
