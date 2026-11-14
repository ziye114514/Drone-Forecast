"""Append completed optimization mechanisms and the explicitly running job."""
from pathlib import Path
from datetime import datetime,timezone
import json
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;OUTPUT=WORK/'output'
OUT=OUTPUT/'optimization_resume_20261026'
def read(name):return json.loads((OUTPUT/name).read_text('utf-8'))
data=dict(utc=datetime.now(timezone.utc).isoformat(),
    periodic_recovery=read('periodic_center_recovery_20261026/summary.json'),
    detector_window=read('detector_window_v2_20261026/summary.json'),
    direct_turn={k:v for k,v in read('live_turn_fit_v3_20261026/comparison.json').items() if k!='results'},
    periodic_packet={k:v for k,v in read('periodic_packet_20261026/comparison.json').items() if k!='rows'},
    periodic_packet_RAW=read('periodic_packet_RAW_20261026/completion.json'),
    ROI_refine=read('roi_refine_20261026/summary.json'),
    train_preparation={k:v for k,v in read('roi_augmented_detector_20261026/preparation.json').items() if k!='rows'},
    primary_remaining_last_read=43,weekly_remaining_last_read=66,
    controller_last_verified='paused; user asked to resume manually; authorized heartbeat work is running',
    running_job=dict(session_id=64989,kind='Single fixed10epoch crop-augmented detector training',
        source='tools/train_roi_adaptation.py',output='output/roi_augmented_detector_20261026',
        terminal_completion_not_yet_verified=True))
(OUT/'progress.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
p=OUTPUT/'checkpoints/latest_handoff.json';handoff=json.loads(p.read_text('utf-8'))
handoff.update(utc=data['utc'],experiments_running=True,
    in_progress='One fixed10epoch ROI-augmentation detector training, exec64989; poll before any restart.',
    next_action='Recover exec64989 or authoritative train_completion.json. If done, evaluate unchanged original test3/negative5 and fresh ROI input with selected checkpoint before any default promotion. Check both quota windows before next batch.',
    current_running_jobs=[data['running_job']],resume_progress_file=str(OUT/'progress.json'))
p.write_text(json.dumps(handoff,ensure_ascii=False,indent=2),encoding='utf-8')
marker='### 检测空档重新定位与周期包减负（2026-10-26）';log=WORK/'WORKLOG.md'
if marker not in log.read_text('utf-8'):
    with log.open('a',encoding='utf-8') as f:
        f.write('\n\n'+marker+'\n\n当前授权heartbeat还在工作，goal控制器最后仍paused；任一账户余量<=10%停止。周期过去偏差标定补位：side缺框4、train2缺框60，恢复0。train2 .4秒段内可人工插值的36个当前时刻全部检出；40.88—42.60秒44次缺框无人工可见标注，不能视作已证实漏检。\n\n固定20/60ms检测输入共205+65+50原生片，20ms检出train2 41/101、side27/31、空场0/26，60ms无额外检出，不接入。SDK OnDemand左边界为严格大于，初版恰好边界1像素控制失败后停；v2保留原Base20ms输入并核查60ms边界，失败目录保留。\n\n直接匀转弯拟合固定AICc/6px噪声/1.96se门槛，真实六队列0次支持，无预测变化，停止不降门槛。初版numpy二维标量转换失败、第二版慢速理想控制缺乏噪声下可观测性，尚未评分；最终高信噪比理想控制通过，真实结果output/live_turn_fit_v3_20261026。\n\n周期包提前保留每像素最后合法记录，保留全部非法记录保证排除计数。65片30个当前观测摘要/线/点逐项一致，离线包18222120→2733912字节（-85%）。可选--periodic-packet latest-pixel已接live，实际65片/16981921事件，31产生/30跟踪/6发布/3有效，包6929712→1040616字节，源/三方共享图SHA通过。与前次quarter4有效不同，年龄P95 161.760ms，未证明稳定端到端提速或精度提升，不升默认。旧实际copy计时包含compaction，分项不可直接求和。\n\n局部放大精定位直接调用现模型：side96个有框时刻0成功精定位；train2 16个未截边框也0，其他25截边跳过。原当前图与NN框全部一致。裁剪后输入域不适配，只能作为诊断负结果，不再靠放宽匹配门槛强取框。source tools/probe_roi_refine.py。\n\n开始一次实质训练改动：复用40现有20ms原生图/弱框，train1/2/4，val6，test3不参与。355训练原图+199放大图、180验证原图+93放大图，固定10epochs、AdamW lr1e-4、freeze前10层梯度、rect=True，单次验证fitness选best。既有缩放训练思想/SAHI相关方法，不主张新网络。训练用GT裁剪可行；实际推理仍只能由当前检测框提出ROI。原权重ed11fada...保留，训练guard阻断test与密集预测标签，输出roi_augmented_detector_20261026。exec64989启动，尚未验证结束，必须先poll或读train_completion再续；不是已改进的检测或预测成绩。\n')
page=OUT/'index.html';s=page.read_text('utf-8')
if 'id="later-progress"' not in s:
    s=s.replace('</html>','''<section id="later-progress"><h2>后续：空档判断与真正的训练改动</h2><p>train2 有人工参考的36个当前时刻全部检出；40.88—42.60秒连续44次缺框没有人工可见标注，不能直接算成漏检。60ms积累、周期线补位、直接匀转弯拟合均未增加有效证据，不接入默认。</p><p>当前周期包去重保持30次摘要/线/点完全一致，离线减少约85%字节；实际短RAW传输6.93→1.04MB。有效输出3/6，仍未证明稳定端到端加速或精度提升。</p><p>直接放大ROI后旧检测器不适配，正在用原有训练图片做一次固定10轮局部放大增强微调。原模型保留，原测试录像不参与训练。目前只能称训练进行中，效果待其完成后决定。</p><p><a href="progress.json">详细当前状态及运行任务</a> · <a href="../detector_window_v2_20261026/summary.json">窗口负结果</a> · <a href="../periodic_packet_20261026/comparison.json">无损周期包</a> · <a href="../roi_augmented_detector_20261026/protocol.json">固定训练设计</a></p></section></html>''')
    page.write_text(s,encoding='utf-8')
print(json.dumps(dict(checkpoint_written=True,running_session=64989),ensure_ascii=False))
