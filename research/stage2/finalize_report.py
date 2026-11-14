"""Append the identity guard's results after retaining the failed integration."""
from pathlib import Path
import html,json
from study import OUT,save
from build_report import score_own,own_video,table,number,traces

def main():
    folder=OUT/'guarded/own_train6';metrics,absent=score_own(folder,'guarded_experimental');own_video(folder)
    summary=json.loads((folder/'summary.json').read_text());fred=json.loads((OUT/'guarded/fred3/summary.json').read_text())
    rows=[[r['run'],r['method'],f"{r['available']}/{r['targets']}",number(r['error_px']),number(r['common_error_px'])] for r in metrics if r['horizon_s']==.4]
    section=f'''<section><h2>追加：身份约束，避免把任意活动当成无人机</h2>
<p><b>无约束活动回退在自有录像上退化，未采用为默认入口。</b>新增run_guarded.py始终保留原桨盘测量；仅在最近0.5秒内出现合格四桨、最近桨盘测量丢失不超过0.3秒，且位置门控与已学习的观测中心偏移一致时，才接受活动位置。活动位置不更新姿态或推力。该修改是看到失败后的开发修复，不是新测试集上的算法胜出。</p>
<p>train6前30秒：检测{summary['detected_frames']}/{summary['frames']}帧，物理分支{summary['physics_active_frames']}帧；位置来源{html.escape(json.dumps(summary['source_detection_frames'],ensure_ascii=False))}。</p>
{table(['版本','分支','可用/端点','0.4秒误差px','共同端点误差px'],rows)}
<p>空场景诊断：{html.escape(json.dumps(absent,ensure_ascii=False))}。确认来自算法桨盘观测，并非独立身份真值；原桨盘检测的误报仍可能保留。这是有边界的接续，不宣称完整语义识别。</p>
<p>代价同样明确：FRED sequence3没有可辨桨盘，受约束入口只检测{fred['detected_frames']}/{fred['frames']}帧；它无法获得无约束活动方案的93.8%覆盖率。不能把两个不同身份假设下的最佳数字拼接成一个系统成绩。</p>
<video controls preload="metadata" poster="guarded/own_train6/自有录像_24秒.png" src="guarded/own_train6/自有录像_真实未来对照.mp4"></video>
<p>这是当前推荐用于继续四桨论文复现的实验入口；run_hybrid.py只保留为无约束回退失败对照。原run_paper.py与原v0.3结果保持不变。</p>
<p><a href="guarded/protocol.json">身份约束冻结协议</a> · <a href="guarded/own_train6/evaluation.json">完整评分与空场景误报</a></p></section>'''
    path=OUT/'第二阶段实验报告.html';document=path.read_text(encoding='utf-8')
    marker='<section><h2>现在能运行什么，下一步做什么</h2>'
    assert marker in document
    document=document.replace(marker,section+marker).replace('event_drone_forecast\\run_hybrid.py','event_drone_forecast\\run_guarded.py')
    document=document.replace('新增独立实验入口run_hybrid.py','新增受身份约束的实验入口run_guarded.py')
    path.write_text(document,encoding='utf-8')
    d=json.loads((OUT/'stage2_summary.json').read_text());d.update(guarded_own=summary,guarded_sparse_scores=metrics,guarded_absence=absent,
        guarded_fred3=fred,recommended_experimental_entry='run_guarded.py',unrestricted_activity_rejected_as_default=True)
    save(OUT/'stage2_summary.json',d)
    md=OUT/'阶段报告.md'
    with md.open('a',encoding='utf-8') as f:
        f.write('\n追加身份约束修复：run_guarded.py。无约束活动回退仅作失败对照。严格身份确认牺牲FRED小目标覆盖率，不能合并不同协议的最好成绩。完整评分见guarded/own_train6/evaluation.json。\n')
    save(OUT/'stage2_completion.json',dict(geometry_cases=2160,raster_cases=96,public_development_sequences=3,
        own_clip_seconds=30,failed_unrestricted_activity_retained=True,guarded_entry_validated=True,
        innovation_claimed=False,independent_real_rpm_pose_truth_available=False))
    print(json.dumps(dict(guarded_summary=summary,absence=absent,metrics_04s=[r for r in metrics if r['horizon_s']==.4])),flush=True)

if __name__=='__main__':main()
