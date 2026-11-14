"""RAW pipeline delivery report; unlabelled recordings remain qualitative."""
import json,csv
from study24 import ROOT,OUT,CASES,load,save,verify


def table(headers,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+str(x)+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</table></div>'


def main():
    verify();a=load(OUT/'analysis.json')
    for case in CASES:
        rr=list(map(json.loads,(OUT/case/'observations_forecasts.jsonl').read_text().splitlines()))
        with (OUT/case/'primary_forecasts.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f);writer.writerow(['t','current_detection','forecast_valid','primary_model','x_.1','y_.1','x_.2','y_.2','x_.4','y_.4'])
            for r in rr:
                p=r['primary_prediction'];writer.writerow([r['t'],r['observation']['detected'],r['forecasting']['valid'],r['primary'],*[x for h in ['0.1','0.2','0.4'] for x in (p[h] if p else [None,None])]])
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第二十四阶段：RAW研究流水线</title><style>body{font:17px/1.75 "Microsoft YaHei",sans-serif;margin:0;background:#f3f6fa;color:#243342}main{max-width:1250px;margin:auto;background:white;padding:32px}h1{font-size:31px;line-height:1.3}h2{margin-top:36px;border-bottom:2px solid #dce4ec}.note{padding:18px;background:#edf6fa;border-left:5px solid #3685a5}.warn{background:#fff4dd;border-color:#c68d26}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;white-space:nowrap;text-align:left;border-bottom:1px solid #dce4ec}th{background:#edf2f7}a{color:#176e99}video,img{max-width:100%;height:auto}pre{background:#edf2f7;padding:14px;overflow:auto}details{padding:15px;border:1px solid #dce4ec;margin:18px 0}summary{cursor:pointer;font-weight:bold}</style><main><h1>第二十四阶段：只输入RAW的研究预测流水线</h1><p>2026-10-18 · 清晰回归+额外训练录像+日间/夜间/黄昏 · 总计20秒/1000时刻</p><div class="note"><b>已接成可运行入口：</b>原生事件→当前桨盘/整机观测→CV与稳健历史预测→0.1/0.2/0.4秒图像位置，另可启用原逐桨频率诊断。单次推理不读标签、manifest或缓存结果。</div><h2>1. 实际运行统计</h2>'''
    text+=table(['片段','当前检测发布/总时刻','预测锚点','四频率内部有效','处理中位ms','P95 ms','含读写视频wall s','精度依据'],[[c['case'],f"{c['detected']}/{c['frames']}",c['forecast_anchors'],c['ticks_four_frequencies_internal'],f"{c['processing_median_ms']:.2f}",f"{c['processing_p95_ms']:.2f}",f"{c['wall_including_open_seek_decode_write_preview_s']:.2f}",('已有阶段回归' if c['case']=='self_clear' else '稀疏框3/12覆盖' if c['case']=='train3_extra' else '无标签，仅定性')] for c in a['cases']])
    text+='<p>发布次数不是召回率。未标注实验录像没有独立精度参考。train3 12个有机标签仅3次当前定位发布，说明模型对这段更小/不清楚的桨盘泛化不足；不能因为日间视频出现框线就称已解决。</p><p>清晰self_clear的100个观测与阶段21、全部模型预测与阶段23逐项一致；启用原RotorSensing没有反向改变这些预测。74时刻“四频率内部有效”不等于74帧四桨都当帧可见或真实转速准确：还涉及频率时效、双叶假设和谐波，RPM/推力均未独立验证。</p><div class="note warn">可选频率只作诊断，目前预测使用位置历史，未声称频率/推力带来增益。清晰含频率P95约252ms，2秒素材实际18.34秒；其余片段也未实现50Hz实时。视频50fps按源时间轴播放，不表示处理速度。</div><h2>2. 查看预测视频</h2><p>绿色为当前整机框，蓝色为CV，橙色为主显示模型trusted_acc的未来位置；标注0.1/0.2/0.4秒。缺少当前测量时会明确显示；超过160ms不继续发新预测。线是当时预报，不是未来真值。</p>'
    for case in CASES:
        text+=f'<details'+(' open' if case=='self_clear' else '')+f'><summary>{case}</summary><video controls preload="metadata" src="{case}/forecasts.mp4"></video><p><a href="{case}/observations_forecasts.jsonl">逐时刻JSON</a> · <a href="{case}/primary_forecasts.csv">主模型CSV</a> · <a href="{case}/completion.json">运行统计</a></p></details>'
    if (OUT/'additional_contact.png').exists():text+='<h2>3. 额外录像抽样</h2><img src="additional_contact.png"><p>每段20/40/60/80%视频时刻，带模型输出。仅作事后可观测性检查，不是独立标注。夜间/黄昏输出稀疏，不能声称其每次发布对应正确无人机。</p>'
    text+='<h2>4. 自己运行</h2><pre>&amp; "D:\\drone forecast\\.venv-event311\\Scripts\\python.exe" "D:\\drone forecast\\event_drone_forecast\\run_research_forecast.py" --raw "E:\\dataset\\train\\6\\media\\event.raw" --start 22 --duration 2 --output "D:\\drone forecast\\output\\my_research_prediction" --preview --rotor-features</pre><p>输出目录必须不存在，start/duration为20ms整数倍；只有RAW是传感器输入。默认native20+trusted_acc是开发数据选出的研究配置，CV始终同时保存。可用--primary cv作基线，--frontend native10等作前端对照；去掉--rotor-features可减少频率计算，但不能据此宣称检测精度改善。</p><h2>5. 保留的边界</h2><p>单次推理不需要质量/风速/桨盘尺寸/推力系数，也不声称恢复这些真实参数。输出是图像平面像素。原run.py等默认保持，新入口为实验分支。新experiment录像在配置冻结前未读画面，但没有框真值，所以只能作定性检查；train3此前被抽样过，不称全局盲测。</p><p>下一步先用受控事件删减检查“先清楚建模、随后只见部分桨叶”的机制，继续保留已有真实框FDE与CV，不拿模拟删减或内部频率标志代替自然遮挡/真实RPM真值。</p><p><a href="protocol.json">冻结协议</a>、<a href="analysis.json">汇总</a>、<a href="self_clear/regression.json">回归一致性</a>、<a href="阶段结论.md">阶段结论</a>。无G，未访问或执行yolov10-main，旧源码及结果保留。</p></main></html>'
    (OUT/'第二十四阶段RAW研究流水线报告.html').write_text(text,encoding='utf-8');print('Stage24 report rendered')


if __name__=='__main__':main()
