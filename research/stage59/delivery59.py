"""Consolidate the completed line pilot without modifying frozen inference."""
from pathlib import Path
import datetime,html,json
import study59 as current
import lsd59
OUT=current.OUT

def run():
    current.verify();lsd59.verify();a=current.load(OUT/'analysis.json');b=current.load(OUT/'lsd_results.json')
    assert current.load(OUT/'completion.json')['native10']==b['native10']==450
    def table(rows):
        cols=list(rows[0]);return '<table><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r[c]))+'</td>' for c in cols)+'</tr>' for r in rows)+'</table>'
    display=[]
    for case in current.CASES:
        for name in ['base','hard1p5','hard2']:
            x=next(r for r in a['real'] if r['case']==case and r['variant']==name);y=next(r for r in b['table'] if r['case']==case and r['variant']==name)
            display.append({'片段':case,'阈值':name,'总时刻':x['ticks'],'薄连通线候选帧':x['ticks_with_lines'],'LSD边缘线帧':y['ticks_with_edge_lines'],'LSD边缘线总数':y['total_edge_lines']})
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>侧视线段与分支融合：结论</title><style>body{max-width:1350px;margin:32px auto;font:17px/1.85 Arial;color:#243e4b}h1{font-size:30px}img{width:100%}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #c9d5dc;padding:7px}.box{padding:18px;background:#eef5f7;border-left:5px solid #2b7f94}a{color:#176584}</style><h1>侧视用线段，正视用椭圆：方向合理，关键在端点与身份</h1><div class="box">可以将它做成侧视专用观测分支，并与椭圆观测平滑融合。完整、对称且可分离的单个桨盘，接近侧视时可用线段中点近似桨心；当前不能把任意高事件线的中点都当作真实轴心。</div><h2>本轮已经验证的内容</h2><p>450个不同原生10ms当前时刻：清晰1秒、侧视2秒、3号1秒、空场0.5秒。主试验和LSD补充各重读这450时刻，共900次RAW当前窗口处理；不是900个独立样本。主试验原57几何逐项精确，补充全图密度阈值逐项精确。事前固定阈值倍率1/1.5/2，原冻结代码未改；固定25/50/75%画面，所有失败保留。</p>'''+table(display)+'''<p>侧视LSD可提到更多直边，但边缘线可能属于一个桨盘的两侧、机臂或碎片，数量不是桨心准确率。提高阈值有时让图像更干净，同时会删除端点或整个桨盘。本轮没有稳定观测收益证明，更没有新增未来位置成绩。以下画面黄色为LSD边缘，紫色为其几何中点，均未认证为真实桨心。</p><img src="lsd_side6_0149.png"><img src="lsd_train3_0049.png"><h2>为什么线段中点需要条件</h2><p>受控完整对称线中点误差0；一侧缺失30px时中点偏15px。活动强度沿线不均，阈值增加时误差0→11.5→28px。两个共线桨盘粘连，整段中点不能代替任意单桨轴心；一个共线离群点也会带偏极值端点，分位数端点能缓解离群点，但不能补回真实遮挡。透视也不保持三维中点的二维投影。</p><img src="line_midpoint_controls.png"><h2>建议的分支与权重</h2><ol><li>正视/斜视椭圆分支提供中心、轴比和法向线索；接近侧视时，由线段分支提供无向轴和有条件的中心。它们共同更新同一个整机/桨槽状态，使用同一套运动动力学。</li><li>先验证线段身份：与过去桨槽和整机位置对应、双边合成中心线、端点支持完整，并检查周期、跨阈值和时间连续性。大块内部任意直边不能直接接成四个桨心。</li><li>权重由轴比和观测质量共同决定。示意w_line=s(r)q_line/[s(r)q_line+(1-s(r))q_ellipse]，s随轴比变小而增加；q_line须包含端点/身份可靠性，不能只等于事件数。当前line59.axis_weight仅为未标定示意，没有接入默认预测。</li><li>权重平滑并加滞回，避免模式来回跳。无向轴用双角平均再作时间消歧，避免相反法线直接相加。两观测来自同一事件窗，不能当作独立重复证据。</li><li>端点缺失或身份不清时，仅更新姿态轴，沿线中心保留较大不确定度；转速证据不足时不强行更新推力。下一步再做同一整机位置上的历史/椭圆/线段/融合未来轨迹消融。</li></ol><h2>复用与局限</h2><p>查阅RT-EvLDT官方论文/作者库、2018低延迟事件线跟踪、ELiSeD。此次实际运行未修改的本地geometry._line、57/9/8和已有OpenCV LSD；RT-EvLDT未编译执行，无新依赖。主路径虽已有line模式，中心仍主要来自fitEllipse，故用户建议有明确的可改进实现位置。58的共同圆锥在中等倾角合成控制有效、极端侧视仍退化，也支持保留不同观测分支；58不提供实采姿态真值。</p><p>原54部署和完整15预测模型保持。无同步RPM/姿态/桨心真值，不能据本轮候选覆盖或漂亮线条宣称预测已经更准或超过论文。E盘数据只读，未访问/修改yolov10-main，不使用G。</p><p><a href="第五十九阶段侧视线段中点小试验.html">59完整主试验</a> · <a href="侧视LSD线段补充诊断.html">LSD完整补充</a> · <a href="../stage58_projected_centers_20261021/第五十八阶段投影圆心与共同桨盘报告.html">58圆心与近侧视失败</a> · <a href="https://github.com/event-driven-robotics/RT-EvLDT">RT-EvLDT作者库</a> · <a href="https://arxiv.org/html/2510.06829v1">RT-EvLDT论文</a> · <a href="https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2018.00004/full">2018事件线跟踪</a></p></html>'''
    (OUT/'结论与分支融合方案.html').write_text(page,encoding='utf-8')
    current.save(OUT/'delivery59.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),helper_sha256=current.sha(Path(__file__)),unique_native10=450,total_RAW_window_reads=900,
        previous58_synthetic_trials=28,previous58_current_samples=130,production_promoted=False,new_future_scoring=False,terminal_batches=[38541,76310],table=display))
    print('59 consolidated delivery complete',flush=True)

if __name__=='__main__':run()
