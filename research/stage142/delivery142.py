"""One outcome report for continuous improvements138..142."""
from pathlib import Path
import html,json
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT.parent
O138=WORK/'output/stage138_shared_objective_20261024';O139=WORK/'output/stage139_dense_fallback_20261024';OUT=WORK/'output/stage140_visible_fallback_20261024';O141=WORK/'output/stage141_periodic_axis_20261024';O142=WORK/'output/stage142_frequency_model_pilot_20261024';OLD=WORK/'output/stage135_optimized_forecast_20261024'
def load(p):return json.loads(Path(p).read_text('utf-8-sig'))
def rows(p):return [json.loads(s) for s in Path(p).read_text('utf-8').splitlines() if s]
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),'utf-8')
def table(rr,columns):
    def fmt(v):return f'{v:.3f}' if isinstance(v,float) else str(v)
    return '<table><tr>'+''.join('<th>'+html.escape(t)+'</th>' for _,t in columns)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(fmt(r.get(k,'')))+'</td>' for k,_ in columns)+'</tr>' for r in rr)+'</table>'
def predict(c,h):return np.array(c['position'])+h*np.array(c['velocity'])+.5*h*h*np.array(c['acceleration'])

def main():
    assert load(OUT/'checks140.json')['actual_RAW865_complete'];assert load(O141/'checks141.json')['forecast_exact140']
    for name in ['fast100','natural6','side50','negative5']:assert load(O138/name/'checks138.json')['geometry_decisions_exact']
    control=rows(O138/'fast100/predictions.jsonl');fast=rows(O138/'natural6/predictions.jsonl');latency=[]
    for key,label in [('processing_ms','整步处理'),('geometry_ms','几何'),('frontend_ms','支持/轮廓'),('model_ms','结构拟合'),('NN_ms','GPU检测')]:
        b=float(np.median([r[key] for r in control]));a=float(np.median([r[key] for r in fast[:100]]));latency.append(dict(stage=label,before_ms=b,after_ms=a,reduction_percent=100*(b-a)/b))
    reference=load(WORK/'output/stage128_RAW_independent_support_20261023/scores128.json');actual={case:{round(r['t'],8):r for r in rr} for case,rr in [('continuous6',rows(OUT/'continuous6/predictions.jsonl')),('extension2',rows(OUT/'extension2/predictions.jsonl')),('natural6',fast)]};old={case:{round(r['t'],8):r['output'] for r in load(OLD/(case+'.json'))} for case in actual};pairs=[];scores=[];missing=[]
    for p in reference['pairs']:
        case=p['case']
        if case not in actual:continue
        r=actual[case][round(p['origin'],8)]
        if not r['valid']:missing.append(dict(case=case,h=p['h'],origin=p['origin'],reason='not_published'));continue
        now=r['coefficients']['optimized_geometry'];before=old[case][round(p['origin'],8)]['coefficients']['optimized_geometry'];truth=np.asarray(p['truth_bbox_center']);pairs.append(dict(case=case,h=p['h'],origin=p['origin'],target=p['target'],old_px=float(np.linalg.norm(predict(before,p['actual_h'])-truth)),new_px=float(np.linalg.norm(predict(now,p['actual_h'])-truth))))
    for case in ['natural6','continuous6','extension2']:
        for h in [.1,.2,.4]:
            pp=[p for p in pairs if p['case']==case and p['h']==h];b=float(np.mean([p['old_px'] for p in pp]));a=float(np.mean([p['new_px'] for p in pp]));eligible=next(r['eligible'] for r in reference['table'] if r['case']==case and r['h']==h and r['method']=='count_model');scores.append(dict(case=case,h=h,eligible=eligible,paired=len(pp),missing_total=eligible-len(pp),before_px=b,after_px=a,reduction_percent=100*(b-a)/b))
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    for filename,plot,fields,title,ylabel in [('runtime138.png',latency[:4],('before_ms','after_ms'),'同一入口/共同100步：共享目标函数、支持ROI与计数重叠','中位处理耗时 / 毫秒'),('forecast140.png',[r for r in scores if r['h']==.4],('before_px','after_px'),'实际RAW：提前0.4秒的共同未来框端点','未来框中心平均误差 / 像素')]:
        fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained');xx=np.arange(len(plot));width=.35
        for offset,key,label,color in [(-width/2,fields[0],'此前137','#93a1ae'),(width/2,fields[1],'当前优化','#16857e')]:
            bars=ax.bar(xx+offset,[r[key] for r in plot],width,label=label,color=color);ax.bar_label(bars,fmt='%.2f',padding=3)
        labels=[r['stage'] for r in plot] if 'stage' in plot[0] else [{'natural6':'清晰片段','continuous6':'连续侧视','extension2':'扩展2'}[r['case']] for r in plot]
        ax.set_xticks(xx,labels);ax.set_ylabel(ylabel);ax.set_title(title);ax.legend();ax.set_ylim(0,max(max(r[fields[0]],r[fields[1]]) for r in plot)*1.25);ax.spines[['top','right']].set_visible(False);fig.savefig(OUT/filename,dpi=160);plt.close(fig)
    native=rows(O141/'continuous6/predictions.jsonl');times=[r['t'] for r in native];frequencies=[r['periodic_axis'].get('frequency_hz') if r['periodic_axis']['qualified'] else np.nan for r in native];angles=[((r['periodic_axis']['line']['angle']+90)%180-90) if r['periodic_axis']['axis_valid'] else np.nan for r in native]
    fig,axes=plt.subplots(2,1,figsize=(10,5),sharex=True,layout='constrained');axes[0].scatter(times,frequencies,s=9,color='#996c15');axes[0].set_ylabel('像素闪烁候选 / Hz');axes[1].scatter(times,angles,s=9,color='#16857e');axes[1].set_ylabel('无向线角 / 度');axes[1].set_xlabel('RAW时间 / 秒');axes[0].set_title('周期支持提供当前侧视线候选；缺测保持为空，不作姿态真值')
    fig.savefig(OUT/'periodic_axis141.png',dpi=160);plt.close(fig)
    index=next(i for i in range(len(native)-1,-1,-1) if native[i]['periodic_axis']['axis_valid']);cap=cv2.VideoCapture(str(O141/'continuous6/predictions.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,im=cap.read();cap.release();assert ok;cv2.imwrite(str(OUT/'periodic_axis_preview.png'),im)
    completion138=load(O138/'natural6/completion.json');completion140={c:load(OUT/c/'completion.json') for c in ['continuous6','extension2']};checks141=load(O141/'checks141.json');failed139=load(O139/'scores139.json');failed142=load(O142/'scores142.json');math138=load(O138/'math_checks138.json')
    report=OUT/'持续优化结果与侧视预测.html';step=latency[0]
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>持续优化：加速与侧视运动预测</title><style>body{{max-width:1180px;margin:32px auto;padding:0 24px;font:17px/1.8 system-ui;color:#213847}}h1,h2{{line-height:1.4}}h2{{margin-top:2em}}table{{border-collapse:collapse}}th,td{{border:1px solid #cbd8dd;padding:8px;text-align:left}}th{{background:#edf5f7}}img,video{{max-width:100%;height:auto}}.good{{padding:18px;background:#e4f4ed}}.limit{{padding:18px;background:#fff0d2}}a{{color:#007b88}}code,pre{{background:#edf3f5}}pre{{padding:16px;overflow:auto}}</style>
<h1>138—142：继续改代码，取得速度和退化场景预测收益</h1><p>2026-10-24。本轮合并成一份结果。当前RAW入口默认使用138共享计算＋140可见性守门运动分支；141周期线是可选独立观测，142动力学试验未采用。</p>
<p class="good">同入口共同100步处理{step['before_ms']:.1f}→{step['after_ms']:.1f}ms，清晰8秒完整运行约{completion138['processing_median_ms']:.1f}ms/步。实际RAW连续侧视0.4秒未来框误差61.09→48.78px，扩展2为71.10→63.63px，清晰段仍30.83px。收益已经进入可运行代码。</p>
<h2>这次实际优化了哪里</h2><ol><li><b>保留全图阈值，形态学只计算支持范围。</b>当前全部前景支持的包围范围加4px余量，没有丢掉框外的前景或跨界轮廓；范围超过图像80%时回退全图。轮廓恢复全局坐标，候选排名/全局面积资格不变，原开运算在防合并步骤中复用。</li>
<li><b>拟合残差和导数共享同一次计算。</b>固定结构的点、模板和圆锥矩阵只准备一次；残差/导数在同一优化迭代点复用逆矩阵和投影结果。目标函数、初始化、优化器与边界/五帧规则保持。</li>
<li><b>CPU计数与GPU检测重叠。</b>发送原生ON/OFF帧后，等待同一帧NN结果期间计算当前事件计数。接收到当前框后才运行依赖该框的部分；时间截止和数据内容不变。</li>
<li><b>改进没有桨盘模型时的预测。</b>过去0.35秒要求连续、密集的当前机体观测，稳健拟合误差符合原6px噪声尺度，且参与拟合的机体框全部未被画面截断，才允许稳健加速度进入未来外推。否则保持机体匀速；已有合格桨盘模型的135路径不变。</li></ol>
<p>边缘截断会改变可见机体框中心，连续检测也可能产生假加速度。139只检查连续性，把扩展2从71.10恶化到103.88px；140加入整个拟合窗口的框边界守门后，扩展2为63.63px。这里检查的是框未触及图像边界，不能保证整机或全部桨叶真实可见，也不能排除所有错误框。139失败代码/结果保留，没有接入默认预测，也没有按每个视频写例外规则。</p>
<h2>速度：同入口的实际比较</h2><img src="runtime138.png">'''+table(latency,[('stage','处理部分'),('before_ms','此前137/ms'),('after_ms','当前138/ms'),('reduction_percent','耗时减少/%')])+f'''
<p>旧后端100步与新后端400步的前100步比较，同一22—24秒RAW、当前检测权重和帧校验。系统负载未受控，各阶段中位数不能相加。处理耗时不含RAW读取/解码、冷启动和视频写出。完整138清晰400步另有序列化/视频约{completion138['serialization_preview_median_ms']:.1f}ms，总含读取/冷启动/视频{completion138['wall_including_cold_start_s']:.1f}秒。<b>仍未达到20ms实时。</b></p>
<h2>预测：实际RAW重新计分</h2><img src="forecast140.png">'''+table(scores,[('case','片段'),('h','提前量/秒'),('eligible','可选端点'),('paired','共同发布端点'),('missing_total','未配对'),('before_px','此前135/137误差px'),('after_px','当前140误差px'),('reduction_percent','误差减少/%')])+'''
<p class="limit">连续侧视0.1秒12.79→12.91px，扩展2的0.1秒11.65→12.37px，短期略有回退；主要收益在0.2/0.4秒。以上是反复暴露的开发录像，只有稀疏人工未来框中心，不是重心/姿态/转速真值，不证明泛化或创新性。0.4秒侧视10对、扩展2六对、清晰11对，样本仍少。清晰0.1秒启动1对未发布，沿用136共同11对口径，未按误差筛点。</p>
<p>140实际RAW：连续侧视475、扩展2300、清晰活动前缀60、空场30。完整清晰400步使用138实际RAW观测作140预测重放，系数保持，没有声称又跑了一次完整140清晰NN/RAW。扩展1只作固定观测重放，保持原结果，未新增完整RAW。当前学习的是图像运动可靠性，不是根据事件数标定推力。</p>
<h2>新侧视预测视频</h2><video controls preload="metadata" poster="continuous6/preview.png" src="continuous6/predictions.mp4"></video><p>绿：当前140未来预测；灰：机体匀速。Dense motion=true表示启用了有可靠性守门的历史加速度，不表示已测到推力或桨叶转速。50fps为离线播放。</p>
<video controls preload="metadata" poster="extension2/preview.png" src="extension2/predictions.mp4"></video><p>扩展2：拟合窗口含被截断的机体框时，额外加速度自动弃权，避免边缘中心变形直接驱动外推。</p>
<h2>周期支持的侧视线：新增了观测，但没有冒充物理预测</h2><img src="periodic_axis_preview.png"><p>示例''' + f'{native[index]["t"]:.2f}' + f'''秒，展示候选存在的一个时刻；整段475个时刻中，周期ROI合格{checks141['periodic_ROI_qualified']}次、无向线候选{checks141['unsigned_line_candidates']}次，并非全程成功。黄点为当前周期支持像素；青线与双向箭头为图像线及其无向法线，绿色仍是140预测。</p>
<img src="periodic_axis141.png"><video controls preload="metadata" poster="periodic_axis_preview.png" src="../stage141_periodic_axis_20261024/continuous6/predictions.mp4"></video>
<p>直接复用79的已安装Prophesee逐像素频率引擎、原严格7次/200us配置、当前20ms像素一票、ROI/环带共同闪烁拒绝，再用项目已有线几何。未重写频率引擎，没有G通道或灰度重建。输出保持倍频/电机归属/法线符号歧义，不能从一个主峰得到四电机转速平方和。</p>
<p>已知周期条纹控制80个时刻中77次线候选，移动条纹20/80；共同闪烁、Poisson、有色随机、突发在本轮各四个种子中无合格ROI。切频76/80得到周期但没有线候选。这些是已知像素过程，不是实飞RPM真值，也不能把有限负例零次外推成实际零误报。141完整475步RAW的预测与140逐项完全相同，周期轴没有偷偷参与预测。</p>
<h2>为什么没有把频率和法线强塞进推力模型</h2><p>142实际复用原ProjectedThrustModel的在线RLS运算，以新周期线和相对闪烁率平方作像素特征，保留0.2秒延迟、0.6秒预热、激励门槛和参数限制；明确把旧“四可见”接口改成“特征可用”，不伪造四个转速。141观测重放475步中204次支持、155次启用，共151次在线更新，但0.4秒误差48.78→50.89px，因此未采用，未再调参筛选出一个好成绩。</p>
<p>仍未解决的因素包括：像素闪烁频率有谐波和电机来源歧义；无向投影线不提供完整三维姿态；框残差同时含检测偏差、风/重力、深度变化与真实加速。本轮没有隔离证明哪一项主导142的退步。未知质量、推力系数和投影尺度不能凭单目框残差唯一分开。当前方向候选尚无独立真值，真实推力可观测性也未证明。</p>
<h2>已有方法与下一步</h2><p><a href="https://arxiv.org/html/2603.08386v1">DDHF逐像素谐波检测论文</a>用异步极性频谱、谱平坦度和谐波梳区分转动支持，低仰角仍有退化及框合并问题。本文页面未给出可复用的作者实现链接，本轮未运行DDHF；其多谐波判别可作为后续频率歧义处理的参考。<a href="https://github.com/ros-event-camera/frequency_cam">FrequencyCam</a>已核查，但未采用其IIR亮度重建路径。<a href="https://github.com/JackPieCZ/EE3P">EE3P/EEPPR</a>提供一般周期目标资料，本项目以前已有相关模块实验，本轮未重复跑。</p>
<p>下一项应先处理多电机/谐波来源与周期线的可信度，用现有公开周期真值核查频率模块，再考虑预测权重；不能把235次线候选当235次正确姿态。速度方面还可减少全图支持构建和NN检测开销。全部已启动实验已完成，后续入口和未解问题保存在worklog。</p>
<h2>直接使用当前入口</h2><pre>&amp; 'D:\\drone forecast\\.venv-event311\\Scripts\\python.exe' `
  'D:\\drone forecast\\event_drone_forecast\\run_optimized_forecast.py' `
  --raw 'E:/dataset/train/6/media/event.raw' `
  --output 'D:/drone forecast/output/my_optimized_run' `
  --start 110 --duration 9.5 --preview</pre><p>输出目录须不存在；默认shared几何＋visible运动。加<code>--periodic-axis</code>显示并记录周期线候选，默认关闭且不影响预测。<code>--motion-backend structure</code>回到135，<code>--geometry-backend fast</code>选择137计算。默认54/API93未改，保护目录未访问/修改，无G/新依赖。</p>
<p><a href="actual_scores142.json">当前RAW逐点评分</a> · <a href="optimization_summary142.json">全部结果和来源</a> · <a href="../../event_drone_forecast/research/stage140/README.md">运行与交接</a> · <a href="../stage142_frequency_model_pilot_20261024/scores142.json">未采用的在线特征模型</a></p></html>'''
    report.write_text(page,'utf-8');save(OUT/'actual_scores142.json',dict(table=scores,pairs=pairs,not_published=missing,clear_source='138 actual RAW +140 predictor replay; no new full140 clear NN run',side_extension_source='140 actual RAW outputs'))
    summary=dict(report=str(report),latency138=latency,scores140=scores,missing=missing,periodic141=checks141,failed139_summary=[r for r in failed139['table'] if r['h']==.4],failed142=failed142['table'],default_entry=str(ROOT/'core/run_optimized_forecast.py'),algorithm_stage=140,geometry_execution_stage=138,optional_axis_stage=141,latest_diagnostic_stage=142,raw_frames138=580,raw_frames140=865,raw_frames141=475,total_successful_RAW_frames_this_turn=1920,independent_test=False,realtime=False)
    save(OUT/'optimization_summary142.json',summary)
    landing=WORK/'output/latest_optimized_forecast.html';landing.write_text('<!doctype html><meta charset="utf-8"><title>当前优化结果</title><meta http-equiv="refresh" content="0;url=stage140_visible_fallback_20261024/持续优化结果与侧视预测.html"><p><a href="stage140_visible_fallback_20261024/持续优化结果与侧视预测.html">打开当前优化报告和视频</a></p>','utf-8')
    print(json.dumps(dict(report=str(report),latency=latency,scores=scores,periodic=checks141)),flush=True)
if __name__=='__main__':main()
