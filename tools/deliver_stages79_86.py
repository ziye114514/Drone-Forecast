"""Evidence-backed delivery after all frozen RAW and module checks complete."""
from pathlib import Path
import hashlib,html,json
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent/'output'
DIRS={79:BASE/'stage79_native_frequency_20261021',80:BASE/'stage80_exact_compute_20261021',81:BASE/'stage81_RAW_exact_compute_20261021',
    82:BASE/'stage82_streamed_window_20261021',83:BASE/'stage83_RAW_streamed_window_v2_20261021',84:BASE/'stage84_batched_rollout_20261021',
    85:BASE/'stage85_RAW_batched_rollout_20261021',86:BASE/'stage86_latency_deadlines_20261021'}
OUT=DIRS[85];load=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tr(values):return '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>'
def run():
    checks={n:load(p/f'checks{n}.json') for n,p in DIRS.items() if n!=86};audit=load(DIRS[86]/'audit86.json')
    for n in [79,80,82,84]:assert checks[n]['parent']==2185 and checks[n]['native10']==4370
    for n in [81,83,85]:assert checks[n]['parent']==835 and checks[n]['native10']==1670 and checks[n]['video_frames_decoded']==600
    assert checks[85]['all49_exact78'] and checks[84]['all49_and_current_exact78'] and not checks[85]['default_promoted']
    bench80=load(DIRS[80]/'benchmark80.json');bench82=load(DIRS[82]/'benchmark82.json');bench84=load(DIRS[84]/'benchmark84.json')
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    cases=['clear6','side6','train3','extra2'];stages=[78,81,83,85];colors=['#98a4b4','#d49c4e','#4d91c7','#2a9a79']
    timing={(r['stage'],r['case']):r for r in audit['table']};fig,ax=plt.subplots(figsize=(11,4.8));x=np.arange(4)
    for k,stage in enumerate(stages):
        values=[timing[stage,c]['median_processing_ms'] for c in cases]
        bars=ax.bar(x+(k-1.5)*.19,values,.18,color=colors[k],label=f'阶段{stage}')
        for bar in bars:ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+3,f'{bar.get_height():.0f}',ha='center',fontsize=8)
    ax.axhline(20,color='#c13e44',linestyle='--',label='50Hz的20ms预算');ax.set_xticks(x,cases);ax.set_ylabel('完整流程单步中位耗时 / ms')
    ax.set_title('相同源时间前缀的实际记录：全部49路，预测值保持');ax.legend(ncol=3);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(OUT/'full_pipeline_timing.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.3))
    specs=[(bench80,'old78','exact80','total','80：观测+额外预测模块'),(bench82,'exact80','stream82',None,'82：活动提取模块'),(bench84,'stream82_reference','batch84','forecast','84：额外预测模块')]
    for ax,(bench,old,new,component,title) in zip(axes,specs):
        old_values=[r[old][component]['median_ms'] if component else r[old]['median_ms'] for r in bench['table']]
        new_values=[r[new][component]['median_ms'] if component else r[new]['median_ms'] for r in bench['table']]
        x=np.arange(3);ax.bar(x-.18,old_values,.35,color='#98a4b4',label='前版');ax.bar(x+.18,new_values,.35,color='#2a9a79',label='优化后')
        ax.set_xticks(x,[r['case'] for r in bench['table']]);ax.set_title(title);ax.set_ylabel('交错基准中位ms');ax.grid(axis='y',alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'paired_module_timing.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for ax,case in zip(axes,['clear6','side6']):
        for stage,col in [(78,colors[0]),(85,colors[-1])]:
            r=timing[stage,case];ax.plot(r['source_relative_s'],r['optimistic_FIFO_age_s'],label=f'阶段{stage}',color=col)
        ax.axhline(.4,linestyle='--',color='#c13e44',label='0.4s提前量');ax.set_title(case);ax.set_xlabel('假设50Hz到达的源时间 / s');ax.set_ylabel('乐观FIFO排队后的观测年龄 / s');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'deadline_audit.png',dpi=150);plt.close(fig)
    fixed=[]
    for case in cases:
        video=OUT/case/'forecasts.mp4';cap=cv2.VideoCapture(str(video));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));frame_index=count//2
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index);ok,im=cap.read();cap.release();assert ok and im.shape==(720,1280,3)
        png=OUT/(case+'_fixed_prediction.png');encoded=cv2.imencode('.png',im)[1];encoded.tofile(str(png));fixed.append(dict(case=case,frame_index=frame_index,path=png.name))
        cap=cv2.VideoCapture(str(video));quarter=count//4;cap.set(cv2.CAP_PROP_POS_FRAMES,quarter);ok,im=cap.read();cap.release();assert ok
        png=OUT/(case+'_quarter_prediction.png');cv2.imencode('.png',im)[1].tofile(str(png));fixed[-1].update(quarter_frame_index=quarter,quarter_path=png.name)
    timing_rows=[];deadline_rows=[]
    for case in cases:
        vals=[timing[n,case]['median_processing_ms'] for n in stages]
        timing_rows.append(tr([case]+[f'{x:.2f}' for x in vals]))
        r=timing[85,case];h=r['lead_deadlines']['0.4']
        deadline_rows.append(tr([case,f'{r["optimistic_no_decode_processing_fps"]:.2f}',f'{r["source_time_realtime_factor"]:.1f}',
            f'{h["isolated_processing_before_target"]}/{h["valid_forecasts"]}',f'{h["optimistic_FIFO_before_target"]}/{h["valid_forecasts"]}']))
    benchmark_rows=[]
    for number,bench,old,new,component,_ in [(80,*specs[0]),(82,*specs[1]),(84,*specs[2])]:
        for r in bench['table']:
            a=r[old][component]['median_ms'] if component else r[old]['median_ms'];b=r[new][component]['median_ms'] if component else r[new]['median_ms']
            benchmark_rows.append(tr([number,r['case'],f'{a:.3f}',f'{b:.3f}',component or 'activity']))
    videos=''
    for frame in fixed:
        case=frame['case'];videos+=f'<h3>{case}</h3><p>固定取源视频四分之一及中间帧，未按预测误差挑选。蓝：CV；紫：旧IMM；黄：隔离后的主审阅分支。缺少当前物理支持时黄线回历史；目标/预测出画的情况也保留。</p><p>四分之一帧</p><img src="{frame["quarter_path"]}"><p>中间帧</p><img src="{frame["path"]}"><video controls preload="metadata" src="{case}/forecasts.mp4"></video>'
    report=OUT/'第七十九至八十六阶段输入诊断与计算优化报告.html'
    report.write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>79–86：输入诊断与计算优化</title>
<style>body{{font:17px/1.75 "Microsoft YaHei",sans-serif;max-width:1150px;margin:35px auto;padding:0 20px;color:#253248}}h1,h2,h3{{line-height:1.35}}img,video{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{border:1px solid #ccd4df;padding:8px;text-align:right}}td:first-child,th:first-child{{text-align:left}}.lead{{padding:18px;background:#e6f3ef;border-left:5px solid #2a9a79}}.limit{{padding:16px;background:#fff0d9}}a{{color:#235eaa}}code{{background:#eee;padding:2px 4px}}</style>
<h1>79–86：保持预测值，减少计算开销</h1>
<div class="lead"><b>本轮取得的是计算优化，尚未提高既有预测精度或实现实时。</b><p>实际复用SDK做了逐像素频率对照，未发现普遍补齐缺失的效果，因此没有接入推力。随后三项计算优化保留全部49路；最终85完整RAW的当前观测、预测、学习/模式状态与78逐项精确，两真实截断前缀通过，600视频帧全解码。</p></div>
<h2>完整RAW链路：相同时间前缀</h2><p>七段835父/1670原生窗口，其中四段视频共600帧，另有空场和两个重复截断前缀。下表只比较相同源时间前缀；这是分次实际记录，环境/温度/worker调度可能不同，并非随机交错整条链路试验。source50fps视频是播放时间轴，不是计算帧率。</p>
<table><tr><th>片段</th><th>78原版ms</th><th>81字典/缓存ms</th><th>83流式取样ms</th><th>85批量投影ms</th></tr>{''.join(timing_rows)}</table><img src="full_pipeline_timing.png">
<h2>改动与独立模块证据</h2>
<ol><li><b>80：减少完整预测字典的深复制。</b>新根、系数映射、预测映射单独复制，旧值只读；小系数/学习/延迟队列仍深复制。同一窗口重复背景tone缓存。全部49/当前观测及先前行不被后续修改检查通过；背景缓存对观测耗时帮助小，主要收益来自预测复制。</li>
<li><b>82：先按原全局索引取样，再拼接。</b>不再把全部160ms事件拼成大副本后才取样。全局stride和跨包相位、事件顺序/时间完全保持，不能在每包重启取样。80个人工控制含百万事件/空包/部分边界/包内无序，真实当前activity全部字段精确78。</li>
<li><b>84：批量计算26个未来时距投影。</b>复用<a href="https://numpy.org/doc/stable/reference/generated/numpy.matmul.html">NumPy矩阵堆叠运算</a>，原滤波更新、作者IMM/MMAE交互/概率与推力学习保持。120随机银行及真实全部49的严格相等检查通过，未用预先允许的1e-8容差放宽。</li></ol>
<p>80、82、84各有2185父/4370原生模块回归，含相同上下文重复和前缀，不能相加当新独立样本。独立基准采用同一步交错前/后顺序，输入、输出完全相同；decode/NN/worker/JSON和一致性检查耗时不包含在模块计时。下面不是端到端实时指标。</p>
<table><tr><th>阶段</th><th>片段</th><th>前版中位ms</th><th>优化中位ms</th><th>计时范围</th></tr>{''.join(benchmark_rows)}</table><img src="paired_module_timing.png">
<p>侧视前两秒多数没有当前有效NN框，活动/预测中位值接近空调用，不能由该前缀宣称全侧视都同样快。完整链路和全段分布应结合上表。profile只用于定位，未当运行吞吐量。</p>
<h2>79的逐像素频率结果为何没有部署</h2><p>实际调用<a href="https://docs.prophesee.ai/stable/api/cpp/cv/algorithms.html">Prophesee FrequencyAlgorithm</a>，修复本机Python24B记录/16B视图步长适配，已知控制和分块输出通过；不修改SDK。严格逐像素在3号、1号、2号片段零合格支持，对齐侧视有少量补充，清晰覆盖也未超原频谱。亮度周期候选不等于机械RPM。详细失败、人工负控制、首次越频带输出与曲线见<a href="../stage79_native_frequency_20261021/第七十九阶段逐像素频率与SDK接口诊断.html">79完整报告</a>。</p>
<h2>86：处理变快后，仍要考虑输出是否过时</h2>
<p class="limit">离线源时间上的0.4秒预测误差，不等于实际发布后仍提前0.4秒。单步处理时间超过预测提前量时，目标时刻可能已过去；若50Hz到达却不丢帧，排队还会持续增长。下面只是记录耗时的乐观单服务FIFO推算，未做新的实时摄像机测试；decode/JSON/预览/冷启动尚未计入，所以不能当实际延迟测量。</p>
<table><tr><th>片段</th><th>仅计算乐观fps</th><th>计算/源时间倍数</th><th>有效0.4s预测单步能按时</th><th>乐观FIFO还能按时</th></tr>{''.join(deadline_rows)}</table><img src="deadline_audit.png">
<p>实时下一步应是有界队列、解码/NN/桨叶异步、基于真实观测时间的更新及跳帧条件下因果验证。不能直接复用这里已算好的预测来声称跳帧后准确率不变；跳帧会改变学习、周期缓存和状态，需新实验。<a href="../stage86_latency_deadlines_20261021/audit86.json">86全部记录与边界</a>。</p>
<h2>预测精度：沿用78，所有失败仍成立</h2><p>本轮数学输出精确保持，所以不重挑专家或重复宣称误差改善。原0.4s主审阅分支clear13.75、side63.38、train397.44px；额外2号60.50差于CV46.97，额外4号55.86差于强历史49.91。匹配常量经常接近或更好，未证明RPM独立增益、未知物体泛化或真实姿态/推力准确性。<a href="../stage78_RAW_separated_forecast_20261021/第七十六至七十八阶段状态隔离与预测修正报告.html">78稀疏真值对照与全部失败</a>。</p>
<h2>固定画面与新入口视频</h2><p>以下展示当前检测和已保存预测，不把未来检测框当独立真值。完整播放为源50fps。新入口<code>run_batched_forecast.py</code>；默认54和旧入口保留。</p>{videos}
<h2>可复现记录与交接</h2><p>83首版清晰100帧数值完成后，外层包装在审计钩子命名空间清理后写元数据失败，已保留原源/协议/事后检查；修复后在v2目录完整重跑，未用失败批次冒充成功。没有安装新依赖，没有读取/修改yolov10-main，没有G重建输入。</p>
<p><a href="protocol.json">85冻结协议</a> · <a href="checks85.json">85完整RAW检查</a> · <a href="../stage84_batched_rollout_20261021/checks84.json">84严格模块检查</a> · <a href="../stage84_batched_rollout_20261021/benchmark84.json">84基准</a> · <a href="../stage82_streamed_window_20261021/benchmark82.json">82基准</a> · <a href="../stage83_RAW_streamed_window_20261021/failure_postmortem83.json">83首版失败</a> · <a href="../../event_drone_forecast/research/stage85/README.md">入口说明</a> · <a href="../../WORKLOG.md">工作日志</a></p></html>''',encoding='utf-8')
    data=dict(report=str(report),helper_sha256=sha(__file__),checks={n:dict(parent=c['parent'],native10=c['native10'],video_frames_decoded=c.get('video_frames_decoded')) for n,c in checks.items()},
        timing_table=[{k:r[k] for k in ['stage','case','median_processing_ms','optimistic_no_decode_processing_fps','source_time_realtime_factor','lead_deadlines']} for r in audit['table']],
        benchmark80=bench80,benchmark82=bench82,benchmark84=bench84,fixed_frames=fixed,all49_exact78=True,accuracy_changed=False,
        default_promoted=False,stable_high_accuracy_realtime_achieved=False,event_RPM_advantage_proven=False)
    (OUT/'delivery_data85.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(report,flush=True)
if __name__=='__main__':run()
