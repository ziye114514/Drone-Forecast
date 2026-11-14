"""One outcome report for optimization130..136; score the actual RAW output."""
from pathlib import Path
import html,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT.parent
OUT=WORK/'output/stage135_optimized_forecast_20261024'

def load(p):return json.loads(Path(p).read_text('utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),'utf-8')
def rows(p):return [json.loads(s) for s in Path(p).read_text('utf-8').splitlines() if s]
def table(rr,columns):
    def fmt(v):return f'{v:.2f}' if isinstance(v,float) else str(v)
    return '<table><tr>'+''.join('<th>'+html.escape(label)+'</th>' for _,label in columns)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(fmt(r.get(key,'')))+'</td>' for key,_ in columns)+'</tr>' for r in rr)+'</table>'

def main():
    old=load(WORK/'output/stage128_RAW_independent_support_20261023/scores128.json');new=load(OUT/'optimization135.json')
    actual=rows(OUT/'raw/natural6_fast/predictions.jsonl');bytime={round(r['t'],8):r for r in actual};pairs=[];scores=[]
    for p in old['pairs']:
        if p['case']!='natural6':continue
        r=bytime[round(p['origin'],8)];c=r['coefficients']['optimized_geometry'];h=p['actual_h'];pred=np.array(c['position'])+h*np.array(c['velocity'])+.5*h*h*np.array(c['acceleration'])
        err=float(np.linalg.norm(pred-p['truth_bbox_center']));pairs.append(dict(p,published=r['valid'],error_px=p['predictions']['count_model']['error_px'],new_prediction=pred.tolist(),new_error_px=err))
    for h in [.1,.2,.4]:
        allp=[p for p in pairs if p['h']==h];pp=[p for p in allp if p['published']];base=next(r for r in old['table'] if r['case']=='natural6' and r['method']=='count_model' and r['h']==h)
        cached=next(r for r in new['table'] if r['case']=='natural6' and r['method']=='optimized_geometry' and r['h']==h);assert abs(float(np.mean([p['new_error_px'] for p in allp]))-cached['mean_px'])<1e-8
        value=float(np.mean([p['new_error_px'] for p in pp]));before=float(np.mean([p['error_px'] for p in pp]));scores.append(dict(horizon_s=h,paired=len(pp),eligible=base['eligible'],missing_total=base['eligible']-len(pp),additional_not_published=len(allp)-len(pp),old_px=before,new_px=value,reduction_percent=(before-value)/before*100))
    save(OUT/'actual_RAW_scores136.json',dict(source='raw/natural6_fast/predictions.jsonl',truth='Previously exposed sparse manual future bounding-box centers; not COM/pose/RPM; inference exited before scoring.',table=scores,pairs=pairs))
    for case in ['natural6','natural6_fast','prefix_natural6','negative5']:assert load(OUT/'raw'/case/'checks135.json')['predictor_exact135_replay']
    first=load(OUT/'raw/natural6/completion.json');latest=load(OUT/'raw/natural6_fast/completion.json');oldtime=load(WORK/'output/stage128_RAW_independent_support_20261023/raw/natural6/completion.json')
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained');xx=np.arange(3);width=.35
    for offset,key,label,color in [(-width/2,'old_px','旧计数模型','#8b9aa8'),(width/2,'new_px','新相对几何运动＋稳健外推','#16857e')]:
        bars=ax.bar(xx+offset,[r[key] for r in scores],width,label=label,color=color)
        ax.bar_label(bars,fmt='%.2f',padding=3)
    ax.set_xticks(xx,[f'{r["horizon_s"]:.1f} 秒' for r in scores]);ax.set_ylabel('未来框中心平均误差 / 像素');ax.set_ylim(0,43);ax.legend(loc='upper left');ax.set_title('清晰片段：同一批稀疏人工端点，实际 RAW 输出重新计分');ax.spines[['top','right']].set_visible(False);fig.savefig(OUT/'accuracy136.png',dpi=160);plt.close(fig)
    pp=[p for p in pairs if p['h']==.4];fig,ax=plt.subplots(figsize=(10,4.2),layout='constrained');tt=[p['origin'] for p in pp]
    ax.plot(tt,[p['error_px'] for p in pp],'o-',color='#8b9aa8',label='旧计数模型');ax.plot(tt,[p['new_error_px'] for p in pp],'o-',color='#16857e',label='新方法');ax.set_xlabel('发出预测时刻 / 秒');ax.set_ylabel('0.4 秒未来框误差 / 像素');ax.set_title('不是每个端点都改善：保留全部 11 个端点');ax.legend();ax.spines[['top','right']].set_visible(False);fig.savefig(OUT/'endpoints136.png',dpi=160);plt.close(fig)
    remaining=[]
    for case in ['extension1','extension2','continuous6']:
        a=next(r for r in old['table'] if r['case']==case and r['method']=='count_model' and r['h']==.4);b=next(r for r in new['table'] if r['case']==case and r['method']=='optimized_geometry' and r['h']==.4)
        remaining.append(dict(case=case,paired=b['paired'],eligible=b['eligible'],old_px=a['mean_px'],new_px=b['mean_px'],mode='无合格结构，保持机体匀速预测；本轮为固定观测重放'))
    matched=next(r for r in new['table'] if r['case']=='natural6' and r['method']=='matched_body' and r['h']==.4)
    report=OUT/'预测算法与精简RAW入口优化报告.html'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>预测算法与RAW入口优化</title><style>body{max-width:1150px;margin:32px auto;padding:0 24px;font:17px/1.8 system-ui;color:#203648}h1,h2{line-height:1.4}h2{margin-top:2em}img,video{max-width:100%;height:auto}table{border-collapse:collapse}th,td{border:1px solid #ccd9df;padding:9px;text-align:left}th{background:#edf4f7}.result{background:#e5f4ed;padding:18px}.limit{background:#fff1d6;padding:18px}a{color:#007a83}pre{background:#edf3f5;padding:16px;overflow:auto}code{font-size:.9em}</style>
<h1>这轮实际改了预测器和运行入口</h1><p>2026-10-24｜130—136合并为一份优化结果。输入仍是自采事件RAW和项目现有检测权重；不需要人工给定机体尺寸、质量或桨盘大小。无G重建，未访问或修改受保护的yolov10-main。</p>
<p class="result">已交付一条精简RAW预测入口：位置由机体检测维持，桨盘模型提供相对运动来修正速度，历史位置的稳健加速度只用于未来外推。清晰8秒片段的0.4秒未来框平均误差从33.68降到30.83像素；0.2秒从18.95降到11.31像素。</p>
<h2>代码层面改了什么</h2><ol><li><b>去掉第一帧桨盘坐标的永久位置锚点。</b>旧方案用首次几何与框中心对齐，此时的偏差可能长期进入位置预测。新方案在每个来源自己的短历史中计算几何中心速度，恒定坐标偏差在差分中抵消；机体框继续承担绝对位置观测。</li>
<li><b>把加速度放在外推层。</b>复用项目已有稳健二次拟合，用过去0.35秒位置估计加速度，按估计误差收缩并限幅；不让不确定加速度反复反馈污染滤波位置状态。预测形式为 p(h)=p+h·v+½h²·a。</li>
<li><b>模型学成前保留匀速支路。</b>两套经过边界守门的桨盘模型均未学成时，使用原机体匀速预测。学成后才启用更丰富的外推；只有当前真实合格的几何观测才能修正速度，形状记忆本身不算新测量。失去同一身份超过原0.5秒则重置。</li>
<li><b>新增可直接运行的精简入口。</b>同一20ms ON/OFF计数只算一遍；保留计数/占据各自的结构模型，只输出优化、同条件机体对照和匀速三路预测，去掉旧研究入口中几十路历史对照。结构学成后，跳过随后必被固定结构拟合替换的自由结构拟合。</li></ol>
<p>桨盘提取门槛、FOV完整性、五帧建模和缺桨资格未放宽。这里改进的是图像平面的运动估计，还没有完成“转速→推力→真实动力学”的标定链路；也不能把它称为侧飞推力预测已成功。</p>
<h2>预测精度实际变了多少</h2><img src="accuracy136.png">'''+table(scores,[('horizon_s','提前量/秒'),('eligible','可选目标端点'),('paired','共同发布端点'),('missing_total','未能配对'),('old_px','旧计数模型/px'),('new_px','新方法/px'),('reduction_percent','误差减少/%')])+f'''
<p>0.1秒原有12对中，最早22.14秒尚未累计五次当前机体观测，当前入口不发布，因此同时从两方法的误差均值排除，另计1次未发布；其余11对对比约15.89→8.58px。135固定系数重放中，包含该未发布起点时为16.07→9.37px。0.2秒原有12个可选端点中先前已缺1对；0.2/0.4秒原来配对的11对本次全部发布，未再减少。未因误差选择评分点。</p>
<p>不是只和弱匀速对照比较：上述旧方案是先前更好的count_model（33.68px），而非退步后的129位置融合（37.64px）。相同“结构学成才启用加速度”条件下，机体速度对照为{matched['mean_px']:.2f}px，新几何速度为30.83px；相对几何运动贡献不全来自增加加速度或结构开关。</p>
<img src="endpoints136.png"><p class="limit">以上均为已反复开发过的录像，评分用原有稀疏人工未来<b>框中心</b>，不是重心或飞控真值。实际RAW新输出已重新计分；同11个0.4秒端点，不能据此证明泛化、急转优势、统计显著性或论文创新性。保留所有评分端点，未因误差大删点。</p>
<h2>实际RAW预测视频</h2><video controls preload="metadata" poster="raw/natural6_fast/preview.png" src="raw/natural6_fast/predictions.mp4"></video><p>绿线：新方法的未来轨迹，末端为0.4秒；灰线：机体匀速对照。50fps是离线播放速度。图表中的旧count_model是更强的历史基准，与视频灰线不是同一方法。</p>
<h2>速度提升到哪里</h2><p>旧128研究入口的清晰段中位处理耗时为{oldtime['processing_median_ms']:.1f}ms/步；本轮精简入口两次完整400步为{first['processing_median_ms']:.1f}和{latest['processing_median_ms']:.1f}ms/步（后者为当前代码）。主要来自去掉几十路历史对照、共享计数，约3.7—4.2倍；这是不同负载时的运行记录，不是受控硬件基准。</p>
<p>当前完整运行：几何阶段中位{latest['geometry_median_ms']:.1f}ms，GPU检测{latest['gpu_median_ms']:.1f}ms，另有写记录/绘图{latest['serialization_preview_median_ms']:.1f}ms。加上结构拟合精简后本轮总耗时并未低于前一轮，不能单独宣称这项改动带来加速。仍远超20ms，<b>还没有实时</b>。</p>
<h2>哪些地方尚未改善</h2>'''+table(remaining,[('case','片段'),('paired','已评分端点'),('eligible','可选端点'),('old_px','旧0.4秒/px'),('new_px','新0.4秒/px'),('mode','实际行为')])+'''
<p>这些片段的原有观测完成了新算法重放，但本轮未重新跑完整RAW检测；不能与清晰400帧的实际全链路混为一谈。侧视仍缺乏可靠当前法线和可用转速，不宜强塞加速度换取看上去更复杂的预测。</p>
<h2>试过但没有采用的改动</h2><p>130仅自适应偏移约37.50px、仅相对几何速度约36.45px；131把加速度直接放入六维滤波状态反而约50.67—51.30px；132直接普遍启用加速度会使断续检测片段大幅变差；133延迟误差守门清晰约34.07px；134收缩/一致性变体约33.05px，另有片段回退。负结果保留小脚本和结果JSON，未扩展成多份独立报告。最终135采用有合格结构支持才启用的组合。</p>
<h2>直接运行与交接</h2><pre>&amp; 'D:\\drone forecast\\.venv-event311\\Scripts\\python.exe' `
  'D:\\drone forecast\\event_drone_forecast\\run_optimized_forecast.py' `
  --raw 'E:/dataset/train/6/media/event.raw' `
  --output 'D:/drone forecast/output/my_optimized_run' `
  --start 22 --duration 8 --preview</pre><p>输出目录必须尚不存在。输出逐时刻预测、有效性、当前几何来源、处理耗时和预览视频。无需传入历史实验输出或人工标注。当前固定1280×720/20ms，依赖本机现有SDK/检测权重与环境。默认54/API93没有改；新增入口可单独试用。</p>
<p>必要检查仅为确认改动确实进入RAW：新400帧预测与冻结135系数逐项一致；60帧实际RAW前缀一致；起始0秒空场完整30帧且不发布有效预测；事件均在发布时刻以前，推理不读取结果缓存/标注，视频400帧可解码。首次空场因热身跳过前4帧导致输出计数错误，目录/日志保留，已修复并完整重跑；不是一次成功的空场实验。</p>
<p>下一项实际优化应集中在：把几何求解放入机体ROI、降低重复全图处理和非必要重拟合的开销，同时保留轮廓边界；随后用当前模型约束侧视线法线，并把真实周期证据接入预测。先改变运行代码，再用一个固定短段决定是否保留，不再扩张历史对照树。</p>
<p><a href="actual_RAW_scores136.json">实际RAW逐点评分</a> · <a href="optimization135.json">全部开发片段与对照结果</a> · <a href="protocol135.json">冻结的预测算法</a> · <a href="raw/natural6_fast/protocol.json">当前RAW入口源码指纹</a> · <a href="raw/natural6_fast/completion.json">实际运行耗时</a> · <a href="../../event_drone_forecast/research/stage135/README.md">运行与交接说明</a></p></html>'''
    report.write_text(page,'utf-8');save(OUT/'delivery136.json',dict(report=str(report),actual_raw_frames=400,prefix_frames=60,negative_frames=30,raw_repeat_frames=400,latest_entry_stage=136,predictor_stage=135,old_raw_processing_ms=oldtime['processing_median_ms'],lean_v1_ms=first['processing_median_ms'],lean_current_ms=latest['processing_median_ms'],actual_scores=scores,side_improvement=False,realtime=False,independent_test=False))
    print(json.dumps(dict(report=str(report),scores=scores),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
