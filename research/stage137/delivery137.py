"""One report for the measured code optimization, not new prediction tuning."""
from pathlib import Path
import html,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT.parent;OUT=WORK/'output/stage137_fast_geometry_20261024';OLD=WORK/'output/stage135_optimized_forecast_20261024'
def load(p):return json.loads(Path(p).read_text('utf-8-sig'))
def rows(p):return [json.loads(s) for s in Path(p).read_text('utf-8').splitlines() if s]
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),'utf-8')
def table(rr,columns):
    def fmt(v):return f'{v:.3f}' if isinstance(v,float) else str(v)
    return '<table><tr>'+''.join('<th>'+html.escape(t)+'</th>' for _,t in columns)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(fmt(r.get(k,'')))+'</td>' for k,_ in columns)+'</tr>' for r in rr)+'</table>'

def main():
    checks=[load(OUT/name/'checks137.json') for name in ['legacy100','natural6','side50','negative5']];current=rows(OUT/'natural6/predictions.jsonl');control=rows(OUT/'legacy100/predictions.jsonl');first=current[:len(control)];assert len(first)==len(control)==100
    timing=[]
    for field,label in [('processing_ms','整步处理'),('geometry_ms','几何合计'),('frontend_ms','轮廓前端'),('model_ms','结构拟合'),('NN_ms','GPU检测')]:
        before=float(np.median([r[field] for r in control]));after=float(np.median([r[field] for r in first]));timing.append(dict(stage=label,before_ms=before,after_ms=after,reduction_percent=100*(before-after)/before))
    bytime={round(r['t'],8):r for r in current};previous=load(OLD/'actual_RAW_scores136.json');scores=[];pairs=[]
    for p in previous['pairs']:
        if not p['published']:continue
        r=bytime[round(p['origin'],8)];assert r['valid'];c=r['coefficients']['optimized_geometry'];h=p['actual_h'];pred=np.array(c['position'])+h*np.array(c['velocity'])+.5*h*h*np.array(c['acceleration']);err=float(np.linalg.norm(pred-p['truth_bbox_center']));pairs.append(dict(h=p['h'],origin=p['origin'],target=p['target'],previous_px=p['new_error_px'],current_px=err))
    for h in [.1,.2,.4]:
        pp=[r for r in pairs if r['h']==h];before=float(np.mean([r['previous_px'] for r in pp]));after=float(np.mean([r['current_px'] for r in pp]));assert abs(before-after)<1e-4;scores.append(dict(horizon_s=h,pairs=len(pp),previous_px=before,current_px=after))
    completion=load(OUT/'natural6/completion.json');jac=load(OUT/'jacobian_checks137.json');gap=next(r for r in timing if r['stage']=='整步处理');plot=timing[:4]
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained');xx=np.arange(len(plot));width=.35
    for offset,key,label,color in [(-width/2,'before_ms','原几何后端','#93a1ae'),(width/2,'after_ms','批量导数＋共享掩膜','#16857e')]:
        bars=ax.bar(xx+offset,[r[key] for r in plot],width,label=label,color=color);ax.bar_label(bars,fmt='%.1f',padding=3)
    ax.set_xticks(xx,[r['stage'] for r in plot]);ax.set_ylabel('中位耗时 / 毫秒');ax.set_title('同一入口、同一 RAW 的前 100 步：不改变预测逻辑');ax.legend();ax.set_ylim(0,max(r['before_ms'] for r in plot)*1.27);ax.spines[['top','right']].set_visible(False);fig.savefig(OUT/'latency137.png',dpi=160);plt.close(fig)
    report=OUT/'几何计算加速与预测结果.html'
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>几何计算实际加速</title><style>body{{max-width:1120px;margin:32px auto;padding:0 24px;font:17px/1.8 system-ui;color:#213847}}h1,h2{{line-height:1.4}}h2{{margin-top:2em}}table{{border-collapse:collapse}}th,td{{border:1px solid #c8d8de;padding:9px;text-align:left}}th{{background:#edf5f6}}img,video{{max-width:100%;height:auto}}.good{{background:#e5f4ed;padding:18px}}.limit{{background:#fff1d6;padding:18px}}a{{color:#007b88}}code{{background:#edf3f6}}</style>
<h1>137：减少几何计算，预测精度保持</h1><p>2026-10-24｜继续优化实际运行代码。本轮不调预测系数、不扩张历史控制，只比较旧/新几何后端。</p>
<p class="good">同入口前100个20ms时刻，整步处理中位耗时从{gap['before_ms']:.1f}降到{gap['after_ms']:.1f}ms，减少{gap['reduction_percent']:.1f}%。新后端完整8秒400步为{completion['processing_median_ms']:.1f}ms/步；未来框预测误差保持约30.83px（提前0.4秒，同11对）。</p>
<h2>具体改了三处</h2><ol><li><b>批量计算固定结构拟合的导数。</b>旧117/113按每个桨盘和每个参数循环，还计算固定尺寸下用不上的半径/长宽比导数。新实现按四个桨盘和八个单应参数做同一公式的数组运算，只输出实际需要的八列。优化器、目标函数、初始化、采样点和迭代上限不变。</li>
<li><b>计数分支复用124已实现的掩膜缓存。</b>原前端已经得到当前Gaussian密度与阈值掩膜，防合并步骤又重算一次。直接使用既有count_global缓存路径，继续使用原全图阈值和完整轮廓。</li>
<li><b>占据分支跳过没有使用的全图分位数。</b>它最终使用固定0.5支持阈值，旧通用函数仍会提取全图正值并计算分位数。删去这两个无效计算，保留同一个占据阈值、形态学、ROI中心资格和模型。</li></ol>
<p>没有直接裁切ROI：全图阈值依赖背景统计，ROI截断也会改变轮廓。此次先保留观测语义，再减少冗余。原1280×720、20ms每步处理、边界完整性和五帧建模规则均保留。入口新增<code>--geometry-backend legacy</code>可运行同一版本的旧后端；默认fast为本轮优化。</p>
<h2>实际耗时</h2><img src="latency137.png">'''+table(timing,[('stage','处理部分'),('before_ms','旧后端/ms'),('after_ms','新后端/ms'),('reduction_percent','减少/%')])+f'''
<p>上表使用同一新入口、同一22—24秒RAW的共同100步；先跑旧后端100步，再跑新后端完整400步，取前100步比较。均开启共享帧校验，未对推理做cProfile。系统负载没有受控，不能当作跨硬件基准；各阶段中位数不能相加。整步处理不包含RAW解码、写记录/视频或冷启动，后者另有日志。</p>
<p>新后端完整400步的序列化/视频中位{completion['serialization_preview_median_ms']:.1f}ms，包含冷启动、读取和视频输出的总运行时间{completion['wall_including_cold_start_s']:.1f}秒。仍超20ms，<b>没有达到实时</b>。</p>
<h2>预测有没有因加速变坏</h2>'''+table(scores,[('horizon_s','提前量/秒'),('pairs','共同发布端点'),('previous_px','136误差/px'),('current_px','137误差/px')])+f'''
<p>新400步重新读取原始RAW，机体观测、前端桨盘输出和当前几何资格与136比较。最大未来坐标差{checks[1]['maximum_forecast_difference_px']:.3g}px，属于浮点运算次序差异；精度改进仍来自上一轮135算法，本轮收益是计算速度。0.1秒启动期1对未发布、0.2秒原1对缺失的覆盖口径与136完全相同；没有根据误差重新选择评分点。</p>
<p class="limit">这些仍是已用于开发的清晰录像与稀疏人工框中心，不是重心/姿态/转速真值或独立泛化成绩。侧视1秒只确认入口能运行并保持原预测，不代表侧视法线或推力预测改善。</p>
<h2>实际RAW预测视频</h2><video controls preload="metadata" poster="natural6/preview.png" src="natural6/predictions.mp4"></video><p>绿：优化预测；灰：机体匀速。50fps为离线播放。预测逻辑仍是机体绝对位置＋相对几何速度＋有结构支持时的稳健加速度外推，没有新增真实推力标定。</p>
<h2>复用依据与必要检查</h2><p>数值优化继续使用<a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html">SciPy现有最小二乘及可调用Jacobian接口</a>；批量数组计算使用<a href="https://numpy.org/doc/stable/user/basics.broadcasting.html">NumPy广播/向量化</a>。掩膜共享直接复用项目124实现，非新轮廓检测算法。检查过OpenCV线程接口和threadpoolctl，但本轮未新增依赖或更改线程配置；这项工程加速不称论文创新。</p>
<p>26组实际拟合点的导数与旧实现最大差{jac['old_new_max_abs_difference']:.3g}，另与独立中心数值差分方向导数相符。交替同输入调用的单次导数中位{jac['old_median_ms']:.3f}→{jac['new_median_ms']:.3f}ms。实际RAW：旧100步、新清晰400步、侧视50步、空场30步；空场无有效预测，cutoff和不读缓存/标注通过，新视频400帧解码。</p>
<p>首次cProfile因相对入口文件名与源码指纹的绝对根路径不兼容，在启动NN前失败，空目录与日志保留；改用绝对路径完成100步剖析。入口已修复此路径兼容问题。带cProfile的292ms不是性能对照，未计入上表。</p>
<h2>下一步实际瓶颈</h2><p>几何拟合和全图支持前端仍占明显时间。下一项可把当前有效机体ROI用于共享局部支持，但必须保留原全图阈值统计、轮廓跨边界时回退全图，不能把被裁截的弧标成完整桨盘。侧视方向与周期的接入仍需单独完成，事件密度不直接等于转速或推力。</p>
<p><a href="summary137.json">全部耗时与误差</a> · <a href="natural6/protocol.json">当前入口源码指纹</a> · <a href="natural6/completion.json">完整运行记录</a> · <a href="natural6/checks137.json">实际预测数值比较</a> · <a href="../../event_drone_forecast/research/stage137/README.md">复现/交接</a></p></html>'''
    report.write_text(page,'utf-8');save(OUT/'summary137.json',dict(report=str(report),timing_common100=timing,full400_completion=completion,scores=scores,pairs=pairs,checks=checks,actual_RAW580_complete=True,profile_RAW100_additional=True,precision_improved_this_stage=False,realtime=False,side_physics_improved=False))
    print(json.dumps(dict(report=str(report),timing=timing,scores=scores)),flush=True)
if __name__=='__main__':main()
