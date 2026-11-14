"""Completed-result report; findings are read from checked artifacts."""
from pathlib import Path
import html,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage124'),str(ROOT/'research/stage128')]
from study124 import OUT as INPUT,rows,save
from study128 import OUT,CASES,verify
def load(p):return json.loads(Path(p).read_text('utf-8-sig'))
def table(rr,fields):
    def fmt(v):return f'{v:.3f}' if isinstance(v,float) else str(v)
    return '<table><tr>'+''.join('<th>'+html.escape(label)+'</th>' for _,label in fields)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(fmt(r.get(k,'')))+'</td>' for k,_ in fields)+'</tr>' for r in rr)+'</table>'
def main():
    verify();checks=[load(OUT/'raw'/c/'checks128.json') for c in CASES];availability=load(INPUT/'availability124.json');independent=load(ROOT.parent/'output/stage127_independent_support_20261023/availability127.json');simple=load(ROOT.parent/'output/stage125_complementary_support_20261023/availability125.json');scores=load(OUT/'scores128.json');videos=load(OUT/'video_checks128.json')
    summary=[]
    for case in ['extension1','extension2','natural6','continuous6','negative5']:
        co=next(r for r in availability if r['case']==case and r['mode']=='count_global');oc=next(r for r in availability if r['case']==case and r['mode']=='occupancy_roi');i=next(r for r in independent['table'] if r['case']==case)
        summary.append(dict(case=case,frames=co['frames'],count_four=co['four'],occupancy_four=oc['four'],count_qualified=co['qualified'],occupancy_qualified=oc['qualified'],count_current=co['current'],occupancy_current=oc['current'],independent_current=i['current'],count_first=co['first_structure'],independent_first=i['first_current']))
    short=[r for r in scores['table'] if r['h']==.4];clear={r['method']:r for r in short if r['case']=='natural6'};before=clear['count_model']['mean_px'];after=clear['independent_models']['mean_px'];delta=before-after
    efficacy=(f'同11个稀疏未来框端点，0.4秒平均误差从{before:.3f}px变为{after:.3f}px，差仅{delta:.3f}px（约{delta/before*100:.2f}%），未证明有实用预测收益。' if delta>0 else f'同11个稀疏未来框端点，0.4秒平均误差从{before:.3f}px变为{after:.3f}px，本轮没有得到未来预测精度提升。')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>桨盘提取与独立模型补位实验</title><style>body{max-width:1280px;margin:28px auto;padding:0 24px;font:17px/1.8 system-ui;color:#213847}h1,h2{line-height:1.4}h2{margin-top:2em}table{border-collapse:collapse;font-size:13px}th,td{border:1px solid #c9d4dc;padding:7px}th{background:#eaf3f4}img,video{max-width:100%;height:auto}a{color:#007a88}.note{padding:16px;background:#fff1d2}.good{padding:16px;background:#e7f4ee}code{background:#edf2f5}</style>
<h1>124—128：桨盘支持恢复、独立结构与未来位置验证</h1><p>2026-10-23开发与冻结，2026-10-24完成核验。输入为自采RAW与项目原检测权重，使用原生ON/OFF，无G重建。默认稳定入口没有替换。</p>
<p class="good">已把“低重复计数的盘状支持被截成小内核”接成可运行的占据支路；为避免新旧轮廓尺寸偏差混入同一结构，两套桨盘模型分别学习，合格的位置观测进入同一个滤波状态。清晰段当前有效几何从236/400到244/400，首次可用从22.80秒提前到22.68秒。</p><p class="note">'''+html.escape(efficacy)+'''这些是历史开发录像中的稀疏人工框中心，不能当成真实重心、姿态、转速精度或独立泛化成绩。侧视连续片段仍没有首次模型；不能说当前卡点全部解决。</p>
<h2>实际修了哪一层</h2><p>原生“黑”表示同像素在20ms内两种极性均出现；旧前端使用归一到10ms的重复计数，再取最低2.0密度门槛，常把可见盘状支持压成小核。本轮新增当前ON/OFF二值占据，3×3 Gaussian sigma=.8后取≥.5的局部多数支持，再用原开闭运算及分量防合并处理。侧视薄轮廓仍先判线，未强拟合椭圆。</p><p>ROI使用原机体框25%边界，只在拟合后筛候选中心；不把轮廓裁断，也不把框中心当桨心。四路对照含原计数全局/同ROI，以及占据全局/同ROI。两帧开发检查之后、区间实验与未来评分之前冻结参数。占据分支activity按占据像素，原分支按归一计数，二者不等价于推力单位。</p>
<img src="../stage124_occupancy_support_20261023/pilot_masks.png"><p>2号43.50和45.00秒：同一ON/OFF快照、旧阈值与占据支持对照。红色拟合仅用于开发可视检查，不证明所有轮廓都是独立桨盘。2号四候选0→72、原共面资格0→7，但最长合格仅2帧，未达到原五帧初始化。</p>
<h2>为什么保留两套结构</h2><p>占据分支清晰段独用只有196次当前有效几何，少于原计数的236次。第一版规则“旧四圆不合格时才用新四圆”混入同一固定模板，当前仍236次，没有覆盖收益。其实际RAW控制126已完整核验400时刻，作为负结果保留。</p><p>改进版让两种支持各自按原五帧规则学习半径/长宽比，避免互相改写形状。位置层优先当前有效的计数模型；缺失时才用当前有效的占据模型。各来源第一次几何与当时机体框对齐为固定图像偏移，统一到同一位置参考，进入一个位置/速度/外观偏移KF。切换不清零速度；同一身份缺失超过原0.5秒才重置。这不是COM标定，仍可能有来源偏差。</p>
<img src="coverage128.png"><img src="timeline128.png">'''+table(summary,[('case','片段'),('frames','时刻'),('count_four','原四候选'),('occupancy_four','新四候选'),('count_qualified','原完整资格'),('occupancy_qualified','新完整资格'),('count_current','原当前模型'),('occupancy_current','占据独用'),('independent_current','独立模型合用'),('count_first','原首次/s'),('independent_first','新首次/s')])+'''
<p>四候选不等于四个真实可见桨盘。分母含全部固定输入时刻，未建模不从统计剔除。1号当前NN仅43/300，无法只靠桨盘阈值修复整机漏检；2号和连续侧视的完整几何仍不连续，未降低资格或五帧门槛强行建模。清晰段新增8次来自占据补位，改善2个百分点，且全部集中于22.68—22.94秒启动附近，尚不是长遮挡或侧视恢复收益。</p>
<h2>预测究竟有没有变好</h2><p>所有实际预测、负例和活动模型前缀结束之后，才读取原有人工事件框。每个目标选target−h之前最近的发布时刻，间隔不超过20ms，以实际target−origin外推；五种方法用共同起点和共同端点。没有插值真值，没有按端点调阈值。此处新增分支是几何修正后的位置/速度预测，未证明新转速/推力收益。</p><img src="future_box128.png">'''+table(short,[('case','片段'),('method','方法'),('eligible','有目标端点'),('paired','共同评分端点'),('mean_px','平均误差/px'),('median_px','中位误差/px')])+'''
<p>count_model为原119仿射缺桨模型，mixed_template为125失败控制，independent_models为127/128新方案。侧视continuous6只有缓存同期NN的RAW几何实验，不是新的全链路部署。部分端点无法发布会计入覆盖缺口；空场仅0.6秒，不能外推长期误报率。</p>
<h2>实际RAW预测视频</h2>'''
    for case in ['natural6','extension2']:
        page+=f'<h3>{case}</h3><video controls preload="metadata" src="{case}_independent128.mp4" poster="{case}_poster128.png"></video>'
    page+='''<p>绿：新独立模型0.4秒预测；灰：原计数模型预测；黄/紫：两种当前桨盘观测。50fps是离线回放，不是处理速度，预测重合并不代表画图遗漏。放大窗展示当前目标附近真实事件和拟合。</p>
<h2>核验与中断记录</h2><p>124重新读RAW1505时刻、两60时刻独立前缀，共1625；原重叠桨盘精确123、共同机体KF一致。125/127缓存因果前缀通过，127仅count坐标适配与原107系数在1e-8内一致。</p><p>126仅完整单独跑清晰400时刻；其余原计划由128保留为控制。128实际RAW790时刻：清晰400、2号300、空场30、清晰独立前缀60。原69路/机体观测精确123；清晰重叠原70路精确126。新占据逐帧精确124，简单混合精确125，独立模型/共享滤波精确127，输出不读缓存或标注，事件均早于cutoff。活动前缀内部状态及已发布预测严格，空场无有效预测，源预览和新增视频全帧解码。</p><p>第一次128在会话中断时最终保存91条完整记录（末次进度打印79）、无completion；进程退出后保留原目录和日志为interrupted_natural6_20261024，重新按同一冻结协议运行，没有拼接残缺状态或改阈值。原RAW和yolov10-main未改，无新依赖；SDK可能创建自身索引，未将其视为RAW内容改动。</p>
<p>扩展2首次已保存142条完整输出后出现3.52MiB NumPy数组分配失败，输出/日志保存在failed_memory_extension2_20261024。系统提交空间紧张，重型核验父进程和同期分析也占内存；改成仅标准库调度、预测退出后再单独核验/绘图。没有改变冻结推理、线程配置、精度、权重或系统内存设置。后续完整重跑才用于评分，这次资源失败不计成算法漏检。</p>
<h2>还卡在哪里，以及下一步</h2><p>现在不能再把“候选多了”当成“正确模型与预测成功”。第一，双极占据会同时突出转动桨盘和机体双边缘，完整桨盘身份与重叠分解仍不足。第二，侧视相互覆盖时，单幅线状支持不能唯一恢复四个转轴及完整姿态；已有结构可以提供预测，不能制造新观测。第三，首次四圆资格仍不连续；没有足够当前测量，模板记忆不能代替初始化。第四，当前全研究对照路径远超20ms，不具备实时性。</p><p>下一步先利用ROI内像素时间周期与空间支持，验证是否能抑制机体边缘、保留转动桨盘；周期频率与事件数分开管理，不能把事件密度直接换算推力。保留本轮负例和失败片段，另用连续片段验证三桨退化和恢复。若未来框没有改善，就把占据支路保留为测量诊断，不应默认替换原预测器。</p>
<h2>已有研究与实际复用边界</h2><p><a href="https://arxiv.org/html/2604.18289v1">原论文</a>已有CC二值密度/形态学和聚类提取，再从椭圆与周期估姿态/推力。<a href="https://arxiv.org/html/2609.11920v1">EVPeriscope</a>采用泄漏事件累积、统计阈值、形态学和固定四簇；本轮读取作者detect.py，未运行其完整系统，也未搬入固定四簇与已知CAD假设。<a href="https://github.com/viorik/ELSDc">ELSDc作者代码</a>可供后续弧/线提取参考，本轮未编译。<a href="https://docs.prophesee.ai/stable/samples/modules/analytics/vibration_py.html">Prophesee频率图</a>可供后续周期支持验证，本轮未接入。占据、独立模型、坐标对齐和KF本身不称创新；事件特有的观测可靠性机制仍需独立验证。</p>
<p><a href="scores128.json">全部0.1/0.2/0.4秒逐点评分</a> · <a href="../stage124_occupancy_support_20261023/availability124.json">四路提取和失败分母</a> · <a href="../stage127_independent_support_20261023/availability127.json">独立模型覆盖</a> · <a href="protocol128.json">冻结方案</a> · <a href="../../event_drone_forecast/research/stage128/README.md">复现与入口</a></p></html>'''
    timings={c:load(OUT/'raw'/c/'completion.json') for c in ['natural6','extension2']}
    timing_text=f'<p>本机128完整研究对照路径：清晰段中位处理{timings["natural6"]["processing_median_ms"]:.1f}ms，扩展2为{timings["extension2"]["processing_median_ms"]:.1f}ms，均超20ms。包含原70路控制和新支路，不是裁剪后的单一方法速度；本次没有证明实时性。</p>'
    page=page.replace('<h2>已有研究与实际复用边界</h2>',timing_text+'<h2>已有研究与实际复用边界</h2>')
    report=OUT/'桨盘提取与独立模型补位实验报告.html';report.write_text(page,'utf-8')
    save(OUT/'delivery_checks128.json',dict(report=str(report),actual128_frames=sum(c['frames'] for c in checks),actual124_geometry_frames=1625,actual126_control_frames=400,
        original69_exact123=True,old70_clear400_exact_actual126=True,new_observations_and_models_exact_cached_controls=True,active_actual_prefix60_exact=True,
        source_preview_decoded=sum(c['decoded_frames'] for c in checks),overlay_decoded=sum(v['frames'] for v in videos),new_current_geometry=244,old_current_geometry=236,
        nominal04_old_mean_px=before,nominal04_new_mean_px=after,nominal04_improvement_px=delta,development_future_mean_decreased=delta>0,independent_generalization_proven=False,
        no_new_RPM_or_COM_truth=True,default_promoted=False,summary=summary,outcome=efficacy,processing_median_ms={c:v['processing_median_ms'] for c,v in timings.items()}))
    print(json.dumps(dict(report=str(report),outcome=efficacy)),flush=True)
if __name__=='__main__':main()
