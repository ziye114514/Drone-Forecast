from pathlib import Path
import html,json
from experiments import OUT,ROOT,save

def number(x):return '—' if x is None else f'{x:.3f}'
def table(headers,rows):
    return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</table></div>'

def main():
    module=json.loads((OUT/'module_analysis.json').read_text());raw=json.loads((OUT/'raw_analysis.json').read_text());primary=[]
    for c in ('clean','one_arc120','two_arcs180','one_wrong_plane','missing_one','missing_two','all_arcs120'):
        for method in ('frozen_up_prior','joint_all','joint_audit'):
            r=next(r for r in module['multi'] if r['condition']==c and r['method']==method and float(r['focal_scale'])==1.)
            primary.append([c,method,f"{r['accepted']}/{r['cases']}",number(r['median_error_deg']),number(r['p95_error_deg']),r['accepted_wrong_over10deg']])
    singles=[[r['method'],r['arc_deg'],f"{r['accepted']}/{r['cases']}",number(r['median_error_deg']),f"{r['accepted_wrong_over10deg']}/{r['accepted']}"] for r in module['single']]
    calibration=[[r['method'],r['focal_scale'],f"{r['accepted']}/{r['cases']}",number(r['median_error_deg']),r['accepted_wrong_over10deg']] for r in module['multi'] if r['condition']=='clean']
    temporal=[[r['method'],r['horizon_s'],number(r['mean_error_px']),number(r['seed_sd_px'])] for r in module['temporal']]
    metrics=[]
    for case in ('train6','side6','train2'):
        for arm in ('frozen','consensus','consensus_audit'):
            for method in ('cv','paper_adaptive','paper_fused'):
                r=next(r for r in raw['metrics'] if r['case']==case and r['arm']==arm and r['method']==method and r['horizon_s']==.4 and r['subset']=='all')
                metrics.append([case,arm,method,f"{r['available']}/{r['targets']}",number(r['error_px']),r['physics_at_scored_anchors'],r['cv_equal_endpoints']])
    admission=[[r['case'],arm,r['frames'],c['active'],c['physics'],r['updates'][arm]] for r in raw['summaries'] for arm,c in r['arms'].items()]
    four=[[r['case'],r['arm'],f"{r['available']}/{r['targets']}",number(r['error_px'])] for r in raw['metrics'] if r['horizon_s']==.4 and r['subset']=='four_visible' and r['method']=='paper_fused']
    pair=module['temporal_pair'];summary=dict(single_cases=json.loads((OUT/'single_completion.json').read_text())['cases'],
        multi_cases=json.loads((OUT/'multi_completion.json').read_text())['cases'],temporal_seeds=10,
        raw_frames=sum(r['frames'] for r in raw['summaries']),raw_events=sum(r['events'] for r in raw['summaries']),
        default_upgrade_selected=False,real_pose_accuracy_claimed=False,raw_fused_04s=[r for r in raw['metrics'] if r['horizon_s']==.4 and r['subset']=='all' and r['method']=='paper_fused'])
    save(OUT/'stage3_summary.json',summary)
    (OUT/'LITERATURE.md').write_bytes((Path(__file__).parent/'LITERATURE.md').read_bytes())
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第三阶段：多桨法向与残弧置信度</title>
<style>body{{margin:0;background:#f3f6f8;color:#1d2b36;font-family:"Microsoft YaHei",sans-serif;line-height:1.8}}main{{max-width:1260px;margin:auto;padding:28px}}section{{background:white;border:1px solid #dce4e9;padding:22px;margin:20px 0;border-radius:10px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #dce4e9;text-align:left}}th{{background:#eef3f6}}.scroll{{overflow:auto}}.note{{background:#fff0d7;border-left:4px solid #d3a04b;padding:18px}}a{{color:#126b95}}img,video{{max-width:100%}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style><main>
<h1>第三阶段：多桨法向消歧与残弧置信度</h1><p>2026-10-16 · 已有思路的强对照与独立实现 · 无真实内参的RAW不冒充已标定姿态</p>
<div class="note"><b>结论：</b>已知标定、完整四桨的合成投影中，共同法向能明显减少初始化双解错误；残弧敏感性门控仍存在“稳定但错误”的误接收。真实RAW的二维预测没有稳定改进：一段略好，另两段略差，改善幅度也很小。因此保留为研究分支，未替换默认预测器。</div>
<p><a href="protocol.json">事前模块协议</a> · <a href="raw_v2/protocol.json">实际RAW对照协议</a> · <a href="stage3_summary.json">结果概要</a> · <a href="LITERATURE.md">论文/GitHub核查</a> · <a href="raw_metrics.csv">全部RAW分组成绩</a> · <a href="../stage2_module_validation_20261016/第二阶段实验报告.html">第二阶段</a></p>
<section><h2>实现了什么？为什么先做它们？</h2><p>首轮冻结后运行{summary['single_cases']}组单圆/残弧、{summary['multi_cases']}组多桨/内参扰动，另有10种子动态观测测试。单圆姿态有多个法向候选；旧实现先逐桨用图像向上选解再融合，容易在初始化时选错。新对照联合考察至多16种四桨候选组合，以共同平面法向、基础离群检查和过去估计作选择，不输入机体尺寸或真实姿态。</p>
<p>置信度对照结合局部残差灵敏度和连续块删除后的拟合变化；输出是敏感性诊断，尚非经过覆盖率校准的概率置信区间。单圆评分仍用“最接近真值候选”作下界；多桨评分则是算法实际选择的解，二者不可混称。</p>
<p>这些构件已有研究：<a href="https://arxiv.org/abs/1804.04922">单圆姿态歧义</a>、<a href="https://link.springer.com/article/10.1186/s41074-018-0050-y">共面二次曲线歧义</a>、<a href="https://arxiv.org/abs/2207.07953">不确定性加权多对应位姿</a>。本轮不将它们包装为新算法，相关代码与适用条件见核查文件。</p></section>
<section><h2>已标定多桨：实际选出的法向是否更准？</h2><p>四个圆由独立三维相机投影生成，共面且等半径，位置偏移为±2R，但布局与尺寸没有送给估计器。已知真实K只是这一模块测试条件。覆盖噪声、目标大小、离轴、残弧、缺桨和一个错误平面。每条件240组是参数网格和噪声重复，不是实飞次数；不同方案接受集合不同。</p>
<img src="figures/multi_normal.png">
{table(['情形','方法','接受/总数','法向误差中位°','P95°','接受但>10°'],primary)}
<p>完整四桨：原向上选解中位21.320°，直接共同法向0.115°；门控并非总更优，干净情形直接共同法向没有>10°错误，门控版本仍有1例。一个错误平面时，共同法向直接融合仍可能被带偏，置信度与离群检查以拒绝部分样本换取较低接受误差。缺两桨时，静态保守方案拒绝输出；动态方案才允许借助过去状态接续。</p>
<h3>内参不准会怎样？</h3><p>下面将假定焦距扰动±20%，主点和像素纵横比仍正确。不能以“粗略内参有时可用”推断任意默认内参都足够。未把这些合成K写入真实RAW配置。</p>
{table(['方法','假定焦距/真焦距','接受/总数','误差中位°','接受但>10°'],calibration)}</section>
<section><h2>残弧：诊断稳定不代表椭圆正确</h2><p>单圆测试加入可见弧180/120/60°，以及闭合遮挡切边。下表统计接受率和已接受样本中错误比例；不只报告拒绝后变小的平均误差。近线段不计为完整姿态。</p>
<img src="figures/arc_risk_coverage.png">
{table(['方法','可见弧°','接受/总数','候选误差中位°','接受且>10°/接受'],singles)}
<p><b>明确失败：</b>60°残弧中，原拟合经敏感性门控仍接受13例，13例全错超过10°；120°亦有同类问题。错误的椭圆可以对删点很稳定，局部残差也小。这不能靠简单把某个自助抽样分位数命名为置信度解决。四桨共同结构或可靠历史可能增加约束，但本轮尚未验证它们能覆盖所有短弧、切边与相关误差。</p></section>
<section><h2>动态机制测试：姿态变准能带来多少预测变化？</h2><p>120步、10个噪声种子，同一预设像素运动；当前真实位置/速度与有效推力增益已知，只有法向需要估计。先完整桨盘，再错桨、缺两桨、短弧，最后恢复。预测只使用当前/过去观测，60步前缀与完整运行精确一致。这是隔离姿态作用的诊断，不是只输入RAW的部署成绩，也没有证明未知质量/推力系数的恢复。</p>
<img src="figures/temporal_mechanism.png">
{table(['方法','预测秒数','平均误差px','种子间标准差px'],temporal)}
<p>0.4秒新方法减原法向方法为{pair['new_minus_frozen_error_px']:.4f}px，种子配对自助抽样95%区间{html.escape(str(pair['paired_seed_bootstrap95']))}。量级很小；区间只描述此轨迹上的噪声重复，不能推广为真实飞行的显著提升。</p></section>
<section><h2>真实RAW：共享位置与转速，只改法向和准入</h2><p>三个预选片段：train6前30秒、train6侧视108–114秒、train2前30秒，总{summary['raw_frames']}帧、{summary['raw_events']}事件。各臂使用完全相同的位置观测、转速和CV状态；train6/train2的原版分支还与先前冻结运行的全部预测状态逐项相同。侧视为108秒冷启动，只比较本轮同起点三臂。</p>
<p>真实RAW没有提供K，所以只运行弱透视轴向共识并沿用时间/图像向上先验；不声称真实透视法向双解已经解决。三臂为原法向、共同轴向、共同轴向加置信度门控。保留CV作为强基线。新分支更换完整法向滤波/准入实现，包括测量噪声处理；未学习原版未标定的电机差分映射，其兼容字段报告0。因此这不是只替换一个选解函数的单因素消融，真实成绩变化不能全部归因于候选选择。</p>
{table(['片段','方案','分支','可用/标注','0.4秒误差px','评分锚点物理启用数','与CV相同端点数'],metrics)}
<p>train6融合误差74.285→73.849px，约降低0.59%；加门控74.341px反而略差。侧视与train2的两种改法均略差。train6的CV为73.940px；因此不能只拿融合的轻微波动宣称物理预测已稳定获益。标签少且时间相关，录像也用于过往开发，不是盲测成绩。</p>
<h3>四桨当前可见的子集</h3>{table(['片段','方案','可用/四桨锚点标注','融合0.4秒误差px'],four)}
<h3>启用和在线更新是否被门控改变？</h3>{table(['片段','方案','总帧','追踪有效帧','物理帧','自适应更新次数'],admission)}
<p>三臂跟踪覆盖率相同；物理分支的启用数不同。新方法若退回CV，必须计入解释，不能称之为更准的姿态/推力。空场景诊断也保持原结果，见raw_analysis.json。</p></section>
<section><h2>看真实0.4秒后的画面</h2><p>左上当前RAW，其他三格均为真实未来RAW。紫色融合预测、橙色CV预测，来源是此前已经发布的状态。四格使用由当前跟踪中心决定的同一裁剪，未用未来目标位置居中；预测出框或无输出照常保留。绿色椭圆为当前观测，不是真值。</p>
<video controls preload="metadata" poster="raw_v2/train6/comparison_frame39.png" src="raw_v2/train6/actual_future_comparison.mp4"></video>
<p>侧视与不完整观测：</p><video controls preload="metadata" poster="raw_v2/side6/comparison_frame39.png" src="raw_v2/side6/actual_future_comparison.mp4"></video></section>
<section><h2>当前交付与下一步</h2><p>保留run_paper.py及前两阶段入口。run_normals.py是三臂研究入口，尚不选默认升级：</p>
<pre>&amp; 'D:\\drone forecast\\.venv-event311\\Scripts\\python.exe' 'D:\\drone forecast\\event_drone_forecast\\run_normals.py' --raw 'E:\\dataset\\train\\6\\media\\event.raw' --duration 30 --output 'D:\\drone forecast\\output\\my_normal_ablation'</pre>
<p>目前的证据支持：已知标定的多圆几何能解决部分选解错误；真实输入中的姿态可靠性、相机标定缺口和预测器对物理信息的利用仍需分别处理。下一步应验证真实数据上可辨识的相机/共同平面约束，以及短弧错误的结构性检查，再与在线参数学习联合评估。不是继续叠加未经验证的置信度名称。</p>
<p>首轮RAW组合因日志接口缺少differential_gain字段停止，raw/保留失败输入与源码快照；修复仅在适配层报告该未启用增益为0，raw_v2/另行冻结重跑，没有改变测量公式或阈值。当前计时含并发运行与三臂/审计开销，未达到实时，也不用于公平算法速度排名。</p>
<p><a href="single_cases.csv">逐单圆结果</a> · <a href="multi_cases.csv">逐多圆结果</a> · <a href="temporal_seed_metrics.csv">逐种子预测成绩</a> · <a href="raw_endpoint_details.csv">逐真实端点误差</a></p></section></main></html>'''
    (OUT/'第三阶段实验报告.html').write_text(doc,encoding='utf-8')
    (OUT/'阶段报告.md').write_text('# 第三阶段结果\n\n已完成1620组单圆/残弧、2160组多圆/内参扰动、10种子动态诊断，以及三个RAW片段共6600帧。\n\n'+
        '已标定完整四桨的法向选解明显改善；短弧置信度存在严重误接收。真实RAW的0.4秒融合误差：train6原74.285px，共识73.849px，共识+门控74.341px；另两段略差。未选默认升级，未证明真实姿态、转速、质量、推力系数或风速精度。\n\n'+
        '协议、全部CSV、源码快照、因果前缀检查与真实未来对照视频均保留。详见第三阶段实验报告.html。\n',encoding='utf-8')
    save(OUT/'stage3_completion.json',dict(module_and_raw_experiments_complete=True,default_upgrade=False,real_pose_truth=False,innovation_claimed=False))
    print('Stage3 report built',flush=True)

if __name__=='__main__':main()
