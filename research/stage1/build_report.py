"""Stage1 mechanism report, including negative results and evidence boundaries."""
from pathlib import Path
import html,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mechanism import METHODS,HORIZONS,SCALES
from run_study import OUT,save,csvsave

def table(headers,rows):
    return '<table><tr>'+''.join(f'<th>{html.escape(str(x))}</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in row)+'</tr>' for row in rows)+'</table>'
def number(x):return '—' if x is None else f'{x:.3f}'

def main():
    d=json.loads((OUT/'controlled_analysis.json').read_text());raw=json.loads((OUT/'raw_analysis.json').read_text())
    protocol=json.loads((OUT/'protocol.json').read_text());metrics=d['metrics'];(OUT/'figures').mkdir(exist_ok=True)
    methods=['fixed','ordinary_rls','quality_rls','v030_rls','ordinary_reset_oracle']
    colors=['#647d91','#dc6652','#3486a1','#9c6aaa','#47a280']
    def get(scenario,method,window='post_early',world='matched'):
        return next(r for r in metrics if r['mode']=='estimated' and r['world']==world and r['scenario']==scenario and r['method']==method and r['window']==window and r['horizon_s']==.4)
    scenarios=[s for s in protocol['scenarios'] if s!='clean']
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax,world,title in zip(axes,['matched','gain_mismatch'],['Matched initial gain','True gain30% above prior']):
        x=np.arange(len(scenarios));width=.15
        for j,(method,color) in enumerate(zip(methods,colors)):
            rows=[get(s,method,world=world) for s in scenarios]
            y=np.array([r['paired_excess_error_px'] for r in rows]);lo=np.array([r['excess_ci95_low_px'] for r in rows]);hi=np.array([r['excess_ci95_high_px'] for r in rows])
            ax.bar(x+(j-2)*width,y,width,color=color,label=method)
            ax.errorbar(x+(j-2)*width,y,yerr=np.maximum(0,[y-lo,hi-y]),fmt='none',ecolor='#303030',elinewidth=.7,capsize=1)
        ax.set_xticks(x,scenarios,rotation=60,ha='right',fontsize=8);ax.set_title(title);ax.axhline(0,color='black',lw=.7);ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Post-restoration paired excess0.4s error(px)');axes[1].legend(fontsize=8)
    fig.suptitle('Stage1 synthetic observation errors:20 paired seeds; reset is ORACLE DIAGNOSTIC')
    fig.tight_layout();fig.savefig(OUT/'figures/post_fault_excess.png',dpi=160);plt.close(fig)
    directory=OUT/'controlled/estimated/matched/seed_100';clean=np.load(directory/'clean.npz');q=np.load(directory/'double_confident.npz');truth=np.load(directory/'truth.npz');times=q['times'];N=len(times)-20
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
    for method,color in zip(methods,colors):
        j=METHODS.index(method);err=np.linalg.norm(q['prediction'][:N,j,-1]-truth['position'][20:],axis=1)
        base=np.linalg.norm(clean['prediction'][:N,j,-1]-truth['position'][20:],axis=1)
        axes[0].plot(times[:N],err-base,label=method,color=color)
        axes[1].plot(times,q['parameters'][:,j,0],color=color)
        axes[2].plot(times,np.linalg.norm(q['disturbance'][:,j]-clean['disturbance'][:,j],axis=1),color=color)
    axes[0].set(ylabel='Excess0.4s error(px)',title='Preselected seed100, matched world:1s confident double-frequency fault');axes[0].legend(ncol=3,fontsize=8)
    axes[1].set_ylabel('Effective thrust gain');axes[2].set(ylabel='Residual drift(px/s²)',xlabel='Time(s)')
    for ax in axes:ax.axvspan(5,6,color='#e3af78',alpha=.3);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'figures/double_frequency_mechanism.png',dpi=160);plt.close(fig)
    rows=[]
    for scenario in scenarios:
        a=get(scenario,'ordinary_rls');b=get(scenario,'quality_rls');c=get(scenario,'v030_rls');reset=get(scenario,'ordinary_reset_oracle')
        rows.append([scenario,number(a['paired_excess_error_px']),f"[{a['excess_ci95_low_px']:.3f}, {a['excess_ci95_high_px']:.3f}]",
                     number(b['paired_excess_error_px']),number(c['paired_excess_error_px']),number(reset['paired_excess_error_px'])])
    raw_rows=[[r['condition'],r['method'],r['common_available_frames'],number(r['mean_paired_prediction_disagreement_px']),number(r['normalized_parameter_drift'])]
              for r in raw['paired_diagnostics'] if r['window']=='early_restore' and r['method'] in ('ordinary_rls','quality_rls','v030_rls','ordinary_reset_oracle')]
    pub='public_truth_substitutes';public=(OUT/pub/'replacement_plan.json').exists()
    public_status=json.loads((OUT/pub/'public_baseline_summary.json').read_text()) if (OUT/pub/'public_baseline_summary.json').exists() else None
    optional_public='<p>公开数据实际推理/评分尚未运行，本页仅记录现有材料准备状态。</p>'
    if public_status:
        public_rows=[]
        for method in ('cv','paper_adaptive','oracle_past_boxes_cv'):
            for r in public_status['metrics']:
                if r['method']==method:
                    public_rows.append([method,r['horizon_s'],f"{r['available']}/{r['targets']}",f"{100*r['coverage']:.1f}%",number(r['mean_error_px'])])
        eeppr=json.loads((OUT/pub/'combined_results.json').read_text())
        periodic_rows=[[r['sequence'],r['gt_freq_hz'],number(r['unrounded_median_hz']),number(r['unrounded_absolute_error_hz']),f"{r['valid_windows']}/{r['total_windows']}"] for r in eeppr]
        optional_public='''<h3>第二阶段已执行：FRED train1前20秒</h3>
<p>事前固定sequence1前20秒，原v0.3不改参数，输入仅RAW，处理2000帧、18477968事件。检测139帧，只有4帧出现四椭圆；物理分支激活0帧，在线物理更新0次。各物理命名输出因此等于回退CV，不能解释为物理方法验证成功。</p>
<p>下表是可用时刻的平均端点距离，必须与覆盖率一起读。历史标注输入的CV单独列出，它不是只输入RAW的结果，也不是与自动检测公平比较的成绩；其目标数因历史起点略有不同。该行仅检查标注能否支持二维评分。</p>'''+table(['方法/输入协议','预测秒数','可用/目标','覆盖率','平均误差px'],public_rows)+'''
<p>处理20秒数据耗时44.75秒，包含流式解码与推理；当前这一运行仍未达到实时。它只是一段官方训练序列上的迁移诊断，不能代替官方完整测试。</p>
<p>对齐抽查：下面四个时刻并列作者事件PNG与独立解码RAW，绿色框为同一份提供标注。10、15、18秒目标明显落入框内；5.1秒事件稀疏，未据此判定错位。这排除了这些抽查点上的明显整体坐标偏移，不证明全序列逐帧标注精度。</p>
<a href="public_truth_substitutes/public_alignment_audit.png"><img src="public_truth_substitutes/public_alignment_audit.png" alt="FRED四时刻原始事件与标注对齐抽查"></a>
<h3>复用已完成的EEPPR公开数据实验</h3>
<p>运行作者核心算法、默认1秒数据和提供的完整ROI；SDK桥接逐事件核对。下面使用未四舍五入结果，不能将显示到一位小数相同误称为零误差。真值本身也仅按作者给定精度记录。它验证原EEPPR的一般周期估计，不是当前项目测速模块、四桨机械RPM或实飞姿态的验证。</p>'''+table(['序列','提供真值Hz','未取整估计Hz','绝对误差Hz','有效窗口'],periodic_rows)+'''
<p><a href="public_truth_substitutes/public_baseline_summary.json">公开数据完整成绩</a> · <a href="public_truth_substitutes/public_protocol.json">运行前冻结的公开数据协议</a> · <a href="public_truth_substitutes/替代真值验证路线.md">没有同步日志时的完整验证路线</a></p>'''
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第一阶段：观测错误与在线学习</title>
<style>body{{font-family:"Microsoft YaHei",sans-serif;line-height:1.8;background:#f4f6f8;color:#182b38;margin:0}}main{{max-width:1180px;margin:auto;padding:28px}}section{{background:white;border:1px solid #dae2e8;border-radius:10px;padding:22px;margin:20px 0}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border-bottom:1px solid #dbe3e8;text-align:left;padding:8px}}th{{background:#eef3f7}}img,video{{max-width:100%}}a{{color:#136f9c}}.note{{background:#fff4d8;padding:16px;border-left:4px solid #d29d35}}pre{{white-space:pre-wrap;font-size:12px}}.scroll{{overflow:auto}}</style><main>
<h1>第一阶段：坏观测是否会把动力学“学坏”？</h1>
<p>2026-10-16 · 先验证机制，再开发方法 · 原v0.3推理代码与既往成绩保持原样</p>
<div class="note"><b>结论：</b>在冻结的可控实验中，持续1秒的倍频、半频和错误几何，确实可让在线模型在输入恢复后仍有额外预测误差。能发现低质量时，已有门控大多有效；高置信度错误则穿过现有门控。这支持继续研究这一具体失效机制，但没有证明新算法或真实飞行参数识别。</div>
<p><a href="阶段报告.md">文字报告</a> · <a href="protocol.json">事前冻结协议</a> · <a href="controlled_aggregate_metrics.csv">全部分组成绩</a> · <a href="controlled_trial_metrics.csv">逐随机种子结果</a> · <a href="controlled_recovery.csv">恢复时间与未恢复记录</a> · <a href="raw_sparse_endpoint_errors.csv">RAW稀疏端点评分</a></p>
<section><h2>实验设计</h2><p>360组主要实验：20个固定随机种子×2种初始增益条件×9种观测情形；另18组当前状态真值诊断，总计378组。真实动力学在每组干预前后不变，噪声在干净/污染输入之间逐样本相同。除干预字段外不改真实轨迹。20种子是同一预设运动上的噪声重复，不是20次独立实飞。</p>
<p>六个输出：CV、固定参数、普通RLS、同方程质量门控RLS、冻结v0.3学习器、理想重置诊断。所有模型共享当前位置/速度估计，只隔离未来加速度和学习效果；不是原端到端版本的复评分。所有更新只使用已到达的观测。</p>
<p><b>理想重置仅用于因果诊断：</b>错误结束时，将并行干净运行中的学习器状态移入污染分支，部署时不可获得。当前状态真值诊断也明确不属于RAW-only部署成绩。</p></section>
<section><h2>输入恢复后还剩多少损失？</h2><p>下表是初始增益匹配条件，错误发生于5–6秒；评分使用6.4–8秒发布的0.4秒预测。数值为“污染输入误差−同模型干净输入误差”，不是绝对误差。95%区间按20个种子配对均值自助抽样，不把连续帧当独立样本；未作多重比较显著性声明。</p>
<div class="scroll">{table(['干预','普通RLS额外误差px','种子95%区间','质量门控额外误差','原v0.3额外误差','理想重置诊断'],rows)}</div>
<img src="figures/post_fault_excess.png"><img src="figures/double_frequency_mechanism.png">
<p>倍频期间，固定参数模型的额外误差较大，而普通学习可部分补偿错误频率；频率恢复后，学习得到的补偿反而留下损失。因此不能得出“总是冻结优于学习”的结论。需要同时评错误期间和恢复阶段。</p>
<p>反例同样保留：只置换ID对总推力完全不变；中心偏差未在本设置中导致平均恢复误差变差，理想学习器重置甚至可能让它暂时更差，因为位置滤波状态仍有偏差。不能把参数变化一概解释为有害。</p></section>
<section><h2>真实RAW事件删除复核</h2><p>使用train6完整0–24秒共同历史，复制同一个状态；24–25秒在固定图像区域删除事件，或随机删除一半，继续处理至30秒。矩形仅由24秒及以前观测决定。无人工框或未来标签进入推理。原始文件未修改。</p>
<video controls preload="metadata" src="raw_interventions/paired_raw.mp4" poster="raw_interventions/comparison_200.png"></video>
<p>四格分别为干净RAW、单桨矩形、左侧双桨矩形、随机删半。橙色十字是普通RLS的0.4秒预测。<b>这不是实际未来对照画面，也不是新的预测准确率证明。</b>固定传感器坐标的矩形不会随无人机移动；事件删除也不模拟遮挡物产生的新事件。</p>
<p>下表是删除结束后25–27秒、双方都在0.16秒追踪有效期内的输出差异。差异大只表示对干预敏感，不等于误差大。真实准确率只可由另存的少量人工端点检查，不能把干净预测当真值。</p>
<div class="scroll">{table(['干预','学习器','共同有效帧','与干净预测差异px','归一化参数差异'],raw_rows)}</div>
<p><a href="raw_analysis.json">完整RAW分析与边界</a> · <a href="raw_interventions/intervention_manifest.json">矩形、输入和步骤记录</a></p></section>
<section><h2>没有自采真值，第二阶段怎样继续？</h2><p>可以用分模块公开真值加终端位置标注替代。E盘已有FRED原生RAW和位置标注，已将sequence1相关文件准备到此目录；它属于官方train，当前用途是公开数据开发验证，不是官方测试成绩。已有EEPPR正/侧视频率真值与原代码结果可验证一般周期估计，但不是四桨实飞真值。TQE有四桨同步RPM，却需向作者申请，不能假称已下载。</p>
<p>仿真验证已知机制，公开频率验证测速，FRED验证二维预测；三者不能合并为真实质量、推力或姿态已经测准。独立测速记录不足时，将研究结论限定在有效像素模型与二维预测，是可行的研究路线。</p>
{optional_public}
<p><a href="public_truth_substitutes/replacement_plan.json">替代证据路线</a> · <a href="public_truth_substitutes/fred_inventory.json">本地FRED材料与指纹</a> · <a href="public_truth_substitutes/combined_results.json">已有EEPPR原始结果</a></p></section>
<section><h2>下一步边界</h2><p>第一阶段支持研究“高置信度错误与自适应混淆”，不支持泛泛宣称所有观测退化都导致持续污染。第二阶段先验证现有测速/几何在公开数据中的可靠性，再开发离散候选与连续参数的联合机制；简单质量门控必须作为强基线。</p>
<p>频率全乘2、推力增益除4会产生相同简化加速度；缺少额外相位/叶片数/标定信息时，不能保证仅用轨迹选出真实RPM。故不承诺同时唯一恢复质量、推力系数和风速。</p></section></main></html>'''
    (OUT/'第一阶段实验报告.html').write_text(text,encoding='utf-8')
    a=get('double_confident','ordinary_rls');b=get('double_flagged','quality_rls');c=get('ellipse_confident','ordinary_rls')
    md=f'''# 第一阶段实验结果

完成378组预先冻结的可控对照和一段真实RAW事件删除实验。360组主要实验是20噪声种子×2初始增益条件×9场景；另18组当前真值状态诊断。不得当作378次独立实飞。

## 核心结果

在初始增益匹配条件下，1秒高置信度倍频结束后6.4–8秒的0.4秒预测，普通RLS较自身干净对照多{a['paired_excess_error_px']:.3f}px，20种子配对95%区间[{a['excess_ci95_low_px']:.3f},{a['excess_ci95_high_px']:.3f}]。高置信度错误几何多{c['paired_excess_error_px']:.3f}px。低置信度已知时，质量门控将倍频后的额外误差减至{b['paired_excess_error_px']:.3f}px。

同一试验在错误结束时理想移植干净学习器状态，可消除频率/几何干预造成的后续差异；这是不可部署的因果诊断，不是新方法。当前状态真值诊断仍出现这一机制，说明不能全归咎于位置滤波噪声。

反例：纯ID置换对总推力与全部预测完全不变；中心偏差并非一定导致更差的恢复预测。固定模型在错误期间可能损失更大，而学习模型在错误结束后有尾部损失，必须报告二者。普通RLS在干净输入上也未被假定优于固定模型。

## 真实RAW复核

完整0–24秒共同预热，24–25秒固定区域/随机删除事件，再观测至30秒。四个条件、同一历史状态、同一RAW输入；无未来标签参与推理。差异曲线只反映干预敏感性，不能当作真实预测误差。真实端点只使用既有稀疏人工框，未插值真值。

## 真值替代

用户不能自采同步真值。已准备现有E:/FRED/1.zip中的RAW与coordinates.txt，并复用已完成的EEPPR公共正侧视频率验证。FRED支持二维位置评分，EEPPR支持一般周期验证；都不能提供真实飞行推力、质量或完整姿态真值。TQE需作者批准访问，尚未取得。

## 贡献判断

已有方法能解决被识别的低质量错误；仍值得研究的是高置信度错误和未知参数互相补偿。现在尚未提出或验证新算法。先完成公开数据模块验证，再开展新机制，必须比较普通RLS、简单质量门控和原v0.3；否则不能把工程组合写成创新。

公开FRED train1前20秒已完成：2000帧中检测139帧，物理分支激活0帧，故物理输出全部回退CV。0.4秒自动预测可用66/317个端点，平均误差178.005px；低覆盖率与物理未启用必须一同报告。公开数据不能掩盖当前检测迁移不足。对齐图已人工抽查，无明显整体偏移；非完整标注审核。

报告入口：第一阶段实验报告.html。事前协议、源码快照、逐种子CSV、恢复时间（含未恢复）和RAW日志均已保存。旧版推理源码未修改。
'''
    (OUT/'阶段报告.md').write_text(md,encoding='utf-8')
    save(OUT/'stage1_completion.json',dict(controlled_trials=378,raw_conditions=4,raw_test_seconds=6,
        stage1_mechanism_in_simulation_supported=True,real_rpm_pose_truth_validated=False,new_algorithm_claimed=False,
        primary_post_double_excess_px=a['paired_excess_error_px'],primary_post_ellipse_excess_px=c['paired_excess_error_px'],
        next_stage='Public ground-truth substitutes, then mechanism development; not a promise of full physical parameter recovery.'))
    print('Stage1 report built',flush=True)

if __name__=='__main__':main()
