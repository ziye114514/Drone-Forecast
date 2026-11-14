"""Explain stage160 with frozen observations; no new detection or forecasting."""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Ellipse

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from rotor_only_front import filtered_rgb
from physical_publication import evaluate
from compare_final_pipeline import reference_at

P=ROOT.parent/'output/rotor_pipeline_20261101'
OUT=P/'gaps_explained'
FONT=Path('C:/Windows/Fonts/msyh.ttc')
if FONT.exists():font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,
    'font.size':11,'figure.dpi':150,'savefig.dpi':180,
    'axes.spines.top':False,'axes.spines.right':False})
BLUE='#2563eb';ORANGE='#d97706';RED='#dc2626';GREEN='#15803d';GRAY='#64748b'


def rows(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
def save(fig,name):
    fig.savefig(OUT/name,bbox_inches='tight',facecolor='white');plt.close(fig)
def current_before(data,q):return [r for r in data if r['t']<=q+1e-9][-1]
def ellipse(ax,r,color=BLUE,style='-'):
    ax.add_patch(Ellipse((r['cx'],r['cy']),r['major'],r['minor'],
        angle=r['angle'],fill=False,color=color,lw=1.5,ls=style))
def rgb(z,mask):return filtered_rgb(z['on'],z['off'],mask)


def main():
    OUT.mkdir(exist_ok=True)
    clear=rows(P/'clear_bank/predictions.jsonl')
    side=rows(P/'side_bank/predictions.jsonl')
    muted=rows(P/'side_bank/without_period_input.jsonl')
    summary=json.loads((P/'final_summary.json').read_text(encoding='utf-8'))
    # 1: Actual contour/rate disagreement, not an illustrative synthetic UAV.
    r=clear[8]
    with np.load(P/'clear_bank/images/0008_evidence.npz') as z:
        fig,axs=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
        for ax,mask,title in zip(axs,[np.ones_like(z['period']),z['period'],z['native_arcs']],
                ['原生 ON/OFF','严格周期检验：374像素','恢复原生轮廓：4个椭圆']):
            ax.imshow(rgb(z,mask));ax.set_xlim(815,1110);ax.set_ylim(150,30);ax.set_title(title)
            ax.set_xticks([]);ax.set_yticks([])
        for i,(g,rate) in enumerate(zip(r['geometry']['rotors'],r['rates'])):
            color=GREEN if rate['valid'] else RED;ellipse(axs[2],g,color)
            label=f'R{i+1}: {rate["frequency_hz"]:.0f} Hz'
            axs[2].text(g['cx'],g['cy']-g['minor']/2-3,label,color='white',fontsize=9,ha='center')
        fig.suptitle('实测26.500s：四个轮廓成立，逐桨周期局部资格仅3/4；本帧总推力代理仍未初始化',fontsize=13)
        fig.supxlabel('R1–R4是当前候选区域编号，不是机头机尾；Hz是事件周期候选，不是真实转速。显示裁剪仅用于放大。',fontsize=10)
        save(fig,'01_contours_and_periods.png')
    # 2: Pooled frequency switching plus exact current qualification.
    ts=np.array([r['t'] for r in side]);fs=np.array([r['period']['frequency_hz'] or np.nan for r in side])
    active=np.array([r['force']['active'] for r in side])
    fig,(ax,bx)=plt.subplots(2,1,figsize=(11.4,5.8),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[3,1]})
    ax.plot(ts,fs,'o-',color=BLUE,ms=4,lw=1,label='当前池化事件周期候选')
    ax.scatter(ts[~active],fs[~active],facecolors='white',edgecolors=RED,s=55,label='当前力输入未启用',zorder=3)
    ax.axhline(150,color=GRAY,lw=.8,ls=':');ax.axhline(300,color=GRAY,lw=.8,ls=':')
    ax.annotate('150 → 285 Hz\n若直接平方：×3.61\n不等于已证实真实转速突变',xy=(110.533328,285),xytext=(110.77,390),
        arrowprops={'arrowstyle':'->','color':GRAY},fontsize=10)
    ax.set_ylim(80,max(450,float(np.nanmax(fs))*1.12));ax.set_ylabel('事件周期候选 / Hz');ax.legend(loc='upper right',fontsize=9)
    bx.step(ts,active.astype(int),where='post',color=ORANGE,lw=1.5)
    bx.set_yticks([0,1],['停用/保持/回退','当前代理启用']);bx.set_ylim(-.2,1.2);bx.set_xlabel('原始事件截止时刻 / s')
    fig.suptitle('侧视：频率阶次尚未解决；已有一致性门会拒绝部分跳变，但不能证明其余读数就是RPM',fontsize=12)
    save(fig,'02_frequency_candidates.png')
    # 3: Concrete dominant failure: stale residual overwhelms direction.
    q=110.4;on=current_before(side,q);off=current_before(muted,q)
    a_pos=np.array(off['force']['acceleration_px_s2']);a_on=np.array(on['force']['acceleration_px_s2'])
    known=on['force']['effective_gain_px_s2']*np.array(on['force']['input_u'])
    disturbance=np.array(on['force']['disturbance_px_s2'])
    labels=json.loads((ROOT.parent/'output/dense_labels_side_110_1195/labels.json').read_text(encoding='utf-8'))
    truth=sorted(labels['records'].values(),key=lambda z:z['t'])
    future=reference_at(truth,q+.4);now=reference_at(truth,q)
    fig,(ax,bx)=plt.subplots(1,2,figsize=(13.5,5.3),layout='constrained')
    ix=(ts<=110.8)
    ax.plot(ts[ix],[r['coefficient']['acceleration'][0] for r in side if r['t']<=110.8],color=BLUE,label='位置状态估计的水平净加速度')
    ax.plot(ts[ix],[r['force']['acceleration_px_s2'][0] for r in side if r['t']<=110.8],color=RED,label='周期+未知作用模型的水平净加速度')
    ax.plot(ts[ix],[r['force']['disturbance_px_s2'][0] if r['force'].get('disturbance_px_s2') else np.nan for r in side if r['t']<=110.8],
        color=ORANGE,ls='--',label='未知作用估计 d_x（不是风力真值）')
    ax.axvspan(110.033333,110.133332,color=GRAY,alpha=.12,label='位置状态尚未valid')
    ax.axvline(q,color=GRAY,lw=1,ls=':');ax.axhline(0,color=GRAY,lw=.7)
    ax.annotate('110.4s：位置状态 -468\n力学分支 +1085',xy=(q,a_on[0]),xytext=(110.43,2100),
        arrowprops={'arrowstyle':'->','color':GRAY},fontsize=10)
    ax.set_xlabel('源时刻 / s');ax.set_ylabel('图像水平加速度 / px/s²');ax.legend(fontsize=8,loc='lower left')
    horizons=np.linspace(0,.4,50)
    for data,color,title in [(on,RED,'开启周期/残差分支'),(off,BLUE,'关闭周期分支')]:
        age=q-data['t'];values=evaluate(data,[float(age+h) for h in horizons])
        curve=np.array([values[str(float(age+h))] for h in horizons])
        bx.plot(curve[:,0],curve[:,1],color=color,label=title,lw=2)
        bx.scatter(*curve[-1],color=color,s=65,marker='x')
    refs=[reference_at(truth,float(t)) for t in np.linspace(q,q+.4,9)]
    refs=np.array([r for r in refs if r is not None]);bx.plot(refs[:,0],refs[:,1],color=GREEN,ls='--',label='事后人工中心参考轨迹')
    bx.scatter(*future,color=GREEN,marker='*',s=130,label='未来0.4s参考')
    bx.scatter(*now,color=GRAY,s=35,label='当前人工参考')
    bx.invert_yaxis();bx.set_aspect('equal',adjustable='datalim');bx.set_xlabel('图像 x / px');bx.set_ylabel('图像 y / px');bx.legend(fontsize=8)
    fig.suptitle('最大坏例110.4s：当前中心误差0.84px，未来误差却从62.34px变成157.33px',fontsize=13)
    fig.supxlabel('左侧两种加速度都是算法估计，非传感器真值。右侧绿线只用于事后评分；两个预测使用完全相同的位置/速度状态。',fontsize=9)
    save(fig,'03_residual_failure.png')
    # 4: What is current, inferred, or merely a side-axis candidate.
    fig,axs=plt.subplots(1,2,figsize=(12.5,4.8),layout='constrained')
    for ax,name,index,title in [(axs[0],'clear_bank',14,'26.700s：3个候选，模板只接受2个'),(axs[1],'side_bank',23,'110.800s：程序降级为axis_only')]:
        rr=(clear if name=='clear_bank' else side)[index]
        with np.load(P/name/'images'/f'{index:04d}_evidence.npz') as z:
            ax.imshow(rgb(z,z['native_arcs']))
            yy,xx=np.nonzero(z['native_arcs'])
            ax.set_xlim(max(0,xx.min()-25),min(1280,xx.max()+25));ax.set_ylim(min(720,yy.max()+25),max(0,yy.min()-25))
        hubs=rr['structure'].get('model_centers')
        if hubs is not None:
            hubs=np.array(hubs)
            for ident in rr['structure'].get('observed_model_ids',[]):
                ax.scatter(*hubs[ident],s=60,color=BLUE,edgecolors='white',label='有当前观测支持的模型点' if ident==rr['structure']['observed_model_ids'][0] else None)
            for ident in rr['structure'].get('inferred_model_ids',[]):
                ax.scatter(*hubs[ident],s=110,facecolors='none',edgecolors=ORANGE,linewidths=2,label='无当前观测的模型点' if ident==rr['structure']['inferred_model_ids'][0] else None)
            ax.legend(fontsize=8,loc='lower left')
        else:
            center=np.array(rr['body']['center']);normal=np.array(rr['structure']['axis_xy'])
            scale=30;ax.annotate('',xy=center+normal*scale,xytext=center-normal*scale,
                arrowprops={'arrowstyle':'<->','lw':2,'color':ORANGE})
            ax.text(center[0],center[1]+18,'法线投影有正负候选\n当前用“图像向上”先验',color='white',ha='center',fontsize=9)
        ax.set_title(title,fontsize=11);ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('当前可见部分、补出的模型点、侧视法线，必须保持不同资格',fontsize=13)
    fig.supxlabel('空心点不提供隐藏转速；side是历史片段名，不是人工视角真值。右图多个凸起仍可见，降级可能含算法分离/资格问题。',fontsize=9)
    save(fig,'04_visibility_and_inference.png')
    # 5: Error distribution and deadline, all frozen results.
    fixed=summary['source_force_ablation']['side_final']['fixed_source_queries'];points=fixed['rows']
    fig,(ax,bx)=plt.subplots(1,2,figsize=(13.1,4.6),layout='constrained')
    qtimes=[r['query_t'] for r in points]
    for key,color,title in [('old_body_baseline_px',GRAY,'原整机基线'),('rotor_period_off_px',BLUE,'新桨叶，周期关'),('rotor_period_on_px',RED,'新桨叶，周期开')]:
        vals=[r[key] for r in points];ax.plot(qtimes,vals,'o-',color=color,ms=4,label=f'{title}：均值{np.mean(vals):.2f}px')
    ax.set_xlabel('固定源查询时刻 / s');ax.set_ylabel('未来0.4s误差 / px');ax.legend(fontsize=8);ax.set_ylim(0,175)
    runs=summary['runs'];means=[];medians=[];p95=[]
    for name in ['clear_final','side_final']:
        medians.append(runs[name]['compute_ms_median']);p95.append(runs[name]['compute_ms_p95'])
    x=np.arange(2);bx.bar(x-.17,medians,.32,color=BLUE,label='核心计算中位数')
    bx.bar(x+.17,p95,.32,color=ORANGE,label='核心计算P95')
    bx.axhline(1000/30,color=RED,lw=1.5,ls='--',label='30Hz预算：33.3ms')
    for a,b in zip(x,medians):bx.text(a-.17,b+7,f'{b:.0f}ms',ha='center',fontsize=9)
    for a,b in zip(x,p95):bx.text(a+.17,b+7,f'{b:.0f}ms',ha='center',fontsize=9)
    bx.set_xticks(x,['清晰','侧视']);bx.set_ylabel('处理耗时 / ms');bx.set_ylim(0,510);bx.legend(fontsize=8)
    fig.suptitle('预测收益不稳，而且当前还远不是实时系统',fontsize=13)
    fig.supxlabel('左：9个共同源查询，整机人工中心代理；不计真实运行延迟。右：核心耗时，尚未计入当前导出/回放开销。',fontsize=9)
    save(fig,'05_error_and_runtime.png')
    gain={name:dict(ever_learned=sum(r['force'].get('gain_ever_learned',False) for r in data),
        min=min(r['force']['effective_gain_px_s2'] for r in data),max=max(r['force']['effective_gain_px_s2'] for r in data))
        for name,data in [('clear',clear),('side',side)]}
    diagnosis=dict(worst_query=110.4,source_t=on['t'],gain=gain,period_candidate_hz=on['period']['frequency_hz'],
        current_u=on['force']['input_u'],known_proxy_acceleration=known.tolist(),
        unknown_residual_acceleration=disturbance.tolist(),total_force_branch_acceleration=a_on.tolist(),
        position_state_acceleration=a_pos.tolist(),current_center_error_px=float(np.linalg.norm(np.array(on['body']['center'])-now)),
        future_on_error_px=float(np.linalg.norm(np.array(evaluate(on,[float(q-on['t']+.4)])[str(float(q-on['t']+.4))])-future)),
        future_off_error_px=float(np.linalg.norm(np.array(evaluate(off,[float(q-off['t']+.4)])[str(float(q-off['t']+.4))])-future)),
        active_inputs_before_position_valid=[r['t'] for r in side if r['force']['active'] and not r['valid']],
        position_validity_is_not_currently_a_force_learning_input_gate=True,
        causal_problem_inferred_from_code_and_records_not_true_wind_measurement=True,
        horizontal_force_terms_at_bad_query=dict(known=float(known[0]),unknown=float(disturbance[0]),total=float(a_on[0])),
        local_qualification_clear_frame8=[dict(frequency=r['frequency_hz'],valid=r['valid'],spread=r['log_frequency_MAD']) for r in clear[8]['rates']],
        side_is_historical_clip_name_not_human_view_label=True)
    (OUT/'diagnosis.json').write_text(json.dumps(diagnosis,ensure_ascii=False,indent=2),'utf-8')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>当前缺口逐项图解</title>
    <style>body{font:17px/1.8 system-ui;max-width:1250px;margin:30px auto;padding:0 25px;color:#25374a}img{width:100%}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:9px}.note{background:#fff1d9;padding:16px}h2{margin-top:40px}code{background:#eef2f6}</style>
    <h1>当前缺口：从真实记录说明，而不是泛称“检测不够准”</h1><p>2026-11-01。仅分析阶段160已保存结果，未重跑RAW/训练/测速，未修改算法。所有图都是冻结数据或带标记的推断，没有RPM/力学真值。</p>
    <p class="note">最重要的新发现：110.4秒的大坏例主要由残差模型给出的巨大旧水平作用项主导；这个时刻周期候选为300Hz，没有证据把它单独归因于半频跳变。转速资格不稳是另一个真实缺口。</p>
    <h2>1．四个轮廓，不等于四个合格转速</h2><img src="01_contours_and_periods.png"><p>26.5秒确有四个轮廓，局部周期候选为300、240、120、135Hz。240Hz区域的频率离散度约0.381，超过0.08门槛，局部不合格；其余也还需要跨帧一致性和模型关联。本段37个包只有2个时刻真正使用完整四桨频率平方和。其他时刻不能冒充已经测得全部推力。</p>
    <h2>2．周期候选阶次与混桨未解决</h2><img src="02_frequency_candidates.png"><p>150到285Hz的变化若直接平方，会放大成3.61倍。实际已有一致性门会停用其中部分输入，因此图中跳变不等于算法已把它全部当推力突变；无独立RPM也不能断言每次跳变都不是真实变化。当前池化数值是所有合格周期像素的中位数，某个桨占据的像素数变化也可能导致中位数换到另一个桨，即使各桨转速都没变。需要区分真实转速变化、谐波换阶和像素权重改变，不能只做数值平滑。</p>
    <h2>3．最大坏例：未知作用估计覆盖了当前减速</h2><img src="03_residual_failure.png">
    <p>110.4秒当前中心与人工参考差0.84px。位置状态估计a=(-468,-289)px/s²；周期已知项约(-4.53,-246.87)，未知残差d=(1089.47,743.33)，相加变成a=(1084.94,496.46)。开启分支比关闭分支将终点移动约98px，预测误差62.34→157.33px。</p>
    <p><b>代码缺口</b>：力学校准接收了位置状态尚未valid时的起步样本；到期0.2秒积分残差又代表一段历史平均，不能直接视为当前瞬时外力。冷启动速度误差、定位漂移也可能被吸收到d。当前两段的增益k都一直是默认200，没有一次成功学出新增益。记录证明大d主导该坏例；真实风力多大、每项误差各占多少仍未识别。</p>
    <h2>4．部分可见与侧视有分支，但资格仍不完整</h2><img src="04_visibility_and_inference.png"><p>补出桨心不等于补出隐藏转速。四桨全可见时编号置换不改变总和，机头机尾并非必要；部分可见时要防止重复/漏计同一真实转子的贡献。side是历史片段名，程序axis_only不是人工侧视角度标注；图中多个凸起仍可见，不能把所有降级都归咎像素没有信息。部分桨盘粘连/分离和当前几何资格仍有改进空间。侧视池化信号不需要给每个桨编号，但不能自动决定隐藏桨总推力。法线投影、图像向上先验和当前/保持姿态要分开。</p>
    <p>清晰段中心来源有17次几何中心、19次筛选后平移续接；可见性变化时的参考坐标不一致可能制造假运动。当前平移匹配只取最大相关峰，没有双向一致性/峰值歧义资格检验，因此局部旋转/桨盘形变可能被当成平移。此为源码中可确认的资格缺口，尚没有悬停真值可量化其真实误差，不能直接宣称已解决“桨盘摆动而中心不动”。</p>
    <h2>5．固定查询没改善，运行速度也未达标</h2><img src="05_error_and_runtime.png"><p>9点均值：旧34.84、新周期关44.79、开53.96px。新前端/状态整套链与旧链之间还有差距；相同位置输入中，周期/残差分支又增加约9.17px。一个大坏例对均值的影响很大。另一种11个源起点抽样的76.06→64.05不能替代这张固定查询表。</p>
    <p>核心中位约233/213ms，30Hz预算只有33.3ms；当前约3.9计算FPS。单个调用级剖析中，CUDA工作进程往返约64ms、原生重建约48ms、几何约26ms，各项不是长期平均，也不能把包含关系重复相加。含图/视频导出的缓存回放更慢。不应把已有30FPS视频当实时运行。</p>
    <h2>实际力学意义</h2><p>目前是图像加速度模型 a_img = k·q·n + d，而非真实牛顿力传感器。真实转桨推力通常写作 T=C_Tρn²D⁴；C_T、桨径、空气密度、质量和投影比例都不由一个像素频率直接给出。若d不受约束，改变k再用d抵消可以得到相同运动，在线校准需时间尺度/激励/置信约束，不能仅靠“有偏差就改未知力”唯一恢复物理参数。</p>
    <h2>建议顺序</h2><ol><li>先修冷启动/位置资格与残差时间对齐，保留当前减速信息，避免旧平均作用强行覆盖。</li><li>基频/谐波用多假设和不确定性处理，资格不够时减小影响，不能只把数值平滑。</li><li>明确当前/保持法线、隐藏先验、可见性变化下的中心坐标。</li><li>减少全图重复数组与频谱重算，做设备驻留的稀疏更新；保留全局重发现。</li></ol>
    <p>参考：<a href="https://arxiv.org/html/2603.08386v1">DDHF第3.2节：叶片通过频率与谐波</a>；<a href="https://m-selig.ae.illinois.edu/props/volume-1/Brandt_2005_UIUC-MS-Thesis.pdf">UIUC桨叶系数定义</a>。本地简化实现并非作者完整复现。</p>
    <p><a href="diagnosis.json">坏例数值与证据</a> · <a href="../final_summary.json">完整冻结评测</a> · <a href="../index.html">上一轮报告和视频</a></p></html>'''
    (OUT/'index.html').write_text(html,'utf-8')
    print(json.dumps(dict(output=str(OUT),figures=5,unknown=disturbance.tolist(),gain=gain),ensure_ascii=True))


if __name__=='__main__':main()
