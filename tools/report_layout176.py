"""Report the fixed-reference contract and its small, mixed accuracy effects."""
from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from pure_visibility170 import window
from rotor_support_front import accumulate
OUT=ROOT.parent/'output/layout176_20261107'
def load(p):return json.loads(p.read_text('utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text('utf-8').splitlines()]

if __name__=='__main__':
    scores=load(OUT/'integrated/scores.json');source=load(OUT/'source_control/scores.json')
    old=scores['side']['control173']['score'];new=scores['side']['fixed']['score']
    summary=dict(promoted='fixed native-initialized image reference + uncertain layout offset',scores=scores,source_control=source,
        goal30px30Hz_achieved=False,new_RAW_FPS_measured=False,development_data_only=True,
        mean_improved_px=old['error']['mean']-new['error']['mean'],
        tail_cost=dict(old_p95=old['error']['p95'],new_p95=new['error']['p95'],
            old_max=max(r.get('error_px',0) for r in old['records']),new_max=max(r.get('error_px',0) for r in new['records'])))
    (OUT/'final_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),'utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib import font_manager
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc');plt.rcParams['font.family']=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name();plt.rcParams['axes.unicode_minus']=False
    r=next(r for r in rows(OUT/'integrated/side_fixed.jsonl') if r['state'].get('layout_reference_initialized'))
    e=window(ROOT.parent/'output/side_temporal164_20261101/cache_side_warm',r['t']);on,off,_=accumulate(e,r['t'],window_us=20000)
    x,y,X,Y=r['geometry']['roi'];p=np.array(r['coefficient']['position']);g=np.array(r['motion']['layout_observation']['center'])
    limit=max(1.,np.percentile((on+off)[y:Y,x:X],99.7));rgb=np.zeros(on.shape+(3,));rgb[:,:,0]=np.clip(np.log1p(on)/np.log1p(limit),0,1);rgb[:,:,2]=np.clip(np.log1p(off)/np.log1p(limit),0,1)
    fig,ax=plt.subplots(figsize=(9,3.9),constrained_layout=True);ax.imshow(rgb);ax.set_xlim(x,X);ax.set_ylim(max(Y,p[1]+15),y)
    for z in r['geometry'].get('rotors',[]):ax.add_patch(Ellipse((z['cx'],z['cy']),z['major'],z['minor'],angle=z['angle'],fill=False,ec='#69d6cb',lw=1.1,ls='--'))
    ax.plot(*g,'+',color='#68f6c4',ms=14,mew=2,label='g：桨盘布局中心')
    ax.plot(*p,'x',color='#ffb43c',ms=11,mew=2,label='p：固定输出参考')
    ax.annotate('b = g − p',xy=g,xytext=(p[0]+12,p[1]+7),color='white',arrowprops=dict(arrowstyle='->',color='white'))
    ax.set_title(f'首次几何中心 t={r["t"]:.6f}s：只建立未知偏置，不替换轨迹点\n原生20ms ON/OFF；两个点均不称真实质量重心',fontsize=12);ax.legend(fontsize=9,loc='upper right');ax.set_xlabel('原始像素 x');ax.set_ylabel('y')
    fig.savefig(OUT/'fixed_reference_points.png',dpi=165);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3.7),constrained_layout=True)
    for score,label in [(old,'173控制'),(new,'176固定参考')]:
        rr=[q for q in score['records'] if q['valid']];ax.plot([q['t'] for q in rr],[q['error_px'] for q in rr],'o-',label=label,ms=4)
    ax.set_title('均值略降，部分起点仍退化；不是每个起点都更准确');ax.set_xlabel('源时刻预测起点 / s');ax.set_ylabel('0.4s误差 / px');ax.grid(alpha=.2);ax.legend()
    fig.savefig(OUT/'all_query_errors.png',dpi=160);plt.close(fig)
    table=''.join(f'<tr><td>{name}</td><td>{scores[case]["control173"]["score"]["error"]["mean"]:.4f}</td><td>{scores[case]["fixed"]["score"]["error"]["mean"]:.4f}</td><td>{scores[case]["fixed"]["score"]["available"]}/{scores[case]["fixed"]["score"]["eligible"]}</td></tr>' for case,name in [('side','侧视'),('train2','另一录像')])
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>固定输出参考与桨盘布局中心</title>
<style>body{max-width:1080px;margin:35px auto;padding:0 22px;background:#f4f7fb;color:#20334c;font:17px/1.85 "Microsoft YaHei",sans-serif}h1{font-size:29px}h2{margin-top:34px}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #c9d5e4;padding:10px}img{max-width:100%}a{color:#175fa5}code{background:#e3eaf4;padding:2px 5px}.note{background:#fff0ca;border-left:5px solid #c48b20;padding:16px}</style>
<h1>固定输出参考：几何中心不再直接替换轨迹点</h1><p>2026-11-07 · 175模板参考同步、176固定参考已实现 · 无NN、无新转速或推力公式。</p>
<p class="note">参考点混用与模板重复平移的具体通路已修。当前纯桨叶relative入口默认fixed参考，旧模式可显式恢复。侧视0.4s均值38.7329→38.6166px，另一录像87.0806px不变；收益很小，侧视P95和最大误差略升，未实现普遍稳定提精度。30px/持续RAW30Hz仍未完成。</p>
<h2>1. 原来怎样多算了一次平移</h2><p>前端先用原生区域中心驱动五圆模板、当前弧的跨帧引导和待确认布局，然后才把body.center替换为新几何中心。模板内部保存的body/anchor/previous_body仍是旧值，下一帧用“新body−旧anchor”传播图像桨心，就可能把参考差当成额外运动。</p>
<p>175增加统一参考传递，只移动这些body锚点，不移动图像中的桨心、椭圆张量、共享仿射系数或协方差。模拟中旧31.62px幽灵位移变0，真实5/−2px平移仍保留。实录单独修此处仅38.7329→38.7434px，故不能称它已经改善轨迹精度。</p>
<h2>2. 固定哪个点，几何中心提供什么</h2><img src="fixed_reference_points.png" alt="固定输出参考和布局中心分别表示">
<p>当前输出p定义为与初始原生支撑对齐的图像参考。桨盘布局中心g是另一种观测，既不冒充人工整机框中心，也不称真实COM。建立<code>g = p + b</code>，b是未知图像参考偏置，不是从标签拟合的固定20px。</p>
<p>第一次g出现，仅初始化<code>b = g − p</code>及它与p的负交叉协方差，p/v/d/k不变。后续g通过p+b约束同一状态；原生位移与布局信息取既有对半信息权重，避免直接按独立观测相乘。协方差仍是工程代理，不声称统计校准。相对位移clone、可见贡献和未知项仍在同一个11维状态内，没有加一条运行时NN/轨迹分支。</p>
<p>若原生部分出画支撑初始化后来修正，p和前一位置clone同步加δ，b减δ，保证图像布局p+b不移动；五圆模板的body锚点同步更新。这是参考维护，不是给真实运动增加δ。两个点的绝对COM偏差仍不可凭相对位移唯一辨识。</p>
<h2>3. 完整短回放结果与代价</h2><table><tr><th>同一固定查询</th><th>173控制均值 / px</th><th>176默认均值 / px</th><th>可评分/计划</th></tr>TABLE</table>
<p>侧视133包、115有效预测；另一段67包、54有效预测，均保持。173控制复原旧p/v/a最大差0。推理共享实际GPU原生前端结果，禁读坐标标签，输出落盘后才评分。不是新盲测，也没有重跑RAW解码FPS。</p>
<img src="all_query_errors.png" alt="每个起点的误差均保留">
<p><b>侧视尾部未改善：</b>P95为68.81→69.45px，最大69.98→72.49px；112.2s变差、112.4/112.6s变好，全部保留。默认选择这项参考语义修复，不把0.116px平均变化包装为显著稳定提升。另一段缺少合格布局中心，因此几何更新次数为0，这项修复不会降低其43.6/44.0s变向大误差。</p>
<h2>4. 几何数值是否真的有作用</h2><p>同一固定参考前端、候选资格和协方差配方，把布局数值换为自身原生位移的来源对照：侧视真实布局38.6166px，替代数值38.7518px。两者均2次布局更新，说明差异不全来自参考/权重改变。但只有一个历史片段的两次相关更新，不能据此证明独立普适创新或推力预测成功。</p>
<p>新174原弧候选接入固定参考后侧视39.2834px，来源对照38.7738px；它的候选数值仍不够可靠。因此旧原弧提取仍为默认，174reconsider_small_cores=False保持，代码与负例保留。不能因为椭圆更大，就给它更大的位置权力。</p>
<h2>5. 检查、入口与复用范围</h2><p>已检查：无布局时复原旧相对观测器；首次布局不改p/v/a；给全部布局加任意坐标偏移，输出运动不变（约2e−12）；修正支撑参考后p+b不变；隐藏力转移不改净运动；前端确实把g和p分开。当前公共工厂/新模式对已保存实际输入回放最大差0，两个CLI有显式控制。</p>
<p>当前run_rotor_only.py及run_fast_structure.py --rotor-only配合--rotor-relative-observation，默认fixed参考；<code>--rotor-reference-mode legacy --rotor-gap-prediction short</code>恢复173。原NN入口和NN发布快照未改。旧实验驱动已显式选择原reference，避免默认更新悄悄改变控制。实际试验快照与最终仅工厂/分派整理的代码分别保留，不称所有字节相同。</p>
<p>理论参考仍是<a href="https://authors.library.caltech.edu/records/enrkd-zam24">stochastic cloning</a>的跨时刻相关性处理；共享几何与旋转状态可继续参考<a href="https://arxiv.org/html/2610.31384v1">CoHelix</a>。本轮沿用仓库观测器并增加偏置状态，没有复制作者实现，也不把参考维护当成新的滤波理论。</p>
<h2>6. 下一步只攻剩下的输入资格问题</h2><p>另一录像整个短窗口仍0次合格布局观测；当前三小核不等于三真实桨盘，两个/侧线也不直接提供完整中心。下一步先处理孤立小核的外围原弧支持、三桨部分结构在不歧义时的初始化/续接，并先核对已有作者代码与本机资源。参考问题解决前的随意绝对替换已禁止，不再靠额外滤波或标签偏移守分。</p>
<p>当前不能保证任意侧视/遮挡/未来新指令都小于30px。布局观测目前仍要求当前原生位置/位移资格，尚不能独立救回所有丢测；.4s参考有效期与现有模型一致。真实COM、机械RPM/牛顿推力和持续RAW30Hz仍未验证。</p>
<p><a href="final_summary.json">全部结果</a> · <a href="mechanism.json">偏置参考机制</a> · <a href="front_mechanism.json">前端分离检查</a> · <a href="checks.json">公共入口检查</a> · <a href="integrated/protocol.json">实际回放协议</a> · <a href="source_control/scores.json">同输入来源对照</a> · <a href="../reference175_20261107/mechanism.json">175模板参考检查</a></p></html>'''.replace('TABLE',table)
    (OUT/'index.html').write_text(doc,'utf-8');print('Wrote176 report and two figures; tail regression remains explicit.')
