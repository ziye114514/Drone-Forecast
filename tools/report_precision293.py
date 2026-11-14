"""Compact factual report of this work batch, with all regressions visible."""
from pathlib import Path
import sys,json,html,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pure_visibility170 import window

if __name__=='__main__':
    OUT=ROOT.parent/'output/precision293_20261108';OUT.mkdir(exist_ok=False)
    data=json.loads((ROOT.parent/'output/identifiable289_20261108/results.json').read_text('utf8'))
    reviewed=json.loads((ROOT.parent/'output/reference290_20261108/results.json').read_text('utf8'))
    scores={n:data['scores'][n] for n in ['13','15','16','11']};scores['3']=dict(control=reviewed['scores']['control'],tangent=reviewed['scores']['identifiable289'])
    labels=['13 (dev)','15 (dev)','16 (dev)','11 (dev)','3 (new/reviewed)'];names=list(scores)
    fig,ax=plt.subplots(figsize=(10,4));x=np.arange(5)
    for offset,key,label,color in [(-.18,'control','Original','#64748b'),(.18,'tangent','Identifiable tangent response','#2563eb')]:
        vals=[scores[n][key]['error']['mean'] for n in names];ax.bar(x+offset,vals,.36,label=label,color=color)
        for xx,v in zip(x+offset,vals):ax.text(xx,v+.7,f'{v:.2f}',ha='center',fontsize=8)
    ax.axhline(30,color='#b91c1c',ls='--',label='30px target');ax.set_xticks(x,labels);ax.set_ylabel('0.4s source-time mean FDE (px)');ax.legend(fontsize=8);ax.set_ylim(0,82)
    fig.tight_layout();fig.savefig(OUT/'means.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for ax,n in zip(axes.flat,names[:4]):
        for key,label,color in [('control','Original','#64748b'),('tangent','Learned response','#2563eb')]:
            rs=[r for r in scores[n][key]['records'] if r['valid']];ax.plot([r['t'] for r in rs],[r['error_px'] for r in rs],'.-',label=label,color=color)
        ax.axhline(30,color='#b91c1c',ls='--',lw=.7);ax.set_title(f'Recording{n}, every eligible origin');ax.set_xlabel('Source time (s)');ax.set_ylabel('FDE (px)')
    axes[0,0].legend();fig.tight_layout();fig.savefig(OUT/'all_origins.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for stage,label,color in [('tube282','Without inertia centre','#dc2626'),('tube283','With inertia centre','#2563eb')]:
        rows=[json.loads(s) for s in (ROOT.parent/f'output/{stage}_20261108/recording1/tube.jsonl').read_text('utf8').splitlines()]
        current=[r for r in rows if 12.0<=r['t']<=12.4 and r['motion'].get('estimate')]
        axes[0].plot([r['t'] for r in current],[r['motion']['estimate']['dx'] for r in current],'.-',label=label,color=color)
        r=[r for r in rows if r['t']<=12.2][-1]
        axes[1].scatter(*r['physical']['0.4'],s=80,c=color,label=f'{label}: forecast')
    ref=json.loads((ROOT.parent/'output/frozen_unseen278_20261108/references.json').read_text('utf8'))['records']['12.6'];b=ref['box'];target=(np.array(b[:2])+b[2:])/2
    axes[1].scatter(*target,c='green',marker='x',s=100,label='Future native-outline reference')
    axes[0].axhline(0,color='black',lw=.7);axes[0].set_xlabel('Source time (s)');axes[0].set_ylabel('Matched inter-packet dx (px)');axes[0].legend(fontsize=8);axes[0].set_title('Repeated propeller pattern: false reverse match')
    axes[1].invert_yaxis();axes[1].legend(fontsize=7);axes[1].set_title('12.2s origin: 900.27px -> 112.48px');axes[1].set_xlabel('Camera x (px)');axes[1].set_ylabel('Camera y (px)')
    fig.tight_layout();fig.savefig(OUT/'match_alias.png',dpi=150);plt.close(fig)
    e=window(ROOT.parent/'output/frozen_tangent287_20261108/cache',15.);e=e[e['t']>=15000000-20000]
    ids=e['y'].astype(np.int64)*1280+e['x'];count=np.bincount(ids,minlength=720*1280).reshape(720,1280)
    old=json.loads((ROOT.parent/'output/frozen_tangent287_20261108/references.json').read_text('utf8'))['records']['15.0']['box']
    new=json.loads((ROOT.parent/'output/reference290_20261108/references.json').read_text('utf8'))['records']['15.0']['box']
    fig,ax=plt.subplots(figsize=(12,5));ax.imshow(255-np.clip(count*20,0,255),cmap='gray',vmin=0,vmax=255)
    from matplotlib.patches import Rectangle
    for b,color,label in [(old,'red','Invalid whole-image reference (includes foliage)'),(new,'green','Image-reviewed target enclosure, evaluation only')]:
        ax.add_patch(Rectangle(b[:2],b[2]-b[0],b[3]-b[1],fill=False,color=color,lw=1.5,label=label))
    ax.legend(fontsize=8,loc='lower right');ax.set_title('Recording3: reference error is not drone forecast error');fig.tight_layout();fig.savefig(OUT/'reference_error.png',dpi=150);plt.close(fig)
    table=''
    for n in names:
        a,b=scores[n]['control'],scores[n]['tangent'];am,bm=a['error']['mean'],b['error']['mean']
        table+=f'<tr><td>{n}</td><td>{a["available"]}/{a["eligible"]}</td><td>{am:.2f}</td><td>{bm:.2f}</td><td>{am-bm:+.2f}</td><td>{a["error"]["p95"]:.2f} → {b["error"]["p95"]:.2f}</td></tr>'
    factory=json.loads((ROOT.parent/'output/factory292_20261108/results.json').read_text('utf8'))
    summary=dict(scores=scores,factory=factory,goal30px_achieved=False,new_default_promoted=False,
        research_is_native_rotor_only_without_NN=True,source_time_not_wall_clock=True,
        new3_invalid_unreviewed_scores_replaced_by_separate_reviewed_annotation_not_model_tuning=True)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),'utf8')
    body=f'''<!doctype html><meta charset="utf-8"><title>精度主线与错配根因</title><style>body{{font:17px/1.7 system-ui,"Microsoft YaHei";max-width:1120px;margin:32px auto;padding:0 24px;color:#152238}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #cbd5e1;text-align:left}}.note{{padding:18px;background:#f1f5f9;border-left:5px solid #2563eb}}code{{background:#eef2f7}}a{{color:#1d4ed8}}</style>
    <h1>本轮精度主线：匹配错配与未知力延续</h1>
    <p class="note"><b>30px还没达到，旧默认保留。</b>没有NN、没有按录像调参数。新的切向未知响应学习已接成显式RAW实验参数；它在三段开发录像和一段新录像上有小幅均值收益，另一段退化。不能称普适成功，也不是已经求出真实推力。</p>
    <h2>实际预测成绩：同一算法、同一固定起点</h2>
    <table><tr><th>录像</th><th>可评分/固定起点</th><th>原版均值px</th><th>新模块px</th><th>下降px</th><th>P95 px（尾部）</th></tr>{table}</table>
    <p>全部为0.4秒源时间FDE，不是实际延迟后的实时成绩。13/15/16/11均开发数据；3的候选源码在检查该录像结果前冻结。3仅5个起点，且使用助手图像核验后的事件可见轮廓参考，没有独立人工COM真值。尾部退化逐项保留，不混用不同版本拼成绩。</p><img src="means.png"><img src="all_origins.png">
    <h2>真正改了什么</h2>
    <p>仍是RAW原生ON/OFF → 桨弧、五圆与可见性 → 当前可观测贡献＋未知合力 → 预测。新模块对未知合力加一个“随速度变化”的有效响应，不分别要求测风、重力、质量。局部近似 a=Hn+b−cv；沿桨盘法线的垂直方向投影，可消去包括隐藏桨在内的法向推力Hn，再用过去的原生位移差学习c。c只用于未来力的有效延续时间，不是物理阻力标定。</p>
    <p>进一步先消去未知常量b，再计算c的数据可辨识信息，避免把零均值风/重力先验错误当成学到了阻力。恒速不可辨识时不改原预测；加入任意常量外力、旋转坐标后，系数不应改变，已通过机制检查。没有未来轨迹标签、没有录像ID分支。</p>
    <p>动力学启发来自<a href="https://rpg.ifi.uzh.ch/docs/RAL18_Faessler.pdf">Faessler等的rotor-drag原论文</a>；这里是自己的二维有效响应近似，不是其完整3D控制器复现。<a href="../plane_ablation291_20261108/results.json">固定向上法线对照</a>表明真实几何在四段开发数据上优于固定向上，但新3上反而较弱，因此不宣称几何增量已经普适。</p>
    <h2>900px坏例是怎样发生的</h2><img src="match_alias.png">
    <p>这900px发生在新恢复原型，不是旧版原始精度。高速横穿超过零中心±40px匹配范围，先撞边界，再把另一个相似桨盘匹配成dx=−37.55px；启用项目已有惯性搜索中心、保留当前区域重锁后，该起点900.27→112.48px。仍很差，不能称问题全解。整段原版没有可评分输出，恢复版4个起点均45.52px。更早的直接对齐图版本曾27.75px，但在另一录像明显退化，没有选择性发布它。</p>
    <p>恢复分支仅用补偿证明周期，再将支持映回原事件坐标。虚拟像素频率不当作原生motorRPM，新增未标定推力全归未知项。短时图像桥接另有0.12秒身份时效与双向匹配，但无精度收益，未默认。原15/16同起点误差不变，13略退，因此恢复不是本轮选定的精度默认。</p>
    <h2>新录像中的500px不能当预测成绩</h2><img src="reference_error.png">
    <p>旧的自动参考方法把树木与无人机一起框了进去。该次自动分数已经标为无效，原记录保留。逐张检查原生图后，用同一个天空包围区域生成独立于模型的评估标注，修正为69.68→68.47px。评估区域从未进入检测、状态估计或参数学习；全部25个查询仍保留，目标不在场就不提供参考。助手标注也不能冒充独立人工真值。</p>
    <h2>怎样使用与当前卡点</h2>
    <p>现有<code>run_rotor_candidate.py</code>显式增加<code>--rotor-tangential-memory</code>可试新模块。默认不启用。真实公共工厂181包与冻结实验预测最大差{factory['actual_factory_prediction_max_px']:.3g}px，当前位置/速度/加速度保持原状态；不支持的组合在GPU启动前拒绝。没有新30FPS认证。</p>
    <p><b>目前主要卡点仍是变向时未来净力的方向和延续。</b>新3在15.6s当前位置只有约4.4px误差，但0.4秒预测175.5px；其后可见轨迹确有反向。是否在发出预测前已有可靠桨叶征兆尚未验证，不能统称不可预测。后续应只攻这一环：可靠的当前桨盘变化、同源周期差动与未知力响应，证明它能在变向开始时提供增量；不再回头大规模调检测框中心或换NN主干。</p>
    <p><a href="summary.json">完整数据</a> · <a href="../../event_drone_forecast/docs/HANDOFF_20261108_TANGENT292.md">交接文档</a></p>'''
    (OUT/'index.html').write_text(body,'utf8')
    print(OUT/'index.html')
