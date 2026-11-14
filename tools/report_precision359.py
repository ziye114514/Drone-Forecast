"""Publish cumulative precision, coverage and negative results without cherry-picking."""
from pathlib import Path
import json,sys,html,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from physical_publication import evaluate
WORK=ROOT.parent


def read(path):return json.loads((WORK/path).read_text('utf8'))
def rows(path):return [json.loads(s) for s in (WORK/path).read_text('utf8').splitlines()]


if __name__=='__main__':
    out=WORK/'output/precision359_20261109';out.mkdir(exist_ok=True)
    force=read('output/force357_20261109/results.json');cases=force['cases']
    fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    table=[];oldall=[];newall=[]
    for ax,(n,c) in zip(axes.flat,cases.items()):
        a=[r for r in c['control']['records'] if r['valid']];b=[r for r in c['candidate']['records'] if r['valid']]
        assert [r['t'] for r in a]==[r['t'] for r in b]
        ax.plot([r['t'] for r in a],[r['error_px'] for r in a],'o-',color='#7c879a',label='inverse326')
        ax.plot([r['t'] for r in b],[r['error_px'] for r in b],'o-',color='#138a72',label='disk-response357')
        ax.axhline(30,color='#c04843',ls='--',lw=1);ax.set(title=f'Recording {n} | same {len(a)} origins',xlabel='query time (s)',ylabel='0.4s FDE (px)');ax.grid(alpha=.2)
        oldall.extend(r['error_px'] for r in a);newall.extend(r['error_px'] for r in b)
        sa=c['control']['error'];sb=c['candidate']['error'];table.append(f"<tr><td>{n}</td><td>{len(a)}</td><td>{sa['mean']:.2f}</td><td>{sb['mean']:.2f}</td><td>{sb['mean']-sa['mean']:+.2f}</td><td>{sa['p95']:.2f} → {sb['p95']:.2f}</td></tr>")
    axes.flat[0].legend();fig.savefig(out/'all_origins.png',dpi=150);plt.close(fig)
    continuation={}
    for n in ('13','15','16','11','3','4'):
        batch='structural348_20261109' if n in ('13','11','4') else 'continuation339_20261109' if n in ('15','16') else 'continuation344_20261109'
        p=WORK/f'output/{batch}/recording{n}/results.json'
        if not p.exists():
            #15/16 have no forecast change at these common origins. The
            # original continuation batch is identified by its stored data.
            found=[q for q in (WORK/'output').glob(f'*339*/recording{n}/results.json')]
            if not found:continue
            p=found[0]
        continuation[n]=json.loads(p.read_text('utf8'))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    names=list(continuation);xx=np.arange(len(names));w=.36
    for ax,key,title in ((axes[0],'detections','Current detections /181 packets'),(axes[1],'precision','Common-origin 0.4s FDE (px)')):
        values=[]
        for arm in ('control','continued'):
            v=[continuation[n]['diagnostics'][arm]['current_detections'] if key=='detections' else continuation[n]['common_origins'][arm]['mean'] for n in names];values.append(v)
        ax.bar(xx-w/2,values[0],w,color='#7c879a',label='inverse326');ax.bar(xx+w/2,values[1],w,color='#1886b3',label='current-period continuation')
        ax.set(title=title,xticks=xx,xticklabels=names,xlabel='recording');ax.grid(axis='y',alpha=.2)
    axes[0].legend();fig.savefig(out/'coverage_precision.png',dpi=150);plt.close(fig)
    # Two fixed common origins: preserve one improvement AND one regression.
    rr=rows('output/structural348_20261109/recording4/continued.jsonl');cc=rows('output/structural348_20261109/recording4/control.jsonl')
    scores=read('output/structural348_20261109/recording4/results.json')['cases']
    cache=WORK/'output/frozen324_20261109/cache';manifest=read('output/frozen324_20261109/cache/manifest.json')['frames']
    fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    for ax,t in zip(axes,(13.2,13.4)):
        i=max(i for i,r in enumerate(rr) if r['t']<=t+1e-9);r=rr[i];e=np.load(cache/manifest[i]['file'])
        ids=e['y'].astype(np.int64)*1280+e['x'];on=np.bincount(ids[e['p']>0],minlength=1280*720).reshape(720,1280);off=np.bincount(ids[e['p']==0],minlength=1280*720).reshape(720,1280)
        scale=max(1,float(np.percentile(np.r_[on[on>0],off[off>0]],99)));image=np.zeros((720,1280,3));image[:,:,0]=np.clip(np.log1p(on)/np.log1p(scale),0,1);image[:,:,2]=np.clip(np.log1p(off)/np.log1p(scale),0,1)
        ax.imshow(image,origin='upper');target=next(q['target'] for q in scores['continued']['records'] if q['valid'] and q['t']==t)
        points=[]
        for source,color,label in ((cc[i],'#ee8ced','inverse326'),(r,'#31e1dc','continuation348')):
            pred=np.asarray(evaluate(source,[t+.4-source['t']])[str(float(t+.4-source['t']))]);points.append(pred)
            ax.scatter(*pred,c=color,marker='x',s=90,label=label)
        ax.scatter(*target,c='#ffdc61',marker='+',s=150,label='future weak reference')
        pos=np.asarray(r['coefficient']['position']);ax.scatter(*pos,c='white',s=24,label='current state')
        pts=np.vstack([points,target,pos]);lo=pts.min(0)-100;hi=pts.max(0)+100
        ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.set_title(f'Recording4, query {t:.1f}s → {t+.4:.1f}s')
        if hi[0]>1280:ax.axvline(1280,color='#555',ls='--');ax.axvspan(1280,hi[0],color='#ddd',alpha=.4)
        if lo[0]<0:ax.axvspan(lo[0],0,color='#ddd',alpha=.4)
        ax.legend(fontsize=8,loc='upper left');ax.set_xlabel('image x (px)');ax.set_ylabel('image y (px)')
    fig.savefig(out/'native_examples.png',dpi=150);plt.close(fig)
    negatives=[]
    for stage,batch in ((349,'leverage349'),(351,'keyframe351'),(352,'occupancy352'),(353,'combined353')):
        p=WORK/f'output/{batch}_20261109/recording11/results.json'
        if p.exists():
            v=json.loads(p.read_text('utf8'));negatives.append(f"<li>{stage}：11号共同15起点 {v['common_origins']['continued']['mean']:.2f}px，326为37.18px；保留实验，不接默认。</li>")
    aggregate=dict(common_origins=len(oldall),control_mean=float(np.mean(oldall)),candidate_mean=float(np.mean(newall)),control_p95=float(np.percentile(oldall,95)),candidate_p95=float(np.percentile(newall,95)),control_within30=float(np.mean(np.asarray(oldall)<=30)),candidate_within30=float(np.mean(np.asarray(newall)<=30)))
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>桨叶主线：精度与连续性改进</title><style>body{{max-width:1180px;margin:30px auto;padding:0 20px;font:16px/1.85 system-ui;color:#20374c;background:#f6f8fb}}section{{background:white;padding:22px;margin:20px 0;border-radius:10px}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border:1px solid #c8d4e1}}code{{background:#eef2f5}}.lead{{border-left:5px solid #198b72}}</style><h1>334–359：精度收益、退化与已接入口</h1>
<section class="lead"><b>30px／持续30FPS仍未达成。</b>新力学候选在六段相同102起点，0.4s均值{aggregate['control_mean']:.2f}→{aggregate['candidate_mean']:.2f}px，30px内比例{aggregate['control_within30']:.1%}→{aggregate['candidate_within30']:.1%}。改善小，且有录像退化；没有“所有录像稳定变好”的证据。已新增统一可运行候选，旧基线／NN发布包保留；没有按录像选算法。</section>
<section><h2>1. 二力模型：把有数据支持的方向分开</h2><p>仍是当前可观测桨叶贡献＋不可直接观测响应。四／三桨用当前合格弧和近期共享几何，两桨／侧线不补造隐藏桨的频率。新候选复用326过去切向逆问题，只让切向响应随可观测数据变化，法向响应保留先验，并包含平面转动时阻力矩阵变化项。当前p/v/a、当前检测和隐藏力归属不变。它是二维投影响应近似，不是质量、CT、真实风速或牛顿推力辨识。</p><table><tr><th>录像</th><th>同起点数</th><th>326均值px</th><th>357均值px</th><th>差值</th><th>P95</th></tr>{''.join(table)}</table><img src="all_origins.png"><p>所有起点均展示；没有剔除出画困难点或按录像调整阈值。102起点合并P95为{aggregate['control_p95']:.2f}→{aggregate['candidate_p95']:.2f}px，尾部问题仍然明显。</p></section>
<section><h2>2. 稀疏周期续接：确实补回输出，但不能等同于更准</h2><p>冷启动原12像素门槛不动；已有目标续接需要同一当前双极性组件中的严格周期像素，以及当前原生画面的双向匹配。当前支撑与当前周期证据每次重验；不把预测模板当新观测。不需要四桨模板，不新增隐藏RPM或总推力。已复用当前共享仿射／退化线相似变换处理位移。</p><img src="coverage_precision.png"><p>4号当前检出52→72，共同10起点62.74→55.94px，额外新增1起点；11号99→124，却在共同15起点37.18→39.61px。新增起点与共同起点分开统计。故续接只提供显式可选开关，未作为精度候选默认。下图保留同一录像的一处收益与一处退化；背景是当前原生ON/OFF，十字是0.4s后的预测和弱参考位置，不是同帧真值。</p><img src="native_examples.png"></section>
<section><h2>3. 失败也保留</h2><ul><li>匿名单环持续相位334／336未补回真实断档；337合成通过不代表真实效果。</li>{''.join(negatives)}<li>355原生推力变化率通路在固定起点上没有收益。356使用已有慢CPU相位，13/15/16仅36.219→36.231、44.828→44.819、41.102→41.098px，不能称有效预测提升或实时能力。</li></ul><p>348共享位移与347侧线位置约束合并也未解决11号，因此没有继续扫描权重。双向闭合只说明匹配自洽，并不证明它就是机体位移。11号续接会保留早期参考偏差；线位置变好后，速度／未来净力延续依然可能变坏。</p></section>
<section><h2>4. 如何运行与交接</h2><p>精度候选：<code>event_drone_forecast/run_rotor_force_candidate.py</code>；当前严格周期续接另加<code>--rotor-periodic-continuation</code>。变化率实验另加<code>--rotor-visible-force-rate</code>，默认不启用。完整命令和限制见<a href="../../event_drone_forecast/docs/HANDOFF_20261109_PRECISION359.md">最新交接</a>。<a href="../force357_20261109/results.json">六段完整结果</a>，<a href="../factory358_20261109/results.json">实际工厂对照</a>。</p><p>所有六段都是已暴露开发数据；参考来自原生事件可见轮廓，不是独立人工COM、姿态或RPM真值。11号旧源212在15.633s之后和当前FOVguard不同，但所有固定评分源时间≤15.6s的控制前缀差为0，尾差单独保留。354/355/356试验JSONL的旧physical显示字段未更新，评分本来就重新调用evaluate，不受该字段影响；可视化以357规范输出为准。</p><p>下一单项仍是“持续可观测的桨叶运动／周期如何约束未来法向力幅值”，而非继续堆检测数、任意放大K或按片段调参。现有无真值数据尚不能分离每个真实推力系数；方向、频率、位移错误都不应统称为风。</p></section>
<section><h2>相关研究及使用边界</h2><p><a href="https://arxiv.org/html/2603.09235v1">HelixTrack</a>和<a href="https://github.com/radimspetlik/helixtrack">作者代码</a>用于既有CPU相位适配；没有修改作者头文件。<a href="https://arxiv.org/html/2610.31384v1">CoHelix</a>涉及共享多桨几何；<a href="https://arxiv.org/html/2603.08386v1">DDHF</a>涉及极性频谱和谐波，极端侧视仍有失败。<a href="https://arxiv.org/html/2511.13100v2">EventPro</a>采用几何引导运动补偿，本项目没有复现它的论文指标。<a href="https://rpg.ifi.uzh.ch/docs/RAL18_Faessler.pdf">Faessler等的方向性转子阻力模型</a>启发本轮二维条件模型，本文论文是已知机载状态的控制任务，与外部RAW预测不是同一评测。<a href="https://arxiv.org/abs/2611.07713">SEPT</a>、<a href="https://arxiv.org/abs/2107.04536">连续时间事件跟踪</a>及<a href="https://arxiv.org/abs/2607.09103">相关特征解耦滤波</a>用于研究比对，没有引入NN或声称作者复现。</p></section></html>'''
    (out/'index.html').write_text(body,'utf8');(out/'summary.json').write_text(json.dumps(dict(force=aggregate,continuation_sources=list(continuation),goal_30px_30FPS_completed=False),indent=2),'utf8')
    print(str(out/'index.html'));print(aggregate)
