"""Summarize saved inference before opening manual scoring references."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from compare_final_pipeline import reference_at,stats
from physical_publication import evaluate


def read(path):return json.loads(path.read_text('utf-8'))
def rows(path):return [json.loads(s) for s in path.read_text('utf-8').splitlines()]


def main(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    out=Path(output);base=ROOT.parent/'output/rotor_pipeline_20261101'
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'figure.dpi':140})
    runs={name:read(out/name/'summary.json') for name in
        ['clear_cpu_legacy_control','side_cpu_legacy_control','clear_final','side_final','raw_entry_final']}
    data={c:rows(out/f'{c}_final/predictions.jsonl') for c in ['clear','side']}
    baseline={c:rows(base/f'{c}_bank/predictions.jsonl') for c in ['clear','side']}
    equivalence={}
    for c in data:
        pairs=list(zip(data[c],baseline[c]));center=[]
        for a,b in pairs:
            if a['body'].get('detected') and b['body'].get('detected'):
                center.append(float(np.linalg.norm(np.array(a['body']['center'])-b['body']['center'])))
        equivalence[c]=dict(frames=len(pairs),max_center_difference_px=max(center,default=0.),
            current_detection_flags_equal=all(a['body']['detected']==b['body']['detected'] for a,b in pairs),
            body_records_equal=all(a['body']==b['body'] for a,b in pairs),
            period_records_equal=all(a['period']==b['period'] for a,b in pairs),
            rotor_rate_records_equal=all(a['rates']==b['rates'] for a,b in pairs),
            geometry_auxiliary_unequal_frames=sum(a['geometry']!=b['geometry'] for a,b in pairs))
    # Inference above is already on disk. Labels are used only below for scores.
    truth=sorted(read(ROOT.parent/'output/dense_labels_side_110_1195/labels.json')['records'].values(),key=lambda r:r['t'])
    old=read(base/'final_summary.json')['source_force_ablation']['side_final']['fixed_source_queries']
    arms=dict(old=baseline['side'],new=data['side'],
        muted=rows(out/'side_observer_off.jsonl'))
    scored=[]
    for record in old['rows']:
        q=record['query_t'];target=reference_at(truth,q+.4);item=dict(record)
        for name,arm in arms.items():
            r=[r for r in arm if r['t']<=q+1e-9][-1]
            assert r['valid'] and q-r['t']<=.16 and target is not None
            h=float(q-r['t']+.4)
            item[name+'_error_px']=float(np.linalg.norm(np.array(evaluate(r,[h])[str(h)])-target))
        scored.append(item)
    score={name:stats([r[name+'_error_px'] for r in scored]) for name in arms}
    check=read(out/'mechanism_checks.json')
    fig,axs=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    q=[r['query_t'] for r in scored]
    for name,label,color in [('old','原桨叶入口＋旧残差','#dc2626'),('new','新联合二力状态','#2563eb'),('muted','新状态，关闭周期输入','#64748b')]:
        axs[0].plot(q,[r[name+'_error_px'] for r in scored],'o-',label=label,color=color,ms=4)
    axs[0].axhline(25,color='#15803d',ls=':',label='25px目标')
    axs[0].set_xlabel('固定源查询时刻 / s');axs[0].set_ylabel('0.4秒终点误差 / px');axs[0].legend(fontsize=8)
    medians=[runs[n]['compute_ms_median'] for n in ['clear_cpu_legacy_control','clear_final','side_cpu_legacy_control','side_final']]
    axs[1].bar(['清晰原前端','清晰GPU前端','侧视原前端','侧视GPU前端'],medians,color=['#94a3b8','#2563eb','#94a3b8','#2563eb'])
    axs[1].axhline(1000/30,color='#dc2626',ls='--',label='30Hz预算33.3ms')
    axs[1].set_ylabel('核心计算中位耗时 / ms');axs[1].legend(fontsize=9)
    for i,value in enumerate(medians):axs[1].text(i,value+4,f'{value:.0f}',ha='center')
    fig.suptitle('真实保存事件的小试：预测长尾改善，GPU加速仍未达到30Hz',fontsize=12)
    fig.savefig(out/'results.png',bbox_inches='tight');plt.close(fig)
    # The difficult instant uses the actual source-time image and the two
    # already-saved predictions; the future star is a scoring reference only.
    q=110.4
    new=[r for r in data['side'] if r['t']<=q+1e-9][-1]
    oldrow=[r for r in baseline['side'] if r['t']<=q+1e-9][-1]
    index=min(range(len(data['side'])),key=lambda i:abs(data['side'][i]['t']-new['t']))
    from rotor_support_front import accumulate
    from rotor_only_front import filtered_rgb
    with np.load(base/f'cache/side/{index:04d}.npz') as z:
        on,off,_=accumulate(z['events'],float(z['t']))
    image=filtered_rgb(on,off,np.ones(on.shape,bool))
    fig,ax=plt.subplots(figsize=(10,5.5),layout='constrained');ax.imshow(image)
    curves=[]
    for r,label,color in [(oldrow,'旧残差模型','#dc2626'),(new,'联合二力状态','#2563eb')]:
        horizons=np.linspace(0,.4,50)+(q-r['t'])
        values=evaluate(r,[float(h) for h in horizons]);curve=np.array([values[str(float(h))] for h in horizons]);curves.append(curve)
        ax.plot(curve[:,0],curve[:,1],label=label,color=color,lw=2)
        ax.scatter(*curve[-1],color=color,marker='x',s=80)
    target=reference_at(truth,q+.4);ax.scatter(*target,color='#22c55e',marker='*',s=180,label='事后人工0.4秒参考')
    points=np.vstack(curves+[target[None],np.asarray(new['body']['bbox']).reshape(2,2)])
    ax.set_xlim(max(0,points[:,0].min()-65),min(1280,points[:,0].max()+65))
    ax.set_ylim(min(720,points[:,1].max()+70),max(0,points[:,1].min()-70))
    ax.set_title('110.4秒：同一当前桨叶观测，旧157.33px → 新约30.42px')
    ax.set_xlabel('当前原生ON/OFF画面；未来绿星只用于评分，不参与推理');ax.legend(fontsize=9)
    fig.savefig(out/'bad_case.png',bbox_inches='tight');plt.close(fig)
    result=dict(runs=runs,observation_equivalence=equivalence,fixed_side_scores=score,fixed_side_rows=scored,
        mechanism_checks=check,observer_checks=read(out/'observer_mechanism_checks.json'),
        new_independent_RPM_or_thrust_proven=False,actual_30Hz_proven=False,
        evaluation='Two exposed development clips, nine fixed side queries, same manual whole-body center proxy. Source time only; computation delay not included.',
        original_NN_default_preserved=True,rotor_only_default_uses_joint_observer=True)
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    table=''
    for c,title in [('clear','清晰1.2秒'),('side','侧视2秒')]:
        a,b=runs[c+'_cpu_legacy_control'],runs[c+'_final']
        table+=f'<tr><td>{title}</td><td>{a["compute_capacity_fps"]:.2f} → {b["compute_capacity_fps"]:.2f}</td><td>{a["compute_ms_median"]:.1f} → {b["compute_ms_median"]:.1f}</td><td>{b["compute_ms_p95"]:.1f}</td><td>{b["current_rotor_detections"]}/{b["frames"]}</td></tr>'
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>五圆二力与GPU前端优化</title>
    <style>body{{font:17px/1.75 system-ui;max-width:1120px;margin:30px auto;padding:0 20px;color:#24384a}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #bbc7d2;padding:8px}}.note{{background:#fff0cf;padding:16px}}pre{{background:#eef3f8;padding:15px;white-space:pre-wrap}}</style>
    <h1>五圆模型、二力状态与GPU前端：实际改动</h1><p>2026-11-01。纯RAW ON/OFF → 桨叶周期筛选 → 原生支撑 → 五圆/部分桨/侧视 → 周期推力代理＋合并未知作用 → 位置预测。没有引入整机NN或G重建，原整机基线保留。</p>
    <p>新联合状态为位置、速度、合并未知加速度和有效周期增益。上一个输入作用于已经过去的区间；当前输入只改变当前及未来加速度。前四次观测不启用周期力，缺桨先验不用于学习增益。旧的0.2秒位移残差覆盖路径已从新默认桨叶入口移除。</p>
    <h2>预测与速度</h2><img src="results.png"><p>固定九个侧视查询的0.4秒均值：旧桨叶入口 {score['old']['mean']:.2f}px → 联合二力状态 {score['new']['mean']:.2f}px；新状态关闭周期输入 {score['muted']['mean']:.2f}px。周期本身的额外收益很小，主要改进来自状态/残差机制，不能宣称精确推力已经成为主要贡献。仍有单点退化，尚未满足稳定25px。</p>
    <p class="note">这是源时间离线小试，未计计算结束到预测目标的真实延迟；九个相关查询不是独立大规模测试。人工参考仍为整机框中心，不能冒充桨心/重心或RPM真值。清晰段没有足够同口径密集参考，不编造其预测精度。</p>
    <table><tr><th>同缓存短片</th><th>核心计算能力FPS</th><th>中位耗时ms</th><th>新版P95 ms</th><th>当前区域候选</th></tr>{table}</table>
    <p>本轮新旧前端均关闭图片/视频导出，保留推理/诊断；旧前端本身也使用CUDA频谱。核心耗时不含RAW解码和GPU启动，不能称实际持续30Hz。硬件与负载下的短段结果存在波动。</p>
    <h2>110.4秒坏例</h2><img src="bad_case.png"><p>当前定位相同，旧残差产生错误的强向右加速度；联合估计避免将过去位移偏差直接覆盖当前状态。这是某个坏例的改善，不代表所有变向均成功。</p>
    <h2>GPU实际搬了什么</h2><p>ON/OFF累积、整数密度筛选、像素分组、频谱及后半窗验证、高斯与形态学支撑在同一CUDA工作进程。只返回稀疏计数、周期和支撑；CPU在支撑局部做连通域，保留小规模几何与状态更新。九个真实保存窗口的密度、周期和原生支撑掩膜与参考一致。CPU匹配也只在搜索ROI内生成过滤纹理，未使用掩膜外机体/背景。</p>
    <p>短窗口稀疏回传约0.18–1.81MB，原接口三张全幅结果图约8.29MB。输入最多400万原生事件，超出会明确报错，不悄悄丢事件或退回慢CPU。GPU和CPU仍同步，持续30Hz和有界异步流水线尚未完成。</p>
    <h2>RAW入口已执行</h2><p>直接只读E盘train6的26.2–26.4秒：{runs['raw_entry_final']['frames']}个包，{runs['raw_entry_final']['current_rotor_detections']}次当前候选，{runs['raw_entry_final']['valid_forecasts']}次有效源预测。该0.2秒启动小试周期银行未完成初始化，当前推力代理输入为0，不把它写成已完成四桨推力验证。“侧视”是历史片段名，不是每一帧的人工视角标签。</p>
    <pre>python run_fast_structure.py --rotor-only --rotor-gpu-front --no-preview --gpu-python &lt;已有CUDA Python路径&gt; --raw &lt;event.raw&gt; --start 26.2 --duration 1.2 --output &lt;新目录&gt;</pre>
    <p>--rotor-legacy-force 可回放原状态/残差控制；不加--rotor-gpu-front仍可走原CPU空间前端。默认整机NN入口、模型、RAW、保护目录和已发布ZIP未修改。</p>
    <h2>仍未做完的部分</h2><p>30Hz还受设备调度、CPU几何和匹配、稀疏接口开销限制；尚未完成有界流水线及失锁时宽频/锁定时窄频调度。HelixTrack历史适配保留，不重新从头接入，也未把作者代码混入发行包。周期谐波资格、真实RPM、部分桨总推力尺度及法线符号仍需继续解决。</p>
    <p><a href="summary.json">完整数值与逐项检查</a></p></html>'''
    (out/'index.html').write_text(html,'utf-8')
    print(json.dumps(dict(scores=score,runtime={k:dict(fps=v['compute_capacity_fps'],median_ms=v['compute_ms_median']) for k,v in runs.items()},equivalence=equivalence),ensure_ascii=False))


if __name__=='__main__':main(sys.argv[1])
