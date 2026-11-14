"""Summarize measured improvements and remaining failures, no score retuning."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,html,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pure_visibility170 import window

def main():
    out=ROOT.parent/'output/precision333_20261109';out.mkdir(exist_ok=False)
    d=json.loads((ROOT.parent/'output/inverse326_20261109/results.json').read_text('utf8'));cases=d['cases']
    cases['4']=json.loads((ROOT.parent/'output/inverse326_20261109/extension4.json').read_text('utf8'))
    colors=dict(control='#ba3a36',tangent289='#a79060',inverse320='#176b9c',combined='#00856a')
    names=dict(control='Original rotor baseline',tangent289='Earlier tangent memory289',inverse320='Inverse normal response326',combined='326 +289')
    fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True)
    for ax,n in zip(axes,('13','15','16')):
        for arm in ('control','inverse320','combined'):
            rr=[r for r in cases[n][arm]['records'] if r['valid']];ax.plot([r['t'] for r in rr],[r['error_px'] for r in rr],'.-',color=colors[arm],label=names[arm],lw=1.3)
        ax.axhline(30,color='#777777',ls='--',lw=1);ax.set_ylabel('0.4s FDE (px)');ax.set_title(f'Recording {n}: all fixed origins retained');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);axes[-1].set_xlabel('Prediction origin (s)');fig.tight_layout();fig.savefig(out/'fixed_errors.png',dpi=140);plt.close(fig)
    nms=list(cases);fig,ax=plt.subplots(figsize=(10,4));xx=np.arange(len(nms))
    for i,arm in enumerate(('control','inverse320','combined')):ax.bar(xx+(i-1)*.24,[cases[n][arm]['error']['mean'] for n in nms],.24,label=names[arm],color=colors[arm])
    ax.axhline(30,color='#777777',ls='--');ax.set_xticks(xx,[f'{n} (n={cases[n]["control"]["available"]})' for n in nms]);ax.set_ylabel('Mean0.4s FDE(px)');ax.legend(fontsize=8);ax.set_xlabel('All previously exposed/development recordings;4 is frozen extension');fig.tight_layout();fig.savefig(out/'means.png',dpi=140);plt.close(fig)
    # Show one largest improvement AND one unresolved worst case. Future native
    # images are for post-hoc visualization only, never predictor input.
    a=cases['16']['control']['records'];b=cases['16']['inverse320']['records'];pairs=[(x,y) for x,y in zip(a,b) if x['valid'] and y['valid']];best=max(pairs,key=lambda pair:pair[0]['error_px']-pair[1]['error_px'])
    a=cases['3']['control']['records'];b=cases['3']['inverse320']['records'];worst=max([(x,y) for x,y in zip(a,b) if x['valid'] and y['valid']],key=lambda pair:pair[1]['error_px'])
    examples=[]
    for n,pair,title in [('16',best,'Largest measured improvement'),('3',worst,'Unresolved worst case')]:
        old,new=pair;t=old['t']+.4;cache=ROOT.parent/('output/frozen_tangent287_20261108/cache' if n=='3' else 'output/phase185_20261107/cache_recording16')
        e=window(cache,t);e=e[e['t']>=round(t*1e6)-20000];counts=np.bincount(e['y'].astype(np.int64)*1280+e['x'],minlength=1280*720).reshape(720,1280);im=np.uint8(255-np.clip(counts*20,0,255))
        fig,ax=plt.subplots(figsize=(12,6));ax.imshow(im,cmap='gray',vmin=0,vmax=255)
        target=np.asarray(old['target']);p0=np.asarray(old['prediction']);p1=np.asarray(new['prediction'])
        ax.scatter(*target,c='#14934b',s=100,marker='+',label='Weak visible-outline reference');ax.scatter(*p0,c=colors['control'],s=80,marker='x',label=f'Original {old["error_px"]:.2f}px');ax.scatter(*p1,c=colors['inverse320'],s=80,marker='x',label=f'New {new["error_px"]:.2f}px')
        for p,color in [(p0,colors['control']),(p1,colors['inverse320'])]:ax.plot([target[0],p[0]],[target[1],p[1]],color=color,lw=1)
        ax.set_xlim(min(0,p0[0]-30,p1[0]-30),max(1280,p0[0]+30,p1[0]+30));ax.set_ylim(max(720,p0[1]+30,p1[1]+30),min(0,p0[1]-30,p1[1]-30));ax.legend(fontsize=8);ax.set_title(f'{title}: recording{n}, origin{old["t"]:.1f}s, actual future native image at{t:.1f}s');fig.tight_layout();file=f'example{n}.png';fig.savefig(out/file,dpi=130);plt.close(fig);examples.append(dict(recording=n,file=file,baseline=old,new=new))
    rows=''
    for n,c in cases.items():
        vals=[c[a]['error']['mean'] for a in ('control','inverse320','combined')];rows+=f'<tr><td>{n}</td><td>{c["control"]["available"]}/25</td>'+''.join(f'<td>{v:.2f}</td>' for v in vals)+f'<td>{vals[0]-vals[1]:+.2f}</td></tr>'
    factory=json.loads((ROOT.parent/'output/factory331_20261109/results.json').read_text('utf8'));check=json.loads((ROOT.parent/'output/invariants328_20261109/results.json').read_text('utf8'))
    switch=ROOT.parent/'output/switch332_20261109/recording4/results.json';switchtext='尚未完成，不作结论。'
    if switch.exists():
        q=json.loads(switch.read_text('utf8'));switchtext=html.escape(json.dumps(dict(errors={k:v['error'] for k,v in q['cases'].items()},current_axis_qualified=q['current_axis_qualified']),ensure_ascii=False))
    document=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>桨叶主线：未知法向响应与实际精度</title><style>body{{font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif;max-width:1080px;margin:36px auto;padding:0 20px;color:#243244}}h1,h2{{line-height:1.3}}h2{{margin-top:38px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #dce3e9;text-align:right}}td:first-child,th:first-child{{text-align:left}}img{{max-width:100%;height:auto}}pre{{white-space:pre-wrap;background:#eef3f6;padding:18px}}.note{{background:#fff3d6;padding:16px}}a{{color:#176b9c}}</style>
<h1>桨叶主线：未知法向响应开始产生实际预测收益</h1><p>2026-11-09 · 303–333批次收录 · 原始ON/OFF与几何路线，无整机NN。</p>
<p class="note"><b>30px / 持续30Hz仍未达成。</b>新机制在13/15/16/11降低平均误差约1.30/3.65/3.75/2.45px，3与4基本无收益。不能称普适成功；没有新的完整RAW实时FPS成绩。以下参考是事件可见轮廓，非独立质心、姿态或转速真值。</p>
<h2>同一起点的实际精度</h2><table><tr><th>录像</th><th>可评分/原定起点</th><th>原基线px</th><th>新326 px</th><th>326＋旧289 px</th><th>原−新px</th></tr>{rows}</table>
<p>预测0.4秒；固定10.8–15.6秒、间隔0.2秒；保留所有25个查询及缺失原因。全部已有录像已暴露给开发；4号仅表示候选冻结后的扩展，历史项目可能读过，不能称独立盲测。不按录像选最优分支，也不把不同实验的收益直接相加。</p><img src="means.png" alt="六段录像平均误差"><img src="fixed_errors.png" alt="三个主要录像全部固定起点误差">
<h2>真正改了什么</h2><p>当前已测桨输入与未知力仍分开；隐藏桨推力留在未知项。新增机制利用当前桨盘平面的切向，先消除任意、彼此不等的未知桨推力，再从过去事件位移估计外部偏置和速度响应。只有可观测的方向参与幅值校准；恒定平面不能分清法向推力与外部偏置时回退。重力、风、阻力仍是图像域未知响应，未测出质量或牛顿推力。</p>
<pre>原生RAW → ON/OFF周期资格 → 仅原始像素重建桨叶支撑
 → 椭圆/原弧与五圆模型，4/3/2/侧线保持资格分工
 → 原生相对位移＋当前平面 → 同一个位置/速度/二力状态
 → 过去切向运动消去隐藏推力 → 估计可观测外部残差
 → 校准未来法向响应幅值＋法线变化通过阻力响应积分
 → 输出0.1/0.2/0.4秒位置（当前p/v/a不改）</pre>
<p>原先把“未来法线旋转”和“净加速度衰减”直接相乘；若衰减用于速度阻力，正确关系是 <code>ȧ = −βa + A ṅ</code>（当前先取A短时不变）。新核按这个动力学积分。新幅值是模型条件估计，不是当前逐桨已测力；没有把隐藏推力伪装成观测。椭圆法线的±二义性也统一到原观察器的图像向上先验。</p>
<h2>为什么没有继续只优化椭圆和转速</h2><p>连续相位303–311增加了多环周期覆盖，311已能仅从原生事件自动初始化、无需旧作者周期文件；但向预测转化仍微小，且CPU研究版远不够实时，因此未默认接入RAW。314的四源增益重分配虽然保持当前状态与协方差，却使15号退化，未采用。316/317的更精细未来角度平均收益不足1px。</p>
<p>318/319进行了刻意使用未来的诊断：提前代入后续测得的桨盘方向几乎不改善，提前代入后续滤波净加速度可到约23–24px。<b>这不是因果算法、不是实际成绩，也不是独立加速度真值。</b>把未来法向与外部残差单独替换反而常变差，说明两项抵消且分解不稳，这才引出了当前幅值校准。</p>
<h2>成功例和仍然失败的例子</h2><img src="example16.png" alt="实际收益最大的16号同起点"><img src="example3.png" alt="3号仍未解决的大误差"><p>图像是预测时刻之后的实际原生事件，仅用于事后展示。预测计算没有读取未来画面。</p>
<h2>4号录像与参考错误</h2><p>自动全画面count框曾把树木选为目标，约217px的初评分已标INVALID，原文件保留。全帧目视核验后只在评价端给目标粗区域，保持count8/count4与面积16规则；未改推理、未改25起点。修正后原62.71px，新62.74px，未产生收益。13.6秒目标与树枝相接，参考仍可能含偏差，已明确注记。此段只有52/181包当前检出、60/181包有效状态、51包有合格方向；这些不是全数据集召回率。<a href="../reference323_20261109/corrected_crops.png">原生参考核验</a>。</p>
<h2>分支资格小试</h2><p>重用既有中心/角度不确定度门槛，测试低置信椭圆不再压住新的侧线观测，默认不改变。结果：<code>{switchtext}</code></p>
<h2>已完成的有效检查</h2><p>实际RAW工厂181包：每条分支都核验事件数、检测与有效状态，候选与冻结回放最大差{max(factory['actual_factory_forecast_max_delta'].values()):.3g}px，当前状态最大差{factory['current_state_max_delta']:.3g}。法线正负翻转后预测差{check['headless_sign_flip_forecast_max_px']:.3g}px，含角加速度的阻力积分核对数值积分误差{check['variable_angular_acceleration_drag_kernel_max_error']:.3g}。这类缓存/工厂核验不算完整RAW30FPS。</p>
<p class="note">322、327多分支脚本曾只给第一条分支一次性GPU缓存，其他分支读空事件，导致空比较。两批已标INVALID；329因期望版本过旧主动中止；331修复后重新完成真实核验。单路324与CPU321/326不受此错误影响。</p>
<h2>调用与保留范围</h2><pre>在原run_rotor_candidate.py命令中追加：
  --rotor-inverse-normal-response

若同时试旧切向记忆，再追加：
  --rotor-tangential-memory</pre><p>新开关默认关闭，原算法和公开NN发布包保留。没有修改两处yolov10-main、原RAW、SDK或权重，没有训练、安装或上传。</p>
<h2>文献依据与边界</h2><p><a href="https://rpg.ifi.uzh.ch/docs/RAL18_Faessler.pdf">Faessler等的含阻力多旋翼动力学</a>提供力沿机体轴、阻力与速度相关的物理依据；这里改写成未标定图像响应，不能照搬其真实参数精度。<a href="https://arxiv.org/html/2610.31384v1">CoHelix</a>与<a href="https://github.com/radimspetlik/helixtrack">HelixTrack</a>用于持续相位/共享几何调查。<a href="https://www.aau.at/wp-content/uploads/2023/08/2023_Framework_for_UAV_System-Parameter_Estimation_with_Disturbance_Detection.pdf">在线参数与扰动可观测性框架</a>提醒不能把未知参数分解当作无条件可辨识。本文工程组合不声称首次提出这些原理。</p>
<p><a href="../../event_drone_forecast/docs/HANDOFF_20261109_INVERSE333.md">详细交接</a> · <a href="results.json">结果数据</a> · <a href="../inverse326_20261109/results.json">所有逐起点评分</a></p></html>'''
    (out/'index.html').write_text(document,'utf8');(out/'results.json').write_text(json.dumps(dict(cases=cases,examples=examples,factory=factory,invariants=check,goal30px30Hz_achieved=False,NN_used=False),indent=2),'utf8');(out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print(out/'index.html')

if __name__=='__main__':main()
