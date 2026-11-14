"""Build a local report from frozen outputs, including adverse examples."""
from pathlib import Path
import sys,json,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from physical_publication import evaluate
from nn_rotor_gate import evaluate_gated
from live_decode import join_native

OUT=ROOT.parent/'output/nn_rotor166_20261105'
NAMES={'clear':'清晰片段（仍含退化）','side':'侧视','train2':'另一录像'}
FONT=Path('C:/Windows/Fonts/msyh.ttc')
if FONT.exists():
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(FONT));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams['axes.unicode_minus']=False


def load(path):return [json.loads(s) for s in path.read_text('utf-8').splitlines()]


def native_window(path,t):
    m=json.loads((path/'manifest.json').read_text('utf-8'));end=round(t*1e6);begin=end-20000
    prev=round(m['spec']['start']*1e6);pieces=[]
    for item in m['frames']:
        stop=round(item['t']*1e6)
        if stop>begin and prev<end:
            e=np.load(path/item['file']);pieces.append(e[(e['t']>=begin)&(e['t']<end)])
        prev=stop
    return join_native(pieces,pieces[0].dtype)


def illustration(clip,query,path,scores):
    rows=load(OUT/f'{clip}_gated_native_visible.jsonl');r=max((r for r in rows if r['t']<=query+1e-8),key=lambda r:r['t'])
    obs={r['t']:r for r in load(OUT/f'{clip}_observations.jsonl')}[r['t']]
    events=native_window(path,r['t']);on=np.zeros((720,1280),np.float32);off=on.copy()
    np.add.at(on,(events['y'][events['p']>0],events['x'][events['p']>0]),1)
    np.add.at(off,(events['y'][events['p']==0],events['x'][events['p']==0]),1)
    image=np.ones((720,1280,3),np.float32)
    a=np.clip(np.log1p(on)/np.log(32.),0,1);b=np.clip(np.log1p(off)/np.log(32.),0,1)
    image[:,:,0]-=.8*a;image[:,:,1]-=.8*np.maximum(a,b);image[:,:,2]-=.8*b
    old=next(x for x in scores[clip]['arms']['legacy_exact']['records'] if abs(x['query_t']-query)<1e-6)
    new=next(x for x in scores[clip]['arms']['native_visible']['records'] if abs(x['query_t']-query)<1e-6)
    fig,ax=plt.subplots(figsize=(10,5.3),layout='constrained');ax.imshow(image)
    for rotor in obs['geometry']['rotors']:
        ax.add_patch(Ellipse((rotor['cx'],rotor['cy']),rotor['major'],rotor['minor'],angle=rotor['angle'],fill=False,edgecolor='#d49500',linewidth=1.6))
    center=np.asarray(r['coefficient']['position']);ax.scatter(*center,s=50,c='black',marker='x',label='当前 NN 状态位置')
    for record,color,label in [(old,'#3778bf','旧入口 0.4s 预测'),(new,'#22945a','新增量预测')]:
        ax.scatter(*record['prediction'],s=85,facecolors='none',edgecolors=color,linewidths=2,label=label)
        ax.plot([center[0],record['prediction'][0]],[center[1],record['prediction'][1]],c=color,lw=1)
    ax.scatter(*old['target'],s=160,c='#ba3156',marker='*',label='未来 0.4s 人工参考终点')
    box=np.asarray(obs['body']['bbox']);points=np.array([box[:2],box[2:],old['prediction'],new['prediction'],old['target']])
    lower=points.min(0)-35;upper=points.max(0)+35
    ax.set_xlim(max(0,lower[0]),min(1280,upper[0]));ax.set_ylim(min(720,upper[1]),max(0,lower[1]))
    ax.set_title(f"{NAMES[clip]}：源时刻 {r['t']:.3f}s，模式 {obs['structure']['mode']}\n旧 {old['error_px']:.2f}px → 新 {new['error_px']:.2f}px；当前原生 20ms 事件")
    ax.legend(loc='best',fontsize=9);ax.set_xlabel('像素 x');ax.set_ylabel('像素 y');fig.savefig(OUT/f'example_{clip}.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    scores=json.loads((OUT/'gated_scores.json').read_text('utf-8'))
    first=json.loads((OUT/'scores.json').read_text('utf-8'))['clips']
    direct=json.loads((OUT/'preserved_scores.json').read_text('utf-8'))
    features=json.loads((OUT/'summary.json').read_text('utf-8'))
    fig,axes=plt.subplots(3,1,figsize=(10,9),layout='constrained')
    for ax,clip in zip(axes,['clear','side','train2']):
        for mode,color,label in [('legacy_exact','#3778bf','旧完整入口'),('native_visible','#22945a','新桨叶增量'),('native_evidence_only','#b37920','同资格／已知输入值置零')]:
            records=[r for r in scores[clip]['arms'][mode]['records'] if r['valid']]
            ax.plot([r['query_t'] for r in records],[r['error_px'] for r in records],'.-',c=color,label=label)
        ax.axhline(30,color='gray',linestyle=':',label='30px 目标')
        ax.set_title(NAMES[clip]);ax.set_ylabel('0.4s 终点误差 / px');ax.grid(alpha=.2)
    axes[0].legend(fontsize=9);axes[-1].set_xlabel('查询时刻 / s')
    fig.savefig(OUT/'fixed_query_errors.png',dpi=150);plt.close(fig)
    illustration('clear',29.184013338878,OUT/'cache_clear',scores)
    illustration('side',111.2,ROOT.parent/'output/side_temporal164_20261101/cache_side_warm',scores)
    def table(mode_set,data):
        rows=''
        for clip in ['clear','side','train2']:
            cells=''.join(f"<td>{data[clip]['arms'][mode]['error']['mean']:.3f}</td>" for mode,label in mode_set)
            n=data[clip]['arms'][mode_set[0][0]]
            rows+=f"<tr><th>{NAMES[clip]}</th><td>{n['available']}/{n['eligible']}</td>{cells}</tr>"
        heads=''.join(f'<th>{label}</th>' for mode,label in mode_set)
        return f'<table><tr><th>片段</th><th>可评分／预定</th>{heads}</tr>{rows}</table>'
    main_table=table([('legacy_exact','旧完整入口'),('native_pose','关闭周期'),('native_four','仅四桨'),('native_visible','部分／侧视增量'),('native_evidence_only','同资格，已知输入值置零')],scores)
    weak_table=table([('NN_motion','新共同骨干无桨叶'),('NN_geometry','加几何'),('NN_four_force','加四桨力'),('NN_visible_force','加部分／侧视力'),('saved_old_NN_pipeline','旧完整入口')],first)
    direct_table=table([('legacy_exact','旧完整入口'),('native_four','直接替换，仅四桨'),('native_visible','直接替换，允许部分')],direct)
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>阶段166：NN辅助桨叶预测实测</title>
<style>body{{font:17px/1.75 system-ui,"Microsoft YaHei";color:#202c3b;max-width:1120px;margin:35px auto;padding:0 20px;background:#fafbfd}}h1,h2{{line-height:1.4}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{border:1px solid #d4dce6;padding:9px;text-align:right}}th:first-child{{text-align:left}}img{{max-width:100%;background:white}}.notice{{background:#fff3d3;padding:15px;border-left:5px solid #b8791d}}code,pre{{background:#eaf0f5}}pre{{padding:15px;white-space:pre-wrap}}a{{color:#235f9e}}.flow{{padding:18px;background:#e7f1fc;border-radius:8px}}details{{margin:20px 0}}small{{color:#526477}}</style>
<h1>桨叶逻辑辅助 NN：原型已接通，小收益尚未证明来自推力数值</h1>
<p>2026-11-05 · 阶段166 · 所有预测使用原有冻结 NN/KLT 观测；未训练神经轨迹网络。NN在这里负责定位，不代表新的NN轨迹预测模型。</p>
<div class="notice">清晰开发片段 6 个固定终点：27.555 → 26.889px，下降 0.666px（2.42%）；侧视 37.110 → 37.653px，退化 0.543px；另一录像 68.193px 不变。更严格的同资格控制得到 26.854px，说明清晰段收益主要来自资格与未知合力更新，尚不能归功于周期数值。保留旧默认和纯桨叶入口，未发布此候选。</div>
<h2>已经实现的路线</h2>
<div class="flow">冻结 NN/KLT 当前框与位置/速度 → 只在 ROI 内处理原生 ON/OFF 桨叶支持 → 椭圆／共同结构与侧轴 → 四、三、部分二桨、轴线分工 → 当前可观测贡献 + 未知合力 d → 与旧预测形成修正候选 → 用已经到期的自身 NN 位置学习修正权重 → 输出未来 0.4s 位置</div>
<p>机身无头尾，五圆描述共同结构；三桨可几何补点但不补缺桨RPM；两桨或侧线不强求完整五圆。不可见桨力、重力、风与阻力统一归 d。切换当前观测集合时转移输入参考，避免把“少看见一个桨”误当真实推力突变。两力模型是动力学分账，并非假设合力为零。</p>
<pre>二维有效动力学：p¨ = k · u_observed + d
四／三／部分二桨：u_observed 来自当前合格可见频率平方及像面法线代理。
侧视：局部混合周期只作变化代理，不乘4、不声称是总推力或已知某电机。
预测增量：p_final(h) = p_old(h) + w · [p_native(h) − p_old(h)]，0≤w≤1。
w 从已到期0.4s自身NN位置学习；人工标注只用于之后评分。</pre>
<p>当前 k=200 是图像尺度默认值，不是推力系数。k与d未独立标定，且侧线不能唯一确定三维平面或法线正负。机体仅摆动、中心不平移时，沿用当前位移/速度资格保护，不把盘面摆动直接当成机体平移。</p>
<h2>固定起点的主要结果</h2><p>数值为0.4s终点欧氏误差均值（px），源时刻口径。各组同一NN状态、有效性、时刻和评分终点；不删大误差点。旧完整入口本身已有历史桨叶/力学分支，不能把它称作“纯方框匀速法”。</p>{main_table}
<img src="fixed_query_errors.png" alt="三个片段所有固定查询误差对比">
<p>仅四桨组这次没有有效增量输出：连续四槽当前周期支持过少，在线权重保持0。不能把未激活分支当作已完成四桨推力效果验证。清晰段有18/120次four结构、4次three、5次partial_two，仍有71次axis_only；全四槽当前周期仅2次。侧视无四槽，另一录像只有3次部分／混合输入。几何模式为insufficient时也可能存在当前合格可见锥线法向，故它不等同于“完全无当前方向”。</p>
<h2>好例与坏例都保留</h2><p>图为查询时刻的原生20ms ON/OFF事件；黄色椭圆是当前候选，不能将全部候选当作独立真桨。星号表示未来人工参考终点，叠在当前事件上仅供比较，不作为当时输入。两个例子用于解释，不替代上表全部起点评分。</p>
<img src="example_clear.png" alt="清晰段增益例"><img src="example_side.png" alt="侧视退化例">
<h2>卡点具体在哪里</h2><ol>
<li><b>实际周期数值的预测杠杆太小。</b>同样资格/未知合力学习下，打开与置零已知输入的候选预测差异：侧视86次活动输入均值0.203px、最大0.756px；清晰83次均值0.516px、最大2.235px；另一段3次均值0.032px。它解释不了几十像素的主误差。</li>
<li><b>尺度仍学不起来。</b>新可见力分支这三段的有效增益更新均为0。大多侧视是混合频率，不能无条件把它当可辨识的逐桨转速；少量完整观测又太短，未知项和默认增益难以分开。用更大的固定系数扩大作用不构成解决。</li>
<li><b>在线可靠性也会滞后。</b>侧视111.2/111.4s根据之前到期自身NN位置学到正权重，但人工终点误差变差；111.6s得到改善，之后权重回0。自身NN是带偏差的教师，不是独立真值；未来控制变化也不受该教师保证。</li>
<li><b>前端仍慢且资格断续。</b>本次新增特征平均清晰43.81ms、侧视46.21ms；单候选+门控约1–4ms。这里未测RAW+NN整线，不称30FPS；train2平均包含76个无原生窗口的NN暖启动，17.21ms不能当其真实特征速度。</li></ol>
<h2>下一步应做什么</h2><p>优先在连续、空间来源稳定的桨叶支持上，用冻结NN的速度增量和已知噪声学习“频率/法向变化 → 图像加速度”的有效响应，同时约束未知项变化速率并保留增益不确定度；来源切换只换参考，不重造力。仍先做同资格、已知值置零的对照，证明周期数值本身带来收益，再接正式实时入口。不能继续在这些6/17/7点上扫参数。新增独立录像及更密的未来终点参考之后，再评估泛化与是否值得训练轨迹NN。</p>
<h2>创新性如何表述</h2><p>允许定位NN与桨叶模型互补，并不意味着主思路放弃。可以研究“可见性条件化的结构/周期辅助预测及可观测—未知输入分账”；但本轮仅有开发片段小收益，且更严格消融未支持实际周期数值的贡献。应称研究原型/初步结果，不能写成推力估计成功或稳定优于现有方法。</p>
<p>已有 <a href="https://arxiv.org/abs/2603.01997">RPM-Modulated Kalman Filtering</a> 将转速及变化引入预测，方法评估用真值框/位置隔离检测影响；<a href="https://arxiv.org/abs/2610.31384">CoHelix</a> 研究共享投影下多桨旋转。一般“RPM+KF”或“共享四桨几何”不能单独声称首次。此处为借鉴后自写实现，并非作者方法复现。</p>
<details><summary>前两轮失败结果：保留，避免只展示最好的一轮</summary><p>第一轮换成新共同滤波骨干，桨叶小收益却落后旧系统：</p>{weak_table}<p>第二轮保留旧NN位置/速度，直接替换整个力分支仍退化：</p>{direct_table}<p>随后才采用上表的保留旧预测＋因果增量门控。所有轮次都是开发实验，不能因最终一组较好而当独立盲测。</p></details>
<h2>复现与证据</h2><p>旧入口物理预测已逐帧复原，最大偏差3.2e-9px；所有组位置/速度/有效性断言一致；各模式输入前缀重跑输出一致；未到期不能学习、反向证据不加权、观测集合变化无假力阶跃等检查通过。源文件启动前快照、结束哈希守卫保存。人工标签没有进入回放。</p>
<p>侧视固定19起点可评分17个，缺失保留；另一录像7个；清晰用6个精确稀疏人工终点向前0.4s设起点，不跨稀疏标签插值。这些都是既有开发参考，不是新的盲测或独立RPM/姿态真值。</p>
<p><a href="gated_scores.json">最终逐点评分</a> · <a href="preserved_scores.json">直接替换评分</a> · <a href="scores.json">共同骨干评分</a> · <a href="protocol.json">原生特征及源哈希</a> · <a href="gated_summary.json">因果回放与前缀检查</a> · <a href="evidence_summary.json">同资格控制</a> · <a href="../../event_drone_forecast/docs/NN_ROTOR166_PROTOCOL_20261105.md">协议</a></p>
<pre>在 D:\\drone forecast\\event_drone_forecast 内，用现有 .venv-event311：
python tools/check_nn_rotor166.py
原生提取：tools/run_nn_rotor166.py（输出目录需另取名；旧结果不可覆盖）
回放：tools/replay_preserved_nn166.py → tools/replay_gated_nn166.py
控制：tools/replay_evidence_control166.py
评分：tools/score_gated_nn166.py（仅推理完成后）</pre>
<small>未动两处yolov10、RAW、模型权重、已发布ZIP；未训练或上传；当前默认入口和纯桨叶入口保持。30px／持续有效30Hz整体目标未完成。</small></html>'''
    (OUT/'index.html').write_text(body,'utf-8')
    print('Report and three figures saved:',OUT/'index.html')
