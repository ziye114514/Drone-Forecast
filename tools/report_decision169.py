"""A bounded decision report; do not disguise geometry consistency as FDE."""
from pathlib import Path
import sys,json,hashlib,copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from report_nn_rotor166 import native_window
OUT=ROOT.parent/'output/decision169_20261105'

if __name__=='__main__':
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    layout=json.loads((OUT/'layout_summary.json').read_text('utf-8'));cases=json.loads((OUT/'layout_cases.json').read_text('utf-8'));scores=json.loads((OUT/'decision_scores.json').read_text('utf-8'))
    valid=[r for r in cases if r['api_center_error_px'] is not None];median=float(np.median([r['api_center_error_px'] for r in valid]))
    case=min(valid,key=lambda r:abs(r['api_center_error_px']-median))
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained');ax=axes[0]
    e=native_window(ROOT.parent/'output/nn_rotor166_20261105/cache_clear',case['t'])
    on=np.zeros((720,1280),np.float32);off=on.copy()
    np.add.at(on,(e['y'][e['p']>0],e['x'][e['p']>0]),1);np.add.at(off,(e['y'][e['p']==0],e['x'][e['p']==0]),1)
    image=np.ones((720,1280,3),np.float32);a=np.clip(np.log1p(on)/np.log(32),0,1);b=np.clip(np.log1p(off)/np.log(32),0,1)
    image[:,:,0]-=.75*a;image[:,:,1]-=.75*np.maximum(a,b);image[:,:,2]-=.75*b;ax.imshow(image)
    p=np.asarray(case['visible_centers']);hidden=np.asarray(case['withheld_reference_hub']);center=np.asarray(case['full_reference_center'])
    ax.scatter(p[:,0],p[:,1],s=45,c='#2076b6',label='实际输入的3个当前中心')
    ax.scatter(*hidden,s=60,facecolors='none',edgecolors='gray',label='隐藏观测：仅评分参考')
    ax.scatter(*case['api_center'],s=90,c='#22945b',marker='+',label='五圆布局中心估计')
    ax.scatter(*case['independent_centroid'],s=70,c='#cf7b24',marker='x',label='直接取3点平均')
    for point in p:ax.plot([point[0],case['api_center'][0]],[point[1],case['api_center'][1]],c='#2076b6',lw=1)
    allp=np.vstack((p,hidden));ax.set_xlim(allp[:,0].min()-45,allp[:,0].max()+45);ax.set_ylim(allp[:,1].max()+50,allp[:,1].min()-50)
    ax.set_title(f"真实候选：{case['t']:.3f}s\n仅删除几何观测，原图没有模拟遮挡");ax.legend(fontsize=8);ax.set_xlabel('像素 x');ax.set_ylabel('像素 y')
    names=['可见3点平均','仅历史速度预测','近期布局＋共享平移','原仿射模型'];values=[layout['baseline_common']['mean'],layout['history_only_common']['mean'],layout['api_geometry_consistency']['mean'],layout['model_common']['mean']]
    bars=axes[1].bar(np.arange(4),values,color=['#cf7b24','#888888','#22945b','#457cb3']);axes[1].set_yscale('log');axes[1].set_xticks(np.arange(4),names,rotation=15);axes[1].set_ylabel('相对完整检测布局的误差 / px（对数轴）')
    axes[1].set_title('64个受控缺测案例；不是运动预测误差')
    for bar,value in zip(bars,values):axes[1].text(bar.get_x()+bar.get_width()/2,value*1.08,f'{value:.3f}',ha='center')
    fig.savefig(OUT/'geometry_value.png',dpi=150);plt.close(fig)
    table=''
    labels={'NN_clear':'NN辅助／清晰3终点','NN_side':'NN辅助／侧视9终点','NN_train2':'NN辅助／另一录像3终点','pure_side':'纯桨叶／侧视7终点','pure_train2':'纯桨叶／另一录像3终点'}
    for name,label in labels.items():
        c=scores[name];a=c['arms'];table+=f"<tr><th>{label}</th><td>{c['uncalibrated_baseline']['mean']:.2f}</td><td>{a['aligned']['error']['mean']:.2f}</td><td>{a['zero']['error']['mean']:.2f}</td><td>{a['past_delayed']['error']['mean']:.2f}</td></tr>"
    status=dict(scope='bounded batch finished; project prediction target incomplete',
        primary_value='visibility-conditioned rotor evidence and geometry consistency',
        force_response_status='experimental_not_validated',more_force_parameter_search_on_current_clips=False,
        actual_occlusion_accuracy_proven=False,true_COM_or_RPM_or_thrust_proven=False,global_novelty_proven=False,
        prediction_HEAD_test_passed=scores['decision']['all_suffixes_beat_both_controls'],
        supported_geometry_metric=layout['api_geometry_consistency'],best_simple_control=layout['shared_translation_common'])
    (OUT/'DECISION_STATUS.json').write_text(json.dumps(status,indent=2),'utf-8')
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>阶段169：收口与可交付价值</title>
<style>body{{font:17px/1.8 system-ui,"Microsoft YaHei";max-width:1120px;margin:32px auto;padding:0 20px;color:#26364b;background:#fafbfd}}h1,h2{{line-height:1.4}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{border:1px solid #d5deea;padding:8px;text-align:right}}th:first-child{{text-align:left}}img{{max-width:100%}}.note{{background:#eaf4ee;border-left:5px solid #379767;padding:18px}}.warn{{background:#fff2d1;padding:18px}}pre{{background:#e8f0f9;padding:18px;white-space:pre-wrap}}a{{color:#245e9c}}</style>
<h1>将项目收敛到有证据的模块，不再无止境调推力</h1>
<p>2026-11-05 · 阶段169 · 用户的五圆／可见数量分工保留；NN没有升级或训练。本批只交付独立观测模块、一个受控缺测基准和一次冻结后缀判断。</p>
<div class="note"><b>当前能交付的价值：</b>可见桨数变化时，使用历史完整布局和当前三点，减少直接可见点平均造成的中心偏移。真实视频候选的64个受控缺测案例中，布局中心一致性均值21.258→0.325px；只有历史速度预测为0.837px。共享平移已与原复杂仿射0.348px相当，所以优先简单模型，不为复杂度继续堆分支。</div>
<div class="warn"><b>限定范围：</b>0.325px不是0.4s运动预测误差，不是真实COM误差，也不是实际遮挡下的检测精度。参考是同一帧完整检测布局，测试只删除几何观测，候选可能有误且部分弧拟合依赖先验。上游候选来自原生事件及既有NN辅助ROI，恢复模块没有使用当前NN机体位置。不能据此宣称世界首次或整套创新已成功。</div>
<h2>你的结构思路已经落实成什么</h2><p><code>rotor_evidence.py</code>提供两个可单独调用的类：</p><ol>
<li><b>RotorEvidence：</b>把当前实测区域、周期合格数、完整模板状态、时间支持、隐藏补点分开。四路当前周期可求无序平方和，不要求持久电机ID或完整模板锁定；三/二路只保留实测贡献，不补隐藏转速。</li>
<li><b>VisibilityLayout：</b>四点提供布局中心候选；三点在近期模板下做共享平移，形变确实需要时才复用旧仿射形状检查；两点/侧线不虚构完整中心，明确给方向、先验时效或拒绝输出。</li></ol>
<p>完整模型锁定与当前桨叶证据不再是一道门。完整帧能提供模板，但不会被解释为“一帧之后永远成功”；参考超过.4s会失去可用资格，推断点始终标为未测量且没有虚构频率。当前四区域只是候选假设，不保证就是四个真实电机。</p>
<h2>几何试验如何避免偷看当前隐藏点</h2><p>从既有清晰段选出20个满足固定资格的四区域帧。16帧有两份近期历史参考，分别删掉四个中心中的一个，共64例。模型只看到过去完整布局和当前3点；当前完整布局中心及被隐藏点只用于评分。没有把当前完整中心或当前NN机体位置交给恢复模型。</p>
<img src="geometry_value.png" alt="真实候选受控缺测与布局中心一致性比较">
<table><tr><th>方法</th><th>均值px</th><th>中位px</th><th>P95 px</th></tr>
<tr><th>3点直接平均</th><td>21.258</td><td>20.776</td><td>24.816</td></tr>
<tr><th>只有过去布局的速度预测</th><td>0.837</td><td>0.844</td><td>1.147</td></tr>
<tr><th>共享平移／独立模块默认</th><td>0.325</td><td>0.289</td><td>0.727</td></tr>
<tr><th>项目原有仿射</th><td>0.348</td><td>0.315</td><td>0.911</td></tr></table>
<p>64例均输出；4个没有足够历史的完整帧未进入缺测测试，原因保留。原仿射补出的第四中心相对被隐藏检测点误差均值1.010px。这支撑“结构约束可减少组成变化偏移”，并未证明仿射优于已有简单刚体跟踪；本模块复用项目旧数学，不包装为新发明。</p>
<h2>推力分支的一次性判断</h2><p>预先固定标定/后缀时段，最多拟合一个非负响应系数与二维未知偏置。对齐输入、数值置零、只取过去.4s的错时序输入各自用同样标定段；参数冻结后输出后缀预测，拟合进程禁止打开后缀目标文件。人工标定标签被明确使用，因此不是无需真值的部署结果，也不是所有模型的数学性能上限。</p>
<table><tr><th>后缀：0.4s均值px</th><th>未标定基线</th><th>对齐输入</th><th>置零控制</th><th>过去错时序</th></tr>{table}</table>
<p><b>预定通过条件未满足。</b>对齐输入没有稳定胜过置零/错时序；纯桨叶侧视响应系数被非负拟合压成0，NN清晰虽好于两个标定控制却仍差于未标定基线。不能拿训练段好看或单段小胜证明推力预测有效。该版本结束在这些旧片段上的推力参数寻找，保留代码和负结果，主张标为experimental。</p>
<h2>现在适合怎样描述项目</h2><pre>面向事件相机四旋翼的可见性条件化桨叶观测与布局恢复：
区分当前测量、历史补点和侧视不可观测量，提供原生前端后的JSON/API；
在真实视频候选的受控单桨缺测一致性测试中减少组成变化造成的中心偏移。
推力辅助运动预测尚未获得稳定增益，不声称真实推力、质量或风已辨识。</pre>
<p>目前是可复用的观测后端和研究原型。下一步只应针对实际像素遮挡/真实可见桨数变化收集独立标注来验证检测与中心稳定性；不是继续把0.325px换名为轨迹精度。具体新颖性仍需已有方法对照，不能仅凭“五圆”或刚体模板本身声称首次。</p>
<h2>可直接运行</h2><pre>在 D:\\drone forecast\\event_drone_forecast 中：
python tools/rotor_model169.py --input "原生入口输出/predictions.jsonl" --output "新的模块输出目录"

from rotor_evidence import RotorEvidence, VisibilityLayout
evidence, layout = RotorEvidence(), VisibilityLayout()
packet = evidence.step(t, body, structure, current_arcs, current_period_tests)
state = layout.step(packet)  # current_center允许为空；不把prior当当前测量</pre>
<p>已有<a href="exports/clear/rotor_model.jsonl">清晰候选导出</a>（上游NN辅助ROI）和<a href="exports/pure_side/rotor_model.jsonl">纯桨叶侧视导出</a>（不用NN）。导出器不加载NN；输入是已提取候选，不是只靠一个新文件就完成原始事件检测。</p>
<p><a href="layout_summary.json">几何完整统计</a> · <a href="layout_cases.json">64例逐例数据</a> · <a href="decision_scores.json">冻结后缀逐点结果</a> · <a href="fit_completion.json">参数冻结与源码守卫</a> · <a href="DECISION_STATUS.json">收口决策</a> · <a href="../../event_drone_forecast/docs/ROTOR_MODEL_V1.md">模块说明与可引用表述</a> · <a href="../../event_drone_forecast/docs/DECISION169_PROTOCOL_20261105.md">预定协议</a></p>
<p>机制检查通过：无模板四频率和、置换不变、三点不补RPM、缺周期拒绝、侧视不造完整中心、共享平移与形变回退。预测头前后源码守卫通过，后缀坐标未进拟合；所有数据仍为旧开发片段。原RAW/权重/保护目录/发布ZIP不变，本批没有新增FPS或端到端预测成功声明。</p></html>'''
    (OUT/'index.html').write_text(page,'utf-8')
    names=['rotor_evidence.py','calibrated_force_head.py','tools/rotor_model169.py','tools/layout_value169.py','tools/run_decision169.py','tools/check_rotor_evidence169.py',
           'tools/export_evidence169.py','tools/prepare_decision169.py','tools/score_decision169.py','tools/report_decision169.py','tools/finalize_decision169.py',
           'docs/ROTOR_MODEL_V1.md','docs/HANDOFF_20261105_DECISION169.md','docs/DECISION169_PROTOCOL_20261105.md']
    (OUT/'final_source_hashes.json').write_text(json.dumps({n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},indent=2),'utf-8')
    print('Saved bounded decision report:',OUT/'index.html')
