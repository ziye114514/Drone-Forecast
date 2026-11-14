"""Scientific figures, verified RAW previews and honest112-118 delivery."""
from pathlib import Path
import html,json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage118'),str(ROOT/'research/stage117'),str(ROOT/'research/stage110')]
from study118 import OUT,verify,CASES,rows,save,stable
from study117 import OUT as MODEL
BASE=ROOT.parent/'output'

def table(records,fields):
    return '<table><tr>'+''.join('<th>'+html.escape(label)+'</th>' for key,label in fields)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(key,'')))+'</td>' for key,label in fields)+'</tr>' for r in records)+'</table>'

def model_summary():
    results=[]
    for case in ['clear6','continuous6','side6','negative5']:
        rr=rows(MODEL/case/'records.jsonl')
        for mode in ['free_structure','fixed_structure','fixed_guarded']:
            dd=[r['outputs'][mode]['center'] for r in rr]
            results.append(dict(case=case,mode=mode,frames=len(rr),current=sum(d['current'] for d in dd),
                structure_first=next((r['t'] for r in rr if r['outputs'][mode]['center'].get('structure_learned_now')),None),
                fixed_solves=sum(d.get('fixed_structure_used',False) for d in dd),
                initializations=sum(d['initialized_now'] for d in dd),shape=next((d['structure'] for d in dd if d.get('structure_ready')),None)))
    save(OUT/'model_summary118.json',results);return results

def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rr=rows(MODEL/'clear6/records.jsonl');ts=np.array([r['t'] for r in rr]);fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
    for mode,color in [('free_structure','#d77c22'),('fixed_structure','#158c7e')]:
        a=[];rad=[];v=[]
        for r in rr:
            q=r['outputs'][mode];d=q['center'];m=d['measurement']
            aspect=m.get('frame_aspect',np.nan);radius=m.get('radius',np.nan)
            if mode=='fixed_structure':
                shape=d.get('structure');radius,aspect=shape if shape is not None else [np.nan,np.nan]
            elif aspect<1:radius,aspect=radius/aspect,1/aspect
            a.append(aspect);rad.append(radius)
            c=q['forecast']['coefficients'].get('center_bias_fixed107');v.append(np.linalg.norm(c['velocity']) if c else np.nan)
        axes[0].plot(ts,a,color=color,label=mode);axes[1].plot(ts,rad,color=color,label=mode);axes[2].plot(ts,v,color=color,label=mode)
    axes[0].set_ylabel('Canonical template aspect');axes[1].set_ylabel('Canonical template radius');axes[2].set_ylabel('Estimated speed (px/s)');axes[2].set_xlabel('Source time (s)')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Learned dimensionless structure vs free fitting; no real COM truth');fig.tight_layout();fig.savefig(OUT/'structure_and_speed118.png',dpi=150);plt.close(fig)
    # Same target coverage, all candidate failures visible.
    ss=json.loads((MODEL/'scores110.json').read_text('utf-8'))['table'];methods=['free_structure/center_bias_fixed107','fixed_structure/center_bias_fixed107','original_fusion']
    fig,ax=plt.subplots(figsize=(9,4.5));xx=np.arange(2)
    for i,name in enumerate(methods):
        values=[next(r['mean_px'] for r in ss if r['case']==case and r['h']==.4 and r['method']==name) for case in ['clear6','side6']]
        bars=ax.bar(xx+(i-1)*.24,values,.24,label=name.split('/')[0])
        ax.bar_label(bars,fmt='%.2f',fontsize=8)
    ax.set_xticks(xx,['clear6 (7/8 sparse box endpoints)','side6 (7/8 sparse box endpoints)']);ax.set_ylabel('Mean0.4s endpoint error (px)');ax.legend(fontsize=8);ax.set_ylim(0,98);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(OUT/'forecast_comparison118.png',dpi=150);plt.close(fig)

def videos():
    checks=[]
    for case in ['clear6','continuous6','negative5']:
        folder=OUT/'raw'/case;rr=rows(folder/'observations_forecasts.jsonl');cap=cv2.VideoCapture(str(folder/'forecasts.mp4'))
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));assert w>=1280 and h>=720
        dest=folder/'learned_structure.mp4';writer=cv2.VideoWriter(str(dest),cv2.VideoWriter_fourcc(*'mp4v'),50.,(w+440,h));assert writer.isOpened()
        midpoint=None
        for i,r in enumerate(rr):
            ok,im=cap.read();assert ok;display=np.full((h,w+440,3),248,np.uint8);display[:,:w]=im
            q=r['frontends']['delayed_feedback'];worker=q['learned_structure118']
            for z in worker['geometry'].get('rotors',[]):cv2.ellipse(display,((z['cx'],z['cy']),(z['major'],z['minor']),z['angle']),(170,160,30),1)
            for name,color in [('model_free20_kf118',(25,135,225)),('model_fixed_kf118',(70,175,25))]:
                c=q['coefficients'].get(name)
                if q['valid'] and c:
                    points=np.asarray([np.asarray(c['position'])+dt*np.asarray(c['velocity']) for dt in [0,.1,.2,.3,.4]])
                    if np.isfinite(points).all():cv2.polylines(display,[np.rint(points).astype(np.int32)],False,color,2)
            lines=[f'RAW source t={r["t"]:.2f}s','50fps source replay, not realtime','Orange: free current20 model','Green: fixed learned structure','Cyan ellipses: current measurements','No hidden motor points inserted','']
            for mode in ['free_structure','fixed_structure']:
                d=worker['outputs'][mode]['center'];lines.extend([mode,f'current pose: {d["current"]}',f'pose initialized: {d["ready"]}',d['measurement']['reason']])
                if d.get('structure') is not None:lines.append('learned r/aspect: '+', '.join(f'{x:.3f}' for x in d['structure']))
            for j,line in enumerate(lines):cv2.putText(display,line,(w+12,28+j*26),0,.48,(25,45,55),1)
            body=r['body_observation']
            if body.get('detected'):
                box=np.asarray(body['bbox']);lo=np.maximum(np.floor(box[:2]-12).astype(int),0);hi=np.minimum(np.ceil(box[2:]+12).astype(int),[1280,720])
                crop=display[lo[1]:hi[1],lo[0]:hi[0]];top=465
                if crop.size and h>top+35:
                    scale=min(416/crop.shape[1],(h-top-30)/crop.shape[0]);small=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
                    cv2.putText(display,'Current ROI zoom (same event pixels)',(w+12,top-8),0,.45,(25,45,55),1)
                    display[top:top+small.shape[0],w+12:w+12+small.shape[1]]=small
            writer.write(display)
            if i==len(rr)//2:midpoint=display.copy()
        ok,_=cap.read();assert not ok;cap.release();writer.release()
        check=cv2.VideoCapture(str(dest));decoded=0
        while True:
            ok,_=check.read()
            if not ok:break
            decoded+=1
        check.release();assert decoded==len(rr);cv2.imwrite(str(OUT/(case+'_midpoint118.png')),midpoint)
        checks.append(dict(case=case,frames=decoded,original_sensor_region_preserved=True,source_fps=50.,not_processing_fps=True,path=str(dest)))
    save(OUT/'video_checks118.json',checks)

def report():
    verify();summary=model_summary();figures();videos()
    benchmark=json.loads((BASE/'stage116_analytic_solver_benchmark_20261022/benchmark116.json').read_text('utf-8'))['summary']
    scores=json.loads((MODEL/'scores110.json').read_text('utf-8'))['table']
    selected=[r for r in scores if r['h']==.4 and (r['method'] in ['free_structure/center_bias_fixed107','fixed_structure/center_bias_fixed107','original_fusion'])]
    rawchecks={case:json.loads((OUT/'raw'/case/'checks118.json').read_text('utf-8')) for case in CASES}
    native=sum(r['frames'] for r in rawchecks.values());preview=sum(r['decoded_frames'] for r in rawchecks.values())
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>112—118：真实轮廓、结构记忆与RAW预测</title><style>body{max-width:1400px;margin:25px auto;padding:0 20px;font:17px/1.8 system-ui;color:#243b4a}img,video{max-width:100%;height:auto}table{border-collapse:collapse;font-size:14px}td,th{padding:6px;border:1px solid #cbd5df}th{background:#edf4f7}a{color:#087c87}.note{background:#fff1d6;padding:12px}code{background:#edf2f5}</style><h1>112—118：把“模型建好后只估姿态”接入真实RAW</h1>
<p>本轮新增完整RAW实验入口 <code>event_drone_forecast/run_learned_structure_forecast.py</code>。原107的64路保留，新增当前20ms自由结构、已学固定结构、固定结构加条件不确定性三个对照。使用同一个RAW及原检测权重，不向运行时传入框/轮廓缓存；原默认54/API93未替换。</p>
<p class="note">结构建模已更接近用户设想，当前精度收益仍小。清晰0.4秒相同7/8个稀疏框端点：自由结构16.843px，固定结构16.780px，原完整融合13.708px。侧视/连续段没有首次成功建模，仍不能声称完成连续侧视恢复或稳定实时预测。</p>
<h2>固定的结构与新鲜的姿态分开</h2><p>旧104每帧自由求H8、桨盘半径、机架长宽比，首次五帧只产生一个事后尺寸筛选量。本轮在原五帧资格成立后，学习无量纲半径和机架比例，规范90度重排歧义并固定；以后只求当前投影H8。保留结构记忆不等于保留当前姿态，缺桨仍不伪造观测。身份超过0.5秒无当前机体支持就重置；旧姿态有效期、五帧重新建模和位置滤波规则保持。</p>
<p>每段独立启动。没有把22秒学到的模型或位姿跨八十多秒注入110秒片段，没有提前提供真实尺寸、质量、推力系数或风速。</p>'''+table(summary,[('case','片段'),('mode','方法'),('frames','时刻'),('structure_first','结构首次学到/s'),('fixed_solves','固定结构求解时刻'),('current','当前几何更新'),('initializations','姿态初始化')])+'''<img src="structure_and_speed118.png">
<h2>预测效果和实际RAW验证</h2>'''+table(selected,[('case','片段'),('method','方法'),('n','有效端点'),('targets','全部端点'),('mean_px','0.4秒误差/px')])+'''<img src="forecast_comparison118.png"><p>目标是先前稀疏标注的未来框中心，不是实际机体重心。数据属于已检查开发片段，不能当盲测或前沿基准。固定结构相对匹配对照仅改善约0.063px，不能据此证明泛化或独立创新。</p>
<p>实际RAW'''+str(native)+'''个20ms父时刻（清晰250、连续150、无机30、两个独立实际截断各60），原64路系数与预测逐项精确107，新3路与117的模型/学习/滤波状态严格一致（仅排除耗时）。当前事件时间上界检查、body-onlyKF对照、实际前缀、负例和'''+str(preview)+'''个原预览帧全解码通过。新增几何运行在隔离进程，避免额外椭圆拟合改变旧OpenCV随机状态；隔离进程禁止读取output缓存/标注文件。</p>
<h3>清晰片段：实际RAW与两种模型预测</h3><video controls preload="metadata" src="raw/clear6/learned_structure.mp4"></video><p>橙为匹配自由结构，绿为固定结构；曲线接近，不能靠视频夸大局部误差差异。</p>
<h3>连续片段：仍未学到结构</h3><video controls preload="metadata" src="raw/continuous6/learned_structure.mp4"></video><p>视频按源50fps播放，当前整链路计算仍超20ms预算。无首次锁定时新分支依赖机体位置历史，不声称获得了桨盘模型的侧视收益。</p>
<h2>真实轮廓与弱边界：本轮保留的负结果</h2><p>112实际重读RAW留存当前轮廓，并严格匹配110所有选中观测。未匹配的20个候选均为既有arc推断模式，未作完整轮廓输入。113直接用阈值轮廓/曲线筛选点替代拟合椭圆的完整周长，加入标准稳健损失与完整参数局部协方差；清晰0.4秒退步至约21.64px，连续仍0初始化。</p>
<p>114利用高密度区域作种子，以半阈值双极计数及标准OpenCV分水岭恢复弱边界，C单帧能在80次预算内合格；115整段测试清晰却退至bodyKF约22.86px、当前几何仅14次，连续仍0初始化。单帧改善没有泛化，不纳入新RAW默认分支。</p><img src="../stage114_weak_boundary_pilot_20261022/weak_boundary_contact114.png"><p>这里的轮廓是事件密度区域边界，可能受机体边缘、遮挡切边和时间累积影响，尚无物理桨盘边缘真值。当前证据不足以把失败全部归因于某一种污染。</p>
<h2>数值与控制检查</h2><p>113解析Sampson导数通过中央差分检查，适配后的原数值导数目标严格等于原58。116独立交错模块基准'''+str(benchmark['pairs'])+'''对，跳过'''+str(benchmark['skipped'])+'''个预设非四完整椭圆时刻：原有限差分中位'''+f"{benchmark['fd_median_ms']:.2f}"+'''ms，解析导数含局部协方差'''+f"{benchmark['analytic_median_ms']:.2f}"+'''ms，配对比中位'''+f"{benchmark['median_pair_speed_ratio']:.3f}"+'''倍。最大中心差'''+f"{benchmark['maximum_center_difference_px']:.8f}"+'''px、终止状态无不一致；不是全链路速度提升或逐位相等。</p>
<p>117另外运行静止中心姿态摆动、真实平移、突然变向、缺桨后恢复的已知投影圆轮廓序列。固定结构没有把真实运动压成静止，缺桨期间相对位置RMSE仍约10.23px、0.4秒误差约19.36px。结构记忆不能代替缺失的当前观测。合成是带噪轮廓层，不是事件传感器仿真。</p>
<p>局部协方差仍未校准：113含遮挡切边的已知真值控制中，约0.86px中心误差对应约0.23px局部sigma；它没有覆盖模型偏差。117的位姿协方差还以已学结构固定为条件，没有包含结构学习误差，不能宣称95%置信覆盖。</p>
<h2>相关方法与下一步</h2><p>查阅<a href="https://github.com/zygmuntszpak/guaranteed-ellipse-fitting-with-a-confidence-region-and-an-uncertainty-measure">Szpak等的椭圆不确定性作者代码</a>、<a href="https://link.springer.com/article/10.1186/s41074-016-0007-y">椭圆支持弧段筛选论文</a>及<a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html">SciPy求解文档</a>。本轮未运行/移植作者MATLAB系统，实际复用现有项目几何辅助、原58/104/107数学和已安装OpenCV/SciPy，不称这些基础方法原创。</p>
<p>下一步优先：在同一连续录像中验证结构已学到后出现退化、再恢复；为缺失桨盘保留“未观测”状态，利用已有结构关联剩余真实支持。围绕真实事件中的机体/桨盘污染、结构不确定性与连续观测建立方法，避免继续仅在单帧阈值上调参。独立RPM收益和实时调度仍需后续证据。</p>
<p><a href="protocol118.json">118冻结完整RAW方案</a> · <a href="../stage113_joint_observed_arcs_20261022/protocol113.json">113拟合对照</a> · <a href="../stage115_RAW_weak_support_20261022/protocol115.json">115弱边界对照</a> · <a href="../stage117_persistent_structure_20261022/protocol117.json">117结构与合成</a> · <a href="model_summary118.json">结构状态汇总</a> · <a href="video_checks118.json">叠加视频全解码检查</a></p></html>'''
    path=OUT/'第一百一十二至一百一十八阶段观测与固定结构预测报告.html';path.write_text(page,encoding='utf-8')
    save(OUT/'delivery_checks118.json',dict(report=str(path),actual_RAW_parent=native,raw_checks=rawchecks,original_preview_decoded=preview,new_preview_decoded=preview,
        original64_preserved=True,default_promoted=False,first_side_model_initialized=False,independent_RPM_gain_proven=False,real_time=False))
    print(str(path),flush=True)

if __name__=='__main__':report()
