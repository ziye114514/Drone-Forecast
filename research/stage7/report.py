"""Aggregate frozen experiment, render actual future views, validate artifacts."""
from pathlib import Path
import collections,hashlib,html,json,sys,zipfile,time
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study import OUT,ROOT,FRONTS,INPUTS,METHODS,HORIZONS,load,save,verify,read_labels,states,summarize
from models import predict,MotionBank

def aggregate():
    protocol=verify();rows=[];runs=[]
    for i in range(11):
        folder=OUT/'runs'/f'sequence{i}';runs.append(load(folder/'completion.json'))
        rows.extend(json.loads(x) for x in (folder/'endpoint_details.jsonl').read_text().splitlines())
    groups={'all_local':list(range(11)),'previously_inspected_train':[1,2,3],
        'additional_train_transfer':[0,4,5,6,7,10],'official_test_subset':[8,9]}
    summary=[];paired=[]
    for split,ids in groups.items():
        for front in FRONTS:
            for method in METHODS:
                for h in HORIZONS:
                    for source in INPUTS:
                        selected=[r for r in rows if r['sequence'] in ids and r['front']==front and r['input']==source and r['method']==method and r['horizon_s']==h]
                        per=[summarize([r for r in selected if r['sequence']==i]) for i in ids]
                        means=[m['mean_px'] for m in per if m['mean_px'] is not None]
                        summary.append(dict(split=split,front=front,method=method,input=source,horizon_s=h,
                            **summarize(selected),sequence_macro_mean_px=float(np.mean(means)) if means else None))
                    g={}
                    for r in rows:
                        if r['sequence'] in ids and r['front']==front and r['method']==method and r['horizon_s']==h:
                            g.setdefault((r['sequence'],r['target_t']),{})[r['input']]=r
                    common=[x for x in g.values() if all(x[s]['error_px'] is not None for s in ['matched_auto','matched_oracle'])]
                    auto=summarize([x['matched_auto'] for x in common]);oracle=summarize([x['matched_oracle'] for x in common])
                    paired.append(dict(split=split,front=front,method=method,horizon_s=h,n=len(common),auto=auto,oracle=oracle,
                        oracle_error_reduction=1-oracle['mean_px']/auto['mean_px'] if auto['mean_px'] else None,
                        caution='Intervention effect on matched support, not additive decomposition or fraction of physical error uniquely explained.'))
    save(OUT/'aggregate.json',dict(groups=groups,metrics=summary,matched_interventions=paired,runs=runs))
    return rows,summary,paired,runs

def plots(rows,summary):
    ids=list(range(11));fig,axes=plt.subplots(2,1,figsize=(11,8),constrained_layout=True)
    for ax,front in zip(axes,FRONTS):
        aa=[];oo=[]
        for i in ids:
            a=[r for r in rows if r['sequence']==i and r['front']==front and r['method']=='cv' and r['horizon_s']==.4]
            by={}
            for r in a:by.setdefault(r['target_t'],{})[r['input']]=r
            co=[g for g in by.values() if g['matched_auto']['error_px'] is not None and g['matched_oracle']['error_px'] is not None]
            aa.append(np.mean([g['matched_auto']['error_px'] for g in co]) if co else np.nan)
            oo.append(np.mean([g['matched_oracle']['error_px'] for g in co]) if co else np.nan)
        x=np.arange(11);ax.bar(x-.18,aa,.36,label='Automatic center, matched cadence',color='#e68b36')
        ax.bar(x+.18,oo,.36,label='Past label center, same missing mask',color='#4a92bf')
        ax.set_xticks(x,[str(i)+('*' if i in [8,9] else '') for i in ids]);ax.set_ylabel('Mean endpoint error (px)')
        ax.set_title(f'{front}: same OpenCV CV filter, 0.4 s nominal horizon');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'localization_intervention.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
    for ax,split in zip(axes,['additional_train_transfer','official_test_subset']):
        for source,color in [('dense_auto','#d17832'),('matched_auto','#bd4d6a'),('matched_oracle','#3b91b7'),('oracle_all','#53883f')]:
            rr=[next(r for r in summary if r['split']==split and r['front']=='activity' and r['method']=='cv' and r['input']==source and r['horizon_s']==h) for h in HORIZONS]
            ax.plot(HORIZONS,[r['mean_px'] for r in rr],'o-',color=color,label=source)
        ax.set_title(split);ax.set_xlabel('Nominal horizon (s)');ax.set_ylabel('Mean error on available endpoints (px)');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(OUT/'horizon_comparison.png',dpi=150);plt.close(fig)

def video(sequence):
    folder=OUT/'runs'/f'sequence{sequence}'
    records,times,labels,dense,paired,oracle=states(sequence)
    data=np.load(folder/'event_frames.npz');images=data['images'];ft=data['times'];lt=np.array([r['t'] for r in labels])
    meta=load(folder/'completion.json');sx=640/meta['width'];sy=360/meta['height']
    path=folder/'prediction_future.mp4';writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(1280,440));assert writer.isOpened()
    def point(im,p,color,label):
        xy=np.round(np.array(p)*[sx,sy]).astype(int)
        if 0<=xy[0]<640 and 0<=xy[1]<360:
            cv2.drawMarker(im,tuple(xy),color,cv2.MARKER_CROSS,18,2)
    count=0;preview=None
    for fi,t in enumerate(ft):
        if t>19.6:break
        future=t+.4;fj=int(np.argmin(abs(ft-future)));assert abs(ft[fj]-future)<.011
        left=cv2.cvtColor(images[fi],cv2.COLOR_GRAY2BGR);right=cv2.cvtColor(images[fj],cv2.COLOR_GRAY2BGR)
        k=int(np.searchsorted(times,t+1e-9,side='right')-1)
        if k<0:continue
        for front,method,color,label in [('rotor','cv',(20,145,255),'rotor CV'),('activity','cv',(255,210,50),'activity CV'),('activity','ca_history',(230,80,230),'activity CA')]:
            bank=dense[front][k]
            if bank is not None:
                point(right,predict(bank[method],future-times[k]),color,label)
                o=records[k][front]
                if o.get('detected'):point(left,o['center'],color,front)
        j=int(np.searchsorted(lt,t+1e-9,side='right')-1)
        if j>=0 and t-lt[j]<.075 and oracle[j] is not None:point(right,predict(oracle[j]['cv'],future-lt[j]),(50,220,255),'GT history CV')
        g=int(np.argmin(abs(lt-future))) if len(lt) else -1
        if g>=0 and abs(lt[g]-future)<.001:
            point(right,labels[g]['center'],(70,255,70),'future label')
            box=np.array(labels[g]['bbox'])*[sx,sy,sx,sy];b=box.astype(int);cv2.rectangle(right,tuple(b[:2]),tuple(b[2:]),(70,255,70),1)
        cv2.putText(left,f'Seq {sequence}: current events t={t:.2f}s',(12,26),0,.7,(240,240,240),2)
        cv2.putText(right,f'ACTUAL FUTURE events t={ft[fj]:.2f}s',(12,26),0,.7,(240,240,240),2)
        # Inset is centered on the past-only activity CV prediction, never future labels/events.
        if dense['activity'][k] is not None:
            c=np.round(predict(dense['activity'][k]['cv'],future-times[k])*[sx,sy]).astype(int)
            if 0<=c[0]<640 and 0<=c[1]<360:
                x0=int(np.clip(c[0]-60,0,520));y0=int(np.clip(c[1]-60,0,240))
                zoom=cv2.resize(right[y0:y0+120,x0:x0+120],(190,190),interpolation=cv2.INTER_NEAREST)
                right[155:345,435:625]=zoom;cv2.rectangle(right,(434,154),(626,346),(160,160,160),1)
                cv2.putText(right,'Past-predicted ROI zoom',(434,148),0,.38,(220,220,220),1)
        canvas=np.zeros((440,1280,3),np.uint8);canvas[:360]=np.hstack((left,right))
        for x,label,color in [(12,'Rotor CV',(20,145,255)),(180,'Activity CV',(255,210,50)),(370,'Activity CA',(230,80,230)),(570,'GT-history CV: diagnostic',(50,220,255)),(970,'Future label',(70,255,70))]:
            cv2.putText(canvas,label,(x,384),0,.55,color,1)
        cv2.putText(canvas,'Yellow uses past annotations, NOT RAW-only. Full view retained; inset uses past prediction only.',(12,409),0,.57,(220,220,220),1)
        cv2.putText(canvas,'Out-of-view predictions are not drawn. Future images and labels never enter the predictor.',(12,433),0,.55,(220,220,220),1)
        writer.write(canvas);count+=1
        if abs(t-17.5)<.011:preview=canvas.copy()
    writer.release();cap=cv2.VideoCapture(str(path));decoded=0
    while True:
        ok,_=cap.read()
        if not ok:break
        decoded+=1
    cap.release();assert decoded==count and count>0
    if preview is not None:(folder/'preview.png').write_bytes(cv2.imencode('.png',preview)[1].tobytes())
    save(folder/'video_manifest.json',dict(frames=count,decoded_all_frames=decoded,fps=10,width=1280,height=440,
        horizon_s=.4,rendered_future_observation_only=True,oracle_drawn_separately=True,
        inset='Past-only activity CV prediction determines fixed120px crop; future labels/events never select the crop.'))

def f(x,d=2):return '—' if x is None else f'{x:.{d}f}'
def pct(x):return f'{100*x:.1f}%'

def report(rows,summary,paired,runs):
    def metric(split,front,source,method='cv',h=.4):return next(r for r in summary if r['split']==split and r['front']==front and r['input']==source and r['method']==method and r['horizon_s']==h)
    tables=[]
    for split,title in [('additional_train_transfer','新增训练序列迁移检查：0、4、5、6、7、10'),('official_test_subset','官方测试子集：8、9，各前20秒')]:
        trs=[]
        for front in FRONTS:
            for source in INPUTS:
                for method in METHODS:
                    r=metric(split,front,source,method)
                    trs.append(f"<tr><td>{front}</td><td>{source}</td><td>{method}</td><td>{r['available']}/{r['targets']}</td><td>{f(r['mean_px'])}</td><td>{f(r['median_px'])}</td><td>{f(r['p95_px'])}</td><td>{pct(r['within_quarter_diagonal'])}</td></tr>")
        tables.append(f'<h2>{title}</h2><div class="scroll"><table><tr><th>前端</th><th>位置输入</th><th>预测器</th><th>输出/端点</th><th>均值 px</th><th>中位数</th><th>P95</th><th>输出且误差≤目标对角线25%</th></tr>'+''.join(trs)+'</table></div>')
    interventions=[]
    for split in ['additional_train_transfer','official_test_subset']:
        for front in FRONTS:
            for method in METHODS:
                r=next(r for r in paired if r['split']==split and r['front']==front and r['method']==method and r['horizon_s']==.4)
                interventions.append(f"<tr><td>{split}</td><td>{front}</td><td>{method}</td><td>{r['n']}</td><td>{f(r['auto']['mean_px'])}</td><td>{f(r['oracle']['mean_px'])}</td><td>{'—' if r['oracle_error_reduction'] is None else pct(r['oracle_error_reduction'])}</td></tr>")
    runrows=[]
    for r in runs:
        timing=r['timings_ms'];rotor_p95=timing['rotor']['p95']
        if r.get('rotor_cached'):rotor_p95=load(Path(r['rotor_cache_source'])/'completion.json')['timings_ms']['rotor']['p95']
        runrows.append(f"<tr><td>{r['sequence']}</td><td>{r['role']}</td><td>{r['detected_frames']['rotor']}/{r['frames']}</td><td>{r['detected_frames']['activity']}/{r['frames']}</td><td>{r['four_rotor_frames']}</td><td>{r['frequency_quality_frames']}</td><td>{f(r['wall_s'])}{'*' if r.get('rotor_cached') else ''}</td><td>{f(rotor_p95)}</td><td>{f(timing['activity']['p95'])}</td></tr>")
    sourceaudit='''<ul><li><a href="https://arxiv.org/html/2603.01997v1">RPMKF</a> 已有转速调噪声与历史标注框预测；本阶段没有声称复现作者完整实现。</li><li><a href="https://arxiv.org/html/2609.00839v1">Residual Kalman Dynamics</a> 已有强卡尔曼基线加加速度残差、运动先验去相关；本次检索未确认作者公开代码，不把自行实现写成官方复现。</li><li><a href="https://github.com/miccunifi/FRED">FRED 官方</a> 固定提交2bf89c5，实际使用其 canonical 划分；新版 loader 的六字段解析不兼容本地四字段旧标注，因此复用项目已有解析器。</li><li><a href="https://github.com/takara-ai/DroneStalker">DroneStalker</a> 固定提交93e846e，核查目录与源文件，确有预测及检测权重；README宣称检测器在整个FRED训练，训练来源未被独立确认，因此没有拿它在8、9号上充当无污染基线。它适合作为之后自有视频的迁移候选，本轮未执行其权重。</li><li><a href="https://docs.opencv.org/4.x/dd/d6a/classcv_1_1KalmanFilter.html">OpenCV KalmanFilter</a> 本次实际调用已安装5.0.0的predict/correct；RAW解码、桨叶检测和活动跟踪也直接复用现有模块。新增的是实验协议、数据适配和误差定位，不是重新编写卡尔曼库。</li></ul>'''
    cards=[]
    for front in FRONTS:
        r=next(x for x in paired if x['split']=='official_test_subset' and x['front']==front and x['method']=='cv' and x['horizon_s']==.4)
        cards.append(f"<div class='card'><b>{front}：匹配对照</b><p>0.4秒，{r['n']}个共同端点<br>自动位置 {f(r['auto']['mean_px'])} px → 标注位置 {f(r['oracle']['mean_px'])} px</p></div>")
    videos=''.join(f'<h3>序列{i}</h3><video controls preload="metadata" src="runs/sequence{i}/prediction_future.mp4"></video><p><a href="runs/sequence{i}/endpoint_details.jsonl">逐端点预测与误差</a> · <a href="runs/sequence{i}/evaluation.json">该序列完整评分</a></p>' for i in [5,8,9])
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第七阶段：FRED定位误差与预测能力对照</title><style>body{font-family:Microsoft YaHei,system-ui;margin:0;background:#f3f6fb;color:#203044;line-height:1.75}main{max-width:1180px;margin:auto;padding:30px}h1{font-size:29px}h2{margin-top:36px}.cards{display:flex;gap:16px;flex-wrap:wrap}.card,section{background:white;border-radius:12px;padding:20px;margin:16px 0}.card{flex:1;min-width:260px}table{border-collapse:collapse;width:100%;font-size:13px;background:white}td,th{padding:8px;border-bottom:1px solid #dce3ee;text-align:left;white-space:nowrap}th{background:#e6edf7}.scroll{overflow:auto}img,video{width:100%;border-radius:10px}a{color:#176ca7}.note{border-left:4px solid #c58627;background:#fff7e6;padding:14px}code{background:#e6edf7;padding:2px 5px}</style><main>'''
    doc+='<h1>第七阶段：FRED 定位误差与预测能力对照</h1><p>2026-10-17 · 11条原生RAW，各20秒，共220秒；只使用本地已有数据。固定协议后运行，未调参挑选测试结果。</p>'
    tc=metric('official_test_subset','activity','dense_auto','cv');ta=metric('official_test_subset','activity','dense_auto','ca_history')
    doc+=f'<section><b>主要结果：</b>8、9号测试前缀上，自动活动定位＋历史加速度的约0.4秒终点误差为 {f(ta["mean_px"])} px，匀速基线为 {f(tc["mean_px"])} px；两者均输出 {ta["available"]}/{ta["targets"]} 个端点。相对改善不等于实用准确度：只有 {pct(ta["within_quarter_diagonal"])} 的全部端点同时满足“输出且误差不超过目标对角线25%”。不同序列的瓶颈不同：4/5/6号位置替换收益大，8/9号位置替换未改善；后者更需要运动外推修正。<p>本轮转速质量均未达到既定有效门槛，没有验证推力信息的额外预测收益。下一步优先做定位可靠性与目标身份约束，并保留强历史加速度基线；物理分支放在自有近距离、桨盘足够清晰的片段验证。</p></section>'
    doc+='<div class="note">这是检测与预测误差诊断，不是完整FRED排行榜，也不是第六阶段整机侧影分支的正式FRED成绩。本轮隔离测试原桨叶定位和已有活动定位两个基础输入源，没有运行整机侧影或RPM推力分支。自动系统只读RAW；matched_oracle与oracle_all使用过去标注，属于明确分开的诊断对照。</div><div class="cards">'+''.join(cards)+'</div>'
    doc+='<h2>先看：误差是否主要来自位置输入？</h2><p>matched_auto和matched_oracle使用相同滤波器、相同更新时间和相同缺失模式，主动替换的是位置输入。差异同时可能包含定位偏差、错误目标关联、最多50ms的测量滞后，以及由它们引起的速度估计污染；不能解释成各误差来源的独立可加百分比。标注中心也不是质心真值，不保证替换后所有预测器必然更好。</p><img src="localization_intervention.png">'
    doc+='<div class="scroll"><table><tr><th>分组</th><th>前端</th><th>预测器</th><th>共同端点</th><th>自动位置 px</th><th>标注位置 px</th><th>误差降低</th></tr>'+''.join(interventions)+'</table></div>'
    doc+='<h2>输入和评分含义</h2><ul><li>dense_auto：100Hz自动位置流，实际部署输入。</li><li>matched_auto：在过去标注时刻，仅采用过去50ms内最新自动测量。</li><li>matched_oracle：完全相同的更新和缺失模式，仅把坐标换成该时刻标注中心。</li><li>oracle_all：全部过去标注中心，提供更理想输入的参考，不是RAW-only输出。</li><li>cv：匀速卡尔曼；ca_kf：匀加速卡尔曼；ca_history：同一CV状态加最近0.35秒位置的有界二次拟合加速度。</li><li>统一以原始标注时刻为预测起点，终点也只用实际标注。预测时长为名义值至名义值+75ms，实际时长保存在逐项结果；不插值制造真值，不使用未来标注更新过去状态。</li><li>表格为可输出端点的终点误差，未输出比例单列。“输出且准确”指标的分母包含所有可评测端点，避免只报告容易成功的样本。无密集轨迹ADE、真实姿态、转速或推力精度声明。</li></ul>'
    doc+='<img src="horizon_comparison.png">'+''.join(tables)
    if (OUT/'observation_budget.json').exists():
        obs=load(OUT/'observation_budget.json');trs=[]
        for r in obs['results']:
            if r['group'] not in ['additional_train_transfer','official_test_subset']:continue
            trs.append(f"<tr><td>{r['group']}</td><td>{r['front']}</td><td>{r['size_band']}</td><td>{r['fresh']}/{r['label_times']}</td><td>{f(r['mean_center_error_px'])}</td><td>{pct(r['fraction_fresh_and_within_quarter_diagonal'])}</td><td>{r['four_current_times']}</td></tr>")
        doc+='<h2>当前定位质量与目标尺度</h2><p>按标注框对角线分组。当前误差也包含最多50ms观测滞后，不等于姿态或质心误差。候选四桨不代表确实看清四桨。</p><div class="scroll"><table><tr><th>分组</th><th>前端</th><th>目标对角线</th><th>新鲜观测/标注</th><th>当前均值 px</th><th>观测且误差≤对角线25%</th><th>算法报四桨次数</th></tr>'+''.join(trs)+'</table></div><p><a href="observation_budget.json">完整观测质量审计</a></p>'
    doc+='<h2>实际未来画面对照</h2><p>左为当前事件图，右为真实0.4秒后的事件图。橙色是桨叶CV，青色是活动CV，紫色是活动CA，黄色是过去标注输入的CV诊断。绿色标记只在与未来图像相差不超过1ms时显示；预测到画外时不把坐标钳制回图内。右下放大区域由过去的活动CV预测决定，保留全幅画面，没有用未来真值选取裁剪区域。活动前端不是无人机语义分类器。</p>'+videos
    doc+='<h2>运行完整性和耗时</h2><p>各段均从0秒连续处理至20秒，额外0.4秒只用于未来画面。四桨列只是算法报告的四个候选，未被真值证实。带*的总时间复用此前未改变的桨叶观测，只重算活动，不是完整双前端速度；该行桨叶P95来自原始运行。其他行是含输出的双前端研究流水线，也不等于单个部署模型的速度。解码、检测、活动和IO分项均已保存；未据此声称持续实时运行。</p><div class="scroll"><table><tr><th>序列</th><th>归属</th><th>桨叶检测帧</th><th>活动检测帧</th><th>算法报四桨帧</th><th>频率质量≥.35帧</th><th>本次秒*</th><th>桨叶P95 ms</th><th>活动P95 ms</th></tr>'+''.join(runrows)+'</table></div>'
    doc+='<h2>已有工作和实际复用</h2>'+sourceaudit
    doc+='<h2>证据和限制</h2><p>本地旧版coordinates无目标ID。没有把tracks.txt中的另一个跟踪器ID强行当作标注身份；中断超0.5秒时重置状态。未标注时刻不能直接计为无人机不存在。两条测试序列的短前缀不能代表完整测试集、多目标身份保持或长时连续性能。v1在区间外零面积标注检查处停止、尚未评分；v2误引用第二阶段保留的失败版活动器，在测试8/9运行前停止。v3改为已有修复版并通过实际重新捕获测试，所有活动观测从RAW重跑；原始桨叶观测仅在源文件指纹相同的已完成片段复用。模型参数未据此调优，失败记录保留。</p>'
    if (OUT/'alignment_audit.png').exists():doc+='<h2>原生RAW与作者事件帧的固定时刻对齐抽查</h2><p>只用于检查坐标与时间，未优化时间偏移；作者渲染方式和本次20ms积累不同。</p><img src="alignment_audit.png">'
    doc+='<p><a href="阶段结论.md">阶段结论与下一步</a> · <a href="protocol.json">冻结协议</a> · <a href="aggregate.json">汇总JSON</a> · <a href="local_inventory.json">本地数据指纹</a> · <a href="upstream/audit.json">上游代码核查</a></p></main></html>'
    (OUT/'第七阶段定位与预测对照报告.html').write_text(doc,encoding='utf-8')

def validate():
    protocol=verify();manifest=[]
    for i in range(11):
        folder=OUT/'runs'/f'sequence{i}';meta=load(folder/'completion.json');assert meta['frames']==2000
        assert meta['labels_in_inference'] is False
        observations=[json.loads(x) for x in (folder/'observations.jsonl').read_text().splitlines()]
        assert len(observations)==2000
        # Actual automatic stream prefix reproducibility, independent of future label targets.
        for front in FRONTS:
            a=MotionBank();b=MotionBank();short=[];long=[]
            for r in observations:
                o=r[front];z=o.get('center') if o['detected'] else None
                long.append(a.step(r['t'],z))
                if r['t']<=6:short.append(b.step(r['t'],z))
            assert short==long[:len(short)]
        rows=[json.loads(x) for x in (folder/'endpoint_details.jsonl').read_text().splitlines()]
        groups={}
        for r in rows:
            assert r['anchor_t']<=r['target_t']-r['horizon_s']+1e-9
            if r['prediction'] is not None:assert np.isfinite(r['prediction']).all() and np.isfinite(r['error_px'])
            groups.setdefault((r['front'],r['method'],r['horizon_s'],r['target_t']),{})[r['input']]=r
        assert all((g['matched_auto']['prediction'] is None)==(g['matched_oracle']['prediction'] is None) for g in groups.values())
        manifest.append(dict(sequence=i,frames=2000,finite=True,actual_stream_prefix_equal=True,matched_missing_masks_equal=True))
    save(OUT/'validation.json',dict(source_hashes_unchanged=True,checks=manifest))

if __name__=='__main__':
    rows,summary,paired,runs=aggregate();plots(rows,summary)
    for i in [5,8,9]:video(i)
    report(rows,summary,paired,runs);validate();print('Report and validation completed',flush=True)
