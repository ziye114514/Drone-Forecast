"""Figures, actual-RAW overlay video and bounded119-122 conclusions."""
from pathlib import Path
import html,json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage122'),str(ROOT/'research/stage119')]
from study122 import OUT,verify,rows,save,stable,CASES
from study119 import OUT as PARTIAL,REFERENCE,NAMES
VALID=ROOT.parent/'output/stage121_causal_model_validation_20261023'

def load(p):return json.loads(Path(p).read_text('utf-8-sig'))
def table(rr,fields):
    return '<table><tr>'+''.join('<th>'+html.escape(label)+'</th>' for key,label in fields)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(f'{r.get(k):.3f}' if isinstance(r.get(k),float) else str(r.get(k,'')))+'</td>' for k,_ in fields)+'</tr>' for r in rr)+'</table>'

def summary():
    a=load(PARTIAL/'scores119.json');b=load(VALID/'scores121.json');real=[]
    for mode in ['body_hold','affine3','conic3','guarded3','validated121']:
        rr=[r for r in (b['real_consistency'] if mode=='validated121' else a['real_consistency']) if r['mode']==mode]
        real.append(dict(mode=mode,center_consistency_rmse_px=float(np.sqrt(np.mean([r['center_consistency_rmse_px']**2 for r in rr]))),
            reference_fde04_mean_px=float(np.mean([r['reference_fde04_mean_px'] for r in rr])),center_reference_pairs=sum(r['center_reference_pairs'] for r in rr),
            future_reference_pairs=sum(r['reference_forecast_pairs'] for r in rr),distinct_raw_sequences=1,distinct_masks=4))
    save(OUT/'summary122.json',dict(real=real,synthetic=a['synthetic']+b['synthetic'],real_COM_truth=False,physical_occlusion=False));return real

def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    real=summary();names=['body_hold','affine3','conic3','validated121'];colors=['#79838d','#5680c2','#df8827','#169a7b']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    realvalues=[next(r['center_consistency_rmse_px'] for r in real if r['mode']==m) for m in names]
    syn=load(PARTIAL/'scores119.json')['synthetic']+load(VALID/'scores121.json')['synthetic']
    synthetic=[next(r['relative_position_rmse_px'] for r in syn if r['case']=='stationary_wobble' and r['mode']==m) for m in names]
    for ax,values,title in zip(axes,[realvalues,synthetic],['Real feature dropout: full-four consistency ONLY','Ideal stationary wobble: known relative-motion truth']):
        bars=ax.bar(names,values,color=colors);ax.bar_label(bars,fmt='%.2f');ax.tick_params(axis='x',labelrotation=18);ax.set_ylabel('Center error / RMSE (px)');ax.set_title(title,fontsize=10);ax.set_ylim(0,max(values)*1.22);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Current evidence: no single partial observer wins both domains');fig.tight_layout();fig.savefig(OUT/'partial_comparison122.png',dpi=150);plt.close(fig)
    ideal=load(PARTIAL/'synthetic119.json')['cases'];chosen=load(VALID/'synthetic121.json')['cases'];fig,axes=plt.subplots(2,1,figsize=(11,7))
    for ax,case in zip(axes,['stationary_wobble','translation_wobble']):
        rr=next(c['records'] for c in ideal if c['case']==case);ss=next(c['records'] for c in chosen if c['case']==case)
        for name,color in zip(names,colors):
            first=next((r for r in rr if r['outputs'][name]['center']['current']),None) if name!='validated121' else next(r for r in ss if r['output']['center']['current'])
            q=first['outputs'][name] if name!='validated121' else first['output'];offset=np.array(q['forecast']['coefficients'][NAMES[2]]['position'])-first['truth'];xx=[];yy=[]
            for i,r in enumerate(rr):
                q=r['outputs'][name] if name!='validated121' else ss[i]['output'];c=q['forecast']['coefficients'].get(NAMES[2])
                if c:xx.append(r['t']);yy.append(c['position'][0]-offset[0])
            ax.plot(xx,yy,color=color,label=name,linewidth=1.6)
        ax.plot([r['t'] for r in rr],[r['truth'][0] for r in rr],'k--',label='known true center x');ax.axvspan(.9,1.98,color='#f5d7b8',alpha=.4,label='one observation removed');ax.set_title(case);ax.set_ylabel('Aligned center x (px)');ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=3)
    axes[-1].set_xlabel('Source time (s)');fig.tight_layout();fig.savefig(OUT/'wobble_translation122.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,6.8))
    realrr=load(VALID/'clear6_mask0.json')['records'];synrr=next(c['records'] for c in chosen if c['case']=='stationary_wobble')
    for ax,rr,title in zip(axes,[realrr,synrr],['Real RAW-derived observations: favors affine3','Ideal circles with perspective wobble: favors conic3']):
        for mode,color in [('affine3',colors[1]),('conic3',colors[2])]:
            values=[r['output']['selection']['prior_errors_px2'] for r in rr]
            ax.plot([r['t'] for r in rr],[np.nan if v is None else max(1e-6,v[mode]) for v in values],label=mode,color=color)
        ax.set_yscale('log');ax.set_ylabel('Past validation MSE (px^2)');ax.set_title(title);ax.legend();ax.grid(alpha=.2)
    axes[-1].set_xlabel('Source time (s)');fig.tight_layout();fig.savefig(OUT/'causal_validation122.png',dpi=150);plt.close(fig)

def video():
    folder=OUT/'raw/clear6';rr=rows(folder/'observations_forecasts.jsonl');full=rows(ROOT.parent/'output/stage110_merge_safe_geometry_20261022/clear6/records.jsonl')
    cap=cv2.VideoCapture(str(folder/'forecasts.mp4'));w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));dest=OUT/'validated_partial_forecast122.mp4'
    writer=cv2.VideoWriter(str(dest),cv2.VideoWriter_fourcc(*'mp4v'),50.,(w+480,h));assert writer.isOpened();poster=None
    for i,r in enumerate(rr):
        ok,im=cap.read();assert ok;display=np.full((h,w+480,3),248,np.uint8);display[:,:w]=im;q=r['frontends']['delayed_feedback'];worker=q['learned_structure118'];v=worker['validated122'];s=v['selection'];body=r['body_observation']
        for z in worker['input_rotors']:cv2.ellipse(display,((z['cx'],z['cy']),(z['major'],z['minor']),z['angle']),(210,150,20),2)
        if worker['removed_one_observation']:
            seen=np.array([[z['cx'],z['cy']] for z in worker['input_rotors']])
            for z in full[i]['outputs']['merge_safe']['geometry'].get('rotors',[]):
                if np.min(np.linalg.norm(seen-[z['cx'],z['cy']],axis=1))>1:
                    cv2.ellipse(display,((z['cx'],z['cy']),(z['major'],z['minor']),z['angle']),(30,35,225),2)
                    cv2.drawMarker(display,(round(z['cx']),round(z['cy'])),(30,35,225),cv2.MARKER_TILTED_CROSS,12,2)
        for name,color in [('partial_body_hold_kf120',(120,120,120)),('partial_conic3_kf120',(35,125,225)),('partial_validated_kf122',(65,165,20))]:
            c=q['coefficients'].get(name)
            if q['valid'] and c:
                points=np.rint([np.array(c['position'])+dt*np.array(c['velocity']) for dt in [0,.1,.2,.3,.4]]).astype(np.int32)
                cv2.polylines(display,[points],False,color,2);cv2.circle(display,tuple(points[-1]),4,color,1)
        lines=[f'RAW t={r["t"]:.2f}s; source50fps','Offline replay, NOT realtime','Green: validated0.4s forecast','Orange: always three-conic model','Gray: original body-hold fallback','Cyan: current observed ellipses','Red: held OUT of estimator only','Pixels are NOT physically occluded','',f'Current rotor observations: {len(worker["input_rotors"])}',f'Chosen branch: {s["chosen"]}',f'Past full-view checks: {s["checks_before"]}','No fourth input read on3-input ticks']
        if s['prior_errors_px2']:
            lines += [f'Past affine MSE: {s["prior_errors_px2"]["affine3"]:.2f}px^2',f'Past conic MSE: {s["prior_errors_px2"]["conic3"]:.2f}px^2']
        for j,line in enumerate(lines):cv2.putText(display,line,(w+12,25+j*25),0,.48,(25,45,55),1)
        if body.get('detected'):
            b=np.array(body['bbox']);lo=np.maximum(np.floor(b[:2]-12).astype(int),0);hi=np.minimum(np.ceil(b[2:]+12).astype(int),[1280,720]);crop=display[lo[1]:hi[1],lo[0]:hi[0]]
            if crop.size:
                scale=min(450/crop.shape[1],240/crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST);display[465:465+crop.shape[0],w+12:w+12+crop.shape[1]]=crop
        writer.write(display)
        if i==85:poster=display.copy()
    cap.release();writer.release();cap=cv2.VideoCapture(str(dest));decoded=0
    while True:
        ok,_=cap.read()
        if not ok:break
        decoded+=1
    cap.release();assert decoded==200;cv2.imwrite(str(OUT/'dropout_midpoint122.png'),poster)
    save(OUT/'video_checks122.json',dict(frames_decoded=decoded,path=str(dest),source_fps=50,not_processing_fps=True,visual_annotation_only_after_inference=True))

def report():
    verify();figures();video();real=summary();syn=load(VALID/'scores121.json')['synthetic'];checks={c:load(OUT/'raw'/c/'checks122.json') for c in CASES}
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>119—122：缺桨观测与模型自检</title><style>body{max-width:1250px;margin:30px auto;padding:0 22px;font:17px/1.8 system-ui;color:#243947}img,video{max-width:100%;height:auto}table{border-collapse:collapse;font-size:14px}td,th{padding:7px;border:1px solid #cbd5df}th{background:#eaf3f4}.note{padding:14px;background:#fff0cf}a{color:#087c87}code{background:#edf2f5}h2{margin-top:2em}</style><h1>119—122：缺一个桨盘后，怎样知道模型还值不值得信</h1>
<p>2026-10-23。目标收窄为一个可验证的改进：在四桨清楚时学习结构并做自检；少一个桨盘时，用过去的自检结果选择三个中心的仿射更新或三个椭圆的投影模型；持续估计中心与速度。极端侧视仍退回机体历史分支。本轮不是新的转速/真实推力估计。</p>
<p class="note">目前有具体机制和小实验支持，仍是创新候选。四组真实对照来自同一段RAW的四种“删去一个特征”，不是四次独立飞行，也不是物理遮挡。表中的真实数据误差衡量与四桨完整估计的一致性，不能当作实际重心精度或前沿比较。</p>
<h2>先把失败说明白</h2><p>固定117的无量纲结构后，119仅用当前三个椭圆的周长约束H8，不把缺的第四个桨盘塞成观测。也加入了24种一一关联、局部雅可比条件、五份已学结构重拟合的中心散布检查。但在四种真实缺桨对照中，这些检查全部放行，没有识别出模型偏差。</p><p>理想圆投影里三椭圆很好，真实数据里三中心仿射却更稳。真实事件提取的椭圆可能偏离等半径共面圆模型；当前证据能说明模型适配性不足，不能把原因全部确定为某一种轮廓污染。</p>
<h2>新增的自检怎样工作</h2><p>只在四个当前观测均存在、模型有效时，轮流隐藏其中一个；分别用剩下三个中心/椭圆估计整机中心，与当前四桨模型中心比较。将四种隐藏方式的误差平方取均值，按0.2秒时间常数更新历史分数；至少五次有效检查后，缺桨时按严格过去的分数选分支。当前缺桨帧不更新自检，也不读取隐藏观测。</p><p>选择的是当前中心观测，再进入同一个位置、速度、外观中心偏移滤波器；切换时不清零速度。不清楚、模型过期或两桨缺失时保持原回退。结构记忆、当前姿态和真实观测仍分开保存。模型仍假设四桨等半径、共面、近矩形机架；不提供米制尺寸或真实重心。</p>
<img src="causal_validation122.png"><h2>真实数据：四种缺桨方式的汇总</h2>'''+table(real,[('mode','方法'),('center_consistency_rmse_px','中心一致性RMSE/px'),('reference_fde04_mean_px','0.4秒相对完整参考误差/px'),('center_reference_pairs','中心比较时刻'),('future_reference_pairs','未来比较时刻')])+'''
<p>每种方式200时刻，其中90时刻删除一个完整桨盘观测。首次删去位于23.32—24.50秒，第二次25.12—25.70秒。四种方式合计360个中心比较、340个0.4秒比较，数据高度重叠。自检360次都选仿射，因此与“始终仿射”相同；本真实片段不能证明自检优于始终仿射，也不能宣称0.4秒真实位置误差下降到表中数值。</p>
<img src="partial_comparison122.png"><h2>已知真值：静止摆动、真实平移和变向</h2>'''+table(syn,[('case','合成情形'),('relative_position_rmse_px','相对位置RMSE/px'),('fde04_mean_px','0.4秒误差/px'),('choices','缺桨时选择')])+'''
<p>在理想的带噪投影圆轮廓中，自检在正常视角选择三椭圆：静止姿态摆动相对位置RMSE约0.58px，真实平移约0.71px，突然变向约0.84px；它没有把所有运动压成静止。但突然变向0.4秒误差仍约18.66px，因为本轮输出仍按当前速度外推，没有提前知道未来控制指令。近侧视及缺两个桨盘仍退回原分支，未解决退化。</p><img src="wobble_translation122.png">
<h2>实际RAW运行与视频</h2><p>120先完成330个实际RAW父时刻；122再次完成清晰200、独立截断100、负例30，共330。输入为E盘RAW和既有检测权重，隔离几何进程禁止读output缓存/标注；122原68路系数/预测严格保持120，新选择器状态/系数严格匹配121，实际截断和事件时间界通过。视频为真实RAW重读后的离线预览。</p>
<video controls preload="metadata" poster="dropout_midpoint122.png" src="validated_partial_forecast122.mp4"></video><p>绿线为自检分支的0.4秒预测，橙线为始终三椭圆，灰线为原机体回退。红色椭圆只是推理结束后的可视化标记：其特征被从新估计器输入删除，事件像素没有被挡住；旧64条对照仍接收原数据。影片按源50fps播放，绝不表示计算达到50fps。</p>
<h2>与已有研究的边界</h2><p><a href="https://robots.ox.ac.uk/~vgg/publications/1992/Rothwell92c/">Rothwell等1992</a>已研究已知共面曲线/圆锥曲线的位姿；<a href="https://link.springer.com/article/10.1186/s41074-018-0050-y">Cai与Wu2018</a>分析共面圆锥曲线单应及歧义。因此“三个椭圆/已知模型求姿态”不是原创。</p>
<p><a href="https://arxiv.org/html/2609.11920v1">EVPeriscope</a>第3.2节当前实现要求四个桨心，也明确说原则上三个足够；它使用已知非对称机架和PnP/EKF。<a href="https://github.com/Zibin6/LOPET">LOPET作者MATLAB项目</a>已实现事件线与模型匹配跟踪；本轮只查阅，未声称运行/移植作者系统。</p>
<p><a href="https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/trit.2019.0107">Jiang等2020</a>已将事件计数图的检测器、在线跟踪器和Kalman融合。因此“多个分支互补、在线适应、滤波”也不是独立创新。<a href="https://arxiv.org/html/2512.21053v1">Liu等2025</a>继续研究模型与事件角点/边缘的关联，说明部分特征模型跟踪已有广泛研究。</p>
<p>本项目值得继续验证的窄差异是：针对事件桨盘的投影模型失配，利用四桨可见期间的逐桨隐藏自检，为后续缺桨选择几何观测；并检验它能否减少姿态造成的伪平移。检索未找到完全相同的事件四旋翼实现，不代表证明文献中不存在。交叉检验、误差滑动平均、选择器本身均为已有工具；当前收益在理想/真实两域分开出现，尚未在同一真实录像证明动态切换优势。</p>
<h2>当前卡点与下一步</h2><p>第一，真实桨盘拟合仍有系统偏差，自检的四桨参考自身也可能错。第二，三条独立完整椭圆仍是要求，极端侧视/重叠弧段不满足。第三，缺桨只做了特征删除，需要真实连续退化与恢复录像。第四，数十条对照与同步重复求解仍很慢，当前新增模块是研究原型，不是实时部署。</p>
<p>下一轮优先冻结方法，在此前未用于调参的自采连续RAW上验证“四桨清楚→部分丢失→恢复”，将中心一致性、人工可见机体中心误差和未来框误差分别报告；比较始终仿射、始终投影、原回退、自检四组。若真实数据仍全选仿射，就应保留简化方案，把创新收敛到可靠性判别，不能为了保留复杂模型宣称优越。之后再裁剪最小运行路径和做速度基准。</p>
<p>119方案文字将edge_on区间误写为35—104，冻结的实际代码始终为45—99，本报告按代码的55时刻报告；未按结果改区间。另核验该控制的三个中心扁平比约0.095—0.150、最小椭圆轴比0.114—0.194，属于扁平投影及姿态突变，不能当成极端成线视角的充分测试。</p>
<p>本轮保留两个核验脚本断言修正记录：120启动时不是所有分支已初始化；119原生清晰段也含自然三桨时刻。仅修正核验假定，未改推理或按结果调参数。<a href="verification_amendment120.json">120核验修正记录</a>；<a href="controls119.json">119控制检查</a>；<a href="summary122.json">汇总数值</a>。</p></html>'''
    # Copies keep all report links local and reviewable.
    save(OUT/'verification_amendment120.json',load(ROOT.parent/'output/stage120_RAW_partial_structure_20261023/verification_amendment120.json'))
    save(OUT/'controls119.json',load(PARTIAL/'controls119.json'))
    report=OUT/'第一百一十九至一百二十二阶段缺桨与模型自检报告.html';report.write_text(page,encoding='utf-8')
    save(OUT/'delivery_checks122.json',dict(report=str(report),actual_RAW122=sum(z['frames'] for z in checks.values()),actual_RAW120=330,
        original_RAW122_preview_frames=sum(z['decoded_frames'] for z in checks.values()),new_overlay_decoded=200,old68_preserved=True,
        selector_matches121=True,actual_prefix100=True,negative30=True,no_default_promotion=True,real_COM_accuracy_proven=False,extreme_side_recovery=False,realtime=False))
    print(json.dumps(dict(report=str(report),real=real)),flush=True)

if __name__=='__main__':report()
