"""Sparse-reference forecast report; future RAW is used only for post-score figures."""
import json,html,sys
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study22 import ROOT,OUT,INPUT,CASES,FRONTENDS,MODELS,load,save,verify


def fmt(x):return '—' if x is None else f'{x:.2f}'
def table(headers,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+str(x)+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>'


def figures(a):
    from pathlib import Path
    sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'src'))
    from event_drone_forecast.raw_io import open_raw,raw_chunks
    source=load(INPUT/'protocol.json');chosen=[]
    for case in ['clear6','side6','review4']:
        valid=[d for d in a['details'] if d['case']==case and d['frontend']=='native20' and d['model']=='cv_hist_acc' and d['nominal_h']==.4 and d.get('error_px') is not None]
        if not valid:continue
        if case=='side6':d=max(valid,key=lambda d:d['error_px']);why='worst_history_acceleration'
        else:
            median=np.median([d['error_px'] for d in valid]);d=min(valid,key=lambda d:abs(d['error_px']-median));why='median_history_acceleration'
        t=d['target_t'];cache=OUT/f'future_raw_{case}_{t:.6f}.npz'
        if not cache.exists():
            reader=open_raw(source['cases'][case]['raw']);h,w=reader.get_size();chunks=list(raw_chunks(reader,t-.02,.02,20000));assert len(chunks)==1
            e=chunks[0][1];p=e['y'].astype(np.int32)*w+e['x'];np.savez_compressed(cache,total=np.bincount(p,minlength=w*h).reshape(h,w))
        count=np.load(cache)['total'];h,w=count.shape;grey=np.uint8(255-230*np.clip(np.log1p(count)/np.log(8),0,1))
        label=next(l for l in load(ROOT.parent/'output/raw_only_project/evaluation_data'/('train4.json' if case=='review4' else 'train6.json'))['labels'] if l['t']==t)
        b=label['bbox'];target=np.array(d['target_center']);entries=[next(x for x in a['details'] if x['case']==case and x['frontend']=='native20' and x['model']==m and x['nominal_h']==.4 and x['target_t']==t) for m in MODELS]
        fig,axes=plt.subplots(1,2,figsize=(14,5.8),layout='constrained');points=[target]
        for ax in axes:
            ax.imshow(grey,cmap='gray',vmin=0,vmax=255,extent=(0,w,h,0))
            ax.add_patch(plt.Rectangle(b[:2],b[2]-b[0],b[3]-b[1],fill=False,color='#1b9b50',lw=2));ax.scatter(*target,color='#1b9b50',marker='+',s=80,label='archived box center')
            for q,c in zip(entries,['#777777','#1676bd','#df7d20']):
                p=np.array(q['predicted']);points.append(p);ax.scatter(*p,color=c,marker='x',s=70,label=f"{q['model']}: {q['error_px']:.1f}px")
            ax.set_xlabel('native x [px]');ax.set_ylabel('native y [px]');ax.grid(alpha=.2)
        axes[0].set_xlim(0,w);axes[0].set_ylim(h,0);axes[0].set_title(f'Future observed RAW / {case} T={t:.3f}s')
        p=np.array(points);lo=np.minimum([0,0],p.min(0)-[60,60]);hi=np.maximum([w,h],p.max(0)+[60,60])
        axes[1].set_xlim(lo[0],hi[0]);axes[1].set_ylim(hi[1],lo[1]);axes[1].set_title(f"Saved forecasts from {d['anchor_t']:.2f}s / actualH={d['actual_h']:.3f}s")
        axes[1].legend(loc='best',fontsize=9)
        name=f'forecast_{case}_{t:.3f}.png';fig.savefig(OUT/name,dpi=135);plt.close(fig)
        chosen.append(dict(case=case,target_t=t,anchor_t=d['anchor_t'],actual_h=d['actual_h'],selection=why,file=name,
            scope='Post-score illustration; future RAW and archived box are never fed to predictor. Selection explicitly diagnostic, not new score sample.'))
    save(OUT/'visual_selection.json',chosen);return chosen


def main():
    verify();a=load(OUT/'analysis.json');images=figures(a)
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第二十二阶段：未来位置端点核验</title><style>body{font:17px/1.75 "Microsoft YaHei",sans-serif;margin:0;background:#f3f6fa;color:#243342}main{max-width:1250px;margin:auto;background:white;padding:32px}h1{font-size:31px;line-height:1.3}h2{margin-top:38px;border-bottom:2px solid #dce5ed}.note{padding:18px;background:#edf7fb;border-left:5px solid #3387a7}.warn{background:#fff4de;border-color:#c68b25}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;white-space:nowrap;text-align:left;border-bottom:1px solid #dce5ec}th{background:#edf2f7}a{color:#176c98}img{max-width:100%;height:auto}details{padding:14px;border:1px solid #dce5ec;margin:18px 0}summary{cursor:pointer;font-weight:bold}figure{margin:24px 0}figcaption{font-size:14px;color:#637484}</style><main><h1>第二十二阶段：用存档整机框核验未来位置端点</h1><p>2026-10-18 · 四种前端 · hold/CV/历史加速度 · 先保存预测再读取评分标签</p><div class="note"><b>这轮是未来位置评分，不是当前检测评分。</b>每个标注目标时刻T，选择不晚于T−H的预测锚点，再用当时保存的状态计算T处端点。参考是存档整机框中心，已不再是检测器未来输出。</div><h2>1. 评测口径与更正</h2><p>此前把“稀疏标签不够评测”同时用于ADE/FDE过于绝对：它们不足以支持完整连续轨迹ADE，但可以核验标注时刻的未来端点。本轮补上该评分，不将其称为标准密集轨迹基准。</p><p>锚点为T−H前最近20ms截止，间隔≤25ms；用锚点保存的二次多项式在精确T求值。实际前瞻是H至H+25ms，不偷用未来状态。需至少5次真实观测，最后实测后160ms停止新预测，500ms重置。缺失计入覆盖，未用未来数据补齐。</p><p>CV与历史加速度直接复用阶段14实现，另加保持最后位置hold基线；所有前端同样的门控。每份forecasts.jsonl全部保存后，评分才读取标签。</p><h2>2. 全部有效样本与缺失</h2><div class="note warn">样本少、存档框中心/时间轴有近似误差。尤其不能只挑各前端共同样本或只报告中位数：历史加速度在少数时刻会严重过冲，必须同时查看均值、最大值、覆盖。</div>'''
    for case in ['clear6','side6','review4']:
        text+=f'<details'+(' open' if case=='clear6' else '')+f'><summary>{case}：未来端点原始评分</summary>'
        for h in [.1,.2,.4]:
            rr=[r for r in a['table'] if r['case']==case and r['nominal_h']==h]
            text+=f'<h3>名义H={h:.1f}s，实际略长最多25ms</h3>'+table(['前端','预测器','有机目标数','有锚点','有效预测','平均FDEpx','中位px','最大px','联合通过','无机标签内视野预测'],[[r['frontend'],r['model'],r['present_targets'],r['anchors_available'],r['valid_forecasts'],fmt(r['mean_fde_px']),fmt(r['median_fde_px']),fmt(r['max_fde_px']),r['joint_success'],f"{r['absent_forecasts_inside']}/{r['absent_targets']}"] for r in rr])
        text+='</details>'
    text+='<p>联合通过=有预测且目标端点误差≤整机框对角线25%；框较大时门槛也宽，不能替代像素误差。无机标签“内视野预测”表示较早锚点预报未来仍在画面，不是当前检测误报。train5四前端均无预测发布。</p><h2>3. 关键观察</h2><p>以native20为例：清晰名义0.4s，CV平均24.66px，保持位置53.03px；侧视CV131.00px，保持150.57px。历史加速度侧视均值293.03px，比CV更差；它在阶段14对检测器未来输出较好，不代表对存档框也稳定最好。</p><p>native10在train4当前定位5/5，不等于0.4s预测覆盖5/5：此前锚点常未形成足够轨迹，只有1/5有效。清晰native10的0.4s历史加速度也出现平均56.01px，CV20.84px。前端瞬时检出、轨迹历史可用与运动外推必须分开统计。</p><h2>4. 四前端共同样本（补充，不替代全表）</h2>'
    for h in [.1,.2,.4]:
        rr=[r for r in a['common'] if r['nominal_h']==h and r['case']!='negative5']
        text+=f'<h3>名义{h:.1f}s</h3>'+table(['片段','共同目标数','前端','hold','CV','历史加速度'],[[r['case'],r['n'],f,*[fmt(r['mean_fde_px'][f][m]) for m in MODELS]] for r in rr for f in FRONTENDS])
    text+='<p>例如native20侧视0.4s，全部有效样本历史加速度均值293.03px；删到四前端共同11点后变72.33px。这种变化主要来自样本筛选，不能只用共同表宣称稳健。</p><h2>5. 查看预测与之后实际到达的事件画面</h2><p>绿色为目标时刻存档整机框；点为此前保存的预测。左显示原视野，右扩大坐标轴以保留视野外的失败预测。背景是未来时刻到达后读取的原生20ms ON/OFF计数，仅作事后展示，绝不参与过去预测。</p>'
    for d in images:text+=f'<figure><a href="{d["file"]}"><img src="{d["file"]}" loading="lazy"></a><figcaption>{d["case"]}，T={d["target_t"]:.3f}s，锚点{d["anchor_t"]:.2f}s，实际前瞻{d["actual_h"]:.3f}s；事后选例：{d["selection"]}。</figcaption></figure>'
    worst=sorted([d for d in a['details'] if d.get('error_px') is not None],key=lambda d:-d['error_px'])[:10]
    text+='<h2>6. 最差十个端点，保留失败</h2>'+table(['片段','前端','模型','目标s','锚点s','实际H','误差px'],[[d['case'],d['frontend'],d['model'],f"{d['target_t']:.3f}",f"{d['anchor_t']:.2f}",f"{d['actual_h']:.3f}",fmt(d['error_px'])] for d in worst])
    text+='<h2>7. 下一步与边界</h2><p>先处理历史加速度的离群过冲：用已有稳健拟合或不确定性门控作预先冻结消融，继续保留CV/hold与全部缺失。再检验自适应时间窗及模型活跃短段；不根据少数标签选择一次漂亮结果作为统一升级。</p><p>参考是二维粗框中心，不是质心、三维位置、姿态或机械RPM；已在同批数据开发，未宣称盲测/前沿水平。无G、无yolov10-main访问；旧src/default保留。<a href="protocol.json">协议与指纹</a>、<a href="module_checks.json">因果/门控检查</a>、<a href="analysis.json">逐目标评分</a>、<a href="阶段结论.md">阶段结论</a>。</p></main></html>'
    (OUT/'第二十二阶段未来端点评测报告.html').write_text(text,encoding='utf-8');print('Stage22 report rendered',flush=True)


if __name__=='__main__':main()
