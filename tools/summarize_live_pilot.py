"""Summarize one frozen short live-mode pilot, including precision tradeoffs."""
from pathlib import Path
import argparse,bisect,hashlib,html,json,shutil,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage154'))
from turn154 import forecast as curve


def read_rows(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s]
def stats(values):
    return dict(n=len(values),mean=float(np.mean(values)) if values else None,
                median=float(np.median(values)) if values else None,
                p95=float(np.percentile(values,95)) if values else None)


def reference_at(records,t):
    stamps=[r['t'] for r in records];i=bisect.bisect_left(stamps,t)
    if i<len(records) and abs(stamps[i]-t)<1e-7:pair=[records[i],records[i]]
    elif 0<i<len(records) and stamps[i]-stamps[i-1]<=.1500001:pair=[records[i-1],records[i]]
    else:return None
    if any(r.get('state')!='present' or not r.get('box') for r in pair):return None
    centers=[(np.asarray(r['box'][:2])+r['box'][2:])/2 for r in pair]
    alpha=0. if pair[0]['t']==pair[1]['t'] else (t-pair[0]['t'])/(pair[1]['t']-pair[0]['t'])
    return (1-alpha)*centers[0]+alpha*centers[1]


def old_forecast(row,h):
    c=row['coefficients'].get('optimized_geometry')
    if c is None:return None
    if not row['predictor']['structure_supported']:return curve(c,h)
    p=np.asarray(c['position']);delta=np.asarray(c['velocity'])*h+.5*h*h*np.asarray(c['acceleration'])
    return (p+max(.5,1.-.9*h*h)*delta).tolist()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();summary={};scores={};checks={}
    for name,duration in [('clear',2.),('side',4.),('train2',4.)]:
        run=out/name;rows=read_rows(run/'predictions.jsonl');tracking=read_rows(run/'tracking.jsonl')
        complete=json.loads((run/'completion.json').read_text(encoding='utf-8'))
        protocol=json.loads((run/'protocol.json').read_text(encoding='utf-8'))
        valid=[r for r in rows if r['valid']]
        summary[name]=dict(completion=complete,source_path=str(run),duration_s=duration,
                           publication_slots_per_source_second=len(rows)/duration,valid_per_source_second=len(valid)/duration,
                           valid_age_ms=stats([r['observation_age_ms'] for r in valid]),
                           publication_count=len(rows),valid_count=len(valid))
        cutoff=all(r['max_event_t'] is None or r['max_event_t']<round(r['t']*1e6) for r in tracking)
        target=all(r['observation_t']<=r['t'] and all(abs(r['forecast_target_t'][str(h)]-r['t']-h)<1e-8 for h in (.1,.2,.4)) for r in rows)
        cadence=all(b['t']-aa['t']>=.2-1e-6 for aa,b in zip(rows,rows[1:]))
        expected_native=round((duration+.1)*50)
        assert cutoff and target and cadence and complete['sources_unchanged'] and complete['native_batches']==expected_native
        # Save the precise executed code, not whichever version is edited later.
        for relative,digest in protocol['sources'].items():
            source=ROOT/relative;assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
            dest=out/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        import cv2
        cap=cv2.VideoCapture(str(run/'predictions.mp4'));frames=round(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert frames==len(rows)
        for index in sorted(set([0,max(0,frames//2),max(0,frames-1)])):
            cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,image=cap.read();assert ok
            if index==frames//2:cv2.imwrite(str(run/'preview_checked.png'),image)
        cap.release()
        checks[name]=dict(native_cutoff=True,publish_target_time=True,minimum_interval=True,
                          native_batches=complete['native_batches'],video_frames=frames,
                          first_middle_last_decoded=True,sources_match_snapshot=True)
        if name=='clear':continue
        label=ROOT.parent/'output'/('dense_labels_side_110_1195' if name=='side' else 'dense_labels_train2_40_46')/'labels.json'
        records=sorted(json.loads(label.read_text(encoding='utf-8'))['records'].values(),key=lambda q:q['t'])
        old={round(r['t'],6):r for r in read_rows(ROOT.parent/'output/stage156_parallel_geometry_20261024'/name/'predictions.jsonl')}
        scores[name]={}
        for h in (.1,.2,.4):
            errors={k:[] for k in ['live_compensated','live_without_age','old_state_same_target','body_cv']};detail=[];missing=0
            for row in valid:
                gt=reference_at(records,row['t']+h);previous=old.get(round(row['observation_t'],6))
                if gt is None or previous is None or not previous.get('valid'):
                    missing+=1;continue
                base=old_forecast(previous,h+row['t']-row['observation_t'])
                if base is None:missing+=1;continue
                values=dict(live_compensated=row['predictions']['live_motion'][str(h)],
                            live_without_age=row['observation_predictions']['live_motion'][str(h)],
                            old_state_same_target=base,body_cv=row['predictions']['body_cv'][str(h)])
                error={k:float(np.linalg.norm(np.asarray(point)-gt)) for k,point in values.items()}
                for k,v in error.items():errors[k].append(v)
                detail.append(dict(publication_t=row['t'],observation_t=row['observation_t'],target_t=row['t']+h,errors=error))
            scores[name][str(h)]=dict(metrics={k:stats(v) for k,v in errors.items()},missing=missing,detail=detail)
    report=dict(timing=summary,precision=scores,checks=checks,
                precision_reference='Linear interpolation only between adjacent present manual boxes <=150ms apart. Approximate reference; not new ground truth.',
                baseline='Previous156 state at EXACT same native observation time, re-extrapolated to SAME publication target. Offline reference, not a real-time deployment comparison.',
                default_research_algorithm_changed=False,physical_RPM_or_thrust_gain_proven=False)
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(summary);fig,axes=plt.subplots(1,2,figsize=(11,4))
    x=np.arange(3)
    axes[0].bar(x-.18,[summary[n]['completion']['observation_age_median_ms'] for n in names],.36,label='All slots median')
    axes[0].bar(x+.18,[summary[n]['completion']['observation_age_p95_ms'] for n in names],.36,label='All slots p95')
    axes[0].set(xticks=x,xticklabels=names,ylabel='Observation age at publication (ms)');axes[0].legend()
    axes[1].bar(x-.18,[summary[n]['publication_slots_per_source_second'] for n in names],.36,label='Publication slots / source second')
    axes[1].bar(x+.18,[summary[n]['valid_per_source_second'] for n in names],.36,label='Valid predictions / source second')
    axes[1].axhline(5,color='gray',ls='--');axes[1].set(xticks=x,xticklabels=names,ylabel='Hz');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'timing.png',dpi=150);plt.close(fig)
    timing_rows=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [name,
        f"{q['publication_count']}/{q['valid_count']}",f"{q['publication_slots_per_source_second']:.2f}/{q['valid_per_source_second']:.2f}",
        f"{q['completion']['observation_age_median_ms']:.1f}/{q['completion']['observation_age_p95_ms']:.1f}",
        f"{q['completion']['processing_median_ms']:.1f}"])+'</tr>' for name,q in summary.items())
    score_rows=''
    for name,hh in scores.items():
        for h,item in hh.items():
            m=item['metrics'];vs=[name,h,m['live_compensated']['n'],item['missing']]+[None if m[k]['mean'] is None else round(m[k]['mean'],2) for k in ('live_compensated','live_without_age','old_state_same_target','body_cv')]
            score_rows+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in vs)+'</tr>'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>低延迟位置通道与5Hz发布</title><style>body{max-width:1150px;margin:35px auto;padding:0 20px;font:17px/1.8 system-ui;color:#203442;background:#f5f7fa}section{background:white;padding:22px;margin:20px 0;border-radius:12px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #cfd9e4;padding:7px}img,video{width:100%}code{overflow-wrap:anywhere}</style>
<h1>低延迟位置通道与5Hz发布</h1><p>2026-10-25 · 新增可选live模式；原研究算法默认保留</p>
<section><h2>实际改动</h2><p>RAW解码/SDK独立进程，持续处理原生20ms事件；位置跟踪目标25Hz；发布目标5Hz。
只保留最新待处理帧，完整推理预热放在回放开始前。过旧输入在推理前弃用；预测目标为实际发布时刻之后0.1/0.2/0.4秒，补偿观测年龄。</p>
<p>使用官方EventsIterator替代live模式的循环缓冲读取。同一4.1秒、205切片的x/y/p/t逐片SHA256完全一致；顺序探针读取中位2.65→1.20ms，运行中读取峰值173.85→29.83ms。缓存/负载未受控，不据此宣称固定倍数硬件加速。</p>
<p>这是位置预测通道，未使用桨盘姿态或推力。事件周期诊断可选且仍在原始时间戳上工作，不等于转速/推力已解。原research与实验桨盘模式保留。</p></section>
<section><h2>最终短段实测</h2><p>冷启动不计入下面运行中观测年龄，完整启动耗时保存在completion.json。总体Hz按整个请求录像时长计算，包含启动/失锁空档，不用活跃期平均频率掩盖漏输出。</p>
<table><tr><th>片段</th><th>发布/有效数量</th><th>总体发布/有效Hz</th><th>观测年龄中位/P95 ms</th><th>处理中位ms</th></tr>'''+timing_rows+'''</table><img src="timing.png"></section>
<section><h2>精度与代价</h2><p>参考为相邻人工可见框中心间的线性插值（间隔≤150ms），属于近似核查，不是新增真实轨迹。只比较live有效、旧状态也有效的同起点/同未来目标，所有缺失另列。
旧156列使用同一观测时刻的离线状态，重新外推到相同发布时间+h；它没有实际低延迟部署，不能当完全相同的端到端比较，也不能与旧45.594px跨口径排名。</p>
<table><tr><th>片段</th><th>h秒</th><th>共同对</th><th>缺失</th><th>live补延迟</th><th>live不补延迟</th><th>旧156状态同目标</th><th>live匀速</th></tr>'''+score_rows+'''</table>
<p>没有把速度收益说成精度收益；样本较少，若live更差，仍作为低延迟可选通道，不能升级为精度更好的默认算法。</p></section>
<section><h2>预测视频</h2><p>固定5fps只用于展示发布画面。缺失时段会被压缩，实际时间/有效状态以JSON与画面时间戳为准。</p>
<h3>侧视</h3><video controls preload="metadata" src="side/predictions.mp4"></video>
<h3>清晰</h3><video controls preload="metadata" src="clear/predictions.mp4"></video>
<h3>另一段</h3><video controls preload="metadata" src="train2/predictions.mp4"></video></section>
<section><h2>运行</h2><pre>python forecast.py run --profile live --raw &lt;event.raw&gt; --output &lt;新目录&gt; --start 110 --duration 4 --publish-hz 5 --tracker-hz 25 --preview</pre>
<p>当前是本机按真实时间回放RAW，尚未接入实时相机，也没有稳定有效5Hz和未知录像泛化保证。下一步优先处理有效输出空档、测量时序与当前几何资格，推力代理仍保留实验。</p>
<p><a href="summary.json">完整数值/检查</a> · <a href="../live_decoder_comparison_20261025.json">读取器逐事件对照</a> · <a href="https://docs.prophesee.ai/stable/guides/frames_generators.html">SDK帧生成文档</a></p></section></html>'''
    (out/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(timing=summary,precision={n:{h:q['metrics'] for h,q in hs.items()} for n,hs in scores.items()}),ensure_ascii=False))


if __name__=='__main__':main()
