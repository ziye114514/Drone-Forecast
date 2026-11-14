"""Actual clocks and unchanged-output evidence for frozen42 instrumentation."""
import html,json
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from profile43 import ROOT,OUT,CASES,load,save,verify

def rows(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
def main():
    verify();old={r['t']:r for r in rows(ROOT.parent/'output/stage42_raw_body_bridge_20261018/train3/observations_forecasts.jsonl')};current=rows(OUT/'train3/observations_forecasts.jsonl');assert len(current)==100
    for r in current:
        for key in ['t','event_count','filtered_event_count','observations','forecasts','body_boxes']:assert r[key]==old[r['t']][key],(r['t'],key)
    checks=[];summaries=[];times={}
    for case in CASES:
        samples=load(OUT/case/'component_times.json');times[case]=samples;rr=rows(OUT/case/'observations_forecasts.jsonl');assert len(samples)==len(rr)
        cap=cv2.VideoCapture(str(OUT/case/'forecasts.mp4'));n=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape==(720,1280,3);n+=1
        cap.release();assert n==len(rr);checks.append(dict(case=case,decoded_video_frames=n))
        s=load(OUT/case/'component_summary.json');summaries.append(dict(case=case,frames=n,geometry_median_ms=s['geometry_ms']['median_ms'],worker_network_median_ms=float(np.median([r['worker_profile']['ultralytics_speed_ms']['inference'] for r in samples])),png_median_ms=s['png_encode_ms']['median_ms'],ipc_with_model_median_ms=s['ipc_model_ms']['median_ms'],raw_next_median_ms=s['raw_next_ms']['median_ms'],processing_median_ms=s['parent_processing_ms']['median_ms'],processing_p95_ms=s['parent_processing_ms']['p95_ms']))
    save(OUT/'regression.json',dict(original42_full_observations_all_forecasts_and_boxes_exact=100,decoded_videos=checks,sources_unchanged=True,labels_read=False));save(OUT/'analysis.json',dict(table=summaries,scope='Current workstation wall-clock decomposition, no matched-hardware comparison to previous-day timings. Same42 computations and outputs.'))
    keys=['geometry_ms','noise_ms','render_ms','png_encode_ms','ipc_model_ms','history_ms'];labels=['Geometry','SDK filter','Render','PNG encode','IPC + model','History + bookkeeping'];colors=['#277da8','#4caaa0','#92cdb7','#edc55a','#e99450','#aa78ad'];fig,ax=plt.subplots(figsize=(10,5));left=np.zeros(3)
    for key,label,color in zip(keys,labels,colors):
        values=np.array([np.mean([r[key] for r in times[c]]) for c in CASES]);ax.barh(list(CASES),values,left=left,label=label,color=color);left+=values
    ax.set_xlabel('Mean milliseconds per native20ms tick (means add; decode/write/preview excluded)');ax.set_title('Stage43: same42 computations, actual component clocks');ax.legend(ncols=3,loc='upper center',bbox_to_anchor=(.5,-.18));fig.tight_layout();fig.savefig(OUT/'latency_components.png',dpi=150,bbox_inches='tight');plt.close(fig)
    headers=['case','frames','geometry_median_ms','worker_network_median_ms','png_median_ms','ipc_with_model_median_ms','raw_next_median_ms','processing_median_ms','processing_p95_ms'];table='<table><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(round(r[h],3) if isinstance(r[h],float) else r[h]))+'</td>' for h in headers)+'</tr>' for r in summaries)+'</table>'
    text='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>第四十三阶段实际耗时拆解</title><style>body{font:16px/1.8 Arial;max-width:1300px;margin:30px auto;color:#263e4c}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ccd8e0}img{width:100%}</style><h1>第四十三阶段：慢在几何与网络，RAW解码中位并非主因</h1><p>2026-10-19任务恢复后，先对42原代码的内存副本加时钟，原文件/模型/当前输入和操作顺序不动。train3两秒100帧、side两秒100帧、空场一秒50帧。100个train3的全部观测、四路六预测和网络框与42精确相同；250视频帧全解码通过。无需用当前定位标签评价这次纯计时。</p><p><b>几何中位26–43ms、网络内部约43ms，另有PNG编码6–8ms和进程通信。</b>RAW取下一块中位约0.3–1.4ms，侧视尾延迟仍可能高，不能说解码完全免费。单纯换通信不足以达到20ms预算，优先检查现成图执行后端，再考虑无损共享内存。</p>'''+table+'''<img src="latency_components.png"><p>图为均值可加分解，表为中位/P95，不能把各项中位相加当总中位。IPC已包含子进程解码、预处理、网络、后处理和响应，网络内部不能再重复计入总数。parent processing排除了RAW next、JSON写入和视频；这些都在component_times单列。history含四路六模型及少量行整理，真实主流水线仍不实时。</p><p>同日43中位约94–130ms，不能拿它比昨天42约102ms宣称优化退步：冷启动、负载和运行组织不同，43没有改计算。后续44将在同一进程交替比较PyTorch与现成ONNX Runtime，记录实际provider和检测差异，当前未宣称加速成功。</p><p>复用<a href="https://docs.ultralytics.com/modes/predict/">官方Results.speed分项</a>及Python perf_counter，未提出新检测算法。原始逐帧<a href="train3/component_times.json">train3时钟</a>、<a href="analysis.json">结果</a>、<a href="regression.json">回归</a>、<a href="protocol.json">冻结协议</a>。</p></html>'''
    (OUT/'第四十三阶段实际耗时拆解报告.html').write_text(text,encoding='utf-8');(OUT/'阶段结论.md').write_text('主耗时几何与网络，PNG通信有额外代价，RAW中位不是主因。原42完整100观测/全部预测/框精确，250视频帧通过。不同日负载不作速度因果比较，下一步同进程后端对照。',encoding='utf-8');print('Stage43 report verified',flush=True)
if __name__=='__main__':main()
