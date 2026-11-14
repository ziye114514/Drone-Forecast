import html,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graph45 import *
def main():
    verify();b=load(OUT/'benchmark.json');seq=OUT/'sequence';a=load(seq/'analysis.json');old=load(common44.INPUT/'analysis.json');lookup={(r['case'],r['frontend'],r['model'],r['nominal_h']):r for r in old['table']};table=[];discrete=[]
    for r in a['table']:
        original=lookup[r['case'],r['frontend'],r['model'],r['nominal_h']]
        for key in ['valid_forecasts','joint_success','absent_forecasts_inside','known_negative_published_anchors']:
            if r[key]!=original[key]:discrete.append(dict(case=r['case'],method=r['frontend'],model=r['model'],h=r['nominal_h'],field=key,new=r[key],original=original[key]))
        if r['frontend']=='body_first' and r['nominal_h']==.4 and r['model'] in ['cv','trusted_acc'] and r['case']!='negative5':table.append(dict(case=r['case'],model=r['model'],targets=r['present_targets'],valid=r['valid_forecasts'],original_mean=original['mean_fde_px'],graph_mean=r['mean_fde_px'],original_success=original['joint_success'],graph_success=r['joint_success']))
    original={(r['case'],r['t']):r for r in map(json.loads,(common44.INPUT/'detections.jsonl').read_text(encoding='utf-8').splitlines()) if r['method']=='finetuned'};n=same=0;maxdiff=0.;new_counts={}
    for r in map(json.loads,(seq/'detections.jsonl').read_text(encoding='utf-8').splitlines()):
        if r['method']!='finetuned':continue
        q=original[r['case'],r['t']];n+=1;same+=len(r['boxes'])==len(q['boxes'])
        if len(r['boxes'])==len(q['boxes']):
            for x,y in zip(r['boxes'],q['boxes']):maxdiff=max(maxdiff,float(abs(np.array(x['bbox'])-y['bbox']).max()))
    assert n==1550;assert same==n;assert not discrete
    save(OUT/'quality_regression.json',dict(frames=n,per_frame_box_count_same=same,max_retained_bbox_coordinate_difference_px=maxdiff,all_horizons_all_models_coverage_joint_success_absent_outputs_equal=True,original_reference='41 same1550 native input frames; not new blind data.',table=table))
    rr=[]
    for scope,methods in b['times'].items():
        for name,x in methods.items():rr.append(dict(scope=scope,method=name,n=x['n'],median_ms=x['median_ms'],p95_ms=x['p95_ms']))
    def tab(rows):
        cols=list(rows[0]);return '<table><tr>'+''.join('<th>'+x+'</th>' for x in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(round(r[x],5) if isinstance(r[x],float) else r[x]))+'</td>' for x in cols)+'</tr>' for r in rows)+'</table>'
    names=list(b['times']['image']);fig,ax=plt.subplots(figsize=(9,4));med=[b['times']['image'][n]['median_ms'] for n in names];ax.bar(names,med,color=['#3c7ca1','#79b8ba','#b4b7bd','#e4a042']);ax.axhline(20,color='#bb4e46',ls='--',label='20ms whole-tick target');ax.set_ylabel('Median inference chain ms, batch1');ax.set_title('Native-image preprocessing + network + official NMS, same-process alternating');ax.legend();fig.tight_layout();fig.savefig(OUT/'graph_latency.png',dpi=150);plt.close(fig)
    text='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>第四十五阶段CUDA Graph与完整预测回归</title><style>body{font:16px/1.8 Arial;max-width:1300px;margin:30px auto;color:#233e4c}table{border-collapse:collapse;font-size:14px;width:100%}th,td{padding:7px;border:1px solid #cbd8e0}img{width:100%}</style><h1>第四十五阶段：CUDA Graph使模型链约快2.1倍，完整未来评分基本保持</h1><p><b>同进程单图交替：PyTorch47.77ms、FP32 CUDA Graph22.86ms。</b>输入仍native1280×720，网络实际384×640；保持batch1、FP32、固定conf.25和官方NMS。ORT各支显式关闭TF32，不用半精度。绑定GPU固定地址，当前输入update_inplace后才replay；没有批次等待或取未来帧。</p><p>12预设图×5次交替，先做5预热。普通run/仅binding未显著改善，graph才减少调用开销。tensor是输入已预处理，但包含HtoD/推理/回传/NMS；image另含原生图预处理。CPU预处理12张与原PyTorch输入逐元素精确；首图→次图→首图回放精确且输入变化确实改变输出，防止把旧输出重复当新检测。</p>'''+tab(rr)+'''<img src="graph_latency.png"><p>首次capture约0.55秒另记，表不包含RAW、几何、进程通信、写视频。模型链仍略高于整帧20ms预算，尚未整系统实时。初次脚本在正式计时前遇到PyTorch inference tensor被NMS原地修改的上下文错误；保存补充后整次benchmark统一inference_mode，计算不改，原冻结脚本保留。</p><h2>1550时刻完整质量复评</h2><p>先完成全部31秒独立神经输出，再按原41五支/六模型因果回放，最后读取粗框。全部1550每帧框数量与41一致；所有0.1/0.2/0.4秒、六模型的有效数量、联合成功和无机输出统计保持。神经框与未来浮点值不强称bitwise。</p>'''+tab(table)+'''<p>train3仍当前12/12、0.4秒11/12可预报，收缩约110px且成功仅2/12；side联合成功6/14，清晰7/8，负例400当前/预报零。这里证明后端优化基本保留原质量，没有改善转弯意图，也没有额外姿态/RPM真值。原几何/公开模型缓存只作固定参考，新的神经输出没有取未来定位代理。</p><p>采用<a href="https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#using-cuda-graphs-preview">ORT官方CUDA Graph与IO绑定</a>，属于现成部署优化，不称算法创新。下一步16位候选需再完整重评，然后接无损共享内存RAW入口；当前旧默认/42不替换。<a href="quality_regression.json">完整质量回归</a> · <a href="sequence/analysis.json">全部未来评分</a> · <a href="benchmark.json">计时原始数据</a> · <a href="protocol.json">冻结协议</a></p></html>'''
    (OUT/'第四十五阶段CUDA图与预测回归报告.html').write_text(text,encoding='utf-8');(OUT/'阶段结论.md').write_text('同进程模型链47.77→22.86ms约2.09x，非整帧实时。1550框数、所有时距/模型覆盖/联合成功/无机统计保持，未来误差仅微小浮点差。原始难机动仍约110px。标准部署优化非创新算法，旧默认保留。',encoding='utf-8');print('Stage45 report and quality regression complete',flush=True)
if __name__=='__main__':main()
