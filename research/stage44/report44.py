import html
import numpy as np
from common44 import *
def main():
    verify();p=load(OUT/'benchmark.json');a=load(OUT/'benchmark_no_profiling.json');assert not a['failures'];assert p['provider_node_counts']['ort_cuda'].get('CUDAExecutionProvider',0)>0
    rows=[];ref={r['id']:r['boxes'] for r in a['detections'] if r['method']=='pytorch'};quality=[]
    for method,stats in a['times'].items():rows.append(dict(method=method,n=stats['n'],median_ms=stats['median_ms'],p95_ms=stats['p95_ms']))
    for method in ['ort_cuda','ort_cpu']:
        same=0;delta=[]
        for r in a['detections']:
            if r['method']!=method:continue
            b=ref[r['id']];same+=len(r['boxes'])==len(b)
            if len(r['boxes'])==len(b) and len(b):delta.append(float(abs(np.array(r['boxes'])[:,:4]-np.array(b)[:,:4]).max()))
        quality.append(dict(method=method,samples=12,same_box_count=same,retained_box_max_coordinate_diff_px=max(delta,default=0),raw_max_coordinate_diff_px=max(r['coordinate_max_abs'] for r in a['raw_parity'] if r['method']==method),raw_score_max_diff=max(r['score_max_abs'] for r in a['raw_parity'] if r['method']==method)))
    save(OUT/'summary.json',dict(timing=rows,parity=quality,onnx_sha256=load(OUT/'export.json')['sha256'],scope='Small fixed backend pilot, not new whole-UAV/trajectory accuracy evidence. Default CUDA math may use TF32; following45 disables it explicitly.'))
    def table(rr):
        cols=list(rr[0]);return '<table><tr>'+''.join('<th>'+h+'</th>' for h in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(round(r[h],5) if isinstance(r[h],float) else r[h]))+'</td>' for h in cols)+'</tr>' for r in rr)+'</table>'
    text='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>第四十四阶段ONNX后端对照</title><style>body{font:16px/1.8 Arial;max-width:1250px;margin:30px auto;color:#233e4c}table{border-collapse:collapse;font-size:14px;width:100%}th,td{padding:8px;border:1px solid #ccd8df}</style><h1>第四十四阶段：普通ONNX CUDA有小幅加速，仍不足整帧实时</h1><p>复用现有Ultralytics官方导出和ONNX Runtime1.30，未安装新包。40本地可信best复制到新44目录后导出，原40不写。固定FP32/384×640/batch1/opset17，无简化/量化/NMS导出；native1280×720经过原网络最小矩形padding实际就是384×640，没有进一步缩小画面。ONNX checker通过。</p><p>12幅预设画面(每clear/side/train3/negative序列1/6、1/2、5/6)，不读真值。每方法5预热+60交替测量，单图预处理/模型/官方NMS，排除RAW/几何/IPC。<b>普通ORTCUDA中位30.55ms，PyTorch40.72ms，约1.33倍；CPU47.35ms较慢。</b>不是完整系统20ms实时。</p>'''+table(rows)+'''<h2>模型与输入是否一致</h2>'''+table(quality)+'''<p>三方法12画面框数量均相同，但不是bitwise。原始输出包含未保留提案，最大差大于NMS后框差，均记录。这里FP32指输入/权重存储；默认CUDA EP可能采用TF32数学，45进一步明确禁用。小样本一致不能替代完整1550连续当前/未来重评。</p><h2>计时修正与真实provider</h2><p>第一次开启ORT逐节点profiling而PyTorch未开，计时不公平，完整保留在benchmark.json；CUDA节点29722、CPU节点0证明实际GPU执行。之后在新会话关闭profiling，单独冻结unprofiled_supplement和helper、写benchmark_no_profiling.json作为主表；没有改样本/阈值。首次带分析时CUDA40.42ms不能与主表混用。下一步按官方CUDA Graph/IO binding减少调度和拷贝开销，不重复训练。</p><p><a href="https://docs.ultralytics.com/modes/export/">Ultralytics官方导出</a> · <a href="https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html">ORT CUDA官方文档</a> · <a href="summary.json">结果</a> · <a href="protocol.json">冻结协议</a>。未声称新算法、姿态/RPM准确或跨机型泛化。</p></html>'''
    (OUT/'第四十四阶段ONNX后端对照报告.html').write_text(text,encoding='utf-8');(OUT/'阶段结论.md').write_text('关闭ORT profiling后普通CUDA30.55ms/PyTorch40.72约1.33x，仅模型链非整系统；12预设画面框数相同、非bitwise。实际CUDA节点确认，CPU较慢。下一步官方CUDA graph，未装依赖/未改旧目录。',encoding='utf-8');print('Stage44 report complete',flush=True)
if __name__=='__main__':main()
