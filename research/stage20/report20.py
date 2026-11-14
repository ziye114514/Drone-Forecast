"""Numeric performance/equivalence report, no inferred accuracy gain."""
from study20 import OUT,CASES,load,save,verify


def table(headers,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+str(x)+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>'


def main():
    verify();results=[];bench=[]
    for case in CASES:
        c=load(OUT/case/'completion.json');b=load(OUT/case/'paired_benchmark.json')[0]
        results.append([case,c['frames'],c['exact_vs_stage18'],f"{c['frontend_median_ms']:.2f}",f"{c['frontend_p95_ms']:.2f}",c['prefix_exact']])
        bench.append([case,f"{b['t']:.2f}",f"{b['dense_median_ms']:.2f}",f"{b['sparse_median_ms']:.2f}",f"{b['dense_median_ms']/b['sparse_median_ms']:.2f}×",f"{b['dense_hist_bytes']/1e6:.2f}",f"{b['sparse_hist_bytes']/1e6:.2f}"])
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第二十阶段：稀疏缓存等价与性能</title><style>body{font:17px/1.75 "Microsoft YaHei",sans-serif;color:#243342;margin:0;background:#f3f6fa}main{max-width:1150px;margin:auto;padding:32px;background:white}h1{font-size:30px;line-height:1.35}h2{margin-top:36px;border-bottom:2px solid #dce5ec}.note{padding:18px;background:#edf6fb;border-left:5px solid #3888a8}table{border-collapse:collapse;width:100%;font-size:15px}td,th{text-align:left;white-space:nowrap;padding:10px;border-bottom:1px solid #dbe3ea}th{background:#edf2f7}.scroll{overflow:auto}a{color:#176b98}</style><main><h1>第二十阶段：稀疏分块缓存的等价性与性能</h1><p>2026-10-18 · 35秒自采RAW · 1750个截止时刻 · 固定参数/源码后运行</p><div class="note"><b>有计算收益，没有检测精度变化：</b>1750个完整当前观测与阶段18严格分支逐项相同。低事件量时避免了大缓存固定开销；清晰完整前端中位约54ms，仍未达到20ms实时。</div><h2>1. 只改变缓存，不改变频谱与几何</h2><p>原版每帧处理14400×400的稠密环形表；新版保留5个左右互不重叠的20ms分块直方图，各块仅保存有事件的空间单元。过期块直接扣除，部分过期按250µs格裁去；只为100ms内达到原门槛的块拼装FFT序列。原阈值、频带、三帧发布、几何/状态代码均保持。</p>'''
    text+=table(['片段','处理时刻','与阶段18完整观测一致','完整前端中位ms','P95 ms','前缀一致时刻'],results)
    text+='<p>随机流44个时刻同时与原整窗重算、阶段18稠密增量比较，quality/hz/mask/区域/事件数逐项相同；含30ms跳步使最旧块部分过期的测试、清空及未来事件隔离。真实RAW清晰前缀15时刻一致。见<a href="module_checks.json">module_checks.json</a>。</p><h2>2. 七次交替配对计时</h2><p>同一输入与历史状态，比较“稠密增量入环+FFT”与“稀疏增量更新+FFT”；状态副本构造在双方计时外，新事件计数成本都计入。四个时间点均在运行前固定，包含无机低事件量片段。</p>'
    text+=table(['片段','时刻s','稠密中位ms','稀疏中位ms','模块倍速','稠密缓存MB','稀疏缓存MB'],bench)
    text+='<p>缓存MB只统计持久直方图、索引和计数数组nbytes，不是进程RSS或峰值内存；不包含原始事件、FFT临时数组及图像。模块倍速不是端到端倍速。各片段paired_benchmark.json保存7次原始测量。</p><h2>3. 取舍与限制</h2><p>稀疏缓存可作为已核验的计算模块复用；它保留的严格前端仍有阶段17/18的侧视9/14标签覆盖，不能因为更快就当成更准。原默认预测入口保留；该缓存也可为后续软证据分支供相同频谱。当前检测/定位尚未证明机械RPM、姿态或未来轨迹准确。</p><p>完整调用耗时含缓存更新、频谱和原几何，不含解码/JSON。研究循环另含seek、解码及配对基准，不能称持续实时。下一步重点转回原生时间窗、整机分组及独立评测。</p><p><a href="protocol.json">冻结协议</a>；<a href="阶段结论.md">阶段结论</a>；<a href="delivery_checks.json">完整性复核</a>。未用G，未访问执行yolov10-main，旧源码及历史输出保留。</p></main></html>'
    (OUT/'第二十阶段稀疏缓存报告.html').write_text(text,encoding='utf-8');print('Stage20 report rendered')


if __name__=='__main__':main()
