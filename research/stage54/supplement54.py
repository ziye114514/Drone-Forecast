"""Present actual deployment measurements, including the negative speed result."""
from pathlib import Path
import argparse,datetime,hashlib
import numpy as np
import study54 as current
OUT=current.OUT
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze():
    current.verify();p=OUT/'delivery_supplement.json';assert not p.exists()
    current.save(p,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),helper_sha256=sha(Path(__file__)),
        design='Only summarize saved53/54 actual runs. Separate three research groups from one complete15-model group, show same-scope dual/shared negative median result and wall time; keep prefix-padding validation correction visible. No performance or model tuning.'))
def verify():
    assert current.load(OUT/'delivery_supplement.json')['helper_sha256']==sha(Path(__file__));current.verify()
def render():
    verify()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a=current.load(OUT/'runtime_summary.json');side=next(r for r in a['table'] if r['case']=='side6');dual=next(r for r in a['table'] if r['case']=='dual_side6');old=current.load(current.REFERENCE/'side6/completion.json')
    values=[old['processing_median_ms'],dual['median_ms'],side['median_ms']];walls=[old['wall_including_cold_decode_video_s'],dual['wall_s'],side['wall_s']]
    labels=['53: dual\n3 research groups','54: dual\n1 complete group','54: shared\n1 complete group'];colors=['#a3a3a3','#277da1','#e76f51']
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained');x=np.arange(3)
    bars=axes[0].bar(x,values,color=colors);axes[0].bar_label(bars,fmt='%.1f',padding=3);axes[0].axhline(20,ls='--',color='#333',label='20 ms budget');axes[0].set_xticks(x,labels,fontsize=8);axes[0].set_ylabel('Processing median (ms)');axes[0].set_title('Side: shared is not faster per step');axes[0].legend(fontsize=8)
    bars=axes[1].bar(x,walls,color=colors);axes[1].bar_label(bars,fmt='%.1f',padding=3);axes[1].set_xticks(x,labels,fontsize=8);axes[1].set_ylabel('Wall time (s)');axes[1].set_title('Includes cold start, seeks and video')
    fields=['parent_geometry_ms','image_preparation_ms','communication_ms','physics_ms'];x=np.arange(4)
    for j,(c,label,color) in enumerate([('dual_side6','Dual','#277da1'),('side6','Shared','#e76f51')]):
        val=[next(r['median_ms'] for r in a['components'] if r['case']==c and r['field']==field) for field in fields]
        axes[2].bar(x+(j-.5)*.32,val,.32,label=label,color=color)
    axes[2].set_xticks(x,['Geometry20','Image prep','Wait/transport','Physics'],rotation=15,fontsize=8);axes[2].set_ylabel('Stage median (ms)');axes[2].set_title('Stage medians are not additive');axes[2].legend(fontsize=8)
    fig.suptitle('Same predictions; uncontrolled cross-run load; extra event SHA in shared research mode',fontsize=11)
    fig.savefig(OUT/'deployment_comparison.png',dpi=160);plt.close(fig)
    costs=[]
    for case in current.SCORING:
        rr=current.rows(OUT/case/'observations_forecasts.jsonl');tt=current.rows(OUT/case/'rotor_ticks.jsonl')
        residual=[r['rotor_request_ms']-sum(t['processing_ms'] for t in tt[j*2:j*2+2]) for j,r in enumerate(rr)]
        costs.append(dict(case=case,rotor_split_decode_log_residual_median_ms=float(np.median(residual)),p95_ms=float(np.percentile(residual,95))))
    current.save(OUT/'additional_cost_diagnostic.json',dict(table=costs,interpretation='Residual includes splitting/decoder/logging, not a pure IPC or detector timer. No new algorithm change.'))
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>54部署优化结果与校验说明</title><style>body{max-width:1400px;margin:30px auto;font:17px/1.8 Arial;color:#233e4c}img{width:100%}</style><h1>完整物理功能保留，单次解码尚未稳定更快</h1><p>所有15模型、逐桨频率/姿态/推力/在线学习保持，只去掉两组重复研究对照。四个正式shared片段中位清晰175.4ms、侧视153.7ms、3号120.7ms、空场83.1ms，仍不实时。相比53三组对照减少了计算，但不同范围不能当成独立算法加速。</p><img src="deployment_comparison.png"><p>同范围侧视，单分支dual142.2ms/shared153.7ms，shared单步未胜；wall82.7→76.0s，包含冷启动/seek/视频。共享研究模式额外做原生事件双端SHA，跨次机器负载未严格控制。报告工程现象，不宣传稳定倍数。</p><p>侧视原生20ms包峰值21,151,920字节，32MiB缓冲未溢出，没有降采样。发布/校验中位0.765ms，主要耗时仍是10ms桨叶几何/频率与20ms几何；wait包含worker实际计算，不是纯传输成本。下一步优先用已有NN整机框约束桨盘区域，而非只继续微调共享内存。</p><h2>前缀校验的更正</h2><p>原跨运行全结构体SHA检查在108.04s失败，记录完整保留。两个独立RAW读取器对照确认：x/y/p/t所有值与顺序完全相同，变化仅在16字节结构体中第6/7两个未使用填充字节。原先把不参与事件语义的填充值当成跨运行不变量不合适。</p><p>每次传输内仍要求父/子完整字节SHA相同，未放宽。跨运行用全部实际数据/图像/频率/15模型预测/学习状态和事件数核对，225整机/450桨叶的独立RAW前缀通过，225行未来评分精确53；预测器源码和任何数值容差均未修改。原字节失败与补充诊断都保留。原始全程payload未存档，因此诊断不声称重建当时的填充内容。</p><p><a href="failed_prefix_byte_sha_v1.json">原失败</a> · <a href="structured_padding_diagnostic.json">独立读取器字节诊断</a> · <a href="raw_prefix_checks.json">最终语义/模型前缀</a> · <a href="prefix_validation_supplement.json">补充协议</a></p><p><a href="第五十四阶段单次解码完整物理部署报告.html">完整报告、视频与使用命令</a> · <a href="score_regression.json">所有未来评分保持</a></p></html>'''
    (OUT/'部署结果图与前缀校验说明.html').write_text(page,encoding='utf-8');print('Deployment figures and prefix-padding explanation complete',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','render']);globals()[p.parse_args().action]()
